# Copyright 2022 (c) Microsoft Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from scipy.io.wavfile import read
from preprocess import MAX_WAV_VALUE, get_mel, normalize, spectral_normalize_torch

device = torch.device("cuda")

def parse_filelist(filelist_path):
    with open(filelist_path, 'r') as f:
        filelist = [line.strip() for line in f.readlines()]
    return filelist


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, filelist, params, is_training=True):
        super().__init__()
        self.data_root = Path(data_root)
        self.params = params
        self.filenames = []
        self.filenames = parse_filelist(filelist)
        if not is_training:
            self.filenames = sorted(self.filenames)
        self.hop_samples = params.hop_samples
        self.is_training = is_training

        self.use_prior = params.use_prior
        self.max_energy_override = params.max_energy_override if hasattr(params, 'max_energy_override') else None

        if self.is_training:
            self.compute_stats()

        if self.use_prior:
            # build frame energy data for priorgrad
            self.energy_max = float(np.load(str(self.data_root.joinpath('stats_priorgrad', 'energy_max_train.npy')),
                                      allow_pickle=True))
            self.energy_min = float(np.load(str(self.data_root.joinpath('stats_priorgrad', 'energy_min_train.npy')),
                                      allow_pickle=True))
            print("INFO: loaded frame-level waveform stats : max {} min {}".format(self.energy_max, self.energy_min))
            if self.max_energy_override is not None:
                print("overriding max energy to {}".format(self.max_energy_override))
                self.energy_max = self.max_energy_override
            self.std_min = params.std_min

    def compute_stats(self):
        if os.path.exists(self.data_root.joinpath("stats_priorgrad/energy_max_train.npy")) and \
                os.path.exists(self.data_root.joinpath("stats_priorgrad/energy_min_train.npy")):
            return
        # compute audio stats from the dataset
        # goal: pre-calculate variance of the frame-level part of the waveform
        # which will be used for the modified Gaussian base distribution for PriorGrad model

        energy_list = []
        print("INFO: computing training set waveform statistics for PriorGrad training...")
        for i in tqdm(range(len(self.filenames))):
            sr, audio = read(self.filenames[i])
            if self.params.sample_rate != sr:
                raise ValueError(f'Invalid sample rate {sr}.')
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            # match audio length to self.hop_size * n for evaluation
            if (audio.shape[0] % self.params.hop_samples) != 0:
                audio = audio[:-(audio.shape[0] % self.params.hop_samples)]
            audio = torch.FloatTensor(audio)
            spectrogram = get_mel(audio, self.params)
            energy = (spectrogram.exp()).sum(1).sqrt()
            energy_list.append(energy.squeeze(0))

        energy_list = torch.cat(energy_list)
        energy_max = energy_list.max().numpy()
        energy_min = energy_list.min().numpy()

        self.data_root.joinpath("stats_priorgrad").mkdir(exist_ok=True)
        print("INFO: stats computed: max energy {} min energy {}".format(energy_max, energy_min))
        np.save(str(self.data_root.joinpath("stats_priorgrad/energy_max_train.npy")), energy_max)
        np.save(str(self.data_root.joinpath("stats_priorgrad/energy_min_train.npy")), energy_min)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        pfilename = Path(audio_filename)
        sr, audio = read(audio_filename)
        
        if self.params.sample_rate != sr:
            raise ValueError(f'Invalid sample rate {sr}.')
        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95

        # match audio length to self.hop_size * n for evaluation
        if (audio.shape[0] % self.params.hop_samples) != 0:
            audio = audio[:-(audio.shape[0] % self.params.hop_samples)]
        
        audio = torch.FloatTensor(audio)
        #print(f"audio.shape: {audio.shape}")

        glot = None
        if self.params.use_glot:
            glot_path = os.path.join(pfilename.parents[1], 'glots', pfilename.stem + '-glot.npy')
            glot = np.load(glot_path)
            glot = glot / MAX_WAV_VALUE
            glot = normalize(glot) * 0.95
            glot = glot[:audio.shape[0]]
            glot = torch.FloatTensor(glot)
            #print(f"glot.shape: {glot.shape}")

        if self.is_training:
            # get segment of audio
            start = random.randint(0, audio.shape[0] - (self.params.crop_mel_frames * self.params.hop_samples))
            end = start + (self.params.crop_mel_frames * self.params.hop_samples)
            audio = audio[start:end]

            if self.params.degraded_mels:
                audio = audio + (self.params.degraded_scale) ** 0.5 * torch.randn_like(audio)

            if self.params.pretrained_mels == 0:
                spectrogram = get_mel(audio, self.params)
            elif self.params.pretrained_mels == 1: # tacotron 2
                mels_path = os.path.join(pfilename.parents[1], 'mels', pfilename.stem + '.npy')
                spectrogram = torch.tensor(np.load(mels_path), dtype=torch.float32)
                spectrogram = spectral_normalize_torch(spectrogram).unsqueeze(0)
                starts = random.randint(0, spectrogram.shape[2] - self.params.crop_mel_frames)
                ends = starts + self.params.crop_mel_frames * self.params.hop_samples
                spectrogram = spectrogram[:,:,starts + ends]
            elif self.params.pretrained_mels == 2: # fast speech
                spectrogram = get_mel(audio, self.params)
        else:
            if self.params.inf_pretrained_mels == 0:
                spectrogram = get_mel(audio, self.params)
            elif self.params.inf_pretrained_mels == 1: # tacotron 2
                mels_path = os.path.join(pfilename.parents[1], 'mels', pfilename.stem + '.npy')
                spectrogram = torch.tensor(np.load(mels_path), dtype=torch.float32)
                spectrogram = spectral_normalize_torch(spectrogram).unsqueeze(0)

                if audio.shape[0] >= spectrogram.shape[2] * self.params.hop_samples:
                    audio = audio[0:(spectrogram.shape[2] * self.params.hop_samples)]
                else:
                    spectrogram = spectrogram[:,:,:audio.shape[0] // self.params.hop_samples]

            elif self.params.inf_pretrained_mels == 2: # fast speech
                spectrogram = get_mel(audio, self.params)

        #print(f"spectrogram: {spectrogram.shape}")

        if self.use_prior:
            energy = (spectrogram.exp()).sum(1).sqrt()

            if self.max_energy_override is not None:
                energy = torch.clamp(energy, None, self.max_energy_override)
            # normalize to 0~1
            target_std = torch.clamp((energy - self.energy_min) / (self.energy_max - self.energy_min), self.std_min, None)
        else:
            target_std = torch.ones_like(spectrogram[:, 0, :])
            #target_std = torch.ones(1, audio.shape[0] // self.params.hop_samples)

        #print(f"spectrogram.shape: {spectrogram.shape}, target_std: {target_std.shape}")

        # add glot
        if self.params.use_glot:
            if self.is_training:
                # get segment of glot
                glot = glot[start:end]

            if glot.shape[0] < audio.shape[0]:
                pad = audio.shape[0] - glot.shape[0]
                glot = F.pad(glot, (0, pad), 'constant', 0)
            else:
                glot = glot[:audio.shape[0]]

        #print(f"glot.shape: {glot.shape}")

        return {
            'audio': audio, # [T_time]
            'glot': glot,
            'spectrogram': spectrogram[0].T, # [T_mel, 80]
            'target_std': target_std[0], # [T_mel],
            'filename': audio_filename
        }


class Collator:
    def __init__(self, params, is_training=True):
        self.params = params
        self.is_training = is_training

    def collate(self, minibatch):
        samples_per_frame = self.params.hop_samples
        for record in minibatch:
            #Filter out records that aren't long enough.
            if record['spectrogram'] is not None:
                if len(record['spectrogram']) < self.params.crop_mel_frames:
                    del record['spectrogram']
                    del record['audio']
                    del record['glot']
                    continue

                record['spectrogram'] = record['spectrogram'].T
            
            record['target_std'] = record['target_std']
            record['target_std'] = torch.repeat_interleave(record['target_std'], samples_per_frame)
            record['audio'] = record['audio']
            #record['glot'] = record['glot']

            #print(f"audio.shape: {record['audio'].shape}, target_std: {record['target_std'].shape}")
            assert record['audio'].shape == record['target_std'].shape

        audio = torch.stack([record['audio'] for record in minibatch if 'audio' in record])
        filename = [record['filename'] for record in minibatch if 'filename' in record]

        #spectrogram = None
        #if self.params.use_mels:
        spectrogram = torch.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
        
        target_std = torch.stack([record['target_std'] for record in minibatch if 'target_std' in record])

        glot = None
        if self.params.use_glot:
            glot = torch.stack([record['glot'] for record in minibatch if 'glot' in record])
        
        return {
            'audio': audio,
            'glot': glot,
            'spectrogram': spectrogram,
            'target_std': target_std,
            'filename': filename
        }

def from_path(data_root, filelist, params, is_distributed=False):
    dataset = NumpyDataset(data_root, filelist, params, is_training=True)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=Collator(params, is_training=True).collate,
        shuffle=not is_distributed,
        num_workers=1,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=False,
        drop_last=True)


def from_path_valid(data_root, filelist, params, is_distributed=False):
    dataset = NumpyDataset(data_root, filelist, params, is_training=False)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=Collator(params, is_training=False).collate,
        shuffle=False,
        num_workers=1,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=False,
        drop_last=False)
