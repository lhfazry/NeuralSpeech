import os
import subprocess
import sys
import glob
import librosa
from preprocess import get_mel
import numpy as np
import torch
from nnmnkwii.metrics import melcd
from argparse import ArgumentParser
from preprocess import MAX_WAV_VALUE, get_mel, normalize
from params import params
from scipy.io.wavfile import read

def main(args):
    smels = []
    omels = []

    for fname in os.listdir(args.sdir):
        synthetic_mels = load_mels(os.path.join(args.sdir, fname))
        original_mels = load_mels(os.path.join(args.odir, fname))
        smels.append(synthetic_mels)
        omels.append(original_mels)
        #result = melcd(synthetic_mels.numpy(), original_mels.numpy() , lengths=None)
        #print(f"{fname} ==> {result}")

        #results.append(dict(fname=fname, score=result))
        #results += result
        #total += 1

    result = melcd(torch.stack(smels), torch.stack(omels))
    print(f"result: {result}")

def load_mels(audio_file):
    sr, audio = read(audio_file)

    if params.sample_rate != sr:
        raise ValueError(f'Invalid sample rate {sr}.')

    audio = audio / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95

    # match audio length to self.hop_size * n for evaluation
    if (audio.shape[0] % params.hop_samples) != 0:
        audio = audio[:-(audio.shape[0] % params.hop_samples)]
    
    audio = torch.FloatTensor(audio)
    spectrogram = get_mel(audio, params)

    return spectrogram
    
if __name__ == '__main__':
    parser = ArgumentParser(description='Calculate MCD')
    parser.add_argument('--sdir', help='Synthetic directory of waveform')
    parser.add_argument('--odir', help='Original directory of waveform')
    
    main(parser.parse_args())
