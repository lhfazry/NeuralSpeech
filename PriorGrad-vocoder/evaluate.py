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
from params import params
from scipy.io.wavfile import read

def main(args):
    sr = 16000

    results = []

    for fname in os.listdir(args.sdir):
        _, synthetic_wav = read(os.path.join(args.sdir, fname))
        _, original_wav = read(os.path.join(args.odir, fname))

        #synthetic_wav = synthetic_wav.astype(np.float64)
        #f0_1, timeaxis_1 = pyworld.harvest(synthetic_wav, sr, frame_period=5.0, f0_floor=71.0, f0_ceil=800.0)
        #sp1 = pyworld.cheaptrick(synthetic_wav, f0_1, timeaxis_1, sr)  

        #original_wav = original_wav.astype(np.float64)
        #f0_2, timeaxis_2 = pyworld.harvest(original_wav, sr, frame_period=5.0, f0_floor=71.0, f0_ceil=800.0)
        #sp2 = pyworld.cheaptrick(original_wav, f0_2, timeaxis_2, sr)

        # mel-cepstrum
        #coded_sp_1 = pyworld.code_spectral_envelope(sp1, sr, 24)
        #coded_sp_2 = pyworld.code_spectral_envelope(sp2, sr, 24)
        synthetic_mels = get_mel(torch.tensor(synthetic_wav), params)
        original_mels = get_mel(torch.tensor(original_wav), params)

        result = melcd(synthetic_mels.numpy(), original_mels.numpy() , lengths=None)
        print(result)

        results.append(dict(fname=fname, score=result))

    print(results)

if __name__ == '__main__':
    parser = ArgumentParser(description='Calculate MCD')
    parser.add_argument('--sdir', help='Synthetic directory of waveform')
    parser.add_argument('--odir', help='Original directory of waveform')
    
    main(parser.parse_args())
