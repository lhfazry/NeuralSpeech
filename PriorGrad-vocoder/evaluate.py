import os
import subprocess
import sys
import glob
import librosa
import pyworld
import numpy as np
from nnmnkwii.metrics import melcd
from argparse import ArgumentParser

def main(args):
    sr = 16000

    results = []

    for fname in os.listdir(args.sdir):
        synthetic_wav, _ = librosa.load(os.path.join(args.sdir, fname), sr=sr, mono=True)
        original_wav, _ = librosa.load(os.path.join(args.odir, fname), sr=sr, mono=True)

        synthetic_wav = synthetic_wav.astype(np.float64)
        f0_1, timeaxis_1 = pyworld.harvest(synthetic_wav, sr, frame_period=5.0, f0_floor=71.0, f0_ceil=800.0)
        sp1 = pyworld.cheaptrick(synthetic_wav, f0_1, timeaxis_1, sr)  

        original_wav = original_wav.astype(np.float64)
        f0_2, timeaxis_2 = pyworld.harvest(original_wav, sr, frame_period=5.0, f0_floor=71.0, f0_ceil=800.0)
        sp2 = pyworld.cheaptrick(original_wav, f0_2, timeaxis_2, sr)

        # mel-cepstrum
        coded_sp_1 = pyworld.code_spectral_envelope(sp1, sr, 24)
        coded_sp_2 = pyworld.code_spectral_envelope(sp2, sr, 24)

        result = melcd(coded_sp_1,coded_sp_2 , lengths=None)
        print(result)
        results.append(dict(fname=fname, score=result))

    print(results)

if __name__ == '__main__':
    parser = ArgumentParser(description='Calculate MCD')
    parser.add_argument('sdir', help='Synthetic directory of waveform')
    parser.add_argument('odir', help='Original directory of waveform')
    
    main(parser.parse_args())
