import numpy as np
import soundfile as sf
import os
from argparse import ArgumentParser
from scipy.io.wavfile import read
from preprocess import MAX_WAV_VALUE, get_mel, normalize
from pathlib import Path
import shutil
#from speechbrain.processing.speech_augmentation import AddBabble
#import pytest

def noise_psd(N, psd = lambda f: 1):
    print(f"N: {N}")
    X_white = np.fft.rfft(np.random.randn(N))
    print(f"X_white: {X_white.shape}")
    S = psd(np.fft.rfftfreq(N))
    print(f"S: {S.shape}")
    # Normalize S
    S = S / np.sqrt(np.mean(S**2))
    print(f"S2: {S.shape}")
    X_shaped = X_white * S
    print(f"X_shaped: {X_shaped.shape}")

    X_final = np.fft.irfft(X_shaped)
    print(f"X_final: {X_final.shape}")
    return X_final

def PSDGenerator(f):
    return lambda N: noise_psd(N, f)

@PSDGenerator
def white_noise(f):
    return 1;

@PSDGenerator
def blue_noise(f):
    return np.sqrt(f);

@PSDGenerator
def violet_noise(f):
    return f;

@PSDGenerator
def brownian_noise(f):
    return 1/np.where(f == 0, float('inf'), f)

@PSDGenerator
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))

def get_babble_noise(audio, sr, noise):
    if noise == 'cafe':
        sr, noise_sample = read('noises/avsr_noise_data_cafeteria_babble.wav')
    elif noise == 'street':
        sr, noise_sample = read('noises/avsr_noise_data_street_noise_downtown.wav')
    
    N = audio.shape[0]
    i_start = np.random.randint(0,len(noise_sample)-N-1)
    noise = noise_sample[i_start:i_start+N]
    
    e = np.linalg.norm(audio)
    en = np.linalg.norm(noise)
    gain = 10.0**(-1.0*sr/20.0)
    noise = gain * noise * e / en

    return noise
    
def get_noisy_audio(args):
    sr, audio = read(args.audio_path)
    
    audio = audio / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95
    
    T = audio.shape
    
    noise_schedule = np.linspace(1e-4, 0.05, args.max_step).tolist(), # [beta_start, beta_end, num_diffusion_step]

    beta = np.array(noise_schedule)
    noise_level = np.cumprod(1 - beta)

    #t = np.random.randint(0, len(noise_schedule), [N])
    for step in args.steps:
        t = int(step)
        noise_scale = noise_level[t]
        noise_scale_sqrt = noise_scale ** 0.5
        
        # noise = np.random.randn(*audio.shape)
        if args.color == 'white':
            noise = white_noise(*audio.shape)#np.random.randn(*audio.shape)
        elif  args.color == 'blue':
            noise = blue_noise(*audio.shape)
        elif  args.color == 'violet':
            noise = violet_noise(*audio.shape)
        elif  args.color == 'brown':
            noise = brownian_noise(*audio.shape)
        elif  args.color == 'pink':
            noise = pink_noise(*audio.shape)
        elif args.color == 'babble':
            noise = get_babble_noise(audio, sr, args.babble)

        noisy_audio = noise_scale_sqrt * audio[:len(noise)] + (1.0 - noise_scale) ** 0.5 * noise
        print(noisy_audio)

        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        filename = Path(args.audio_path).stem
        shutil.copyfile(args.audio_path, os.path.join(output_dir, Path(args.audio_path).name))

        if args.color == 'babble':
            sf.write(os.path.join(output_dir, filename + f"_noisy_{args.color}_{args.babble}_{t}.wav"), noisy_audio, sr, subtype='PCM_24')
        else:
            sf.write(os.path.join(output_dir, filename + f"_noisy_{args.color}_{t}.wav"), noisy_audio, sr, subtype='PCM_24')


if __name__ == '__main__':
  parser = ArgumentParser(description='Get noisy audio')
  parser.add_argument('--audio_path', default=None, type=str,
      help='audio path')
  parser.add_argument('--color', default='white', type=str,
      help='diffusion step')
  parser.add_argument('--babble', default='cafe', type=str,
      help='diffusion step')
  parser.add_argument('--max_step', default=400, type=int,
      help='diffusion step')
  parser.add_argument('--steps', required=True,
      nargs='+',
      help='diffusion step')
  parser.add_argument('--output_dir', default='output', type=str,
      help='diffusion step')
  get_noisy_audio(parser.parse_args())