import numpy as np
from argparse import ArgumentParser
from scipy.io.wavfile import read
from preprocess import MAX_WAV_VALUE, get_mel, normalize


def get_noisy_audio(args):  
    sr, audio = read(args.audio_path)
    
    audio = audio / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95
    
    T = audio.shape
    
    noise_schedule = np.linspace(1e-4, 0.05, args.max_step).tolist(), # [beta_start, beta_end, num_diffusion_step]

    beta = np.array(noise_schedule)
    noise_level = np.cumprod(1 - beta)

    #t = np.random.randint(0, len(noise_schedule), [N])
    t = args.step
    noise_scale = noise_level[t].unsqueeze(1)
    noise_scale_sqrt = noise_scale ** 0.5
    noise = np.random.randn(*audio.shape)
    noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise
    print(noisy_audio)

if __name__ == '__main__':
  parser = ArgumentParser(description='Get noisy audio')
  parser.add_argument('--audio_path', default=None, type=str,
      help='audio path')
  parser.add_argument('--max_step', default=400, type=int,
      help='diffusion step')
  parser.add_argument('--step', default=2, type=int,
      help='diffusion step')
  get_noisy_audio(parser.parse_args())