import numpy as np
import os
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import matplotlib.pyplot as plt

def erb_bandwidth(f):
    return 24.7 * (4.37e-3 * f + 1.0)

def gammatone_filter(f, f_center, sr, NFFT):
    t = np.linspace(0, 1, sr)[:NFFT // 2 + 1]
    gamma_tone = np.power(t, 3) * np.exp(-2 * np.pi * erb_bandwidth(f_center) * t) * np.cos(2 * np.pi * f * t)
    return np.abs(np.fft.rfft(gamma_tone, NFFT)[:NFFT // 2 + 1])

def make_gammatone_filterbank(nfilt, NFFT, sr):
    fmin = 20
    fmax = sr / 2
    fcenter = np.geomspace(fmin, fmax, nfilt)
    gt_filters = np.zeros((nfilt, NFFT // 2 + 1))
    for i in range(nfilt):
        gt_filters[i] = gammatone_filter(fcenter[i], fcenter[i], sr, NFFT)
    return gt_filters

def extract_gfcc(audio_file, n_gfcc=13):
    # Load the audio file using scipy
    sr, y = wav.read(audio_file)

    # Pre-emphasis
    pre_emphasis = 0.97
    emphasized_signal = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # Framing
    frame_size = 0.025
    frame_stride = 0.01
    frame_length, frame_step = frame_size * sr, frame_stride * sr
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Windowing
    frames *= np.hamming(frame_length)

    # Fourier Transform and Power Spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))

    # Gammatone Filter Bank
    nfilt = 40
    gt_filters = make_gammatone_filterbank(nfilt, NFFT, sr)
    filter_banks = np.dot(pow_frames, gt_filters.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)

    # DCT to get GFCC
    gfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (n_gfcc + 1)]

    return gfcc


def main():
    # Directory containing audio files
    audio_dir = "../Pemsu"

    # Dictionary to store GFCC features for each audio file
    gfcc_features = {}

    # Iterate over each audio file in the directory
    for audio_file in os.listdir(audio_dir):
        if audio_file.endswith(".wav"):
            full_path = os.path.join(audio_dir, audio_file)
            gfcc = extract_gfcc(full_path)
            gfcc_features[audio_file] = gfcc

    # Display the extracted features
    for key, value in gfcc_features.items():
        print(f"{key}:\n{value}\n")

        # Visualization
        plt.figure(figsize=(10, 4))
        plt.imshow(value, cmap='viridis', origin='lower', aspect='auto')
        plt.title(f'GFCC - {key}')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.ylabel('Coefficients')
        plt.xlabel('Frames')
        plt.show()


# Execute the main function
main()
