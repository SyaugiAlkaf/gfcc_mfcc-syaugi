import os
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import matplotlib.pyplot as plt

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def make_mel_filterbank(nfilt, NFFT, sr):
    fmin = 0
    fmax = sr / 2
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, nfilt + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((NFFT + 1) * hz_points / sr).astype(int)

    filters = np.zeros((nfilt, NFFT // 2 + 1))
    for i in range(1, nfilt + 1):
        filters[i - 1, bin_points[i - 1]:bin_points[i]] = (np.arange(bin_points[i - 1], bin_points[i]) - bin_points[i - 1]) / (bin_points[i] - bin_points[i - 1])
        filters[i - 1, bin_points[i]:bin_points[i + 1]] = 1 - (np.arange(bin_points[i], bin_points[i + 1]) - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
    return filters

def extract_mfcc(audio_file, n_mfcc=13):
    sr, y = wav.read(audio_file)
    pre_emphasis = 0.97
    emphasized_signal = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
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
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))
    nfilt = 40
    mel_filters = make_mel_filterbank(nfilt, NFFT, sr)
    filter_banks = np.dot(pow_frames, mel_filters.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (n_mfcc + 1)]
    return mfcc

# Your directory path
data_dir = "./separated_binary_audio_files"

X = []  # Features (MFCC)
Y = []  # Labels (Diagnosis)

# MFCC parameters
num_cepstral = 13
nfft = 512

# Loop through diagnosis folders
for diagnosis in os.listdir(data_dir):
    diagnosis_dir = os.path.join(data_dir, diagnosis)

    if os.path.isdir(diagnosis_dir):
        for audio_file in os.listdir(diagnosis_dir):
            if audio_file.endswith(".wav"):
                file_path = os.path.join(diagnosis_dir, audio_file)
                mfcc_features = extract_mfcc(file_path, n_mfcc=num_cepstral)
                X.append(mfcc_features)
                Y.append(diagnosis)

# Determine the maximum frame length among non-empty audio files
frame_lengths = [x.shape[0] for x in X if x.shape[0] > 0]
if frame_lengths:
    max_frame_length = max(frame_lengths)
else:
    max_frame_length = 0  # No valid frames found

# Pad sequences to the maximum frame length
X_padded = []
for x in X:
    if x.shape[0] == 0:
        # Skip empty sequences
        continue
    padding_frames = max_frame_length - x.shape[0]
    X_padded.append(np.pad(x, ((0, padding_frames), (0, 0)), mode='constant'))

# Convert the list to a numpy array
X = np.array(X_padded)
Y = np.array(Y)

# Now, X contains the MFCC features, and Y contains the corresponding labels
print("X (MFCC Features):")
print(X)
print("Y (Diagnosis Labels):")
print(Y)

# Ensure that X and Y have consistent shapes
X = np.array(X_padded)
Y = np.array(Y)

# Simpan X (fitur MFCC) dan Y (label diagnosis) ke dalam file CSV
output_csv_file = "./separated_binary_audio_files/HealthyMFCC2.csv"

# Buat DataFrame dengan X dan Y
data = {"MFCC": [mfcc.tolist() for mfcc in X], "Diagnosis": Y}
df = pd.DataFrame(data)

# Simpan DataFrame ke dalam file CSV
df.to_csv(output_csv_file, index=False)

print(f"Data berhasil disimpan ke {output_csv_file}")
