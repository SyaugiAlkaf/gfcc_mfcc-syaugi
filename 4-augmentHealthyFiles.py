import os
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import librosa
import soundfile as sf
import shutil

# Inisialisasi augmentor
augmentor = Compose([
    AddGaussianNoise(p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(p=0.5),
])

# Path to the healthy audio and text files
input_folder = "./separated_binary_audio_files/Healthy"

# Define the output folder for augmented files
output_folder = "./separated_binary_audio_files/Healthy_augmented"

# Number of augmentations per file
num_augmentations = 3

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through the healthy audio files
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".wav"):
            # Load the original audio file using librosa
            audio_path = os.path.join(root, file)
            audio, sample_rate = librosa.load(audio_path, sr=None)

            # Create a text file copy
            text_path = audio_path.replace(".wav", ".txt")
            text_copy_path = os.path.join(output_folder, os.path.basename(text_path))
            shutil.copy(text_path, text_copy_path)

            # Perform augmentations
            for i in range(num_augmentations):
                augmented_audio = augmentor(samples=audio, sample_rate=sample_rate)
                augmented_audio_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_aug_{i}.wav")

                # Save the augmented audio using soundfile
                sf.write(augmented_audio_path, augmented_audio, sample_rate)

print("Healthy files augmented using custom augmentation and text file copies created.")
