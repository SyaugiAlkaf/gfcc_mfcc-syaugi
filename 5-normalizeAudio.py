from pydub import AudioSegment
import os

# Directory containing the audio files to be normalized
input_folder = "./separated_binary_audio_files"

# Loop through the audio files in the directory
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".wav"):
            audio_path = os.path.join(root, file)

            # Load the audio file using pydub
            audio = AudioSegment.from_file(audio_path)

            # Normalize the audio
            normalized_audio = audio.normalize()

            # Save the normalized audio back to the same file
            normalized_audio.export(audio_path, format="wav")

print("Audio files in the directory have been normalized.")
