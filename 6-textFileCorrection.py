import os
import shutil

# Path to the directory containing augmented audio files
augmented_audio_folder = "./separated_binary_audio_files/Healthy_augmented"

# Loop through the augmented audio files
for root, dirs, files in os.walk(augmented_audio_folder):
    for file in files:
        if file.endswith(".txt"):
            # Determine the corresponding text file name
            base_name = os.path.splitext(file)[0]
            text_file = base_name + ".txt"

            # Create copies of the text file for each augmentation
            for i in range(3):  # Change this number to match the number of augmentations
                augmented_text_file = f"{base_name}_aug_{i}.txt"
                augmented_text_path = os.path.join(augmented_audio_folder, augmented_text_file)

                # Copy the text file
                shutil.copy(os.path.join(augmented_audio_folder, text_file), augmented_text_path)

print("Text files copied and renamed for augmented audio.")
