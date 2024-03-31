import os

output_folder = "./separated_audio_files1"

# Create a dictionary to store the count of files for each diagnosis
diagnosis_counts = {}

# Iterate through the diagnosis folders
for diagnosis_folder in os.listdir(output_folder):
    if os.path.isdir(os.path.join(output_folder, diagnosis_folder)):
        diagnosis_files = os.listdir(os.path.join(output_folder, diagnosis_folder))
        count = len(diagnosis_files)
        diagnosis_counts[diagnosis_folder] = count

# Print the data distribution
print("Data Distribution by Diagnosis:")
for diagnosis, count in diagnosis_counts.items():
    print(f"{diagnosis}: {count} files")
