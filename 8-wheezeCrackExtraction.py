import os
import pandas as pd

# Directory containing the text files
txt_directory = "./separated_binary_audio_files"

# Output directory for the consolidated CSV file
output_directory = "./separated_binary_audio_files"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Create an empty list to store data
data = []

# Process text files in each diagnosis folder
diagnoses = os.listdir(txt_directory)

for diagnosis in diagnoses:
    diagnosis_dir = os.path.join(txt_directory, diagnosis)
    if not os.path.isdir(diagnosis_dir):
        continue

    for filename in os.listdir(diagnosis_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(diagnosis_dir, filename)

            # Initialize counters for presence and absence of crackles and wheezes
            presence_crackles_count = 0
            absence_crackles_count = 0
            presence_wheezes_count = 0
            absence_wheezes_count = 0

            with open(file_path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    _, _, crackles, wheezes = line.strip().split("\t")
                    if crackles == "1":
                        presence_crackles_count += 1
                    else:
                        absence_crackles_count += 1
                    if wheezes == "1":
                        presence_wheezes_count += 1
                    else:
                        absence_wheezes_count += 1

            # Add the data to the list
            data.append([presence_crackles_count, absence_crackles_count, presence_wheezes_count, absence_wheezes_count, diagnosis])

# Create a DataFrame from the data list
df = pd.DataFrame(data, columns=["Presence of Crackles", "Absence of Crackles", "Presence of Wheezes", "Absence of Wheezes", "Diagnosis"])

# Save the consolidated counts to a CSV file
output_file = os.path.join(output_directory, "Crackles2.csv")
df.to_csv(output_file, index=False)

print(f"Consolidated counts saved to {output_file}")
