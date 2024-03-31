import os
import csv
import shutil

# Define the directories and file paths
audio_txt_dir = "./respiratorydataset/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files"
diagnosis_csv_path = "./respiratorydataset/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv"
output_folder = "./separated_audio_files"

# Create a dictionary to store patient diagnoses
patient_diagnoses = {}

# Read patient diagnoses from the CSV file
with open(diagnosis_csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        patient_id, diagnosis = int(row[0]), row[1]
        patient_diagnoses[patient_id] = diagnosis

# Iterate through the audio and text files directory
for root, dirs, files in os.walk(audio_txt_dir):
    for file in files:
        if file.endswith(".wav"):
            # Extract patient ID from the audio file name
            patient_id = int(file.split("_")[0])

            # Get the diagnosis for this patient
            if patient_id in patient_diagnoses:
                diagnosis = patient_diagnoses[patient_id]

                # Create a folder for the diagnosis if it doesn't exist
                diagnosis_folder = os.path.join(output_folder, diagnosis)
                os.makedirs(diagnosis_folder, exist_ok=True)

                # Move the audio file to the corresponding folder
                audio_src_path = os.path.join(root, file)
                audio_dest_path = os.path.join(diagnosis_folder, file)

                # Check if the destination file already exists before moving
                if not os.path.exists(audio_dest_path):
                    shutil.move(audio_src_path, audio_dest_path)

                # Find and move the corresponding text file
                text_file = file.replace(".wav", ".txt")
                text_src_path = os.path.join(root, text_file)
                text_dest_path = os.path.join(diagnosis_folder, text_file)

                # Check if the text file exists before moving
                if os.path.exists(text_src_path):
                    shutil.move(text_src_path, text_dest_path)

print("Files organized based on patient diagnosis and saved in the 'separated_audio_files2' folder.")
