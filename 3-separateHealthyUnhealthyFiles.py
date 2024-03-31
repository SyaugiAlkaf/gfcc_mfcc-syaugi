import os
import random

output_folder = "./separated_audio_files"
output_healthy_folder = "./separated_audio_files/healthy"
output_unhealthy_folder = "./separated_audio_files/unhealthy"

# Create output folders for healthy and unhealthy files
os.makedirs(output_healthy_folder, exist_ok=True)
os.makedirs(output_unhealthy_folder, exist_ok=True)

# List of abnormal diagnoses
abnormal_diagnoses = ["Asthma", "Bronchiectasis", "Bronchiolitis", "Pneumonia", "URTI"]

# Percentage of COPD files to take
percentage_to_take = 0.05  # 5%

# Iterate through the diagnosis folders
for diagnosis_folder in os.listdir(output_folder):
    if os.path.isdir(os.path.join(output_folder, diagnosis_folder)):
        diagnosis_files = os.listdir(os.path.join(output_folder, diagnosis_folder))

        # Determine the category (healthy or unhealthy)
        if diagnosis_folder == "Healthy":
            category = "healthy"
        elif diagnosis_folder in abnormal_diagnoses:
            category = "unhealthy"
        elif diagnosis_folder == "COPD":
            # Take a random sample of files from COPD
            num_files = len(diagnosis_files)
            num_to_take = int(num_files * percentage_to_take)
            sampled_files = random.sample(diagnosis_files, num_to_take)

            for file in sampled_files:
                src_path = os.path.join(output_folder, diagnosis_folder, file)
                dest_path = os.path.join(output_unhealthy_folder, file)
                os.rename(src_path, dest_path)

            continue
        else:
            category = "unhealthy"

        # Move the files to the corresponding category folder
        for file in diagnosis_files:
            src_path = os.path.join(output_folder, diagnosis_folder, file)
            dest_folder = os.path.join(output_healthy_folder if category == "healthy" else output_unhealthy_folder)
            dest_path = os.path.join(dest_folder, file)
            os.rename(src_path, dest_path)

print("Files separated into 'healthy' and 'unhealthy' categories.")
