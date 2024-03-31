import pandas as pd

# Load the MFCC data CSV
mfcc_csv = pd.read_csv("./separated_binary_audio_files/HealthyMFCC2.csv")

# Load the counts data CSV
counts_csv = pd.read_csv("./separated_binary_audio_files/Crackles2.csv")

# Merge the two DataFrames based on the index (sequence)
merged_data = pd.concat([mfcc_csv, counts_csv], axis=1)

# Save the merged data to a new CSV file
merged_output_file = "./separated_binary_audio_files/merged_data2.csv"
merged_data.to_csv(merged_output_file, index=False)

print(f"Merged data saved to {merged_output_file}")
