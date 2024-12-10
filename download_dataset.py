import os

# Define the dataset identifier and download path
dataset_identifier = "samahsadiq/benign-and-malicious-qr-codes"
download_path = "Quishing/Data"  # Update this path

# Download and unzip the dataset
os.system(f"kaggle datasets download -d {dataset_identifier} --unzip -p {download_path}")

print("Dataset downloaded successfully!")
