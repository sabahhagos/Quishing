import os
import csv
import pandas as pd
from multiprocessing import Pool, cpu_count
from pyzbar.pyzbar import decode
from PIL import Image

# Paths
input_dir = "Data/QR codes"  # Folder containing QR codes
output_file = "Data/decoded_urls.csv"
cleaned_output_file = "Data/cleaned_decoded_urls.csv"

# Function to decode a single QR code
def decode_qr(file_path):
    try:
        img = Image.open(file_path)
        decoded_data = decode(img)
        if decoded_data:
            url = decoded_data[0].data.decode('utf-8')  # Assuming single QR code per image
            label = "malicious" if "malicious" in file_path else "benign"
            return {"file": os.path.basename(file_path), "url": url, "label": label}
        else:
            return {"file": os.path.basename(file_path), "url": "No QR code detected", "label": "unknown"}
    except Exception as e:
        return {"file": os.path.basename(file_path), "url": f"Error: {str(e)}", "label": "error"}

# Function to process files in parallel
def process_files(file_list):
    with Pool(cpu_count()) as pool:
        results = pool.map(decode_qr, file_list)
    return results

# Main function
def main():
    # Get list of all QR code files
    file_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".png"):  # Adjust for your file type
                file_list.append(os.path.join(root, file))
    
    # Decode QR codes in parallel
    results = process_files(file_list)

    # Write results to CSV
    fieldnames = ["file", "url", "label"]
    with open(output_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Decoded URLs saved to {output_file}")

    # Cleaning logic
    print("Cleaning the decoded URLs...")
    df = pd.read_csv(output_file)
    df['url'] = df['url'].str.extract(r'(\bhttps?://\S+)')  # Extract valid URLs
    df['url'] = df['url'].str.strip()  # Remove extra whitespace
    df.to_csv(cleaned_output_file, index=False)
    print(f"Cleaned data saved to {cleaned_output_file}")

if __name__ == "__main__":
    main()
