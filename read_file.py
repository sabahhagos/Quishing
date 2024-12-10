import pandas as pd
try:
    file_path = "Data/updated_features_extracted.csv"
    data = pd.read_csv(file_path)
    print("File read successfully. Shape:", data.shape)
except Exception as e:
    print("Error:", e)
