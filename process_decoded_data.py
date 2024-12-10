import pandas as pd

# Load the decoded URLs
decoded_data = pd.read_csv("Quishing/Data/decoded_urls.csv")

# Preview the first few rows of the data
print("Decoded Data Preview:")
print(decoded_data.head())
