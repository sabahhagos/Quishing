import pandas as pd

# File path
input_file = "Data/updated_features_extracted.csv"

# Load the dataset
data = pd.read_csv(input_file)

# Display first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Check data types
print("\nData types of each column:")
print(data.dtypes)

# Display summary statistics for numerical features
print("\nSummary statistics for numerical features:")
print(data.describe())

# Check unique values in the 'label' column
print("\nUnique values in the 'label' column:")
print(data['label'].unique())

# Analyze the distribution of top-level domains
print("\nTop-Level Domains (TLD) distribution:")
print(data['top_level_domain'].value_counts().head(10))  # Show top 10 most frequent TLDs

# Verify that numerical columns have valid values
numerical_features = [
    'url_length', 'dot_count', 'slash_count', 'protocol_count', '@_sign_count',
    'hyphen_count', 'special_char_count', 'digit_count', 'subdomain_count', 'url_entropy'
]

print("\nChecking for negative values in numerical features:")
for feature in numerical_features:
    if (data[feature] < 0).any():
        print(f"Warning: Negative values found in {feature}")
    else:
        print(f"{feature} has no negative values.")
