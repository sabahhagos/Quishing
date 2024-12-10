import pandas as pd
import tldextract

# Load dataset
input_file = 'Data/fixed_features_extracted.csv'
output_file = 'Data/updated_features_with_tld.csv'

# Function to extract TLD and subdomain count
def extract_tld_features(url):
    extracted = tldextract.extract(url)
    tld = extracted.suffix if extracted.suffix else 'unknown'
    subdomain_count = len(extracted.subdomain.split('.')) if extracted.subdomain else 0
    return tld, subdomain_count

# Read the dataset
data = pd.read_csv(input_file)

# Extract TLD and Subdomain Count
tld_features = data['url'].apply(lambda x: extract_tld_features(str(x)) if pd.notna(x) else ('unknown', 0))
data['top_level_domain'], data['subdomain_count'] = zip(*tld_features)

# Save updated data
data.to_csv(output_file, index=False)
print(f"Updated dataset saved to {output_file}")
