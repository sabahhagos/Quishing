import pandas as pd
import re
from urllib.parse import urlparse
from collections import Counter
from math import log2

# Function to calculate entropy
def calculate_entropy(url):
    """
    Calculate the entropy of a given URL.
    """
    counts = Counter(url)
    length = len(url)
    return -sum((count / length) * log2(count / length) for count in counts.values())

# Function to extract features from a URL
def extract_features(url):
    """
    Extracts features from a given URL.
    """
    if pd.isna(url) or not isinstance(url, str):
        # Handle missing or invalid URLs
        return {
            "url_length": 0,
            "dot_count": 0,
            "slash_count": 0,
            "protocol_count": 0,
            "@_sign_count": 0,
            "hyphen_count": 0,
            "special_char_count": 0,
            "digit_count": 0,
            "top_level_domain": "unknown",
            "subdomain_count": 0,
            "url_entropy": 0.0,
        }

    # Clean URL if it contains unwanted metadata
    url = re.search(r"(http[s]?://[^\s]+)", url)
    if url:
        url = url.group(0)
    else:
        return {
            "url_length": 0,
            "dot_count": 0,
            "slash_count": 0,
            "protocol_count": 0,
            "@_sign_count": 0,
            "hyphen_count": 0,
            "special_char_count": 0,
            "digit_count": 0,
            "top_level_domain": "unknown",
            "subdomain_count": 0,
            "url_entropy": 0.0,
        }

    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path

    features = {
        "url_length": len(url),
        "dot_count": url.count('.'),
        "slash_count": url.count('/'),
        "protocol_count": url.count('://'),
        "@_sign_count": url.count('@'),
        "hyphen_count": url.count('-'),
        "special_char_count": len(re.findall(r'[#&=]', url)),
        "digit_count": sum(char.isdigit() for char in url),
        "top_level_domain": domain.split('.')[-1] if '.' in domain else "unknown",
        "subdomain_count": domain.count('.') - 1 if domain else 0,  # Exclude TLD
        "url_entropy": calculate_entropy(url)
    }
    return features

def main():
    # Input and output files
    input_file = "Data/decoded_urls.csv"
    output_file = "Data/updated_features_extracted.csv"

    # Load the dataset
    df = pd.read_csv(input_file)

    # Clean the 'url' column and extract features
    df['url'] = df['url'].str.extract(r"(http[s]?://[^\s]+)")  # Clean the URL
    features = df['url'].apply(extract_features).apply(pd.Series)

    # Combine the original data with the extracted features
    updated_df = pd.concat([df, features], axis=1)

    # Save the updated dataset
    updated_df.to_csv(output_file, index=False)
    print(f"Features extracted and saved to {output_file}")

if __name__ == "__main__":
    main()
