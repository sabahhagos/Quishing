import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_data(input_file, output_file):
    print("Loading data...")
    data = pd.read_csv(input_file)
    print(f"Data loaded successfully.\nData shape: {data.shape}")
    print(f"Columns: {list(data.columns)}\n")
    
    print("Checking for missing values...")
    print(data.isnull().sum())
    
    # Drop any rows with missing URLs, labels, or top_level_domain
    data = data.dropna(subset=["url", "label", "top_level_domain"])
    print(f"Missing values handled. Shape after cleaning: {data.shape}\n")

    # Encode the target variable (label)
    label_mapping = {"benign": 0, "malicious": 1}
    data["label"] = data["label"].map(label_mapping)

    # Select features for the model
    features = [
        "url_length", "dot_count", "slash_count", "protocol_count",
        "@_sign_count", "hyphen_count", "special_char_count", "digit_count",
        "top_level_domain", "subdomain_count", "url_entropy"
    ]
    X = data[features]
    y = data["label"]

    # Handle categorical features (e.g., top_level_domain)
    print("Encoding categorical features...")
    categorical_features = ["top_level_domain"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_categorical = encoder.fit_transform(X[categorical_features])
    encoded_categorical_df = pd.DataFrame(
        encoded_categorical, 
        columns=encoder.get_feature_names_out(categorical_features)
    )
    
    # Remove the original categorical column and add the encoded features
    X = X.drop(columns=categorical_features).reset_index(drop=True)
    X = pd.concat([X, encoded_categorical_df], axis=1)
    print("Encoding complete.")

    # Standardize numerical features
    numerical_features = X.select_dtypes(include=["float64", "int64"]).columns
    print(f"Standardizing numerical features: {numerical_features}")
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    # Split the data into train and test sets
    print("Splitting the data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save preprocessed datasets
    X_train.to_csv(f"{output_file}_X_train.csv", index=False)
    X_test.to_csv(f"{output_file}_X_test.csv", index=False)
    y_train.to_csv(f"{output_file}_y_train.csv", index=False)
    y_test.to_csv(f"{output_file}_y_test.csv", index=False)

    print(f"Data preprocessed and saved to {output_file}")

if __name__ == "__main__":
    input_file = "Data/updated_features_extracted.csv"
    output_file = "Data/preprocessed_data"
    preprocess_data(input_file, output_file)
