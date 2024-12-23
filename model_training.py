import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import joblib

def main():
    # Load preprocessed data
    print("\nLoading preprocessed data...\n")
    X_train = pd.read_csv("Data/preprocessed_data_X_train.csv")
    X_test = pd.read_csv("Data/preprocessed_data_X_test.csv")
    y_train = pd.read_csv("Data/preprocessed_data_y_train.csv").squeeze()  # Ensure it's a Series
    y_test = pd.read_csv("Data/preprocessed_data_y_test.csv").squeeze()

    # Random Forest Model
    print("\nTraining Random Forest model...\n")
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Models/random_forest_model.pkl", "Random Forest")

    # XGBoost Model
    print("\nTraining XGBoost model...\n")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    evaluate_model(xgb_model, X_test, y_test, "Models/xgboost_model.pkl", "XGBoost")

    # LightGBM Model
    print("\nTraining LightGBM model...\n")
    # Clean column names for LightGBM compatibility
    X_train.columns = X_train.columns.str.replace(r'[^A-Za-z0-9_]', '_', regex=True)
    X_test.columns = X_test.columns.str.replace(r'[^A-Za-z0-9_]', '_', regex=True)

    # Verify sanitized column names
    print("Sanitized column names:")
    print(X_train.columns)

    lgbm_model = lgb.LGBMClassifier(random_state=42)
    lgbm_model.fit(X_train, y_train)
    evaluate_model(lgbm_model, X_test, y_test, "Models/lightgbm_model.pkl", "LightGBM")

def evaluate_model(model, X_test, y_test, model_path, model_name):
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Results:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    joblib.dump(model, model_path)
    print(f"{model_name} model saved as '{model_path}'\n")

if __name__ == "__main__":
    main()
