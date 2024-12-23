import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

def load_preprocessed_data():
    """
    Load the preprocessed training data for model tuning.
    """
    X_train = pd.read_csv("Data/preprocessed_data_X_train.csv")
    y_train = pd.read_csv("Data/preprocessed_data_y_train.csv").squeeze()  # Squeeze to get a Series
    return X_train, y_train

def tune_random_forest(X_train, y_train):
    """
    Perform hyperparameter tuning for Random Forest.
    """
    rf_param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }

    rf_model = RandomForestClassifier(random_state=42)
    rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, 
                                  scoring='f1', cv=3, verbose=2, n_jobs=-1)
    rf_grid_search.fit(X_train, y_train)
    print("Best Random Forest Parameters:", rf_grid_search.best_params_)

    # Save the tuned model
    joblib.dump(rf_grid_search.best_estimator_, "Models/tuned_random_forest.pkl")
    print("Tuned Random Forest model saved to 'Models/tuned_random_forest.pkl'.")

def tune_xgboost(X_train, y_train):
    """
    Perform hyperparameter tuning for XGBoost.
    """
    xgb_param_grid = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 10],
        'min_child_weight': [1, 5, 10],
        'scale_pos_weight': [1, len(y_train[y_train == 0]) / len(y_train[y_train == 1])]
    }

    xgb_model = XGBClassifier(use_label_encoder=False, random_state=42, eval_metric='logloss')
    xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, 
                                   scoring='f1', cv=3, verbose=2, n_jobs=-1)
    xgb_grid_search.fit(X_train, y_train)
    print("Best XGBoost Parameters:", xgb_grid_search.best_params_)

    # Save the tuned model
    joblib.dump(xgb_grid_search.best_estimator_, "Models/tuned_xgboost.pkl")
    print("Tuned XGBoost model saved to 'Models/tuned_xgboost.pkl'.")

def tune_lightgbm(X_train, y_train):
    """
    Perform hyperparameter tuning for LightGBM.
    """
    lgbm_param_grid = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100],
        'min_child_samples': [20, 50, 100],
        'scale_pos_weight': [1, len(y_train[y_train == 0]) / len(y_train[y_train == 1])]
    }

    lgbm_model = LGBMClassifier(random_state=42)
    lgbm_grid_search = GridSearchCV(estimator=lgbm_model, param_grid=lgbm_param_grid, 
                                    scoring='f1', cv=3, verbose=2, n_jobs=-1)
    lgbm_grid_search.fit(X_train, y_train)
    print("Best LightGBM Parameters:", lgbm_grid_search.best_params_)

    # Save the tuned model
    joblib.dump(lgbm_grid_search.best_estimator_, "Models/tuned_lightgbm.pkl")
    print("Tuned LightGBM model saved to 'Models/tuned_lightgbm.pkl'.")

if __name__ == "__main__":
    X_train, y_train = load_preprocessed_data()

    print("Tuning Random Forest...")
    tune_random_forest(X_train, y_train)

    print("\nTuning XGBoost...")
    tune_xgboost(X_train, y_train)

    print("\nTuning LightGBM...")
    tune_lightgbm(X_train, y_train)