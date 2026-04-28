"""
ICP (Intracranial Pressure) Prediction Model
=============================================
XGBoost Regression model for predicting ICP from physiological signals.

Features: HR, HRV, QRS, NPI
Target: ICP
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os
import argparse
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path(__file__).resolve().parent / "train_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "icp_model.joblib"
SCALER_PATH = Path(__file__).resolve().parent / "scaler.joblib"

# Feature and target columns (match CSV headers in XGBoost/train_dataset.csv)
# HR -> HR_bpm, HRV -> HRV_SDNN_ms (or HRV_RMSSD_ms depending on preference)
# QRS -> QRS_duration_ms, Target ICP -> ICP_mmHg
FEATURE_COLS = ["HR_bpm", "HRV_SDNN_ms", "QRS_duration_ms", "NPI"]
TARGET_COL = "ICP_mmHg"

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_data(filepath):
    """Load dataset from CSV file."""
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"Data file not found: {fp}")

    df = pd.read_csv(fp)
    print(f"[INFO] Loaded data: {df.shape[0]} samples, {df.shape[1]} columns")
    return df


def preprocess_data(df):
    """
    Preprocess the dataset:
    - Handle missing values
    - Scale features
    """
    print("\n[INFO] Preprocessing data...")
    
    # Check for missing values
    missing = df[FEATURE_COLS + [TARGET_COL]].isnull().sum()
    print(f"[INFO] Missing values:\n{missing[missing > 0]}")
    
    # Handle missing values - fill with median
    df_clean = df.copy()
    for col in FEATURE_COLS + [TARGET_COL]:
        if df_clean[col].isnull().any():
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"[INFO] Filled missing values in '{col}' with median: {median_val:.4f}")
    
    # Extract features and target
    X = df_clean[FEATURE_COLS].values
    y = df_clean[TARGET_COL].values
    
    print(f"[INFO] Features shape: {X.shape}")
    print(f"[INFO] Target shape: {y.shape}")
    
    return X, y


def scale_features(X_train, X_test):
    """Normalize/Scale features using StandardScaler."""
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"[INFO] Features scaled - Mean: {scaler.mean_}, Std: {scaler.scale_}")
    
    return X_train_scaled, X_test_scaled, scaler

# =============================================================================
# Model Training
# =============================================================================

def train_model(X_train, y_train, use_grid_search=False):
    """
    Train XGBoost Regressor model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    use_grid_search : bool
        Whether to use GridSearchCV for hyperparameter tuning
    
    Returns:
    --------
    model : trained XGBoost model
    """
    print("\n[INFO] Training XGBoost model...")
    
    if use_grid_search:
        print("[INFO] Using GridSearchCV for hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        # Base model
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"[INFO] Best parameters: {grid_search.best_params_}")
        print(f"[INFO] Best CV score (neg MSE): {grid_search.best_score_:.6f}")
        
        model = grid_search.best_estimator_
        
    else:
        # Default hyperparameters
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
            verbosity=1
        )
        
        model.fit(X_train, y_train)
        print("[INFO] Model trained with default hyperparameters")
    
    return model


# =============================================================================
# Model Evaluation
# =============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    
    Parameters:
    -----------
    model : trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    
    Returns:
    --------
    metrics : dict
        Dictionary containing MSE and R2 score
    """
    print("\n[INFO] Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Print results
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"R2 Score: {r2:.6f}")
    print("=" * 50)
    
    # Additional analysis
    print("\n[INFO] Prediction Statistics:")
    print(f"  Actual mean: {y_test.mean():.4f}, std: {y_test.std():.4f}")
    print(f"  Predicted mean: {y_pred.mean():.4f}, std: {y_pred.std():.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

# =============================================================================
# Model Saving/Loading
# =============================================================================

def save_model(model, scaler, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Save trained model and scaler to files."""
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n[INFO] Model saved to: {model_path}")
    print(f"[INFO] Scaler saved to: {scaler_path}")


def load_model(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Load trained model and scaler from files."""
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Model or scaler file not found. Expected {model_path} and {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"[INFO] Model loaded from: {model_path}")
    print(f"[INFO] Scaler loaded from: {scaler_path}")
    return model, scaler

# =============================================================================
# Prediction Function
# =============================================================================

def predict_icp(hr, hrv, qrs, npi, model=None, scaler=None):
    """
    Predict ICP from new input values.
    
    Parameters:
    -----------
    hr : float
        Heart Rate
    hrv : float
        Heart Rate Variability
    qrs : float
        QRS complex duration
    npi : float
        Neural Property Index
    model : trained XGBoost model, optional
        If None, loads from file
    scaler : fitted StandardScaler, optional
        If None, loads from file
    
    Returns:
    --------
    icp_pred : float
        Predicted Intracranial Pressure
    """
    # Load model if not provided
    if model is None or scaler is None:
        model, scaler = load_model()
    
    # Prepare input
    input_features = np.array([[hr, hrv, qrs, npi]])
    
    # Scale features
    input_scaled = scaler.transform(input_features)
    
    # Predict
    icp_pred = model.predict(input_scaled)[0]
    
    return icp_pred


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution block."""
    print("=" * 60)
    print("ICP Prediction Model - XGBoost Regression")
    print("=" * 60)
    
    try:
        # Step 1: Load data
        print("\n[STEP 1] Loading data...")
        df = load_data(DATA_PATH)
        
        # Display data info
        print("\n[INFO] Dataset info:")
        print(df[FEATURE_COLS + [TARGET_COL]].describe())
        
        # Step 2: Preprocess data
        print("\n[STEP 2] Preprocessing data...")
        X, y = preprocess_data(df)
        
        # Step 3: Split data
        print("\n[STEP 3] Splitting data (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"[INFO] Training set: {X_train.shape[0]} samples")
        print(f"[INFO] Test set: {X_test.shape[0]} samples")
        
        # Step 4: Scale features
        print("\n[STEP 4] Scaling features...")
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        # Step 5: Train model
        print("\n[STEP 5] Training model...")
        # Set use_grid_search=True for hyperparameter tuning (slower)
        model = train_model(X_train_scaled, y_train, use_grid_search=False)
        
        # Step 6: Evaluate model
        print("\n[STEP 6] Evaluating model...")
        metrics = evaluate_model(model, X_test_scaled, y_test)
        
        # Step 7: Save model
        print("\n[STEP 7] Saving model...")
        save_model(model, scaler)
        
        # Step 8: Demo prediction
        print("\n[STEP 8] Demo predictions...")
        print("-" * 40)
        
        # Example predictions using the first few test samples
        demo_inputs = X_test[:5]
        demo_actual = y_test[:5]
        
        for i, (inputs, actual) in enumerate(zip(demo_inputs, demo_actual)):
            hr, hrv, qrs, npi = inputs
            pred = model.predict(scaler.transform([inputs]))[0]
            print(f"Sample {i+1}: HR={hr:.2f}, HRV={hrv:.2f}, QRS={qrs:.2f}, NPI={npi:.2f}")
            print(f"         Actual ICP: {actual:.2f}, Predicted ICP: {pred:.2f}")
            print("-" * 40)
        
        print("\n[SUCCESS] Training complete!")
        print("=" * 60)
        
        # Return model and scaler for further use
        return model, scaler, metrics
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print(f"\n[INFO] Please ensure the CSV dataset exists and the path is correct.")
        print("[INFO] Expected columns: HR, HRV, QRS, NPI, ICP")
        raise
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost ICP prediction model.")
    parser.add_argument("--data", "-d", default=str(DATA_PATH), help="Path to training CSV file")
    parser.add_argument("--model", "-m", default=str(MODEL_PATH), help="Output path for trained model")
    parser.add_argument("--scaler", "-s", default=str(SCALER_PATH), help="Output path for scaler")
    parser.add_argument("--grid", action="store_true", help="Run GridSearchCV for hyperparameter tuning")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--demo", action="store_true", help="Run demo predictions after training")

    args = parser.parse_args()

    # Update global paths from CLI
    DATA_PATH = Path(args.data)
    MODEL_PATH = Path(args.model)
    SCALER_PATH = Path(args.scaler)

    # Run main pipeline
    model, scaler, metrics = main()

    if args.demo:
        print("\n" + "=" * 60)
        print("USAGE EXAMPLE - Predicting new ICP values")
        print("=" * 60)
        new_samples = [
            (75, 45, 0.08, 0.5),
            (80, 50, 0.09, 0.6),
            (70, 40, 0.07, 0.4),
        ]
        for hr, hrv, qrs, npi in new_samples:
            icp = predict_icp(hr, hrv, qrs, npi, model, scaler)
            print(f"Input: HR={hr}, HRV={hrv}, QRS={qrs}, NPI={npi} -> Predicted ICP: {icp:.2f}")