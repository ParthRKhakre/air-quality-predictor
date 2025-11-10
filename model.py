import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# --- Configuration ---
FILE_PATH = 'AirQuality.csv'
MODEL_FILENAME = 'rf_air_quality_model.joblib'
METRICS_FILENAME = 'model_metrics.txt'
# Selected features for the model and user input
FEATURES = ['PT08.S1(CO)', 'C6H6(GT)', 'NOx(GT)', 'T', 'RH', 'AH']
TARGET = 'CO_Next_Day'

def load_and_preprocess_data(file_path):
    """Loads, cleans, and aggregates the hourly air quality data to daily data."""
    
    # 1. Load and Initial Cleaning
    df = pd.read_csv(file_path, sep=';', na_values=['-200', 'nan'], engine='python')
    # Drop the typically empty last two columns
    df = df.iloc[:, :-2]
    df.columns = df.columns.str.strip()

    # Helper to clean and convert to numeric (handling comma as decimal separator)
    def clean_and_convert(col):
        if df[col].dtype == 'object':
            return df[col].astype(str).str.replace(',', '.', regex=False).replace('nan', np.nan).astype(float, errors='ignore')
        return df[col]

    for col in df.columns:
        if col not in ['Date', 'Time']:
            df[col] = clean_and_convert(col)

    # Create DateTime index and drop original columns
    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')
    df.set_index('DateTime', inplace=True)
    # Drop NMHC(GT) due to too many NaNs, and original time columns
    df.drop(columns=['Date', 'Time', 'NMHC(GT)'], inplace=True) 

    # Drop rows where the index conversion failed (NaT)
    df = df[df.index.notna()]

    # 2. Daily Aggregation
    # Resample to daily average
    df_daily = df.resample('D').mean()
    
    # 3. Feature Engineering: Next Day Prediction
    # Target is the CO(GT) of the next day (shift target back 1 day)
    df_daily[TARGET] = df_daily['CO(GT)'].shift(-1)
    
    # 4. Final Cleanup: Drop rows with any remaining NaN values
    df_final = df_daily.dropna()

    return df_final

def train_and_evaluate_model(df_final):
    """Trains a Random Forest Regressor and evaluates its performance."""
    
    # Prepare data
    X = df_final[FEATURES]
    y = df_final[TARGET]

    # Split data (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model (Random Forest Regressor)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_split=5)
    model.fit(X_train, y_train)

    # Predict and Evaluate
    y_pred = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

    return model, metrics

def save_artifacts(model, metrics):
    """Saves the trained model and evaluation metrics."""
    joblib.dump(model, MODEL_FILENAME)
    
    with open(METRICS_FILENAME, 'w') as f:
        for key, value in metrics.items():
            f.write(f'{key}: {value}\n')

def make_prediction(model, new_data: dict):
    """
    Makes a prediction on new data (single dictionary input).
    Expects keys from the FEATURES list.
    """
    # Convert input dict to DataFrame
    input_df = pd.DataFrame([new_data], columns=FEATURES)
    # Predict
    prediction = model.predict(input_df)[0]
    return prediction

if __name__ == '__main__':
    # Simple check to avoid re-training if files exist
    if not os.path.exists(MODEL_FILENAME) or not os.path.exists(METRICS_FILENAME):
        print("Starting air quality model training...")
        df_processed = load_and_preprocess_data(FILE_PATH)
        
        if df_processed.empty:
            print("Error: Processed data is empty. Cannot train model.")
        else:
            print(f"Processed data shape: {df_processed.shape}")
            # Train and save
            trained_model, performance_metrics = train_and_evaluate_model(df_processed)
            save_artifacts(trained_model, performance_metrics)
            print("Model training and saving complete.")
    else:
        print("Model and metrics files already exist. Skipping training.")