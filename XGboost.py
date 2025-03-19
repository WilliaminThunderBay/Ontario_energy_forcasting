import pandas as pd
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, recall_score, precision_score, f1_score,
                             confusion_matrix)
import xgboost as xgb

def load_target_series(merged_file, target='hourly_demand'):
    """
    Load the merged data and return the target time series with an hourly frequency.
    Missing values are filled forward.
    """
    df = pd.read_csv(merged_file, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    ts = df[target].asfreq('H').ffill()
    return ts

def create_lag_features(ts, target='hourly_demand', lags=24):
    """
    Create lag features based on the target time series.
    Each row will include the target values for the previous 'lags' hours as features,
    used to predict the target value at the current time.
    """
    df = pd.DataFrame(ts)
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df[target].shift(i)
    df.dropna(inplace=True)
    return df

def train_xgboost(ts, target='hourly_demand', lags=24, forecast_start='2023-01-01', forecast_end='2023-04-03'):
    # Create lag features data
    df_features = create_lag_features(ts, target=target, lags=lags)
    
    # Split into training and testing sets based on time
    train_df = df_features[df_features.index < forecast_start]
    test_df  = df_features[(df_features.index >= forecast_start) & (df_features.index <= forecast_end)]
    
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test  = test_df.drop(columns=[target])
    y_test  = test_df[target]
    
    # Use XGBoost for regression
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    # Calculate regression metrics
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    r2  = r2_score(y_test, pred)
    
    print("[XGBoost] Regression Metrics:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  R²:  {r2:.2f}")
    
    # Convert the regression problem into a binary classification problem using the mean of the training set as threshold
    threshold = y_train.mean()
    y_test_class = (y_test >= threshold).astype(int)
    pred_class = (pred >= threshold).astype(int)
    
    acc  = accuracy_score(y_test_class, pred_class)
    rec  = recall_score(y_test_class, pred_class)
    prec = precision_score(y_test_class, pred_class)
    f1   = f1_score(y_test_class, pred_class)
    cm   = confusion_matrix(y_test_class, pred_class)
    
    print("\n[XGBoost] Classification Metrics (Threshold = {:.2f}):".format(threshold))
    print(f"  Accuracy:  {acc:.2f}")
    print(f"  Recall:    {rec:.2f}")
    print(f"  Precision: {prec:.2f}")
    print(f"  F1 Score:  {f1:.2f}")
    print("  Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    merged_file = "merged_data.csv"  # Ensure this file exists and contains a 'datetime' column and the target field
    target = "hourly_demand"
    ts = load_target_series(merged_file, target=target)
    train_xgboost(ts, target=target, lags=24, forecast_start='2023-01-01', forecast_end='2023-04-03')
