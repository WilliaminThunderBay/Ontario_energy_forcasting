import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import pandas as pd
import numpy as np
from datetime import timedelta
from io import StringIO

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, recall_score, precision_score, f1_score,
                             confusion_matrix)
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#############################################
# 1. Merge Electricity Demand and Weather Data
#############################################
def merge_demand_and_weather(demand_file, weather_file, merged_file='merged_data.csv'):
    df_demand = pd.read_csv(demand_file)
    df_weather = pd.read_csv(weather_file).rename(columns={'Date/Time (LST)': 'datetime'})
    
    def convert_datetime(row):
        if row['hour'] == 24:
            dt = pd.to_datetime(row['date']) + timedelta(days=1)
            return dt.strftime('%Y-%m-%d') + ' 00:00:00'
        else:
            return f"{row['date']} {str(row['hour']).zfill(2)}:00:00"
    
    df_demand['datetime'] = pd.to_datetime(df_demand.apply(convert_datetime, axis=1))
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    
    merged_df = pd.merge(df_demand, df_weather, on='datetime', how='inner').dropna()
    merged_df.to_csv(merged_file, index=False)
    output = f"[INFO] Merged data saved to {merged_file}\n"
    return merged_df.set_index('datetime'), output

#############################################
# 2. Hard-coded Population Data and Conversion
#############################################
def load_population_data_from_string():
    pop_csv = r'''\
"Geography","Q1 2003","Q2 2003","Q3 2003","Q4 2003","Q1 2004","Q2 2004","Q3 2004","Q4 2004","Q1 2005","Q2 2005","Q3 2005","Q4 2005","Q1 2006","Q2 2006","Q3 2006","Q4 2006","Q1 2007","Q2 2007","Q3 2007","Q4 2007","Q1 2008","Q2 2008","Q3 2008","Q4 2008","Q1 2009","Q2 2009","Q3 2009","Q4 2009","Q1 2010","Q2 2010","Q3 2010","Q4 2010","Q1 2011","Q2 2011","Q3 2011","Q4 2011","Q1 2012","Q2 2012","Q3 2012","Q4 2012","Q1 2013","Q2 2013","Q3 2013","Q4 2013","Q1 2014","Q2 2014","Q3 2014","Q4 2014","Q1 2015","Q2 2015","Q3 2015","Q4 2015","Q1 2016","Q2 2016","Q3 2016","Q4 2016","Q1 2017","Q2 2017","Q3 2017","Q4 2017","Q1 2018","Q2 2018","Q3 2018","Q4 2018","Q1 2019","Q2 2019","Q3 2019","Q4 2019","Q1 2020","Q2 2020","Q3 2020","Q4 2020","Q1 2021","Q2 2021","Q3 2021","Q4 2021","Q1 2022","Q2 2022","Q3 2022","Q4 2022","Q1 2023","Q2 2023","Q3 2023","Q4 2023"
,"Persons",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
"Ontario","12,155,691","12,194,269","12,243,641","12,290,116","12,303,516","12,339,813","12,389,641","12,435,931","12,444,755","12,476,560","12,527,581","12,578,240","12,587,149","12,618,321","12,661,953","12,701,324","12,703,327","12,727,090","12,765,133","12,807,497","12,814,686","12,840,482","12,883,824","12,927,520","12,932,742","12,956,924","12,998,941","13,048,442","13,059,426","13,088,924","13,136,481","13,189,987","13,199,081","13,222,146","13,262,345","13,310,604","13,325,337","13,350,123","13,392,364","13,436,625","13,446,276","13,469,151","13,511,902","13,559,499","13,563,311","13,583,220","13,617,763","13,661,282","13,657,423","13,669,290","13,709,293","13,759,762","13,774,364","13,816,652","13,876,500","13,948,180","13,975,516","14,012,209","14,078,499","14,161,084","14,199,811","14,251,136","14,326,746","14,413,055","14,449,986","14,493,612","14,573,565","14,666,727","14,718,155","14,752,374","14,761,811","14,757,582","14,772,726","14,808,093","14,842,488","14,938,314","14,995,433","15,042,458","15,141,455","15,289,550","15,402,095","15,478,287","15,623,207","15,818,465"
'''
    print("[INFO] Reading hard-coded population data...")
    pop_df = pd.read_csv(StringIO(pop_csv), skiprows=2, header=None)
    num_cols = pop_df.shape[1]
    cols = ["Geography"] + [f"Q{i}" for i in range(1, num_cols)]
    pop_df.columns = cols
    pop_df = pop_df[pop_df["Geography"].str.contains("Ontario", na=False)]
    pop_long = pop_df.melt(id_vars=["Geography"], var_name="Quarter", value_name="population")
    pop_long["population"] = pop_long["population"].replace({',':''}, regex=True)
    pop_long["population"] = pd.to_numeric(pop_long["population"], errors='coerce')
    # For simplicity, set all data to a fixed date, e.g., 2013-01-01
    pop_long["date"] = pd.to_datetime("2013-01-01")
    pop_long.set_index("date", inplace=True)
    # Group by index and take mean to ensure unique index
    pop_series = pop_long["population"].groupby(pop_long.index).mean()
    return pop_series

#############################################
# 3. Merge population data into merged_df
#############################################
def merge_population(merged_df, pop_series):
    new_index = pd.date_range(start=merged_df.index.min(), end=merged_df.index.max(), freq='h')
    pop_hourly = pop_series.resample('h').ffill()
    pop_hourly = pop_hourly.reindex(new_index, method='ffill')
    pop_hourly.name = 'population'
    
    merged_df = merged_df.reindex(new_index)
    merged_df['population'] = pop_hourly
    merged_df.dropna(subset=['population'], inplace=True)
    return merged_df

#############################################
# 4. Data Preprocessing
#############################################
def load_preprocessed_data(file_path, features, target='hourly_demand', do_log=True):
    df = pd.read_csv(file_path, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    df = df.dropna(subset=features+[target])
    
    if do_log:
        df['log_demand'] = np.log1p(df[target])
        y_col = 'log_demand'
    else:
        y_col = target
    X = df[features]
    y = df[y_col]
    return df, X, y

#############################################
# 5. Train XGBoost Model with GridSearchCV and Save Model
#############################################
def train_xgboost_grid(X, y):
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }
    xgb_model = xgb.XGBRegressor(random_state=42)
    
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, 
                               scoring='r2', cv=3, verbose=2, n_jobs=-1)
    output = "[INFO] Starting GridSearchCV tuning...\n"
    grid_search.fit(X, y)
    output += "[INFO] GridSearchCV completed. Best parameters:\n"
    output += str(grid_search.best_params_) + "\n"
    best_model = grid_search.best_estimator_
    
    # Save the best model to a file
    best_model.save_model("xgboost_best_model.json")
    output += "[INFO] Best model saved to 'xgboost_best_model.json'\n"
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pred = best_model.predict(X_test)
    
    y_test_real = np.expm1(y_test)
    pred_real = np.expm1(pred)
    
    mae = mean_absolute_error(y_test_real, pred_real)
    mse = mean_squared_error(y_test_real, pred_real)
    r2  = r2_score(y_test_real, pred_real)
    
    output += "\n[Grid XGBoost] Regression Metrics:\n"
    output += f"  MAE: {mae:.2f}\n"
    output += f"  MSE: {mse:.2f}\n"
    output += f"  R²:  {r2:.2f}\n"
    
    threshold = np.expm1(y_train.mean())
    y_test_class = (y_test_real >= threshold).astype(int)
    pred_class = (pred_real >= threshold).astype(int)
    
    acc  = accuracy_score(y_test_class, pred_class)
    rec  = recall_score(y_test_class, pred_class)
    prec = precision_score(y_test_class, pred_class)
    f1   = f1_score(y_test_class, pred_class)
    cm   = confusion_matrix(y_test_class, pred_class)
    
    output += "\n[Grid XGBoost] Classification Metrics (Threshold = {:.2f}):\n".format(threshold)
    output += f"  Accuracy:  {acc:.2f}\n"
    output += f"  Recall:    {rec:.2f}\n"
    output += f"  Precision: {prec:.2f}\n"
    output += f"  F1 Score:  {f1:.2f}\n"
    output += "  Confusion Matrix:\n" + str(cm) + "\n"
    
    return best_model, output

#############################################
# 6. GUI Application using tkinter
#############################################
import tkinter as tk
from tkinter import ttk, scrolledtext

class ModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ontario Energy Demand Forecast")
        self.root.geometry("900x700")
        
        # Model selection frame
        control_frame = ttk.Frame(root)
        control_frame.pack(pady=10)
        
        ttk.Label(control_frame, text="Select Model:").grid(row=0, column=0, padx=5)
        self.model_var = tk.StringVar(value="XGBoost")
        self.model_options = ["XGBoost", "Polynomial+Ridge Regression", "LSTM"]
        self.model_menu = ttk.OptionMenu(control_frame, self.model_var, self.model_options[0], *self.model_options)
        self.model_menu.grid(row=0, column=1, padx=5)
        
        self.execute_button = ttk.Button(control_frame, text="Execute", command=self.execute_model)
        self.execute_button.grid(row=0, column=2, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(root, orient="horizontal", length=600, mode="determinate")
        self.progress.pack(pady=10)
        
        # Output scrolled text
        self.output_text = scrolledtext.ScrolledText(root, width=100, height=35)
        self.output_text.pack(pady=10)
        
    def update_output(self, text):
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)
        
    def execute_model(self):
        self.update_output("")
        self.progress['value'] = 0
        self.root.update()
        thread = threading.Thread(target=self.run_selected_model)
        thread.start()
        
    def run_selected_model(self):
        selected_model = self.model_var.get()
        output = ""
        
        # Step 1: Merge demand and weather data
        self.progress['value'] = 5
        self.root.update()
        demand_file = "ontario_electricity_demand.csv"
        weather_file = "Ontario_Final.csv"
        merged_file = "merged_data.csv"
        if not os.path.exists(merged_file):
            merged_df, out1 = merge_demand_and_weather(demand_file, weather_file, merged_file)
            output += out1
        else:
            merged_df = pd.read_csv(merged_file, parse_dates=['datetime']).set_index('datetime')
            output += f"[INFO] Loaded merged data from {merged_file}\n"
        self.progress['value'] = 15
        self.root.update()
        
        # Step 2: Load and merge population data
        pop_series = load_population_data_from_string()
        merged_df = merge_population(merged_df, pop_series)
        merged_pop_file = "merged_with_pop.csv"
        merged_df.to_csv(merged_pop_file, index_label='datetime')
        output += f"[INFO] Merged data with population saved to {merged_pop_file}\n"
        self.progress['value'] = 30
        self.root.update()
        
        # Step 3: Preprocess data
        features = ["Temp (°C)", "Dew Point Temp (°C)", "Rel Hum (%)",
                    "Wind Spd (km/h)", "Stn Press (kPa)", "population"]
        target = "hourly_demand"
        do_log_transform = True
        df_processed, X, y = load_preprocessed_data(merged_pop_file, features, target=target, do_log=do_log_transform)
        output += "[INFO] Data preprocessed.\n"
        self.progress['value'] = 45
        self.root.update()
        
        # Execute selected model
        if selected_model == "XGBoost":
            best_model, out_model = train_xgboost_grid(X, y)
            output += out_model
        elif selected_model == "Polynomial+Ridge Regression":
            # Use TimeSeriesSplit for CV evaluation (5 splits)
            tscv = TimeSeriesSplit(n_splits=5)
            results = []
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("ridge", Ridge(alpha=1.0, random_state=42))
                ])
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                y_test_real = np.expm1(y_test)
                y_pred_real = np.expm1(y_pred)
                mae = mean_absolute_error(y_test_real, y_pred_real)
                mse = mean_squared_error(y_test_real, y_pred_real)
                r2 = r2_score(y_test_real, y_pred_real)
                threshold = np.median(np.expm1(y_train))
                y_test_class = (y_test_real >= threshold).astype(int)
                y_pred_class = (y_pred_real >= threshold).astype(int)
                acc = accuracy_score(y_test_class, y_pred_class)
                rec = recall_score(y_test_class, y_pred_class)
                prec = precision_score(y_test_class, y_pred_class)
                f1 = f1_score(y_test_class, y_pred_class)
                cm = confusion_matrix(y_test_class, y_pred_class)
                results.append({
                    "mae": mae,
                    "mse": mse,
                    "r2": r2,
                    "accuracy": acc,
                    "recall": rec,
                    "precision": prec,
                    "f1": f1,
                    "confusion_matrix": cm
                })
            avg_mae = np.mean([res["mae"] for res in results])
            avg_mse = np.mean([res["mse"] for res in results])
            avg_r2  = np.mean([res["r2"] for res in results])
            avg_acc = np.mean([res["accuracy"] for res in results])
            avg_rec = np.mean([res["recall"] for res in results])
            avg_prec = np.mean([res["precision"] for res in results])
            avg_f1 = np.mean([res["f1"] for res in results])
            out_model = "[Polynomial+Ridge Regression] TimeSeries CV Metrics:\n"
            out_model += f"  Average MAE: {avg_mae:.2f}\n"
            out_model += f"  Average MSE: {avg_mse:.2f}\n"
            out_model += f"  Average R²:  {avg_r2:.2f}\n"
            out_model += f"  Average Accuracy:  {avg_acc:.2f}\n"
            out_model += f"  Average Recall:    {avg_rec:.2f}\n"
            out_model += f"  Average Precision: {avg_prec:.2f}\n"
            out_model += f"  Average F1 Score:  {avg_f1:.2f}\n\n"
            output += out_model
        elif selected_model == "LSTM":
            # Prepare LSTM data using preprocessed data
            scaler_lstm = StandardScaler()
            X_scaled_lstm = scaler_lstm.fit_transform(df_processed[features])
            y_lstm = df_processed['log_demand'].values if do_log_transform else df_processed[target].values
            def create_lstm_dataset(X, y, steps=24):
                X_lstm, y_lstm = [], []
                for i in range(len(X) - steps):
                    X_seq = X[i:i+steps]
                    y_seq = y[i+steps]
                    X_lstm.append(X_seq)
                    y_lstm.append(y_seq)
                return np.array(X_lstm), np.array(y_lstm)
            X_lstm, y_lstm = create_lstm_dataset(X_scaled_lstm, y_lstm, steps=24)
            split_idx = int(len(X_lstm) * 0.8)
            X_train_lstm, X_test_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
            y_train_lstm, y_test_lstm = y_lstm[:split_idx], y_lstm[split_idx:]
            
            model_lstm = Sequential([
                LSTM(64, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
                Dense(1)
            ])
            model_lstm.compile(optimizer='adam', loss='mse')
            model_lstm.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, verbose=0)
            y_pred_lstm = model_lstm.predict(X_test_lstm).flatten()
            mae = mean_absolute_error(y_test_lstm, y_pred_lstm)
            mse = mean_squared_error(y_test_lstm, y_pred_lstm)
            r2 = r2_score(y_test_lstm, y_pred_lstm)
            out_model = "[LSTM] Regression Metrics:\n"
            out_model += f"  MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.2f}\n"
            threshold = np.mean(y_train_lstm)
            y_test_class = (y_test_lstm >= threshold).astype(int)
            y_pred_class = (y_pred_lstm >= threshold).astype(int)
            acc = accuracy_score(y_test_class, y_pred_class)
            rec = recall_score(y_test_class, y_pred_class)
            prec = precision_score(y_test_class, y_pred_class)
            f1 = f1_score(y_test_class, y_pred_class)
            cm = confusion_matrix(y_test_class, y_pred_class)
            out_model += "[LSTM] Classification Metrics (Threshold = {:.2f}):\n".format(threshold)
            out_model += f"  Accuracy: {acc:.2f}, Recall: {rec:.2f}, Precision: {prec:.2f}, F1 Score: {f1:.2f}\n"
            out_model += "  Confusion Matrix:\n" + str(cm) + "\n"
            output += out_model
        
        # Update progress bar to 100%
        self.progress['value'] = 100
        self.root.update()
        
        # Display output
        self.update_output(output)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModelApp(root)
    root.mainloop()
