"""
build_assay_dataset.py
======================
Extracts a structured, Machine Learning-ready assay-level dataset from raw 
flotation plant historian data.

This script identifies true analytical events (where laboratory targets 
change) and extracts features specific to the time window leading up to 
each assay. This prevents the severe data leakage that occurs when 
interpolating or forward-filling low-frequency lab measurements into 
high-frequency sensor data.

Key Feature Engineering:
  - Sensor window stats  : mean, std, min, max of every sensor during the 
                           interval between the previous assay and the current one.
  - Autoregressive lags  : Previous (t-1, t-2, t-3) assay readings to capture 
                           time-series persistence (AR components).
  - Time features        : Time-of-day, day-of-week, and time elapsed since 
                           the last measurement.

Targets:
  - % Iron Concentrate
  - % Silica Concentrate
"""

import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Suppress warnings for cleaner execution output
warnings.filterwarnings('ignore')

def load_and_clean_data(file_path: str) -> tuple[pd.DataFrame, List[str], List[str]]:
    """
    Loads raw minute-level historian data and separates feature columns from targets.
    
    Args:
        file_path (str): Path to the raw CSV data.
        
    Returns:
        tuple: (cleaned DataFrame, list of target columns, list of sensor columns)
    """
    print("Loading raw data...")
    df = pd.read_csv(file_path, parse_dates=[0], low_memory=False)
    
    # Standardize the datetime column name
    date_col = df.columns[0]
    df = df.rename(columns={date_col: 'date'})
    
    # Coerce all non-date columns to numeric, forcing errors to NaN
    numeric_cols = df.columns.drop('date')
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.sort_values('date').reset_index(drop=True)

    target_cols = ['% Iron Concentrate', '% Silica Concentrate']
    sensor_cols = [c for c in df.columns if c not in ['date'] + target_cols]
    
    print(f"Raw shape: {df.shape}")
    return df, target_cols, sensor_cols

def extract_assay_events(df: pd.DataFrame, target_cols: List[str]) -> tuple[pd.DataFrame, List[int]]:
    """
    Identifies rows where true laboratory assay measurements were recorded.
    A true measurement is identified when the target value changes from the previous row.
    
    Args:
        df (pd.DataFrame): The raw, forward-filled DataFrame.
        target_cols (List[str]): Columns to check for value changes.
        
    Returns:
        tuple: (DataFrame of real assay events, list of original row indices)
    """
    # Detect changes in targets to locate the exact minute an assay result was posted
    change_mask = (df[target_cols].diff().abs() > 0).any(axis=1)
    df_real = df[change_mask].copy().reset_index(drop=True)
    real_indices = df.index[change_mask].tolist()
    
    print(f"Real assay events identified: {len(df_real):,}")
    return df_real, real_indices

def build_features(df: pd.DataFrame, df_real: pd.DataFrame, 
                  real_indices: List[int], sensor_cols: List[str], 
                  target_cols: List[str]) -> pd.DataFrame:
    """
    Constructs interval-based statistical features and instantaneous readings for each assay.
    
    Args:
        df (pd.DataFrame): Raw historian DataFrame.
        df_real (pd.DataFrame): Filtered DataFrame containing only real assays.
        real_indices (List[int]): Original indices mapping df_real back to df.
        sensor_cols (List[str]): Sensor columns to calculate stats for.
        target_cols (List[str]): Target columns to extract.
        
    Returns:
        pd.DataFrame: A structured feature matrix for ML.
    """
    print("Engineering interval and instantaneous features...")
    records = []

    for i, (real_idx, orig_idx) in enumerate(zip(df_real.index, real_indices)):
        row = {}
        dt = df_real.loc[real_idx, 'date']
        
        row['datetime'] = dt
        row['hour_of_day'] = dt.hour
        row['day_of_week'] = dt.dayofweek
        row['month'] = dt.month
        row['day_of_month'] = dt.day

        # Calculate time elapsed since the previous laboratory reading
        if i == 0:
            row['minutes_since_prev_assay'] = np.nan
            window_start = 0
        else:
            prev_orig_idx = real_indices[i - 1]
            minutes_gap = (df.loc[orig_idx, 'date'] - df.loc[prev_orig_idx, 'date']).total_seconds() / 60
            row['minutes_since_prev_assay'] = minutes_gap
            window_start = prev_orig_idx + 1

        window_end = orig_idx + 1  # Inclusive of current row

        # Extract the segment of raw data between the previous assay and the current one
        window_df = df.loc[window_start:window_end - 1, sensor_cols]

        # Compute interval statistics for all sensors
        for col in sensor_cols:
            col_data = window_df[col].dropna()
            safe_col = col.replace('%', 'pct').replace(' ', '_').replace('/', '_')
            
            row[f"{safe_col}_mean"] = col_data.mean() if not col_data.empty else np.nan
            row[f"{safe_col}_std"]  = col_data.std()  if len(col_data) > 1 else np.nan
            row[f"{safe_col}_min"]  = col_data.min()  if not col_data.empty else np.nan
            row[f"{safe_col}_max"]  = col_data.max()  if not col_data.empty else np.nan
            
            # Instantaneous reading at the exact time of the assay
            row[f"{safe_col}_at_assay"] = df_real.loc[real_idx, col]

        # Add target variables
        for col in target_cols:
            safe_col = col.replace('%', 'pct').replace(' ', '_').replace('/', '_')
            row[safe_col] = df_real.loc[real_idx, col]

        records.append(row)

    return pd.DataFrame(records)

def add_autoregressive_features(assay_df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """
    Appends autoregressive lags (t-1, t-2, t-3) and percentage changes to capture 
    time-series persistence (i.e. 'a high grade now predicts a high grade later').
    
    Args:
        assay_df (pd.DataFrame): Structured feature matrix.
        target_cols (List[str]): Target columns to lag.
        
    Returns:
        pd.DataFrame: Feature matrix augmented with AR components.
    """
    print("Adding autoregressive lag features (AR-1, AR-2, AR-3)...")
    for col in target_cols:
        safe_col = col.replace('%', 'pct').replace(' ', '_').replace('/', '_')
        for lag in [1, 2, 3]:
            assay_df[f"{safe_col}_lag{lag}"] = assay_df[safe_col].shift(lag)
        
        assay_df[f"{safe_col}_delta"] = assay_df[safe_col].diff()
        assay_df[f"{safe_col}_pct_change"] = assay_df[safe_col].pct_change() * 100

    # Drop the initial rows that lack sufficient history to populate the lags
    assay_df_ml = assay_df.dropna(subset=[
        'pct_Iron_Concentrate_lag1',
        'pct_Silica_Concentrate_lag1',
        'minutes_since_prev_assay'
    ]).reset_index(drop=True)
    
    return assay_df_ml

def main():
    """Main execution pipeline."""
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    raw_data_path = str(PROJECT_ROOT / 'data' / 'iron_ore_flotation.csv')
    out_csv = str(PROJECT_ROOT / 'data' / 'assay_dataset.csv')
    out_xlsx = str(PROJECT_ROOT / 'data' / 'assay_dataset.xlsx')
    
    if not Path(raw_data_path).exists():
        print(f"Error: Raw data file not found at {raw_data_path}")
        return

    # Execute pipeline
    df, target_cols, sensor_cols = load_and_clean_data(raw_data_path)
    df_real, real_indices = extract_assay_events(df, target_cols)
    assay_df = build_features(df, df_real, real_indices, sensor_cols, target_cols)
    assay_df_ml = add_autoregressive_features(assay_df, target_cols)

    # Console Report
    print("\n" + "=" * 60)
    print("ASSAY-LEVEL ML DATASET SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(assay_df_ml):,}")
    print(f"Total columns: {len(assay_df_ml.columns)}")
    print(f"Date range: {assay_df_ml['datetime'].min()} to {assay_df_ml['datetime'].max()}\n")

    intervals = assay_df_ml['minutes_since_prev_assay']
    print(f"Assay interval stats (minutes) - Mean: {intervals.mean():.1f}, Std: {intervals.std():.1f}")
    
    # Save Outputs
    assay_df_ml.to_csv(out_csv, index=False)
    print(f"\nSaved ML-ready CSV : {out_csv}")
    
    try:
        assay_df_ml.to_excel(out_xlsx, index=False)
        print(f"Saved ML-ready XLSX: {out_xlsx}")
    except Exception as e:
        print(f"XLSX save skipped: {e}")

if __name__ == "__main__":
    main()
