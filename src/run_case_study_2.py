"""
Full Case Study 2 Pipeline on FRESH iron ore flotation data.
Steps:
  1. Load raw historian
  2. Extract assay events (rows where lab columns actually change)
  3. Build window-level features + lag features
  4. Train XGBoost (temporal split 80/20)
  5. Train LSTM (temporal split 80/20)
  6. Print all results for paper update
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CSV = str(PROJECT_ROOT / 'data' / 'iron_ore_flotation.csv')

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD RAW DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: Loading raw historian")
print("=" * 70)
df = pd.read_csv(CSV, sep=',', decimal=',', parse_dates=['date'])
print(f"  Raw shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
span_days = (df['date'].max() - df['date'].min()).days
print(f"  Duration: {span_days} days (~{span_days/30:.1f} months)")

# Column definitions
target_cols   = ['% Iron Concentrate', '% Silica Concentrate']
lab_feed_cols = ['% Iron Feed', '% Silica Feed']
lab_all       = target_cols + lab_feed_cols
sensor_cols   = [c for c in df.columns if c not in lab_all + ['date']]

# ═══════════════════════════════════════════════════════════════════════════════
# 2. ASSAY-EVENT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: Assay-event extraction")
print("=" * 70)

# Identify rows where ANY lab column changes value
change_mask = df[lab_all].diff().fillna(1).abs().sum(axis=1) > 0
assay_idx = df.index[change_mask]
n_events = len(assay_idx)
print(f"  Assay events extracted: {n_events:,} (from {len(df):,} raw rows)")
print(f"  Compression ratio: {len(df)/n_events:.0f}x")

# For each assay event, compute window-level stats from sensor data
print("  Building window-level features...")
records = []
for i, idx in enumerate(assay_idx):
    if i == 0:
        prev_idx = 0
    else:
        prev_idx = assay_idx[i - 1] + 1

    window = df.iloc[prev_idx:idx + 1]
    row = {}

    # Instantaneous sensor values at assay moment
    for col in sensor_cols:
        row[f'{col}_inst'] = df.loc[idx, col]

    # Window statistics
    for col in sensor_cols:
        row[f'{col}_mean'] = window[col].mean()
        row[f'{col}_std']  = window[col].std()
        row[f'{col}_min']  = window[col].min()
        row[f'{col}_max']  = window[col].max()

    # Lab values at this event
    for col in lab_all:
        row[col] = df.loc[idx, col]

    row['date'] = df.loc[idx, 'date']

    # Time features
    row['hour'] = df.loc[idx, 'date'].hour
    row['dow']  = df.loc[idx, 'date'].dayofweek

    records.append(row)

    if (i + 1) % 10000 == 0:
        print(f"    Processed {i+1:,}/{n_events:,} events...")

assay_df = pd.DataFrame(records)
print(f"  Assay dataset shape: {assay_df.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. LAG FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: Adding autoregressive lag features")
print("=" * 70)

for target in target_cols:
    for lag in [1, 2, 3]:
        assay_df[f'{target}_lag{lag}'] = assay_df[target].shift(lag)
        assay_df[f'{target}_diff{lag}'] = assay_df[target].diff(lag)

# Drop rows with NaN from lagging
assay_df = assay_df.dropna().reset_index(drop=True)
print(f"  Final dataset: {assay_df.shape[0]:,} rows x {assay_df.shape[1]} features")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TEMPORAL SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
split_idx = int(0.8 * len(assay_df))
train = assay_df.iloc[:split_idx]
test  = assay_df.iloc[split_idx:]
print(f"  Train: {len(train):,} rows | Test: {len(test):,} rows")
print(f"  Train date range: {train['date'].min()} to {train['date'].max()}")
print(f"  Test  date range: {test['date'].min()} to {test['date'].max()}")

feature_cols = [c for c in assay_df.columns if c not in target_cols + lab_feed_cols + ['date']]
X_train = train[feature_cols].values
X_test  = test[feature_cols].values

# ═══════════════════════════════════════════════════════════════════════════════
# 5. XGBOOST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: XGBoost Training")
print("=" * 70)

xgb_results = {}
for target in target_cols:
    y_train = train[target].values
    y_test  = test[target].values

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    xgb_results[target] = {'r2': r2, 'rmse': rmse, 'mae': mae}
    print(f"  {target:<25}  R²={r2:.6f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    # Feature importance (top 10)
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top10 = importances.nlargest(10)
    print(f"    Top 10 features:")
    for fname, fimp in top10.items():
        print(f"      {fname:<40}  importance={fimp:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. LSTM
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: LSTM Training")
print("=" * 70)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    # Scale features
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train)
    X_test_s  = scaler_X.transform(X_test)

    for target in target_cols:
        y_train = train[target].values.reshape(-1, 1)
        y_test  = test[target].values.reshape(-1, 1)

        scaler_y = StandardScaler()
        y_train_s = scaler_y.fit_transform(y_train)
        y_test_s  = scaler_y.transform(y_test)

        # Create sequences
        seq_len = 20
        def make_sequences(X, y, seq_len):
            Xs, ys = [], []
            for i in range(len(X) - seq_len):
                Xs.append(X[i:i+seq_len])
                ys.append(y[i+seq_len])
            return np.array(Xs), np.array(ys)

        X_tr_seq, y_tr_seq = make_sequences(X_train_s, y_train_s, seq_len)
        X_te_seq, y_te_seq = make_sequences(X_test_s,  y_test_s,  seq_len)

        # Model
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden=128, layers=2, dropout=0.3):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=dropout)
                self.fc = nn.Linear(hidden, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = LSTMModel(X_tr_seq.shape[2]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_tr_seq))
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=False)

        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(200):
            model.train()
            epoch_loss = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break

        model.eval()
        with torch.no_grad():
            y_pred_s = model(torch.FloatTensor(X_te_seq).to(device)).cpu().numpy()
        y_pred = scaler_y.inverse_transform(y_pred_s)
        y_actual = scaler_y.inverse_transform(y_te_seq)

        r2   = r2_score(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        print(f"  {target:<25}  R²={r2:.4f}  RMSE={rmse:.4f}  (stopped at epoch {epoch+1})")

except ImportError:
    print("  PyTorch not installed — skipping LSTM.")
    print("  Install with: pip install torch")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. SPEARMAN CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 6: Spearman correlations (sensor features vs targets)")
print("=" * 70)

from scipy.stats import spearmanr, normaltest

# Normality check on targets
for col in target_cols:
    stat, p = normaltest(assay_df[col].dropna())
    print(f"  Normality test {col}: stat={stat:.2f}, p={p:.6f}, normal={'Yes' if p >= 0.05 else 'No'}")

print(f"\n  {'Feature':<45}  {'Spearman vs %Fe':>16}  {'Spearman vs %SiO2':>18}")
print(f"  {'-'*45}  {'-'*16}  {'-'*18}")
for feat in sensor_cols:
    rho_fe, _ = spearmanr(assay_df[f'{feat}_inst'], assay_df['% Iron Concentrate'])
    rho_si, _ = spearmanr(assay_df[f'{feat}_inst'], assay_df['% Silica Concentrate'])
    if abs(rho_fe) > 0.05 or abs(rho_si) > 0.05:
        print(f"  {feat:<45}  {rho_fe:>+16.4f}  {rho_si:>+18.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. AUTOCORRELATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 7: Autocorrelation at assay-event level")
print("=" * 70)
for col in target_cols:
    for lag in [1, 2, 3, 5, 10]:
        acf = assay_df[col].autocorr(lag=lag)
        print(f"  {col}  lag-{lag:>2}: r = {acf:.4f}")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
