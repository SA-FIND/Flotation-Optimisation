"""
Full Case Study 2 Pipeline on FRESH iron ore flotation data.
Steps:
  1. Load raw historian
  2. Extract assay events (rows where lab columns actually change)
  3. Build window-level features + lag features
  4. Train Naive Persistence Baseline
  5. Train Pure Sensor-Only Model (Soft Sensor ablation)
  6. Train Hybrid XGBoost with AR Lags
  7. Train LSTM Model
  8. Statistical summaries (Spearman Rank correlation & Autocorrelation)
"""
import sys
from pathlib import Path
import warnings

# Add project root to sys.path to allow imports from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, normaltest

from src.models.tree_models import get_xgboost_model
from src.models.lstm import TORCH_AVAILABLE
if TORCH_AVAILABLE:
    from src.models.lstm import AssayLSTM, AssayDataset
from src.utils.config import load_config, get_project_root

CSV = str(PROJECT_ROOT / 'data' / 'iron_ore_flotation.csv')

def main():
    print("=" * 70)
    print("FLOTATION ML: CASE STUDY 2 PIPELINE")
    print(f"Config loaded from: {PROJECT_ROOT / 'config.yaml'}")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════════════
    # 1. LOAD RAW DATA
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 1: Loading raw historian")
    print("=" * 70)
    if not Path(CSV).exists():
        print(f"Error: Dataset not found at {CSV}")
        return

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
    print("STEP 2: Assay-event extraction (Eliminating Forward-Fill Leakage)")
    print("=" * 70)

    # Identify rows where ANY lab column changes value
    change_mask = df[lab_all].diff().fillna(1).abs().sum(axis=1) > 0
    assay_idx = df.index[change_mask]
    n_events = len(assay_idx)
    print(f"  Assay events extracted: {n_events:,} (from {len(df):,} raw rows)")
    print(f"  Compression ratio: {len(df)/n_events:.0f}x reduction")

    # For each assay event, compute window-level stats from sensor data
    print("  Building interval & window statistics...")
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
        row['hour'] = df.loc[idx, 'date'].hour
        row['dow']  = df.loc[idx, 'date'].dayofweek

        records.append(row)

    assay_df = pd.DataFrame(records)
    print(f"  Assay dataset shape: {assay_df.shape}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # 3. LAG FEATURES
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 3: Adding autoregressive lag features")
    print("=" * 70)

    # Only PAST assay values may be used. y.diff(lag) = y_t - y_(t-lag) contains
    # the current target y_t, so (lag1 + diff1) reproduces y_t exactly. That was
    # the source of the earlier R2 ~0.99. Momentum features are now built from
    # past lags only.
    for target in target_cols:
        for lag in [1, 2, 3]:
            assay_df[f'{target}_lag{lag}'] = assay_df[target].shift(lag)
        assay_df[f'{target}_mom12'] = assay_df[f'{target}_lag1'] - assay_df[f'{target}_lag2']
        assay_df[f'{target}_mom13'] = assay_df[f'{target}_lag1'] - assay_df[f'{target}_lag3']

    # Drop initial rows with NaN from lagging
    assay_df = assay_df.dropna().reset_index(drop=True)
    print(f"  Final clean dataset: {assay_df.shape[0]:,} rows x {assay_df.shape[1]} features")

    # ═══════════════════════════════════════════════════════════════════════════════
    # 4. TEMPORAL SPLIT (80/20)
    # ═══════════════════════════════════════════════════════════════════════════════
    split_idx = int(0.8 * len(assay_df))
    train = assay_df.iloc[:split_idx]
    test  = assay_df.iloc[split_idx:]
    print(f"  Train: {len(train):,} rows | Test: {len(test):,} rows")
    print(f"  Train date range: {train['date'].min()} to {train['date'].max()}")
    print(f"  Test  date range: {test['date'].min()} to {test['date'].max()}")

    # Feature groups
    lag_feature_cols = [c for c in assay_df.columns if any(c.startswith(f'{t}_lag') or c.startswith(f'{t}_mom') for t in target_cols)]
    sensor_feature_cols = [c for c in assay_df.columns if c not in target_cols + lab_feed_cols + ['date'] + lag_feature_cols]
    all_feature_cols = sensor_feature_cols + lag_feature_cols

    # ═══════════════════════════════════════════════════════════════════════════════
    # 5. BENCHMARK ABLATIONS
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 4: Model Evaluation & Ablation Comparison")
    print("=" * 70)

    for target in target_cols:
        y_train = train[target].values
        y_test  = test[target].values
        print(f"\n--- Target: {target} ---")

        # 1. Naive Persistence Baseline: y_hat(t) = y(t-1)
        y_pred_naive = test[f'{target}_lag1'].values
        r2_naive = r2_score(y_test, y_pred_naive)
        rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive))
        mae_naive = mean_absolute_error(y_test, y_pred_naive)
        print(f"  1. Naive Persistence (y_t = y_t-1):    R² = {r2_naive:.4f} | RMSE = {rmse_naive:.4f} | MAE = {mae_naive:.4f}")

        # 2. Pure Sensor-Only XGBoost (Soft Sensor without target lag leakage)
        X_train_sensor = train[sensor_feature_cols].values
        X_test_sensor  = test[sensor_feature_cols].values
        model_sensor = get_xgboost_model()
        model_sensor.fit(X_train_sensor, y_train, verbose=False)
        y_pred_sensor = model_sensor.predict(X_test_sensor)
        r2_sensor = r2_score(y_test, y_pred_sensor)
        rmse_sensor = np.sqrt(mean_squared_error(y_test, y_pred_sensor))
        mae_sensor = mean_absolute_error(y_test, y_pred_sensor)
        print(f"  2. Sensor-Only XGBoost (Soft Sensor):  R² = {r2_sensor:.4f} | RMSE = {rmse_sensor:.4f} | MAE = {mae_sensor:.4f}")

        # 2b. AR-only XGBoost (own past values only, no sensors)
        ar_cols = [c for c in lag_feature_cols if c.startswith(target)]
        model_ar = get_xgboost_model()
        model_ar.fit(train[ar_cols].values, y_train, verbose=False)
        y_pred_ar = model_ar.predict(test[ar_cols].values)
        r2_ar = r2_score(y_test, y_pred_ar)
        rmse_ar = np.sqrt(mean_squared_error(y_test, y_pred_ar))
        print(f"  2b. AR-only XGBoost (past assays):     R² = {r2_ar:.4f} | RMSE = {rmse_ar:.4f}")

        # 3. Hybrid XGBoost (Sensors + AR Lags)
        X_train_all = train[all_feature_cols].values
        X_test_all  = test[all_feature_cols].values
        model_hybrid = get_xgboost_model()
        model_hybrid.fit(X_train_all, y_train, verbose=False)  # test set never seen in training
        y_pred_hybrid = model_hybrid.predict(X_test_all)
        r2_hybrid = r2_score(y_test, y_pred_hybrid)
        rmse_hybrid = np.sqrt(mean_squared_error(y_test, y_pred_hybrid))
        mae_hybrid = mean_absolute_error(y_test, y_pred_hybrid)
        print(f"  3. Hybrid XGBoost (Sensors + AR Lags): R² = {r2_hybrid:.4f} | RMSE = {rmse_hybrid:.4f} | MAE = {mae_hybrid:.4f}")
        print(f"     -> Gain over Naive Persistence:     ΔR² = {r2_hybrid - r2_naive:+.4f}")
        skill = 1 - (rmse_hybrid / rmse_naive) ** 2
        print(f"     -> Skill vs persistence (1 - MSE_model/MSE_persist): {skill:+.3f}")

        # Feature importance for hybrid model
        importances = pd.Series(model_hybrid.feature_importances_, index=all_feature_cols)
        top5 = importances.nlargest(5)
        print(f"     Top 5 Features:")
        for fname, fimp in top5.items():
            print(f"       * {fname:<35} : {fimp:.4f}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # 6. LSTM MODEL
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 5: LSTM Sequence Training")
    print("=" * 70)

    if not TORCH_AVAILABLE:
        print("  PyTorch not installed — skipping LSTM training.")
    else:
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        scaler_X = StandardScaler()
        X_train_s = scaler_X.fit_transform(train[all_feature_cols].values)
        X_test_s  = scaler_X.transform(test[all_feature_cols].values)

        for target in target_cols:
            y_train = train[target].values.reshape(-1, 1)
            y_test  = test[target].values.reshape(-1, 1)

            scaler_y = StandardScaler()
            y_train_s = scaler_y.fit_transform(y_train)
            y_test_s  = scaler_y.transform(y_test)

            seq_len = 20
            def make_sequences(X, y, sl):
                Xs, ys = [], []
                for idx_seq in range(len(X) - sl):
                    Xs.append(X[idx_seq:idx_seq + sl])
                    ys.append(y[idx_seq + sl])
                return np.array(Xs), np.array(ys)

            X_tr_seq, y_tr_seq = make_sequences(X_train_s, y_train_s, seq_len)
            X_te_seq, y_te_seq = make_sequences(X_test_s,  y_test_s,  seq_len)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = AssayLSTM(n_features=X_tr_seq.shape[2], hidden=128, layers=2, dropout=0.3, output_size=1).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.MSELoss()

            train_ds = TensorDataset(torch.FloatTensor(X_tr_seq), torch.FloatTensor(y_tr_seq))
            train_dl = DataLoader(train_ds, batch_size=64, shuffle=False)

            best_loss = float('inf')
            patience_counter = 0
            for epoch in range(150):
                model.train()
                epoch_loss = 0.0
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
            mae  = mean_absolute_error(y_actual, y_pred)
            print(f"  LSTM ({target:<25}): R² = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f} (Stopped at epoch {epoch+1})")

    # ═══════════════════════════════════════════════════════════════════════════════
    # 7. SPEARMAN CORRELATION ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STEP 6: Spearman Rank Correlation & Normality Tests")
    print("=" * 70)

    for col in target_cols:
        stat, p = normaltest(assay_df[col].dropna())
        print(f"  Normality test ({col}): stat={stat:.2f}, p={p:.6e}, Normal: {'Yes' if p >= 0.05 else 'No (Use Spearman)'}")

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
    print("STEP 7: Autocorrelation Structure at Assay-Event Level")
    print("=" * 70)
    for col in target_cols:
        for lag in [1, 2, 3, 5, 10]:
            acf = assay_df[col].autocorr(lag=lag)
            print(f"  {col:<25}  lag-{lag:>2}: r = {acf:.4f}")

    print("\n" + "=" * 70)
    print("CASE STUDY 2 PIPELINE EXECUTION COMPLETED")
    print("=" * 70)

if __name__ == '__main__':
    main()
