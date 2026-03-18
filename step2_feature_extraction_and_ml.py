"""
Step 2: PPG Feature Extraction + ML Baseline Model
- Extracts time/frequency domain features from PPG segments
- Trains XGBoost/LightGBM to predict SBP, DBP, MAP
- Uses Leave-One-Subject-Out (LOSO) cross-validation
"""
import os
import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────── Config ────────────────────────────
PROCESSED_DIR = 'C:/Users/jaege/Desktop/Study/PPG2ABP/processed'
RESULTS_DIR = 'C:/Users/jaege/Desktop/Study/PPG2ABP/results'
SAMPLE_RATE = 125
os.makedirs(RESULTS_DIR, exist_ok=True)


# ──────────────────────── Feature Extraction ────────────────────
def extract_ppg_features(segment, fs=125):
    """
    Extract features from a single PPG segment (1D array).
    Returns a feature vector.
    """
    features = {}
    sig = segment

    # ── 1. Basic statistics ──
    features['mean'] = np.mean(sig)
    features['std'] = np.std(sig)
    features['skewness'] = skew(sig)
    features['kurtosis'] = kurtosis(sig)
    features['ptp'] = np.ptp(sig)  # peak-to-peak
    features['rms'] = np.sqrt(np.mean(sig**2))

    # ── 2. Peak detection ──
    # Find systolic peaks
    peaks, peak_props = find_peaks(sig, distance=int(fs*0.4), height=np.mean(sig))
    # Find valleys (troughs)
    valleys, _ = find_peaks(-sig, distance=int(fs*0.4))

    n_peaks = len(peaks)
    features['n_peaks'] = n_peaks
    features['heart_rate'] = n_peaks * (60.0 / (len(sig) / fs)) if n_peaks > 0 else 0

    if n_peaks >= 2:
        # Peak-to-peak intervals
        ppi = np.diff(peaks) / fs  # in seconds
        features['ppi_mean'] = np.mean(ppi)
        features['ppi_std'] = np.std(ppi)
        features['ppi_cv'] = features['ppi_std'] / features['ppi_mean'] if features['ppi_mean'] > 0 else 0

        # Peak amplitudes
        peak_amps = sig[peaks]
        features['peak_amp_mean'] = np.mean(peak_amps)
        features['peak_amp_std'] = np.std(peak_amps)
    else:
        features['ppi_mean'] = 0
        features['ppi_std'] = 0
        features['ppi_cv'] = 0
        features['peak_amp_mean'] = 0
        features['peak_amp_std'] = 0

    if len(valleys) >= 2:
        valley_amps = sig[valleys]
        features['valley_amp_mean'] = np.mean(valley_amps)
        features['valley_amp_std'] = np.std(valley_amps)
    else:
        features['valley_amp_mean'] = np.min(sig)
        features['valley_amp_std'] = 0

    # Pulse amplitude (peak - preceding valley)
    if n_peaks >= 1 and len(valleys) >= 1:
        pulse_amps = []
        for p in peaks:
            prev_valleys = valleys[valleys < p]
            if len(prev_valleys) > 0:
                pulse_amps.append(sig[p] - sig[prev_valleys[-1]])
        if len(pulse_amps) > 0:
            features['pulse_amp_mean'] = np.mean(pulse_amps)
            features['pulse_amp_std'] = np.std(pulse_amps)
        else:
            features['pulse_amp_mean'] = 0
            features['pulse_amp_std'] = 0
    else:
        features['pulse_amp_mean'] = 0
        features['pulse_amp_std'] = 0

    # ── 3. Systolic / Diastolic timing ──
    if n_peaks >= 1 and len(valleys) >= 2:
        sys_times = []
        dia_times = []
        for i, p in enumerate(peaks):
            prev_v = valleys[valleys < p]
            next_v = valleys[valleys > p]
            if len(prev_v) > 0:
                sys_times.append((p - prev_v[-1]) / fs)
            if len(next_v) > 0:
                dia_times.append((next_v[0] - p) / fs)

        features['sys_time_mean'] = np.mean(sys_times) if sys_times else 0
        features['dia_time_mean'] = np.mean(dia_times) if dia_times else 0
        features['sys_dia_ratio'] = (features['sys_time_mean'] / features['dia_time_mean']
                                     if features['dia_time_mean'] > 0 else 0)
    else:
        features['sys_time_mean'] = 0
        features['dia_time_mean'] = 0
        features['sys_dia_ratio'] = 0

    # ── 4. Area under curve ──
    features['auc_total'] = np.trapz(np.abs(sig)) / len(sig)

    # ── 5. Derivative features (VPG, APG) ──
    vpg = np.gradient(sig, 1/fs)  # 1st derivative
    apg = np.gradient(vpg, 1/fs)  # 2nd derivative

    features['vpg_max'] = np.max(vpg)
    features['vpg_min'] = np.min(vpg)
    features['vpg_std'] = np.std(vpg)
    features['apg_max'] = np.max(apg)
    features['apg_min'] = np.min(apg)
    features['apg_std'] = np.std(apg)

    # ── 6. Frequency domain features ──
    freqs, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)))

    # Band powers
    vlf_mask = (freqs >= 0.04) & (freqs < 0.15)
    lf_mask = (freqs >= 0.15) & (freqs < 0.4)
    hf_mask = (freqs >= 0.4) & (freqs < 4.0)
    cardiac_mask = (freqs >= 0.5) & (freqs < 5.0)

    total_power = np.trapz(psd, freqs)
    features['vlf_power'] = np.trapz(psd[vlf_mask], freqs[vlf_mask]) if vlf_mask.any() else 0
    features['lf_power'] = np.trapz(psd[lf_mask], freqs[lf_mask]) if lf_mask.any() else 0
    features['hf_power'] = np.trapz(psd[hf_mask], freqs[hf_mask]) if hf_mask.any() else 0
    features['cardiac_power'] = np.trapz(psd[cardiac_mask], freqs[cardiac_mask]) if cardiac_mask.any() else 0

    if total_power > 0:
        features['lf_hf_ratio'] = features['lf_power'] / features['hf_power'] if features['hf_power'] > 0 else 0
        features['cardiac_ratio'] = features['cardiac_power'] / total_power
    else:
        features['lf_hf_ratio'] = 0
        features['cardiac_ratio'] = 0

    # Dominant frequency
    features['dominant_freq'] = freqs[np.argmax(psd)]
    features['spectral_entropy'] = -np.sum((psd/total_power) * np.log2(psd/total_power + 1e-12)) if total_power > 0 else 0

    return features


# ──────────────────────── Load & Extract ────────────────────────
print("Loading processed data and extracting features...")

all_features = []
all_sbp = []
all_dbp = []
all_mbp = []
all_subject_ids = []
feature_names = None

case_files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.endswith('.npz')])
print(f"Found {len(case_files)} processed cases")

for case_file in case_files:
    case_name = case_file.replace('.npz', '')
    data = np.load(os.path.join(PROCESSED_DIR, case_file))

    pleth = data['pleth']
    sbp = data['sbp']
    dbp = data['dbp']
    mbp = data['mbp']
    age = float(data['age'])
    sex = int(data['sex'])

    n_segs = len(pleth)
    print(f"  {case_name}: extracting features from {n_segs} segments...", end=" ")

    case_features = []
    valid_indices = []

    for i in range(n_segs):
        # Skip segments with NaN BP values
        if np.isnan(sbp[i]) or np.isnan(dbp[i]):
            continue
        # Skip physiologically implausible BP
        if sbp[i] < 30 or sbp[i] > 250 or dbp[i] < 10 or dbp[i] > 200:
            continue
        if sbp[i] <= dbp[i]:
            continue

        try:
            feat = extract_ppg_features(pleth[i])
            feat['age'] = age
            feat['sex'] = sex
            case_features.append(feat)
            valid_indices.append(i)
        except Exception:
            continue

    if len(case_features) == 0:
        print("no valid segments!")
        continue

    if feature_names is None:
        feature_names = list(case_features[0].keys())

    for j, idx in enumerate(valid_indices):
        feat_vec = [case_features[j][fn] for fn in feature_names]
        all_features.append(feat_vec)
        all_sbp.append(sbp[idx])
        all_dbp.append(dbp[idx])
        all_mbp.append(mbp[idx])
        all_subject_ids.append(case_name)

    print(f"{len(case_features)} valid")

X = np.array(all_features, dtype=np.float32)
y_sbp = np.array(all_sbp, dtype=np.float32)
y_dbp = np.array(all_dbp, dtype=np.float32)
y_mbp = np.array(all_mbp, dtype=np.float32)
subjects = np.array(all_subject_ids)

# Replace any inf/nan in features
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Features: {feature_names}")
print(f"Unique subjects: {len(np.unique(subjects))}")
print(f"SBP range: {y_sbp.min():.0f} ~ {y_sbp.max():.0f} mmHg (mean={y_sbp.mean():.0f})")
print(f"DBP range: {y_dbp.min():.0f} ~ {y_dbp.max():.0f} mmHg (mean={y_dbp.mean():.0f})")


# ──────────────────────── LOSO Cross-Validation ─────────────────
def loso_evaluate(X, y, subjects, model_type='xgboost', target_name='SBP'):
    """Leave-One-Subject-Out cross-validation."""
    unique_subjects = np.unique(subjects)
    n_subjects = len(unique_subjects)

    all_preds = np.zeros_like(y)
    all_true = np.zeros_like(y)
    per_subject_mae = []

    for i, test_subj in enumerate(unique_subjects):
        test_mask = subjects == test_subj
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Standardize features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if model_type == 'xgboost':
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
            )
        elif model_type == 'lightgbm':
            model = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
            )

        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

        all_preds[test_mask] = preds
        all_true[test_mask] = y_test

        subj_mae = mean_absolute_error(y_test, preds)
        per_subject_mae.append(subj_mae)

    # Overall metrics
    mae = mean_absolute_error(all_true, all_preds)
    rmse = np.sqrt(mean_squared_error(all_true, all_preds))
    r2 = r2_score(all_true, all_preds)

    # Bland-Altman
    diff = all_preds - all_true
    bias = np.mean(diff)
    std_diff = np.std(diff)

    # BHS grading
    abs_diff = np.abs(diff)
    pct_5 = (abs_diff <= 5).mean() * 100
    pct_10 = (abs_diff <= 10).mean() * 100
    pct_15 = (abs_diff <= 15).mean() * 100

    bhs_grade = 'D'
    if pct_5 >= 60 and pct_10 >= 85 and pct_15 >= 95:
        bhs_grade = 'A'
    elif pct_5 >= 50 and pct_10 >= 75 and pct_15 >= 90:
        bhs_grade = 'B'
    elif pct_5 >= 40 and pct_10 >= 65 and pct_15 >= 85:
        bhs_grade = 'C'

    # AAMI check
    aami_pass = mae <= 5 and std_diff <= 8

    print(f"\n{'='*60}")
    print(f"  {target_name} - {model_type.upper()} (LOSO)")
    print(f"{'='*60}")
    print(f"  MAE:  {mae:.2f} mmHg")
    print(f"  RMSE: {rmse:.2f} mmHg")
    print(f"  R²:   {r2:.4f}")
    print(f"  Bias: {bias:.2f} ± {std_diff:.2f} mmHg")
    print(f"  BHS:  ≤5mmHg={pct_5:.1f}%, ≤10mmHg={pct_10:.1f}%, ≤15mmHg={pct_15:.1f}% → Grade {bhs_grade}")
    print(f"  AAMI: MAE≤5={mae<=5}, SD≤8={std_diff<=8} → {'PASS' if aami_pass else 'FAIL'}")
    print(f"  Per-subject MAE: {np.mean(per_subject_mae):.2f} ± {np.std(per_subject_mae):.2f}")

    return {
        'mae': mae, 'rmse': rmse, 'r2': r2,
        'bias': bias, 'std_diff': std_diff,
        'bhs_grade': bhs_grade, 'aami_pass': aami_pass,
        'preds': all_preds, 'true': all_true,
        'per_subject_mae': per_subject_mae,
    }


# ──────────────────────── Run Experiments ───────────────────────
print("\n" + "="*60)
print("RUNNING LOSO CROSS-VALIDATION")
print("="*60)

results = {}
for target_name, y in [('SBP', y_sbp), ('DBP', y_dbp), ('MBP', y_mbp)]:
    for model_type in ['xgboost', 'lightgbm']:
        key = f"{target_name}_{model_type}"
        results[key] = loso_evaluate(X, y, subjects, model_type, target_name)


# ──────────────────────── Feature Importance ────────────────────
print("\n" + "="*60)
print("TOP 15 FEATURE IMPORTANCE (XGBoost, SBP)")
print("="*60)

# Train final model on all data for feature importance
scaler = StandardScaler()
X_s = scaler.fit_transform(X)
model_fi = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, verbosity=0, random_state=42)
model_fi.fit(X_s, y_sbp)

importances = model_fi.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
for rank, idx in enumerate(sorted_idx[:15]):
    print(f"  {rank+1:2d}. {feature_names[idx]:25s}  {importances[idx]:.4f}")


# ──────────────────────── Save Results ──────────────────────────
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"{'Target':<6} {'Model':<10} {'MAE':>6} {'RMSE':>6} {'R²':>7} {'BHS':>5} {'AAMI':>6}")
print("-" * 50)
for key, res in results.items():
    target, model = key.split('_')
    print(f"{target:<6} {model:<10} {res['mae']:>6.2f} {res['rmse']:>6.2f} {res['r2']:>7.4f} "
          f"{res['bhs_grade']:>5} {'PASS' if res['aami_pass'] else 'FAIL':>6}")

# Save predictions for later analysis
np.savez(
    os.path.join(RESULTS_DIR, 'ml_baseline_results.npz'),
    feature_names=feature_names,
    X=X, y_sbp=y_sbp, y_dbp=y_dbp, y_mbp=y_mbp,
    subjects=subjects,
    **{f"{k}_preds": v['preds'] for k, v in results.items()},
    **{f"{k}_true": v['true'] for k, v in results.items()},
)
print(f"\nResults saved to {RESULTS_DIR}/ml_baseline_results.npz")
