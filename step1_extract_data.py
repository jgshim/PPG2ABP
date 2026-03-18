"""
Step 1: Extract PLETH (PPG) and IBP1 (ABP) waveforms from .vital files.
Saves preprocessed data as .npz files for fast loading.
"""
import os
import numpy as np
import vitaldb
import openpyxl
from scipy.signal import butter, filtfilt, resample_poly
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────── Config ────────────────────────────
CASES_DIR = 'C:/Users/jaege/Desktop/Study/PPG2ABP/cases'
XLSX_PATH = 'C:/Users/jaege/Desktop/Study/PPG2ABP/PPG_ABP_cases.xlsx'
OUTPUT_DIR = 'C:/Users/jaege/Desktop/Study/PPG2ABP/processed'
SAMPLE_RATE = 125  # Target sample rate (Hz) — standard for PPG analysis
SEGMENT_SEC = 10   # Segment length in seconds
MIN_VALID_RATIO = 0.95  # Minimum non-NaN ratio to keep a segment

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────── Load demographics ─────────────────
wb = openpyxl.load_workbook(XLSX_PATH)
ws = wb.active
demographics = {}
for row in ws.iter_rows(min_row=2, values_only=True):
    case_id, sex, age = row
    # Handle case13's formula "=3/12"
    if isinstance(age, str):
        age = eval(age.lstrip('='))
    demographics[case_id] = {
        'sex': 1 if sex == 'M' else 0,  # M=1, F=0
        'age': float(age)
    }
print(f"Loaded demographics for {len(demographics)} cases")


# ──────────────────────────── Bandpass filters ──────────────────
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Apply zero-phase Butterworth bandpass filter."""
    nyq = fs / 2
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)


def process_case(case_name, vital_path):
    """Extract and preprocess PLETH/IBP1 from a single .vital file."""
    print(f"\n{'='*60}")
    print(f"Processing {case_name}...")

    # Read vital file at target sample rate
    vf = vitaldb.read_vital(vital_path)
    interval = 1.0 / SAMPLE_RATE

    pleth = vf.to_numpy('PLETH', interval).flatten()
    ibp1 = vf.to_numpy('IBP1', interval).flatten()

    # Also extract numeric BP values (lower sample rate, ~1Hz)
    sbp = vf.to_numpy('ART1_SBP', interval).flatten()
    dbp = vf.to_numpy('ART1_DBP', interval).flatten()
    mbp = vf.to_numpy('ART1_MBP', interval).flatten()

    print(f"  Raw length: {len(pleth)} samples ({len(pleth)/SAMPLE_RATE:.0f} sec)")
    print(f"  PLETH NaN: {np.isnan(pleth).mean():.1%}, IBP1 NaN: {np.isnan(ibp1).mean():.1%}")

    # ── Find valid regions (both PLETH and IBP1 are non-NaN) ──
    valid_mask = ~np.isnan(pleth) & ~np.isnan(ibp1)

    # ── Segment into fixed-length windows ──
    seg_len = SEGMENT_SEC * SAMPLE_RATE  # samples per segment
    n_segments = len(pleth) // seg_len

    pleth_segments = []
    ibp1_segments = []
    sbp_values = []
    dbp_values = []
    mbp_values = []

    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len

        # Check validity ratio
        if valid_mask[start:end].mean() < MIN_VALID_RATIO:
            continue

        seg_pleth = pleth[start:end].copy()
        seg_ibp1 = ibp1[start:end].copy()

        # Interpolate tiny NaN gaps
        for seg in [seg_pleth, seg_ibp1]:
            nans = np.isnan(seg)
            if nans.any():
                seg[nans] = np.interp(
                    np.where(nans)[0],
                    np.where(~nans)[0],
                    seg[~nans]
                )

        # Basic range checks
        if seg_pleth.std() < 0.1:  # flat signal
            continue
        if seg_ibp1.max() > 300 or seg_ibp1.min() < 0:  # physiologically implausible
            continue

        # Apply bandpass filters
        try:
            seg_pleth_f = bandpass_filter(seg_pleth, 0.5, 8.0, SAMPLE_RATE)
            seg_ibp1_f = bandpass_filter(seg_ibp1, 0.5, 40.0, SAMPLE_RATE)
        except Exception:
            continue

        pleth_segments.append(seg_pleth_f.astype(np.float32))
        ibp1_segments.append(seg_ibp1_f.astype(np.float32))

        # Get BP values for this segment (use median of non-NaN values)
        seg_sbp = sbp[start:end]
        seg_dbp = dbp[start:end]
        seg_mbp = mbp[start:end]
        sbp_values.append(np.nanmedian(seg_sbp) if not np.all(np.isnan(seg_sbp)) else np.nan)
        dbp_values.append(np.nanmedian(seg_dbp) if not np.all(np.isnan(seg_dbp)) else np.nan)
        mbp_values.append(np.nanmedian(seg_mbp) if not np.all(np.isnan(seg_mbp)) else np.nan)

    if len(pleth_segments) == 0:
        print(f"  WARNING: No valid segments for {case_name}!")
        return None

    pleth_segments = np.array(pleth_segments)
    ibp1_segments = np.array(ibp1_segments)
    sbp_values = np.array(sbp_values, dtype=np.float32)
    dbp_values = np.array(dbp_values, dtype=np.float32)
    mbp_values = np.array(mbp_values, dtype=np.float32)

    print(f"  Valid segments: {len(pleth_segments)}")
    print(f"  Segment shape: {pleth_segments.shape}")
    print(f"  SBP range: {np.nanmin(sbp_values):.1f} ~ {np.nanmax(sbp_values):.1f} mmHg")
    print(f"  DBP range: {np.nanmin(dbp_values):.1f} ~ {np.nanmax(dbp_values):.1f} mmHg")

    return {
        'pleth': pleth_segments,
        'ibp1': ibp1_segments,
        'sbp': sbp_values,
        'dbp': dbp_values,
        'mbp': mbp_values,
    }


# ──────────────────────────── Process all cases ─────────────────
all_stats = []

for case_name in sorted(demographics.keys()):
    vital_path = os.path.join(CASES_DIR, f"{case_name}.vital")
    if not os.path.exists(vital_path):
        print(f"  File not found: {vital_path}")
        continue

    result = process_case(case_name, vital_path)
    if result is None:
        continue

    demo = demographics[case_name]
    out_path = os.path.join(OUTPUT_DIR, f"{case_name}.npz")
    np.savez_compressed(
        out_path,
        pleth=result['pleth'],
        ibp1=result['ibp1'],
        sbp=result['sbp'],
        dbp=result['dbp'],
        mbp=result['mbp'],
        age=demo['age'],
        sex=demo['sex'],
        sample_rate=SAMPLE_RATE,
        segment_sec=SEGMENT_SEC,
    )

    all_stats.append({
        'case': case_name,
        'n_segments': len(result['pleth']),
        'age': demo['age'],
        'sex': 'M' if demo['sex'] == 1 else 'F',
        'sbp_mean': np.nanmean(result['sbp']),
        'dbp_mean': np.nanmean(result['dbp']),
    })

# ──────────────────────────── Summary ───────────────────────────
print(f"\n{'='*60}")
print(f"EXTRACTION COMPLETE")
print(f"{'='*60}")
total_segments = sum(s['n_segments'] for s in all_stats)
print(f"Total cases processed: {len(all_stats)}")
print(f"Total segments: {total_segments}")
print(f"Segment duration: {SEGMENT_SEC}s at {SAMPLE_RATE}Hz = {SEGMENT_SEC * SAMPLE_RATE} samples")
for s in all_stats:
    print(f"  {s['case']}: {s['n_segments']:>5} segs, {s['sex']}, age={s['age']}, "
          f"SBP={s['sbp_mean']:.0f}, DBP={s['dbp_mean']:.0f}")
