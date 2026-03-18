"""
Step 3: 1D-CNN Deep Learning Model for PPG → ABP Prediction
- Subsampled for CPU efficiency (max 500 segments per patient)
- Lightweight CNN + late fusion of age/sex
- 5-Fold Group CV (patient-level split)
- Multi-task: SBP, DBP, MBP
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

PROCESSED_DIR = 'C:/Users/jaege/Desktop/Study/PPG2ABP/processed'
RESULTS_DIR = 'C:/Users/jaege/Desktop/Study/PPG2ABP/results'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

BATCH_SIZE = 256
EPOCHS = 20
LR = 2e-3
PATIENCE = 5
N_FOLDS = 5
MAX_SEG_PER_PATIENT = 500  # Subsample for speed
os.makedirs(RESULTS_DIR, exist_ok=True)

class PPGDataset(Dataset):
    def __init__(self, pleth, demo, sbp, dbp, mbp):
        self.pleth = torch.FloatTensor(pleth).unsqueeze(1)
        self.demo = torch.FloatTensor(demo)
        self.sbp = torch.FloatTensor(sbp)
        self.dbp = torch.FloatTensor(dbp)
        self.mbp = torch.FloatTensor(mbp)
    def __len__(self): return len(self.pleth)
    def __getitem__(self, i):
        return self.pleth[i], self.demo[i], self.sbp[i], self.dbp[i], self.mbp[i]

class PPG2BP_CNN(nn.Module):
    """Lightweight 1D-CNN for CPU training."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 15, padding=7), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),   # →(16,312)
            nn.Conv1d(16, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),   # →(32,78)
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),   # →(64,19)
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # →(128,1)
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 3),
        )
    def forward(self, ppg, demo):
        x = self.features(ppg).squeeze(-1)
        x = torch.cat([x, demo], dim=1)
        return self.head(x)

# ── Load & subsample ──
print("Loading data (subsampled)...")
all_pleth, all_demo, all_sbp, all_dbp, all_mbp, all_subjects = [], [], [], [], [], []
rng = np.random.RandomState(42)

for case_file in sorted(os.listdir(PROCESSED_DIR)):
    if not case_file.endswith('.npz'): continue
    case_name = case_file.replace('.npz', '')
    data = np.load(os.path.join(PROCESSED_DIR, case_file))
    pleth, sbp, dbp, mbp = data['pleth'], data['sbp'], data['dbp'], data['mbp']
    age, sex = float(data['age']), int(data['sex'])

    valid = []
    for i in range(len(pleth)):
        if np.isnan(sbp[i]) or np.isnan(dbp[i]): continue
        if sbp[i] < 30 or sbp[i] > 250 or dbp[i] < 10 or dbp[i] > 200 or sbp[i] <= dbp[i]: continue
        valid.append(i)

    if len(valid) > MAX_SEG_PER_PATIENT:
        valid = rng.choice(valid, MAX_SEG_PER_PATIENT, replace=False)

    for i in valid:
        all_pleth.append(pleth[i])
        all_demo.append([age, sex])
        all_sbp.append(sbp[i]); all_dbp.append(dbp[i]); all_mbp.append(mbp[i])
        all_subjects.append(case_name)

all_pleth = np.array(all_pleth, dtype=np.float32)
all_demo = np.array(all_demo, dtype=np.float32)
all_sbp = np.array(all_sbp, dtype=np.float32)
all_dbp = np.array(all_dbp, dtype=np.float32)
all_mbp = np.array(all_mbp, dtype=np.float32)
all_subjects = np.array(all_subjects)

demo_scaler = StandardScaler()
all_demo_scaled = demo_scaler.fit_transform(all_demo).astype(np.float32)

subject_names = np.unique(all_subjects)
subj_map = {s: i for i, s in enumerate(subject_names)}
subject_ids = np.array([subj_map[s] for s in all_subjects])

print(f"Samples: {len(all_pleth)}, Subjects: {len(subject_names)}")

# ── Training ──
def train_fold(fold, train_idx, test_idx):
    tr_p, te_p = all_pleth[train_idx], all_pleth[test_idx]
    mu, sigma = tr_p.mean(), tr_p.std()
    tr_pn, te_pn = (tr_p - mu) / (sigma + 1e-8), (te_p - mu) / (sigma + 1e-8)

    train_ds = PPGDataset(tr_pn, all_demo_scaled[train_idx], all_sbp[train_idx], all_dbp[train_idx], all_mbp[train_idx])
    test_ds = PPGDataset(te_pn, all_demo_scaled[test_idx], all_sbp[test_idx], all_dbp[test_idx], all_mbp[test_idx])
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = PPG2BP_CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    crit = nn.SmoothL1Loss()

    best_vl, patience_cnt, best_st = float('inf'), 0, None
    t0 = time.time()

    for ep in range(EPOCHS):
        model.train()
        for ppg, demo, sbp, dbp, mbp in train_ld:
            ppg, demo = ppg.to(DEVICE), demo.to(DEVICE)
            tgt = torch.stack([sbp, dbp, mbp], dim=1).to(DEVICE)
            opt.zero_grad()
            loss = crit(model(ppg, demo), tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vl = 0
        with torch.no_grad():
            for ppg, demo, sbp, dbp, mbp in test_ld:
                ppg, demo = ppg.to(DEVICE), demo.to(DEVICE)
                tgt = torch.stack([sbp, dbp, mbp], dim=1).to(DEVICE)
                vl += crit(model(ppg, demo), tgt).item() * len(ppg)
        vl /= len(test_ds)
        sched.step(vl)

        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    Early stop ep {ep+1}, {time.time()-t0:.0f}s")
                break

    if patience_cnt < PATIENCE:
        print(f"    Done {EPOCHS} eps, {time.time()-t0:.0f}s")

    model.load_state_dict(best_st)
    model.eval()
    preds = []
    with torch.no_grad():
        for ppg, demo, *_ in test_ld:
            preds.append(model(ppg.to(DEVICE), demo.to(DEVICE)).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return preds[:, 0], preds[:, 1], preds[:, 2]

print(f"\n{'='*60}")
print(f"5-Fold Group CV (max {MAX_SEG_PER_PATIENT} segs/patient)")
print(f"{'='*60}")

gkf = GroupKFold(n_splits=N_FOLDS)
res = {t: {'p': [], 't': []} for t in ['SBP','DBP','MBP']}

for fold, (tri, tei) in enumerate(gkf.split(all_pleth, groups=subject_ids)):
    te_subj = np.unique(all_subjects[tei])
    print(f"\nFold {fold+1}/{N_FOLDS}: train={len(tri)} test={len(tei)} ({len(te_subj)} subj)")

    ps, pd, pm = train_fold(fold, tri, tei)
    for t, p, true_arr in [('SBP',ps,all_sbp[tei]),('DBP',pd,all_dbp[tei]),('MBP',pm,all_mbp[tei])]:
        res[t]['p'].extend(p); res[t]['t'].extend(true_arr)

    print(f"  SBP MAE={mean_absolute_error(all_sbp[tei],ps):.2f} | "
          f"DBP MAE={mean_absolute_error(all_dbp[tei],pd):.2f} | "
          f"MBP MAE={mean_absolute_error(all_mbp[tei],pm):.2f}")

# ── Results ──
print(f"\n{'='*60}")
print("1D-CNN RESULTS")
print(f"{'='*60}")
print(f"{'Target':<6} {'MAE':>6} {'RMSE':>6} {'R²':>7} {'Bias±SD':>14} {'≤5':>5} {'≤10':>5} {'≤15':>5} {'BHS':>4}")
print("-"*62)

for t in ['SBP','DBP','MBP']:
    p, tr = np.array(res[t]['p']), np.array(res[t]['t'])
    mae = mean_absolute_error(tr, p)
    rmse = np.sqrt(mean_squared_error(tr, p))
    r2 = r2_score(tr, p)
    d = p - tr; bias, sd = np.mean(d), np.std(d)
    ad = np.abs(d)
    p5, p10, p15 = (ad<=5).mean()*100, (ad<=10).mean()*100, (ad<=15).mean()*100
    bhs = 'D'
    if p5>=60 and p10>=85 and p15>=95: bhs='A'
    elif p5>=50 and p10>=75 and p15>=90: bhs='B'
    elif p5>=40 and p10>=65 and p15>=85: bhs='C'
    print(f"{t:<6} {mae:>6.2f} {rmse:>6.2f} {r2:>7.4f} {bias:>+5.1f}±{sd:<5.1f} {p5:>5.1f} {p10:>5.1f} {p15:>5.1f} {bhs:>4}")

# Comparison
print(f"\n{'='*60}")
print("COMPARISON: XGBoost(LOSO) vs 1D-CNN(5-Fold)")
print(f"{'='*60}")
ml_path = os.path.join(RESULTS_DIR, 'ml_baseline_results.npz')
if os.path.exists(ml_path):
    ml = np.load(ml_path, allow_pickle=True)
    print(f"{'Target':<6} {'Model':<15} {'MAE':>6} {'RMSE':>6} {'R²':>7}")
    print("-"*45)
    for t in ['SBP','DBP','MBP']:
        mk = f"{t}_xgboost"
        mp, mt = ml[f"{mk}_preds"], ml[f"{mk}_true"]
        cp, ct = np.array(res[t]['p']), np.array(res[t]['t'])
        print(f"{t:<6} {'XGBoost(LOSO)':<15} {mean_absolute_error(mt,mp):>6.2f} {np.sqrt(mean_squared_error(mt,mp)):>6.2f} {r2_score(mt,mp):>7.4f}")
        print(f"{'':6} {'CNN(5-Fold)':<15} {mean_absolute_error(ct,cp):>6.2f} {np.sqrt(mean_squared_error(ct,cp)):>6.2f} {r2_score(ct,cp):>7.4f}")

np.savez(os.path.join(RESULTS_DIR, 'cnn_5fold_results.npz'),
    **{f"{t}_preds": np.array(res[t]['p']) for t in ['SBP','DBP','MBP']},
    **{f"{t}_true": np.array(res[t]['t']) for t in ['SBP','DBP','MBP']})
print(f"\nSaved to {RESULTS_DIR}/cnn_5fold_results.npz")
