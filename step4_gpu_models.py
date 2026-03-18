"""
Step 4: GPU-accelerated models with FULL data + 5-Fold Group CV
- Model A: Improved 1D-CNN (deeper, full data)
- Model B: ResNet1D (residual connections)
- Model C: U-Net 1D (waveform reconstruction PPG -> ABP)
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

BATCH_SIZE = 512
LR = 1e-3
PATIENCE = 8
N_FOLDS = 5
EPOCHS_NUMERIC = 40
EPOCHS_UNET = 40
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================== Datasets ===============================
class PPGDataset(Dataset):
    def __init__(self, pleth, demo, sbp, dbp, mbp):
        self.pleth = torch.FloatTensor(pleth).unsqueeze(1)
        self.demo = torch.FloatTensor(demo)
        self.targets = torch.FloatTensor(np.stack([sbp, dbp, mbp], axis=1))
    def __len__(self): return len(self.pleth)
    def __getitem__(self, i):
        return self.pleth[i], self.demo[i], self.targets[i]

class WaveformDataset(Dataset):
    def __init__(self, pleth, ibp1, demo):
        self.pleth = torch.FloatTensor(pleth).unsqueeze(1)
        self.ibp1 = torch.FloatTensor(ibp1).unsqueeze(1)
        self.demo = torch.FloatTensor(demo)
    def __len__(self): return len(self.pleth)
    def __getitem__(self, i):
        return self.pleth[i], self.ibp1[i], self.demo[i]


# =============================== Model A: Improved CNN ===============================
class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.1))
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, 9, padding=4), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 9, padding=4), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.1))
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2))
        self.block4 = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.2))
        self.block5 = nn.Sequential(
            nn.Conv1d(256, 512, 3, padding=1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(
            nn.Linear(512 + 2, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 3))

    def forward(self, x, demo):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x).squeeze(-1)
        return self.head(torch.cat([x, demo], dim=1))


# =============================== Model B: ResNet1D ===============================
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, stride=1, downsample=None):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 15, stride=2, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(512 + 2, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 3))

    def _make_layer(self, in_ch, out_ch, n_blocks, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride), nn.BatchNorm1d(out_ch))
        layers = [ResBlock1D(in_ch, out_ch, stride=stride, downsample=downsample)]
        for _ in range(1, n_blocks):
            layers.append(ResBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x, demo):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.gap(x).squeeze(-1)
        return self.head(torch.cat([x, demo], dim=1))


# =============================== Model C: U-Net 1D ===============================
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=k//2),
            nn.BatchNorm1d(out_ch), nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, k, padding=k//2),
            nn.BatchNorm1d(out_ch), nn.ReLU())
    def forward(self, x): return self.conv(x)

class UNet1D(nn.Module):
    """1D U-Net: PPG waveform -> ABP waveform with FiLM conditioning."""
    def __init__(self, n_demo=2):
        super().__init__()
        self.enc1 = UNetBlock(1, 32)
        self.enc2 = UNetBlock(32, 64)
        self.enc3 = UNetBlock(64, 128)
        self.enc4 = UNetBlock(128, 256)
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = UNetBlock(256, 512)
        # FiLM conditioning
        self.film_gamma = nn.Linear(n_demo, 512)
        self.film_beta = nn.Linear(n_demo, 512)
        # Decoder
        self.up4 = nn.ConvTranspose1d(512, 256, 2, stride=2)
        self.dec4 = UNetBlock(512, 256)
        self.up3 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.dec3 = UNetBlock(256, 128)
        self.up2 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.dec2 = UNetBlock(128, 64)
        self.up1 = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.dec1 = UNetBlock(64, 32)
        self.final = nn.Conv1d(32, 1, 1)

    def forward(self, ppg, demo):
        # Pad to multiple of 16
        orig_len = ppg.size(-1)
        pad_len = (16 - orig_len % 16) % 16
        if pad_len > 0:
            ppg = F.pad(ppg, (0, pad_len))
        # Encoder
        e1 = self.enc1(ppg)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        # Bottleneck + FiLM
        b = self.bottleneck(self.pool(e4))
        gamma = self.film_gamma(demo).unsqueeze(-1)
        beta = self.film_beta(demo).unsqueeze(-1)
        b = gamma * b + beta
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b)[:,:,:e4.size(2)], e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4)[:,:,:e3.size(2)], e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3)[:,:,:e2.size(2)], e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2)[:,:,:e1.size(2)], e1], dim=1))
        out = self.final(d1)
        return out[:, :, :orig_len]


# =============================== Load Data ===============================
print("\nLoading full data...")
all_pleth, all_ibp1, all_demo = [], [], []
all_sbp, all_dbp, all_mbp = [], [], []
all_subjects = []

for case_file in sorted(os.listdir(PROCESSED_DIR)):
    if not case_file.endswith('.npz'): continue
    case_name = case_file.replace('.npz', '')
    data = np.load(os.path.join(PROCESSED_DIR, case_file))
    pleth, ibp1 = data['pleth'], data['ibp1']
    sbp, dbp, mbp = data['sbp'], data['dbp'], data['mbp']
    age, sex = float(data['age']), int(data['sex'])
    for i in range(len(pleth)):
        if np.isnan(sbp[i]) or np.isnan(dbp[i]): continue
        if sbp[i] < 30 or sbp[i] > 250 or dbp[i] < 10 or dbp[i] > 200 or sbp[i] <= dbp[i]: continue
        all_pleth.append(pleth[i]); all_ibp1.append(ibp1[i])
        all_demo.append([age, sex])
        all_sbp.append(sbp[i]); all_dbp.append(dbp[i]); all_mbp.append(mbp[i])
        all_subjects.append(case_name)

all_pleth = np.array(all_pleth, dtype=np.float32)
all_ibp1 = np.array(all_ibp1, dtype=np.float32)
all_demo = np.array(all_demo, dtype=np.float32)
all_sbp = np.array(all_sbp, dtype=np.float32)
all_dbp = np.array(all_dbp, dtype=np.float32)
all_mbp = np.array(all_mbp, dtype=np.float32)
all_subjects = np.array(all_subjects)

demo_scaler = StandardScaler()
all_demo_s = demo_scaler.fit_transform(all_demo).astype(np.float32)

unique_subjects = np.unique(all_subjects)
subj_map = {s: i for i, s in enumerate(unique_subjects)}
subject_ids = np.array([subj_map[s] for s in all_subjects])

print(f"Total: {len(all_pleth)} samples, {len(unique_subjects)} subjects")
print(f"SBP: {all_sbp.mean():.0f}+/-{all_sbp.std():.0f}, DBP: {all_dbp.mean():.0f}+/-{all_dbp.std():.0f}")


# =============================== Training: Numeric Models ===============================
def train_numeric(model_class, model_name, epochs=EPOCHS_NUMERIC):
    print(f"\n{'='*70}")
    print(f"  {model_name} - 5-Fold Group CV, {len(all_pleth)} samples, GPU")
    print(f"{'='*70}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    all_preds = np.zeros((len(all_pleth), 3), dtype=np.float32)
    t_total = time.time()

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(all_pleth, groups=subject_ids)):
        t0 = time.time()
        te_subj = np.unique(all_subjects[te_idx])

        # Normalize PPG
        mu, sig = all_pleth[tr_idx].mean(), all_pleth[tr_idx].std()
        tr_p = (all_pleth[tr_idx] - mu) / (sig + 1e-8)
        te_p = (all_pleth[te_idx] - mu) / (sig + 1e-8)

        tr_ds = PPGDataset(tr_p, all_demo_s[tr_idx], all_sbp[tr_idx], all_dbp[tr_idx], all_mbp[tr_idx])
        te_ds = PPGDataset(te_p, all_demo_s[te_idx], all_sbp[te_idx], all_dbp[te_idx], all_mbp[te_idx])
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        model = model_class().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.SmoothL1Loss()

        best_vl, pat_cnt, best_st = float('inf'), 0, None
        for ep in range(epochs):
            model.train()
            for ppg, demo, tgt in tr_ld:
                ppg, demo, tgt = ppg.to(DEVICE, non_blocking=True), demo.to(DEVICE, non_blocking=True), tgt.to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(ppg, demo), tgt)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            vl = 0
            with torch.no_grad():
                for ppg, demo, tgt in te_ld:
                    ppg, demo, tgt = ppg.to(DEVICE, non_blocking=True), demo.to(DEVICE, non_blocking=True), tgt.to(DEVICE, non_blocking=True)
                    vl += crit(model(ppg, demo), tgt).item() * len(ppg)
            vl /= len(te_ds)
            if vl < best_vl:
                best_vl = vl; best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}; pat_cnt = 0
            else:
                pat_cnt += 1
                if pat_cnt >= PATIENCE: break

        model.load_state_dict(best_st); model.eval()
        preds = []
        with torch.no_grad():
            for ppg, demo, _ in te_ld:
                preds.append(model(ppg.to(DEVICE, non_blocking=True), demo.to(DEVICE, non_blocking=True)).cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        all_preds[te_idx] = preds

        s_mae = mean_absolute_error(all_sbp[te_idx], preds[:, 0])
        d_mae = mean_absolute_error(all_dbp[te_idx], preds[:, 1])
        m_mae = mean_absolute_error(all_mbp[te_idx], preds[:, 2])
        print(f"  Fold {fold+1}/{N_FOLDS}: {len(te_subj)} subj ({len(te_idx)} seg) "
              f"SBP={s_mae:.2f} DBP={d_mae:.2f} MBP={m_mae:.2f} ep={ep+1} [{time.time()-t0:.0f}s]")

    total_time = time.time() - t_total
    return print_metrics(model_name, all_preds, total_time)


# =============================== Training: U-Net ===============================
def train_unet_model(epochs=EPOCHS_UNET):
    model_name = "U-Net 1D"
    print(f"\n{'='*70}")
    print(f"  {model_name} - 5-Fold Group CV, {len(all_pleth)} samples, GPU")
    print(f"{'='*70}")

    gkf = GroupKFold(n_splits=N_FOLDS)
    all_recon = np.zeros_like(all_ibp1)
    t_total = time.time()

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(all_pleth, groups=subject_ids)):
        t0 = time.time()
        te_subj = np.unique(all_subjects[te_idx])

        # Normalize
        p_mu, p_sig = all_pleth[tr_idx].mean(), all_pleth[tr_idx].std()
        a_mu, a_sig = all_ibp1[tr_idx].mean(), all_ibp1[tr_idx].std()
        tr_p = (all_pleth[tr_idx] - p_mu) / (p_sig + 1e-8)
        te_p = (all_pleth[te_idx] - p_mu) / (p_sig + 1e-8)
        tr_a = (all_ibp1[tr_idx] - a_mu) / (a_sig + 1e-8)

        tr_ds = WaveformDataset(tr_p, tr_a, all_demo_s[tr_idx])
        te_ds = WaveformDataset(te_p, np.zeros_like(all_ibp1[te_idx]), all_demo_s[te_idx])
        tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        model = UNet1D().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        # Combined loss: MSE + waveform gradient loss
        def combined_loss(pred, target):
            mse = F.mse_loss(pred, target)
            # Gradient loss (preserve waveform shape)
            grad_pred = pred[:, :, 1:] - pred[:, :, :-1]
            grad_target = target[:, :, 1:] - target[:, :, :-1]
            grad_loss = F.mse_loss(grad_pred, grad_target)
            return mse + 0.5 * grad_loss

        best_vl, pat_cnt, best_st = float('inf'), 0, None
        for ep in range(epochs):
            model.train()
            ep_loss = 0
            for ppg, abp, demo in tr_ld:
                ppg = ppg.to(DEVICE, non_blocking=True)
                abp = abp.to(DEVICE, non_blocking=True)
                demo = demo.to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                pred = model(ppg, demo)
                loss = combined_loss(pred, abp)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item() * len(ppg)
            sched.step()
            ep_loss /= len(tr_ds)

            if ep_loss < best_vl:
                best_vl = ep_loss
                best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                pat_cnt = 0
            else:
                pat_cnt += 1
                if pat_cnt >= PATIENCE: break

        model.load_state_dict(best_st); model.eval()
        recon_list = []
        with torch.no_grad():
            for ppg, _, demo in te_ld:
                pred = model(ppg.to(DEVICE, non_blocking=True), demo.to(DEVICE, non_blocking=True))
                recon_list.append(pred.cpu().numpy())
        recon = np.concatenate(recon_list, axis=0).squeeze(1)
        # Denormalize
        recon_denorm = recon * a_sig + a_mu
        all_recon[te_idx] = recon_denorm

        # Waveform correlation
        te_abp = all_ibp1[te_idx]
        corrs = []
        for i in range(len(te_abp)):
            c = np.corrcoef(recon_denorm[i], te_abp[i])[0, 1]
            if not np.isnan(c): corrs.append(c)
        mean_corr = np.mean(corrs) if corrs else 0

        # Waveform RMSE
        wf_rmse = np.sqrt(np.mean((recon_denorm - te_abp) ** 2))

        print(f"  Fold {fold+1}/{N_FOLDS}: {len(te_subj)} subj ({len(te_idx)} seg) "
              f"WF_RMSE={wf_rmse:.2f} Corr={mean_corr:.3f} ep={ep+1} [{time.time()-t0:.0f}s]")

    total_time = time.time() - t_total

    # Global waveform metrics
    wf_rmse_all = np.sqrt(np.mean((all_recon - all_ibp1) ** 2))
    corrs_all = []
    for i in range(len(all_ibp1)):
        c = np.corrcoef(all_recon[i], all_ibp1[i])[0, 1]
        if not np.isnan(c): corrs_all.append(c)
    wf_corr_all = np.mean(corrs_all)

    print(f"\n  Waveform Metrics:")
    print(f"    RMSE: {wf_rmse_all:.2f}")
    print(f"    Correlation: {wf_corr_all:.4f}")

    # Extract SBP/DBP/MBP from reconstructed waveform
    pred_sbp = np.array([seg.max() for seg in all_recon])
    pred_dbp = np.array([seg.min() for seg in all_recon])
    pred_mbp = np.array([seg.mean() for seg in all_recon])
    all_preds = np.stack([pred_sbp, pred_dbp, pred_mbp], axis=1)

    result = print_metrics(model_name, all_preds, total_time)
    result['wf_rmse'] = wf_rmse_all
    result['wf_corr'] = wf_corr_all
    result['recon'] = all_recon
    return result


# =============================== Evaluation ===============================
def print_metrics(model_name, all_preds, total_time):
    print(f"\n  --- {model_name} ({total_time:.0f}s) ---")
    print(f"  {'Tgt':<4} {'MAE':>6} {'RMSE':>6} {'R2':>7} {'Bias+/-SD':>14} "
          f"{'<=5':>5} {'<=10':>5} {'<=15':>5} {'BHS':>4}")
    print(f"  {'-'*62}")

    result = {'model': model_name, 'time': total_time}
    for t, idx, true in [('SBP',0,all_sbp), ('DBP',1,all_dbp), ('MBP',2,all_mbp)]:
        p = all_preds[:, idx]
        mae = mean_absolute_error(true, p)
        rmse = np.sqrt(mean_squared_error(true, p))
        r2 = r2_score(true, p)
        d = p - true; bias, sd = np.mean(d), np.std(d)
        ad = np.abs(d)
        p5, p10, p15 = (ad<=5).mean()*100, (ad<=10).mean()*100, (ad<=15).mean()*100
        bhs = 'D'
        if p5>=60 and p10>=85 and p15>=95: bhs='A'
        elif p5>=50 and p10>=75 and p15>=90: bhs='B'
        elif p5>=40 and p10>=65 and p15>=85: bhs='C'

        print(f"  {t:<4} {mae:>6.2f} {rmse:>6.2f} {r2:>7.4f} {bias:>+5.1f}+/-{sd:<5.1f} "
              f"{p5:>5.1f} {p10:>5.1f} {p15:>5.1f} {bhs:>4}")
        result[f'{t}_mae'] = mae; result[f'{t}_rmse'] = rmse
        result[f'{t}_r2'] = r2; result[f'{t}_bhs'] = bhs
        result[f'{t}_preds'] = p
    return result


# =============================== Run All ===============================
results = {}

# Model A
results['CNN'] = train_numeric(ImprovedCNN, "Improved 1D-CNN", EPOCHS_NUMERIC)
torch.cuda.empty_cache()

# Model B
results['ResNet'] = train_numeric(ResNet1D, "ResNet1D", EPOCHS_NUMERIC)
torch.cuda.empty_cache()

# Model C
results['UNet'] = train_unet_model(EPOCHS_UNET)
torch.cuda.empty_cache()


# =============================== Final Comparison ===============================
print(f"\n{'='*80}")
print(f"  FINAL COMPARISON - All Models (5-Fold Group CV, {len(all_pleth)} samples)")
print(f"{'='*80}")
print(f"  {'Model':<22} {'SBP MAE':>8} {'DBP MAE':>8} {'MBP MAE':>8} "
      f"{'SBP R2':>8} {'DBP R2':>8} {'DBP BHS':>8} {'Time':>7}")
print(f"  {'-'*82}")

# Previous ML baseline (LOSO)
ml_path = os.path.join(RESULTS_DIR, 'ml_baseline_results.npz')
if os.path.exists(ml_path):
    ml = np.load(ml_path, allow_pickle=True)
    for m in ['xgboost', 'lightgbm']:
        s_mae = mean_absolute_error(ml[f'SBP_{m}_true'], ml[f'SBP_{m}_preds'])
        d_mae = mean_absolute_error(ml[f'DBP_{m}_true'], ml[f'DBP_{m}_preds'])
        m_mae = mean_absolute_error(ml[f'MBP_{m}_true'], ml[f'MBP_{m}_preds'])
        s_r2 = r2_score(ml[f'SBP_{m}_true'], ml[f'SBP_{m}_preds'])
        d_r2 = r2_score(ml[f'DBP_{m}_true'], ml[f'DBP_{m}_preds'])
        print(f"  {m.upper()+' (LOSO)':<22} {s_mae:>8.2f} {d_mae:>8.2f} {m_mae:>8.2f} "
              f"{s_r2:>8.4f} {d_r2:>8.4f} {'D':>8} {'N/A':>7}")

# Previous CNN (5-fold, subsampled)
prev_path = os.path.join(RESULTS_DIR, 'cnn_5fold_results.npz')
if os.path.exists(prev_path):
    prev = np.load(prev_path)
    s_mae = mean_absolute_error(prev['SBP_true'], prev['SBP_preds'])
    d_mae = mean_absolute_error(prev['DBP_true'], prev['DBP_preds'])
    m_mae = mean_absolute_error(prev['MBP_true'], prev['MBP_preds'])
    s_r2 = r2_score(prev['SBP_true'], prev['SBP_preds'])
    d_r2 = r2_score(prev['DBP_true'], prev['DBP_preds'])
    print(f"  {'Prev CNN (sub)':<22} {s_mae:>8.2f} {d_mae:>8.2f} {m_mae:>8.2f} "
          f"{s_r2:>8.4f} {d_r2:>8.4f} {'C':>8} {'N/A':>7}")

# New models
for name, res in results.items():
    extra = ''
    if 'wf_corr' in res:
        extra = f" WF_corr={res['wf_corr']:.3f}"
    print(f"  {res['model']:<22} {res['SBP_mae']:>8.2f} {res['DBP_mae']:>8.2f} {res['MBP_mae']:>8.2f} "
          f"{res['SBP_r2']:>8.4f} {res['DBP_r2']:>8.4f} {res['DBP_bhs']:>8} "
          f"{res['time']:>5.0f}s{extra}")

# Save
save_dict = {}
for name, res in results.items():
    for t in ['SBP', 'DBP', 'MBP']:
        save_dict[f'{name}_{t}_preds'] = res[f'{t}_preds']
save_dict['true_sbp'] = all_sbp
save_dict['true_dbp'] = all_dbp
save_dict['true_mbp'] = all_mbp
save_dict['subjects'] = all_subjects
if 'recon' in results.get('UNet', {}):
    save_dict['unet_recon'] = results['UNet']['recon']
    save_dict['true_ibp1'] = all_ibp1
np.savez(os.path.join(RESULTS_DIR, 'gpu_all_models_results.npz'), **save_dict)
print(f"\nResults saved to {RESULTS_DIR}/gpu_all_models_results.npz")
