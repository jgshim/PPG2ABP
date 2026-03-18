"""
Train U-Net 1D and CNN-LSTM waveform models (1-fold) and save weights
for use in the web app Research tab.
"""
import os, sys, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(line_buffering=True)

BASE = 'C:/Users/jaege/Desktop/Study/PPG2ABP'
PROCESSED_DIR = os.path.join(BASE, 'processed')
RESULTS_DIR = os.path.join(BASE, 'results')
MODELS_DIR = os.path.join(BASE, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

BATCH_SIZE = 256
LR = 1e-3
PATIENCE = 10
EPOCHS = 50

# ── Dataset ──
class WaveformDataset(Dataset):
    def __init__(self, pleth, ibp1, demo):
        self.pleth = torch.FloatTensor(pleth).unsqueeze(1)
        self.ibp1 = torch.FloatTensor(ibp1).unsqueeze(1)
        self.demo = torch.FloatTensor(demo)
    def __len__(self): return len(self.pleth)
    def __getitem__(self, i):
        return self.pleth[i], self.ibp1[i], self.demo[i]

# ── U-Net 1D ──
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
    def __init__(self, n_demo=2):
        super().__init__()
        self.enc1 = UNetBlock(1, 32)
        self.enc2 = UNetBlock(32, 64)
        self.enc3 = UNetBlock(64, 128)
        self.enc4 = UNetBlock(128, 256)
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = UNetBlock(256, 512)
        self.film_gamma = nn.Linear(n_demo, 512)
        self.film_beta = nn.Linear(n_demo, 512)
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
        orig_len = ppg.size(-1)
        pad_len = (16 - orig_len % 16) % 16
        if pad_len > 0: ppg = F.pad(ppg, (0, pad_len))
        e1 = self.enc1(ppg)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        gamma = self.film_gamma(demo).unsqueeze(-1)
        beta = self.film_beta(demo).unsqueeze(-1)
        b = gamma * b + beta
        d4 = self.dec4(torch.cat([self.up4(b)[:,:,:e4.size(2)], e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4)[:,:,:e3.size(2)], e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3)[:,:,:e2.size(2)], e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2)[:,:,:e1.size(2)], e1], dim=1))
        return self.final(d1)[:, :, :orig_len]

# ── CNN-LSTM ──
class CNNLSTM_WaveformModel(nn.Module):
    def __init__(self, n_demo=2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.film_gamma = nn.Linear(n_demo, 128)
        self.film_beta = nn.Linear(n_demo, 128)
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True,
                           bidirectional=True, dropout=0.2)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 1, 1),
        )

    def forward(self, ppg, demo):
        orig_len = ppg.size(-1)
        pad_len = (4 - orig_len % 4) % 4
        if pad_len > 0: ppg = F.pad(ppg, (0, pad_len))
        x = self.enc(ppg)
        gamma = self.film_gamma(demo).unsqueeze(-1)
        beta = self.film_beta(demo).unsqueeze(-1)
        x = gamma * x + beta
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.dec(x)
        if x.size(-1) >= orig_len:
            x = x[:, :, :orig_len]
        else:
            x = F.pad(x, (0, orig_len - x.size(-1)))
        return x

# ── Loss ──
def waveform_loss(pred, target):
    mse = F.mse_loss(pred, target)
    grad_pred = pred[:, :, 1:] - pred[:, :, :-1]
    grad_target = target[:, :, 1:] - target[:, :, :-1]
    grad_loss = F.mse_loss(grad_pred, grad_target)
    return mse + 0.3 * grad_loss

# ── Load Data ──
print("\nLoading data...")
all_pleth, all_ibp1, all_demo = [], [], []
all_subjects = []

for case_file in sorted(os.listdir(PROCESSED_DIR)):
    if not case_file.endswith('.npz'): continue
    data = np.load(os.path.join(PROCESSED_DIR, case_file))
    pleth, ibp1 = data['pleth'], data['ibp1']
    sbp, dbp = data['sbp'], data['dbp']
    age, sex = float(data['age']), int(data['sex'])
    for i in range(len(pleth)):
        if np.isnan(sbp[i]) or np.isnan(dbp[i]): continue
        if sbp[i] < 30 or sbp[i] > 250 or dbp[i] < 10 or dbp[i] > 200 or sbp[i] <= dbp[i]: continue
        all_pleth.append(pleth[i]); all_ibp1.append(ibp1[i])
        all_demo.append([age, sex])
        all_subjects.append(case_file.replace('.npz', ''))

all_pleth = np.array(all_pleth, dtype=np.float32)
all_ibp1 = np.array(all_ibp1, dtype=np.float32)
all_demo = np.array(all_demo, dtype=np.float32)
all_subjects = np.array(all_subjects)

demo_scaler = StandardScaler()
all_demo_s = demo_scaler.fit_transform(all_demo).astype(np.float32)

unique_subjects = np.unique(all_subjects)
subj_map = {s: i for i, s in enumerate(unique_subjects)}
subject_ids = np.array([subj_map[s] for s in all_subjects])
print(f"Total: {len(all_pleth)} samples, {len(unique_subjects)} subjects")

# ── Train & Save ──
def train_and_save(model_class, model_name, fold_idx=0):
    print(f"\n{'='*60}")
    print(f"  Training {model_name} - Fold {fold_idx+1}")
    print(f"{'='*60}")

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(all_pleth, groups=subject_ids))
    tr_idx, te_idx = folds[fold_idx]

    # Normalize
    p_mu, p_sig = float(all_pleth[tr_idx].mean()), float(all_pleth[tr_idx].std())
    a_mu, a_sig = float(all_ibp1[tr_idx].mean()), float(all_ibp1[tr_idx].std())
    tr_p = (all_pleth[tr_idx] - p_mu) / (p_sig + 1e-8)
    tr_a = (all_ibp1[tr_idx] - a_mu) / (a_sig + 1e-8)

    # 90/10 train/val split
    n_tr = len(tr_idx)
    perm = np.random.RandomState(42).permutation(n_tr)
    n_val = int(n_tr * 0.1)
    val_perm, trn_perm = perm[:n_val], perm[n_val:]

    trn_ds = WaveformDataset(tr_p[trn_perm], tr_a[trn_perm], all_demo_s[tr_idx][trn_perm])
    val_ds = WaveformDataset(tr_p[val_perm], tr_a[val_perm], all_demo_s[tr_idx][val_perm])

    trn_ld = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    model = model_class().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_vl, pat_cnt, best_st = float('inf'), 0, None
    t0 = time.time()

    for ep in range(EPOCHS):
        model.train()
        ep_loss = 0
        for ppg, abp, demo in trn_ld:
            ppg = ppg.to(DEVICE, non_blocking=True)
            abp = abp.to(DEVICE, non_blocking=True)
            demo = demo.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(ppg, demo)
            loss = waveform_loss(pred, abp)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(ppg)
        sched.step()
        ep_loss /= len(trn_ds)

        model.eval()
        vl = 0
        with torch.no_grad():
            for ppg, abp, demo in val_ld:
                ppg = ppg.to(DEVICE, non_blocking=True)
                abp = abp.to(DEVICE, non_blocking=True)
                demo = demo.to(DEVICE, non_blocking=True)
                vl += waveform_loss(model(ppg, demo), abp).item() * len(ppg)
        vl /= len(val_ds)

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    Epoch {ep+1:3d}: train={ep_loss:.6f} val={vl:.6f} [{time.time()-t0:.0f}s]")

        if vl < best_vl:
            best_vl = vl
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat_cnt = 0
        else:
            pat_cnt += 1
            if pat_cnt >= PATIENCE:
                print(f"    Early stop at epoch {ep+1}")
                break

    print(f"  Training: {time.time()-t0:.0f}s")

    # Save model weights + normalization params
    save_path = os.path.join(MODELS_DIR, f'{model_name}.pt')
    torch.save({
        'model_state_dict': best_st,
        'p_mu': p_mu, 'p_sig': p_sig,
        'a_mu': a_mu, 'a_sig': a_sig,
        'demo_scaler_mean': demo_scaler.mean_.tolist(),
        'demo_scaler_scale': demo_scaler.scale_.tolist(),
    }, save_path)
    print(f"  Saved: {save_path}")

    del model; gc.collect(); torch.cuda.empty_cache()


# Train both models
train_and_save(UNet1D, 'unet1d', fold_idx=0)
train_and_save(CNNLSTM_WaveformModel, 'cnn_lstm', fold_idx=0)

print("\nDone! Models saved to:", MODELS_DIR)
