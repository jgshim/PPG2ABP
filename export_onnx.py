"""
Export PyTorch U-Net 1D and CNN-LSTM models to ONNX format for Vercel deployment.
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MODEL_DIR = 'C:/Users/jaege/Desktop/Study/PPG2ABP/models'
OUTPUT_DIR = 'C:/Users/jaege/Desktop/Study/EEG_PostopPain/Vital/240524/models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEG_LEN = 1250  # 125Hz * 10s

# ── Model definitions (must match training code) ──
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
        self.enc1=UNetBlock(1,32);self.enc2=UNetBlock(32,64);self.enc3=UNetBlock(64,128);self.enc4=UNetBlock(128,256)
        self.pool=nn.MaxPool1d(2);self.bottleneck=UNetBlock(256,512)
        self.film_gamma=nn.Linear(n_demo,512);self.film_beta=nn.Linear(n_demo,512)
        self.up4=nn.ConvTranspose1d(512,256,2,stride=2);self.dec4=UNetBlock(512,256)
        self.up3=nn.ConvTranspose1d(256,128,2,stride=2);self.dec3=UNetBlock(256,128)
        self.up2=nn.ConvTranspose1d(128,64,2,stride=2);self.dec2=UNetBlock(128,64)
        self.up1=nn.ConvTranspose1d(64,32,2,stride=2);self.dec1=UNetBlock(64,32)
        self.final=nn.Conv1d(32,1,1)
    def forward(self, ppg, demo):
        ol=ppg.size(-1);pl=(16-ol%16)%16
        if pl>0:ppg=F.pad(ppg,(0,pl))
        e1=self.enc1(ppg);e2=self.enc2(self.pool(e1));e3=self.enc3(self.pool(e2));e4=self.enc4(self.pool(e3))
        b=self.bottleneck(self.pool(e4))
        g=self.film_gamma(demo).unsqueeze(-1);bt=self.film_beta(demo).unsqueeze(-1);b=g*b+bt
        d4=self.dec4(torch.cat([self.up4(b)[:,:,:e4.size(2)],e4],1))
        d3=self.dec3(torch.cat([self.up3(d4)[:,:,:e3.size(2)],e3],1))
        d2=self.dec2(torch.cat([self.up2(d3)[:,:,:e2.size(2)],e2],1))
        d1=self.dec1(torch.cat([self.up1(d2)[:,:,:e1.size(2)],e1],1))
        return self.final(d1)[:,:,:ol]

class CNNLSTM_WaveformModel(nn.Module):
    def __init__(self, n_demo=2):
        super().__init__()
        self.enc=nn.Sequential(
            nn.Conv1d(1,32,7,padding=3),nn.BatchNorm1d(32),nn.ReLU(),
            nn.Conv1d(32,64,5,padding=2),nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(2),
            nn.Conv1d(64,128,5,padding=2),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Conv1d(128,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(2))
        self.film_gamma=nn.Linear(n_demo,128);self.film_beta=nn.Linear(n_demo,128)
        self.lstm=nn.LSTM(128,128,num_layers=2,batch_first=True,bidirectional=True,dropout=0.2)
        self.dec=nn.Sequential(
            nn.ConvTranspose1d(256,128,4,stride=2,padding=1),nn.BatchNorm1d(128),nn.ReLU(),
            nn.ConvTranspose1d(128,64,4,stride=2,padding=1),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Conv1d(64,32,3,padding=1),nn.BatchNorm1d(32),nn.ReLU(),nn.Conv1d(32,1,1))
    def forward(self, ppg, demo):
        ol=ppg.size(-1);pl=(4-ol%4)%4
        if pl>0:ppg=F.pad(ppg,(0,pl))
        x=self.enc(ppg)
        g=self.film_gamma(demo).unsqueeze(-1);bt=self.film_beta(demo).unsqueeze(-1);x=g*x+bt
        x=x.permute(0,2,1);x,_=self.lstm(x);x=x.permute(0,2,1)
        x=self.dec(x)
        return x[:,:,:ol] if x.size(-1)>=ol else F.pad(x,(0,ol-x.size(-1)))

# ── Export ──
for model_name, ModelClass, pt_file in [
    ('unet1d', UNet1D, 'unet1d.pt'),
    ('cnn_lstm', CNNLSTM_WaveformModel, 'cnn_lstm.pt'),
]:
    print(f"\n{'='*50}")
    print(f"  Exporting {model_name}")
    print(f"{'='*50}")

    ckpt = torch.load(os.path.join(MODEL_DIR, pt_file), map_location='cpu', weights_only=False)
    model = ModelClass()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Dummy inputs
    dummy_ppg = torch.randn(1, 1, SEG_LEN)
    dummy_demo = torch.randn(1, 2)

    # Export to ONNX
    onnx_path = os.path.join(OUTPUT_DIR, f'{model_name}.onnx')
    torch.onnx.export(
        model,
        (dummy_ppg, dummy_demo),
        onnx_path,
        input_names=['ppg', 'demo'],
        output_names=['abp'],
        dynamic_axes={
            'ppg': {0: 'batch'},
            'demo': {0: 'batch'},
            'abp': {0: 'batch'},
        },
        opset_version=17,
    )
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  Saved: {onnx_path} ({size_mb:.1f} MB)")

    # Save normalization params as JSON
    norm = {
        'p_mu': float(ckpt['p_mu']),
        'p_sig': float(ckpt['p_sig']),
        'a_mu': float(ckpt['a_mu']),
        'a_sig': float(ckpt['a_sig']),
        'demo_mean': [float(x) for x in ckpt['demo_scaler_mean']],
        'demo_scale': [float(x) for x in ckpt['demo_scaler_scale']],
    }
    norm_path = os.path.join(OUTPUT_DIR, f'{model_name}_norm.json')
    with open(norm_path, 'w') as f:
        json.dump(norm, f, indent=2)
    print(f"  Saved: {norm_path}")

    # Verify: compare PyTorch vs ONNX output
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    with torch.no_grad():
        pt_out = model(dummy_ppg, dummy_demo).numpy()
    onnx_out = sess.run(None, {
        'ppg': dummy_ppg.numpy(),
        'demo': dummy_demo.numpy(),
    })[0]
    diff = np.abs(pt_out - onnx_out).max()
    print(f"  Max diff (PyTorch vs ONNX): {diff:.8f}")

print("\nDone!")
