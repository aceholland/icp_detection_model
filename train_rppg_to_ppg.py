"""
Model 1: rPPG → PPG
Trains a 1D UNet to convert noisy camera rPPG to clean PPG
Dataset: UBFC-rPPG (or synthetic until you get it)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, resample
import json, os

from project_paths import MODELS_DIR

# ══════════════════════════════════════════════════════
# MODEL — 1D UNet
# ══════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 9, padding=4),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, 9, padding=4),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)

class UNet1D(nn.Module):
    """
    1D UNet — encoder compresses, decoder reconstructs
    Skip connections preserve signal detail
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Decoder
        self.up3   = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.dec3  = ConvBlock(256, 128)
        self.up2   = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.dec2  = ConvBlock(128, 64)
        self.up1   = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.dec1  = ConvBlock(64, 32)

        # Output
        self.out   = nn.Conv1d(32, 1, 1)

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b  = self.bottleneck(self.pool(e3))

        # Decode with skip connections
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)

# ══════════════════════════════════════════════════════
# DATASET — synthetic until UBFC arrives
# ══════════════════════════════════════════════════════
def bandpass(sig, lo, hi, fs, order=4):
    nyq  = fs / 2
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig)

def generate_synthetic_pair(hr_bpm=72, fs=30, duration=30):
    """
    Generate paired (noisy rPPG, clean PPG) for one person
    """
    N   = duration * fs
    t   = np.arange(N) / fs
    hr  = hr_bpm / 60

    # Clean PPG — realistic waveform with harmonics
    ppg_clean  = np.sin(2 * np.pi * hr * t)
    ppg_clean += 0.4 * np.sin(4 * np.pi * hr * t)
    ppg_clean += 0.1 * np.sin(6 * np.pi * hr * t)
    ppg_clean  = ppg_clean / np.std(ppg_clean)

    # rPPG — same signal but with realistic noise
    motion     = 0.8 * np.sin(2 * np.pi * 0.3 * t)   # motion artifact
    lighting   = 1.2 * np.sin(2 * np.pi * 0.05 * t)  # lighting flicker
    skin_noise = np.random.normal(0, 0.4, N)           # skin texture noise
    rppg_noisy = ppg_clean + motion + lighting + skin_noise

    # Apply CHROM-like normalization
    rppg_noisy = bandpass(rppg_noisy, 0.7, 4.0, fs)
    rppg_noisy = rppg_noisy / (np.std(rppg_noisy) + 1e-8)

    return rppg_noisy.astype(np.float32), ppg_clean.astype(np.float32)

class SyntheticrPPGDataset(Dataset):
    def __init__(self, n_samples=500, window=256, fs=30):
        self.windows = []
        hr_values = np.random.uniform(50, 110, n_samples)

        for hr in hr_values:
            rppg, ppg = generate_synthetic_pair(hr_bpm=hr, fs=fs)
            # Sliding windows
            for start in range(0, len(rppg) - window, window // 2):
                r = rppg[start:start+window]
                p = ppg[start:start+window]
                self.windows.append((r, p))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        r, p = self.windows[idx]
        return (
            torch.tensor(r).unsqueeze(0),  # [1, window]
            torch.tensor(p).unsqueeze(0)   # [1, window]
        )

# ══════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════
def train_rppg_to_ppg():
    print("\n[Model 1] Training rPPG → PPG UNet...")

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Model 1] Device: {device}")

    dataset   = SyntheticrPPGDataset(n_samples=500, window=256)
    n_train   = int(0.8 * len(dataset))
    n_val     = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32)

    model     = UNet1D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val  = float('inf')

    for epoch in range(80):
        # Train
        model.train()
        train_loss = 0
        for rppg, ppg in train_loader:
            rppg, ppg = rppg.to(device), ppg.to(device)
            optimizer.zero_grad()
            pred = model(rppg)
            loss = criterion(pred, ppg)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for rppg, ppg in val_loader:
                rppg, ppg = rppg.to(device), ppg.to(device)
                pred = model(rppg)
                val_loss += criterion(pred, ppg).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODELS_DIR / 'rppg_to_ppg.pth')

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | "
                  f"train {train_loss:.4f} | "
                  f"val {val_loss:.4f}")

    print(f"[Model 1] Done — best val loss: {best_val:.4f}")
    print("[Model 1] Saved → rppg_to_ppg.pth")
    return model

if __name__ == "__main__":
    train_rppg_to_ppg()