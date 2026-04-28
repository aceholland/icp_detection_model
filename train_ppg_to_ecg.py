"""
Model 2: PPG → ECG
Trains a 1D CNN on BIDMC dataset (real PPG + ECG pairs)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, resample
import os, glob

# ══════════════════════════════════════════════════════
# MODEL — 1D ResNet
# ══════════════════════════════════════════════════════
class ResBlock1D(nn.Module):
    def __init__(self, ch, kernel=9):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(ch, ch, kernel, padding=kernel//2),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Conv1d(ch, ch, kernel, padding=kernel//2),
            nn.BatchNorm1d(ch)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))

class PPG2ECG(nn.Module):
    """
    PPG → ECG reconstruction
    Input:  [batch, 1, 512] normalized PPG segment
    Output: [batch, 1, 512] reconstructed ECG
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64,  15, padding=7), nn.ReLU(),
            ResBlock1D(64),
            nn.Conv1d(64, 128, 9, padding=4), nn.ReLU(),
            ResBlock1D(128),
            nn.Conv1d(128, 256, 7, padding=3), nn.ReLU(),
            ResBlock1D(256),
        )
        self.decoder = nn.Sequential(
            ResBlock1D(256),
            nn.Conv1d(256, 128, 7, padding=3), nn.ReLU(),
            ResBlock1D(128),
            nn.Conv1d(128, 64,  9, padding=4), nn.ReLU(),
            ResBlock1D(64),
            nn.Conv1d(64,  1,  15, padding=7),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ══════════════════════════════════════════════════════
# DATASET — BIDMC (real) or synthetic fallback
# ══════════════════════════════════════════════════════
def bandpass(sig, lo, hi, fs, order=4):
    nyq  = fs / 2
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig)

def load_bidmc(data_dir='./bidmc_data', fs_target=128, window=512):
    """Load real BIDMC PPG+ECG pairs"""
    import wfdb
    windows = []
    records = glob.glob(os.path.join(data_dir, 'bidmc*.hea'))

    for rec_path in records[:40]:  # use 40 records for training
        rec_name = rec_path.replace('.hea', '')
        try:
            record = wfdb.rdrecord(rec_name)
            fs     = record.fs
            sigs   = record.p_signal

            # Find PPG and ECG channels
            ch_names = [c.lower() for c in record.sig_name]
            ppg_idx  = next((i for i, c in enumerate(ch_names)
                             if 'ppg' in c or 'pleth' in c), None)
            ecg_idx  = next((i for i, c in enumerate(ch_names)
                             if 'ecg' in c or 'ii' in c), None)

            if ppg_idx is None or ecg_idx is None:
                continue

            ppg = sigs[:, ppg_idx]
            ecg = sigs[:, ecg_idx]

            # Remove NaN
            mask = ~(np.isnan(ppg) | np.isnan(ecg))
            ppg  = ppg[mask]
            ecg  = ecg[mask]

            # Resample to target fs
            n_out = int(len(ppg) / fs * fs_target)
            ppg   = resample(ppg, n_out).astype(np.float32)
            ecg   = resample(ecg, n_out).astype(np.float32)

            # Filter
            ppg = bandpass(ppg, 0.5, 8.0,  fs_target).astype(np.float32)
            ecg = bandpass(ecg, 0.5, 40.0, fs_target).astype(np.float32)

            # Sliding windows
            for start in range(0, len(ppg) - window, window // 2):
                p = ppg[start:start+window]
                e = ecg[start:start+window]
                # Normalize to [-1, 1]
                p = p / (np.max(np.abs(p)) + 1e-8)
                e = e / (np.max(np.abs(e)) + 1e-8)
                windows.append((p, e))

        except Exception as ex:
            print(f"  Skipping {rec_name}: {ex}")
            continue

    print(f"[Dataset] Loaded {len(windows)} windows from BIDMC")
    return windows

def generate_synthetic_ppg_ecg(n=1000, window=512, fs=128):
    """Synthetic fallback if BIDMC not downloaded yet"""
    windows = []
    for _ in range(n):
        hr     = np.random.uniform(50, 110)
        hr_hz  = hr / 60
        t      = np.arange(window) / fs

        # PPG
        ppg    = np.sin(2*np.pi*hr_hz*t)
        ppg   += 0.4*np.sin(4*np.pi*hr_hz*t)
        ppg   += 0.1*np.sin(6*np.pi*hr_hz*t)
        ppg   += np.random.normal(0, 0.05, window)
        ppg    = ppg / (np.max(np.abs(ppg)) + 1e-8)

        # ECG — sharper QRS
        ecg    = 0.2*np.sin(2*np.pi*hr_hz*t)  # P+T background
        # Add QRS spikes at peak positions
        period = int(fs / hr_hz)
        for start in range(0, window, period):
            center = start + int(period * 0.42)
            if center < window:
                spike  = np.zeros(window)
                width  = max(int(period * 0.04), 3)
                for k in range(-width, width+1):
                    if 0 <= center+k < window:
                        spike[center+k] = np.exp(-k**2 / (2*(width/3)**2))
                ecg += spike
        ecg = ecg / (np.max(np.abs(ecg)) + 1e-8)
        windows.append((ppg.astype(np.float32),
                        ecg.astype(np.float32)))
    return windows

class PPGECGDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        p, e = self.windows[idx]
        return (
            torch.tensor(p).unsqueeze(0),
            torch.tensor(e).unsqueeze(0)
        )

# ══════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════
def train_ppg_to_ecg():
    print("\n[Model 2] Training PPG → ECG model...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Model 2] Device: {device}")

    # Try loading real BIDMC data first
    if os.path.exists('./bidmc_data'):
        print("[Model 2] Loading BIDMC dataset...")
        windows = load_bidmc('./bidmc_data')
    else:
        print("[Model 2] BIDMC not found — using synthetic data")
        print("[Model 2] Run: python -c \"import wfdb; wfdb.dl_database('bidmc', dl_dir='./bidmc_data')\"")
        windows = generate_synthetic_ppg_ecg(n=1000)

    n_train  = int(0.8 * len(windows))
    train_ds = PPGECGDataset(windows[:n_train])
    val_ds   = PPGECGDataset(windows[n_train:])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32)

    model     = PPG2ECG().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )
    # Combined loss: MSE + gradient similarity
    mse_loss  = nn.MSELoss()
    l1_loss   = nn.L1Loss()

    best_val  = float('inf')

    for epoch in range(100):
        model.train()
        train_loss = 0
        for ppg, ecg in train_loader:
            ppg, ecg = ppg.to(device), ecg.to(device)
            optimizer.zero_grad()
            pred = model(ppg)
            # MSE + L1 for sharp peaks
            loss = 0.7 * mse_loss(pred, ecg) + 0.3 * l1_loss(pred, ecg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for ppg, ecg in val_loader:
                ppg, ecg = ppg.to(device), ecg.to(device)
                pred = model(ppg)
                val_loss += mse_loss(pred, ecg).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'ppg_to_ecg.pth')

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | "
                  f"train {train_loss:.4f} | "
                  f"val {val_loss:.4f}")

    print(f"[Model 2] Done — best val loss: {best_val:.4f}")
    print("[Model 2] Saved → ppg_to_ecg.pth")

if __name__ == "__main__":
    train_ppg_to_ecg()