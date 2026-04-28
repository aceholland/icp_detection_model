"""
Stage 2c — Full pipeline:
rPPG → Model1 → PPG → Model2 → ECG → features
"""

import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, resample

from project_paths import ARTIFACTS_DIR, MODELS_DIR

JSON_PATH  = ARTIFACTS_DIR / "stage1_output.json"
FPS        = 30
TARGET_FS  = 128
WINDOW     = 512

# ── Load models ────────────────────────────────────────
# Copy model definitions here (same classes as above)
# [paste UNet1D and PPG2ECG classes here]

def bandpass(sig, lo, hi, fs, order=4):
    nyq  = fs / 2
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig)

def run_pipeline():
    print("\n[Stage 2c] rPPG → PPG → ECG pipeline")

    # Load BVP
    with open(JSON_PATH) as f:
        data = json.load(f)

    bvp = np.array(data.get("model_bvp", data["rgb_buffers"]["G"]))
    bvp = bvp[FPS*5:]  # drop warmup

    # Resample to TARGET_FS
    n_out = int(len(bvp) / FPS * TARGET_FS)
    bvp_r = resample(bvp, n_out).astype(np.float32)
    bvp_r = bandpass(bvp_r, 0.7, 4.0, TARGET_FS)
    bvp_r = bvp_r / (np.std(bvp_r) + 1e-8)

    device = torch.device('cpu')

    # ── Model 1: rPPG → PPG ───────────────────────────
    ppg_signal = bvp_r  # fallback
    try:
        m1 = UNet1D().to(device)
        m1.load_state_dict(torch.load(MODELS_DIR / 'rppg_to_ppg.pth', map_location=device))
        m1.eval()
        with torch.no_grad():
            inp = torch.tensor(bvp_r[:WINDOW]).unsqueeze(0).unsqueeze(0)
            ppg_signal = m1(inp).squeeze().numpy()
        print("[Stage 2c] Model 1 (rPPG→PPG) applied ✓")
    except:
        print("[Stage 2c] Model 1 not found — using raw BVP as PPG")

    # ── Model 2: PPG → ECG ────────────────────────────
    ecg_signal = ppg_signal  # fallback
    try:
        m2 = PPG2ECG().to(device)
        m2.load_state_dict(torch.load('ppg_to_ecg.pth', map_location=device))
        m2.eval()

        # Sliding window inference
        segments = []
        step = WINDOW // 2
        for start in range(0, len(ppg_signal) - WINDOW, step):
            seg = ppg_signal[start:start+WINDOW]
            seg = seg / (np.max(np.abs(seg)) + 1e-8)
            inp = torch.tensor(seg).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                out = m2(inp).squeeze().numpy()
            segments.append((start, out))

        # Overlap-add reconstruction
        ecg_signal = np.zeros(len(ppg_signal))
        counts     = np.zeros(len(ppg_signal))
        for start, seg in segments:
            ecg_signal[start:start+WINDOW] += seg
            counts[start:start+WINDOW]     += 1
        counts[counts == 0] = 1
        ecg_signal /= counts
        print("[Stage 2c] Model 2 (PPG→ECG) applied ✓")
    except:
        print("[Stage 2c] Model 2 not found — train it first")

    # ── Feature extraction ────────────────────────────
    peaks, _ = find_peaks(ecg_signal,
                          distance=int(TARGET_FS*0.4),
                          prominence=0.3*np.std(ecg_signal))
    rr_ms = np.diff(peaks) / TARGET_FS * 1000
    rr_ms = rr_ms[(rr_ms > 333) & (rr_ms < 1500)]

    hr    = round(60000/np.mean(rr_ms), 1)  if len(rr_ms) >= 2 else 0
    sdnn  = round(float(np.std(rr_ms)), 2)  if len(rr_ms) >= 2 else 0
    rmssd = round(float(np.sqrt(np.mean(np.diff(rr_ms)**2))), 2) if len(rr_ms) >= 3 else 0

    print(f"\n{'='*40}")
    print(f"  HR    : {hr} BPM")
    print(f"  SDNN  : {sdnn} ms")
    print(f"  RMSSD : {rmssd} ms")
    print(f"{'='*40}")

    # Save
    data["reconstructed_ecg"] = ecg_signal.tolist()
    data["hr_ecg"]   = hr
    data["sdnn_ecg"] = sdnn
    data["rmssd_ecg"]= rmssd
    with open(JSON_PATH, "w") as f:
        json.dump(data, f)
    print("[Stage 2c] Saved → stage1_output.json")

if __name__ == "__main__":
    run_pipeline()