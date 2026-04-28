import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import json

from project_paths import DATA_DIR

np.random.seed(42)
N = 5000  # synthetic patients

def simulate_features(icp_mmhg):
    """
    Simulate realistic feature values based on known ICP physiology.
    Higher ICP → lower HR, lower HRV, higher P2/P1, abnormal pupils
    """
    # Cushing response: high ICP → bradycardia
    HR = np.random.normal(75 - (icp_mmhg * 0.8), 5)
    HR = np.clip(HR, 40, 120)

    # High ICP → low HRV
    SDNN = np.random.normal(50 - (icp_mmhg * 1.2), 8)
    SDNN = np.clip(SDNN, 5, 120)

    RMSSD = np.random.normal(42 - (icp_mmhg * 1.0), 7)
    RMSSD = np.clip(RMSSD, 5, 100)

    # High ICP → high P2/P1
    P2_P1 = np.random.normal(0.6 + (icp_mmhg * 0.02), 0.05)
    P2_P1 = np.clip(P2_P1, 0.3, 1.5)

    # High ICP → abnormal resp
    RespRate = np.random.normal(16 + (icp_mmhg * 0.1), 2)
    RespRate = np.clip(RespRate, 8, 35)

    # High ICP → pupil dilation + asymmetry
    pupil_L = np.random.normal(3.5 + (icp_mmhg * 0.05), 0.3)
    pupil_R = pupil_L + np.random.normal(icp_mmhg * 0.02, 0.2)
    asymmetry = abs(pupil_L - pupil_R)

    # NPI drops with ICP
    NPI = np.random.normal(4.5 - (icp_mmhg * 0.08), 0.3)
    NPI = np.clip(NPI, 0, 5)

    return {
        "HR":           round(HR, 2),
        "SDNN":         round(SDNN, 2),
        "RMSSD":        round(RMSSD, 2),
        "RespRate":     round(RespRate, 2),
        "P2_P1_ratio":  round(P2_P1, 3),
        "pupil_L_px":   round(pupil_L, 2),
        "pupil_R_px":   round(pupil_R, 2),
        "asymmetry_px": round(asymmetry, 2),
        "NPI_proxy":    round(NPI, 2),
        "ICP_mmhg":     round(icp_mmhg, 1)  # ← target label
    }

# Generate samples across full ICP range
# Normal ICP: 5–15 mmHg
# Elevated:   15–20 mmHg
# Critical:   20–40 mmHg
icp_values = np.concatenate([
    np.random.uniform(5,  15, 2500),   # normal (50%)
    np.random.uniform(15, 20, 1250),   # elevated (25%)
    np.random.uniform(20, 40, 1250),   # critical (25%)
])
np.random.shuffle(icp_values)

rows = [simulate_features(icp) for icp in icp_values]
df   = pd.DataFrame(rows)

df.to_csv(DATA_DIR / "training_data.csv", index=False)
print(f"[Data] Generated {len(df)} synthetic samples")
print(df.head())
print(f"\nICP distribution:")
print(f"  Normal   (<15):  {(df.ICP_mmhg < 15).sum()} samples")
print(f"  Elevated (15-20):{((df.ICP_mmhg>=15)&(df.ICP_mmhg<20)).sum()} samples")
print(f"  Critical (>20):  {(df.ICP_mmhg >= 20).sum()} samples")