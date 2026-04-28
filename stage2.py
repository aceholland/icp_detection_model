import json
import os
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from project_paths import ARTIFACTS_DIR

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:
    plt = None
    gridspec = None
    FancyBboxPatch = None
    MATPLOTLIB_AVAILABLE = False

# ── Config ─────────────────────────────────────────────────────────────────
JSON_PATH     = ARTIFACTS_DIR / "stage1_output.json"
FPS           = 30
WARMUP_SEC    = 5
BP_LO, BP_HI  = 0.7, 4.0
PUPIL_SMOOTH  = 7

# ── Palette ────────────────────────────────────────────────────────────────
BG      = "#0a0c10"
SURFACE = "#111318"
BORDER  = "#1e2128"
DIM     = "#3d4451"
MUTED   = "#6b7280"
TEXT    = "#e2e8f0"
ACCENT  = "#38bdf8"
GREEN   = "#4ade80"
RED     = "#f87171"
AMBER   = "#fbbf24"
PURPLE  = "#a78bfa"

# ── Load Stage 1 output ────────────────────────────────────────────────────
print("\n[Stage 2] Loading stage1_output.json ...")
with open(JSON_PATH) as f:
    data = json.load(f)

R_raw = np.array(data["rgb_buffers"]["R"])
G_raw = np.array(data["rgb_buffers"]["G"])
B_raw = np.array(data["rgb_buffers"]["B"])
PL    = np.array(data["pupil_buffers"]["left_px"])
PR    = np.array(data["pupil_buffers"]["right_px"])

# Drop first 5s warmup
warmup = WARMUP_SEC * FPS
R  = R_raw[warmup:];  G = G_raw[warmup:];  B = B_raw[warmup:]
PL = PL[warmup:] if len(PL) > warmup else PL
PR = PR[warmup:] if len(PR) > warmup else PR

N = len(G)
t = np.arange(N) / FPS
print(f"[Stage 2] {N} frames loaded ({N/FPS:.1f}s after warmup)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — CHROM ALGORITHM
# ══════════════════════════════════════════════════════════════════════════
print("[Stage 2] Running CHROM algorithm...")

def norm(c):
    mu = np.mean(c)
    return c / mu if mu else c

Rn, Gn, Bn = norm(R), norm(G), norm(B)
Xs = 3*Rn - 2*Gn
Ys = 1.5*Rn + Gn - 1.5*Bn
alpha     = np.std(Xs) / np.std(Ys) if np.std(Ys) else 1.0
chrom_raw = Xs - alpha * Ys
print("[Stage 2] CHROM done ✓")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — BANDPASS FILTER  0.7 – 4.0 Hz
# ══════════════════════════════════════════════════════════════════════════
def bandpass(sig, lo, hi, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig)

chrom_filt = bandpass(chrom_raw, BP_LO, BP_HI, FPS)
print("[Stage 2] Bandpass filter done ✓")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — PEAK DETECTION + HRV
# ══════════════════════════════════════════════════════════════════════════
print("[Stage 2] Detecting peaks...")

peaks, _ = find_peaks(
    chrom_filt,
    distance   = int(FPS * 0.33),
    prominence = 0.4 * np.std(chrom_filt)
)

if len(peaks) >= 2:
    rr    = np.diff(peaks) / FPS * 1000   # ms
    # Keep only physiologically valid RR (400–1500ms = 40–150 BPM)
    rr    = rr[(rr > 400) & (rr < 1500)]
    if len(rr) >= 2:
        hr    = round(60000 / np.mean(rr), 1)
        sdnn  = round(float(np.std(rr)), 2)
        rmssd = round(float(np.sqrt(np.mean(np.diff(rr)**2))), 2)
        pnn50 = round(float(np.sum(np.abs(np.diff(rr)) > 50) / len(rr) * 100), 1)
    else:
        hr = sdnn = rmssd = pnn50 = 0.0
        rr = np.array([])
    print(f"[Stage 2] {len(peaks)} peaks found → HR {hr} BPM")
else:
    hr = sdnn = rmssd = pnn50 = 0.0
    rr = np.array([])
    print("[Stage 2] WARNING: Not enough peaks — sit still & re-run Stage 1 for 30s+")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — RESPIRATORY RATE
# ══════════════════════════════════════════════════════════════════════════
resp_signal   = bandpass(chrom_raw, 0.1, 0.5, FPS, order=2)
resp_peaks, _ = find_peaks(resp_signal, distance=FPS*2)
resp_rate     = round(len(resp_peaks) / (N / FPS) * 60, 1)
print(f"[Stage 2] Respiratory rate: {resp_rate} breaths/min")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — P2/P1 WAVEFORM MORPHOLOGY (ICP biomarker)
# ══════════════════════════════════════════════════════════════════════════
p2_p1_ratios = []
for i in range(len(peaks) - 1):
    segment = chrom_filt[peaks[i]:peaks[i+1]]
    if len(segment) < 4:
        continue
    half = len(segment) // 2
    P1   = segment[:half].max()
    P2   = segment[half:].max()
    if P1 > 0:
        p2_p1_ratios.append(P2 / P1)

P2_P1 = round(float(np.mean(p2_p1_ratios)), 3) if p2_p1_ratios else 0.0
print(f"[Stage 2] P2/P1 ratio: {P2_P1} {'⚠ HIGH (ICP risk)' if P2_P1 > 1.0 else '✓ normal'}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6 — PUPILLOMETRY FEATURES
# ══════════════════════════════════════════════════════════════════════════
def roll_med(a, w):
    h = w // 2
    return np.array([
        np.median(a[max(0, i-h):min(len(a), i+h+1)])
        for i in range(len(a))
    ])

tp   = np.arange(len(PL)) / FPS
PL_s = roll_med(PL, PUPIL_SMOOTH) if len(PL) else PL
PR_s = roll_med(PR, PUPIL_SMOOTH) if len(PR) else PR

# Scale raw pixels to mm (approx factor 0.25 for 640x480 frame)
PUPIL_SCALE = 0.25 
pupil_L_mean = round(float(np.mean(PL_s)) * PUPIL_SCALE, 2) if len(PL_s) else 4.0
pupil_R_mean = round(float(np.mean(PR_s)) * PUPIL_SCALE, 2) if len(PR_s) else 4.0
asym         = round(float(np.mean(np.abs(PL_s - PR_s))) * PUPIL_SCALE, 2) if len(PL_s) else 0.0
NPI_proxy    = 1.5 if asym > 1.5 else 2.5 if asym > 0.8 else 4.0

print(f"[Stage 2] Pupil asymmetry: {asym} px — NPI proxy: {NPI_proxy}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7 — FEATURE VECTOR → save for Stage 3
# ══════════════════════════════════════════════════════════════════════════
feature_vector = {
    "HR":           hr,
    "SDNN":         sdnn,
    "RMSSD":        rmssd,
    "RespRate":     resp_rate,
    "P2_P1_ratio":  P2_P1,
    "pupil_L_px":   pupil_L_mean,
    "pupil_R_px":   pupil_R_mean,
    "asymmetry_px": asym,
    "NPI_proxy":    NPI_proxy
}

with open(ARTIFACTS_DIR / "stage2_output.json", "w") as f:
    json.dump(feature_vector, f, indent=2)

print("\n" + "="*45)
print("  STAGE 2 — FEATURE VECTOR")
print("="*45)
for k, v in feature_vector.items():
    print(f"  {k:<20}: {v}")
print("="*45)
print("\n[Stage 2] Saved → stage2_output.json")
print("[Stage 2] Ready for Stage 3\n")

if os.environ.get("HEADLESS") == "1" or not MATPLOTLIB_AVAILABLE:
    print("[Stage 2] Headless mode or Matplotlib unavailable; skipping plots.")
    raise SystemExit(0)

# ══════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════
def style_ax(ax, xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=0)
    ax.grid(axis='y', color=BORDER, lw=0.6, zorder=0)
    ax.grid(axis='x', visible=False)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=8, labelpad=5)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=8, labelpad=5)

def set_title(ax, txt):
    ax.set_title(txt, color=TEXT, fontsize=8,
                 fontweight='semibold', pad=7, loc='left')

fig = plt.figure(figsize=(18, 10), facecolor=BG)
gs  = gridspec.GridSpec(
    3, 2, figure=fig,
    width_ratios=[1.55, 1],
    hspace=0.55, wspace=0.35,
    left=0.055, right=0.975, top=0.90, bottom=0.07
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[0, 1])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[2, 1])

# Raw RGB
style_ax(ax1, ylabel="Pixel mean")
set_title(ax1, "Raw RGB  ·  forehead ROI")
ax1.plot(t, R, color="#ef4444", lw=0.7, alpha=0.85, label="R")
ax1.plot(t, G, color="#22c55e", lw=0.7, alpha=0.85, label="G")
ax1.plot(t, B, color="#3b82f6", lw=0.7, alpha=0.85, label="B")
ax1.legend(loc="upper right", fontsize=7, framealpha=0,
           labelcolor=MUTED, handlelength=1.2)

# CHROM raw
style_ax(ax2, ylabel="Amplitude")
set_title(ax2, "CHROM raw pulse  ·  skin-tone cancelled")
ax2.plot(t, chrom_raw, color=AMBER, lw=0.75, alpha=0.9)
ax2.axhline(0, color=DIM, lw=0.5, ls='--')

# Filtered + peaks
style_ax(ax3, xlabel="Time (s)", ylabel="Amplitude")
set_title(ax3, f"Filtered 0.7–4Hz  ·  HR {hr} BPM  ·  SDNN {sdnn} ms  ·  RMSSD {rmssd} ms")
ax3.plot(t, chrom_filt, color=GREEN, lw=0.9)
ax3.axhline(0, color=DIM, lw=0.5, ls='--')
if len(peaks):
    ax3.scatter(peaks/FPS, chrom_filt[peaks],
                color=RED, s=22, zorder=5,
                label=f"{len(peaks)} peaks", marker='o', linewidths=0)
    ax3.legend(loc="upper right", fontsize=7, framealpha=0,
               labelcolor=MUTED, handlelength=1.2)

# Metric cards
ax4.set_facecolor(SURFACE)
for spine in ax4.spines.values(): spine.set_visible(False)
ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
ax4.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
set_title(ax4, "Vitals summary")

cards = [
    ("Heart Rate", f"{hr:.1f}",    "BPM", 60 <= hr <= 100),
    ("SDNN",       f"{sdnn:.0f}",  "ms",  20 <= sdnn <= 100),
    ("RMSSD",      f"{rmssd:.0f}", "ms",  20 <= rmssd <= 50),
    ("pNN50",      f"{pnn50:.1f}", "%",   3 <= pnn50 <= 20),
]
cw, ch = 0.44, 0.36
ox, oy = 0.04, 0.10
for i, (label, val, unit, ok) in enumerate(cards):
    cx = ox + (i % 2) * (cw + 0.08)
    cy = 1.0 - oy - (i // 2) * (ch + 0.14) - ch
    color = GREEN if ok else RED
    rect  = FancyBboxPatch(
        (cx, cy), cw, ch,
        boxstyle="round,pad=0.02",
        facecolor=BORDER, edgecolor=color,
        linewidth=1.0, transform=ax4.transAxes, zorder=2
    )
    ax4.add_patch(rect)
    ax4.text(cx+cw/2, cy+ch*0.72, label,
             transform=ax4.transAxes, ha='center', va='center',
             fontsize=7.5, color=MUTED)
    ax4.text(cx+cw/2, cy+ch*0.38, val,
             transform=ax4.transAxes, ha='center', va='center',
             fontsize=17, color=color, fontweight='bold')
    ax4.text(cx+cw/2, cy+ch*0.10, unit,
             transform=ax4.transAxes, ha='center', va='center',
             fontsize=7, color=DIM)

# Pupil diameters
style_ax(ax5, xlabel="Time (s)", ylabel="Diameter (px)")
set_title(ax5, f"Pupil diameters  ·  asymmetry {asym} px")
if len(PL_s):
    ax5.plot(tp, PL_s, color=ACCENT,  lw=0.9, label="Left")
    ax5.plot(tp, PR_s, color=PURPLE,  lw=0.9, label="Right")
    ax5.axhline(pupil_L_mean, color=ACCENT,  ls='--', lw=0.5, alpha=0.45)
    ax5.axhline(pupil_R_mean, color=PURPLE,  ls='--', lw=0.5, alpha=0.45)
    ax5.legend(loc="upper right", fontsize=7, framealpha=0,
               labelcolor=MUTED, handlelength=1.2)

# Poincare plot
style_ax(ax6, xlabel="RR[n] (ms)", ylabel="RR[n+1] (ms)")
set_title(ax6, "Poincaré  ·  RR[n] vs RR[n+1]")
if len(rr) > 1:
    rr1, rr2 = rr[:-1], rr[1:]
    ax6.scatter(rr1, rr2, color=PURPLE, s=14, alpha=0.75, linewidths=0)
    lim = [min(rr1.min(), rr2.min())-30, max(rr1.max(), rr2.max())+30]
    ax6.plot(lim, lim, color=DIM, lw=0.7, ls='--', zorder=0)

# Header
fig.text(0.055, 0.962,
         "Stage 2  —  rPPG Signal Processing + Feature Extraction",
         color=ACCENT, fontsize=12, fontweight='bold', va='top')
fig.text(0.975, 0.962,
         f"{N/FPS:.0f}s  ·  {N} frames  ·  30 fps",
         color=MUTED, fontsize=8, va='top', ha='right')

plt.savefig("stage2_output.png", dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.show()
print("[Stage 2] Plot saved → stage2_output.png")