import json
import os
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.interpolate import interp1d

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
JSON_PATH    = ARTIFACTS_DIR / "stage1_output.json"
FPS          = 30
WARMUP_SEC   = 5
BP_PULSE_LO, BP_PULSE_HI = 0.7, 4.0
BP_P2P1_LO,  BP_P2P1_HI  = 0.5, 10.0
BP_RESP_LO,  BP_RESP_HI   = 0.1, 0.5
MIN_DIST_S   = 0.33
PROM_FACTOR  = 0.4
LF_LO, LF_HI = 0.04, 0.15
HF_LO, HF_HI = 0.15, 0.40

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

# ── Load ───────────────────────────────────────────────────────────────────
with open(JSON_PATH) as f:
    data = json.load(f)

warmup = WARMUP_SEC * FPS
R = np.array(data["rgb_buffers"]["R"])[warmup:]
G = np.array(data["rgb_buffers"]["G"])[warmup:]
B = np.array(data["rgb_buffers"]["B"])[warmup:]
N = len(G)
t = np.arange(N) / FPS
print(f"[Stage 3] {N} frames  ({N/FPS:.1f}s) after warmup strip")

# ── CHROM ──────────────────────────────────────────────────────────────────
def norm(c): mu = np.mean(c); return c / mu if mu else c

Rn, Gn, Bn = norm(R), norm(G), norm(B)
Xs = 3*Rn - 2*Gn
Ys = 1.5*Rn + Gn - 1.5*Bn
alpha     = np.std(Xs) / np.std(Ys) if np.std(Ys) else 1.0
chrom_raw = Xs - alpha * Ys

# ── Ensemble with model BVP if available ───────────────────────────────────
ensemble_label = "CHROM only"
if "model_bvp" in data:
    mb = np.array(data["model_bvp"])[warmup:]
    if len(mb) == N:
        chrom_norm = (chrom_raw - np.mean(chrom_raw)) / (np.std(chrom_raw) + 1e-8)
        mb_norm    = (mb - np.mean(mb)) / (np.std(mb) + 1e-8)
        chrom_raw  = (chrom_norm + mb_norm) / 2.0
        ensemble_label = "CHROM + EfficientPhys average"
        print(f"[Stage 3] Ensembled: {ensemble_label}")
    else:
        print(f"[Stage 3] model_bvp length mismatch ({len(mb)} vs {N}), using CHROM only")
else:
    print("[Stage 3] No model_bvp — using CHROM only (run stage2_model.py to enable ensemble)")

# ── Filters ────────────────────────────────────────────────────────────────
def bandpass(sig, lo, hi, fs=FPS, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return filtfilt(b, a, sig)

pulse_filt = bandpass(chrom_raw, BP_PULSE_LO, BP_PULSE_HI)
p2p1_filt  = bandpass(chrom_raw, BP_P2P1_LO,  BP_P2P1_HI)
resp_filt  = bandpass(chrom_raw, BP_RESP_LO,  BP_RESP_HI)

# ── Peaks & HRV ────────────────────────────────────────────────────────────
min_dist = int(FPS * MIN_DIST_S)
prom     = PROM_FACTOR * np.std(pulse_filt)
peaks, _ = find_peaks(pulse_filt, distance=min_dist, prominence=prom)

# ── RR outlier rejection ───────────────────────────────────────────────────
# physiologically valid RR: 333ms (180 BPM) to 1500ms (40 BPM)
# reject peaks that produce RR intervals outside this range
def clean_peaks(peaks, fps, rr_min_ms=333, rr_max_ms=1500):
    if len(peaks) < 2:
        return peaks
    keep = [peaks[0]]
    for i in range(1, len(peaks)):
        rr = (peaks[i] - keep[-1]) / fps * 1000
        if rr_min_ms <= rr <= rr_max_ms:
            keep.append(peaks[i])
        # else: skip this peak — it's a false detection
    return np.array(keep)

peaks = clean_peaks(peaks, FPS)

rr_ms = np.array([])
hr = sdnn = rmssd = pnn50 = 0.0

if len(peaks) >= 2:
    rr_ms  = np.diff(peaks) / FPS * 1000
    hr     = 60 / np.mean(np.diff(peaks) / FPS)
    sdnn   = np.std(rr_ms)
    rmssd  = np.sqrt(np.mean(np.diff(rr_ms)**2)) if len(rr_ms) >= 2 else 0.0
    pnn50  = (np.sum(np.abs(np.diff(rr_ms)) > 50) / len(rr_ms) * 100
              if len(rr_ms) >= 2 else 0.0)

print(f"  HR     : {hr:.1f} BPM")
print(f"  SDNN   : {sdnn:.1f} ms")
print(f"  RMSSD  : {rmssd:.1f} ms")
print(f"  pNN50  : {pnn50:.1f}%")

# ── LF/HF ─────────────────────────────────────────────────────────────────
lf_hf = np.nan
freqs = psd = None

if len(rr_ms) >= 8:
    rr_t      = (peaks[:-1] + peaks[1:]) / 2 / FPS
    interp_fs = 4.0
    t_uniform = np.arange(rr_t[0], rr_t[-1], 1/interp_fs)
    if len(t_uniform) > 8:
        rr_interp = interp1d(rr_t, rr_ms/1000, kind='cubic')(t_uniform)
        freqs, psd = welch(rr_interp, fs=interp_fs,
                           nperseg=min(len(rr_interp), 64))
        lf = np.trapezoid(psd[(freqs>=LF_LO)&(freqs<=LF_HI)],
                      freqs[(freqs>=LF_LO)&(freqs<=LF_HI)])
        hf = np.trapezoid(psd[(freqs>=HF_LO)&(freqs<=HF_HI)],
                      freqs[(freqs>=HF_LO)&(freqs<=HF_HI)])
        lf_hf = lf / hf if hf > 0 else np.nan

print(f"  LF/HF  : {lf_hf:.2f}" if not np.isnan(lf_hf) else "  LF/HF  : insufficient data")

# ── Respiratory rate ───────────────────────────────────────────────────────
resp_rate = 0.0
freqs_resp, psd_resp = welch(resp_filt, fs=FPS, nperseg=min(N, FPS*10))
resp_mask = (freqs_resp >= BP_RESP_LO) & (freqs_resp <= BP_RESP_HI)
if resp_mask.any():
    resp_rate = freqs_resp[resp_mask][np.argmax(psd_resp[resp_mask])] * 60

print(f"  Resp   : {resp_rate:.1f} breaths/min")

# ── P2/P1 ─────────────────────────────────────────────────────────────────
p2p1_ratios = []
for i in range(len(peaks) - 1):
    seg = p2p1_filt[peaks[i]:peaks[i+1]]
    if len(seg) < 6: continue
    P1 = seg[0]
    mid = len(seg) // 2
    notch_r = seg[mid:]
    if len(notch_r) < 3: continue
    notch_idx = mid + np.argmin(notch_r)
    post = seg[notch_idx:]
    if len(post) < 2: continue
    p2p, _ = find_peaks(post, prominence=0.001)
    P2 = post[p2p[0]] if len(p2p) else np.max(post)
    if P1 != 0:
        r = P2 / P1
        if 0.1 < r < 2.5:
            p2p1_ratios.append(r)

p2p1_mean = np.mean(p2p1_ratios) if p2p1_ratios else np.nan
p2p1_flag = "HIGH ICP RISK ⚠" if (not np.isnan(p2p1_mean) and p2p1_mean >= 1.0) else "NORMAL"
print(f"  P2/P1  : {p2p1_mean:.3f}  →  {p2p1_flag}" if not np.isnan(p2p1_mean)
      else "  P2/P1  : insufficient beats")

# ── Save JSON ──────────────────────────────────────────────────────────────
results = {
    "HR_bpm":       round(hr, 1),
    "SDNN_ms":      round(sdnn, 1),
    "RMSSD_ms":     round(rmssd, 1),
    "pNN50_pct":    round(pnn50, 1),
    "LF_HF":        round(lf_hf, 3) if not np.isnan(lf_hf) else None,
    "resp_bpm":     round(resp_rate, 1),
    "P2P1_mean":    round(p2p1_mean, 3) if not np.isnan(p2p1_mean) else None,
    "P2P1_flag":    p2p1_flag,
    "ensemble":     ensemble_label,
    "n_beats":      len(peaks),
}
with open(ARTIFACTS_DIR / "stage3_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[Stage 3] Saved → stage3_results.json  [{ensemble_label}]")

if os.environ.get("HEADLESS") == "1" or not MATPLOTLIB_AVAILABLE:
    print("[Stage 3] Headless mode or Matplotlib unavailable; skipping plots.")
    raise SystemExit(0)

# ══════════════════════════════════════════════════════════════════════════
#  PLOT
# ══════════════════════════════════════════════════════════════════════════
def style_ax(ax, xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=0)
    ax.grid(axis='y', color=BORDER, lw=0.6, zorder=0)
    ax.grid(axis='x', visible=False)
    if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=8, labelpad=5)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=8, labelpad=5)

def ttl(ax, txt):
    ax.set_title(txt, color=TEXT, fontsize=8.5,
                 fontweight='semibold', pad=7, loc='left')

plt.style.use('default')
fig = plt.figure(figsize=(18, 11), facecolor=BG)
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.55, wspace=0.32,
                        left=0.055, right=0.975,
                        top=0.91, bottom=0.07)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[1, 2])
ax5 = fig.add_subplot(gs[2, 0])
ax6 = fig.add_subplot(gs[2, 1])
ax7 = fig.add_subplot(gs[2, 2])

# ax1
style_ax(ax1, xlabel="Time (s)", ylabel="Amplitude")
ttl(ax1, f"Filtered pulse  ·  {ensemble_label}  ·  HR {hr:.1f} BPM  ·  SDNN {sdnn:.0f} ms  ·  RMSSD {rmssd:.0f} ms  ·  Resp {resp_rate:.1f} br/min")
ax1.plot(t, pulse_filt, color=GREEN, lw=0.9)
ax1.axhline(0, color=DIM, lw=0.5, ls='--')
if len(peaks):
    ax1.scatter(peaks/FPS, pulse_filt[peaks], color=RED, s=24,
                zorder=5, linewidths=0, label=f"{len(peaks)} beats")
    ax1.legend(loc="upper right", fontsize=7, framealpha=0, labelcolor=MUTED)

# ax2
style_ax(ax2, xlabel="Beat index", ylabel="P2/P1 ratio")
p2p1_color = RED if (not np.isnan(p2p1_mean) and p2p1_mean >= 1.0) else GREEN
ttl(ax2, f"P2/P1  ·  mean {p2p1_mean:.3f}  ·  {p2p1_flag}" if not np.isnan(p2p1_mean)
    else "P2/P1  ·  insufficient data")
if p2p1_ratios:
    ax2.bar(range(len(p2p1_ratios)), p2p1_ratios,
            color=[RED if r >= 1.0 else GREEN for r in p2p1_ratios],
            width=0.7, alpha=0.85)
    ax2.axhline(1.0, color=AMBER, lw=1.0, ls='--', label="threshold")
    ax2.axhline(p2p1_mean, color=p2p1_color, lw=0.8, ls=':', label=f"mean {p2p1_mean:.3f}")
    ax2.legend(loc="upper right", fontsize=7, framealpha=0, labelcolor=MUTED)

# ax3
style_ax(ax3, xlabel="Frequency (Hz)", ylabel="PSD (s²/Hz)")
lf_hf_str = f"{lf_hf:.2f}" if not np.isnan(lf_hf) else "n/a"
ttl(ax3, f"HRV Power Spectrum  ·  LF/HF {lf_hf_str}")
if freqs is not None and not np.isnan(lf_hf):
    ax3.fill_between(freqs[(freqs>=LF_LO)&(freqs<=LF_HI)],
                     psd[(freqs>=LF_LO)&(freqs<=LF_HI)], color=AMBER, alpha=0.6, label="LF")
    ax3.fill_between(freqs[(freqs>=HF_LO)&(freqs<=HF_HI)],
                     psd[(freqs>=HF_LO)&(freqs<=HF_HI)], color=ACCENT, alpha=0.6, label="HF")
    ax3.plot(freqs, psd, color=MUTED, lw=0.7)
    ax3.set_xlim(0, 0.5)
    ax3.legend(loc="upper right", fontsize=7, framealpha=0, labelcolor=MUTED)
else:
    ax3.text(0.5, 0.5, "Need ≥8 RR intervals\nfor LF/HF",
             transform=ax3.transAxes, ha='center', va='center', color=MUTED, fontsize=8)

# ax4
style_ax(ax4, xlabel="Time (s)", ylabel="Amplitude")
ttl(ax4, f"Respiratory  ·  0.1–0.5 Hz  ·  {resp_rate:.1f} br/min")
ax4.plot(t, resp_filt, color=PURPLE, lw=0.8)
ax4.axhline(0, color=DIM, lw=0.5, ls='--')

# ax5
style_ax(ax5, xlabel="RR[n] (ms)", ylabel="RR[n+1] (ms)")
ttl(ax5, "Poincaré  ·  RR[n] vs RR[n+1]")
if len(rr_ms) > 1:
    ax5.scatter(rr_ms[:-1], rr_ms[1:], color=PURPLE, s=16, alpha=0.75, linewidths=0)
    lim = [rr_ms.min()-30, rr_ms.max()+30]
    ax5.plot(lim, lim, color=DIM, lw=0.7, ls='--')

# ax6
style_ax(ax6, xlabel="Beat index", ylabel="RR interval (ms)")
ttl(ax6, "RR Tachogram")
if len(rr_ms):
    ax6.plot(range(len(rr_ms)), rr_ms, color=AMBER, lw=0.9,
             marker='o', markersize=3, markerfacecolor=AMBER)
    ax6.axhline(np.mean(rr_ms), color=DIM, lw=0.6, ls='--')

# ax7 — summary cards
ax7.set_facecolor(SURFACE)
for s in ax7.spines.values(): s.set_visible(False)
ax7.set_xlim(0,1); ax7.set_ylim(0,1)
ax7.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ttl(ax7, "Summary")

cards = [
    ("HR",    f"{hr:.0f}",   "BPM",    60<=hr<=100),
    ("SDNN",  f"{sdnn:.0f}", "ms",     20<=sdnn<=100),
    ("LF/HF", f"{lf_hf:.2f}" if not np.isnan(lf_hf) else "–", "",
              0.5<=lf_hf<=2.0 if not np.isnan(lf_hf) else True),
    ("Resp",  f"{resp_rate:.0f}", "br/min", 12<=resp_rate<=20),
    ("P2/P1", f"{p2p1_mean:.3f}" if not np.isnan(p2p1_mean) else "–", "",
              np.isnan(p2p1_mean) or p2p1_mean < 1.0),
    ("RMSSD", f"{rmssd:.0f}", "ms",    20<=rmssd<=50),
]

cw, ch = 0.27, 0.27
for i, (label, val, unit, ok) in enumerate(cards):
    cx = 0.03 + (i % 3) * (cw + 0.04)
    cy = 0.55 - (i // 3) * (ch + 0.10)
    color = GREEN if ok else RED
    ax7.add_patch(FancyBboxPatch((cx, cy), cw, ch,
                                  boxstyle="round,pad=0.02",
                                  facecolor=BORDER, edgecolor=color,
                                  linewidth=1.0, transform=ax7.transAxes, zorder=2))
    ax7.text(cx+cw/2, cy+ch*0.75, label, transform=ax7.transAxes,
             ha='center', va='center', fontsize=7, color=MUTED)
    ax7.text(cx+cw/2, cy+ch*0.40, val, transform=ax7.transAxes,
             ha='center', va='center', fontsize=13, color=color, fontweight='bold')
    if unit:
        ax7.text(cx+cw/2, cy+ch*0.10, unit, transform=ax7.transAxes,
                 ha='center', va='center', fontsize=6.5, color=DIM)

icp_color = RED if p2p1_flag != "NORMAL" else GREEN
ax7.text(0.5, 0.08, p2p1_flag, transform=ax7.transAxes,
         ha='center', va='center', fontsize=9, color=icp_color, fontweight='bold')

# header
fig.text(0.055, 0.962,
         f"Stage 3  —  rPPG Signal Processing  ·  {ensemble_label}",
         color=ACCENT, fontsize=12, fontweight='bold', va='top')
fig.text(0.975, 0.962, f"{N/FPS:.0f}s  ·  {len(peaks)} beats  ·  30 fps",
         color=MUTED, fontsize=8, va='top', ha='right')

plt.savefig("stage3_output.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("[Stage 3] Saved → stage3_output.png")