import cv2
import numpy as np
import time
import json
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

from project_paths import ARTIFACTS_DIR, MODELS_DIR

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH   = MODELS_DIR / "face_landmarker.task"
FPS          = 30
BASELINE_SEC = 2.0      # pre-flash recording
FLASH_SEC    = 0.40     # 400ms white flash (long enough to reliably trigger PLR)
RECORD_SEC   = 3.0      # post-flash recording
PUPIL_SMOOTH = 3        # light smoothing during capture
WINDOW       = "Stage 2b  —  PLR Test"

# ── NPI thresholds (from published normative data) ─────────────────────────
LAT_NORMAL   = 300      # ms  — latency cutoff
PCT_FULL     = 40.0     # %   — full credit constriction
CV_FULL      = 80.0     # px/s — full credit constriction velocity
DV_FULL      = 40.0     # px/s — full credit dilation velocity
MIN_BASELINE = 10.0     # px  — minimum valid baseline diameter

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

# ── MediaPipe setup ────────────────────────────────────────────────────────
import urllib.request, os
if not os.path.exists(MODEL_PATH):
    print("[PLR] Downloading MediaPipe model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )

BaseOptions       = mp.tasks.BaseOptions
FaceLandmarker    = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOpt = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# ── Helpers ────────────────────────────────────────────────────────────────
def lm_px(lm, idx, W, H):
    return (int(lm[idx].x * W), int(lm[idx].y * H))

def iris_diameter(lm, center_idx, edge_idx, W, H):
    c = lm_px(lm, center_idx, W, H)
    e = lm_px(lm, edge_idx,   W, H)
    return 2 * np.sqrt((c[0]-e[0])**2 + (c[1]-e[1])**2), c

def roll_med(a, w):
    h = w // 2
    return np.array([np.median(a[max(0,i-h):min(len(a),i+h+1)])
                     for i in range(len(a))])

def draw_overlay(frame, phase, elapsed, dL, dR):
    H, W = frame.shape[:2]

    # ── dark panel at top ─────────────────────────────────────────────────
    panel_h = 64
    panel = np.zeros((panel_h, W, 3), dtype=np.uint8)
    panel[:] = (14, 17, 23)
    frame[0:panel_h] = cv2.addWeighted(frame[0:panel_h], 0.15, panel, 0.85, 0)

    # ── phase config ──────────────────────────────────────────────────────
    phase_cfg = {
        "WAITING":   {"label": "WAITING FOR START",  "color": (120, 120, 120), "icon": "·"},
        "BASELINE":  {"label": "BASELINE",            "color": ( 74, 222, 128), "icon": "●"},
        "FLASH":     {"label": "FLASH",               "color": (255, 255, 255), "icon": "◉"},
        "RECORDING": {"label": "RECORDING",           "color": ( 56, 189, 248), "icon": "●"},
        "DONE":      {"label": "COMPLETE",            "color": (167, 139, 250), "icon": "✓"},
    }
    cfg   = phase_cfg.get(phase, phase_cfg["WAITING"])
    color = cfg["color"]

    # ── title ─────────────────────────────────────────────────────────────
    cv2.putText(frame, "PLR TEST", (16, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 1, cv2.LINE_AA)

    # ── phase pill ────────────────────────────────────────────────────────
    label = cfg["label"]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    px, py = W//2 - tw//2 - 10, 14
    cv2.rectangle(frame, (px-4, py-2), (px+tw+14, py+th+4),
                  tuple(max(0,c//4) for c in color), -1)
    cv2.rectangle(frame, (px-4, py-2), (px+tw+14, py+th+4), color, 1)
    cv2.putText(frame, label, (px+4, py+th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # ── elapsed timer ─────────────────────────────────────────────────────
    timer_str = f"{elapsed:.1f}s"
    cv2.putText(frame, timer_str, (W-70, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (107, 114, 128), 1, cv2.LINE_AA)

    # ── pupil diameter readouts ───────────────────────────────────────────
    if dL and dR:
        # left eye
        cv2.putText(frame, "L", (16, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (107, 114, 128), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{dL:.1f}px", (28, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 80), 1, cv2.LINE_AA)
        # right eye
        cv2.putText(frame, "R", (100, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (107, 114, 128), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{dR:.1f}px", (112, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 220, 130), 1, cv2.LINE_AA)

    # ── bottom panel ──────────────────────────────────────────────────────
    bot_h = 48
    bot_panel = np.zeros((bot_h, W, 3), dtype=np.uint8)
    bot_panel[:] = (14, 17, 23)
    frame[H-bot_h:H] = cv2.addWeighted(
        frame[H-bot_h:H], 0.15, bot_panel, 0.85, 0)

    # ── progress bar ──────────────────────────────────────────────────────
    total_dur = BASELINE_SEC + FLASH_SEC + RECORD_SEC
    if elapsed > 0:
        prog      = min(elapsed / total_dur, 1.0)
        bar_x1    = 16
        bar_x2    = W - 16
        bar_y     = H - 28
        bar_h_px  = 6
        bar_w     = bar_x2 - bar_x1

        # track
        cv2.rectangle(frame, (bar_x1, bar_y),
                      (bar_x2, bar_y + bar_h_px), (40, 44, 52), -1)

        # baseline segment (green)
        b_end = int(bar_x1 + bar_w * (BASELINE_SEC / total_dur))
        if phase in ("BASELINE", "RECORDING", "DONE"):
            fill_end = int(bar_x1 + bar_w * min(elapsed / total_dur,
                           BASELINE_SEC / total_dur))
            cv2.rectangle(frame, (bar_x1, bar_y),
                          (fill_end, bar_y + bar_h_px), (74, 222, 128), -1)

        # flash marker
        cv2.rectangle(frame, (b_end-1, bar_y-2),
                      (b_end+1, bar_y + bar_h_px + 2), (251, 191, 36), -1)

        # recording segment (blue)
        if phase in ("RECORDING", "DONE"):
            rec_start = b_end + int(bar_w * (FLASH_SEC / total_dur))
            rec_fill  = int(bar_x1 + bar_w * min(elapsed / total_dur, 1.0))
            cv2.rectangle(frame, (rec_start, bar_y),
                          (rec_fill, bar_y + bar_h_px), (56, 189, 248), -1)

        # border
        cv2.rectangle(frame, (bar_x1, bar_y),
                      (bar_x2, bar_y + bar_h_px), (60, 65, 75), 1)

    # ── hint text ─────────────────────────────────────────────────────────
    hint = "SPACE to start" if phase == "WAITING" else (
           "Look at the screen" if phase in ("BASELINE", "RECORDING") else
           "Analysing..." if phase == "DONE" else "")
    if hint:
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(frame, hint, (W//2 - hw//2, H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (107, 114, 128), 1, cv2.LINE_AA)

    return frame


# ── NPI computation ────────────────────────────────────────────────────────
def compute_npi(baseline, minimum, latency_ms, cv, dv):
    if baseline < MIN_BASELINE:
        return 0.0, {}

    pct  = (baseline - minimum) / baseline * 100

    lat_s  = 1.0 if latency_ms < LAT_NORMAL  else max(0, 1-(latency_ms-LAT_NORMAL)/300)
    pct_s  = min(pct  / PCT_FULL, 1.0)
    cv_s   = min(cv   / CV_FULL,  1.0)
    dv_s   = min(dv   / DV_FULL,  1.0)
    size_s = min(baseline / MIN_BASELINE, 1.0)

    npi = lat_s + pct_s + cv_s + dv_s + size_s   # 0–5

    components = {
        "latency":     lat_s,
        "pct_constr":  pct_s,
        "cv":          cv_s,
        "dv":          dv_s,
        "baseline_sz": size_s,
    }
    return round(npi, 2), components

# ── Main capture loop ──────────────────────────────────────────────────────
def run_plr():
    options = FaceLandmarkerOpt(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          FPS)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    timestamps  = []
    diameters_L = []
    diameters_R = []
    flash_time  = None

    phase      = "WAITING"
    session_start = None

    total_dur = BASELINE_SEC + FLASH_SEC + RECORD_SEC

    print("\n[PLR] Press SPACE to start the test. Press Q to quit.\n")
    print(f"  Baseline : {BASELINE_SEC}s")
    print(f"  Flash    : {FLASH_SEC*1000:.0f}ms  (screen turns white)")
    print(f"  Record   : {RECORD_SEC}s post-flash\n")

    with FaceLandmarker.create_from_options(options) as lm_model:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            H, W  = frame.shape[:2]
            now   = time.time()
            elapsed = (now - session_start) if session_start else 0.0

            # ── Phase transitions ──────────────────────────────────────────
            if phase == "WAITING":
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    phase         = "BASELINE"
                    session_start = time.time()
                    print("[PLR] Session started — baseline recording...")
                elif key == ord('q'):
                    break

            elif phase == "BASELINE" and elapsed >= BASELINE_SEC:
                phase      = "FLASH"
                flash_time = time.time()
                print("[PLR] FLASH!")

            elif phase == "FLASH" and (time.time() - flash_time) >= FLASH_SEC:
                phase = "RECORDING"
                print("[PLR] Recording post-flash response...")

            elif phase == "RECORDING" and elapsed >= (BASELINE_SEC + FLASH_SEC + RECORD_SEC):
                phase = "DONE"
                print("[PLR] Session complete.")

            # ── Flash frame: fill window white ────────────────────────────
            if phase == "FLASH":
                white = np.full((H, W, 3), 255, dtype=np.uint8)
                cv2.imshow(WINDOW, white)
                cv2.waitKey(1)
                continue

            # ── Landmark detection ─────────────────────────────────────────
            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            result = lm_model.detect(mp_img)
            dL_val = dR_val = None

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                if len(lm) > 477:
                    dL_val, lc = iris_diameter(lm, 468, 469, W, H)
                    dR_val, rc = iris_diameter(lm, 473, 474, W, H)

                    cv2.circle(frame, lc, int(dL_val/2), (255, 200,  0), 2)
                    cv2.circle(frame, rc, int(dR_val/2), (100, 255, 100), 2)

                    if phase in ("BASELINE", "RECORDING"):
                        timestamps.append(elapsed)
                        diameters_L.append(dL_val)
                        diameters_R.append(dR_val)

            frame = draw_overlay(frame, phase, elapsed, dL_val, dR_val)

            # progress bar
            if session_start and phase != "WAITING":
                prog = min(elapsed / total_dur, 1.0)
                cv2.rectangle(frame, (10, H-38), (10+int(300*prog), H-28),
                              (100,200,255) if phase=="BASELINE" else (180,100,255), -1)
                cv2.rectangle(frame, (10, H-38), (310, H-28), (60,60,60), 1)

            cv2.putText(frame, "SPACE=start  Q=quit",
                        (W-160, H-14), cv2.FONT_HERSHEY_PLAIN, 0.85, (80,80,80), 1)

            cv2.imshow(WINDOW, frame)

            if phase == "DONE":
                cv2.waitKey(800)
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return np.array(timestamps), np.array(diameters_L), np.array(diameters_R)

# ── Analysis ───────────────────────────────────────────────────────────────
def analyse(timestamps, dL, dR):
    if len(dL) < 10:
        print("[PLR] Not enough data recorded.")
        return None

    # smooth
    dL = roll_med(dL, PUPIL_SMOOTH)
    dR = roll_med(dR, PUPIL_SMOOTH)
    avg = (dL + dR) / 2.0

    # split baseline vs post-flash
    baseline_mask = timestamps < BASELINE_SEC
    postflash_mask = timestamps >= BASELINE_SEC

    baseline_vals = avg[baseline_mask]
    post_vals     = avg[postflash_mask]
    post_t        = timestamps[postflash_mask] - BASELINE_SEC

    if len(baseline_vals) == 0 or len(post_vals) == 0:
        print("[PLR] Insufficient baseline or post-flash data.")
        return None

    D_baseline = np.mean(baseline_vals[-int(FPS*0.5):])   # last 0.5s of baseline
    D_min      = np.min(post_vals)
    min_idx    = np.argmin(post_vals)

    # latency: first frame where diameter drops >5% from baseline
    threshold  = D_baseline * 0.95
    lat_frames = np.where(post_vals < threshold)[0]
    latency_ms = (lat_frames[0] / FPS * 1000) if len(lat_frames) else 999.0

    # constriction velocity: baseline → min (px/s)
    constr_time = post_t[min_idx] if min_idx > 0 else 1/FPS
    cv = (D_baseline - D_min) / constr_time if constr_time > 0 else 0.0

    # dilation velocity: min → end (px/s)
    if min_idx < len(post_vals) - 1:
        dil_time = post_t[-1] - post_t[min_idx]
        dv = (post_vals[-1] - D_min) / dil_time if dil_time > 0 else 0.0
    else:
        dv = 0.0

    npi, comps = compute_npi(D_baseline, D_min, latency_ms, cv, dv)

    results = {
        "D_baseline_px":   round(D_baseline, 2),
        "D_min_px":        round(D_min, 2),
        "pct_constriction": round((D_baseline - D_min) / D_baseline * 100, 1),
        "latency_ms":      round(latency_ms, 1),
        "cv_px_per_s":     round(cv, 2),
        "dv_px_per_s":     round(dv, 2),
        "NPI":             npi,
        "NPI_components":  {k: round(v, 3) for k, v in comps.items()},
        "interpretation":  ("NORMAL" if npi >= 3 else
                            "BORDERLINE" if npi >= 2 else "ABNORMAL"),
    }

    print("\n── PLR Results ─────────────────────────────────")
    for k, v in results.items():
        if k != "NPI_components":
            print(f"   {k:22s}: {v}")
    print(f"────────────────────────────────────────────────\n")

    with open(ARTIFACTS_DIR / "plr_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[PLR] Saved → plr_results.json")

    return results, timestamps, avg, D_baseline, D_min, post_t, post_vals

# ── Plot ───────────────────────────────────────────────────────────────────
def plot_plr(pack):
    results, timestamps, avg, D_baseline, D_min, post_t, post_vals = pack

    plt.style.use('default')
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.50, wspace=0.32,
                            left=0.07, right=0.97,
                            top=0.90, bottom=0.08)

    ax1 = fig.add_subplot(gs[0, :])   # full-width timeline
    ax2 = fig.add_subplot(gs[1, 0])   # post-flash zoom
    ax3 = fig.add_subplot(gs[1, 1])   # NPI component bars

    def style_ax(ax, xlabel="", ylabel=""):
        ax.set_facecolor(SURFACE)
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=7.5, length=0)
        ax.grid(axis='y', color=BORDER, lw=0.6, zorder=0)
        ax.grid(axis='x', visible=False)
        if xlabel: ax.set_xlabel(xlabel, color=MUTED, fontsize=8, labelpad=5)
        if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=8, labelpad=5)

    def set_title(ax, txt):
        ax.set_title(txt, color=TEXT, fontsize=8.5,
                     fontweight='semibold', pad=7, loc='left')

    npi   = results["NPI"]
    interp = results["interpretation"]
    npi_color = GREEN if npi >= 3 else AMBER if npi >= 2 else RED

    # ── ax1: full timeline ─────────────────────────────────────────────────
    style_ax(ax1, xlabel="Time (s)", ylabel="Avg pupil diameter (px)")
    set_title(ax1, f"PLR Timeline  ·  NPI {npi:.2f}  ·  {interp}")

    ax1.plot(timestamps, avg, color=ACCENT, lw=1.0)
    ax1.axvline(BASELINE_SEC, color=AMBER, lw=1.0, ls='--', alpha=0.8,
                label="Flash")
    ax1.axhline(D_baseline, color=DIM, lw=0.7, ls='--', alpha=0.7,
                label=f"Baseline {D_baseline:.1f}px")
    ax1.axhline(D_min, color=RED, lw=0.7, ls='--', alpha=0.7,
                label=f"Min {D_min:.1f}px")

    # flash region shading
    ax1.axvspan(BASELINE_SEC, BASELINE_SEC + FLASH_SEC,
                color=AMBER, alpha=0.12)

    ax1.legend(loc="upper right", fontsize=7, framealpha=0,
               labelcolor=MUTED, handlelength=1.5)

    # ── ax2: post-flash zoom ───────────────────────────────────────────────
    style_ax(ax2, xlabel="Time after flash (s)", ylabel="Avg diameter (px)")
    set_title(ax2, f"Post-flash response  ·  latency {results['latency_ms']:.0f}ms  ·  "
              f"constriction {results['pct_constriction']:.1f}%")

    ax2.plot(post_t, post_vals, color=PURPLE, lw=1.0)
    ax2.axhline(D_baseline, color=DIM,  lw=0.7, ls='--', alpha=0.6)
    ax2.axhline(D_min,      color=RED,  lw=0.7, ls='--', alpha=0.6)

    lat_s = results["latency_ms"] / 1000
    ax2.axvline(lat_s, color=AMBER, lw=0.8, ls=':', alpha=0.7,
                label=f"Latency {results['latency_ms']:.0f}ms")
    ax2.legend(loc="upper right", fontsize=7, framealpha=0,
               labelcolor=MUTED, handlelength=1.5)

    # ── ax3: NPI component bars ────────────────────────────────────────────
    style_ax(ax3, ylabel="Score (0–1 each)")
    set_title(ax3, f"NPI Components  ·  Total {npi:.2f} / 5.0")

    comp_labels = ["Latency", "% Constr.", "Constr. V", "Dil. V", "Baseline sz"]
    comp_vals   = list(results["NPI_components"].values())
    colors      = [GREEN if v >= 0.6 else AMBER if v >= 0.3 else RED
                   for v in comp_vals]

    bars = ax3.barh(comp_labels, comp_vals, color=colors, height=0.5)
    for bar, v in zip(bars, comp_vals):
        ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{v:.2f}", va='center', color=TEXT, fontsize=8)
    ax3.set_xlim(0, 1.25)
    ax3.axvline(1.0, color=DIM, lw=0.6, ls='--')

    # ── NPI badge ──────────────────────────────────────────────────────────
    fig.text(0.97, 0.955,
             f"NPI  {npi:.2f}  —  {interp}",
             color=npi_color, fontsize=11, fontweight='bold',
             va='top', ha='right')

    fig.text(0.07, 0.962,
             "Stage 2b  —  Pupillometry / PLR Test",
             color=ACCENT, fontsize=12, fontweight='bold', va='top')

    plt.savefig("plr_output.png", dpi=150, bbox_inches='tight', facecolor=BG)
    plt.show()
    print("[PLR] Saved → plr_output.png")

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    timestamps, dL, dR = run_plr()
    pack = analyse(timestamps, dL, dR)
    if pack:
        plot_plr(pack)