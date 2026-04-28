import cv2
import numpy as np
import json
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from project_paths import ARTIFACTS_DIR, MODELS_DIR


# ── Setup ──────────────────────────────────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ── Download model first (run this once) ──────────────────────────────────
import urllib.request, os
MODEL_PATH = MODELS_DIR / "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("[Stage 1] Downloading MediaPipe face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )
    print("[Stage 1] Model downloaded ✓")

# ── Landmark indices ───────────────────────────────────────────────────────
FOREHEAD_LM = [
    10, 338, 297, 332, 284, 251, 389, 356, 454,
    323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172,  58, 132,
     93, 234, 127, 162,  21,  54, 103,  67, 109
]
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
BUFFER_MAX  = 900
CROP_SIZE   = 72   # EfficientPhys input size

# ── Buffers ────────────────────────────────────────────────────────────────
R_buf, G_buf, B_buf = [], [], []
PL_buf, PR_buf      = [], []
diff_frames_buf     = []   # diff-normalized face crops (privacy-safe)
prev_crop           = None # previous frame for diff computation

# ── Helpers ────────────────────────────────────────────────────────────────
def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def lm_px(lm_list, idx, W, H):
    return (int(lm_list[idx].x * W), int(lm_list[idx].y * H))

def get_roi_rgb(frame, lm_list, indices, W, H):
    pts = [lm_px(lm_list, i, W, H) for i in indices]
    xs  = [p[0] for p in pts]
    ys  = [p[1] for p in pts]
    x1  = max(0, min(xs));  x2 = min(W, max(xs))
    y1  = max(0, min(ys));  y2 = min(H, max(ys))
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None
    mean_b = np.mean(roi[:, :, 0])
    mean_g = np.mean(roi[:, :, 1])
    mean_r = np.mean(roi[:, :, 2])
    return (mean_r, mean_g, mean_b), (x1, y1, x2, y2)

def get_face_bbox(lm_list, W, H, padding=0.15):
    """Loose face bounding box from all landmarks"""
    xs = [lm_list[i].x for i in range(min(468, len(lm_list)))]
    ys = [lm_list[i].y for i in range(min(468, len(lm_list)))]
    x1 = max(0, int((min(xs) - padding) * W))
    y1 = max(0, int((min(ys) - padding) * H))
    x2 = min(W, int((max(xs) + padding) * W))
    y2 = min(H, int((max(ys) + padding) * H))
    return x1, y1, x2, y2

def diff_normalize(curr, prev):
    """
    Diff-normalize: frame-to-frame difference, normalized by std.
    Output is float32, NOT reconstructable back to a face image.
    """
    diff = curr.astype(np.float32) - prev.astype(np.float32)
    std  = np.std(diff)
    if std > 1e-6:
        diff /= std
    return diff

def iris_radius(lm_list, center_idx, edge_idx, W, H):
    c = lm_px(lm_list, center_idx, W, H)
    e = lm_px(lm_list, edge_idx,   W, H)
    return dist(c, e), c

# ── FaceLandmarker setup ───────────────────────────────────────────────────
options = FaceLandmarkerOptions(
    base_options        = BaseOptions(model_asset_path=MODEL_PATH),
    running_mode        = VisionRunningMode.IMAGE,
    num_faces           = 1,
    min_face_detection_confidence = 0.5,
    min_tracking_confidence       = 0.5,
    output_face_blendshapes       = False,
    output_facial_transformation_matrixes = False
)

# ── Main loop ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)

print("\n[Stage 1] Starting... Press Q to quit and export\n")

frame_count = 0

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        H, W = frame.shape[:2]
        frame = cv2.flip(frame, 1)

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        result  = landmarker.detect(mp_image)
        display = frame.copy()

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            # ── Forehead ROI ───────────────────────────────────────────
            rgb_means, bbox = get_roi_rgb(frame, lm, FOREHEAD_LM, W, H)

            if rgb_means and bbox:
                r_mean, g_mean, b_mean = rgb_means
                x1, y1, x2, y2 = bbox

                pts = np.array(
                    [lm_px(lm, i, W, H) for i in FOREHEAD_LM],
                    dtype=np.int32
                )
                overlay = display.copy()
                cv2.fillPoly(overlay, [pts], (0, 100, 255))
                cv2.addWeighted(overlay, 0.2, display, 0.8, 0, display)
                cv2.polylines(display, [pts], True, (0, 140, 255), 1)
                cv2.putText(display, "FOREHEAD ROI -> rPPG",
                            (x1, max(y1-8, 15)),
                            cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 160, 255), 1)

                R_buf.append(r_mean)
                G_buf.append(g_mean)
                B_buf.append(b_mean)
                if len(R_buf) > BUFFER_MAX:
                    R_buf.pop(0); G_buf.pop(0); B_buf.pop(0)

                cv2.putText(display,
                            f"R:{r_mean:.1f}  G:{g_mean:.1f}  B:{b_mean:.1f}",
                            (10, H-60),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 200, 255), 1)

            # ── Face crop → diff-normalize → buffer ────────────────────
            fx1, fy1, fx2, fy2 = get_face_bbox(lm, W, H)
            face_roi = frame[fy1:fy2, fx1:fx2]
            if face_roi.size > 0:
                curr_crop = cv2.resize(
                    cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB),
                    (CROP_SIZE, CROP_SIZE)
                )
                if prev_crop is not None:
                    diff = diff_normalize(curr_crop, prev_crop)
                    diff_frames_buf.append(diff.tolist())
                    if len(diff_frames_buf) > BUFFER_MAX:
                        diff_frames_buf.pop(0)
                prev_crop = curr_crop

            # ── Iris ROI ───────────────────────────────────────────────
            if len(lm) > 477:
                l_rad, l_center = iris_radius(lm, 468, 469, W, H)
                r_rad, r_center = iris_radius(lm, 473, 474, W, H)

                cv2.circle(display, l_center, int(l_rad), (255, 200,  0), 2)
                cv2.circle(display, r_center, int(r_rad), (100, 255, 100), 2)

                for idx in LEFT_IRIS:
                    cv2.circle(display, lm_px(lm, idx, W, H), 2, (255, 200, 0), -1)
                for idx in RIGHT_IRIS:
                    cv2.circle(display, lm_px(lm, idx, W, H), 2, (100, 255, 100), -1)

                cv2.putText(display, f"L:{l_rad*2:.1f}px",
                            (l_center[0]-30, l_center[1]-int(l_rad)-8),
                            cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 200, 0), 1)
                cv2.putText(display, f"R:{r_rad*2:.1f}px",
                            (r_center[0]-30, r_center[1]-int(r_rad)-8),
                            cv2.FONT_HERSHEY_PLAIN, 0.9, (100, 255, 100), 1)

                asym       = abs(l_rad - r_rad)
                asym_color = (0,0,255) if asym>8 else (0,165,255) if asym>4 else (100,255,100)
                cv2.putText(display, f"Asymmetry: {asym:.1f}px",
                            (10, H-40),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, asym_color, 1)

                PL_buf.append(l_rad * 2)
                PR_buf.append(r_rad * 2)
                if len(PL_buf) > BUFFER_MAX:
                    PL_buf.pop(0); PR_buf.pop(0)

            # ── Progress bar ───────────────────────────────────────────
            buf_len  = len(G_buf)
            progress = min(buf_len / 300, 1.0)
            ready    = buf_len >= 300
            cv2.rectangle(display, (10, H-25),
                          (10 + int(300*progress), H-10),
                          (100,255,100) if ready else (0,165,255), -1)
            cv2.rectangle(display, (10, H-25), (310, H-10), (80,80,80), 1)
            cv2.putText(display,
                        f"Buffer:{buf_len}/300 {'| STAGE 2 READY!' if ready else '| filling...'}",
                        (315, H-12), cv2.FONT_HERSHEY_PLAIN, 0.85,
                        (100,255,100) if ready else (200,200,200), 1)

        else:
            cv2.putText(display, "NO FACE DETECTED",
                        (W//2 - 110, H//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.putText(display, f"Frame:{frame_count}",
                    (10, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (150,150,150), 1)
        cv2.putText(display, "Q = quit & export",
                    (10, 38), cv2.FONT_HERSHEY_PLAIN, 1.0, (150,150,150), 1)
        cv2.putText(display, "STAGE 1 | MediaPipe FaceLandmarker",
                    (W-240, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,200,255), 1)

        cv2.imshow("Stage 1 — Face Mesh ROI", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# ── Export JSON for Stage 2 ────────────────────────────────────────────────
output = {
    "meta": {
        "frames":       frame_count,
        "buffer_len":   len(G_buf),
        "diff_frames":  len(diff_frames_buf),
        "timestamp":    datetime.now().isoformat(),
        "stage2_ready": len(G_buf) >= 300
    },
    "rgb_buffers": {"R": R_buf, "G": G_buf, "B": B_buf},
    "pupil_buffers": {
        "left_px":  PL_buf,
        "right_px": PR_buf
    },
    # diff-normalized frames: float values representing frame-to-frame change
    # NOT reconstructable back into a face image — privacy safe
    "diff_frames": diff_frames_buf
}

with open(ARTIFACTS_DIR / "stage1_output.json", "w") as f:
    json.dump(output, f)

print(f"\n[Stage 1] Done — {frame_count} frames")
print(f"[Stage 1] Buffer       : {len(G_buf)} frames")
print(f"[Stage 1] Diff frames  : {len(diff_frames_buf)} frames")
print(f"[Stage 1] Stage 2 ready: {len(G_buf) >= 300}")
print(f"[Stage 1] Saved → stage1_output.json")