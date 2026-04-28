"""
stage2_model.py
Reconstructs uint8 face crops from RGB means (for model inference only),
runs EfficientPhys, extracts BVP, then discards the frames immediately.
No face images are saved to disk.
"""

import json
import numpy as np
import rppg

from project_paths import ARTIFACTS_DIR

# ── Config ─────────────────────────────────────────────────────────────────
JSON_PATH  = ARTIFACTS_DIR / "stage1_output.json"
FPS        = 30
MODEL_NAME = "EfficientPhys.pure"
CROP_SIZE  = 72

# ── Load ───────────────────────────────────────────────────────────────────
with open(JSON_PATH) as f:
    data = json.load(f)

R = np.array(data["rgb_buffers"]["R"])
G = np.array(data["rgb_buffers"]["G"])
B = np.array(data["rgb_buffers"]["B"])
N = len(G)
print(f"[Model] {N} frames loaded")

# ── Build uint8 frames from RGB means (in-memory only, never saved) ────────
# Each frame is a uniform-color 72x72 patch — enough for EfficientPhys
# to pick up the temporal color variation (the signal it was trained on)
frames = np.zeros((N, CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
frames[:, :, :, 0] = np.clip(R, 0, 255).astype(np.uint8)[:, None, None]
frames[:, :, :, 1] = np.clip(G, 0, 255).astype(np.uint8)[:, None, None]
frames[:, :, :, 2] = np.clip(B, 0, 255).astype(np.uint8)[:, None, None]

print(f"[Model] Frames built in memory — shape: {frames.shape}  dtype: {frames.dtype}")
print(f"[Model] Frames are NOT saved to disk")

# ── Run EfficientPhys ──────────────────────────────────────────────────────
print(f"[Model] Loading {MODEL_NAME}...")
model = rppg.Model(MODEL_NAME)

print("[Model] Running inference...")
model.process_faces_tensor(frames, fps=float(FPS))

# discard frames immediately after inference
del frames

# extract BVP
bvp, timestamps = model.bvp()
bvp = np.array(bvp)

# normalise
bvp = (bvp - np.mean(bvp)) / (np.std(bvp) + 1e-8)
print(f"[Model] BVP shape: {bvp.shape}  std: {np.std(bvp):.4f}")

# ── Save BVP only (not frames) ─────────────────────────────────────────────
data["model_bvp"] = bvp.tolist()
with open(JSON_PATH, "w") as f:
    json.dump(data, f)

print(f"[Model] Saved model_bvp → {JSON_PATH}")
print("[Model] Done — run stage3.py next")