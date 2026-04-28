"""
backend.py — NeuraScan FastAPI Backend
=======================================
Receives RGB + pupil buffers from the frontend,
then runs the deployable pipeline in order:

    stage1_output.json (saved from frontend data)
    → stage2.py        (rPPG + pupillometry feature extraction)
    → stage3.py        (richer signal metrics)
    → stage3_predict.py (XGBoost ICP prediction)
    → stage5_gemini.py  (Gemini or local fallback clinical report)

Run with:
    uvicorn backend:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import subprocess
import sys
import os
import numpy as np
import joblib
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from project_paths import ARTIFACTS_DIR, BASE_DIR

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="NeuraScan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema ─────────────────────────────────────────────────────────
class RGBBuffers(BaseModel):
    R: List[float]
    G: List[float]
    B: List[float]

class PupilBuffers(BaseModel):
    left_px:  List[float]
    right_px: List[float]

class ScanPayload(BaseModel):
    rgb_buffers:   RGBBuffers
    pupil_buffers: PupilBuffers
    plr_done:      Optional[bool] = False

# ── Helper: run a python script ────────────────────────────────────────────
def run_stage(script_path):
    script_path = Path(script_path)
    print(f"[Backend] Running {script_path.name} ...")
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("HEADLESS", "1")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        return False, result.stderr
    return True, None


def load_json(path):
    with open(ARTIFACTS_DIR / path) as f:
        return json.load(f)


def save_json(path, payload):
    with open(ARTIFACTS_DIR / path, "w") as f:
        json.dump(payload, f, indent=2)


def derive_confidence(stage2_features, stage3_result, assessment):
    if isinstance(assessment, dict) and assessment.get("confidence_percent") is not None:
        return int(assessment["confidence_percent"])
    if isinstance(stage3_result, dict) and stage3_result.get("confidence_percent") is not None:
        return int(stage3_result["confidence_percent"])
    frames_used = int(stage2_features.get("frames_used", 0) or 0)
    return max(35, min(95, round(frames_used / 9)))


def build_frontend_features(stage2_features, stage3_results, frame_count):
    return {
        "HR": float(stage2_features.get("HR", 0) or 0),
        "SDNN": float(stage2_features.get("SDNN", 0) or 0),
        "RMSSD": float(stage2_features.get("RMSSD", 0) or 0),
        "RespRate": float(stage2_features.get("RespRate", 0) or 0),
        "P2_P1_ratio": float(stage2_features.get("P2_P1_ratio", 0) or 0),
        "pupil_L_px": float(stage2_features.get("pupil_L_px", 0) or 0),
        "pupil_R_px": float(stage2_features.get("pupil_R_px", 0) or 0),
        "asymmetry_px": float(stage2_features.get("asymmetry_px", 0) or 0),
        "NPI_proxy": float(stage2_features.get("NPI_proxy", 0) or 0),
        "LF_HF": float(stage3_results.get("LF_HF") or 0),
        "pNN50": float(stage3_results.get("pNN50_pct") or 0),
        "signal_quality": int(max(30, min(100, round(frame_count / 3)))) if frame_count else 75,
        "frames_used": int(frame_count),
        "duration_sec": int(round(frame_count / 30)) if frame_count else 0,
    }

# ── Predict endpoint ───────────────────────────────────────────────────────
@app.post("/predict")
def predict(payload: ScanPayload):
    frame_count = len(payload.rgb_buffers.G)

    stage1_data = {
        "meta": {
            "frames": frame_count,
            "buffer_len": frame_count,
            "stage2_ready": frame_count >= 300,
        },
        "rgb_buffers": {
            "R": payload.rgb_buffers.R,
            "G": payload.rgb_buffers.G,
            "B": payload.rgb_buffers.B,
        },
        "pupil_buffers": {
            "left_px": payload.pupil_buffers.left_px,
            "right_px": payload.pupil_buffers.right_px,
        },
    }
    save_json("stage1_output.json", stage1_data)
    print("[Backend] stage1_output.json saved ✓")

    ok, err = run_stage(BASE_DIR / "stage2.py")
    if not ok:
        return {"error": "stage2.py failed", "detail": err}
    print("[Backend] stage2.py done ✓")

    ok, err = run_stage(BASE_DIR / "stage3.py")
    if not ok:
        return {"error": "stage3.py failed", "detail": err}
    print("[Backend] stage3.py done ✓")

    ok, err = run_stage(BASE_DIR / "stage3_predict.py")
    if not ok:
        return {"error": "stage3_predict.py failed", "detail": err}
    print("[Backend] stage3_predict.py done ✓")

    ok, err = run_stage(BASE_DIR / "stage5_gemini.py")
    if not ok:
        return {"error": "stage5_gemini.py failed", "detail": err}
    print("[Backend] stage5_gemini.py done ✓")

    stage2_features = load_json("stage2_output.json")
    stage3_results = load_json("stage3_results.json")
    stage3_output = load_json("stage3_output.json")
    stage5 = load_json("stage5_output.json")

    assessment = stage5.get("assessment", {}) or {}
    frontend_features = build_frontend_features(stage2_features, stage3_results, frame_count)
    icp_est = float(stage3_output.get("icp_est_mmhg", 0) or 0)
    risk = (stage3_output.get("risk_level") or assessment.get("risk_level") or "UNKNOWN").upper()

    plr_details = {
        "pct_constriction": round(max(0, (1 - frontend_features.get("NPI_proxy", 4) / 5) * 35), 1),
        "constriction_vel": round(frontend_features.get("NPI_proxy", 4) / 5 * 1.2, 2),
        "latency_frames": 7,
    }

    return {
        "icp_est_mmhg": icp_est,
        "icp_estimate_mmhg": icp_est,
        "risk_level": risk,
        "confidence": derive_confidence(stage2_features, stage3_output, assessment),
        "features": frontend_features,
        "plr_details": plr_details,
        "clinical_report": stage5.get("clinical_report", "Report unavailable."),
        "assessment": assessment,
        "method": "Stage 2 + Stage 3 + XGBoost + Gemini",
    }

# ── Health check ───────────────────────────────────────────────────────────
@app.get("/")
def root():
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"status": "NeuraScan backend running"}


@app.get("/health")
def health():
    return {"status": "NeuraScan backend running"}