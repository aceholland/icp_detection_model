# ICP Detection Model — Camera-based rPPG → ICP

**Project summary:** This repository implements an end-to-end non‑invasive pipeline that captures face video, extracts rPPG from forehead skin color (CHROM + optional EfficientPhys model), computes physiological features (HR / HRV / respiration / P2/P1 waveform morphology / pupillometry), runs an XGBoost regression ensemble that estimates intracranial pressure (ICP), and produces a short clinical report via Gemini.

**High-level signal pipeline**
- Capture: MediaPipe face landmarker → forehead RGB means, pupil sizes, privacy-safe diff frames (`stage1.py`).
- rPPG extraction: CHROM algorithm (Stage 2) + optional EfficientPhys ensemble (`stage2.py`, `stage2_model.py`).
- Feature extraction: HR, HRV (SDNN, RMSSD, pNN50), respiration, P2/P1 waveform metric, pupillometry, NPI proxy (`stage2.py`, `stage3.py`).
- ICP estimation: XGBoost model trained on synthetic + real-derived data (`stage3_train.py`, [`models/icp_model.pkl`](models/icp_model.pkl)).
- Report: Gemini Pro prompt-driven clinical assessment + 4‑sentence report (`stage5_gemini.py`).

**Quick start**
1. Create and activate a Python virtual environment and install dependencies (example):

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

2. Capture a recording (stage 1):

```bash
python stage1.py
# Output: stage1_output.json (contains rgb buffers, pupil buffers, diff frames)
```

3. Run rPPG extraction & features (stage 2):

```bash
python stage2.py
# Output: stage2_output.json, stage2_output.png
```

Optional: run `stage2_model.py` to run EfficientPhys on reconstructed in-memory frames and save an ensembled BVP into `stage1_output.json` before running `stage3.py`.

```bash
python stage2_model.py
# modifies: stage1_output.json (adds model_bvp)
```

4. Advanced processing & final features (stage 3) or model inference:

```bash
python stage3.py        # generates stage3_results.json + stage3_output.png
python stage3_predict.py   # requires trained model and scaler (icp_model.pkl, icp_scaler.pkl)
```

5. Train the ICP model (if you want to re-train):

```bash
python stage3_train.py
# saves icp_model.pkl, icp_scaler.pkl, feature_names.json
```

6. Generate Gemini clinical report:

```bash
export GEMINI_API_KEY=your_key_here   # Windows: setx or use .env + python-dotenv
python stage5_gemini.py
# saves stage5_output.json
```

7. Run the web app:

```bash
uvicorn backend:app --reload --port 8000
# open http://127.0.0.1:8000/
```

The runtime files are now grouped into:
- `models/` for saved model files and metadata
- `artifacts/` for JSON outputs and plots
- `data/` for reusable training datasets
- `legacy/` for older research folders and experimental code

**Web app user guide**
1. Start the server with `uvicorn backend:app --reload --port 8000`.
2. Open `http://127.0.0.1:8000/` in your browser.
3. Click **Start Scan** and allow camera access.
4. Keep your face centered and wait while the app records the rPPG signal.
5. When the app prompts for the PLR test, click it and complete the flash step.
6. Wait for processing to finish, then open the **Results** page.
7. Read the final ICP estimate, risk level, confidence, pupil metrics, and clinical report.

**What you should see**
- `icp_est_mmhg`: the estimated intracranial pressure in mmHg.
- `risk_level`: `NORMAL`, `ELEVATED`, or `CRITICAL`.
- `clinical_report`: the final text summary produced by Gemini or the local fallback.
- If Gemini is unavailable, the app still returns a local report so the workflow stays usable.

**Key files and their roles**
- `backend.py`: FastAPI API and pipeline orchestrator.
- `index.html`: browser UI for the live scan, results, and report views.
- `stage1.py`: Real-time camera capture using MediaPipe FaceLandmarker. Computes forehead RGB means, pupil diameters, and a privacy-safe diff-normalized face-frame buffer. Exports `artifacts/stage1_output.json`.
- `stage2.py`: Implements CHROM rPPG algorithm, bandpass filtering, peak detection, HR/HRV metrics, respiration estimate, P2/P1 waveform analysis, pupillometry stats, and saves `artifacts/stage2_output.json` plus `artifacts/stage2_output.png`.
- `stage2_model.py`: Reconstructs uniform 72×72 RGB frames from the per-frame RGB means (in-memory only) and runs an external EfficientPhys model (`rppg` package) to extract BVP.
- `stage3.py`: More advanced signal processing, optional ensemble (CHROM + model_bvp), LF/HF analysis, Poincaré, and writes `artifacts/stage3_results.json` plus `artifacts/stage3_output.png`.
- `stage3_train.py`: Trains an XGBoost regressor on `data/training_data.csv` (synthetic data generator provided) to predict ICP in mmHg. Produces `models/icp_model.pkl`, `models/icp_scaler.pkl`, and `models/feature_names.json`.
- `stage3_predict.py`: Loads the saved model and scaler from `models/`, reads `artifacts/stage2_output.json`, and writes `artifacts/stage3_output.json`.
- `stage5_gemini.py`: Prepares a medical prompt and calls Gemini (via `google-generativeai`) to produce a structured risk assessment JSON and a 4‑sentence clinical report. Falls back to a local report when `GEMINI_API_KEY` or the Gemini SDK is unavailable.
- `generate_data.py`: Generates synthetic training samples (features + ICP labels) used to build `data/training_data.csv`.
- `train_rppg_to_ppg.py`: PyTorch 1D UNet training pipeline for converting noisy rPPG → clean PPG (synthetic training provided). Saves checkpoints in `models/`.
- `train_ppg_to_ecg.py`: PyTorch model to reconstruct ECG from PPG using BIDMC dataset when available (or synthetic fallback). Uses `bidmc_data/` if present.
- `layers.py`, `layers_patched.py`, `module.py`: supporting model and layer utilities used by the PyTorch training scripts (internals - read these before modifying).
- `legacy/`: older research folders and experimental code kept out of the main app flow.

**Data & models**
- `artifacts/`: runtime JSON outputs and plots generated by the pipeline.
- `models/`: trained model files, scaler files, MediaPipe assets, and model metadata.
- `data/`: reusable datasets such as `training_data.csv`.
- `artifacts/stage1_output.json`: core intermediate artifact. Contains `rgb_buffers` (R/G/B lists) and `pupil_buffers`. Do not commit sensitive recordings.
- `data/training_data.csv`: synthetic dataset produced by `generate_data.py` (used by `stage3_train.py`).
- `models/icp_model.pkl`, `models/icp_scaler.pkl`, `models/feature_names.json`: trained ICP estimators and feature metadata.

**Privacy note**
- The pipeline intentionally keeps `diff_frames` (frame-to-frame differences) which the code documents as not reconstructable back to face images. `stage2_model.py` reconstructs ONLY uniform-color patches from mean RGB values for model inference and never writes full face images to disk.

**Dependencies**
- Main Python packages used across the project: `numpy`, `scipy`, `matplotlib`, `opencv-python`, `mediapipe`, `google-generativeai`, `python-dotenv`, `torch` (for model training), `wfdb` (optional, for BIDMC), `xgboost`, `scikit-learn`, `pandas`.
- Use the top-level `requirements.txt` for the deployable app.
- Older research folders now live under `legacy/` so the main web app stays easier to scan.

**How the ICP model works (concise)**
- Features: HR, SDNN, RMSSD, RespRate, P2_P1_ratio, pupil_L_px, pupil_R_px, asymmetry_px, NPI_proxy.
- Model: XGBoost regressor trained on synthetic data that encodes physiological relationships between ICP and vitals.
- Inference: features extracted from camera-based rPPG are scaled with the trained scaler and fed to the saved model to get an ICP estimate (mmHg). A simple rule maps numeric ICP to risk bins (NORMAL / ELEVATED / CRITICAL).

**Common commands**
- Visualize stage outputs: `python stage2.py` (opens figures) and `python stage3.py`.
- Train ICP model: `python stage3_train.py`.
- Run inference with saved model: `python stage3_predict.py`.
- Generate Gemini report: `python stage5_gemini.py` (ensure `GEMINI_API_KEY` is set).

**Suggested improvements / next steps**
- Add a consolidated top-level `requirements.txt` or `pyproject.toml` and reproducible environment instructions.
- Add CLI wrappers to run the pipeline end‑to‑end with argument flags (input video, webcam, output folder).
- Add unit tests for feature extraction functions (peak detection, P2/P1 computation).
- Add a small Dockerfile for reproducibility and to isolate GPU/CPU dependencies for training scripts.
- Consider storing only aggregated features and encrypted metadata when sharing results to strengthen privacy.

**References & credits**
- CHROM rPPG algorithm and EfficientPhys model are used for camera-based pulse extraction.
- Gemini is used as a natural-language clinical summarizer (requires API key and adherence to usage policies).

If you'd like, I can:
- add a consolidated `requirements.txt` and a short `run_pipeline.sh` script; or
- create a CLI entrypoint that runs capture → features → inference → report in one command.

---
Project root: see main pipeline scripts `stage1.py`, `stage2.py`, `stage2_model.py`, `stage3.py`, `stage3_train.py`, `stage3_predict.py`, `stage5_gemini.py`.