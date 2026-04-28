# NeuraScan — Non-Invasive ICP Detection Model

**Project summary:** NeuraScan is an end-to-end medical AI pipeline that estimates Intracranial Pressure (ICP) using only a standard webcam. It combines camera-based rPPG (Remote Photoplethysmography), real-time pupillometry, and XGBoost regression to provide a non-invasive alternative to surgical ICP monitoring.

---

## 🚀 Quick Start

### 1. Local Development (Virtual Env)
```bash
# Setup environment
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the web app
uvicorn backend:app --reload --port 8000
```
Open [http://localhost:8000](http://localhost:8000) to start a scan.

### 2. Docker (Fastest for Deployment)
The project includes a pre-configured `Dockerfile` and `docker-compose.yml`.
```bash
docker compose up --build
```
This handles all system dependencies (OpenCV, MediaPipe) automatically and maps your scan results to the `./artifacts` folder.

---

## 🛠 Features & Pipeline
1.  **Capture (`stage1.py` / Browser)**: MediaPipe tracks 468 face landmarks to extract forehead RGB means and pupil diameters.
2.  **Signal Processing (`stage2.py`)**: Implements the **CHROM** algorithm to extract heart rate, HRV (SDNN, RMSSD), and respiratory rate from subtle skin color changes.
3.  **Advanced Metrics (`stage3.py`)**: Computes P2/P1 waveform morphology, LF/HF ratios, and Poincaré analysis.
4.  **ICP Estimation (`stage3_predict.py`)**: An XGBoost regressor (trained on MIMIC-III derived data) predicts ICP in mmHg.
5.  **Clinical Report (`stage5_gemini.py`)**: Generates an automated clinical summary using **Gemini Pro**.

---

## 📂 Project Structure
- `backend.py`: FastAPI orchestrator that runs the pipeline stages.
- `index.html`: Stunning, premium glassmorphic UI for live scanning and result visualization.
- `run_pipeline.py`: CLI tool to run capture → inference → report in one command.
- `models/`: Pre-trained XGBoost models, scalers, and MediaPipe assets.
- `artifacts/`: JSON outputs, clinical reports, and signal plots generated during scans.

---

## ☁️ Deployment Options

### Hugging Face + Vercel (Split Deployment)
For high-availability hosting, you can split the app:
- **Backend**: Deploy the `Dockerfile` to **Hugging Face Spaces** (Port 7860).
- **Frontend**: Deploy `index.html` to **Vercel**. Update the `API` constant in `index.html` to point to your HF Space URL.
- Detailed Guide: [HF_Vercel_Deployment.md](./HF_Vercel_Deployment.md)

---

## 📄 Privacy & Research Note
- **Privacy**: The pipeline uses "diff-normalized" frames for model inference, which are mathematically non-reconstructable back to recognizable face images.
- **Disclaimer**: This is a **research prototype**. It is NOT intended for clinical diagnosis or treatment decisions without professional medical validation.

---

## 📜 Key Commands
- **Run Full Pipeline (CLI)**: `python run_pipeline.py`
- **Train ICP Model**: `python stage3_train.py`
- **Generate Synthetic Data**: `python generate_data.py`
- **Clean Environment**: `rm -rf artifacts/*`

---
*Built for emergency triage and resource-limited settings.*