# Project Structure

This repository is organized to separate the runtime app from generated outputs, trained artifacts, and legacy research code.

## Runtime

- `backend.py` - FastAPI app and pipeline orchestrator.
- `index.html` - Browser UI for scanning and results.
- `stage1.py`, `stage2.py`, `stage3.py`, `stage3_predict.py`, `stage5_gemini.py` - pipeline stages used by the web app.

## Models

- `models/` - saved model files, scaler files, MediaPipe assets, and metadata.

## Artifacts

- `artifacts/` - JSON outputs and plots generated during a run.

## Data

- `data/` - training datasets and other reusable input data.

## Legacy / Research

- `legacy/` - older experiment folders and exploratory research code that are not part of the main web app workflow.