"""
ICP Prediction Script
=====================
Load the trained XGBoost model and predict ICP (mmHg) from new input values.

Required files in the same directory:
  - icp_model.joblib
  - scaler.joblib
"""

from pathlib import Path
import csv
import sys
import argparse
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parent
TEST_DATASET_PATH = BASE_DIR / "test_dataset.csv"

# =============================================================================
# Load Model & Scaler
# =============================================================================

# model and scaler will be set by `load_resources()` at runtime so importing
# this module doesn't execute heavy IO or prints.
model = None
scaler = None


def load_resources(model_path=None, scaler_path=None):
    """Load model and scaler from disk and assign to module globals.

    Raises a RuntimeError if loading fails.
    """
    global model, scaler
    model_file = Path(model_path) if model_path else (BASE_DIR / 'icp_model.joblib')
    scaler_file = Path(scaler_path) if scaler_path else (BASE_DIR / 'scaler.joblib')
    try:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
    except Exception as exc:
        print(f"[ERROR] Failed to load resources: {exc}", file=sys.stderr)
        raise RuntimeError("Could not load model/scaler") from exc
    print("[INFO] Model and scaler loaded successfully.")

# =============================================================================
# Prediction Function
# =============================================================================

def predict_icp(hr, hrv_sdnn, qrs, npi):
    """
    Predict ICP value in mmHg.

    Parameters:
    -----------
    hr         : float  — Heart Rate (bpm),           e.g. 84.0
    hrv_sdnn   : float  — HRV SDNN (ms),              e.g. 25.0
    hrv_rmssd  : float  — HRV RMSSD (ms),             e.g. 20.0
    hrv_lf_hf  : float  — HRV LF/HF ratio,            e.g. 2.5
    qrs        : float  — QRS duration (ms),           e.g. 90.0
    npi        : float  — Neurological Pupil Index,    e.g. 3.0

    Returns:
    --------
    icp : float — Predicted ICP in mmHg
    """
    if model is None or scaler is None:
        raise RuntimeError("Model and scaler are not loaded. Call load_resources() first.")
    # Model was trained on 4 features: HR_bpm, HRV_SDNN_ms, QRS_duration_ms, NPI
    features = np.array([[hr, hrv_sdnn, qrs, npi]])
    features_scaled = scaler.transform(features)
    icp = model.predict(features_scaled)[0]
    return round(float(icp), 2)


def load_test_dataset(limit=None):
    """
    Load rows from test_dataset.csv.

    Parameters:
    -----------
    limit : int | None
        Optional maximum number of rows to return.

    Returns:
    --------
    list[dict]
        Parsed rows from the test dataset.
    """
    rows = []
    with TEST_DATASET_PATH.open(newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def predict_icp_from_row(row):
    """
    Predict ICP using a row from test_dataset.csv.
    """
    return predict_icp(
        hr=float(row["HR_bpm"]),
        hrv_sdnn=float(row["HRV_SDNN_ms"]),
        qrs=float(row["QRS_duration_ms"]),
        npi=float(row["NPI"]),
    )
def _format_and_print_single(single_row):
    test_rows = [single_row]
    single_row = test_rows[0]
    icp = predict_icp_from_row(single_row)

    print("\n" + "=" * 50)
    print("SINGLE PATIENT PREDICTION")
    print("=" * 50)

    print(f"  Patient ID : {single_row['patient_id']}")
    print(f"  Time (min) : {single_row['time_min']}")
    print(f"  HR         : {float(single_row['HR_bpm']):.1f} bpm")
    print(f"  HRV SDNN   : {float(single_row['HRV_SDNN_ms']):.2f} ms")
    print(f"  HRV RMSSD  : {float(single_row['HRV_RMSSD_ms']):.2f} ms")
    print(f"  HRV LF/HF  : {float(single_row['HRV_LF_HF']):.3f}")
    print(f"  QRS        : {float(single_row['QRS_duration_ms']):.1f} ms")
    print(f"  NPI        : {float(single_row['NPI']):.1f}")
    print(f"  ICP actual : {float(single_row['ICP_mmHg']):.2f} mmHg")
    print(f"\n  ➜ Predicted ICP: {icp} mmHg")

    # ICP severity interpretation
    if icp < 15:
        status = "Normal (< 15 mmHg)"
    elif icp < 20:
        status = "Borderline (15–20 mmHg)"
    elif icp < 30:
        status = "Elevated — Intervention may be needed (20–30 mmHg)"
    else:
        status = "CRITICAL — Immediate attention required (> 30 mmHg)"

    print(f"  ➜ Status: {status}")


def _batch_predict_and_print(rows):
    print("\n" + "=" * 50)
    print("BATCH PREDICTION — MULTIPLE PATIENTS")
    print("=" * 50)

    print(f"{'Name':<12} {'HR':>6} {'SDNN':>6} {'RMSSD':>7} {'LF/HF':>7} {'QRS':>6} {'NPI':>5} {'ICP (mmHg)':>12} {'Status'}")
    print("-" * 90)

    for row in rows:
        name = f"{row['patient_id']}-{row['time_min']}m"
        hr = float(row["HR_bpm"])
        sdnn = float(row["HRV_SDNN_ms"])
        rmssd = float(row["HRV_RMSSD_ms"])
        lf_hf = float(row["HRV_LF_HF"])
        qrs = float(row["QRS_duration_ms"])
        npi = float(row["NPI"])
        actual_icp = float(row["ICP_mmHg"])
        icp = predict_icp_from_row(row)
        if icp < 15:
            status = "Normal"
        elif icp < 20:
            status = "Borderline"
        elif icp < 30:
            status = "Elevated"
        else:
            status = "CRITICAL"
        print(f"{name:<12} {hr:>6.1f} {sdnn:>6.1f} {rmssd:>7.1f} {lf_hf:>7.2f} {qrs:>6.1f} {npi:>5.1f} {icp:>12.2f}   {status}  (actual: {actual_icp:.2f})")

    print("\n[DONE] Predictions complete.")


def main():
    parser = argparse.ArgumentParser(description="Predict ICP using a pre-trained XGBoost model.")
    parser.add_argument("--test-csv", dest="test_csv", default=str(TEST_DATASET_PATH),
                        help="Path to test CSV file")
    parser.add_argument("--limit", type=int, default=4,
                        help="Number of rows to show in batch mode")
    parser.add_argument("--model", dest="model_path", default=None,
                        help="Path to model joblib file (optional)")
    parser.add_argument("--scaler", dest="scaler_path", default=None,
                        help="Path to scaler joblib file (optional)")
    args = parser.parse_args()

    try:
        load_resources(model_path=args.model_path, scaler_path=args.scaler_path)
    except RuntimeError:
        sys.exit(1)

    # Single patient (first row)
    test_rows = load_test_dataset(limit=1)
    if not test_rows:
        print("[WARN] No rows found in test CSV.")
        return
    _format_and_print_single(test_rows[0])

    # Batch
    patients = load_test_dataset(limit=args.limit)
    _batch_predict_and_print(patients)


if __name__ == "__main__":
    main()
