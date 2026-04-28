import json
import pickle
import numpy as np

from project_paths import ARTIFACTS_DIR, MODELS_DIR

# Load model + scaler
with open(MODELS_DIR / "icp_model.pkl",    "rb") as f: model  = pickle.load(f)
with open(MODELS_DIR / "icp_scaler.pkl",   "rb") as f: scaler = pickle.load(f)
with open(MODELS_DIR / "feature_names.json")    as f: features = json.load(f)
with open(ARTIFACTS_DIR / "stage2_output.json")    as f: data = json.load(f)

# Build feature vector from your real data
X = np.array([[data[f] for f in features]])
X_scaled = scaler.transform(X)

# Predict
icp_pred = model.predict(X_scaled)[0]

# Classify
if icp_pred < 15:
    risk, color = "NORMAL",   "✅"
elif icp_pred < 20:
    risk, color = "ELEVATED", "⚠️"
else:
    risk, color = "CRITICAL", "🚨"

print("\n" + "="*45)
print("  ICP PREDICTION FROM YOUR CAMERA DATA")
print("="*45)
for f in features:
    print(f"  {f:<20}: {data[f]}")
print(f"{'='*45}")
print(f"  Estimated ICP  : {icp_pred:.1f} mmHg")
print(f"  Risk Level     : {color} {risk}")
print("="*45)

# Save for Stage 4 (Gemini report)
result = {
    "features":     data,
    "icp_est_mmhg": round(float(icp_pred), 1),
    "risk_level":   risk
}
with open(ARTIFACTS_DIR / "stage3_output.json", "w") as f:
    json.dump(result, f, indent=2)
print("\n[Stage 3] Saved → stage3_output.json")
print("[Stage 3] Ready for Stage 4 — Gemini clinical report\n")