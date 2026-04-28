import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL = BASE_DIR / 'icp_model.joblib'
SCALER = BASE_DIR / 'scaler.joblib'
TEST = BASE_DIR / 'test_dataset.csv'

print('Loading model and scaler...')
model = joblib.load(MODEL)
scaler = joblib.load(SCALER)
print('Loaded.')

df = pd.read_csv(TEST)
cols = ['HR_bpm','HRV_SDNN_ms','QRS_duration_ms','NPI']
print('Using features:', cols)
X = df[cols].values
X_scaled = scaler.transform(X[:4])
preds = model.predict(X_scaled)
for i,p in enumerate(preds):
    print(f"Row {i}: patient={df.loc[i,'patient_id']}, time_min={df.loc[i,'time_min']}, predicted ICP={p:.2f}, actual ICP={df.loc[i,'ICP_mmHg']}")
