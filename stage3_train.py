import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from project_paths import ARTIFACTS_DIR, DATA_DIR, MODELS_DIR

# ── Load data ──────────────────────────────────────────────────────────────
print("\n[Stage 3] Loading training data...")
df = pd.read_csv(DATA_DIR / "training_data.csv")

FEATURES = [
    "HR", "SDNN", "RMSSD", "RespRate",
    "P2_P1_ratio", "pupil_L_px", "pupil_R_px",
    "asymmetry_px", "NPI_proxy"
]
TARGET = "ICP_mmhg"

X = df[FEATURES].values
y = df[TARGET].values

# ── Split ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Scale ──────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Train XGBoost ──────────────────────────────────────────────────────────
print("[Stage 3] Training XGBoost model...")
model = XGBRegressor(
    n_estimators      = 500,
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    random_state      = 42,
    verbosity         = 0
)
model.fit(
    X_train, y_train,
    eval_set          = [(X_test, y_test)],
    verbose           = False
)

# ── Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

print(f"\n{'='*45}")
print(f"  MODEL PERFORMANCE")
print(f"{'='*45}")
print(f"  MAE  : {mae:.2f} mmHg   (lower = better)")
print(f"  R²   : {r2:.3f}         (1.0 = perfect)")
print(f"{'='*45}")

# Risk classification
def classify_icp(mmhg):
    if mmhg < 15:  return "NORMAL",   (0, 200, 0)
    if mmhg < 20:  return "ELEVATED", (255, 165, 0)
    return              "CRITICAL",   (255, 0, 0)

# Sample predictions
print("\n  Sample predictions:")
for i in range(5):
    pred  = y_pred[i]
    true  = y_test[i]
    label, _ = classify_icp(pred)
    print(f"  True: {true:.1f} mmHg | Pred: {pred:.1f} mmHg | {label}")

# ── Save model + scaler ────────────────────────────────────────────────────
with open(MODELS_DIR / "icp_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open(MODELS_DIR / "icp_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(MODELS_DIR / "feature_names.json", "w") as f:
    json.dump(FEATURES, f)

print("\n[Stage 3] Saved → icp_model.pkl")
print("[Stage 3] Saved → icp_scaler.pkl")
print("[Stage 3] Saved → feature_names.json")

# ── Feature importance plot ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#050a0e')
for ax in axes:
    ax.set_facecolor('#0f1c24')
    ax.tick_params(colors='#c8e8f0')
    for spine in ax.spines.values():
        spine.set_color('#1a3040')

# Feature importance
importance = model.feature_importances_
idx        = np.argsort(importance)
axes[0].barh(
    [FEATURES[i] for i in idx],
    importance[idx],
    color='#00d4ff', alpha=0.8
)
axes[0].set_title('Feature Importance', color='#c8e8f0', pad=8)
axes[0].set_xlabel('Importance score', color='#4a7a8a')

# Predicted vs actual
axes[1].scatter(y_test, y_pred, alpha=0.3, color='#7fff6b', s=10)
axes[1].plot([5,40], [5,40], '--', color='#ff3b5c', linewidth=1.5)
axes[1].set_title(f'Predicted vs Actual ICP\nMAE={mae:.2f} mmHg  R²={r2:.3f}',
                  color='#c8e8f0', pad=8)
axes[1].set_xlabel('Actual ICP (mmHg)',    color='#4a7a8a')
axes[1].set_ylabel('Predicted ICP (mmHg)', color='#4a7a8a')

plt.suptitle('Stage 3 — XGBoost ICP Model', color='#00d4ff',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / 'stage3_model.png', dpi=150,
            bbox_inches='tight', facecolor='#050a0e')
plt.show()
print("[Stage 3] Plot saved → stage3_model.png")