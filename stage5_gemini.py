"""
Stage 5 — Gemini Pro Clinical Report
Reads: stage3_results.json + stage3_output.json
Generates: clinical report + risk assessment
Run: python stage5_gemini.py
"""

import json
import os
from project_paths import ARTIFACTS_DIR
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ══════════════════════════════════════════════════════
# CONFIG — paste your API key here
# ══════════════════════════════════════════════════════
import os

# Optional: load a .env file if python-dotenv is installed. This keeps the
# runtime flexible: users can either export GEMINI_API_KEY or place it in a
# .env file (which should NOT be committed).
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("[Gemini] GEMINI_API_KEY is not set. Using local fallback report.")

# ══════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════
print("\n[Gemini] Loading feature data...")

with open(ARTIFACTS_DIR / "stage3_results.json") as f:
    raw_features = json.load(f)


def normalize_features(data):
    def clean(val, default=0):
        return val if val is not None else default

    return {
        "HR":           clean(data.get("HR", data.get("HR_bpm")), 0),
        "SDNN":         clean(data.get("SDNN", data.get("SDNN_ms")), 0),
        "RMSSD":        clean(data.get("RMSSD", data.get("RMSSD_ms")), 0),
        "pNN50":        clean(data.get("pNN50", data.get("pNN50_pct")), 0),
        "LF_HF":        clean(data.get("LF_HF"), 0),
        "RespRate":     clean(data.get("RespRate", data.get("resp_bpm")), 0),
        "P2_P1_ratio":  clean(data.get("P2_P1_ratio", data.get("P2P1_mean")), 0),
        "pupil_L_px":   clean(data.get("pupil_L_px"), 0),
        "pupil_R_px":   clean(data.get("pupil_R_px"), 0),
        "asymmetry_px": clean(data.get("asymmetry_px"), 0),
        "NPI_proxy":    clean(data.get("NPI_proxy"), 0),
    }


features = normalize_features(raw_features)

# Load ICP estimate if available
icp_est  = 0.0
risk     = "UNKNOWN"
try:
    with open(ARTIFACTS_DIR / "stage3_output.json") as f:
        stage3 = json.load(f)
    icp_est = stage3.get("icp_est_mmhg", 0.0)
    risk    = stage3.get("risk_level", "UNKNOWN")
except:
    # Estimate from P2/P1 if no XGBoost output
    p2p1    = features.get("P2_P1_ratio", 0.0)
    icp_est = round(10 + (p2p1 - 0.6) * 20, 1)
    risk    = "CRITICAL" if icp_est > 20 else "ELEVATED" if icp_est > 15 else "NORMAL"

print(f"[Gemini] ICP estimate : {icp_est} mmHg")
print(f"[Gemini] Risk level   : {risk}")

# ══════════════════════════════════════════════════════
# GEMINI SETUP
# ══════════════════════════════════════════════════════
model = None
if API_KEY and genai is not None:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-flash-latest")

# ══════════════════════════════════════════════════════
# PROMPT 1 — Risk Assessment (structured JSON)
# ══════════════════════════════════════════════════════
assessment_prompt = f"""
You are a neurocritical care AI assistant analyzing non-invasive 
camera-based patient monitoring data.

PATIENT VITALS:
- Heart Rate: {features.get('HR', 0)} BPM (normal: 60-100)
- SDNN: {features.get('SDNN', 0)} ms (normal: 20-100ms)
- RMSSD: {features.get('RMSSD', 0)} ms (normal: 20-50ms)
- pNN50: {features.get('pNN50', 0)}% (normal: 3-20%)
- LF/HF ratio: {features.get('LF_HF', 0)} (normal: 0.5-2.0)
- P2/P1 ratio: {features.get('P2_P1_ratio', 0)} (>1.0 = ICP risk)
- Respiratory Rate: {features.get('RespRate', 0)} breaths/min (normal: 12-20)
- Pupil Left: {features.get('pupil_L_px', 0)} px
- Pupil Right: {features.get('pupil_R_px', 0)} px
- Pupil Asymmetry: {features.get('asymmetry_px', 0)} px (>4px = concern)
- NPI Score: {features.get('NPI_proxy', 0)} (normal >3, abnormal <3)
- Estimated ICP: {icp_est} mmHg (normal: 5-15 mmHg)
- Risk Level: {risk}

Cushing's Triad indicators:
- Bradycardia: {"YES" if features.get('HR', 999) < 60 else "NO"}
- Irregular breathing: {"YES" if features.get('RespRate', 16) < 10 or features.get('RespRate', 16) > 25 else "NO"}
- P2/P1 elevated: {"YES" if features.get('P2_P1_ratio', 0) > 1.0 else "NO"}

Respond ONLY with valid JSON in this exact format, no other text:
{{
  "risk_level": "{risk}",
  "confidence_percent": 85,
  "icp_estimate_mmhg": {icp_est},
  "cushing_triad_detected": false,
  "primary_concern": "brief description of main finding",
  "secondary_concerns": ["concern 1", "concern 2"],
  "recommendation": "clinical recommendation",
  "urgency": "routine",
  "monitor_parameters": ["param1", "param2"]
}}
"""

print("\n[Gemini] Generating risk assessment...")
try:
    if model is None:
        raise RuntimeError("Gemini API key missing")

    response = model.generate_content(assessment_prompt)
    raw_text = response.text.strip()

    # Clean JSON if wrapped in markdown
    if "```json" in raw_text:
        raw_text = raw_text.split("```json")[1].split("```")[0].strip()
    elif "```" in raw_text:
        raw_text = raw_text.split("```")[1].split("```")[0].strip()

    assessment = json.loads(raw_text)
    print("[Gemini] Risk assessment done ✓")

except Exception as e:
    print(f"[Gemini] Assessment fallback used: {e}")
    assessment = {
        "risk_level":            risk,
        "confidence_percent":    75,
        "icp_estimate_mmhg":     icp_est,
        "cushing_triad_detected":features.get("P2_P1_ratio", 0) > 1.0,
        "primary_concern":       f"Estimated ICP {icp_est} mmHg — {risk}",
        "secondary_concerns":    [],
        "recommendation":        "Clinical review recommended",
        "urgency":               "urgent" if risk != "NORMAL" else "routine",
        "monitor_parameters":    ["ICP", "HR", "pupils"]
    }

# ══════════════════════════════════════════════════════
# PROMPT 2 — Clinical Report (natural language)
# ══════════════════════════════════════════════════════
report_prompt = f"""
You are a senior neurologist writing a brief clinical monitoring report.

FINDINGS:
- Non-invasive camera-based ICP monitoring
- Estimated ICP: {icp_est} mmHg
- Risk Level: {risk}
- Heart Rate: {features.get('HR', 0)} BPM
- HRV SDNN: {features.get('SDNN', 0)} ms
- HRV RMSSD: {features.get('RMSSD', 0)} ms
- P2/P1 Ratio: {features.get('P2_P1_ratio', 0)} (ICP waveform biomarker)
- NPI Score: {features.get('NPI_proxy', 0)}
- Pupil Asymmetry: {features.get('asymmetry_px', 0)} px
- Cushing Triad: {assessment.get('cushing_triad_detected', False)}
- Primary Concern: {assessment.get('primary_concern', '')}

Write a professional clinical report in exactly 4 sentences:
1. Summary of monitoring method and key findings
2. Clinical interpretation of the ICP estimate and biomarkers  
3. Pupillometry and autonomic findings
4. Clear recommendation with urgency level

Use professional medical language. Be concise and factual.
"""

print("[Gemini] Generating clinical report...")
try:
    if model is None:
        raise RuntimeError("Gemini API key missing")

    response2 = model.generate_content(report_prompt)
    report = response2.text.strip()
    print("[Gemini] Clinical report done ✓")
except Exception as e:
    print(f"[Gemini] Report fallback used: {e}")
    report = (
        f"Non-invasive camera-based monitoring estimated ICP at {icp_est} mmHg "
        f"with risk classification: {risk}. "
        f"P2/P1 ratio of {features.get('P2_P1_ratio', 0)} indicates "
        f"{'elevated' if features.get('P2_P1_ratio', 0) > 1.0 else 'normal'} "
        f"intracranial waveform morphology. "
        f"NPI score {features.get('NPI_proxy', 0)} with pupil asymmetry "
        f"{features.get('asymmetry_px', 0)}px. "
        f"Recommend {assessment.get('urgency', 'routine')} clinical review."
    )

# ══════════════════════════════════════════════════════
# OUTPUT
# ══════════════════════════════════════════════════════
risk_colors = {
    "NORMAL":   "\033[92m",   # green
    "ELEVATED": "\033[93m",   # yellow
    "CRITICAL": "\033[91m",   # red
    "UNKNOWN":  "\033[97m",   # white
}
RESET = "\033[0m"
color = risk_colors.get(risk, "\033[97m")

print("\n" + "="*55)
print("  GEMINI — RISK ASSESSMENT")
print("="*55)
print(f"  Risk Level    : {color}{assessment.get('risk_level')}{RESET}")
print(f"  ICP Estimate  : {icp_est} mmHg")
print(f"  Confidence    : {assessment.get('confidence_percent')}%")
print(f"  Cushing Triad : {assessment.get('cushing_triad_detected')}")
print(f"  Urgency       : {assessment.get('urgency')}")
print(f"  Primary       : {assessment.get('primary_concern')}")
print(f"  Recommend     : {assessment.get('recommendation')}")
print("="*55)

print("\n" + "="*55)
print("  GEMINI — CLINICAL REPORT")
print("="*55)
print(report)
print("="*55)

# ══════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════
output = {
    "features":        features,
    "icp_est_mmhg":    icp_est,
    "risk_level":      risk,
    "assessment":      assessment,
    "clinical_report": report
}

with open(ARTIFACTS_DIR / "stage5_output.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n[Gemini] Saved → stage5_output.json")
print("[Gemini] Pipeline complete ✓\n")