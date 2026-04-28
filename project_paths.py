from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
LEGACY_DIR = BASE_DIR / "legacy"


for directory in (ARTIFACTS_DIR, MODELS_DIR, DATA_DIR, DOCS_DIR, LEGACY_DIR):
    directory.mkdir(exist_ok=True)