from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"

# Expected folder names inside data/raw.
# Example:
# data/raw/NORMAL/*.jpeg
# data/raw/PNEUMONIA/*.jpeg
CLASS_TO_LABEL = {"NORMAL": 0, "PNEUMONIA": 1}

# Preprocessing defaults
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
RANDOM_SEED = 42

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
