from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CODEBRIM_DIR = DATA_DIR / "codebrim_raw" / "classification_dataset"
SDNET_DIR = DATA_DIR / "sdnet2018"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

CODEBRIM_CLASSES = ["background", "crack", "spallation", "efflorescence", "exposed_bars", "corrosion_stain"]
SDNET_CLASSES = ["no_crack", "crack"]
UNIFIED_CLASSES = ["crack", "spallation", "efflorescence", "exposed_bars", "corrosion_stain"]
SEVERITY_BANDS = ["cosmetic", "watch", "significant"]

IMAGE_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS_DEFAULT = 15
BACKBONE = "resnet18"

WINDOW_SIZE = 224
WINDOW_STRIDE = 112
CONFIDENCE_THRESHOLD = 0.6
NMS_IOU_THRESHOLD = 0.3
