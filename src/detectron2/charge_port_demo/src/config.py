from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FRAME_DIR = OUTPUT_DIR / "frames"
LOG_DIR = OUTPUT_DIR / "logs"

CONF_THRESHOLD = 0.60
ALPHA = 0.4
BETA = 0.6

USE_PSEUDO_3D = True

CAMERA_INTRINSICS = {
    "fx": 900.0,
    "fy": 900.0,
    "cx": 640.0,
    "cy": 360.0,
}

DEFAULT_INIT_VIEW = "view_far"
DEFAULT_CAR_ID = "car_A"
VIDEO_FPS = 2.0
VIDEO_HOLD_SECONDS = 2.5
