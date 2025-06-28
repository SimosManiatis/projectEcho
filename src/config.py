# config.py

import os
import random
import logging
import numpy as np
import torch
import warnings
from torch.jit import TracerWarning

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
JSON_FOLDER               = "I:\GIT\projectEcho\dataset\training_validation"
EXTRA_VALIDATION_FOLDER   = "I:\GIT\projectEcho\dataset\EXTRA_VALIDATION_FOLDER"
EXTRA_GROUND_TRUTH_FOLDER = "I:\GIT\projectEcho\dataset\EXTRA_GROUND_TRUTH_FOLDER"
VIZ_FOLDER                = "I:\GIT\projectEcho\src\viz_folder"
MODEL_SAVE_FOLDER         = "I:\GIT\projectEcho\src\trained_model"

# ──────────────────────────────────────────────────────────────────────────────
# Training flags
# ──────────────────────────────────────────────────────────────────────────────
TRAIN                  = True                   # If False, skip training and load saved model
ENABLE_VISUALIZATIONS  = True                   # Save static plots if True
PLOT_VISUALIZATIONS    = True                     # Show plots interactively if True

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
BATCH_SIZE      = 16
NUM_EPOCHS      = 500
LEARNING_RATE   = 5e-5
TRAIN_SPLIT     = 0.50
SEED            = 42
LAMBDA_SIGN     = 2  # weight for the sign-regularisation penalty

# ──────────────────────────────────────────────────────────────────────────────
# Model dimensions
# ──────────────────────────────────────────────────────────────────────────────
IN_FEATS     = 11    # number of input features per node
HIDDEN_FEATS = 32
OUT_FEATS    = 2     # predicting two outputs per node
DROPOUT_RATE = 0.35

# ──────────────────────────────────────────────────────────────────────────────
# Feature / Target names
# ──────────────────────────────────────────────────────────────────────────────
INPUT_FEATURE_NAMES  = [
    "length_m", "width_m", "lengthkey", "widthkey",
    "DoorCount", "WindowCount", "OpeningArea", "Level",
    "AreaAfterOpenings", "SideIntersectionArea", "total_modules"
]
OUTPUT_FEATURE_NAMES = ["ObjectCount", "ObjectVolume"]

# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# numpy print
# ──────────────────────────────────────────────────────────────────────────────
np.set_printoptions(
    suppress=True,
    formatter={'float_kind': lambda x: f"{x:.4f}"}
)

# ──────────────────────────────────────────────────────────────────────────────
# Suppress DGL / Torch JIT tracer warnings during ONNX export
# ──────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=TracerWarning)
warnings.filterwarnings(
    "ignore",
    message=r"Using len to get tensor shape.*",
    module="dgl"
)
warnings.filterwarnings(
    "ignore",
    message=r"Converting a tensor to a Python.*",
    module="dgl"
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
