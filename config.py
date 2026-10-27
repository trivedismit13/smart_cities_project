"""
Configuration parameters for Smart Cities Federated Learning Vehicle Trajectory Prediction.
Dataset: NGSIM (US-101, Lankershim Blvd, Peachtree St)
"""

import os
import torch

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(BASE_DIR, "export.csv")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

for d in [CACHE_DIR, OUTPUT_DIR, PLOTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
RANDOM_SEED = 42

# Trajectory & Sequence Parameters
# NGSIM is 10 Hz. Downsampling by 5 gives 2 Hz (0.5s per frame).
# 6 frames = 3.0 seconds history; 6 frames = 3.0 seconds future prediction.
MAX_RAW_ROWS = 1_000_000  # Explicitly trimmed to 1 Million rows as requested
DOWNSAMPLE_STEP = 5
OBS_STEPS = 6       # 3.0 seconds observation horizon
# PRED_STEPS can be overridden from the environment (e.g. PRED_STEPS=10 for the 5.0 s horizon run)
PRED_STEPS = int(os.environ.get("PRED_STEPS", 6))  # 3.0 seconds prediction horizon by default
TOTAL_STEPS = OBS_STEPS + PRED_STEPS
STRIDE = 3          # Sliding window stride
MAX_VEHICLES_PER_LOC = 600  # Balanced representative vehicle sampling per domain

# Feature Configuration
# Features: [Local_X_rel, Local_Y_rel, v_Vel, v_Acc]
INPUT_DIM = 4       # Relative X, Relative Y, Velocity, Acceleration
OUTPUT_DIM = 2      # Predicted Relative X, Predicted Relative Y

# Model Hyperparameters
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 128

# Federated Learning Configuration
NUM_CLIENTS = 3     # Client 0: US-101, Client 1: Lankershim, Client 2: Peachtree
CLIENT_NAMES = {
    0: "RSU-1_US101_Highway",
    1: "RSU-2_Lankershim_Arterial",
    2: "RSU-3_Peachtree_Downtown"
}
COMMUNICATION_ROUNDS = 8
LOCAL_EPOCHS = 1
FEDPROX_MU = 0.01

# Centralized Training Configuration
CENTRALIZED_EPOCHS = 6

# Suffix for cache/result files of non-default horizons, so a 5 s run never reuses 3 s data
HORIZON_TAG = "" if PRED_STEPS == 6 else f"_pred{PRED_STEPS}"

# Evaluation & Unit conversion
FEET_TO_METERS = 0.3048  # NGSIM raw coordinates are in feet
