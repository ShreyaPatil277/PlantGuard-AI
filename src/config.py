# config.py
# ─────────────────────────────────────────
# All project settings in one place.
# Change values here — affects entire project.
# ─────────────────────────────────────────

import os

# ── Paths ────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR   = os.path.join(BASE_DIR, 'models')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ── Dataset ──────────────────────────────
DATASET_ID  = 'vipoooool/new-plant-diseases-dataset'

# ── Model ────────────────────────────────
IMG_SIZE    = 128
BATCH_SIZE  = 64
EPOCHS_CNN  = 15
EPOCHS_TL   = 10
LEARNING_RATE = 1e-3
FINE_TUNE_LR  = 1e-5
FINE_TUNE_AT  = 50        # unfreeze last 50 layers

# ── Classes ──────────────────────────────
NUM_CLASSES = 38

# ── Saved model names ────────────────────
CNN_MODEL_PATH = os.path.join(MODEL_DIR, 'plant_disease_cnn.h5')
TL_MODEL_PATH  = os.path.join(MODEL_DIR, 'plant_disease_mobilenet.h5')
TFLITE_PATH    = os.path.join(MODEL_DIR, 'plant_disease_best.tflite')
META_PATH      = os.path.join(MODEL_DIR, 'model_metadata.json')