# train.py
# ─────────────────────────────────────────
# Handles full training pipeline:
#   - Data generators
#   - Training Custom CNN
#   - Training MobileNetV2 (Phase 1 + Phase 2)
#   - Saving models
# ─────────────────────────────────────────

import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from model import build_custom_cnn, build_mobilenet_model
from config import *


def create_generators(train_dir, valid_dir):
    """Create train and validation data generators."""

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        shear_range=0.1,
        fill_mode='nearest'
    )
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    valid_gen = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return train_gen, valid_gen


def get_callbacks(model_save_path, monitor='val_accuracy'):
    """Standard callbacks used in all training runs."""
    return [
        ModelCheckpoint(model_save_path,
                        monitor=monitor,
                        save_best_only=True,
                        mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=1e-7, verbose=1)
    ]


def train_cnn(train_gen, valid_gen):
    """Train the Custom CNN model."""
    print('\n🚀 Training Custom CNN...')
    model = build_custom_cnn()
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        train_gen,
        steps_per_epoch=200,
        epochs=EPOCHS_CNN,
        validation_data=valid_gen,
        validation_steps=50,
        callbacks=get_callbacks(CNN_MODEL_PATH),
        verbose=1
    )
    print('✅ CNN training done!')
    return model, history


def train_mobilenet(train_gen, valid_gen):
    """
    Train MobileNetV2 in two phases:
    Phase 1 — Head only (base frozen)
    Phase 2 — Fine-tune top 50 layers
    """
    model, base = build_mobilenet_model()

    # ── Phase 1 ──────────────────────────
    print('\n🚀 Phase 1 — Training head only...')
    model.compile(
        optimizer=optimizers.Adam(LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        train_gen,
        steps_per_epoch=200,
        epochs=10,
        validation_data=valid_gen,
        validation_steps=50,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )

    # ── Phase 2: Fine-tuning ─────────────
    print('\n🔧 Phase 2 — Fine-tuning top layers...')
    base.trainable = True
    for layer in base.layers[:-FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(FINE_TUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        train_gen,
        steps_per_epoch=200,
        epochs=EPOCHS_TL,
        validation_data=valid_gen,
        validation_steps=50,
        callbacks=get_callbacks(TL_MODEL_PATH),
        verbose=1
    )
    print('✅ MobileNetV2 training done!')
    return model, history


def save_metadata(class_names):
    """Save class names and config to JSON."""
    meta = {
        'class_names'  : class_names,
        'num_classes'  : len(class_names),
        'img_size'     : IMG_SIZE,
        'dataset'      : DATASET_ID
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'✅ Metadata saved: {META_PATH}')


# ── Run training ─────────────────────────
if __name__ == '__main__':
    import kagglehub

    # Download dataset
    path = kagglehub.dataset_download(DATASET_ID)

    # Find train/valid dirs
    TRAIN_DIR, VALID_DIR = None, None
    for root, dirs, _ in os.walk(path):
        name = os.path.basename(root).lower()
        if name == 'train' and len(dirs) > 30:
            TRAIN_DIR = root
        if name in ('valid', 'val') and len(dirs) > 30:
            VALID_DIR = root

    train_gen, valid_gen = create_generators(TRAIN_DIR, VALID_DIR)
    save_metadata(list(train_gen.class_indices.keys()))

    cnn_model, cnn_hist     = train_cnn(train_gen, valid_gen)
    tl_model,  tl_hist      = train_mobilenet(train_gen, valid_gen)

    print('\n🎉 All training complete!')
    