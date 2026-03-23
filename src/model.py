# model.py
# ─────────────────────────────────────────
# Contains both model architectures:
#   1. Custom CNN (built from scratch)
#   2. MobileNetV2 (transfer learning)
# ─────────────────────────────────────────

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from config import IMG_SIZE, NUM_CLASSES


def build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                     num_classes=NUM_CLASSES):
    """
    Custom 5-block CNN built from scratch.
    Each block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool -> Dropout
    """
    inputs = layers.Input(shape=input_shape)

    def conv_block(x, filters, dropout=0.25):
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout)(x)
        return x

    x = conv_block(inputs, 32,  dropout=0.20)
    x = conv_block(x,      64,  dropout=0.25)
    x = conv_block(x,      128, dropout=0.25)
    x = conv_block(x,      256, dropout=0.30)
    x = conv_block(x,      512, dropout=0.30)

    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='CustomCNN')


def build_mobilenet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                          num_classes=NUM_CLASSES):
    """
    MobileNetV2 Transfer Learning model.
    Base: pretrained on ImageNet (frozen initially)
    Head: custom classifier for plant diseases
    """
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False   # frozen for Phase 1

    inputs  = layers.Input(shape=input_shape)
    x       = layers.Rescaling(scale=2.0, offset=-1.0)(inputs)
    x       = base(x, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(512, activation='relu')(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.5)(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='MobileNetV2_PlantDisease')
    return model, base