# gradcam.py
# ─────────────────────────────────────────
# Grad-CAM: visualize WHERE the model looks
# in a leaf image to make its prediction.
# Usage:
#   python gradcam.py --image leaf.jpg
# ─────────────────────────────────────────

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, layers
from predict import load_model_and_classes, preprocess_image
from config import IMG_SIZE, TL_MODEL_PATH, META_PATH


def find_last_conv_layer(model):
    """Automatically find the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            return layer.name
        if hasattr(layer, 'layers'):           # nested model (MobileNetV2)
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, layers.Conv2D):
                    return sublayer.name
    raise ValueError('No Conv2D layer found in model!')


def compute_gradcam(img_norm, model, conv_layer_name, pred_index=None):
    """
    Compute Grad-CAM heatmap.
    Returns heatmap array (H, W) with values in [0, 1].
    """
    grad_model = Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(conv_layer_name).output,
            model.output
        ]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_norm[np.newaxis,...])
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_score = preds[:, pred_index]

    grads      = tape.gradient(class_score, conv_out)
    pooled     = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap    = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap    = tf.squeeze(heatmap)
    heatmap    = tf.maximum(heatmap, 0)
    heatmap    = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_img, heatmap, alpha=0.45):
    """Overlay Grad-CAM heatmap on original image."""
    heatmap_r = cv2.resize(heatmap, (original_img.shape[1],
                                      original_img.shape[0]))
    heatmap_c = cv2.applyColorMap(np.uint8(255 * heatmap_r),
                                   cv2.COLORMAP_JET)
    heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB)
    overlay   = (original_img * (1 - alpha) +
                 heatmap_c    *  alpha).astype(np.uint8)
    return heatmap_r, overlay


def visualize_gradcam(image_path, model, class_names,
                       conv_layer=None, save_path=None):
    """Full Grad-CAM pipeline for one image."""
    img, img_norm = preprocess_image(image_path)

    if conv_layer is None:
        conv_layer = find_last_conv_layer(model)
    print(f'Using conv layer: {conv_layer}')

    # Predict
    preds     = model.predict(img_norm[np.newaxis,...], verbose=0)[0]
    pred_idx  = np.argmax(preds)
    conf      = preds[pred_idx] * 100
    pred_cls  = class_names[pred_idx]
    healthy   = 'healthy' in pred_cls.lower()

    # Compute heatmap
    heatmap          = compute_gradcam(img_norm, model, conv_layer, pred_idx)
    heatmap_r, overlay = overlay_heatmap(img, heatmap)

    # Plot: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles    = ['Original Image', 'Grad-CAM Heatmap', 'Overlay']
    images    = [img, heatmap_r, overlay]
    cmaps     = [None, 'jet', None]

    for ax, title, im, cmap in zip(axes, titles, images, cmaps):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontweight='bold')
        ax.axis('off')

    status = '✅ HEALTHY' if healthy else '❌ DISEASED'
    plt.suptitle(
        f'{status}  |  {pred_cls}  |  Conf: {conf:.1f}%',
        fontsize=12, fontweight='bold',
        color='green' if healthy else 'red'
    )
    plt.tight_layout()

    out = save_path or 'outputs/gradcam_result.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.show()
    print(f'✅ Grad-CAM saved: {out}')
    return heatmap_r, overlay


# ── Run from command line ─────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grad-CAM visualization')
    parser.add_argument('--image',  required=True, help='Path to leaf image')
    parser.add_argument('--model',  default=TL_MODEL_PATH)
    parser.add_argument('--layer',  default=None, help='Conv layer name')
    parser.add_argument('--save',   default=None, help='Output path')
    args = parser.parse_args()

    model, class_names = load_model_and_classes(args.model)
    visualize_gradcam(args.image, model, class_names,
                      conv_layer=args.layer, save_path=args.save)