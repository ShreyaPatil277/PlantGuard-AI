# predict.py
# ─────────────────────────────────────────
# Load a saved model and predict disease
# from any new leaf photo.
# Usage:
#   python predict.py --image leaf.jpg
# ─────────────────────────────────────────

import argparse, json, time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from config import IMG_SIZE, META_PATH, TL_MODEL_PATH


def load_model_and_classes(model_path=TL_MODEL_PATH,
                            meta_path=META_PATH):
    """Load saved model and class names."""
    model = tf.keras.models.load_model(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta['class_names']


def preprocess_image(image_path, img_size=IMG_SIZE):
    """Read and preprocess a single image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    return img, img.astype(np.float32) / 255.0


def predict(image_path, model, class_names, top_k=5):
    """
    Run inference on one image.
    Returns predicted class, confidence, top-k predictions.
    """
    img, img_norm = preprocess_image(image_path)

    t0    = time.time()
    preds = model.predict(img_norm[np.newaxis,...], verbose=0)[0]
    ms    = (time.time() - t0) * 1000

    top_idx  = np.argsort(preds)[::-1][:top_k]
    top_preds= [(class_names[i], float(preds[i]) * 100) for i in top_idx]

    pred_cls = top_preds[0][0]
    conf     = top_preds[0][1]
    healthy  = 'healthy' in pred_cls.lower()

    # Parse plant and disease
    parts   = pred_cls.split('___')
    plant   = parts[0].replace('_', ' ')
    disease = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'

    # Print result
    print(f'\n{"="*45}')
    print(f'  {"✅ HEALTHY" if healthy else "❌ DISEASED"}')
    print(f'  Plant    : {plant}')
    print(f'  Condition: {disease}')
    print(f'  Confidence: {conf:.2f}%')
    print(f'  Time     : {ms:.2f} ms')
    print(f'{"="*45}')
    print(f'\nTop-{top_k} Predictions:')
    for cls, c in top_preds:
        bar = '█' * int(c // 5)
        print(f'  {c:6.2f}%  {bar:20s}  {cls[:40]}')

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.imshow(img)
    ax1.set_title(
        f'{"✅ HEALTHY" if healthy else "❌ DISEASED"}\n'
        f'{plant} — {disease}\n'
        f'Confidence: {conf:.2f}%',
        fontsize=11,
        color='green' if healthy else 'red',
        fontweight='bold'
    )
    ax1.axis('off')

    labels     = [p[0].replace('___',' | ')[:35] for p in top_preds]
    values     = [p[1] for p in top_preds]
    bar_colors = ['#2ecc71' if 'healthy' in l.lower() else '#e74c3c'
                  for l in labels]
    ax2.barh(labels[::-1], values[::-1], color=bar_colors[::-1], alpha=0.85)
    ax2.set_xlabel('Confidence (%)')
    ax2.set_xlim(0, 105)
    ax2.set_title(f'Top-{top_k} Predictions', fontweight='bold')
    for i, v in enumerate(values[::-1]):
        ax2.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('outputs/last_prediction.png', dpi=120, bbox_inches='tight')
    plt.show()

    return pred_cls, conf, top_preds


# ── Run from command line ─────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict plant disease')
    parser.add_argument('--image', required=True, help='Path to leaf image')
    parser.add_argument('--model', default=TL_MODEL_PATH, help='Model path')
    parser.add_argument('--topk',  default=5, type=int)
    args = parser.parse_args()

    model, class_names = load_model_and_classes(args.model)
    predict(args.image, model, class_names, top_k=args.topk)
    