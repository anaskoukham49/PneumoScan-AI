from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import auc, confusion_matrix, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "reports" / "manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "model" / "artifacts"
MODEL_PATH = OUTPUT_DIR / "baseline_pneumonia.keras"
METRICS_PATH = OUTPUT_DIR / "baseline_metrics.json"

def load_split(manifest: pd.DataFrame, split_name: str) -> Tuple[np.ndarray, np.ndarray]:
    split_df = manifest[manifest["split"] == split_name].copy()
    x_list = []
    y_list = []
    for _, row in split_df.iterrows():
        image = np.load(row["processed_path"]).astype(np.float32)
        image = np.expand_dims(image, axis=-1)
        x_list.append(image)
        y_list.append(int(row["label"]))

    if not x_list:
        raise RuntimeError(f"No samples found for split '{split_name}'.")

    x = np.stack(x_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    return x, y

def plot_training_history(history: dict, save_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    ax1.plot(history.get('accuracy', []), label='Train Accuracy', color='blue')
    ax1.plot(history.get('val_accuracy', []), label='Val Accuracy', color='orange')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='lower right')

    # Loss
    ax2.plot(history.get('loss', []), label='Train Loss', color='blue')
    ax2.plot(history.get('val_loss', []), label='Val Loss', color='orange')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred_classes, save_path: Path):
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'], 
                yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_pred_probs, save_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    print("Loading test data and model...")
    manifest = pd.read_csv(MANIFEST_PATH)
    x_test, y_test = load_split(manifest, "test")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Running predictions on test set...")
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()

    print("Generating evaluation plots...")
    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred_classes, OUTPUT_DIR / "confusion_matrix.png")
    
    # ROC Curve
    plot_roc_curve(y_test, y_pred_probs, OUTPUT_DIR / "roc_curve.png")

    # Training History
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as fh:
            metrics_data = json.load(fh)
        if "history" in metrics_data:
            plot_training_history(metrics_data["history"], OUTPUT_DIR / "training_history.png")
    
    print(f"Evaluation completed. Plots saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
