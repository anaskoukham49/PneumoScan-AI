from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "reports" / "manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "model" / "artifacts"
MODEL_PATH = OUTPUT_DIR / "baseline_pneumonia.keras"
METRICS_PATH = OUTPUT_DIR / "baseline_metrics.json"

BATCH_SIZE = 32
EPOCHS = 10
SEED = 42


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


def build_baseline_model(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def main() -> None:
    tf.keras.utils.set_random_seed(SEED)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found at {MANIFEST_PATH}. Run ETL first (python -m etl.run_etl)."
        )

    manifest = pd.read_csv(MANIFEST_PATH)
    x_train, y_train = load_split(manifest, "train")
    x_val, y_val = load_split(manifest, "val")
    x_test, y_test = load_split(manifest, "test")

    model = build_baseline_model(input_shape=x_train.shape[1:])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    test_metrics = model.evaluate(x_test, y_test, verbose=0)
    metric_names = model.metrics_names
    metrics_dict = {name: float(value) for name, value in zip(metric_names, test_metrics)}
    metrics_dict["history"] = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metrics_dict, fh, indent=2)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
