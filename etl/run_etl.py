from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from etl.config import (
    CLASS_TO_LABEL,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    REPORTS_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def discover_images(raw_dir: Path) -> pd.DataFrame:
    records: List[Dict[str, str | int]] = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            continue
        for img_path in class_dir.rglob("*"):
            if img_path.suffix.lower() in VALID_EXTENSIONS:
                records.append(
                    {
                        "source_path": str(img_path.resolve()),
                        "class_name": class_name,
                        "label": label,
                    }
                )
    return pd.DataFrame(records)


def preprocess_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return image


def augment_image(image: np.ndarray) -> np.ndarray:
    # Lightweight augmentation for training only.
    # Input is normalized grayscale image in [0, 1].
    flipped = np.fliplr(image)
    angle = np.random.uniform(-10, 10)
    center = (IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        flipped,
        rotation_matrix,
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return rotated.astype(np.float32)


def split_dataset(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if len(df) < 3:
        raise ValueError("Need at least 3 images to build train/val/test split.")

    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-8:
        raise ValueError("Split ratios must sum to 1.0.")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    val_fraction_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    return {"train": train_df, "val": val_df, "test": test_df}


def process_split(name: str, split_df: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    output_images_dir = output_root / "images" / name
    output_images_dir.mkdir(parents=True, exist_ok=True)

    processed_rows: List[Dict[str, str | int]] = []
    for idx, row in tqdm(split_df.reset_index(drop=True).iterrows(), total=len(split_df), desc=f"Processing {name}"):
        source_path = Path(str(row["source_path"]))
        label = int(row["label"])
        class_name = str(row["class_name"])

        processed = preprocess_image(source_path)
        if processed is None:
            continue

        filename = f"{name}_{idx:06d}.npy"
        output_file = output_images_dir / filename
        np.save(output_file, processed)

        processed_rows.append(
            {
                "split": name,
                "processed_path": str(output_file.resolve()),
                "source_path": str(source_path.resolve()),
                "class_name": class_name,
                "label": label,
                "augmented": 0,
            }
        )

        if name == "train":
            aug_filename = f"{name}_{idx:06d}_aug.npy"
            aug_output_file = output_images_dir / aug_filename
            augmented = augment_image(processed)
            np.save(aug_output_file, augmented)
            processed_rows.append(
                {
                    "split": name,
                    "processed_path": str(aug_output_file.resolve()),
                    "source_path": str(source_path.resolve()),
                    "class_name": class_name,
                    "label": label,
                    "augmented": 1,
                }
            )

    return pd.DataFrame(processed_rows)


def write_reports(manifest: pd.DataFrame, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    by_split = manifest.groupby(["split", "class_name"]).size().reset_index(name="count")
    by_split_path = report_dir / "class_distribution_by_split.csv"
    by_split.to_csv(by_split_path, index=False)

    summary = {
        "total_samples": int(len(manifest)),
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "splits": {k: int(v) for k, v in manifest["split"].value_counts().to_dict().items()},
        "classes": {k: int(v) for k, v in manifest["class_name"].value_counts().to_dict().items()},
        "augmented_samples": int(manifest["augmented"].sum()) if "augmented" in manifest else 0,
    }
    with (report_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def main() -> None:
    print("Starting ETL pipeline...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = discover_images(RAW_DIR)
    if df.empty:
        raise RuntimeError(
            f"No images found in {RAW_DIR}. Expected folders: {', '.join(CLASS_TO_LABEL.keys())}."
        )

    print(f"Discovered {len(df)} images.")
    splits = split_dataset(df)

    all_processed = []
    for split_name, split_df in splits.items():
        processed_df = process_split(split_name, split_df, PROCESSED_DIR)
        all_processed.append(processed_df)

    manifest_df = pd.concat(all_processed, ignore_index=True)
    write_reports(manifest_df, REPORTS_DIR)
    print("ETL completed successfully.")
    print(f"Manifest: {REPORTS_DIR / 'manifest.csv'}")
    print(f"Summary: {REPORTS_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
