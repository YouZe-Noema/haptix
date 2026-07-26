#!/usr/bin/env python3
"""
Example 1: Load DIGIT tactile data and train a material classifier.

This example demonstrates the full ML pipeline with the haptix library:
  1. Load DIGIT image frames into .hapt format
  2. Extract image features (color histogram + texture)
  3. Train a simple sklearn classifier to predict material type
  4. Evaluate on a held-out test set

Usage:
    # Run with synthetic demo data (no real DIGIT files needed):
    python examples/digit_material_classifier.py

    # Or point to a directory of .hapt files:
    python examples/digit_material_classifier.py /path/to/hapt_files/

Requirements:
    pip install haptix[all] scikit-learn pillow
"""

import sys
import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from haptix import load, save, get_sensor, list_sensors
from haptix.core import (
    HaptData, RawData, SensorMeta, InteractionMeta, Labels,
)

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(data: HaptData) -> np.ndarray:
    """Extract a feature vector from tactile image data.

    For imaging-mode DIGIT data (T, H, W, C), we compute:
      - Per-frame color histogram (RGB, 32 bins each = 96 values)
      - Per-frame mean + std intensity (3 channels * 2 = 6 values)
      - Frame-to-frame temporal variance (3 values)
      - Global image texture: Laplacian variance proxy via std of diff

    Returns shape (F,) — flattened per-sample feature vector.
    """
    frames = data.raw.numpy()  # (T, H, W, C)

    if frames.ndim != 4:
        raise ValueError(f"Expected 4D frames (T,H,W,C), got shape {frames.shape}")

    T, H, W, C = frames.shape

    # --- Color histogram ---
    n_bins = 32
    hist_features = []
    for t in range(min(T, 5)):  # average over first 5 frames for speed
        for c in range(C):
            hist, _ = np.histogram(frames[t, :, :, c], bins=n_bins, range=(0, 256))
            hist = hist.astype(np.float32) / (H * W)  # normalize
            hist_features.append(hist)

    # --- Channel statistics ---
    stat_features = []
    for c in range(C):
        stat_features.append(float(frames[:, :, :, c].mean()))
        stat_features.append(float(frames[:, :, :, c].std()))

    # --- Temporal variance ---
    temporal_std = [float(frames[:, :, :, c].std(axis=0).mean()) for c in range(C)]

    # Concatenate all features
    features = np.concatenate(hist_features + [np.array(stat_features + temporal_std)])

    return features.astype(np.float32)


def extract_batch_features(hapt_files: list[Path]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract features + labels from a list of .hapt directories.

    Returns:
        X: (N, D) feature matrix
        y: (N,) integer label indices
        class_names: list of unique material names
    """
    materials = []
    feature_list = []

    for path in hapt_files:
        try:
            data = load(path)
        except Exception as e:
            print(f"  [skip] {path.name}: {e}")
            continue

        if data.labels.material is None:
            print(f"  [skip] {path.name}: no material label")
            continue

        feats = extract_features(data)
        feature_list.append(feats)
        materials.append(data.labels.material)

    X = np.stack(feature_list, axis=0)
    class_names = sorted(set(materials))
    label_map = {name: i for i, name in enumerate(class_names)}
    y = np.array([label_map[m] for m in materials])

    return X, y, class_names


# ---------------------------------------------------------------------------
# Synthetic demo data
# ---------------------------------------------------------------------------


def _make_demo_hapt(output_dir: Path, material: str, seed: int, n_frames: int = 10):
    """Create a synthetic .hapt file with DIGIT-style random frames."""
    rng = np.random.RandomState(seed)

    # Simulate different material "signatures" via color tint + texture
    color_map = {
        "metal":      (180, 180, 190),
        "plastic":    (200, 220, 180),
        "fabric":     (160, 130, 140),
        "wood":       (140, 110, 80),
        "rubber":     (60,  60,  65),
    }
    base_r, base_g, base_b = color_map.get(material, (128, 128, 128))

    frames = np.zeros((n_frames, 240, 320, 3), dtype=np.uint8)
    for t in range(n_frames):
        noise = rng.randn(240, 320, 3) * 15
        frame = np.clip(
            np.stack([
                np.full((240, 320), base_r) + noise[:, :, 0],
                np.full((240, 320), base_g) + noise[:, :, 1],
                np.full((240, 320), base_b) + noise[:, :, 2],
            ], axis=-1),
            0, 255,
        ).astype(np.uint8)

        # Add material-specific texture pattern
        if material == "fabric":
            # Horizontal streaks
            frame[::8, :, :] = (frame[::8, :, :] * 0.7).astype(np.uint8)
        elif material == "metal":
            # Specular highlight
            cx, cy = 160, 120
            yy, xx = np.ogrid[:240, :320]
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < 400
            frame[mask] = np.clip(frame[mask].astype(np.int16) + 40, 0, 255).astype(np.uint8)

        frames[t] = frame

    hapt_dir = output_dir / f"{material}_{seed:04d}.hapt"
    data = HaptData(
        raw=RawData(
            array=frames,
            checksum=RawData.compute_checksum(frames),
            dtype="uint8",
            shape=frames.shape,
        ),
        sensor=SensorMeta(type="DIGIT_v2", serial="demo-sensor-001"),
        modality="imaging",
        sampling_rate_hz=60.0,
        interaction=InteractionMeta(
            type="pressing",
            normal_force_N=2.0,
        ),
        labels=Labels(material=material, material_category="demo"),
    )
    save(data, hapt_dir)
    return hapt_dir


def _create_demo_dataset(n_per_class: int = 8) -> tuple[list[Path], list[Path]]:
    """Create synthetic DIGIT .hapt files and split into train/test."""
    tmp = Path(tempfile.mkdtemp(prefix="haptix_demo_"))
    materials = ["metal", "plastic", "fabric", "wood", "rubber"]

    all_files = []
    for mat in materials:
        for i in range(n_per_class):
            p = _make_demo_hapt(tmp, mat, seed=i + 42)
            all_files.append(p)

    rng = np.random.RandomState(0)
    rng.shuffle(all_files)

    split = int(len(all_files) * 0.8)
    train_files = all_files[:split]
    test_files = all_files[split:]
    return train_files, test_files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # 1. Discover available sensors
    print("=" * 60)
    print("haptix — DIGIT Material Classifier Example")
    print("=" * 60)
    print(f"Registered sensors: {list_sensors()}")
    print()

    # 2. Get data — either user-provided path or synthetic demo
    if len(sys.argv) > 1:
        data_root = Path(sys.argv[1])
        if not data_root.is_dir():
            print(f"Error: {data_root} is not a directory")
            sys.exit(1)
        hapt_files = sorted(data_root.glob("*.hapt"))
        if not hapt_files:
            print(f"No .hapt files found in {data_root}")
            sys.exit(1)
        print(f"Loading {len(hapt_files)} .hapt files from {data_root}")
        train_files, test_files = hapt_files[:int(len(hapt_files) * 0.8)], hapt_files[int(len(hapt_files) * 0.8):]
    else:
        print("No path provided — creating synthetic DIGIT demo data...")
        train_files, test_files = _create_demo_dataset(n_per_class=8)
        print(f"Created {len(train_files) + len(test_files)} synthetic .hapt files")
        print(f"  Train: {len(train_files)}  Test: {len(test_files)}")
    print()

    # 3. Extract features
    print("Extracting features from training set...")
    X_train, y_train, class_names = extract_batch_features(train_files)
    print(f"  Features: {X_train.shape[0]} samples x {X_train.shape[1]} dims")
    print(f"  Classes:  {class_names}")
    print()

    print("Extracting features from test set...")
    X_test, y_test, _ = extract_batch_features(test_files)
    print(f"  Features: {X_test.shape[0]} samples x {X_test.shape[1]} dims")
    print()

    # 4. Train classifier
    print("Training Random Forest classifier...")
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, accuracy_score

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\nTest accuracy: {acc:.2%}")
        print()
        print("Classification report:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        # Feature importance (top 10)
        importances = clf.feature_importances_
        top_k = min(10, len(importances))
        top_idx = np.argsort(importances)[-top_k:][::-1]
        print(f"Top {top_k} feature indices: {top_idx}")

    except ImportError:
        print("sklearn not installed. Install with: pip install scikit-learn")
        print("Skipping classifier training.")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
