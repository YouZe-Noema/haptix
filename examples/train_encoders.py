#!/usr/bin/env python3
"""Train per-sensor encoders on real data and save weights.

Trains the v0.3 "pre-trained encoders for common sensor types" weights:

- **GelSight** (imaging): fitted on real YCB-Sight GelSight frames
  (``research/real-data/gelsight/YCBSight-Real/*.zip`` when present;
  falls back to the paired synthetic records generator). Object identity
  is the supervised label → PCA whitening + class-aligned rotation.
- **CoroCapacitive** (dynamic): fitted on the real Lab-CORO CSV
  (``research/real-data/coro/Dataset_01_Real.csv`` when present;
  falls back to synthetic 29-taxel records). Unsupervised whitening.

Weights are saved to ``haptix/encoders/weights/<SensorType>_v1.0.npz``
(the gitignored weights home) and can be served by
``haptix.load_trained(sensor_type)``.

Usage:
    python examples/train_encoders.py [--out DIR] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import numpy as np

import haptix
from haptix.core import HaptData, InteractionMeta, Labels, RawData, SensorMeta

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS_DIR = Path(haptix.__file__).resolve().parent / "encoders" / "weights"

MATERIALS = ["metal", "fabric", "wood", "rubber", "plastic"]


# ── Synthetic fallbacks (mirror examples/end_to_end_demo.py) ────────────


def _texture_template(material: str, h: int, w: int, seed: int) -> np.ndarray:
    """Deterministic per-material micro-texture (H, W, 3) in [0, 1]."""
    rng = np.random.RandomState(seed)
    tex = np.ones((h, w, 3), dtype=np.float64) * 0.85
    if material == "metal":
        for _ in range(12):
            x0, y0 = rng.randint(0, w), rng.randint(0, h)
            tex[y0 : y0 + 2, x0 : x0 + rng.randint(8, 24)] *= 0.55
    elif material == "fabric":
        for _ in range(8):
            x0 = rng.randint(0, w)
            tex[:, x0 : x0 + 2] *= 0.6
    elif material == "wood":
        for _ in range(10):
            y0 = rng.randint(0, h)
            tex[y0 : y0 + 3, :] *= 0.5
    elif material == "rubber":
        for _ in range(20):
            x0, y0, r = rng.randint(0, w), rng.randint(0, h), rng.randint(2, 6)
            yy, xx = np.ogrid[:h, :w]
            tex[(xx - x0) ** 2 + (yy - y0) ** 2 < r**2] *= 0.5
    else:  # plastic
        for _ in range(30):
            x0, y0 = rng.randint(0, w), rng.randint(0, h)
            tex[y0, x0 : x0 + 3] *= 0.7
    return np.clip(tex, 0.05, 1.0)


def _synthetic_imaging_records(trials_per_material: int = 6, frames: int = 8) -> list[HaptData]:
    """Labeled GelSight-style imaging records (fallback when no real data)."""
    records: list[HaptData] = []
    h, w = 48, 64
    for material in MATERIALS:
        seed_m = int(hashlib.sha256(material.encode()).hexdigest(), 16) % (2**31)
        tex = _texture_template(material, h, w, seed=seed_m)
        for trial in range(trials_per_material):
            rng = np.random.RandomState(seed_m + trial)
            base = np.zeros((h, w, 3), dtype=np.float64)
            base[..., 0] = (seed_m >> 0) % 256
            base[..., 1] = (seed_m >> 8) % 256
            base[..., 2] = (seed_m >> 16) % 256
            noise = rng.randint(0, 20, (frames, h, w, 3)).astype(np.float64)
            img = np.clip(base * tex + noise, 0, 255).astype(np.uint8)
            records.append(
                HaptData(
                    raw=RawData(
                        array=img,
                        checksum=RawData.compute_checksum(img),
                        dtype=str(img.dtype),
                        shape=img.shape,
                    ),
                    sensor=SensorMeta(type="GelSight", serial=f"syn-gs-{material}-{trial}"),
                    modality="imaging",
                    sampling_rate_hz=30.0,
                    interaction=InteractionMeta(type="pressing", normal_force_N=2.0),
                    labels=Labels(material=material, material_category=material),
                )
            )
    return records


def _synthetic_dynamic_records(trials_per_material: int = 6, frames: int = 8) -> list[HaptData]:
    """Labeled Coro-style dynamic records (fallback when no real CSV)."""
    records: list[HaptData] = []
    for material in MATERIALS:
        seed_m = int(hashlib.sha256(material.encode()).hexdigest(), 16)
        dyn_offset = np.array(
            [50.0 + ((seed_m >> (8 * (k % 8))) % 200) for k in range(29)],
            dtype=np.float64,
        )
        for trial in range(trials_per_material):
            rng = np.random.RandomState(seed_m % 2**31 + trial)
            dyn = np.broadcast_to(dyn_offset, (frames, 29)).copy()
            dyn = (dyn + rng.randn(frames, 29) * 2.0).astype(np.float32)
            records.append(
                HaptData(
                    raw=RawData(
                        array=dyn,
                        checksum=RawData.compute_checksum(dyn),
                        dtype=str(dyn.dtype),
                        shape=dyn.shape,
                    ),
                    sensor=SensorMeta(type="CoroCapacitive", serial=f"syn-coro-{material}-{trial}"),
                    modality="dynamic",
                    sampling_rate_hz=30.0,
                    interaction=InteractionMeta(type="pressing", normal_force_N=3.0),
                    labels=Labels(material=material, material_category=material),
                )
            )
    return records


# ── Real data loaders ───────────────────────────────────────────────────


def load_ycb_sight_gelsight(zips_dir: Path, segments_per_object: int = 4) -> list[HaptData]:
    """Load GelSight frames from YCB-Sight zip archives.

    Each zip holds one YCB object (``<object>/gelsight/*.jpg``); the object
    id becomes ``object_name`` (supervised label for the class rotation).
    Frames are split into ``segments_per_object`` temporal segments so the
    leave-one-record-out benchmark is honest: each held-out segment is a
    NEW contact window of a KNOWN object (segments of the same object stay
    in the training folds).
    """
    records: list[HaptData] = []
    zips = sorted(zips_dir.glob("YCBSight-Real/*.zip"))
    for zip_path in zips:
        obj_id = zip_path.stem
        frames: list[np.ndarray] = []
        with zipfile.ZipFile(zip_path) as zf:
            names = sorted(
                n for n in zf.namelist() if "/gelsight/" in n and n.lower().endswith(".jpg")
            )

            # Stable temporal order: the filename embeds a frame index
            # (gelsight_<idx>_<timestamp>.jpg), so sort numerically.
            def _frame_idx(name: str) -> int:
                try:
                    return int(name.split("gelsight_")[1].split("_")[0])
                except (IndexError, ValueError):
                    return 0

            for name in sorted(names, key=_frame_idx):
                data = zf.read(name)
                try:
                    from PIL import Image
                    import io

                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    frames.append(np.asarray(img, dtype=np.uint8))
                except Exception:
                    continue
        if not frames:
            print(f"  ! no frames loaded from {zip_path.name}")
            continue
        arr = np.stack(frames)  # [T, H, W, 3]
        seg_size = max(1, arr.shape[0] // segments_per_object)
        for seg in range(segments_per_object):
            seg_arr = arr[seg * seg_size : (seg + 1) * seg_size]
            if seg_arr.shape[0] == 0:
                continue
            records.append(
                HaptData(
                    raw=RawData(
                        array=seg_arr,
                        checksum=RawData.compute_checksum(seg_arr),
                        dtype=str(seg_arr.dtype),
                        shape=seg_arr.shape,
                    ),
                    sensor=SensorMeta(type="GelSight", serial=f"ycbsight-{obj_id}-seg{seg}"),
                    modality="imaging",
                    sampling_rate_hz=30.0,
                    interaction=InteractionMeta(type="pressing", normal_force_N=2.0),
                    labels=Labels(object_name=obj_id, material_category="object"),
                )
            )
        print(f"  ✓ {obj_id}: {len(frames)} frames -> {segments_per_object} segments")
    return records


def load_coro_real_csv(csv_path: Path) -> list[HaptData]:
    """Load the real Lab-CORO CSV as one dynamic record per taxel group.

    The CSV is ``force, tax1..tax28, Path`` — rows group by the Path column
    (taxel id). Returns one HaptData per group so the encoder sees real
    taxel dynamics.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    records: list[HaptData] = []
    for group_name, group in df.groupby("Path"):
        arr = group.drop(columns=["Path"]).to_numpy(dtype=np.float32)
        if arr.size == 0:
            continue
        records.append(
            HaptData(
                raw=RawData(
                    array=arr,
                    checksum=RawData.compute_checksum(arr),
                    dtype=str(arr.dtype),
                    shape=arr.shape,
                ),
                sensor=SensorMeta(type="CoroCapacitive", serial=f"coro-real-{group_name}"),
                modality="dynamic",
                sampling_rate_hz=30.0,
                interaction=InteractionMeta(type="pressing", normal_force_N=3.0),
                labels=Labels(material="unknown"),
            )
        )
    return records


# ── Main ────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--force", action="store_true", help="overwrite existing weight files")
    args = parser.parse_args()

    print("=" * 62)
    print("  haptix — train per-sensor encoders (v0.3)")
    print("=" * 62)

    args.out.mkdir(parents=True, exist_ok=True)
    real_gelsight_dir = REPO_ROOT / "research" / "real-data" / "gelsight"
    real_coro_csv = REPO_ROOT / "research" / "real-data" / "coro" / "Dataset_01_Real.csv"

    # ── GelSight ────────────────────────────────────────────────────────
    print("\n[1/2] GelSight encoder")
    gs_records = load_ycb_sight_gelsight(real_gelsight_dir) if real_gelsight_dir.is_dir() else []
    source = "YCB-Sight real"
    if not gs_records:
        print("  ! no real YCB-Sight data — using synthetic GelSight-style records")
        gs_records = _synthetic_imaging_records()
        source = "synthetic (fallback)"
    gs_enc = haptix.get_encoder("GelSight").fit(gs_records, label_key="object_name")
    print(f"  fitted on {len(gs_records)} records ({source}), {gs_enc.version}")
    print(f"  benchmark: {gs_enc.benchmark()}")

    gs_path = args.out / "GelSight_v1.0.npz"
    if gs_path.exists() and not args.force:
        print(f"  ! {gs_path.name} exists (use --force to overwrite)")
    else:
        gs_enc.save(gs_path)
        print(f"  ✓ saved {gs_path}")

    # ── CoroCapacitive ──────────────────────────────────────────────────
    print("\n[2/2] CoroCapacitive encoder")
    coro_records = load_coro_real_csv(real_coro_csv) if real_coro_csv.is_file() else []
    coro_source = "Coro real CSV"
    if not coro_records:
        print("  ! no real Coro CSV — using synthetic 29-taxel records")
        coro_records = _synthetic_dynamic_records()
        coro_source = "synthetic (fallback)"
    # Coro real CSV has no material labels → unsupervised whitening.
    coro_enc = haptix.get_encoder("CoroCapacitive").fit(coro_records, label_key="material")
    print(f"  fitted on {len(coro_records)} records ({coro_source}), {coro_enc.version}")
    print(f"  benchmark: {coro_enc.benchmark()}")

    coro_path = args.out / "CoroCapacitive_v1.0.npz"
    if coro_path.exists() and not args.force:
        print(f"  ! {coro_path.name} exists (use --force to overwrite)")
    else:
        coro_enc.save(coro_path)
        print(f"  ✓ saved {coro_path}")

    # ── Verification ────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Verification")
    print("=" * 62)
    for sensor in ("GelSight", "CoroCapacitive"):
        try:
            loaded = haptix.load_trained(sensor, weights_dir=str(args.out))
            assert loaded.trained is True
            print(f"  ✓ load_trained({sensor}) -> trained={loaded.trained} {loaded.version}")
        except FileNotFoundError as exc:
            print(f"  ! load_trained({sensor}) failed: {exc}")

    for path in sorted(args.out.glob("*_v1.0.npz")):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"  sha256 {path.name}: {sha}")

    print("\n  Done. Weights are ready for catalog publication (HF Hub, pending).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
