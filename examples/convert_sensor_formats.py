#!/usr/bin/env python3
"""
Example 2: Convert between sensor formats.

This example demonstrates how to:
  1. Load native DIGIT image frames using the DIGIT sensor adapter
  2. Convert to the .hapt directory format
  3. Create and store a unified (cross-sensor) representation
  4. Round-trip: load back from .hapt and verify integrity

The .hapt format stores raw sensor data verbatim plus optional unified
representations that can be shared across sensor types — enabling models
trained on one sensor to transfer to another.

Usage:
    python examples/convert_sensor_formats.py

    # Or point to a directory of PNG frames:
    python examples/convert_sensor_formats.py /path/to/digit_frames/

Requirements:
    pip install haptix[all] pillow
"""

import sys
import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from haptix import load, save, get_sensor, list_sensors
from haptix.core import (
    HaptData, RawData, UnifiedData, SensorMeta, InteractionMeta, Labels,
)
from haptix.io import ChecksumError

# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------


def _make_digit_frames(output_dir: Path, n_frames: int = 8) -> Path:
    """Create a directory of synthetic DIGIT PNG frames (640x480 RGB)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(99)

    for i in range(n_frames):
        # Fake fingerprint of a pressed elastomer
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :, 0] = rng.poisson(lam=20, size=(480, 640)).clip(0, 255).astype(np.uint8)
        img[:, :, 1] = rng.poisson(lam=30, size=(480, 640)).clip(0, 255).astype(np.uint8)
        img[:, :, 2] = rng.poisson(lam=25, size=(480, 640)).clip(0, 255).astype(np.uint8)

        # Contact patch — dark center
        cy, cx = 240, 320
        yy, xx = np.ogrid[:480, :640]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < 8000
        img[mask] = (img[mask] * 0.4).astype(np.uint8)

        Image.fromarray(img).save(output_dir / f"frame_{i:04d}.png")

    return output_dir


# ---------------------------------------------------------------------------
# Step-by-step
# ---------------------------------------------------------------------------


def step1_load_native_digit(path: Path) -> HaptData:
    """Load DIGIT frames via the sensor adapter."""
    print(f"  Adapter found: {list_sensors()}")

    digit = get_sensor("DIGIT_v2")
    print(f"  Sensor adapter: {type(digit).__name__}")
    print(f"  can_load({path}) = {digit.can_load(path)}")

    data = digit.load(
        path,
        interaction=InteractionMeta(
            type="pressing",
            normal_force_N=3.0,
            approach_angle_deg=90,
        ),
        labels=Labels(
            material="foam",
            material_category="soft_porous",
            task="pressing",
        ),
        sensor_meta=SensorMeta(
            type="DIGIT_v2",
            serial="DIGIT-2026-001",
            calibration_date="2026-01-15",
        ),
    )

    print(f"  Loaded: {data}")
    print(f"  Raw shape: {data.raw.shape}  dtype: {data.raw.dtype}")
    print(f"  Sampling rate: {data.sampling_rate_hz} Hz")
    print(f"  Checksum valid: {data.raw.verify()}")

    return data


def step2_save_as_hapt(data: HaptData, output_path: Path) -> Path:
    """Save to .hapt directory format with checksum."""
    saved = save(data, output_path)

    print(f"\n  Saved to: {saved}")
    print(f"  Directory listing:")
    for p in sorted(saved.rglob("*")):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            print(f"    {p.relative_to(saved)}  ({size_kb:.1f} KB)")

    # Inspect manifest
    with open(saved / "manifest.json") as f:
        manifest = json.load(f)
    print(f"\n  Manifest keys: {list(manifest.keys())}")
    print(f"  Sensor: {manifest['sensor']}")
    print(f"  Interaction: {manifest['interaction']}")
    print(f"  Sampling: {manifest['sampling']}")

    return saved


def step3_roundtrip_load(hapt_path: Path) -> HaptData:
    """Load the .hapt file back and verify integrity."""
    loaded = load(hapt_path)

    print(f"\n  Loaded: {loaded}")
    print(f"  Sensor type:    {loaded.sensor.type} (serial: {loaded.sensor.serial})")
    print(f"  Modality:       {loaded.modality}")
    print(f"  Interaction:    {loaded.interaction.type} @ {loaded.interaction.normal_force_N}N")
    print(f"  Material label: {loaded.labels.material}")
    print(f"  Raw data       {loaded.raw.shape}  checksum valid: {loaded.raw.verify()}")

    return loaded


def step4_add_unified_representation(data: HaptData) -> HaptData:
    """Attach a unified (cross-sensor) representation.

    This simulates converting tactile images to a force-domain representation
    that could be shared with force-sensor data. In practice you'd use a
    learned model; here we just compress to a 16D force-like vector per frame.
    """
    print("\n  Computing unified representation (imaging -> force)...")
    frames = data.raw.numpy()  # (T, H, W, C)
    T, H, W, C = frames.shape

    # Simulate a force estimator: spatial mean + variance per channel per frame
    unified_arr = np.zeros((T, 16), dtype=np.float32)
    for t in range(T):
        frame = frames[t].astype(np.float32)
        # Global stats
        mu = frame.mean(axis=(0, 1))  # (C,)
        sigma = frame.std(axis=(0, 1))  # (C,)
        # Grid-based spatial features (4x4 grid)
        grid_h, grid_w = 4, 4
        cell_h, cell_w = H // grid_h, W // grid_w
        spatial = []
        for gy in range(grid_h):
            for gx in range(grid_w):
                cell = frame[gy * cell_h:(gy + 1) * cell_h,
                             gx * cell_w:(gx + 1) * cell_w]
                spatial.append(float(cell.mean()))
                if len(spatial) >= 10:  # cap at 10 spatial features
                    break
            if len(spatial) >= 10:
                break
        unified_arr[t] = np.concatenate([mu, sigma, np.array(spatial[:10])])

    unified = UnifiedData(
        array=unified_arr,
        method="demo_force_estimator_v0",
        source_modality="imaging",
        target_modality="force",
        is_lossy=True,
        checksum=UnifiedData(
            array=unified_arr,
            method="",
            source_modality="",
            target_modality="",
            is_lossy=False,
            checksum="",
        ).checksum,
    )

    # Rebuild HaptData with unified attached
    hapt_with_unified = HaptData(
        raw=data.raw,
        sensor=data.sensor,
        modality=data.modality,
        sampling_rate_hz=data.sampling_rate_hz,
        interaction=data.interaction,
        labels=data.labels,
        unified=unified,
    )

    print(f"  Unified shape: {unified.array.shape}  dtype: {unified.array.dtype}")
    print(f"  Transform: {unified.method}  lossy: {unified.is_lossy}")

    return hapt_with_unified


def step5_save_and_load_unified(data: HaptData, output_dir: Path):
    """Save with unified data, then load and verify."""
    hapt_path = output_dir / "digit_with_unified.hapt"
    save(data, hapt_path)

    print(f"\n  Saved with unified to: {hapt_path}")
    for p in sorted(hapt_path.rglob("*")):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            print(f"    {p.relative_to(hapt_path)}  ({size_kb:.1f} KB)")

    loaded = load(hapt_path)

    print(f"\n  Loaded back:")
    print(f"    Unified present: {loaded.unified is not None}")
    if loaded.unified is not None:
        print(f"    Unified shape:  {loaded.unified.shape}")
        print(f"    Unified method: {loaded.unified.method}")
        print(f"    Verified: {np.allclose(loaded.unified.array, data.unified.array)}")

    return loaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("haptix — Sensor Format Conversion Example")
    print("=" * 60)

    # Setup
    tmp = Path(tempfile.mkdtemp(prefix="haptix_convert_"))
    print(f"\nWorking directory: {tmp}\n")

    # Step 1: Load native DIGIT frames
    print("—" * 40)
    print("Step 1: Load native DIGIT frames")
    print("—" * 40)
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    else:
        src = tmp / "digit_frames"
        _make_digit_frames(src)
        print(f"  Created {len(list(src.glob('*.png')))} synthetic frames in {src}")

    digit_data = step1_load_native_digit(src)

    # Step 2: Save as .hapt
    print("\n—" * 40)
    print("Step 2: Save to .hapt directory format")
    print("—" * 40)
    hapt_path = step2_save_as_hapt(digit_data, tmp / "digit_demo.hapt")

    # Step 3: Round-trip
    print("\n—" * 40)
    print("Step 3: Round-trip load and verify")
    print("—" * 40)
    loaded_data = step3_roundtrip_load(hapt_path)

    # Verify byte-level identity
    assert np.array_equal(loaded_data.raw.array, digit_data.raw.array), "Data mismatch!"
    print("\n  ✓ Round-trip OK — raw data is byte-identical")

    # Step 4: Add unified representation
    print("\n—" * 40)
    print("Step 4: Attach unified cross-sensor representation")
    print("—" * 40)
    unified_data = step4_add_unified_representation(digit_data)

    # Step 5: Save and reload with unified
    print("\n—" * 40)
    print("Step 5: Save + load with unified data")
    print("—" * 40)
    final = step5_save_and_load_unified(unified_data, tmp)

    # Summary
    print("\n" + "=" * 60)
    print("Summary of supported sensor formats:")
    print("=" * 60)
    print(f"  Sensor adapters loaded: {list_sensors()}")
    print()
    print("  DIGIT native format:  PNG/JPG frames directory or .mp4 video")
    print("  .hapt storage format: directory with manifest.json + raw/ + labels.json")
    print("  Unified extension:    unified/ with transform.json + data.npy")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
