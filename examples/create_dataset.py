#!/usr/bin/env python3
"""Batch-create a tactile dataset from synthetic data with varied parameters."""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from haptix import save
from haptix.core import (
    HaptData,
    InteractionMeta,
    Labels,
    RawData,
    SensorMeta,
)


def make_tactile_recording(
    material: str,
    category: str,
    interaction_type: str,
    speed_mm_s: float,
    force_N: float,
    num_frames: int = 30,
    seed: int = 0,
) -> HaptData:
    """Create a synthetic tactile recording with specified parameters."""
    rng = np.random.RandomState(seed)
    frames = rng.randint(0, 255, (num_frames, 240, 320, 3), dtype=np.uint8)

    return HaptData(
        raw=RawData(
            array=frames,
            checksum=RawData.compute_checksum(frames),
            dtype="uint8",
            shape=frames.shape,
        ),
        sensor=SensorMeta(type="DIGIT_v2", serial="SN-DIGIT-001"),
        modality="imaging",
        sampling_rate_hz=60.0,
        interaction=InteractionMeta(
            type=interaction_type,
            speed_mm_s=speed_mm_s,
            normal_force_N=force_N,
        ),
        labels=Labels(material=material, material_category=category, task=interaction_type),
    )


def main():
    print("=" * 60)
    print("Creating a Tactile Dataset")
    print("=" * 60)

    # --- Define a small dataset with varied materials ---
    materials = [
        ("sandpaper_grit_40", "abrasive", "sliding", 80.0, 1.0),
        ("sandpaper_grit_80", "abrasive", "sliding", 50.0, 2.0),
        ("sandpaper_grit_120", "abrasive", "sliding", 30.0, 1.5),
        ("cotton_fabric", "fabric", "sliding", 100.0, 0.5),
        ("silk_fabric", "fabric", "sliding", 120.0, 0.3),
        ("rubber_block", "elastomer", "pressing", 0.0, 5.0),
        ("steel_plate", "metal", "static", 0.0, 10.0),
    ]

    tmp = Path(tempfile.mkdtemp())
    dataset_dir = tmp / "tactile_dataset"
    dataset_dir.mkdir()

    try:
        print(f"\nGenerating {len(materials)} recordings...")
        for i, (material, category, itype, speed, force) in enumerate(materials):
            data = make_tactile_recording(
                material=material,
                category=category,
                interaction_type=itype,
                speed_mm_s=speed,
                force_N=force,
                num_frames=20,
                seed=i,
            )

            # Save with a descriptive filename
            filename = f"{material}_{itype}_{int(speed)}mms_{force}N.hapt"
            save(data, dataset_dir / filename)
            print(f"   [{i+1}/{len(materials)}] {material:25s} {itype:10s} {speed:3.0f} mm/s  {force:3.1f} N")

        # --- Summary ---
        print(f"\nDataset created at: {dataset_dir}")
        hapt_dirs = sorted(dataset_dir.glob("*.hapt"))
        print(f"Total files: {len(hapt_dirs)}")
        total_size = sum(
            sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            for d in hapt_dirs
        )
        print(f"Total size: {total_size / 1024:.1f} KB")

        # --- Verify round-trip on first file ---
        from haptix import load

        first = load(hapt_dirs[0])
        print(f"\nVerification: {first}")
        assert first.raw.verify()
        print("✅ Checksum verified for first file")

        print("\n" + "=" * 60)
        print("Dataset creation complete!")
        print("=" * 60)

    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
