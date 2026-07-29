#!/usr/bin/env python3
"""Basic usage of haptix: load, inspect, save, and round-trip."""

import shutil
import tempfile
from pathlib import Path

import numpy as np

# === Import ===
import haptix
from haptix.core import (
    HaptData,
    InteractionMeta,
    Labels,
    RawData,
    SensorMeta,
)


def create_example_data() -> HaptData:
    """Create synthetic DIGIT-like tactile data."""
    # Simulate 10 frames of 240x320 RGB from a DIGIT sensor
    frames = np.random.randint(0, 255, (10, 240, 320, 3), dtype=np.uint8)

    return HaptData(
        raw=RawData(
            array=frames,
            checksum=RawData.compute_checksum(frames),
            dtype="uint8",
            shape=frames.shape,
        ),
        sensor=SensorMeta(
            type="DIGIT_v2",
            serial="SN-DIGIT-001",
            calibration_date="2026-07-01",
        ),
        modality="imaging",
        sampling_rate_hz=60.0,
        interaction=InteractionMeta(
            type="sliding",
            speed_mm_s=50.0,
            normal_force_N=2.0,
            approach_angle_deg=90.0,
            temperature_C=23.0,
            humidity_pct=45.0,
        ),
        labels=Labels(
            material="sandpaper_grit_80",
            material_category="abrasive",
            task="sliding",
            custom_tags=["rough", "high_friction"],
        ),
    )


def main():
    print("=" * 60)
    print("haptix — Tactile Data Infrastructure")
    print("=" * 60)

    # --- Create data ---
    print("\n1. Creating synthetic tactile data...")
    data = create_example_data()
    print(f"   {data}")
    print(f"   Raw shape: {data.raw.shape}")
    print(f"   Raw dtype: {data.raw.dtype}")
    print(f"   Sampling rate: {data.sampling_rate_hz} Hz")

    # --- Access metadata ---
    print("\n2. Accessing metadata...")
    print(f"   Sensor type: {data.sensor.type}")
    print(f"   Serial: {data.sensor.serial}")
    print(f"   Interaction type: {data.interaction.type}")
    print(f"   Speed: {data.interaction.speed_mm_s} mm/s")
    print(f"   Force: {data.interaction.normal_force_N} N")
    print(f"   Material: {data.labels.material}")
    print(f"   Category: {data.labels.material_category}")

    # --- Checksum verification ---
    print("\n3. Verifying data integrity...")
    print(f"   Computed checksum: {data.raw.checksum[:16]}...")
    assert data.raw.verify()
    print("   ✅ Checksum verified")

    # --- Numpy access ---
    print("\n4. Accessing raw data as NumPy...")
    frames = data.raw.numpy()
    print(f"   Frame 0 shape: {frames[0].shape}")
    print(f"   Frame 0 pixel (0,0): {frames[0, 0, 0]}")

    # --- Round-trip ---
    print("\n5. Round-trip: save → load → verify...")
    tmp = Path(tempfile.mkdtemp())
    try:
        saved_path = haptix.save(data, tmp / "test.hapt")
        reloaded = haptix.load(saved_path)
        assert np.array_equal(reloaded.raw.array, data.raw.array)
        assert reloaded.raw.checksum == data.raw.checksum
        assert reloaded.sensor.type == data.sensor.type
        assert reloaded.interaction.speed_mm_s == data.interaction.speed_mm_s
        assert reloaded.labels.material == data.labels.material
        print(f"   ✅ Save path: {saved_path}")
        print(f"   ✅ Loaded: {reloaded}")
        print("   ✅ Round-trip successful — byte-level identical")
    finally:
        shutil.rmtree(tmp)

    # --- List sensors ---
    print(f"\n6. Available sensors: {haptix.list_sensors()}")

    print("\n" + "=" * 60)
    print("All examples passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
