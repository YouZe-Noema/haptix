#!/usr/bin/env python3
"""Using the DIGIT sensor adapter to convert native formats to .hapt."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from haptix import get_sensor, save
from haptix.core import InteractionMeta, Labels


def create_synthetic_digit_frames(tmp: Path, num_frames: int = 5):
    """Generate synthetic DIGIT image frames."""
    frames_dir = tmp / "digit_session"
    frames_dir.mkdir()
    for i in range(num_frames):
        img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        Image.fromarray(img).save(frames_dir / f"frame_{i:04d}.png")
    return frames_dir


def main():
    print("=" * 60)
    print("DIGIT Sensor Adapter Example")
    print("=" * 60)

    tmp = Path(tempfile.mkdtemp())
    try:
        # --- Step 1: Create synthetic DIGIT frames ---
        print("\n1. Creating synthetic DIGIT frames...")
        frames_dir = create_synthetic_digit_frames(tmp)
        print(f"   Frames directory: {frames_dir}")
        pngs = sorted(frames_dir.glob("*.png"))
        print(f"   {len(pngs)} PNG files created")

        # --- Step 2: Get the adapter ---
        print("\n2. Getting DIGIT adapter...")
        adapter = get_sensor("DIGIT")
        print(f"   Adapter type: {type(adapter).__name__}")
        print(f"   Sensor type: {adapter.sensor_type}")

        # --- Step 3: Check if it can load ---
        print(f"\n3. can_load? {adapter.can_load(frames_dir)}")

        # --- Step 4: Load the frames ---
        print("\n4. Loading DIGIT frames...")
        data = adapter.load(
            frames_dir,
            interaction=InteractionMeta(
                type="sliding",
                speed_mm_s=50.0,
                normal_force_N=2.0,
            ),
            labels=Labels(
                material="sandpaper_grit_120",
                material_category="abrasive",
                task="sliding",
            ),
        )
        print(f"   Loaded: {data}")
        print(f"   Shape: {data.raw.shape}")
        print(f"   Dtype: {data.raw.dtype}")
        print(f"   Frames: {data.raw.shape[0]}")
        print(f"   Frame size: {data.raw.shape[1]}x{data.raw.shape[2]}")
        print(f"   Channels: {data.raw.shape[3]}")
        print(f"   Framerate: {data.sampling_rate_hz} Hz")

        # --- Step 5: Save as .hapt ---
        print("\n5. Saving as .hapt...")
        output_path = tmp / "outputs" / "sandpaper_120.hapt"
        saved = save(data, output_path)
        print(f"   Saved to: {saved}")

        # Verify on-disk structure
        manifest = saved / "manifest.json"
        labels_file = saved / "labels.json"
        raw_dir = saved / "raw"
        print(f"   manifest.json present: {manifest.exists()}")
        print(f"   labels.json present: {labels_file.exists()}")
        print(f"   raw/ directory present: {raw_dir.exists()}")
        print(f"   raw/data.npy present: {(raw_dir / 'data.npy').exists()}")
        print(f"   raw/checksum.sha256 present: {(raw_dir / 'checksum.sha256').exists()}")

        print("\n" + "=" * 60)
        print("DIGIT example completed successfully!")
        print("=" * 60)

    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
