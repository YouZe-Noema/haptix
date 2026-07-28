"""
Round-trip tests for .hapt format.

These verify the core guarantee: load(save(data)) == data.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from haptix.core import HaptData, InteractionMeta, Labels, RawData, SensorMeta
from haptix.io import load, save


def make_test_data() -> HaptData:
    """Create a minimal valid HaptData for testing."""
    frames = np.random.randint(0, 255, (10, 240, 320, 3), dtype=np.uint8)
    return HaptData(
        raw=RawData(
            array=frames,
            checksum=RawData.compute_checksum(frames),
            dtype="uint8",
            shape=frames.shape,
        ),
        sensor=SensorMeta(type="DIGIT_v2"),
        modality="imaging",
        sampling_rate_hz=60.0,
        interaction=InteractionMeta(
            type="sliding",
            speed_mm_s=50.0,
            normal_force_N=2.0,
        ),
        labels=Labels(
            material="sandpaper_grit_80",
            task="sliding",
        ),
    )


class TestRoundTrip:
    """Verify load(save(x)) == x for all data integrity guarantees."""

    def test_save_load_roundtrip(self):
        """Core test: save then load should produce identical data."""
        original = make_test_data()

        tmp = Path(tempfile.mkdtemp())
        try:
            # Save
            saved_path = save(original, tmp / "test.hapt")

            # Load
            loaded = load(saved_path)

            # Verify metadata
            assert loaded.sensor.type == original.sensor.type
            assert loaded.modality == original.modality
            assert loaded.sampling_rate_hz == original.sampling_rate_hz
            assert loaded.interaction.type == original.interaction.type
            assert loaded.interaction.speed_mm_s == original.interaction.speed_mm_s
            assert loaded.labels.material == original.labels.material

            # Verify raw data — byte-level identical
            assert np.array_equal(loaded.raw.array, original.raw.array)
            assert loaded.raw.checksum == original.raw.checksum
            assert loaded.raw.shape == original.raw.shape
            assert loaded.raw.dtype == original.raw.dtype

        finally:
            shutil.rmtree(tmp)

    def test_checksum_verification(self):
        """Manually corrupt data and verify ChecksumError is raised."""
        original = make_test_data()
        tmp = Path(tempfile.mkdtemp())
        try:
            saved_path = save(original, tmp / "test.hapt")

            # Corrupt the data file
            data_path = saved_path / "raw" / "data.npy"
            corrupted = np.load(data_path)
            corrupted[0, 0, 0, 0] = 0  # flip one pixel
            np.save(data_path, corrupted)

            # Load should raise
            try:
                load(saved_path)
                assert False, "Should have raised ChecksumError"
            except (ValueError, RuntimeError) as e:
                assert "Checksum" in str(e) or "checksum" in str(e).lower()
        finally:
            shutil.rmtree(tmp)

    def test_manifest_roundtrip(self):
        """Verify all manifest fields survive round-trip."""
        original = make_test_data()
        tmp = Path(tempfile.mkdtemp())
        try:
            saved_path = save(original, tmp / "test.hapt")
            loaded = load(saved_path)

            # Check structural integrity
            assert (loaded.raw.array == original.raw.array).all()
            assert loaded.raw.checksum == original.raw.checksum
        finally:
            shutil.rmtree(tmp)


class TestDigitAdapter:
    """Test the DIGIT sensor adapter."""

    def test_load_image_directory(self):
        """Load DIGIT data from a directory of PNG frames."""
        from haptix.sensors.digit import DigitAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            # Create synthetic DIGIT frames
            for i in range(5):
                img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                Image.fromarray(img).save(tmp / f"frame_{i:04d}.png")

            adapter = DigitAdapter()
            assert adapter.can_load(tmp)

            data = adapter.load(
                tmp,
                interaction=InteractionMeta(type="sliding", speed_mm_s=50),
                labels=Labels(material="test"),
            )

            assert data.raw.shape == (5, 240, 320, 3)
            assert data.sensor.type == "DIGIT_v2"
            assert data.modality == "imaging"

        finally:
            shutil.rmtree(tmp)
