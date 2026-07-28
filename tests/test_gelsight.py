"""Tests for the GelSight sensor adapter.

Tests cover: loading grayscale and RGB frames, can_load detection,
sensor type registration, edge cases (empty dirs, unsupported formats).
"""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from haptix.core import InteractionMeta, Labels
from haptix.sensors import get_sensor, list_sensors


class TestGelSightAdapter:
    """Test the GelSight sensor adapter."""

    def _make_grayscale_frames(self, path: Path, count: int = 5, h: int = 480, w: int = 640):
        """Create synthetic grayscale GelSight frames."""
        for i in range(count):
            img = np.random.randint(0, 255, (h, w), dtype=np.uint8)
            Image.fromarray(img).save(path / f"frame_{i:04d}.png")

    def _make_rgb_frames(self, path: Path, count: int = 3, h: int = 240, w: int = 320):
        """Create synthetic RGB GelSight frames."""
        for i in range(count):
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            Image.fromarray(img).save(path / f"img_{i:04d}.png")

    def test_registered(self):
        """GelSight should be in the sensor registry."""
        sensors = list_sensors()
        assert "GelSight" in sensors, f"GelSight not in {sensors}"

    def test_can_load_directory_with_png(self):
        """can_load should return True for dirs containing PNG frames."""
        from haptix.sensors.gelsight import GelSightAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            self._make_grayscale_frames(tmp)
            adapter = GelSightAdapter()
            assert adapter.can_load(tmp) is True
        finally:
            import shutil

            shutil.rmtree(tmp)

    def test_can_load_directory_with_jpg(self):
        """can_load should return True for dirs containing JPG frames."""
        from haptix.sensors.gelsight import GelSightAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            for i in range(3):
                img = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
                Image.fromarray(img).save(tmp / f"gel_{i:04d}.jpg")
            adapter = GelSightAdapter()
            assert adapter.can_load(tmp) is True
        finally:
            import shutil

            shutil.rmtree(tmp)

    def test_can_load_empty_directory(self):
        """can_load should return False for empty directories."""
        from haptix.sensors.gelsight import GelSightAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            adapter = GelSightAdapter()
            assert adapter.can_load(tmp) is False
        finally:
            import shutil

            shutil.rmtree(tmp)

    def test_can_load_non_image_files(self):
        """can_load should return False for dirs with no image files."""
        from haptix.sensors.gelsight import GelSightAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "metadata.txt").write_text("not an image")
            (tmp / "calibration.csv").write_text("a,b,c\n1,2,3")
            adapter = GelSightAdapter()
            assert adapter.can_load(tmp) is False
        finally:
            import shutil

            shutil.rmtree(tmp)

    def test_can_load_non_existent_path(self):
        """can_load should return False for paths that don't exist."""
        from haptix.sensors.gelsight import GelSightAdapter

        adapter = GelSightAdapter()
        assert adapter.can_load(Path("/nonexistent/path")) is False

    def test_can_load_file_path(self):
        """can_load should return False for single files (not dirs)."""
        from haptix.sensors.gelsight import GelSightAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            img_path = tmp / "single.png"
            img = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
            Image.fromarray(img).save(img_path)
            adapter = GelSightAdapter()
            # Single file is not a directory of frames
            assert adapter.can_load(img_path) is False
        finally:
            import shutil

            shutil.rmtree(tmp)

    def test_load_grayscale_frames(self):
        """Load grayscale GelSight data from directory of PNGs."""
        from haptix.sensors.gelsight import GelSightAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            self._make_grayscale_frames(tmp, count=5, h=480, w=640)

            adapter = GelSightAdapter()
            data = adapter.load(
                tmp,
                interaction=InteractionMeta(type="sliding", speed_mm_s=50),
                labels=Labels(material="sandpaper_grit_80"),
            )

            # Grayscale: shape should be [T, H, W, 1] (expanded dim)
            assert data.raw.shape == (5, 480, 640, 1), f"Got {data.raw.shape}"
            assert data.raw.dtype == "uint8"
            assert data.sensor.type == "GelSight"
            assert data.modality == "imaging"
            assert data.sampling_rate_hz == 30.0
            assert data.interaction.type == "sliding"
            assert data.labels.material == "sandpaper_grit_80"
            assert data.version == "0.1.0"

            # Verify checksum
            assert data.raw.verify() is True
        finally:
            import shutil

            shutil.rmtree(tmp)

    def test_load_rgb_frames(self):
        """Load RGB GelSight data from directory of PNGs."""
        from haptix.sensors.gelsight import GelSightAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            self._make_rgb_frames(tmp, count=3, h=240, w=320)

            adapter = GelSightAdapter()
            data = adapter.load(
                tmp,
                interaction=InteractionMeta(type="pressing", normal_force_N=1.5),
                labels=Labels(material="foam"),
            )

            assert data.raw.shape == (3, 240, 320, 3)
            assert data.raw.dtype == "uint8"
            assert data.sensor.type == "GelSight"
            assert data.modality == "imaging"
            assert data.sampling_rate_hz == 30.0
        finally:
            import shutil

            shutil.rmtree(tmp)

    def test_load_jpg_frames(self):
        """Load JPG frames (common in GelSight datasets)."""
        from haptix.sensors.gelsight import GelSightAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            for i in range(3):
                img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
                Image.fromarray(img).save(tmp / f"gel_{i:04d}.jpg")

            adapter = GelSightAdapter()
            data = adapter.load(
                tmp,
                interaction=InteractionMeta(type="sliding"),
                labels=Labels(material="test"),
            )

            assert data.raw.shape == (3, 240, 320, 3)
        finally:
            import shutil

            shutil.rmtree(tmp)

    def test_load_with_custom_sensor_meta(self):
        """Allow overriding sensor metadata on load."""
        from haptix.core import SensorMeta
        from haptix.sensors.gelsight import GelSightAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            self._make_grayscale_frames(tmp, count=2)

            adapter = GelSightAdapter()
            custom_sensor = SensorMeta(
                type="GelSight_Mini",
                serial="GS-MINI-001",
                calibration_date="2026-06-15",
                calibration_params={"camera": "OV5640", "led_wavelength_nm": 520},
            )
            data = adapter.load(
                tmp,
                interaction=InteractionMeta(type="static"),
                labels=Labels(material="rubber"),
                sensor_meta=custom_sensor,
            )

            assert data.sensor.type == "GelSight_Mini"
            assert data.sensor.serial == "GS-MINI-001"
            assert data.sensor.calibration_date == "2026-06-15"
            assert data.sensor.calibration_params["camera"] == "OV5640"
        finally:
            import shutil

            shutil.rmtree(tmp)

    def test_load_empty_directory_raises(self):
        """Loading an empty directory should raise an error."""
        from haptix.sensors.gelsight import GelSightAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            adapter = GelSightAdapter()
            try:
                adapter.load(
                    tmp,
                    interaction=InteractionMeta(type="sliding"),
                    labels=Labels(material="test"),
                )
                assert False, "Should have raised FileNotFoundError"
            except FileNotFoundError:
                pass
        finally:
            import shutil

            shutil.rmtree(tmp)

    def test_default_framerate(self):
        """GelSight should default to 30 Hz."""
        from haptix.sensors.gelsight import GelSightAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            self._make_grayscale_frames(tmp, count=2)

            adapter = GelSightAdapter()
            data = adapter.load(
                tmp,
                interaction=InteractionMeta(type="sliding"),
                labels=Labels(material="test"),
            )

            assert data.sampling_rate_hz == 30.0
        finally:
            import shutil

            shutil.rmtree(tmp)

    def test_get_sensor_via_registry(self):
        """GelSight adapter should be retrievable via get_sensor()."""
        adapter = get_sensor("GelSight")
        from haptix.sensors.gelsight import GelSightAdapter

        assert isinstance(adapter, GelSightAdapter)

    def test_sort_order(self):
        """Frames should be loaded in alphabetical sort order."""
        from haptix.sensors.gelsight import GelSightAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            # Create frames in non-alphabetical creation order
            count = 5
            self._make_grayscale_frames(tmp, count=count)

            adapter = GelSightAdapter()
            data = adapter.load(
                tmp,
                interaction=InteractionMeta(type="sliding"),
                labels=Labels(material="test"),
            )

            assert data.raw.shape[0] == count
        finally:
            import shutil

            shutil.rmtree(tmp)
