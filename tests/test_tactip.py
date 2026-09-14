"""
Tests for TacTip adapter (tactip.py).
"""

import csv
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from haptix.core import InteractionMeta, Labels, SensorMeta
from haptix.io import load, save
from haptix.sensors.tactip import TacTipAdapter


@pytest.fixture
def tactip_image_dir() -> Path:
    """Create a directory of synthetic TacTip-like pin images."""
    tmp = Path(tempfile.mkdtemp())
    n_frames = 10
    h, w = 240, 320

    rng = np.random.RandomState(42)
    for i in range(n_frames):
        # TacTip images show pins as bright dots on dark background
        img = np.zeros((h, w), dtype=np.uint8)
        # Add noise background
        img[:] = rng.randint(10, 30, (h, w)).astype(np.uint8)
        # Add 100+ bright pin dots
        for _ in range(127):
            px, py = rng.randint(20, w - 20), rng.randint(20, h - 20)
            img[py - 1 : py + 2, px - 1 : px + 2] = 255
        Image.fromarray(img).save(tmp / f"frame_{i:04d}.png")

    yield tmp
    shutil.rmtree(tmp)


@pytest.fixture
def tactip_markers_csv() -> Path:
    """Create a TacTip pin positions CSV file."""
    tmp = Path(tempfile.mkdtemp())
    n_frames = 20
    n_pins = 127
    cols = []
    for p in range(n_pins):
        cols.append(f"pin_{p}_x")
        cols.append(f"pin_{p}_y")

    rng = np.random.RandomState(77)
    data = rng.randn(n_frames, len(cols)) * 5 + 100

    csv_path = tmp / "tactip_pins.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in data:
            writer.writerow([f"{v:.2f}" for v in row])

    yield csv_path
    shutil.rmtree(tmp)


class TestTacTipAdapter:
    def test_instance_attributes(self):
        adapter = TacTipAdapter()
        assert adapter.sensor_type == "TacTip"

    def test_can_load_image_dir(self, tactip_image_dir):
        adapter = TacTipAdapter()
        assert adapter.can_load(tactip_image_dir)

    def test_can_load_markers_csv(self, tactip_markers_csv):
        adapter = TacTipAdapter()
        assert adapter.can_load(tactip_markers_csv)

    def test_can_load_rejects_non_tactip(self):
        adapter = TacTipAdapter()
        assert not adapter.can_load(Path("/tmp/nonexistent.png"))

    def test_load_image_mode(self, tactip_image_dir):
        adapter = TacTipAdapter()
        data = adapter.load(
            tactip_image_dir,
            interaction=InteractionMeta(type="pressing", normal_force_N=2.0),
            labels=Labels(material="foam"),
        )
        assert data.raw.shape == (10, 240, 320, 1)
        assert data.raw.dtype == "uint8"
        assert data.raw.verify()
        assert data.sensor.type == "TacTip"
        assert data.modality == "imaging"
        assert data.sampling_rate_hz == 30.0

    def test_load_marker_mode(self, tactip_markers_csv):
        adapter = TacTipAdapter()
        data = adapter.load(
            tactip_markers_csv,
            interaction=InteractionMeta(type="sliding", speed_mm_s=30),
            labels=Labels(material="plastic"),
        )
        assert data.raw.shape == (20, 254)  # 127 pins × 2 coords
        assert data.raw.dtype == "float32"
        assert data.raw.verify()
        assert data.sensor.type == "TacTip"
        assert data.modality == "dynamic"

    def test_load_force_markers_mode(self, tactip_markers_csv):
        adapter = TacTipAdapter()
        data = adapter.load(
            tactip_markers_csv,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(),
            mode="markers",
        )
        assert data.modality == "dynamic"

    def test_load_custom_sensor_meta(self, tactip_image_dir):
        adapter = TacTipAdapter()
        data = adapter.load(
            tactip_image_dir,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(),
            sensor_meta=SensorMeta(type="TacTip_v2", serial="TT-001"),
        )
        assert data.sensor.type == "TacTip_v2"
        assert data.sensor.serial == "TT-001"

    def test_load_custom_sampling_rate(self, tactip_image_dir):
        adapter = TacTipAdapter()
        data = adapter.load(
            tactip_image_dir,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(),
            sampling_rate_hz=60.0,
        )
        assert data.sampling_rate_hz == 60.0

    def test_roundtrip_image_mode(self, tactip_image_dir):
        """Save as .hapt and reload — verify checksum match."""
        adapter = TacTipAdapter()
        original = adapter.load(
            tactip_image_dir,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(material="test"),
        )

        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "test.hapt")
            reloaded = load(saved)
            assert np.array_equal(reloaded.raw.array, original.raw.array)
            assert reloaded.raw.checksum == original.raw.checksum
            assert reloaded.sensor.type == original.sensor.type
        finally:
            shutil.rmtree(tmp)

    def test_roundtrip_marker_mode(self, tactip_markers_csv):
        """Round-trip for marker CSV data."""
        adapter = TacTipAdapter()
        original = adapter.load(
            tactip_markers_csv,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(material="test"),
        )

        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "test.hapt")
            reloaded = load(saved)
            assert np.array_equal(reloaded.raw.array, original.raw.array)
            assert reloaded.raw.checksum == original.raw.checksum
        finally:
            shutil.rmtree(tmp)

    def test_empty_dir_raises(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            adapter = TacTipAdapter()
            with pytest.raises(FileNotFoundError):
                adapter.load(
                    tmp,
                    interaction=InteractionMeta(type="pressing"),
                    labels=Labels(),
                )
        finally:
            shutil.rmtree(tmp)

    def test_can_load_unreadable_csv_returns_false(self, tmp_path):
        """Binary garbage CSV cannot be decoded as text → can_load is False."""
        csv_path = tmp_path / "broken.csv"
        csv_path.write_bytes(b"\xff\xfe\x00\x01not-utf8")
        adapter = TacTipAdapter()
        assert adapter.can_load(csv_path) is False

    def test_load_images_rejects_non_directory(self, tmp_path):
        """Image mode requires a directory; a file path raises FileNotFoundError."""
        file_path = tmp_path / "not_a_dir.png"
        Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(file_path)
        adapter = TacTipAdapter()
        with pytest.raises(FileNotFoundError, match="Not a directory"):
            adapter.load(
                file_path,
                interaction=InteractionMeta(type="pressing"),
                labels=Labels(),
                mode="image",
            )

    def test_load_images_resizes_mixed_resolutions(self, tmp_path):
        """Frames with different HxW are resized to the first frame's shape."""
        Image.fromarray(np.full((20, 30), 10, dtype=np.uint8)).save(tmp_path / "a_ref.png")
        Image.fromarray(np.full((40, 60), 200, dtype=np.uint8)).save(tmp_path / "b_big.png")

        adapter = TacTipAdapter()
        data = adapter.load(
            tmp_path,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(),
        )
        assert data.raw.shape == (2, 20, 30, 1)
        assert data.raw.array[0].shape == (20, 30, 1)
        # Resized frame should retain elevated intensity from the bright source
        assert float(data.raw.array[1].mean()) > 100.0

    def test_load_images_promotes_gray_to_rgb_when_ref_is_rgb(self, tmp_path):
        """Gray follow-on frame is np.repeat'ed to 3 channels to match RGB ref."""
        rgb = np.zeros((16, 24, 3), dtype=np.uint8)
        rgb[:, :, 0] = 50
        Image.fromarray(rgb).save(tmp_path / "a_rgb.png")
        Image.fromarray(np.full((32, 48), 80, dtype=np.uint8)).save(tmp_path / "b_gray.png")

        adapter = TacTipAdapter()
        data = adapter.load(
            tmp_path,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(),
        )
        assert data.raw.shape == (2, 16, 24, 3)
        # Gray→RGB repeat: all three channels equal after resize+repeat
        assert np.array_equal(data.raw.array[1, :, :, 0], data.raw.array[1, :, :, 1])
        assert np.array_equal(data.raw.array[1, :, :, 1], data.raw.array[1, :, :, 2])

    def test_load_images_collapses_rgb_to_gray_when_ref_is_gray(self, tmp_path):
        """RGB follow-on frame is mean-collapsed to 1 channel to match gray ref."""
        Image.fromarray(np.full((16, 24), 30, dtype=np.uint8)).save(tmp_path / "a_gray.png")
        rgb = np.zeros((32, 48, 3), dtype=np.uint8)
        rgb[:, :, 0] = 90
        rgb[:, :, 1] = 60
        rgb[:, :, 2] = 30
        Image.fromarray(rgb).save(tmp_path / "b_rgb.png")

        adapter = TacTipAdapter()
        data = adapter.load(
            tmp_path,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(),
        )
        assert data.raw.shape == (2, 16, 24, 1)
        # Mean of (90, 60, 30) = 60 before resize; after resize should stay near that
        assert 40.0 < float(data.raw.array[1].mean()) < 80.0

    def test_load_markers_empty_csv_raises(self, tmp_path):
        """Empty markers CSV raises ValueError mentioning Empty CSV."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        adapter = TacTipAdapter()
        with pytest.raises(ValueError, match="Empty CSV"):
            adapter.load(
                csv_path,
                interaction=InteractionMeta(type="pressing"),
                labels=Labels(),
                mode="markers",
            )

    def test_load_markers_skips_non_numeric_rows(self, tmp_path):
        """Non-numeric rows are skipped; remaining floats form the array."""
        csv_path = tmp_path / "mixed.csv"
        # No pin header → treated as data rows; junk lines skipped
        csv_path.write_text("1.0,2.0,3.0,4.0\nbad,row,here,x\n5.0,6.0,7.0,8.0\n")
        adapter = TacTipAdapter()
        data = adapter.load(
            csv_path,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(),
            mode="markers",
        )
        assert data.raw.shape == (2, 4)
        np.testing.assert_array_equal(
            data.raw.array, np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        )

    def test_load_markers_no_numeric_rows_raises(self, tmp_path):
        """CSV with only non-numeric content raises ValueError."""
        csv_path = tmp_path / "junk.csv"
        csv_path.write_text("foo,bar,baz\nabc,def,ghi\n")
        adapter = TacTipAdapter()
        with pytest.raises(ValueError, match="No valid numeric data"):
            adapter.load(
                csv_path,
                interaction=InteractionMeta(type="pressing"),
                labels=Labels(),
                mode="markers",
            )
