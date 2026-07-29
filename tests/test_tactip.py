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
