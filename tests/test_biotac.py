"""
Tests for BioTac adapter (biotac.py).
"""

import csv
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from haptix.core import InteractionMeta, Labels, SensorMeta
from haptix.io import load, save
from haptix.sensors.biotac import BioTacAdapter


@pytest.fixture
def biotac_csv() -> Path:
    """Create a minimal BioTac CSV file with header."""
    tmp = Path(tempfile.mkdtemp())
    n_rows = 50
    n_electrodes = 19
    cols = [f"E{i}" for i in range(1, n_electrodes + 1)]
    cols += ["PDC", "PAC", "TDC", "TAC"]

    rng = np.random.RandomState(42)
    data = rng.randn(n_rows, len(cols)) * 100 + 2000

    csv_path = tmp / "biotac_test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in data:
            writer.writerow([f"{v:.2f}" for v in row])

    yield csv_path
    shutil.rmtree(tmp)


@pytest.fixture
def biotac_csv_no_header() -> Path:
    """Create a BioTac CSV without a header row."""
    tmp = Path(tempfile.mkdtemp())
    n_rows = 30
    n_cols = 23  # 19 electrodes + PDC + PAC + TDC + TAC

    rng = np.random.RandomState(99)
    data = rng.randn(n_rows, n_cols) * 50 + 1500

    csv_path = tmp / "biotac_no_header.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow([f"{v:.2f}" for v in row])

    yield csv_path
    shutil.rmtree(tmp)


@pytest.fixture
def biotac_csv_with_timestamp(biotac_csv: Path) -> Path:
    """BioTac CSV with a timestamp column in position 0."""
    import csv

    tmp = biotac_csv.parent

    # Read existing data, prepend timestamp
    rows = []
    with open(biotac_csv, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = ["timestamp"] + header
        rows.append(header)
        for i, row in enumerate(reader):
            rows.append([f"{i * 0.01:.4f}"] + row)

    csv_path = tmp / "biotac_timestamp.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    return csv_path


class TestBioTacAdapter:
    def test_instance_attributes(self):
        adapter = BioTacAdapter()
        assert adapter.sensor_type == "BioTac"

    def test_can_load_csv_with_header(self, biotac_csv):
        adapter = BioTacAdapter()
        assert adapter.can_load(biotac_csv)

    def test_can_load_rejects_non_csv(self):
        adapter = BioTacAdapter()
        assert not adapter.can_load(Path("/tmp/not_a_csv.png"))
        assert not adapter.can_load(Path("/tmp/nonexistent.csv"))

    def test_can_load_rejects_directory(self):
        adapter = BioTacAdapter()
        assert not adapter.can_load(Path(tempfile.mkdtemp()))

    def test_load_with_header(self, biotac_csv):
        adapter = BioTacAdapter()
        data = adapter.load(
            biotac_csv,
            interaction=InteractionMeta(type="pressing", normal_force_N=1.5),
            labels=Labels(material="aluminum"),
        )
        assert data.raw.shape == (50, 23)
        assert data.raw.dtype == "float32"
        assert data.raw.verify()
        assert data.sensor.type == "BioTac_SP"
        assert data.modality == "dynamic"
        assert data.sampling_rate_hz == 100.0
        assert data.interaction.type == "pressing"

    def test_load_no_header(self, biotac_csv_no_header):
        adapter = BioTacAdapter()
        data = adapter.load(
            biotac_csv_no_header,
            interaction=InteractionMeta(type="sliding", speed_mm_s=50),
            labels=Labels(material="sandpaper"),
        )
        # Without header, auto-detects columns
        assert data.raw.shape[0] == 30
        assert data.raw.dtype == "float32"
        assert data.raw.verify()

    def test_load_with_timestamp(self, biotac_csv_with_timestamp):
        adapter = BioTacAdapter()
        data = adapter.load(
            biotac_csv_with_timestamp,
            interaction=InteractionMeta(type="static"),
            labels=Labels(task="calibration"),
        )
        # Timestamp column should be stripped
        assert data.raw.shape == (50, 23)  # same as original without timestamp
        assert data.raw.verify()

    def test_load_custom_sensor_meta(self, biotac_csv):
        adapter = BioTacAdapter()
        data = adapter.load(
            biotac_csv,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(),
            sensor_meta=SensorMeta(
                type="BioTac_Custom",
                serial="BT-001",
                calibration_date="2025-01-15",
            ),
        )
        assert data.sensor.type == "BioTac_Custom"
        assert data.sensor.serial == "BT-001"
        assert data.sensor.calibration_date == "2025-01-15"

    def test_load_custom_sampling_rate(self, biotac_csv):
        adapter = BioTacAdapter()
        data = adapter.load(
            biotac_csv,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(),
            sampling_rate_hz=50.0,
        )
        assert data.sampling_rate_hz == 50.0

    def test_roundtrip(self, biotac_csv):
        """Save as .hapt and reload — verify checksum match."""
        adapter = BioTacAdapter()
        original = adapter.load(
            biotac_csv,
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

    def test_empty_csv_raises(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            csv_path = tmp / "empty.csv"
            csv_path.write_text("")
            adapter = BioTacAdapter()
            with pytest.raises(ValueError):
                adapter.load(
                    csv_path,
                    interaction=InteractionMeta(type="pressing"),
                    labels=Labels(),
                )
        finally:
            shutil.rmtree(tmp)
