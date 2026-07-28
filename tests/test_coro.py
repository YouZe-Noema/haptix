"""Tests for the Lab-CORO capacitive tactile sensor adapter.

Tests cover: loading CSV pressure arrays, can_load detection,
sensor type registration, edge cases (empty dirs, missing CSV columns).
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from haptix.core import InteractionMeta, Labels, SensorMeta
from haptix.sensors import get_sensor, list_sensors

_NUM_TAXELS = 57  # Number of taxels in the Lab-CORO capacitive sensor


def _make_csv_file(path: Path, num_samples: int = 3, num_taxels: int = _NUM_TAXELS):
    """Create a synthetic Lab-CORO CSV file with pressure data.

    Real format: each Path group has 57 rows (one per taxel),
    with small number of data columns (pressure value + optional metadata).
    """
    import pandas as pd

    rows = []
    for sample_idx in range(num_samples):
        path_name = f"Square_Indenter_{sample_idx}"
        for taxel_idx in range(num_taxels):
            pressure = float(np.random.rand() * 100)
            # One pressure value per taxel row, plus optional metadata cols
            row = {
                "Path": path_name,
                "Pressure": pressure,
                "X": float(taxel_idx % 8),  # grid position
                "Y": float(taxel_idx // 8),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path / "Flat_Real_Abaqus.csv", index=False)
    df.to_csv(path / "Flat_Simulation_Abaqus.csv", index=False)
    return path


def _make_csv_file_alt_format(path: Path, num_samples: int = 2, num_taxels: int = _NUM_TAXELS):
    """Create CSV with one-row-per-sample format (taxels as columns)."""
    import pandas as pd

    rows = []
    for sample_idx in range(num_samples):
        path_name = f"Object_{sample_idx}"
        row = {"Path": path_name}
        for t in range(num_taxels):
            row[f"t{t}"] = float(np.random.rand() * 100)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path / "Curved_Real_Abaqus.csv", index=False)
    return path


class TestCoroCapacitiveAdapter:
    """Test the CoroCapacitive sensor adapter."""

    def test_registered(self):
        """CoroCapacitive should be in the sensor registry."""
        sensors = list_sensors()
        assert "CoroCapacitive" in sensors, f"CoroCapacitive not in {sensors}"

    def test_can_load_directory_with_csv(self):
        """can_load should return True for dirs containing CSV files."""
        from haptix.sensors.coro import CoroCapacitiveAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            _make_csv_file(tmp)
            adapter = CoroCapacitiveAdapter()
            assert adapter.can_load(tmp) is True
        finally:
            shutil.rmtree(tmp)

    def test_can_load_empty_directory(self):
        """can_load should return False for empty directories."""
        from haptix.sensors.coro import CoroCapacitiveAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            adapter = CoroCapacitiveAdapter()
            assert adapter.can_load(tmp) is False
        finally:
            shutil.rmtree(tmp)

    def test_can_load_non_csv_directory(self):
        """can_load should return False for dirs with no CSV files."""
        from haptix.sensors.coro import CoroCapacitiveAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "readme.txt").write_text("no csv here")
            (tmp / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            adapter = CoroCapacitiveAdapter()
            assert adapter.can_load(tmp) is False
        finally:
            shutil.rmtree(tmp)

    def test_can_load_nonexistent_path(self):
        """can_load should return False for paths that don't exist."""
        from haptix.sensors.coro import CoroCapacitiveAdapter

        adapter = CoroCapacitiveAdapter()
        assert adapter.can_load(Path("/nonexistent/path")) is False

    def test_load_flat_real_csv(self):
        """Load CoroCapacitive data from Flat_Real_Abaqus.csv."""
        from haptix.sensors.coro import CoroCapacitiveAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            _make_csv_file(tmp, num_samples=3, num_taxels=_NUM_TAXELS)

            adapter = CoroCapacitiveAdapter()
            data = adapter.load(
                tmp,
                interaction=InteractionMeta(type="pressing", normal_force_N=3.0),
                labels=Labels(material="indenter", object_name="Square_Indenter_0"),
            )

            # 3 samples × 57 taxels = 171 rows in the CSV
            # For Flat_Real_Abaqus with Path grouping, we expect [3, 57]
            assert data.raw.shape == (3, _NUM_TAXELS), f"Got {data.raw.shape}"
            assert data.raw.dtype.startswith("float")
            assert data.sensor.type == "CoroCapacitive"
            assert data.modality == "dynamic"
            assert data.sampling_rate_hz == 30.0
            assert data.interaction.type == "pressing"
            assert data.labels.material == "indenter"
            assert data.version == "0.1.0"
            assert data.raw.verify() is True
        finally:
            shutil.rmtree(tmp)

    def test_load_simulation_csv(self):
        """Load CoroCapacitive data from Flat_Simulation_Abaqus.csv."""
        from haptix.sensors.coro import CoroCapacitiveAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            _make_csv_file(tmp, num_samples=2)

            adapter = CoroCapacitiveAdapter()
            data = adapter.load(
                tmp,
                source="simulation",
                interaction=InteractionMeta(type="pressing", normal_force_N=5.0),
                labels=Labels(material="indenter"),
            )

            assert data.raw.shape[0] == 2
            assert data.raw.shape[1] == _NUM_TAXELS
            assert data.modality == "dynamic"
        finally:
            shutil.rmtree(tmp)

    def test_load_curved_csv(self):
        """Load CoroCapacitive data from Curved_Real_Abaqus.csv."""
        from haptix.sensors.coro import CoroCapacitiveAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            _make_csv_file_alt_format(tmp, num_samples=2)

            adapter = CoroCapacitiveAdapter()
            data = adapter.load(
                tmp,
                source="curved_real",
                interaction=InteractionMeta(type="grasping"),
                labels=Labels(object_name="Object_0"),
            )

            assert data.raw.shape[0] == 2
            assert data.raw.shape[1] == _NUM_TAXELS
        finally:
            shutil.rmtree(tmp)

    def test_load_with_custom_sensor_meta(self):
        """Allow overriding sensor metadata on load."""
        from haptix.sensors.coro import CoroCapacitiveAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            _make_csv_file(tmp, num_samples=1)

            adapter = CoroCapacitiveAdapter()
            custom_sensor = SensorMeta(
                type="CoroCapacitive",
                serial="CORO-2024-001",
                calibration_date="2024-06-15",
                calibration_params={"num_taxels": _NUM_TAXELS, "sensor_type": "capacitive"},
            )
            data = adapter.load(
                tmp,
                interaction=InteractionMeta(type="pressing"),
                labels=Labels(material="test"),
                sensor_meta=custom_sensor,
            )

            assert data.sensor.serial == "CORO-2024-001"
            assert data.sensor.calibration_date == "2024-06-15"
            assert data.sensor.calibration_params["num_taxels"] == _NUM_TAXELS
        finally:
            shutil.rmtree(tmp)

    def test_load_empty_directory_raises(self):
        """Loading an empty directory should raise an error."""
        from haptix.sensors.coro import CoroCapacitiveAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            adapter = CoroCapacitiveAdapter()
            with pytest.raises(FileNotFoundError):
                adapter.load(
                    tmp,
                    interaction=InteractionMeta(type="pressing"),
                    labels=Labels(material="test"),
                )
        finally:
            shutil.rmtree(tmp)

    def test_get_sensor_via_registry(self):
        """CoroCapacitive adapter should be retrievable via get_sensor()."""
        adapter = get_sensor("CoroCapacitive")
        from haptix.sensors.coro import CoroCapacitiveAdapter

        assert isinstance(adapter, CoroCapacitiveAdapter)

    def test_default_framerate(self):
        """CoroCapacitive should default to 30 Hz."""
        from haptix.sensors.coro import CoroCapacitiveAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            _make_csv_file(tmp, num_samples=1)

            adapter = CoroCapacitiveAdapter()
            data = adapter.load(
                tmp,
                interaction=InteractionMeta(type="pressing"),
                labels=Labels(material="test"),
            )

            assert data.sampling_rate_hz == 30.0
        finally:
            shutil.rmtree(tmp)

    def test_load_dynamic_modality_preserved(self):
        """Raw data should be [T, D] dynamic with correct modality."""
        from haptix.sensors.coro import CoroCapacitiveAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            _make_csv_file(tmp, num_samples=5, num_taxels=_NUM_TAXELS)

            adapter = CoroCapacitiveAdapter()
            data = adapter.load(
                tmp,
                interaction=InteractionMeta(type="grasping"),
                labels=Labels(material="aluminium"),
            )

            # Dynamic modality: shape[0] = time steps, shape[1] = taxels
            assert len(data.raw.shape) == 2, f"Expected 2D dynamic array, got {data.raw.shape}"
            assert data.raw.shape[0] == 5, f"Expected 5 time steps, got {data.raw.shape[0]}"
            assert data.raw.shape[1] == _NUM_TAXELS, f"Expected {_NUM_TAXELS} taxels"
            assert data.modality == "dynamic"
        finally:
            shutil.rmtree(tmp)
