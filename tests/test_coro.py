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

    def test_load_falls_back_to_first_csv_when_pattern_unmatched(self, tmp_path):
        """When source pattern matches no filename, use the first CSV present."""
        import pandas as pd

        from haptix.sensors.coro import CoroCapacitiveAdapter

        # Filename does not contain any known source pattern
        df = pd.DataFrame(
            {
                "Path": ["A", "A", "B", "B"],
                "Pressure": [1.0, 2.0, 3.0, 4.0],
                "X": [0.0, 1.0, 0.0, 1.0],
            }
        )
        df.to_csv(tmp_path / "orphan_readings.csv", index=False)

        adapter = CoroCapacitiveAdapter()
        data = adapter.load(
            tmp_path,
            source="flat_real",
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(material="test"),
        )
        # 2-9 cols → mean-collapse: 2 Path groups × 2 taxel rows each
        assert data.raw.shape == (2, 2)
        assert data.raw.dtype.startswith("float")
        np.testing.assert_allclose(data.raw.array[0], [0.5, 1.5], atol=1e-5)

    def test_extract_single_data_column_per_path_group(self, tmp_path):
        """Path + one numeric column: each row is one taxel reading."""
        import pandas as pd

        from haptix.sensors.coro import CoroCapacitiveAdapter

        rows = []
        for path_name, values in [("g0", [10.0, 20.0, 30.0]), ("g1", [40.0, 50.0, 60.0])]:
            for v in values:
                rows.append({"Path": path_name, "Pressure": v})
        pd.DataFrame(rows).to_csv(tmp_path / "Flat_Real_Abaqus.csv", index=False)

        adapter = CoroCapacitiveAdapter()
        data = adapter.load(
            tmp_path,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(),
        )
        assert data.raw.shape == (2, 3)
        np.testing.assert_array_equal(
            data.raw.array[0], np.array([10.0, 20.0, 30.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            data.raw.array[1], np.array([40.0, 50.0, 60.0], dtype=np.float32)
        )

    def test_extract_wide_frame_format_concatenates_rows(self, tmp_path):
        """10+ data columns: each row is a full frame; Path groups concatenate."""
        import pandas as pd

        from haptix.sensors.coro import CoroCapacitiveAdapter

        n_taxels = 12
        rows = []
        for path_name, offset in [("p0", 0.0), ("p1", 100.0)]:
            for r in range(2):
                row = {"Path": path_name}
                for t in range(n_taxels):
                    row[f"tax{t}"] = float(offset + r * 10 + t)
                rows.append(row)
        pd.DataFrame(rows).to_csv(tmp_path / "Flat_Real_Abaqus.csv", index=False)

        adapter = CoroCapacitiveAdapter()
        data = adapter.load(
            tmp_path,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(),
        )
        # 2 groups × 2 rows each → 4 frames of 12 taxels
        assert data.raw.shape == (4, n_taxels)
        np.testing.assert_array_equal(data.raw.array[0], np.arange(n_taxels, dtype=np.float32))
        np.testing.assert_array_equal(
            data.raw.array[2], np.arange(n_taxels, dtype=np.float32) + 100.0
        )

    def test_extract_pads_uneven_path_group_lengths(self, tmp_path):
        """1-D frames of unequal length are zero-padded to the longest group."""
        import pandas as pd

        from haptix.sensors.coro import CoroCapacitiveAdapter

        rows = [
            {"Path": "short", "Pressure": 1.0},
            {"Path": "short", "Pressure": 2.0},
            {"Path": "long", "Pressure": 3.0},
            {"Path": "long", "Pressure": 4.0},
            {"Path": "long", "Pressure": 5.0},
        ]
        pd.DataFrame(rows).to_csv(tmp_path / "Flat_Real_Abaqus.csv", index=False)

        adapter = CoroCapacitiveAdapter()
        data = adapter.load(
            tmp_path,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(),
        )
        assert data.raw.shape == (2, 3)
        # groupby(sort=True) orders Path names alphabetically: long, then short
        np.testing.assert_array_equal(
            data.raw.array[0], np.array([3.0, 4.0, 5.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            data.raw.array[1], np.array([1.0, 2.0, 0.0], dtype=np.float32)
        )

    def test_extract_empty_path_groups_returns_zero_frame(self):
        """Empty DataFrame with a Path column yields a single zeroed taxel row."""
        import pandas as pd

        from haptix.sensors.coro import CoroCapacitiveAdapter

        adapter = CoroCapacitiveAdapter()
        df = pd.DataFrame(columns=["Path", "Pressure"])
        result = adapter._extract_pressure_array(df)
        assert result.shape == (1, _NUM_TAXELS)
        assert result.dtype == np.float32
        assert np.all(result == 0.0)

    def test_extract_no_path_column_filters_constant_markers(self):
        """Without Path, drop constant marker columns and keep varying taxels."""
        import pandas as pd

        from haptix.sensors.coro import CoroCapacitiveAdapter

        adapter = CoroCapacitiveAdapter()
        # frame_id is constant → marker, not taxel data
        df = pd.DataFrame(
            {
                "frame_id": [0, 0, 0],
                "t0": [1.0, 2.0, 3.0],
                "t1": [4.0, 5.0, 6.0],
                "note": ["a", "b", "c"],  # non-numeric → dropped
            }
        )
        result = adapter._extract_pressure_array(df)
        assert result.shape == (3, 2)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result[:, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result[:, 1], [4.0, 5.0, 6.0])

    def test_extract_no_path_all_constant_columns_uses_numeric_fallback(self):
        """If every numeric column is constant, still return those columns."""
        import pandas as pd

        from haptix.sensors.coro import CoroCapacitiveAdapter

        adapter = CoroCapacitiveAdapter()
        df = pd.DataFrame({"marker": [7.0, 7.0, 7.0], "calib": [0.0, 0.0, 0.0]})
        result = adapter._extract_pressure_array(df)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result[:, 0], [7.0, 7.0, 7.0])
        np.testing.assert_array_equal(result[:, 1], [0.0, 0.0, 0.0])
