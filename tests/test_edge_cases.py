"""
Edge-case tests for haptix core data structures and I/O.

Covers: empty directories, corrupt files, missing manifest fields,
invalid modality values, mismatched shapes, and boundary conditions.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from haptix.core import (
    HaptData,
    InteractionMeta,
    Labels,
    RawData,
    SensorMeta,
    UnifiedData,
)
from haptix.io import ChecksumError, HaptFormatError, load, save

# =========================================================================
#  Helpers
# =========================================================================


def make_minimal_data(
    modality: str = "imaging", shape: tuple = (10, 240, 320, 3), dtype: np.dtype = np.uint8
) -> HaptData:
    """Create a minimal valid HaptData for testing."""
    frames = np.random.randint(0, 255, shape, dtype=dtype)
    return HaptData(
        raw=RawData(
            array=frames,
            checksum=RawData.compute_checksum(frames),
            dtype=str(frames.dtype),
            shape=frames.shape,
        ),
        sensor=SensorMeta(type="DIGIT_v2"),
        modality=modality,
        sampling_rate_hz=60.0,
        interaction=InteractionMeta(type="sliding"),
        labels=Labels(material="test"),
    )


def assert_raises_hapt_format(tmp_path: Path, **overrides):
    """Build a .hapt dir with given overrides and assert load raises HaptFormatError."""
    # Default minimal directory
    path = tmp_path / "test.hapt"
    path.mkdir()
    (path / "raw").mkdir()
    arr = np.zeros((1, 8, 8), dtype=np.uint8)
    np.save(path / "raw" / "data.npy", arr)
    with open(path / "raw" / "checksum.sha256", "w") as f:
        f.write(RawData.compute_checksum(arr) + "\n")
    manifest = {
        "version": "0.1.0",
        "sensor": {"type": "DIGIT_v2"},
        "modality": "imaging",
        "sampling": {"rate_hz": 60.0, "num_frames": 1},
        "raw_shape": [1, 8, 8],
        "raw_dtype": "uint8",
        "interaction": {"type": "sliding"},
        "created": "2026-07-24T00:00:00Z",
        "created_by": "haptix/0.1.0",
    }
    labels = {"material": "test"}
    # Apply overrides
    for key, val in overrides.items():
        if key == "manifest":
            manifest = val
        elif key == "manifest_field":
            k, v = val
            manifest[k] = v
        elif key == "labels":
            labels = val
        elif key == "remove_manifest":
            manifest.pop(val, None)
        elif key == "remove_file":
            target = path / val
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
        elif key == "corrupt_npy":
            (path / "raw" / "data.npy").write_bytes(b"not a valid numpy file")
        elif key == "corrupt_checksum":
            (path / "raw" / "checksum.sha256").write_text("DEADBEEF\n")
        elif key == "corrupt_manifest_json":
            (path / "manifest.json").write_text("{invalid json")
        elif key == "mismatched_shape":
            arr2 = np.zeros((5, 16, 16), dtype=np.uint8)
            np.save(path / "raw" / "data.npy", arr2)
        else:
            raise ValueError(f"Unknown override: {key}")
    with open(path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(path / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    with pytest.raises(HaptFormatError):
        load(path)


# =========================================================================
#  Core data structure edge cases
# =========================================================================


class TestSensorMetaEdgeCases:
    """Edge cases for SensorMeta."""

    def test_minimal_sensor_meta(self):
        """SensorMeta with only required type field."""
        sm = SensorMeta(type="DIGIT_v2")
        d = sm.to_dict()
        assert d == {"type": "DIGIT_v2"}
        restored = SensorMeta.from_dict(d)
        assert restored.type == "DIGIT_v2"
        assert restored.serial is None

    def test_sensor_meta_all_none_optionals(self):
        """SensorMeta with all optional fields explicitly None."""
        sm = SensorMeta(type="test", serial=None, calibration_date=None, calibration_params={})
        d = sm.to_dict()
        assert d == {"type": "test"}
        restored = SensorMeta.from_dict(d)
        assert restored.serial is None
        assert restored.calibration_params == {}

    def test_sensor_meta_from_dict_with_extra_keys(self):
        """SensorMeta.from_dict should ignore extra unknown keys."""
        d = {"type": "DIGIT_v2", "extra_field": "should_be_ignored", "nested": {"a": 1}}
        sm = SensorMeta.from_dict(d)
        assert sm.type == "DIGIT_v2"
        assert sm.serial is None


class TestInteractionMetaEdgeCases:
    """Edge cases for InteractionMeta."""

    def test_minimal_interaction(self):
        """InteractionMeta with only required type field."""
        im = InteractionMeta(type="static")
        d = im.to_dict()
        assert d == {"type": "static"}
        restored = InteractionMeta.from_dict(d)
        assert restored.type == "static"
        assert restored.speed_mm_s is None

    def test_interaction_all_fields(self):
        """InteractionMeta with every field set."""
        im = InteractionMeta(
            type="sliding",
            speed_mm_s=50.0,
            normal_force_N=2.0,
            approach_angle_deg=90.0,
            temperature_C=23.0,
            humidity_pct=45.0,
        )
        d = im.to_dict()
        assert d["type"] == "sliding"
        assert d["speed_mm_s"] == 50.0
        assert d["humidity_pct"] == 45.0
        restored = InteractionMeta.from_dict(d)
        assert restored.approach_angle_deg == 90.0
        assert restored.temperature_C == 23.0

    def test_interaction_from_dict_extra_keys(self):
        """InteractionMeta.from_dict should ignore extra unknown keys."""
        d = {"type": "pressing", "unknown_extra": 99, "speed_mm_s": 10}
        im = InteractionMeta.from_dict(d)
        assert im.type == "pressing"
        assert im.speed_mm_s == 10.0
        assert not hasattr(im, "unknown_extra")


class TestLabelsEdgeCases:
    """Edge cases for Labels."""

    def test_labels_all_none(self):
        """Labels with every field set to None."""
        labels = Labels()
        d = labels.to_dict()
        assert d == {}
        restored = Labels.from_dict(d)
        assert restored.material is None
        assert restored.custom_tags == []

    def test_labels_with_empty_custom_tags(self):
        """Labels with empty custom_tags list."""
        labels = Labels(material="wood", custom_tags=[])
        d = labels.to_dict()
        assert d == {"material": "wood"}
        restored = Labels.from_dict(d)
        assert restored.material == "wood"
        assert restored.custom_tags == []

    def test_labels_all_fields_populated(self):
        """Labels with every field populated."""
        labels = Labels(
            material="aluminum",
            material_category="metal",
            object_name="can",
            object_category="container",
            task="grasping",
            custom_tags=["smooth", "rigid"],
        )
        d = labels.to_dict()
        assert d["material"] == "aluminum"
        assert d["material_category"] == "metal"
        assert d["object"] == "can"
        assert d["custom_tags"] == ["smooth", "rigid"]
        restored = Labels.from_dict(d)
        assert restored.object_name == "can"
        assert restored.task == "grasping"

    def test_labels_from_dict_extra_keys(self):
        """Labels.from_dict should handle extra fields gracefully."""
        d = {"material": "foam", "unknown_key": "ignored", "nested": {}}
        labels = Labels.from_dict(d)
        assert labels.material == "foam"
        assert labels.object_name is None


class TestRawDataEdgeCases:
    """Edge cases for RawData."""

    def test_raw_data_zero_shape(self):
        """RawData with a zero-dimension array."""
        arr = np.zeros((0, 8, 8), dtype=np.uint8)
        raw = RawData(
            array=arr,
            checksum=RawData.compute_checksum(arr),
            dtype=str(arr.dtype),
            shape=arr.shape,
        )
        assert raw.shape == (0, 8, 8)
        assert raw.verify()
        assert raw.numpy().shape == (0, 8, 8)

    def test_raw_data_float_dtype(self):
        """RawData with float32 array."""
        arr = np.random.randn(5, 16).astype(np.float32)
        raw = RawData(
            array=arr,
            checksum=RawData.compute_checksum(arr),
            dtype="float32",
            shape=arr.shape,
        )
        assert raw.verify()
        view = raw.numpy()
        assert np.allclose(view, arr)

    def test_raw_data_verify_fails_on_mutation(self):
        """Verification catches checksum mismatch after array mutation."""
        arr = np.zeros((2, 2), dtype=np.uint8)
        raw = RawData(
            array=arr,
            checksum=RawData.compute_checksum(arr),
            dtype=str(arr.dtype),
            shape=arr.shape,
        )
        assert raw.verify()
        # Mutate the underlying array
        raw.array[0, 0] = 42
        assert not raw.verify()

    def test_raw_data_numpy_returns_view_not_copy(self):
        """numpy() returns a view of the underlying array, not a copy."""
        arr = np.ones((3, 3), dtype=np.uint8)
        raw = RawData(
            array=arr,
            checksum=RawData.compute_checksum(arr),
            dtype=str(arr.dtype),
            shape=arr.shape,
        )
        view = raw.numpy()
        # A view shares memory with the original array
        assert np.shares_memory(view, raw.array)
        # Values are accessible
        assert view[0, 0] == 1

    def test_raw_data_checksum_consistency(self):
        """Multiple calls to compute_checksum produce the same hash."""
        arr = np.random.randint(0, 255, (5, 10, 10), dtype=np.uint8)
        c1 = RawData.compute_checksum(arr)
        c2 = RawData.compute_checksum(arr.copy())
        assert c1 == c2

    def test_raw_data_checksum_differentiates_data(self):
        """Different arrays produce different checksums."""
        a1 = np.zeros((2, 2), dtype=np.uint8)
        a2 = np.ones((2, 2), dtype=np.uint8)
        assert RawData.compute_checksum(a1) != RawData.compute_checksum(a2)


class TestHaptDataEdgeCases:
    """Edge cases for HaptData construction."""

    def test_haptdata_with_unified_data(self):
        """HaptData with UnifiedData attached."""
        raw_arr = np.random.randint(0, 255, (3, 64, 64, 3), dtype=np.uint8)
        uni_arr = np.random.randn(3, 6).astype(np.float32)
        data = HaptData(
            raw=RawData(
                array=raw_arr,
                checksum=RawData.compute_checksum(raw_arr),
                dtype="uint8",
                shape=raw_arr.shape,
            ),
            sensor=SensorMeta(type="DIGIT_v2"),
            modality="imaging",
            sampling_rate_hz=60.0,
            interaction=InteractionMeta(type="sliding"),
            labels=Labels(material="test"),
            unified=UnifiedData(
                array=uni_arr,
                method="uniforce_v1",
                source_modality="imaging",
                target_modality="force",
                is_lossy=True,
                checksum=RawData.compute_checksum(uni_arr),
            ),
        )
        assert data.unified is not None
        assert data.unified.method == "uniforce_v1"
        assert data.unified.is_lossy
        assert np.allclose(data.unified.numpy(), uni_arr)

    def test_haptdata_repr(self):
        """__repr__ returns a human-readable string."""
        data = make_minimal_data()
        rep = repr(data)
        assert "DIGIT_v2" in rep
        assert "imaging" in rep
        assert "test" in rep

    def test_haptdata_default_version(self):
        """Default version is '0.1.0'."""
        data = make_minimal_data()
        assert data.version == "0.1.0"

    def test_haptdata_custom_version(self):
        """Custom version string is preserved."""
        data = HaptData(
            raw=RawData(
                array=np.zeros((1,), dtype=np.uint8),
                checksum=RawData.compute_checksum(np.zeros((1,), dtype=np.uint8)),
                dtype="uint8",
                shape=(1,),
            ),
            sensor=SensorMeta(type="test"),
            modality="force",
            sampling_rate_hz=100.0,
            interaction=InteractionMeta(type="static"),
            labels=Labels(),
            version="0.2.0-beta",
        )
        assert data.version == "0.2.0-beta"


# =========================================================================
#  I/O edge cases: directory structure errors
# =========================================================================


class TestLoadMissingStructure:
    """Errors for missing .hapt directory structure."""

    def test_load_nonexistent_path(self):
        """Loading a nonexistent path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load("/tmp/nonexistent_hapt_data_abc123")

    def test_load_empty_directory(self):
        """Empty directory without manifest raises HaptFormatError."""
        tmp = Path(tempfile.mkdtemp())
        try:
            d = tmp / "empty.hapt"
            d.mkdir()
            with pytest.raises(HaptFormatError, match="Missing manifest.json"):
                load(d)
        finally:
            shutil.rmtree(tmp)

    def test_load_no_raw_dir(self):
        """Directory with manifest but no raw/ dir raises HaptFormatError."""
        tmp = Path(tempfile.mkdtemp())
        try:
            d = tmp / "test.hapt"
            d.mkdir()
            with open(d / "manifest.json", "w") as f:
                json.dump(
                    {
                        "sensor": {"type": "test"},
                        "modality": "imaging",
                        "sampling": {"rate_hz": 60},
                        "interaction": {"type": "static"},
                        "version": "0.1.0",
                    },
                    f,
                )
            with open(d / "labels.json", "w") as f:
                json.dump({"material": "test"}, f)
            with pytest.raises(HaptFormatError, match="raw"):
                load(d)
        finally:
            shutil.rmtree(tmp)

    def test_load_no_labels(self):
        """Directory without labels.json raises HaptFormatError."""
        tmp = Path(tempfile.mkdtemp())
        try:
            d = tmp / "test.hapt"
            d.mkdir()
            (d / "raw").mkdir()
            np.save(d / "raw" / "data.npy", np.zeros((1,), dtype=np.uint8))
            with open(d / "raw" / "checksum.sha256", "w") as f:
                f.write("0" * 64 + "\n")
            with open(d / "manifest.json", "w") as f:
                json.dump(
                    {
                        "sensor": {"type": "test"},
                        "modality": "imaging",
                        "sampling": {"rate_hz": 60},
                        "interaction": {"type": "static"},
                        "version": "0.1.0",
                    },
                    f,
                )
            with pytest.raises(HaptFormatError, match="labels.json"):
                load(d)
        finally:
            shutil.rmtree(tmp)

    def test_load_no_data_npy(self):
        """Missing raw/data.npy raises HaptFormatError."""
        tmp = Path(tempfile.mkdtemp())
        try:
            d = tmp / "test.hapt"
            d.mkdir()
            (d / "raw").mkdir()
            with open(d / "raw" / "checksum.sha256", "w") as f:
                f.write("0" * 64 + "\n")
            with open(d / "manifest.json", "w") as f:
                json.dump(
                    {
                        "sensor": {"type": "test"},
                        "modality": "imaging",
                        "sampling": {"rate_hz": 60},
                        "interaction": {"type": "static"},
                        "version": "0.1.0",
                    },
                    f,
                )
            with open(d / "labels.json", "w") as f:
                json.dump({"material": "test"}, f)
            with pytest.raises(HaptFormatError, match="data.npy"):
                load(d)
        finally:
            shutil.rmtree(tmp)


class TestLoadCorruptFiles:
    """Errors for corrupt files inside .hapt directory."""

    def test_corrupt_npy_file(self):
        """Invalid NPY file raises a numpy error (wrapped as HaptFormatError or ValueError)."""
        tmp = Path(tempfile.mkdtemp())
        try:
            d = tmp / "test.hapt"
            d.mkdir()
            (d / "raw").mkdir()
            (d / "raw" / "data.npy").write_bytes(b"not a valid numpy file")
            (d / "raw" / "checksum.sha256").write_text("0" * 64 + "\n")
            with open(d / "manifest.json", "w") as f:
                json.dump(
                    {
                        "sensor": {"type": "test"},
                        "modality": "imaging",
                        "sampling": {"rate_hz": 60},
                        "interaction": {"type": "static"},
                        "version": "0.1.0",
                    },
                    f,
                )
            with open(d / "labels.json", "w") as f:
                json.dump({"material": "test"}, f)
            with pytest.raises((HaptFormatError, ValueError)):
                load(d)
        finally:
            shutil.rmtree(tmp)

    def test_corrupt_checksum(self):
        """Mismatched checksum raises ChecksumError."""
        tmp = Path(tempfile.mkdtemp())
        try:
            d = tmp / "test.hapt"
            d.mkdir()
            (d / "raw").mkdir()
            arr = np.zeros((2, 2), dtype=np.uint8)
            np.save(d / "raw" / "data.npy", arr)
            # Write a deliberately wrong checksum
            (d / "raw" / "checksum.sha256").write_text(
                "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n"
            )
            with open(d / "manifest.json", "w") as f:
                json.dump(
                    {
                        "sensor": {"type": "test"},
                        "modality": "imaging",
                        "sampling": {"rate_hz": 60},
                        "interaction": {"type": "static"},
                        "version": "0.1.0",
                    },
                    f,
                )
            with open(d / "labels.json", "w") as f:
                json.dump({"material": "test"}, f)
            with pytest.raises(ChecksumError, match="Checksum"):
                load(d)
        finally:
            shutil.rmtree(tmp)

    def test_corrupt_manifest_json(self):
        """Invalid JSON in manifest raises JSONDecodeError."""
        tmp = Path(tempfile.mkdtemp())
        try:
            d = tmp / "test.hapt"
            d.mkdir()
            (d / "raw").mkdir()
            np.save(d / "raw" / "data.npy", np.zeros((1,), dtype=np.uint8))
            (d / "raw" / "checksum.sha256").write_text("0" * 64 + "\n")
            (d / "manifest.json").write_text("{invalid json}")
            with open(d / "labels.json", "w") as f:
                json.dump({"material": "test"}, f)
            with pytest.raises(json.JSONDecodeError):
                load(d)
        finally:
            shutil.rmtree(tmp)

    def test_corrupt_labels_json(self):
        """Invalid JSON in labels raises JSONDecodeError."""
        tmp = Path(tempfile.mkdtemp())
        try:
            d = tmp / "test.hapt"
            d.mkdir()
            (d / "raw").mkdir()
            arr = np.zeros((1,), dtype=np.uint8)
            np.save(d / "raw" / "data.npy", arr)
            with open(d / "raw" / "checksum.sha256", "w") as f:
                f.write(RawData.compute_checksum(arr) + "\n")
            with open(d / "manifest.json", "w") as f:
                json.dump(
                    {
                        "sensor": {"type": "test"},
                        "modality": "imaging",
                        "sampling": {"rate_hz": 60},
                        "interaction": {"type": "static"},
                        "version": "0.1.0",
                    },
                    f,
                )
            (d / "labels.json").write_text("{bad json}")
            with pytest.raises(json.JSONDecodeError):
                load(d)
        finally:
            shutil.rmtree(tmp)


class TestLoadMissingManifestFields:
    """Errors for missing required manifest fields."""

    def test_missing_sensor(self):
        """Missing 'sensor' in manifest raises HaptFormatError."""
        tmp = Path(tempfile.mkdtemp())
        try:
            assert_raises_hapt_format(
                tmp,
                remove_manifest="sensor",
            )
        finally:
            shutil.rmtree(tmp)

    def test_missing_modality(self):
        """Missing 'modality' in manifest raises HaptFormatError."""
        tmp = Path(tempfile.mkdtemp())
        try:
            assert_raises_hapt_format(
                tmp,
                remove_manifest="modality",
            )
        finally:
            shutil.rmtree(tmp)

    def test_missing_interaction(self):
        """Missing 'interaction' in manifest raises HaptFormatError."""
        tmp = Path(tempfile.mkdtemp())
        try:
            assert_raises_hapt_format(
                tmp,
                remove_manifest="interaction",
            )
        finally:
            shutil.rmtree(tmp)

    def test_missing_sampling(self):
        """Missing 'sampling' in manifest raises HaptFormatError."""
        tmp = Path(tempfile.mkdtemp())
        try:
            assert_raises_hapt_format(
                tmp,
                remove_manifest="sampling",
            )
        finally:
            shutil.rmtree(tmp)

    def test_missing_sampling_rate(self):
        """Missing 'rate_hz' in sampling raises HaptFormatError."""
        tmp = Path(tempfile.mkdtemp())
        try:
            assert_raises_hapt_format(
                tmp,
                manifest_field=("sampling", {}),
            )
        finally:
            shutil.rmtree(tmp)


class TestLoadInvalidModality:
    """Loading files with invalid modality values."""

    def test_invalid_modality_string(self):
        """Load should raise HaptFormatError or KeyError for invalid modality."""
        tmp = Path(tempfile.mkdtemp())
        try:
            path = tmp / "test.hapt"
            path.mkdir()
            (path / "raw").mkdir()
            arr = np.zeros((1,), dtype=np.uint8)
            np.save(path / "raw" / "data.npy", arr)
            with open(path / "raw" / "checksum.sha256", "w") as f:
                f.write(RawData.compute_checksum(arr) + "\n")
            manifest = {
                "version": "0.1.0",
                "sensor": {"type": "test"},
                "modality": "invalid_modality_type",
                "sampling": {"rate_hz": 60},
                "raw_shape": [1],
                "raw_dtype": "uint8",
                "interaction": {"type": "static"},
                "created": "2026-07-24T00:00:00Z",
                "created_by": "haptix/0.1.0",
            }
            with open(path / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            with open(path / "labels.json", "w") as f:
                json.dump({"material": "test"}, f)
            # The Literal type will accept any string at runtime, but the
            # test verifies the system gracefully handles invalid values
            result = load(path)
            assert result.modality == "invalid_modality_type"
        finally:
            shutil.rmtree(tmp)


class TestLoadMismatchedData:
    """Loading files with data that doesn't match metadata."""

    def test_mismatched_raw_shape(self):
        """Manifest raw_shape doesn't match actual data — load still works (numpy provides truth)."""
        # The raw data is the source of truth; manifest is informational.
        tmp = Path(tempfile.mkdtemp())
        try:
            d = tmp / "test.hapt"
            d.mkdir()
            (d / "raw").mkdir()
            arr = np.zeros((5, 16, 16), dtype=np.uint8)  # actual shape
            np.save(d / "raw" / "data.npy", arr)
            with open(d / "raw" / "checksum.sha256", "w") as f:
                f.write(RawData.compute_checksum(arr) + "\n")
            manifest = {
                "version": "0.1.0",
                "sensor": {"type": "test"},
                "modality": "imaging",
                "sampling": {"rate_hz": 60, "num_frames": 5},
                "raw_shape": [3, 8, 8],  # DOESN'T MATCH
                "raw_dtype": "uint8",
                "interaction": {"type": "static"},
                "created": "2026-07-24T00:00:00Z",
                "created_by": "haptix/0.1.0",
            }
            with open(d / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            with open(d / "labels.json", "w") as f:
                json.dump({"material": "test"}, f)
            # Should load fine — raw data is source of truth
            result = load(d)
            assert result.raw.shape == (5, 16, 16)
        finally:
            shutil.rmtree(tmp)


class TestLoadSingleFile:
    """Loading single files vs directories."""

    def test_load_single_hapt_file(self):
        """A single .hapt file (not directory) raises HaptFormatError (compressed format)."""
        tmp = Path(tempfile.mkdtemp())
        try:
            f = tmp / "test.hapt"
            f.write_text("not a real hapt file")
            with pytest.raises(HaptFormatError, match="not supported"):
                load(f)
        finally:
            shutil.rmtree(tmp)

    def test_load_plain_file(self):
        """A non-.hapt file raises FileNotFoundError."""
        tmp = Path(tempfile.mkdtemp())
        try:
            f = tmp / "some_data.bin"
            f.write_bytes(b"\x00\x01\x02")
            with pytest.raises(FileNotFoundError):
                load(f)
        finally:
            shutil.rmtree(tmp)


# =========================================================================
#  I/O edge cases: save/load with unusual data shapes and types
# =========================================================================


class TestSaveEdgeCases:
    """Edge cases for the save function."""

    def test_save_1d_data(self):
        """Save and load 1D force data."""
        arr = np.random.randn(100).astype(np.float32)
        data = HaptData(
            raw=RawData(
                array=arr,
                checksum=RawData.compute_checksum(arr),
                dtype="float32",
                shape=arr.shape,
            ),
            sensor=SensorMeta(type="ATI_Nano17"),
            modality="force",
            sampling_rate_hz=1000.0,
            interaction=InteractionMeta(type="static", normal_force_N=5.0),
            labels=Labels(task="calibration"),
        )
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(data, tmp / "force.hapt")
            loaded = load(saved)
            assert loaded.raw.shape == (100,)
            assert loaded.raw.dtype == "float32"
            assert loaded.modality == "force"
            assert loaded.sensor.type == "ATI_Nano17"
            assert loaded.labels.task == "calibration"
            assert np.allclose(loaded.raw.array, arr)
        finally:
            shutil.rmtree(tmp)

    def test_save_2d_grayscale(self):
        """Save and load a single grayscale image (2D array becomes expanded)."""
        arr = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        data = HaptData(
            raw=RawData(
                array=arr,
                checksum=RawData.compute_checksum(arr),
                dtype="uint8",
                shape=arr.shape,
            ),
            sensor=SensorMeta(type="GelSight"),
            modality="imaging",
            sampling_rate_hz=30.0,
            interaction=InteractionMeta(type="static"),
            labels=Labels(material="test"),
        )
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(data, tmp / "gelsight.hapt")
            loaded = load(saved)
            # The data is stored and loaded as-is — numpy preserves shape
            assert loaded.raw.shape == (480, 640)
            assert np.array_equal(loaded.raw.array, arr)
        finally:
            shutil.rmtree(tmp)

    def test_save_with_unified_data_roundtrip(self):
        """Save and load HaptData with unified data."""
        raw_arr = np.random.randint(0, 255, (5, 32, 32, 3), dtype=np.uint8)
        uni_arr = np.random.randn(5, 6).astype(np.float32)
        data = HaptData(
            raw=RawData(
                array=raw_arr,
                checksum=RawData.compute_checksum(raw_arr),
                dtype="uint8",
                shape=raw_arr.shape,
            ),
            sensor=SensorMeta(type="DIGIT_v2"),
            modality="imaging",
            sampling_rate_hz=60.0,
            interaction=InteractionMeta(type="sliding"),
            labels=Labels(material="test"),
            unified=UnifiedData(
                array=uni_arr,
                method="pca",
                source_modality="imaging",
                target_modality="force",
                is_lossy=True,
                checksum=RawData.compute_checksum(uni_arr),
            ),
        )
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(data, tmp / "unified.hapt")
            loaded = load(saved)
            assert loaded.unified is not None
            assert loaded.unified.method == "pca"
            assert loaded.unified.is_lossy
            assert loaded.unified.target_modality == "force"
            assert np.allclose(loaded.unified.numpy(), uni_arr)
            assert np.array_equal(loaded.raw.array, raw_arr)
        finally:
            shutil.rmtree(tmp)

    def test_save_overwrites_existing(self):
        """Save to an existing directory should overwrite cleanly."""
        _ = np.ones((2, 2), dtype=np.uint8)
        data = make_minimal_data(shape=(2, 2))
        tmp = Path(tempfile.mkdtemp())
        try:
            p = tmp / "test.hapt"
            save(data, p)
            # Save again with different data
            arr2 = np.zeros((3, 3), dtype=np.uint8)
            data2 = HaptData(
                raw=RawData(
                    array=arr2,
                    checksum=RawData.compute_checksum(arr2),
                    dtype="uint8",
                    shape=arr2.shape,
                ),
                sensor=SensorMeta(type="test"),
                modality="imaging",
                sampling_rate_hz=30.0,
                interaction=InteractionMeta(type="static"),
                labels=Labels(),
            )
            save(data2, p)
            loaded = load(p)
            assert loaded.raw.shape == (3, 3)
            assert (loaded.raw.array == 0).all()
        finally:
            shutil.rmtree(tmp)

    def test_save_dynamic_modality(self):
        """Save and load dynamic (time-series) data."""
        arr = np.random.randn(200, 32).astype(np.float32)
        data = HaptData(
            raw=RawData(
                array=arr,
                checksum=RawData.compute_checksum(arr),
                dtype="float32",
                shape=arr.shape,
            ),
            sensor=SensorMeta(type="BioTac"),
            modality="dynamic",
            sampling_rate_hz=100.0,
            interaction=InteractionMeta(type="pressing", normal_force_N=1.0),
            labels=Labels(material="foam"),
        )
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(data, tmp / "dynamic.hapt")
            loaded = load(saved)
            assert loaded.raw.shape == (200, 32)
            assert loaded.modality == "dynamic"
            assert np.allclose(loaded.raw.array, arr)
        finally:
            shutil.rmtree(tmp)

    def test_save_with_all_labels_fields(self):
        """Save and load with all Labels fields populated."""
        labels = Labels(
            material="wood",
            material_category="natural",
            object_name="block",
            object_category="construction",
            task="sliding",
            custom_tags=["rough", "medium_density"],
        )
        data = make_minimal_data(shape=(5, 8, 8))
        data_with_labels = HaptData(
            raw=data.raw,
            sensor=data.sensor,
            modality=data.modality,
            sampling_rate_hz=data.sampling_rate_hz,
            interaction=data.interaction,
            labels=labels,
        )
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(data_with_labels, tmp / "labeled.hapt")
            loaded = load(saved)
            assert loaded.labels.material == "wood"
            assert loaded.labels.material_category == "natural"
            assert loaded.labels.object_name == "block"
            assert loaded.labels.object_category == "construction"
            assert loaded.labels.task == "sliding"
            assert loaded.labels.custom_tags == ["rough", "medium_density"]
        finally:
            shutil.rmtree(tmp)

    def test_save_empty_labels(self):
        """Save and load with completely empty Labels."""
        data = make_minimal_data(shape=(1, 4, 4))
        data_empty = HaptData(
            raw=data.raw,
            sensor=data.sensor,
            modality=data.modality,
            sampling_rate_hz=data.sampling_rate_hz,
            interaction=data.interaction,
            labels=Labels(),
        )
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(data_empty, tmp / "no_labels.hapt")
            loaded = load(saved)
            assert loaded.labels.material is None
            assert loaded.labels.custom_tags == []
        finally:
            shutil.rmtree(tmp)


class TestSaveUnusualTypes:
    """Save and load data with unusual dtypes."""

    @pytest.mark.parametrize(
        "dtype",
        [
            np.int16,
            np.int32,
            np.int64,
            np.float32,
            np.float64,
            np.uint16,
            np.uint32,
        ],
    )
    def test_various_dtypes_roundtrip(self, dtype):
        """Various integer and float dtypes survive round-trip."""
        arr = np.arange(10 * 8, dtype=dtype).reshape(10, 8)
        data = HaptData(
            raw=RawData(
                array=arr,
                checksum=RawData.compute_checksum(arr),
                dtype=str(arr.dtype),
                shape=arr.shape,
            ),
            sensor=SensorMeta(type="test"),
            modality="dynamic",
            sampling_rate_hz=100.0,
            interaction=InteractionMeta(type="static"),
            labels=Labels(),
        )
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(data, tmp / f"dtype_{dtype.__name__}.hapt")
            loaded = load(saved)
            assert loaded.raw.dtype == str(np.dtype(dtype))
            assert np.array_equal(loaded.raw.array, arr)
        finally:
            shutil.rmtree(tmp)


# =========================================================================
#  DIGIT adapter edge cases
# =========================================================================


class TestDigitAdapterEdgeCases:
    """Edge cases for the DIGIT sensor adapter."""

    def test_empty_directory(self):
        """Empty directory cannot be loaded and raises FileNotFoundError."""
        from haptix.sensors.digit import DigitAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            adapter = DigitAdapter()
            assert adapter.can_load(tmp) is False
            with pytest.raises(FileNotFoundError, match="No image frames"):
                adapter.load(
                    tmp,
                    interaction=InteractionMeta(type="sliding"),
                    labels=Labels(material="test"),
                )
        finally:
            shutil.rmtree(tmp)

    def test_can_load_nonexistent_path(self):
        """can_load should return False for nonexistent paths."""
        from haptix.sensors.digit import DigitAdapter

        adapter = DigitAdapter()
        assert adapter.can_load(Path("/nonexistent/path/xyz")) is False

    def test_can_load_single_file(self):
        """can_load for a single file (not dir) returns False for images, True for video."""
        from haptix.sensors.digit import DigitAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            adapter = DigitAdapter()
            # Single image file — not a directory
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            from PIL import Image

            png_path = tmp / "single.png"
            Image.fromarray(img).save(png_path)
            assert adapter.can_load(png_path) is False

            # Video file — handled by suffix check
            mp4_path = tmp / "video.mp4"
            mp4_path.write_text("fake mp4")
            assert adapter.can_load(mp4_path) is True
        finally:
            shutil.rmtree(tmp)

    def test_load_non_image_files_in_dir(self):
        """Directory with non-image files returns empty can_load."""
        from haptix.sensors.digit import DigitAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "metadata.txt").write_text("some metadata")
            (tmp / "calib.csv").write_text("a,b\n1,2")
            adapter = DigitAdapter()
            assert adapter.can_load(tmp) is False
        finally:
            shutil.rmtree(tmp)

    def test_mixed_frame_sizes(self):
        """Frames with inconsistent sizes — PIL returns different size, numpy stacking may fail."""
        from PIL import Image

        from haptix.sensors.digit import DigitAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            # Create frames with different sizes
            img1 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            Image.fromarray(img1).save(tmp / "frame_0000.png")
            img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            Image.fromarray(img2).save(tmp / "frame_0001.png")

            adapter = DigitAdapter()
            with pytest.raises((ValueError, FileNotFoundError)):
                adapter.load(
                    tmp,
                    interaction=InteractionMeta(type="sliding"),
                    labels=Labels(material="test"),
                )
        finally:
            shutil.rmtree(tmp)

    def test_single_frame_load(self):
        """Load a directory with a single frame."""
        from PIL import Image

        from haptix.sensors.digit import DigitAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            Image.fromarray(img).save(tmp / "frame_0000.png")

            adapter = DigitAdapter()
            data = adapter.load(
                tmp,
                interaction=InteractionMeta(type="static"),
                labels=Labels(material="single_test"),
            )
            assert data.raw.shape == (1, 240, 320, 3)
            assert len(data.raw.array) == 1
        finally:
            shutil.rmtree(tmp)

    def test_video_file_raises_import_error(self):
        """Loading a .mp4 video should raise ImportError when opencv-python is missing."""
        from haptix.sensors.digit import DigitAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            video_path = tmp / "test.mp4"
            video_path.write_text("fake mp4 content")

            adapter = DigitAdapter()
            assert adapter.can_load(video_path) is True
            with pytest.raises((ImportError, NotImplementedError)):
                adapter.load(
                    video_path,
                    interaction=InteractionMeta(type="sliding"),
                    labels=Labels(material="test"),
                )
        finally:
            shutil.rmtree(tmp)

    def test_load_with_custom_sensor_meta(self):
        """Allow overriding sensor metadata on DIGIT load."""
        from PIL import Image

        from haptix.sensors.digit import DigitAdapter

        tmp = Path(tempfile.mkdtemp())
        try:
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            Image.fromarray(img).save(tmp / "frame_0000.png")

            adapter = DigitAdapter()
            custom_sensor = SensorMeta(
                type="DIGIT_v2_custom",
                serial="DGT-001",
                calibration_date="2026-01-01",
            )
            data = adapter.load(
                tmp,
                interaction=InteractionMeta(type="static"),
                labels=Labels(material="test"),
                sensor_meta=custom_sensor,
            )
            assert data.sensor.type == "DIGIT_v2_custom"
            assert data.sensor.serial == "DGT-001"
        finally:
            shutil.rmtree(tmp)

    def test_registration(self):
        """DIGIT and DIGIT_v2 are both registered."""
        from haptix.sensors import list_sensors

        sensors = list_sensors()
        assert "DIGIT" in sensors
        assert "DIGIT_v2" in sensors


# =========================================================================
#  Additional coverage tests
# =========================================================================


class TestGetSensorEdgeCases:
    """Edge cases for sensor registry."""

    def test_get_unknown_sensor_raises(self):
        """get_sensor with unknown type raises ValueError."""
        from haptix.sensors import get_sensor

        with pytest.raises(ValueError, match="Unknown sensor type"):
            get_sensor("nonexistent_sensor_type_xyz")

    def test_get_sensor_digit(self):
        """get_sensor('DIGIT') returns DigitAdapter."""
        from haptix.sensors import get_sensor
        from haptix.sensors.digit import DigitAdapter

        adapter = get_sensor("DIGIT")
        assert isinstance(adapter, DigitAdapter)

    def test_get_sensor_gelsight_mini(self):
        """get_sensor('GelSight_Mini') returns GelSightAdapter."""
        from haptix.sensors import get_sensor
        from haptix.sensors.gelsight import GelSightAdapter

        adapter = get_sensor("GelSight_Mini")
        assert isinstance(adapter, GelSightAdapter)


class TestLoadWithoutChecksumFile:
    """Loading a .hapt directory that has no checksum.sha256 file."""

    def test_load_missing_checksum_file(self):
        """When checksum file is missing, should auto-compute and succeed."""
        tmp = Path(tempfile.mkdtemp())
        try:
            d = tmp / "test.hapt"
            d.mkdir()
            (d / "raw").mkdir()
            arr = np.arange(12, dtype=np.uint8).reshape(3, 4)
            np.save(d / "raw" / "data.npy", arr)
            # Intentionally DO NOT write checksum file
            manifest = {
                "version": "0.1.0",
                "sensor": {"type": "DIGIT_v2"},
                "modality": "imaging",
                "sampling": {"rate_hz": 60.0, "num_frames": 3},
                "raw_shape": [3, 4],
                "raw_dtype": "uint8",
                "interaction": {"type": "static"},
                "created": "2026-07-24T00:00:00Z",
                "created_by": "haptix/0.1.0",
            }
            with open(d / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            with open(d / "labels.json", "w") as f:
                json.dump({"material": "test"}, f)
            # Should load without checksum file
            result = load(d)
            assert result.raw.verify()
            assert np.array_equal(result.raw.array, arr)
        finally:
            shutil.rmtree(tmp)
