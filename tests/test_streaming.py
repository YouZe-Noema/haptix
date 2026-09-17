"""Tests for streaming / temporal windowing (haptix.streaming).

Covers: lazy open (no raw array materialized), window slicing correctness
vs eager load(), timestamps slicing, overlapping/non-overlapping windows,
window_count, frame_index_at, streaming verify(), zip/zarr/dir formats,
context manager, and round-trip of a window.
"""

import numpy as np
import pytest

import haptix
from haptix.core import HaptData, InteractionMeta, Labels, RawData, SensorMeta
from haptix.io import ChecksumError, HaptFormatError, load, save
from haptix.streaming import HaptArchive, open_archive


class _ArchiveFixture:
    """(path, original HaptData) pair for a saved archive."""

    def __init__(self, path, data):
        self.path = path
        self.data = data


def _make_long_data(n_frames: int = 300, t=60) -> HaptData:
    """Imaging recording with per-frame timestamps."""
    arr = np.random.RandomState(0).randint(0, 255, (n_frames, 32, 40, 3)).astype(np.uint8)
    return HaptData(
        raw=RawData(
            array=arr,
            checksum=RawData.compute_checksum(arr),
            dtype="uint8",
            shape=arr.shape,
        ),
        sensor=SensorMeta(type="GelSight", serial="stream-test"),
        modality="imaging",
        sampling_rate_hz=30.0,
        interaction=InteractionMeta(type="sliding", speed_mm_s=50.0),
        labels=Labels(material="rubber", task="sliding"),
        timestamps_s=[i / 30.0 for i in range(n_frames)],
        coordinate_frame="sensor_local",
        version="0.2.0",
    )


@pytest.fixture(params=["dir", "zarr", "zip"])
def archive_path(request, tmp_path):
    """Save a long recording in each supported format."""
    data = _make_long_data()
    if request.param == "dir":
        p = save(data, tmp_path / "long.hapt")
    elif request.param == "zarr":
        p = save(data, tmp_path / "long.hapt.zarr")
    else:
        p = save(data, tmp_path / "long.hapt.zip")
    return _ArchiveFixture(p, data)


class TestOpen:
    def test_open_metadata(self, archive_path):
        path = archive_path.path
        with open_archive(path) as arc:
            assert arc.n_frames == 300
            assert arc.shape == (300, 32, 40, 3)
            assert arc.dtype == "uint8"
            assert arc.sensor.type == "GelSight"
            assert arc.modality == "imaging"
            assert arc.sampling_rate_hz == 30.0
            assert arc.labels.material == "rubber"
            assert arc.coordinate_frame == "sensor_local"
            assert arc.version == "0.2.0"
            assert arc.timestamps_s is not None
            assert len(arc.timestamps_s) == 300

    def test_open_does_not_materialize_raw(self, archive_path):
        path = archive_path.path
        arc = open_archive(path)
        try:
            # No public raw accessor until a window is requested.
            assert not hasattr(arc, "raw")
            assert arc.n_frames == 300  # metadata only
        finally:
            arc.close()

    def test_open_exported_top_level(self):
        assert haptix.open_archive is open_archive
        assert haptix.HaptArchive is HaptArchive
        assert "open_archive" in haptix.__all__

    def test_open_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            open_archive(tmp_path / "nope.hapt")

    def test_open_flat_hapt_rejected(self, tmp_path):
        p = tmp_path / "legacy.hapt"
        p.write_bytes(b"legacy")
        with pytest.raises(HaptFormatError):
            open_archive(p)

    def test_close_releases(self, tmp_path):
        data = _make_long_data(n_frames=20)
        p = save(data, tmp_path / "short.hapt")
        arc = open_archive(p)
        arc.close()
        arc.close()  # idempotent
        with pytest.raises(RuntimeError):
            arc.window(0, 5)


class TestWindow:
    def test_window_matches_eager_load(self, archive_path):
        path = archive_path.path
        eager = load(path)
        with open_archive(path) as arc:
            win = arc.window(50, 150)
            assert win.raw.shape == (100, 32, 40, 3)
            assert np.array_equal(win.raw.array, eager.raw.array[50:150])
            assert win.sensor.type == eager.sensor.type
            assert win.labels.material == "rubber"
            assert win.timestamps_s == [i / 30.0 for i in range(50, 150)]

    def test_window_has_own_checksum(self, archive_path):
        path = archive_path.path
        with open_archive(path) as arc:
            win = arc.window(0, 100)
            assert len(win.raw.checksum) == 64
            assert win.raw.verify()

    def test_window_clamps_bounds(self, archive_path):
        path = archive_path.path
        with open_archive(path) as arc:
            win = arc.window(-10, 1000)
            assert win.raw.shape[0] == 300

    def test_window_invalid_raises(self, archive_path):
        path = archive_path.path
        with open_archive(path) as arc, pytest.raises(ValueError):
            arc.window(10, 10)

    def test_window_roundtrip(self, archive_path, tmp_path):
        path = archive_path.path
        with open_archive(path) as arc:
            win = arc.window(0, 64)
            p2 = save(win, tmp_path / "win.hapt")
            reloaded = load(p2)
            assert np.array_equal(reloaded.raw.array, win.raw.array)
            assert reloaded.raw.checksum == win.raw.checksum


class TestIterWindows:
    def test_non_overlapping(self, archive_path):
        path, data = archive_path.path, archive_path.data
        with open_archive(path) as arc:
            wins = list(arc.iter_windows(window_size=64))
            assert len(wins) == 5  # 300 frames / 64 -> 4 full + 1 partial(44)
            assert wins[0].raw.shape[0] == 64
            assert wins[-1].raw.shape[0] == 44
            assert np.array_equal(wins[2].raw.array, data.raw.array[128:192])

    def test_drop_last(self, archive_path):
        path = archive_path.path
        with open_archive(path) as arc:
            wins = list(arc.iter_windows(window_size=64, drop_last=True))
            assert len(wins) == 4
            assert all(w.raw.shape[0] == 64 for w in wins)

    def test_overlapping_stride(self, archive_path):
        path, data = archive_path.path, archive_path.data
        with open_archive(path) as arc:
            wins = list(arc.iter_windows(window_size=64, stride=32))
            # starts at 0,32,...,288 -> 10 windows (last is partial)
            assert len(wins) == 10
            assert np.array_equal(wins[1].raw.array, data.raw.array[32:96])

    def test_start_stop_bounds(self, archive_path):
        path, data = archive_path.path, archive_path.data
        with open_archive(path) as arc:
            wins = list(arc.iter_windows(window_size=20, start=40, stop=100))
            assert len(wins) == 3  # [40,60) [60,80) [80,100)
            assert np.array_equal(wins[0].raw.array, data.raw.array[40:60])

    def test_window_count_matches_iteration(self, archive_path):
        path = archive_path.path
        with open_archive(path) as arc:
            cases = [
                {"window_size": 64},
                {"window_size": 64, "drop_last": True},
                {"window_size": 64, "stride": 32},
                {"window_size": 20, "start": 40, "stop": 100},
            ]
            for kw in cases:
                count = arc.window_count(**kw)
                n_iter = len(list(arc.iter_windows(**kw)))
                assert count == n_iter

    def test_invalid_args(self, archive_path):
        path = archive_path.path
        with open_archive(path) as arc:
            with pytest.raises(ValueError):
                list(arc.iter_windows(window_size=0))
            with pytest.raises(ValueError):
                list(arc.iter_windows(window_size=10, stride=0))


class TestTimeIndexing:
    def test_frame_index_at_with_timestamps(self, archive_path):
        path = archive_path.path
        with open_archive(path) as arc:
            assert arc.frame_index_at(0.0) == 0
            assert arc.frame_index_at(5.0) == 150  # 30 Hz
            assert arc.frame_index_at(1000.0) == 299  # clamped

    def test_frame_index_at_equal_spacing(self, tmp_path):
        arr = np.zeros((60, 4), dtype=np.float32)
        data = HaptData(
            raw=RawData(
                array=arr, checksum=RawData.compute_checksum(arr), dtype="float32", shape=arr.shape
            ),
            sensor=SensorMeta(type="CoroCapacitive"),
            modality="dynamic",
            sampling_rate_hz=30.0,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(material="metal"),
        )
        p = save(data, tmp_path / "dyn.hapt")
        with open_archive(p) as arc:
            assert arc.timestamps_s is None
            assert arc.frame_index_at(1.0) == 30


class TestVerify:
    def test_verify_ok(self, archive_path):
        path = archive_path.path
        with open_archive(path) as arc:
            assert arc.verify() is True

    def test_verify_detects_corruption_dir(self, tmp_path):
        data = _make_long_data(n_frames=50)
        p = save(data, tmp_path / "c.hapt")
        # Corrupt a byte in the raw data.
        raw_path = p / "raw" / "data.npy"
        b = bytearray(raw_path.read_bytes())
        b[-10] ^= 0xFF
        raw_path.write_bytes(bytes(b))
        with open_archive(p) as arc, pytest.raises(ChecksumError):
            arc.verify()

    def test_verify_detects_corruption_zip(self, tmp_path):
        data = _make_long_data(n_frames=50)
        p = save(data, tmp_path / "c.hapt.zip")
        import zipfile

        # Rewrite the archive with a corrupted raw member.
        with zipfile.ZipFile(p, "r") as zf:
            members = {n: zf.read(n) for n in zf.namelist()}
        corrupted = bytearray(members["raw/data.npy"])
        corrupted[-10] ^= 0xFF
        members["raw/data.npy"] = bytes(corrupted)
        with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for n, b in members.items():
                zf.writestr(n, bytes(b))
        with open_archive(p) as arc, pytest.raises(ChecksumError):
            arc.verify()


class TestDynamicFormat:
    def test_dynamic_window(self, tmp_path):
        arr = np.random.RandomState(1).rand(1000, 29).astype(np.float32)
        data = HaptData(
            raw=RawData(
                array=arr, checksum=RawData.compute_checksum(arr), dtype="float32", shape=arr.shape
            ),
            sensor=SensorMeta(type="CoroCapacitive"),
            modality="dynamic",
            sampling_rate_hz=30.0,
            interaction=InteractionMeta(type="pressing", normal_force_N=3.0),
            labels=Labels(material="metal"),
        )
        p = save(data, tmp_path / "dyn.hapt")
        with open_archive(p) as arc:
            wins = list(arc.iter_windows(window_size=128, stride=64, drop_last=True))
            # starts at 0,64,...,832 yield full 128-frame windows (14);
            # start 896 leaves 104 frames -> dropped.
            assert len(wins) == 14
            assert wins[0].raw.shape == (128, 29)
            assert np.array_equal(wins[0].raw.array, arr[:128])


# ---------------------------------------------------------------------------
# Error / edge paths (hardening)
# ---------------------------------------------------------------------------


def _minimal_haptdata(n_frames: int = 20, with_unified: bool = False) -> HaptData:
    from haptix.core import UnifiedData

    arr = np.random.RandomState(2).randn(n_frames, 8).astype(np.float32)
    unified = None
    if with_unified:
        u = np.random.RandomState(3).randn(n_frames, 4).astype(np.float32)
        unified = UnifiedData(
            array=u,
            method="TestEncoder",
            source_modality="dynamic",
            target_modality="latent",
            is_lossy=False,
            checksum=RawData.compute_checksum(u),
        )
    return HaptData(
        raw=RawData(
            array=arr, checksum=RawData.compute_checksum(arr), dtype="float32", shape=arr.shape
        ),
        sensor=SensorMeta(type="CoroCapacitive"),
        modality="dynamic",
        sampling_rate_hz=10.0,
        interaction=InteractionMeta(type="pressing"),
        labels=Labels(material="foam"),
        unified=unified,
    )


def _zip_with_members(path, members: dict):
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, payload)
    return path


class TestOpenZipErrors:
    def test_invalid_zip_raises_hapt_format_error(self, tmp_path):
        p = tmp_path / "bad.hapt.zip"
        p.write_bytes(b"not-a-zip")
        with pytest.raises(HaptFormatError, match="Not a valid .hapt.zip archive"):
            open_archive(p)

    @pytest.mark.parametrize(
        "missing,match",
        [
            ("manifest.json", "Missing manifest.json"),
            ("raw/data.npy", "Missing raw/data.npy"),
            ("labels.json", "Missing labels.json"),
        ],
    )
    def test_zip_missing_required_member(self, tmp_path, missing, match):
        import io
        import json

        arr = np.zeros((4, 2), dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, arr)
        members = {
            "manifest.json": json.dumps(
                {
                    "sensor": {"type": "t"},
                    "modality": "dynamic",
                    "sampling": {"rate_hz": 1.0},
                    "interaction": {"type": "pressing"},
                }
            ).encode(),
            "raw/data.npy": buf.getvalue(),
            "labels.json": b"{}",
        }
        del members[missing]
        p = _zip_with_members(tmp_path / "miss.hapt.zip", members)
        with pytest.raises(HaptFormatError, match=match):
            open_archive(p)


class TestOpenZarrErrors:
    def test_empty_zip_not_a_valid_hapt_path(self, tmp_path):
        import zipfile

        p = tmp_path / "empty.hapt.zarr"
        with zipfile.ZipFile(p, "w"):
            pass
        with pytest.raises(FileNotFoundError, match="Not a valid .hapt path"):
            open_archive(p)

    def test_corrupt_zarr_valueerror_branch(self, tmp_path):
        """Invalid `.zgroup` JSON trips the ValueError → HaptFormatError path."""
        import zipfile

        p = tmp_path / "badfmt.hapt.zarr"
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr(".zgroup", "not-json")
            zf.writestr(".zattrs", "{}")
        with pytest.raises(HaptFormatError, match="Not a valid .hapt.zarr archive"):
            open_archive(p)

    def test_corrupt_zarr_badzipfile_raises_hapt_format_error(self, tmp_path):
        p = tmp_path / "garbage.hapt.zarr"
        p.write_bytes(b"PK\x03\x04" + b"\x00" * 20)  # fake local-file header
        with pytest.raises(HaptFormatError, match="Not a valid .hapt.zarr archive"):
            open_archive(p)

    def test_zarr_missing_raw_data(self, tmp_path):
        from haptix.io import _ensure_zarr, _zarr_group

        zarr, _ = _ensure_zarr()
        p = tmp_path / "noraw.hapt.zarr"
        store = zarr.storage.ZipStore(str(p), mode="w")
        root = _zarr_group(zarr, store)
        root.attrs["manifest"] = {
            "sensor": {"type": "t"},
            "modality": "dynamic",
            "sampling": {"rate_hz": 1.0},
            "interaction": {"type": "pressing"},
        }
        root.attrs["labels"] = {}
        store.close()
        with pytest.raises(HaptFormatError, match="Missing raw/data"):
            open_archive(p)

    def test_zarr_missing_manifest_labels_attrs(self, tmp_path):
        from haptix.io import _ensure_zarr, _zarr_create_array, _zarr_group

        zarr, _ = _ensure_zarr()
        p = tmp_path / "noattrs.hapt.zarr"
        store = zarr.storage.ZipStore(str(p), mode="w")
        root = _zarr_group(zarr, store)
        arr = np.zeros((2, 3), dtype=np.float32)
        zarr_arr = _zarr_create_array(root, "raw/data", arr.shape, arr.dtype, arr.shape, None)
        zarr_arr[:] = arr
        store.close()
        with pytest.raises(HaptFormatError, match="Missing manifest/labels"):
            open_archive(p)


class TestZarrUnifiedMetadata:
    def test_unified_transform_attrs(self, tmp_path):
        data = _minimal_haptdata(n_frames=12, with_unified=True)
        p = save(data, tmp_path / "u.hapt.zarr")
        with open_archive(p) as arc:
            assert arc.unified_shape == (12, 4)
            assert arc.unified_method == "TestEncoder"
            # source/target/is_lossy live on the archive private fields and
            # surface via UnifiedData on window slices.
            assert arc._unified_source == "dynamic"
            assert arc._unified_target == "latent"
            assert arc._unified_is_lossy is False
            win = arc.window(0, 4)
            assert win.unified is not None
            assert win.unified.method == "TestEncoder"
            assert win.unified.source_modality == "dynamic"
            assert win.unified.target_modality == "latent"
            assert win.unified.is_lossy is False


class TestWindowCountEdgeCases:
    def test_window_count_stop_le_start_returns_zero(self, tmp_path):
        p = save(_minimal_haptdata(), tmp_path / "w.hapt")
        with open_archive(p) as arc:
            assert arc.window_count(window_size=4, start=10, stop=10) == 0
            assert arc.window_count(window_size=4, start=15, stop=5) == 0

    def test_window_count_invalid_args(self, tmp_path):
        p = save(_minimal_haptdata(), tmp_path / "w.hapt")
        with open_archive(p) as arc:
            with pytest.raises(ValueError, match="window_size must be >= 1"):
                arc.window_count(window_size=0)
            with pytest.raises(ValueError, match="stride must be >= 1"):
                arc.window_count(window_size=4, stride=0)


class TestFrameIndexClamp:
    def test_timestamps_clamp_low_and_high(self, tmp_path):
        data = _make_long_data(n_frames=30)
        p = save(data, tmp_path / "ts.hapt")
        with open_archive(p) as arc:
            assert arc.frame_index_at(-100.0) == 0
            assert arc.frame_index_at(1e9) == 29

    def test_equal_spacing_clamp_low_and_high(self, tmp_path):
        p = save(_minimal_haptdata(n_frames=60), tmp_path / "eq.hapt")
        with open_archive(p) as arc:
            assert arc.timestamps_s is None
            assert arc.frame_index_at(-5.0) == 0
            assert arc.frame_index_at(1e6) == 59


class TestHaptArchiveRepr:
    def test_repr_contains_path_modality_shape(self, tmp_path):
        p = save(_minimal_haptdata(n_frames=8), tmp_path / "repr.hapt")
        with open_archive(p) as arc:
            text = repr(arc)
            assert str(p) in text
            assert "dynamic" in text
            assert "shape=(8, 8)" in text
            assert text.startswith("HaptArchive(")
