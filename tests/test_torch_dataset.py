"""Tests for the PyTorch-native windowed episode dataset (haptix.torch_dataset).

Covers: window slicing correctness vs eager load, overlapping/non-overlapping
windows, dataset-wide label encoding (classification + regression), transforms,
drop_last, in-memory HaptData sources, multi-episode concatenation (windows
never cross episode boundaries), unified-representation windows, error paths,
close/context-manager lifecycle, pickling for DataLoader workers, and an
end-to-end DataLoader run.

Requires torch:
    pip install 'haptix[torch]'
"""

import pickle
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

import haptix
from haptix.core import HaptData, InteractionMeta, Labels, RawData, SensorMeta
from haptix.io import save
from haptix.torch_dataset import TemporalDataset, WindowedDataset
from haptix.unified import SharedForceEncoder


def _seed_from(material: str) -> int:
    """Deterministic per-material seed so episodes differ across materials."""
    return sum(ord(c) for c in material) % (2**32)


def make_dynamic_data(
    n_frames: int = 60, material: str = "rubber", speed: float = 50.0
) -> HaptData:
    """Dynamic (time-series) recording, [T, D]."""
    arr = np.random.RandomState(_seed_from(material)).randn(n_frames, 16).astype(np.float32)
    return HaptData(
        raw=RawData(
            array=arr,
            checksum=RawData.compute_checksum(arr),
            dtype="float32",
            shape=arr.shape,
        ),
        sensor=SensorMeta(type="CoroCapacitive", serial="ds-test"),
        modality="dynamic",
        sampling_rate_hz=30.0,
        interaction=InteractionMeta(type="pressing", speed_mm_s=speed),
        labels=Labels(material=material, task="pressing"),
    )


def make_imaging_data(n_frames: int = 40, material: str = "rubber") -> HaptData:
    """Imaging recording, [T, H, W, C]."""
    arr = np.random.RandomState(1).randint(0, 255, (n_frames, 24, 32, 3)).astype(np.uint8)
    return HaptData(
        raw=RawData(
            array=arr,
            checksum=RawData.compute_checksum(arr),
            dtype="uint8",
            shape=arr.shape,
        ),
        sensor=SensorMeta(type="GelSight", serial="ds-test"),
        modality="imaging",
        sampling_rate_hz=30.0,
        interaction=InteractionMeta(type="sliding", speed_mm_s=25.0),
        labels=Labels(material=material, task="sliding"),
    )


@pytest.fixture(params=["dir", "zarr", "zip"])
def dynamic_path(request, tmp_path):
    """Save a dynamic recording in each supported format."""
    data = make_dynamic_data()
    if request.param == "dir":
        p = save(data, tmp_path / "dyn.hapt")
    elif request.param == "zarr":
        p = save(data, tmp_path / "dyn.hapt.zarr")
    else:
        p = save(data, tmp_path / "dyn.hapt.zip")
    return p, data


@pytest.fixture
def unified_path(tmp_path):
    """Recording with a unified/ representation (via SharedForceEncoder)."""
    data = make_dynamic_data(n_frames=50, material="rubber")
    enc = SharedForceEncoder(embedding_dim=16)
    data = HaptData(
        raw=data.raw,
        sensor=data.sensor,
        modality=data.modality,
        sampling_rate_hz=data.sampling_rate_hz,
        interaction=data.interaction,
        labels=data.labels,
        unified=enc.encode(data),
    )
    return save(data, tmp_path / "unified.hapt"), data


class TestBasics:
    def test_len_matches_window_count(self, dynamic_path):
        path, _ = dynamic_path
        with WindowedDataset([path], window_size=16) as ds:
            assert len(ds) == 4  # 60 frames, non-overlapping windows of 16

    def test_len_overlapping(self, dynamic_path):
        path, _ = dynamic_path
        with WindowedDataset([path], window_size=16, stride=8) as ds:
            assert len(ds) == 8  # starts at 0, 8, ..., 56 (last is partial)

    def test_getitem_shapes(self, dynamic_path):
        path, _ = dynamic_path
        with WindowedDataset([path], window_size=16) as ds:
            X = ds[0]
            assert isinstance(X, torch.Tensor)
            assert X.shape == (16, 16)
            assert X.dtype == torch.float32

    def test_window_content_matches_eager(self, dynamic_path):
        path, data = dynamic_path
        with WindowedDataset([path], window_size=16, stride=8) as ds:
            X = ds[2].numpy()  # window starting at frame 16
            assert np.allclose(X, data.raw.array[16:32])

    def test_imaging_windows(self, tmp_path):
        data = make_imaging_data()
        p = save(data, tmp_path / "img.hapt")
        with WindowedDataset([p], window_size=8, dtype="float32") as ds:
            X = ds[0]
            assert X.shape == (8, 24, 32, 3)
            assert X.dtype == torch.float32

    def test_in_memory_haptdata_source(self):
        data = make_dynamic_data()
        with WindowedDataset(data, window_size=16) as ds:
            assert len(ds) == 4
            assert ds.source_paths == [None]
            X = ds[0]
            assert X.shape == (16, 16)

    def test_single_path_and_archive_are_accepted(self, tmp_path):
        data = make_dynamic_data()
        p = save(data, tmp_path / "dyn.hapt")
        with haptix.open_archive(p) as arc:
            ds1 = WindowedDataset(p, window_size=16)
            ds2 = WindowedDataset(arc, window_size=16)
            assert len(ds1) == len(ds2) == 4
            assert torch.equal(ds1[0], ds2[0])
            ds1.close()
            ds2.close()

    def test_negative_index(self, dynamic_path):
        path, _ = dynamic_path
        with WindowedDataset([path], window_size=16) as ds:
            assert torch.equal(ds[-1], ds[3])


class TestMultiEpisode:
    def test_concatenation(self, tmp_path):
        mats = ["rubber", "metal", "plastic"]
        paths = []
        for i, m in enumerate(mats):
            paths.append(save(make_dynamic_data(n_frames=30, material=m), tmp_path / f"ep{i}.hapt"))
        with WindowedDataset(paths, window_size=16) as ds:
            # 30 frames -> windows [0:16], [16:32->30] = 2 per episode
            assert len(ds) == 6
            # First window of episode 1 starts at its frame 0, not episode 0's tail.
            X = ds[2].numpy()
            assert np.allclose(X, haptix.load(paths[1]).raw.array[0:16])

    def test_windows_never_cross_boundaries(self, tmp_path):
        p1 = save(make_dynamic_data(n_frames=20, material="rubber"), tmp_path / "a.hapt")
        p2 = save(make_dynamic_data(n_frames=20, material="metal"), tmp_path / "b.hapt")
        with WindowedDataset([p1, p2], window_size=16, stride=16) as ds:
            assert len(ds) == 4  # 1 full window + 1 partial per episode
            # Flat indices 0,1 -> episode 1 (full + partial); 2,3 -> episode 2.
            # Index 2 is the FIRST window of episode 2, not a continuation of
            # episode 1 (data differs across materials).
            X2 = ds[2].numpy()
            assert np.allclose(X2, haptix.load(p2).raw.array[0:16])

    def test_drop_last(self, tmp_path):
        p = save(make_dynamic_data(n_frames=50, material="rubber"), tmp_path / "a.hapt")
        with WindowedDataset([p], window_size=16, drop_last=True) as ds:
            assert len(ds) == 3  # partial 2-frame window dropped

    def test_source_paths(self, tmp_path):
        p1 = save(make_dynamic_data(material="rubber"), tmp_path / "a.hapt")
        p2 = save(make_dynamic_data(material="metal"), tmp_path / "b.hapt")
        with WindowedDataset([p1, p2], window_size=16) as ds:
            assert ds.n_sources == 2
            assert ds.source_paths == [Path(p1), Path(p2)]


class TestLabels:
    def test_classification_global_encoding(self, tmp_path):
        # Same material in two files must map to the same class index.
        p1 = save(make_dynamic_data(material="rubber"), tmp_path / "a.hapt")
        p2 = save(make_dynamic_data(material="rubber"), tmp_path / "b.hapt")
        p3 = save(make_dynamic_data(material="metal"), tmp_path / "c.hapt")
        with WindowedDataset([p1, p2, p3], window_size=16, label="material") as ds:
            # Sorted deterministic encoding: metal -> 1, rubber -> 2; None -> 0.
            assert ds.label_classes == {"metal": 1, "rubber": 2, None: 0}
            _, y0 = ds[0]  # episode 0: rubber
            _, y1 = ds[5]  # episode 1: rubber
            _, y2 = ds[9]  # episode 2: metal
            assert y0.item() == y1.item() == 2
            assert y2.item() == 1
            assert y0.dtype == torch.long

    def test_regression_label(self, tmp_path):
        p1 = save(make_dynamic_data(speed=10.0), tmp_path / "a.hapt")
        p2 = save(make_dynamic_data(speed=42.5), tmp_path / "b.hapt")
        with WindowedDataset([p1, p2], window_size=16, label="speed_mm_s") as ds:
            assert ds.label_classes is None
            _, y0 = ds[0]
            _, y1 = ds[len(ds) - 1]
            assert y0.dtype == torch.float32
            assert y0.item() == 10.0
            assert y1.item() == 42.5

    def test_sensor_type_label(self, tmp_path):
        p = save(make_dynamic_data(material="rubber"), tmp_path / "a.hapt")
        with WindowedDataset([p], window_size=16, label="sensor_type") as ds:
            _, y = ds[0]
            assert ds.label_classes == {"CoroCapacitive": 1, None: 0}
            assert y.item() == 1

    def test_numeric_label_none_in_some_sources_raises(self, tmp_path):
        p1 = save(make_dynamic_data(speed=10.0), tmp_path / "c.hapt")
        p2 = save(make_dynamic_data(speed=None), tmp_path / "d.hapt")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="None in some sources"):
            WindowedDataset([p1, p2], window_size=16, label="speed_mm_s")


class TestTransforms:
    def test_transform_applied(self, dynamic_path):
        path, _ = dynamic_path
        with WindowedDataset([path], window_size=16, transform=lambda x: x * 2) as ds:
            X = ds[0]
            assert np.allclose(X.numpy(), ds._sources[0].window_array(0, 16) * 2)

    def test_target_transform_applied(self, tmp_path):
        p = save(make_dynamic_data(material="rubber"), tmp_path / "a.hapt")
        with WindowedDataset(
            [p], window_size=16, label="material", target_transform=lambda y: y + 10
        ) as ds:
            _, y = ds[0]
            assert y.item() == 11


class TestUnified:
    def test_use_unified(self, unified_path):
        path, _ = unified_path
        with WindowedDataset([path], window_size=16, use_unified=True) as ds:
            X = ds[0]
            assert X.shape == (16, 16)  # embedding dim 16
        with pytest.raises(ValueError, match="Unknown label field"):
            WindowedDataset([path], window_size=16, use_unified=True, label="nope")

    def test_use_unified_requires_unified_data(self, dynamic_path):
        path, _ = dynamic_path
        with pytest.raises(ValueError, match="no unified representation"):
            WindowedDataset([path], window_size=16, use_unified=True)

    def test_unified_shape_metadata(self, unified_path):
        path, _ = unified_path
        with haptix.open_archive(path) as arc:
            assert arc.unified_shape == (50, 16)
            assert arc.unified_method is not None
        plain = save(make_dynamic_data(), path.parent / "plain.hapt")
        with haptix.open_archive(plain) as arc:
            assert arc.unified_shape is None


class TestErrors:
    def test_bad_window_size(self, dynamic_path):
        path, _ = dynamic_path
        with pytest.raises(ValueError, match="window_size"):
            WindowedDataset([path], window_size=0)

    def test_bad_stride(self, dynamic_path):
        path, _ = dynamic_path
        with pytest.raises(ValueError, match="stride"):
            WindowedDataset([path], window_size=16, stride=0)

    def test_empty_sources(self):
        with pytest.raises(ValueError, match="at least one recording"):
            WindowedDataset([], window_size=16)

    def test_bad_source_type(self):
        with pytest.raises(TypeError):
            WindowedDataset([42], window_size=16)

    def test_unknown_label_field(self, dynamic_path):
        path, _ = dynamic_path
        with pytest.raises(ValueError, match="Unknown label field"):
            WindowedDataset([path], window_size=16, label="bogus")

    def test_inconsistent_frame_shapes(self, tmp_path):
        p1 = save(make_dynamic_data(), tmp_path / "a.hapt")
        p2 = save(make_imaging_data(), tmp_path / "b.hapt")
        with pytest.raises(ValueError, match="inconsistent frame shapes"):
            WindowedDataset([p1, p2], window_size=16)

    def test_index_out_of_range(self, dynamic_path):
        path, _ = dynamic_path
        with WindowedDataset([path], window_size=16) as ds, pytest.raises(IndexError):
            ds[99]


class TestLifecycle:
    def test_close_is_idempotent(self, dynamic_path):
        path, _ = dynamic_path
        ds = WindowedDataset([path], window_size=16)
        ds.close()
        ds.close()
        with pytest.raises(RuntimeError, match="closed"):
            ds[0]

    def test_context_manager_closes_owned(self, dynamic_path):
        path, _ = dynamic_path
        with WindowedDataset([path], window_size=16) as ds:
            assert len(ds) == 4
        with pytest.raises(RuntimeError, match="closed"):
            len(ds)

    def test_passed_archive_not_closed(self, tmp_path):
        p = save(make_dynamic_data(), tmp_path / "a.hapt")
        with haptix.open_archive(p) as arc:
            ds = WindowedDataset(arc, window_size=16)
            ds.close()
            # The archive the caller passed in is still usable.
            assert arc.n_frames == 60
            ds.close()

    def test_pickle_roundtrip(self, tmp_path):
        p1 = save(make_dynamic_data(material="rubber"), tmp_path / "a.hapt")
        p2 = save(make_dynamic_data(material="metal"), tmp_path / "b.hapt")
        ds = WindowedDataset([p1, p2], window_size=16, stride=8, label="material")
        n = len(ds)
        payload = pickle.dumps(ds)
        ds2 = pickle.loads(payload)
        ds.close()  # original's archives released; the unpickled copy is independent
        assert len(ds2) == n
        assert ds2.label_classes == {"metal": 1, "rubber": 2, None: 0}
        X, y = ds2[0]
        assert X.shape == (16, 16)
        assert y.item() == 2  # rubber
        assert ds2[5][0].shape == (16, 16)
        ds2.close()


class TestDataLoader:
    def test_end_to_end_batching(self, tmp_path):
        mats = ["rubber", "metal", "plastic"]
        paths = [
            save(make_dynamic_data(n_frames=40, material=m), tmp_path / f"ep{i}.hapt")
            for i, m in enumerate(mats)
        ]
        ds = WindowedDataset(paths, window_size=8, stride=8, label="material")
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, drop_last=True)
        batches = list(loader)
        assert len(batches) == 3  # 15 windows, batch 4 -> 3 full batches
        X, y = batches[0]
        assert X.shape == (4, 8, 16)
        assert y.shape == (4,)
        assert y.dtype == torch.long
        # Sorted encoding: metal=1, plastic=2, rubber=3. The first batch is
        # entirely episode 0 (rubber -> class 3).
        assert set(batches[0][1].tolist()) == {3}
        assert set(batches[1][1].tolist()) <= {1, 3}  # ep0 tail + ep1 head
        ds.close()

    def test_dataloader_with_workers(self, tmp_path):
        # Exercises multi-worker iteration over WindowedDataset.
        #
        # Uses the fork context explicitly: on macOS the default start
        # method is spawn, and this toolchain (Python 3.13.5 + torch 2.5.1)
        # has a bug where spawn workers that convert numpy -> torch via
        # torch.from_numpy wedge the multiprocessing resource_tracker child
        # at interpreter exit (test passes, process never terminates).
        # fork is also the default on Linux CI, so this stays deterministic
        # everywhere. Spawn-side pickling of the dataset is covered
        # separately by test_pickle_roundtrip.
        mats = ["rubber", "metal"]
        paths = [
            save(make_dynamic_data(n_frames=40, material=m), tmp_path / f"ep{i}.hapt")
            for i, m in enumerate(mats)
        ]
        ds = WindowedDataset(paths, window_size=8, stride=8, label="material")
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            drop_last=True,
            multiprocessing_context="fork",
        )
        seen = 0
        for X, y in loader:
            assert X.shape == (8, 8, 16)
            assert y.shape == (8,)
            seen += 1
            if seen >= 2:
                break
        assert seen >= 1
        ds.close()


class TestAlias:
    def test_temporal_dataset_is_windowed_dataset(self):
        assert TemporalDataset is WindowedDataset
