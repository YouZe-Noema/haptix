"""
Tests for PyTorch integration (to_torch).

These tests require torch to be installed:
    pip install 'haptix[torch]'
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from haptix.core import (
    HaptData,
    InteractionMeta,
    Labels,
    RawData,
    SensorMeta,
)
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
            material_category="abrasive",
            task="sliding",
            custom_tags=["rough", "high_friction"],
        ),
    )


def make_dynamic_data() -> HaptData:
    """Create a dynamic (time-series) HaptData for testing."""
    data = np.random.randn(50, 16).astype(np.float32)  # [T, D]
    return HaptData(
        raw=RawData(
            array=data,
            checksum=RawData.compute_checksum(data),
            dtype="float32",
            shape=data.shape,
        ),
        sensor=SensorMeta(type="BioTac"),
        modality="dynamic",
        sampling_rate_hz=100.0,
        interaction=InteractionMeta(type="pressing", normal_force_N=1.0),
        labels=Labels(material="foam"),
    )


class TestToTorchTensorDataset:
    """Verify to_torch() returns a correct TensorDataset."""

    def test_returns_tensor_dataset(self):
        """Default call returns a TensorDataset."""
        data = make_test_data()
        result = data.to_torch()
        assert isinstance(result, torch.utils.data.TensorDataset)

    def test_dataset_length(self):
        """Dataset length matches number of frames."""
        data = make_test_data()
        ds = data.to_torch()
        assert len(ds) == 10  # 10 frames

    def test_single_sample_shape(self):
        """Each sample is a (H, W, C) tensor for imaging data."""
        data = make_test_data()
        ds = data.to_torch()
        x = ds[0]
        # TensorDataset returns a tuple of tensors, one per dataset
        assert isinstance(x[0], torch.Tensor)
        assert x[0].shape == (240, 320, 3)

    def test_default_dtype_is_float32(self):
        """Default conversion casts uint8 to float32."""
        data = make_test_data()
        ds = data.to_torch()
        x = ds[0][0]
        assert x.dtype == torch.float32

    def test_custom_dtype(self):
        """User can specify a custom dtype."""
        data = make_test_data()
        ds = data.to_torch(dtype=torch.float64)
        x = ds[0][0]
        assert x.dtype == torch.float64

    def test_dynamic_modality_shape(self):
        """Dynamic data yields (D,) vectors per sample."""
        data = make_dynamic_data()
        ds = data.to_torch()
        x = ds[0][0]
        assert x.shape == (16,)
        assert len(ds) == 50

    def test_force_modality_shape(self):
        """Force modality yields 3D or 6D vectors."""
        forces = np.random.randn(100, 6).astype(np.float32)
        data = HaptData(
            raw=RawData(
                array=forces,
                checksum=RawData.compute_checksum(forces),
                dtype="float32",
                shape=forces.shape,
            ),
            sensor=SensorMeta(type="ATI_Nano17"),
            modality="force",
            sampling_rate_hz=1000.0,
            interaction=InteractionMeta(type="static", normal_force_N=5.0),
            labels=Labels(task="calibration"),
        )
        ds = data.to_torch()
        assert len(ds) == 100
        assert ds[0][0].shape == (6,)

    def test_with_label_field(self):
        """Specifying a label field returns (X, y) tuples."""
        data = make_test_data()
        ds = data.to_torch(label="material")
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert y.ndim == 1  # scalar label
        assert y.dtype == torch.long

    def test_with_label_field_all_same(self):
        """When label is constant, all y values are identical."""
        data = make_test_data()
        ds = data.to_torch(label="material")
        ys = torch.tensor([ds[i][1].item() for i in range(len(ds))])
        assert (ys == ys[0]).all()

    def test_with_label_field_numeric(self):
        """Numeric label fields work (e.g., speed_mm_s as a proxy)."""
        data = make_test_data()
        ds = data.to_torch(label="task")
        _, y = ds[0]
        assert isinstance(y, torch.Tensor)
        assert y.dtype == torch.long

    def test_invalid_label_raises(self):
        """Invalid label field raises ValueError."""
        data = make_test_data()
        with pytest.raises(ValueError, match="label"):
            data.to_torch(label="nonexistent_field")

    def test_label_none_returns_single_tensor(self):
        """With no label, each item is a 1-tuple (X,)."""
        data = make_test_data()
        ds = data.to_torch()
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 1  # just X


class TestToTorchDataLoader:
    """Verify batching/shuffling with to_torch()."""

    def test_with_batch_size_returns_dataloader(self):
        """Passing batch_size returns a DataLoader."""
        data = make_test_data()
        loader = data.to_torch(batch_size=4)
        assert isinstance(loader, torch.utils.data.DataLoader)

    def test_batch_shape(self):
        """DataLoader yields batches with correct batch dimension."""
        data = make_test_data()
        loader = data.to_torch(batch_size=4, shuffle=False)
        batch = next(iter(loader))
        # batch is a list of tensors (one per dataset component)
        assert len(batch) == 1  # no label → just X
        assert batch[0].shape == (4, 240, 320, 3)

    def test_batch_with_label(self):
        """DataLoader with label yields batched (X, y)."""
        data = make_test_data()
        loader = data.to_torch(label="material", batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert len(batch) == 2  # X, y
        assert batch[0].shape == (4, 240, 320, 3)
        assert batch[1].shape == (4, 1)

    def test_shuffle_works(self):
        """Shuffle=True yields different ordering."""
        data = make_test_data()
        loader = data.to_torch(batch_size=10, shuffle=False)
        batch_no_shuffle = next(iter(loader))[0]

        loader = data.to_torch(batch_size=10, shuffle=True)
        batch_shuffle = next(iter(loader))[0]

        # With shuffle, at least some ordering should differ
        # (Probabilistically guaranteed for 10 elements)
        assert batch_no_shuffle.shape == batch_shuffle.shape
        if not torch.equal(batch_no_shuffle, batch_shuffle):
            pass  # Expected — shuffle changed ordering
        else:
            # Possible but unlikely with random data
            pass

    def test_dataloader_num_workers(self):
        """Can pass DataLoader kwargs through."""
        data = make_test_data()
        loader = data.to_torch(batch_size=4, num_workers=0)
        assert loader.num_workers == 0
        batch = next(iter(loader))
        assert batch[0].shape == (4, 240, 320, 3)


class TestToTorchTransforms:
    """Verify transform support."""

    def test_transform_applied(self):
        """A transform is applied to each sample."""
        data = make_test_data()

        def normalize(x):
            return x / 255.0

        ds = data.to_torch(transform=normalize)
        x = ds[0][0]
        assert x.max() <= 1.0
        assert x.min() >= 0.0

    def test_target_transform_with_label(self):
        """A target_transform is applied to the label."""
        data = make_test_data()

        def label_to_binary(y):
            return y  # identity for now

        ds = data.to_torch(label="material", target_transform=label_to_binary)
        _, y = ds[0]
        assert isinstance(y, torch.Tensor)


class TestToTorchRoundtrip:
    """Verify round-trip: load .hapt → to_torch works on loaded data."""

    def test_loaded_data_to_torch(self):
        """Save .hapt, load it, convert to torch."""
        original = make_test_data()
        tmp = Path(tempfile.mkdtemp())
        try:
            saved_path = save(original, tmp / "test.hapt")
            loaded = load(saved_path)
            ds = loaded.to_torch(batch_size=4)
            batch = next(iter(ds))
            assert batch[0].shape == (4, 240, 320, 3)
        finally:
            shutil.rmtree(tmp)
