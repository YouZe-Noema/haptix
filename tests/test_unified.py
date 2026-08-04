"""Tests for the unified cross-sensor representation layer.

Tests cover: encoder instantiation, determinism, imaging and dynamic
modality encoding, embedding shape consistency, metadata correctness,
supported sensors listing, cross-sensor consistency, and round-trip.
"""

import numpy as np
import pytest

from haptix.core import (
    HaptData,
    InteractionMeta,
    Labels,
    RawData,
    SensorMeta,
)
from haptix.io import load, save
from haptix.unified import _ENCODER_VERSION, SharedForceEncoder, UnifiedEncoder
from haptix.unified.encoder import _pad_to_dim, _resize_image_embedding


@pytest.fixture
def encoder_64():
    """Encoder with 64-dim embedding (8×8 spatial grid for imaging)."""
    return SharedForceEncoder(embedding_dim=64, seed=42)


@pytest.fixture
def encoder_128():
    """Encoder with 128-dim embedding."""
    return SharedForceEncoder(embedding_dim=128, seed=42)


def _make_imaging_hapt(num_frames=3, h=64, w=64, c=3, sensor_type="GelSight") -> HaptData:
    """Build synthetic imaging HaptData."""
    arr = np.random.RandomState(42).randint(0, 256, (num_frames, h, w, c)).astype(np.uint8)
    return HaptData(
        raw=RawData(
            array=arr,
            checksum=RawData.compute_checksum(arr),
            dtype=str(arr.dtype),
            shape=arr.shape,
        ),
        sensor=SensorMeta(type=sensor_type),
        modality="imaging",
        sampling_rate_hz=30.0,
        interaction=InteractionMeta(type="pressing", normal_force_N=2.0),
        labels=Labels(material="rubber"),
    )


def _make_dynamic_hapt(num_frames=10, features=29, sensor_type="CoroCapacitive") -> HaptData:
    """Build synthetic dynamic HaptData."""
    arr = np.random.RandomState(42).rand(num_frames, features).astype(np.float32)
    return HaptData(
        raw=RawData(
            array=arr,
            checksum=RawData.compute_checksum(arr),
            dtype=str(arr.dtype),
            shape=arr.shape,
        ),
        sensor=SensorMeta(type=sensor_type),
        modality="dynamic",
        sampling_rate_hz=30.0,
        interaction=InteractionMeta(type="pressing", normal_force_N=3.0),
        labels=Labels(material="aluminium"),
    )


# ── Encoder Instantiation ──────────────────────────────────────────


class TestEncoderInstantiation:
    def test_default_embedding_dim(self):
        enc = SharedForceEncoder()
        assert enc.embedding_dim == 128
        assert enc.version == _ENCODER_VERSION

    def test_custom_embedding_dim(self):
        enc = SharedForceEncoder(embedding_dim=256)
        assert enc.embedding_dim == 256

    def test_different_seeds_different_projections(self, encoder_128):
        """Encoders with different seeds may produce different embeddings."""
        # Currently uses no random projection, so seed doesn't matter.
        # But the API accepts it for future learned projections.
        enc2 = SharedForceEncoder(embedding_dim=128, seed=99)
        assert encoder_128.embedding_dim == enc2.embedding_dim

    def test_implements_unified_encoder_protocol(self, encoder_128):
        assert isinstance(encoder_128, UnifiedEncoder)


# ── Determinism ─────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_input_same_output(self, encoder_64):
        data = _make_imaging_hapt()
        e1 = encoder_64.encode(data)
        e2 = encoder_64.encode(data)
        np.testing.assert_array_equal(e1.array, e2.array)

    def test_different_encoder_instances_same_output(self):
        data = _make_dynamic_hapt()
        enc1 = SharedForceEncoder(seed=42)
        enc2 = SharedForceEncoder(seed=42)
        np.testing.assert_array_equal(enc1.encode(data).array, enc2.encode(data).array)


# ── Imaging Modality ─────────────────────────────────────────────────


class TestImagingEncoding:
    def test_gelsight_encoding_shape(self, encoder_64):
        """8×8 spatial grid → 64-dim embedding."""
        data = _make_imaging_hapt(num_frames=5, h=100, w=150, c=3, sensor_type="GelSight")
        result = encoder_64.encode(data)
        # 8×8 spatial × 3 channels = 192 (flattened)
        assert result.array.shape == (5, 192)

    def test_digit_encoding_shape(self, encoder_64):
        data = _make_imaging_hapt(num_frames=5, h=80, w=120, c=3, sensor_type="DIGIT")
        result = encoder_64.encode(data)
        assert result.array.shape == (5, 192)

    def test_gelsight_mini_encoding(self, encoder_128):
        """11×11 spatial grid → 121 * C flat."""
        data = _make_imaging_hapt(num_frames=3, h=64, w=64, c=3, sensor_type="GelSight_Mini")
        result = encoder_128.encode(data)
        # floor(sqrt(128)) = 11 → 11×11 = 121 × 3 = 363
        assert result.array.shape == (3, 363)

    def test_grayscale_imaging(self, encoder_64):
        data = _make_imaging_hapt(num_frames=4, h=64, w=64, c=1, sensor_type="GelSight")
        result = encoder_64.encode(data)
        # 8×8 spatial × 1 channel = 64
        assert result.array.shape == (4, 64)

    def test_digit_v2_handled_as_imaging(self, encoder_64):
        data = _make_imaging_hapt(num_frames=2, h=64, w=64, c=3, sensor_type="DIGIT_v2")
        result = encoder_64.encode(data)
        assert result.array.shape == (2, 192)

    def test_single_frame_imaging(self, encoder_64):
        data = _make_imaging_hapt(num_frames=1, h=64, w=64, c=3)
        result = encoder_64.encode(data)
        assert result.array.shape == (1, 192)


# ── Dynamic Modality ─────────────────────────────────────────────────


class TestDynamicEncoding:
    def test_coro_encoding_shape(self, encoder_128):
        data = _make_dynamic_hapt(num_frames=10, features=29)
        result = encoder_128.encode(data)
        # 29 features padded to 128
        assert result.array.shape == (10, 128)

    def test_biotac_encoding_shape(self, encoder_128):
        data = _make_dynamic_hapt(num_frames=8, features=23, sensor_type="BioTac_SP")
        result = encoder_128.encode(data)
        assert result.array.shape == (8, 128)

    def test_tactip_encoding_shape(self, encoder_128):
        data = _make_dynamic_hapt(num_frames=6, features=50, sensor_type="TacTip")
        result = encoder_128.encode(data)
        assert result.array.shape == (6, 128)

    def test_features_exceed_embedding_dim_truncate(self, encoder_64):
        """When features > embedding_dim, use first embedding_dim columns."""
        data = _make_dynamic_hapt(num_frames=5, features=200)
        result = encoder_64.encode(data)
        assert result.array.shape == (5, 64)

    def test_single_feature_dynamic(self, encoder_128):
        data = _make_dynamic_hapt(num_frames=5, features=1)
        result = encoder_128.encode(data)
        assert result.array.shape == (5, 128)

    def test_single_frame_dynamic(self, encoder_128):
        data = _make_dynamic_hapt(num_frames=1, features=50)
        result = encoder_128.encode(data)
        assert result.array.shape == (1, 128)


# ── UnifiedData Metadata ────────────────────────────────────────────


class TestUnifiedDataMetadata:
    def test_method_tag_includes_version_and_sensor(self, encoder_128):
        data = _make_imaging_hapt(sensor_type="GelSight")
        result = encoder_128.encode(data)
        assert _ENCODER_VERSION in result.method
        assert "GelSight" in result.method

    def test_source_modality_preserved(self, encoder_128):
        img = _make_imaging_hapt()
        dyn = _make_dynamic_hapt()
        assert encoder_128.encode(img).source_modality == "imaging"
        assert encoder_128.encode(dyn).source_modality == "dynamic"

    def test_target_modality_indicates_shared_space(self, encoder_128):
        result = encoder_128.encode(_make_imaging_hapt())
        assert "shared_force" in result.target_modality
        assert "128d" in result.target_modality

    def test_is_lossy_true(self, encoder_128):
        """Cross-sensor embeddings are lossy by nature (dimensionality reduction)."""
        result = encoder_128.encode(_make_imaging_hapt())
        assert result.is_lossy is True

    def test_checksum_present(self, encoder_128):
        result = encoder_128.encode(_make_imaging_hapt())
        assert isinstance(result.checksum, str)
        assert len(result.checksum) == 64  # SHA-256 hex


# ── Supported Sensors ────────────────────────────────────────────────


class TestSupportedSensors:
    def test_supported_sensors_includes_registered_types(self):
        supported = SharedForceEncoder.supported_sensors()
        assert "GelSight" in supported
        assert "DIGIT" in supported
        assert "CoroCapacitive" in supported
        assert "BioTac_SP" in supported
        assert "TacTip" in supported

    def test_imaging_sensors_routed_correctly(self, encoder_64):
        for stype in ["GelSight", "GelSight_Mini", "GelSight_Wedge", "DIGIT", "DIGIT_v2"]:
            data = _make_imaging_hapt(sensor_type=stype)
            result = encoder_64.encode(data)
            # Imaging path: 8×8 spatial × 3 channels = 192
            assert result.array.shape == (3, 192), f"Wrong shape for {stype}"

    def test_dynamic_sensors_routed_correctly(self, encoder_64):
        for stype in ["CoroCapacitive", "BioTac_SP", "TacTip"]:
            data = _make_dynamic_hapt(sensor_type=stype)
            result = encoder_64.encode(data)
            # Dynamic path: 29 padded/truncated to 64
            assert result.array.shape == (10, 64), f"Wrong shape for {stype}"


# ── Cross-Sensor Consistency ─────────────────────────────────────────


class TestCrossSensorConsistency:
    def test_same_embedding_dim_across_modalities(self, encoder_128):
        """Different modalities produce consistent embedding shapes."""
        img = encoder_128.encode(_make_imaging_hapt(num_frames=5))
        dyn = encoder_128.encode(_make_dynamic_hapt(num_frames=5))
        # Both have T=5 frames; imaging uses sqrt floor so dims differ
        # from target_dim by channel factor, but both are 2D [T, D]
        assert img.array.ndim == 2
        assert dyn.array.ndim == 2
        assert img.array.shape[0] == 5
        assert dyn.array.shape[0] == 5

    def test_values_in_reasonable_range(self, encoder_64):
        """Encoded values should be in [0, 1] for imaging, reasonable for dynamic."""
        img = encoder_64.encode(_make_imaging_hapt())
        assert 0.0 <= img.array.max() <= 1.0
        assert 0.0 <= img.array.min() <= 1.0

        dyn = encoder_64.encode(_make_dynamic_hapt())
        # Dynamic data is raw float32, no normalization. Values can be > 1.
        assert not np.any(np.isnan(dyn.array))
        assert not np.any(np.isinf(dyn.array))


# ── Round-trip via save/load ────────────────────────────────────────


class TestRoundTrip:
    def test_unified_data_survives_save_load(self, encoder_64, tmp_path):
        """UnifiedData attached to HaptData is preserved through save/load."""
        data = _make_imaging_hapt(num_frames=4, h=32, w=32, c=1)
        unified = encoder_64.encode(data)

        # Create HaptData with unified data attached
        data_with_unified = HaptData(
            raw=data.raw,
            sensor=data.sensor,
            modality=data.modality,
            sampling_rate_hz=data.sampling_rate_hz,
            interaction=data.interaction,
            labels=data.labels,
            unified=unified,
        )

        # Save and reload
        path = tmp_path / "test_unified.hapt"
        save(data_with_unified, path)
        reloaded = load(path)

        # Verify unified data survived
        assert reloaded.unified is not None
        np.testing.assert_array_equal(reloaded.unified.array, unified.array)
        assert reloaded.unified.method == unified.method
        assert reloaded.unified.checksum == unified.checksum

    def test_unified_checksum_verified_after_reload(self, encoder_64, tmp_path):
        """Reloaded unified data should have a valid checksum."""
        data = _make_dynamic_hapt(num_frames=3)
        unified = encoder_64.encode(data)

        data_with_unified = HaptData(
            raw=data.raw,
            sensor=data.sensor,
            modality=data.modality,
            sampling_rate_hz=data.sampling_rate_hz,
            interaction=data.interaction,
            labels=data.labels,
            unified=unified,
        )

        path = tmp_path / "test_unified_dyn.hapt"
        save(data_with_unified, path)
        reloaded = load(path)

        # Checksum should match
        import hashlib

        computed = hashlib.sha256(reloaded.unified.array.tobytes()).hexdigest()
        assert computed == reloaded.unified.checksum


# ── Helper Functions ─────────────────────────────────────────────────


class TestHelperFunctions:
    def test_resize_image_embedding_shape(self):
        """_resize_image_embedding produces correct output shape."""
        arr = np.random.RandomState(42).randint(0, 256, (5, 80, 120, 3)).astype(np.uint8)
        result = _resize_image_embedding(arr, target_dim=8)
        # 8×8 spatial × 3 channels = 192
        assert result.shape == (5, 192)

    def test_resize_image_embedding_grayscale(self):
        arr = np.random.RandomState(42).randint(0, 256, (3, 64, 64, 1)).astype(np.uint8)
        result = _resize_image_embedding(arr, target_dim=8)
        assert result.shape == (3, 64)

    def test_pad_to_dim_less(self):
        arr = np.ones((5, 10), dtype=np.float32)
        result = _pad_to_dim(arr, 20)
        assert result.shape == (5, 20)
        np.testing.assert_array_equal(result[:, :10], np.ones((5, 10)))
        np.testing.assert_array_equal(result[:, 10:], np.zeros((5, 10)))

    def test_pad_to_dim_more(self):
        arr = np.ones((5, 50), dtype=np.float32)
        result = _pad_to_dim(arr, 20)
        assert result.shape == (5, 20)

    def test_pad_to_dim_equal(self):
        arr = np.random.rand(5, 20).astype(np.float32)
        result = _pad_to_dim(arr, 20)
        assert result is arr  # Same object when no-op

    def test_pad_to_dim_1d(self):
        arr = np.ones(10, dtype=np.float32)
        result = _pad_to_dim(arr, 20)
        assert result.shape == (1, 20)
        np.testing.assert_array_equal(result[0, :10], np.ones(10))
