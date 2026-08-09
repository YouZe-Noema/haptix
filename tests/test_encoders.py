"""Tests for the per-sensor encoder registry (haptix.encoders).

Covers: registry behavior (register/get/list), the SensorEncoder protocol,
embedding-dim conventions (256 imaging / 128 dynamic), determinism,
surrogate fallback for unknown sensor types, save/load round-trip,
benchmark() contract, and top-level exports.
"""

import numpy as np
import pytest

import haptix
from haptix.core import HaptData, InteractionMeta, Labels, RawData, SensorMeta
from haptix.encoders import (
    SensorEncoder,
    get_encoder,
    list_encoders,
    register_encoder,
)
from haptix.encoders.base import (
    SurrogateEncoder,
    _default_dim_for,
    _modality_for,
)

SURROGATE_VERSION = "unified/shared-force/v0.1/surrogate"

REGISTERED_SENSORS = ["GelSight", "DIGIT", "CoroCapacitive", "BioTac_SP", "TacTip"]


def _make_imaging_hapt(num_frames=3, h=64, w=64, c=3, sensor_type="GelSight") -> HaptData:
    """Build synthetic imaging HaptData (mirrors tests/test_unified.py)."""
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
    """Build synthetic dynamic HaptData (mirrors tests/test_unified.py)."""
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


# ── Registry ──────────────────────────────────────────────────────────


class TestRegistry:
    def test_lazy_import_populates_registry(self):
        encoders = list_encoders()
        for name in REGISTERED_SENSORS:
            assert name in encoders, f"{name} missing from {encoders}"

    def test_get_encoder_returns_registered_instance(self):
        enc = get_encoder("CoroCapacitive")
        assert enc.sensor_type == "CoroCapacitive"
        assert enc.modality == "dynamic"
        assert enc.embedding_dim == 128

    def test_get_encoder_returns_new_instance_each_call(self):
        assert get_encoder("GelSight") is not get_encoder("GelSight")

    def test_register_encoder_decorator(self):
        @register_encoder("TestSensorXYZ", modality="dynamic")
        class _TestEncoder:
            sensor_type = "TestSensorXYZ"
            modality = "dynamic"
            embedding_dim = 64
            version = "encoders/test/v0.1"

            def encode(self, data):
                return np.zeros((1, 64), dtype=np.float32)

            def save(self, path):
                pass

            @classmethod
            def load(cls, path):
                return cls()

        try:
            assert "TestSensorXYZ" in list_encoders()
            enc = get_encoder("TestSensorXYZ")
            assert isinstance(enc, _TestEncoder)
            assert enc.sensor_type == "TestSensorXYZ"
            assert enc.modality == "dynamic"  # decorator set it
        finally:
            # Do not pollute the registry for other tests.
            from haptix.encoders import _registry

            _registry.pop("TestSensorXYZ", None)

    def test_top_level_exports(self):
        assert haptix.get_encoder is get_encoder
        assert haptix.list_encoders is list_encoders
        assert haptix.register_encoder is register_encoder
        assert haptix.SensorEncoder is SensorEncoder
        for name in ("get_encoder", "list_encoders", "register_encoder", "SensorEncoder"):
            assert name in haptix.__all__


# ── Protocol compliance ───────────────────────────────────────────────


class TestProtocol:
    @pytest.mark.parametrize("name", REGISTERED_SENSORS)
    def test_registered_encoders_satisfy_protocol(self, name):
        assert isinstance(get_encoder(name), SensorEncoder)

    def test_surrogate_satisfies_protocol(self):
        enc = get_encoder("NoSuchSensor")
        assert isinstance(enc, SensorEncoder)

    def test_version_format(self):
        assert get_encoder("GelSight").version == "encoders/gelsight/v0.1"
        assert get_encoder("CoroCapacitive").version == "encoders/coro/v0.1"

    def test_untrained_flag(self):
        for name in REGISTERED_SENSORS:
            assert get_encoder(name).trained is False


# ── Embedding-dim conventions ─────────────────────────────────────────


class TestDimConventions:
    def test_imaging_dim_256(self):
        for name in ("GelSight", "DIGIT"):
            assert get_encoder(name).embedding_dim == 256

    def test_dynamic_dim_128(self):
        for name in ("CoroCapacitive", "BioTac_SP", "TacTip"):
            assert get_encoder(name).embedding_dim == 128

    def test_gelsight_rgb_encodes_to_exact_dim(self):
        emb = get_encoder("GelSight").encode(_make_imaging_hapt(c=3))
        assert emb.shape == (3, 256)

    def test_gelsight_grayscale_encodes_to_exact_dim(self):
        emb = get_encoder("GelSight").encode(_make_imaging_hapt(c=1))
        assert emb.shape == (3, 256)

    def test_digit_encodes_to_exact_dim(self):
        emb = get_encoder("DIGIT").encode(_make_imaging_hapt(sensor_type="DIGIT"))
        assert emb.shape == (3, 256)

    def test_dynamic_pad(self):
        emb = get_encoder("CoroCapacitive").encode(_make_dynamic_hapt(features=5))
        assert emb.shape == (10, 128)

    def test_dynamic_truncate(self):
        emb = get_encoder("CoroCapacitive").encode(_make_dynamic_hapt(features=200))
        assert emb.shape == (10, 128)

    def test_biotac_and_tactip_dynamic(self):
        biotac = get_encoder("BioTac_SP").encode(
            _make_dynamic_hapt(features=23, sensor_type="BioTac_SP")
        )
        tactip = get_encoder("TacTip").encode(_make_dynamic_hapt(features=17, sensor_type="TacTip"))
        assert biotac.shape == (10, 128)
        assert tactip.shape == (10, 128)

    def test_single_frame(self):
        emb = get_encoder("DIGIT").encode(_make_imaging_hapt(num_frames=1))
        assert emb.shape == (1, 256)

    def test_imaging_rejects_wrong_ndim(self):
        arr = np.zeros((4, 5), dtype=np.float32)
        data = HaptData(
            raw=RawData(
                array=arr, checksum=RawData.compute_checksum(arr), dtype="float32", shape=arr.shape
            ),
            sensor=SensorMeta(type="GelSight"),
            modality="imaging",
            sampling_rate_hz=30.0,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(material="rubber"),
        )
        with pytest.raises(ValueError):
            get_encoder("GelSight").encode(data)


# ── Determinism ───────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_input_same_output(self):
        data = _make_imaging_hapt()
        a = get_encoder("GelSight").encode(data)
        b = get_encoder("GelSight").encode(data)
        assert np.array_equal(a, b)

    def test_dynamic_deterministic(self):
        data = _make_dynamic_hapt()
        a = get_encoder("CoroCapacitive").encode(data)
        b = get_encoder("CoroCapacitive").encode(data)
        assert np.array_equal(a, b)

    def test_surrogate_deterministic(self):
        data = _make_dynamic_hapt(sensor_type="UnknownSensor")
        a = get_encoder("UnknownSensor").encode(data)
        b = get_encoder("UnknownSensor").encode(data)
        assert np.array_equal(a, b)


# ── Surrogate fallback ────────────────────────────────────────────────


class TestSurrogateFallback:
    def test_unknown_sensor_returns_surrogate(self):
        enc = get_encoder("NoSuchSensor")
        assert isinstance(enc, SurrogateEncoder)
        assert enc.version == SURROGATE_VERSION
        assert enc.embedding_dim == 128  # unknown -> conservative dynamic default
        assert enc.trained is False

    def test_unknown_sensor_never_raises_and_is_shape_correct(self):
        emb = get_encoder("NoSuchSensor").encode(_make_dynamic_hapt(features=7))
        assert emb.shape == (10, 128)

    def test_known_alias_uses_imaging_dim(self):
        # GelSight_Mini has an adapter but no registered encoder -> surrogate
        # with the imaging dim (256).
        enc = get_encoder("GelSight_Mini")
        assert enc.version == SURROGATE_VERSION
        assert enc.embedding_dim == 256
        emb = enc.encode(_make_imaging_hapt(sensor_type="GelSight_Mini"))
        assert emb.shape == (3, 256)

    def test_dynamic_alias_uses_dynamic_dim(self):
        # "BioTac" (adapter alias) has no registered encoder -> surrogate 128.
        enc = get_encoder("BioTac")
        assert enc.version == SURROGATE_VERSION
        assert enc.embedding_dim == 128


# ── save / load ───────────────────────────────────────────────────────


class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        enc = get_encoder("CoroCapacitive")
        path = tmp_path / "coro_v0.1.npz"
        enc.save(path)
        loaded = type(enc).load(path)
        data = _make_dynamic_hapt()
        assert np.array_equal(enc.encode(data), loaded.encode(data))
        assert loaded.version == enc.version
        assert loaded.embedding_dim == enc.embedding_dim
        assert loaded.sensor_type == enc.sensor_type

    def test_save_load_imaging_roundtrip(self, tmp_path):
        enc = get_encoder("GelSight")
        path = tmp_path / "gelsight_v0.1.npz"
        enc.save(path)
        loaded = type(enc).load(path)
        data = _make_imaging_hapt()
        assert np.array_equal(enc.encode(data), loaded.encode(data))

    def test_load_type_mismatch_raises(self, tmp_path):
        from haptix.encoders.coro import CoroCapacitiveEncoder
        from haptix.encoders.gelsight import GelSightEncoder

        path = tmp_path / "coro.npz"
        CoroCapacitiveEncoder().save(path)
        with pytest.raises(ValueError):
            GelSightEncoder.load(path)

    def test_load_non_encoder_file_raises(self, tmp_path):
        from haptix.encoders.coro import CoroCapacitiveEncoder

        path = tmp_path / "not_an_encoder.npz"
        np.savez(path, foo=1)
        with pytest.raises(ValueError):
            CoroCapacitiveEncoder.load(path)


# ── benchmark contract ────────────────────────────────────────────────


class TestBenchmark:
    def test_structured_report_keys(self):
        report = get_encoder("GelSight").benchmark()
        assert {"dataset", "metric", "score", "split"} <= set(report)

    def test_untrained_score_is_none(self):
        assert get_encoder("GelSight").benchmark()["score"] is None

    def test_surrogate_benchmark(self):
        report = get_encoder("NoSuchSensor").benchmark()
        assert report["score"] is None


# ── dim helpers ───────────────────────────────────────────────────────


class TestDimHelpers:
    def test_default_dim_for(self):
        assert _default_dim_for("GelSight") == 256
        assert _default_dim_for("DIGIT_v2") == 256
        assert _default_dim_for("CoroCapacitive") == 128
        assert _default_dim_for("BioTac_SP") == 128
        assert _default_dim_for("TacTip") == 128
        assert _default_dim_for("TotallyUnknown") == 128  # conservative default

    def test_modality_for(self):
        assert _modality_for("GelSight") == "imaging"
        assert _modality_for("CoroCapacitive") == "dynamic"
        assert _modality_for("TotallyUnknown") == "dynamic"
