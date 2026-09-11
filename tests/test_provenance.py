"""
Dedicated tests for Provenance, Source, and ProcessingStep.

Covers serialization round-trips, optional-field omission, from_dict defaults,
already-ProcessingStep handling, HaptData.provenance wiring, and save/load
auto-provenance persistence.
"""

import numpy as np

import haptix
from haptix.core import (
    HaptData,
    InteractionMeta,
    Labels,
    ProcessingStep,
    Provenance,
    RawData,
    SensorMeta,
    Source,
)
from haptix.io import load, save


def make_test_data(shape=(4, 16, 16, 3), dtype=np.uint8) -> HaptData:
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
        modality="imaging",
        sampling_rate_hz=60.0,
        interaction=InteractionMeta(
            type="sliding",
            speed_mm_s=50.0,
            normal_force_N=2.0,
        ),
        labels=Labels(material="sandpaper_grit_80", task="sliding"),
    )


class TestProcessingStep:
    """Unit tests for ProcessingStep serialization."""

    def test_to_dict_includes_name_and_params(self):
        """to_dict always includes name and params."""
        step = ProcessingStep(name="normalize", params={"axis": 0})
        d = step.to_dict()
        assert d["name"] == "normalize"
        assert d["params"] == {"axis": 0}
        assert "tool" not in d

    def test_to_dict_includes_tool_only_when_set(self):
        """tool key is present only when tool is non-None/non-empty."""
        with_tool = ProcessingStep(name="filter", params={}, tool="scipy")
        without_tool = ProcessingStep(name="filter", params={})
        assert with_tool.to_dict()["tool"] == "scipy"
        assert "tool" not in without_tool.to_dict()

    def test_from_dict_roundtrip(self):
        """ProcessingStep.from_dict(step.to_dict()) equals the original."""
        step = ProcessingStep(name="resize", params={"size": 64}, tool="pillow")
        assert ProcessingStep.from_dict(step.to_dict()) == step

    def test_from_dict_fills_defaults(self):
        """from_dict with only name fills empty params and tool=None."""
        step = ProcessingStep.from_dict({"name": "x"})
        assert step.name == "x"
        assert step.params == {}
        assert step.tool is None


class TestSource:
    """Unit tests for Source serialization."""

    def test_empty_to_dict_is_empty(self):
        """Source() with all None fields serializes to {}."""
        assert Source().to_dict() == {}

    def test_all_six_fields_in_to_dict(self):
        """Source with all fields set includes exactly those six keys."""
        source = Source(
            dataset="Lab-CORO",
            url="https://example.com/data",
            citation="Xia et al. 2026",
            license="CC-BY-4.0",
            collection_date="2026-01-15",
            sensor_calibration="cal_v3.json",
        )
        d = source.to_dict()
        assert d == {
            "dataset": "Lab-CORO",
            "url": "https://example.com/data",
            "citation": "Xia et al. 2026",
            "license": "CC-BY-4.0",
            "collection_date": "2026-01-15",
            "sensor_calibration": "cal_v3.json",
        }

    def test_partial_source_omits_unset_fields(self):
        """Partial Source only includes set keys."""
        assert Source(license="MIT").to_dict() == {"license": "MIT"}

    def test_from_dict_roundtrip(self):
        """Source.from_dict(source.to_dict()) equals the original."""
        source = Source(dataset="D", citation="C", license="L")
        assert Source.from_dict(source.to_dict()) == source

    def test_from_dict_empty_yields_all_none(self):
        """Source.from_dict({}) yields all-None fields."""
        source = Source.from_dict({})
        assert source == Source()
        assert source.dataset is None
        assert source.url is None
        assert source.citation is None
        assert source.license is None
        assert source.collection_date is None
        assert source.sensor_calibration is None


class TestProvenance:
    """Unit tests for Provenance serialization and defaults."""

    def test_minimal_defaults(self):
        """Minimal Provenance(file_hash=...) uses documented defaults."""
        p = Provenance(file_hash="abc")
        assert p.derived_from is None
        assert p.processing == []
        assert p.is_lossy is False
        assert p.source == Source()
        assert p.created == ""
        assert p.created_by == "haptix/0.2.0"

    def test_to_dict_has_all_seven_keys_and_serializes_processing(self):
        """to_dict always has seven keys; processing steps become dicts."""
        p = Provenance(
            file_hash="abc",
            processing=[ProcessingStep(name="denoise", params={"k": 3})],
        )
        d = p.to_dict()
        assert set(d.keys()) == {
            "file_hash",
            "derived_from",
            "processing",
            "is_lossy",
            "source",
            "created",
            "created_by",
        }
        assert len(d["processing"]) == 1
        assert isinstance(d["processing"][0], dict)
        assert d["processing"][0]["name"] == "denoise"

    def test_full_roundtrip_equality(self):
        """Full Provenance round-trips through to_dict/from_dict."""
        p = Provenance(
            file_hash="deadbeef",
            derived_from="parent_hash",
            processing=[
                ProcessingStep(name="crop", params={"roi": [0, 0, 10, 10]}),
                ProcessingStep(name="normalize", params={}, tool="haptix"),
            ],
            is_lossy=True,
            source=Source(dataset="Lab-CORO", license="CC-BY-4.0"),
            created="2026-03-01T12:00:00Z",
            created_by="haptix/0.2.0",
        )
        assert Provenance.from_dict(p.to_dict()) == p

    def test_from_dict_keeps_already_processing_step(self):
        """from_dict else branch keeps an already-ProcessingStep entry."""
        raw_step = ProcessingStep(name="raw")
        p = Provenance.from_dict({"file_hash": "h", "processing": [raw_step]})
        assert p.processing[0] is raw_step
        assert p.processing[0] == ProcessingStep(name="raw")

    def test_from_dict_empty_defaults(self):
        """from_dict({}) yields empty file_hash and default fields."""
        p = Provenance.from_dict({})
        assert p.file_hash == ""
        assert p.processing == []
        assert p.source == Source()
        assert p.derived_from is None
        assert p.is_lossy is False
        assert p.created == ""
        assert p.created_by == "haptix/0.2.0"


class TestHaptDataProvenance:
    """HaptData.provenance property wiring."""

    def test_default_provenance_is_none(self):
        """HaptData built without provenance has .provenance is None."""
        data = make_test_data()
        assert data.provenance is None

    def test_provenance_kwarg_exposed(self):
        """HaptData built with provenance= exposes it via .provenance."""
        prov = Provenance(file_hash="abc123", source=Source(dataset="Lab-CORO"))
        frames = np.zeros((2, 8, 8, 1), dtype=np.uint8)
        data = HaptData(
            raw=RawData(
                array=frames,
                checksum=RawData.compute_checksum(frames),
                dtype=str(frames.dtype),
                shape=frames.shape,
            ),
            sensor=SensorMeta(type="DIGIT_v2"),
            modality="imaging",
            sampling_rate_hz=30.0,
            interaction=InteractionMeta(type="pressing"),
            labels=Labels(material="foam"),
            provenance=prov,
        )
        assert data.provenance is prov
        assert data.provenance.file_hash == "abc123"
        assert data.provenance.source.dataset == "Lab-CORO"


class TestSaveLoadAutoProvenance:
    """Integration: save/load auto-generates provenance when missing."""

    def test_save_without_provenance_persists_defaults(self, tmp_path):
        """Saving data with no provenance writes auto-generated provenance.json."""
        original = make_test_data()
        assert original.provenance is None

        path = tmp_path / "ep.hapt"
        save(original, path)
        loaded = load(path)

        assert loaded.provenance is not None
        assert loaded.provenance.created_by == f"haptix/{haptix.__version__}"
        # _default_provenance() sets file_hash=""; save does not fill it in.
        assert isinstance(loaded.provenance.file_hash, str)
        assert loaded.provenance.file_hash == ""
        assert loaded.provenance.created != ""
        assert loaded.provenance.source == Source()
        assert loaded.provenance.processing == []
        assert loaded.provenance.is_lossy is False
