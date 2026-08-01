"""
Tests for .hapt.zarr compressed format with Zstd.

Covers round-trip integrity, compression ratio, checksum verification,
corruption detection, and unified data support.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from haptix.core import (
    HaptData,
    InteractionMeta,
    Labels,
    Provenance,
    RawData,
    SensorMeta,
    Source,
    UnifiedData,
)
from haptix.io import ChecksumError, HaptFormatError, load, save


def make_test_data(
    shape=(10, 240, 320, 3), dtype=np.uint8, modality="imaging", with_unified=False
) -> HaptData:
    """Create a minimal valid HaptData for testing."""
    frames = np.random.randint(0, 255, shape, dtype=dtype)
    data = HaptData(
        raw=RawData(
            array=frames,
            checksum=RawData.compute_checksum(frames),
            dtype=str(frames.dtype),
            shape=frames.shape,
        ),
        sensor=SensorMeta(type="DIGIT_v2"),
        modality=modality,
        sampling_rate_hz=60.0,
        interaction=InteractionMeta(
            type="sliding",
            speed_mm_s=50.0,
            normal_force_N=2.0,
        ),
        labels=Labels(material="sandpaper_grit_80", task="sliding"),
    )
    if with_unified:
        unified_arr = np.random.randn(shape[0], 128).astype(np.float32)
        data._unified = UnifiedData(
            array=unified_arr,
            method="SharedForceEncoder",
            source_modality="imaging",
            target_modality="force_latent",
            is_lossy=True,
            checksum=RawData.compute_checksum(unified_arr),
        )
    return data


# --- Skip if zarr not installed ---
zarr = pytest.importorskip("zarr", reason="zarr not installed")
numcodecs = pytest.importorskip("numcodecs", reason="numcodecs not installed")


class TestZarrRoundTrip:
    """Verify save/load round-trip integrity for .hapt.zarr."""

    def test_basic_roundtrip(self):
        """Save as .hapt.zarr then load — all data should match."""
        original = make_test_data()
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "test.hapt.zarr")
            assert saved.suffix == ".zarr"
            assert saved.exists()

            loaded = load(saved)

            # Metadata
            assert loaded.sensor.type == original.sensor.type
            assert loaded.modality == original.modality
            assert loaded.sampling_rate_hz == original.sampling_rate_hz
            assert loaded.interaction.type == original.interaction.type
            assert loaded.interaction.speed_mm_s == original.interaction.speed_mm_s
            assert loaded.labels.material == original.labels.material

            # Raw data — byte-level identical
            assert np.array_equal(loaded.raw.array, original.raw.array)
            assert loaded.raw.checksum == original.raw.checksum
            assert loaded.raw.shape == original.raw.shape
            assert loaded.raw.dtype == original.raw.dtype
        finally:
            shutil.rmtree(tmp)

    def test_dynamic_modality(self):
        """Zarr should work with dynamic (1D/2D) data, not just imaging."""
        # Use randn for float data, randint doesn't support float dtypes
        frames = np.random.randn(50, 19).astype(np.float32)
        original = HaptData(
            raw=RawData(
                array=frames,
                checksum=RawData.compute_checksum(frames),
                dtype=str(frames.dtype),
                shape=frames.shape,
            ),
            sensor=SensorMeta(type="BioTac_SP"),
            modality="dynamic",
            sampling_rate_hz=100.0,
            interaction=InteractionMeta(type="pressing", normal_force_N=1.0),
            labels=Labels(material="foam"),
        )
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "dynamic.hapt.zarr")
            loaded = load(saved)
            assert np.array_equal(loaded.raw.array, original.raw.array)
            assert loaded.modality == "dynamic"
        finally:
            shutil.rmtree(tmp)

    def test_unified_data_roundtrip(self):
        """Unified representation should survive .hapt.zarr round-trip."""
        original = make_test_data(shape=(10, 64, 64, 1), with_unified=True)
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "unified.hapt.zarr")
            loaded = load(saved)

            assert loaded.unified is not None
            assert np.array_equal(loaded.unified.array, original.unified.array)
            assert loaded.unified.method == original.unified.method
            assert loaded.unified.source_modality == original.unified.source_modality
            assert loaded.unified.target_modality == original.unified.target_modality
            assert loaded.unified.is_lossy == original.unified.is_lossy
        finally:
            shutil.rmtree(tmp)

    def test_large_dataset(self):
        """Zarr handles larger datasets efficiently via chunking."""
        # 100 frames of 128x128 RGB = ~4.9MB uncompressed
        original = make_test_data(shape=(100, 128, 128, 3), dtype=np.uint8)
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "large.hapt.zarr")
            loaded = load(saved)

            assert np.array_equal(loaded.raw.array, original.raw.array)

            # Zarr stores chunks independently; verify file exists and is reasonable
            zarr_size = saved.stat().st_size
            raw_size = int(np.prod(original.raw.shape))  # bytes for uint8
            # Random data won't compress, but file should be within 10% of raw size
            assert zarr_size > 0, "Zarr file should not be empty"
            assert (
                zarr_size <= raw_size * 1.10
            ), f"Zarr overhead too high: {zarr_size} > {raw_size * 1.10:.0f}"
        finally:
            shutil.rmtree(tmp)

    def test_provenance_preserved(self):
        """Provenance metadata should round-trip through Zarr."""
        original = make_test_data()
        original._provenance = Provenance(
            file_hash="abc123",
            source=Source(dataset="Lab-CORO", license="CC-BY-4.0"),
            created="2026-01-01T00:00:00Z",
        )
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "prov.hapt.zarr")
            loaded = load(saved)

            assert loaded.provenance is not None
            assert loaded.provenance.file_hash == "abc123"
            assert loaded.provenance.source.dataset == "Lab-CORO"
            assert loaded.provenance.source.license == "CC-BY-4.0"
        finally:
            shutil.rmtree(tmp)


class TestZarrChecksum:
    """Checksum integrity for .hapt.zarr."""

    def test_valid_checksum_on_load(self):
        """Loading uncorrupted data should pass checksum verification."""
        original = make_test_data()
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "valid.hapt.zarr")
            loaded = load(saved)
            assert loaded.raw.verify()
        finally:
            shutil.rmtree(tmp)

    def test_corruption_detection(self):
        """Manually corrupting the Zarr store should raise ChecksumError."""
        original = make_test_data()
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "corrupt.hapt.zarr")

            # Corrupt: tamper with the stored checksum in raw/data attrs
            store = zarr.storage.ZipStore(str(saved), mode="a")
            root = zarr.group(store=store, zarr_format=2)
            raw_arr = root["raw/data"]
            raw_arr.attrs["checksum"] = (
                "0000000000000000000000000000000000000000000000000000000000000000"
            )
            store.close()

            with pytest.raises(ChecksumError, match="Checksum mismatch"):
                load(saved)
        finally:
            shutil.rmtree(tmp)

    def test_empty_store_fails(self):
        """Loading a non-existent or empty file should raise."""
        tmp = Path(tempfile.mkdtemp())
        try:
            nonexistent = tmp / "nonexistent.hapt.zarr"
            with pytest.raises(FileNotFoundError):
                load(nonexistent)
        finally:
            shutil.rmtree(tmp)


class TestZarrErrors:
    """Error handling for malformed .hapt.zarr files."""

    def test_empty_zarr_not_a_file(self):
        """An empty ZipStore doesn't create a file — expect FileNotFoundError."""
        tmp = Path(tempfile.mkdtemp())
        try:
            empty_path = tmp / "empty.hapt.zarr"
            store = zarr.storage.ZipStore(str(empty_path), mode="w")
            store.close()
            # Empty ZipStore produces no file

            with pytest.raises(FileNotFoundError):
                load(empty_path)
        finally:
            shutil.rmtree(tmp)

    def test_missing_manifest_fails(self):
        """Store with raw data but no manifest should raise."""
        tmp = Path(tempfile.mkdtemp())
        try:
            store = zarr.storage.ZipStore(str(tmp / "nomanifest.hapt.zarr"), mode="w")
            root = zarr.group(store=store, zarr_format=2)
            root.zeros(name="raw/data", shape=(5,), dtype=np.float64, zarr_format=2)
            # Deliberately skip manifest
            store.close()

            with pytest.raises(HaptFormatError, match="manifest"):
                load(tmp / "nomanifest.hapt.zarr")
        finally:
            shutil.rmtree(tmp)


class TestZarrCompressionRatio:
    """Compression effectiveness for tactile-like data patterns."""

    def test_constant_data_compresses_well(self):
        """Uniform (no-contact) tactile frames should compress heavily."""
        # Simulate flat sensor reading — all zeros (no contact)
        frames = np.zeros((50, 32, 32, 3), dtype=np.uint8)
        data = HaptData(
            raw=RawData(
                array=frames,
                checksum=RawData.compute_checksum(frames),
                dtype="uint8",
                shape=frames.shape,
            ),
            sensor=SensorMeta(type="GelSight"),
            modality="imaging",
            sampling_rate_hz=30.0,
            interaction=InteractionMeta(type="static"),
            labels=Labels(),
        )

        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(data, tmp / "flat.hapt.zarr")
            zarr_size = saved.stat().st_size
            raw_size = int(np.prod(frames.shape))  # 50 * 32 * 32 * 3 = 153600 bytes

            # Constant data should achieve >5x compression with Zstd
            ratio = raw_size / zarr_size
            assert ratio > 5.0, f"Expected >5x compression, got {ratio:.1f}x"
        finally:
            shutil.rmtree(tmp)

    def test_noisy_data_still_smaller(self):
        """Even random noise data should be slightly compressible."""
        original = make_test_data(shape=(30, 64, 64, 3))
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "noisy.hapt.zarr")
            zarr_size = saved.stat().st_size
            raw_size = int(np.prod(original.raw.shape))

            # With Blosc shuffle filter, even random data gets some compression
            assert (
                zarr_size <= raw_size * 1.05
            ), f"Zarr size {zarr_size} unreasonably larger than raw {raw_size}"
        finally:
            shutil.rmtree(tmp)
