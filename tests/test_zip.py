"""
Tests for .hapt.zip single-file archive format (stdlib DEFLATE).

Covers round-trip integrity, checksum verification, corruption detection,
error handling, unified data support, and compression effectiveness.
Mirrors test_zarr.py but uses the stdlib zipfile backend.
"""

import shutil
import tempfile
import zipfile
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


class TestZipRoundTrip:
    """Verify save/load round-trip integrity for .hapt.zip."""

    def test_basic_roundtrip(self):
        """Save as .hapt.zip then load — all data should match."""
        original = make_test_data()
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "test.hapt.zip")
            assert saved.suffix == ".zip"
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

    def test_archive_is_plain_zip(self):
        """The .hapt.zip file must be inspectable with stdlib zipfile."""
        original = make_test_data(shape=(4, 16, 16, 1))
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "plain.hapt.zip")

            with zipfile.ZipFile(saved) as zf:
                names = set(zf.namelist())

            assert "manifest.json" in names
            assert "raw/data.npy" in names
            assert "raw/checksum.sha256" in names
            assert "labels.json" in names
            assert "provenance.json" in names
        finally:
            shutil.rmtree(tmp)

    def test_dynamic_modality(self):
        """ZIP should work with dynamic (1D/2D) data, not just imaging."""
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
            saved = save(original, tmp / "dynamic.hapt.zip")
            loaded = load(saved)
            assert np.array_equal(loaded.raw.array, original.raw.array)
            assert loaded.modality == "dynamic"
        finally:
            shutil.rmtree(tmp)

    def test_unified_data_roundtrip(self):
        """Unified representation should survive .hapt.zip round-trip."""
        original = make_test_data(shape=(10, 64, 64, 1), with_unified=True)
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "unified.hapt.zip")
            loaded = load(saved)

            assert loaded.unified is not None
            assert np.array_equal(loaded.unified.array, original.unified.array)
            assert loaded.unified.method == original.unified.method
            assert loaded.unified.source_modality == original.unified.source_modality
            assert loaded.unified.target_modality == original.unified.target_modality
            assert loaded.unified.is_lossy == original.unified.is_lossy
        finally:
            shutil.rmtree(tmp)

    def test_provenance_preserved(self):
        """Provenance metadata should round-trip through ZIP."""
        original = make_test_data()
        original._provenance = Provenance(
            file_hash="abc123",
            source=Source(dataset="Lab-CORO", license="CC-BY-4.0"),
            created="2026-01-01T00:00:00Z",
        )
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "prov.hapt.zip")
            loaded = load(saved)

            assert loaded.provenance is not None
            assert loaded.provenance.file_hash == "abc123"
            assert loaded.provenance.source.dataset == "Lab-CORO"
            assert loaded.provenance.source.license == "CC-BY-4.0"
        finally:
            shutil.rmtree(tmp)


class TestZipChecksum:
    """Checksum integrity for .hapt.zip."""

    def test_valid_checksum_on_load(self):
        """Loading uncorrupted data should pass checksum verification."""
        original = make_test_data()
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "valid.hapt.zip")
            loaded = load(saved)
            assert loaded.raw.verify()
        finally:
            shutil.rmtree(tmp)

    def test_corruption_detection(self):
        """Corrupting the stored checksum should raise ChecksumError."""
        original = make_test_data()
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "corrupt.hapt.zip")

            # Corrupt: rewrite the checksum member with a wrong hash
            bad = "0" * 64 + "\n"
            with zipfile.ZipFile(saved, "a") as zf:
                # writestr replaces the existing member
                zf.writestr("raw/checksum.sha256", bad)

            with pytest.raises(ChecksumError, match="Checksum mismatch"):
                load(saved)
        finally:
            shutil.rmtree(tmp)

    def test_corrupted_npy_detected(self):
        """Corrupting the raw data itself should also raise ChecksumError."""
        original = make_test_data(shape=(5, 8, 8, 1))
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "corrupt2.hapt.zip")

            # Corrupt raw data: flip bytes inside raw/data.npy member
            with zipfile.ZipFile(saved, "r") as zf:
                members = [(item, zf.read(item.filename)) for item in zf.infolist()]
            raw_bytes = bytearray(next(c for n, c in members if n.filename == "raw/data.npy"))
            raw_bytes[-20] ^= 0xFF  # flip bits near end of file

            with zipfile.ZipFile(saved, "w") as zf:
                for item, content in members:
                    if item.filename == "raw/data.npy":
                        content = bytes(raw_bytes)
                    zf.writestr(item, content)

            with pytest.raises(ChecksumError, match="Checksum mismatch"):
                load(saved)
        finally:
            shutil.rmtree(tmp)


class TestZipErrors:
    """Error handling for malformed .hapt.zip files."""

    def test_missing_file_fails(self):
        """Loading a non-existent .hapt.zip should raise FileNotFoundError."""
        tmp = Path(tempfile.mkdtemp())
        try:
            nonexistent = tmp / "nonexistent.hapt.zip"
            with pytest.raises(FileNotFoundError):
                load(nonexistent)
        finally:
            shutil.rmtree(tmp)

    def test_not_a_zip_fails(self):
        """A .hapt.zip that is not a valid ZIP should raise HaptFormatError."""
        tmp = Path(tempfile.mkdtemp())
        try:
            bogus = tmp / "bogus.hapt.zip"
            bogus.write_bytes(b"this is not a zip archive at all")

            with pytest.raises(HaptFormatError, match="Not a valid"):
                load(bogus)
        finally:
            shutil.rmtree(tmp)

    def test_missing_manifest_fails(self):
        """Archive with raw data but no manifest should raise."""
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = tmp / "nomanifest.hapt.zip"
            with zipfile.ZipFile(saved, "w") as zf:
                zf.writestr("raw/data.npy", b"garbage")
                zf.writestr("raw/checksum.sha256", "0" * 64 + "\n")
                # Deliberately skip manifest.json and labels.json

            with pytest.raises(HaptFormatError, match="manifest"):
                load(saved)
        finally:
            shutil.rmtree(tmp)

    def test_flat_hapt_file_still_unsupported(self):
        """A bare .hapt file (no .zip suffix) stays unsupported."""
        tmp = Path(tempfile.mkdtemp())
        try:
            flat = tmp / "legacy.hapt"
            flat.write_bytes(b"anything")

            with pytest.raises(HaptFormatError, match="deprecated"):
                load(flat)
        finally:
            shutil.rmtree(tmp)


class TestZipCompression:
    """Compression effectiveness for the ZIP archive mode."""

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
            saved = save(data, tmp / "flat.hapt.zip")
            zip_size = saved.stat().st_size
            raw_size = int(np.prod(frames.shape))  # 50 * 32 * 32 * 3 = 153600 bytes

            # Constant data should achieve >10x compression with DEFLATE
            ratio = raw_size / zip_size
            assert ratio > 10.0, f"Expected >10x compression, got {ratio:.1f}x"
        finally:
            shutil.rmtree(tmp)

    def test_zip_vs_zarr_file_sizes(self):
        """ZIP archive should be comparable in size to directory format.

        For incompressible data, DEFLATE adds ~0 overhead beyond the
        archive header, so the ZIP should be no larger than the raw npy
        plus a small constant."""
        original = make_test_data(shape=(30, 64, 64, 3))
        tmp = Path(tempfile.mkdtemp())
        try:
            saved = save(original, tmp / "noisy.hapt.zip")
            zip_size = saved.stat().st_size
            raw_size = int(np.prod(original.raw.shape))

            # Random data: npy header + zip overhead, but no blowup
            assert (
                zip_size <= raw_size * 1.05
            ), f"ZIP size {zip_size} unreasonably larger than raw {raw_size}"
        finally:
            shutil.rmtree(tmp)
