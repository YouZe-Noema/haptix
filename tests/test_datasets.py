"""
Tests for dataset catalog and download/cache utilities.

These tests cover:
  - Catalog listing and lookup
  - Cache directory management
  - Download orchestration (mocked HTTP)
"""

import io
import shutil
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from haptix.datasets import (
    cache_info,
    cached_datasets,
    clear_cache,
    download_dataset,
    get_dataset_info,
    list_datasets,
)


class TestCatalog:
    """Verify the dataset catalog returns correct metadata."""

    def test_list_datasets(self):
        """list_datasets should return expected known datasets."""
        names = list_datasets()
        assert "touch_and_go" in names
        assert "ycb_slide" in names
        assert "robotouch" in names

    def test_get_dataset_info(self):
        """get_dataset_info should return metadata dict with required keys."""
        info = get_dataset_info("touch_and_go")
        assert info["name"] == "touch_and_go"
        assert "url" in info
        assert "description" in info
        assert "size_bytes" in info
        assert "sensor_type" in info
        assert "modality" in info
        assert "num_samples" in info
        assert "citation" in info

    def test_get_dataset_info_ycb(self):
        """YCB-Slide info should have correct sensor types."""
        info = get_dataset_info("ycb_slide")
        assert info["sensor_type"] in ("DIGIT_v2", "GelSight", "DIGIT_v2, GelSight")
        assert info["modality"] == "imaging"

    def test_get_dataset_info_unknown(self):
        """Unknown dataset should raise KeyError."""
        try:
            get_dataset_info("nonexistent_dataset")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_catalog_consistency(self):
        """Every dataset in the catalog must have all required metadata keys."""
        required = {
            "name",
            "url",
            "description",
            "size_bytes",
            "sensor_type",
            "modality",
            "num_samples",
            "citation",
        }
        for name in list_datasets():
            info = get_dataset_info(name)
            for key in required:
                assert key in info, f"{name} missing key: {key}"
            # size_bytes should be positive
            assert info["size_bytes"] > 0
            # num_samples should be positive
            assert info["num_samples"] > 0
            # url should be a valid-looking URL
            assert info["url"].startswith("http")

    def test_catalog_encoder_schema(self):
        """The optional ``encoder`` block (weights_url/weights_sha256) is validated."""
        from haptix.datasets.catalog import _CATALOG, _validate_catalog

        # A well-formed encoder block passes validation (import-time check).
        for name in list_datasets():
            info = get_dataset_info(name)
            if "encoder" in info:
                enc = info["encoder"]
                assert enc["weights_url"].startswith("http")
                assert len(enc["weights_sha256"]) == 64

        # Malformed encoder blocks are rejected by _validate_catalog.
        saved = _CATALOG["coro_tactile"].get("encoder")
        try:
            _CATALOG["coro_tactile"]["encoder"] = {"weights_url": "https://x/y.npz"}
            with pytest.raises(RuntimeError):
                _validate_catalog()
            _CATALOG["coro_tactile"]["encoder"] = {
                "weights_url": "https://x/y.npz",
                "weights_sha256": "not-hex",
            }
            with pytest.raises(RuntimeError):
                _validate_catalog()
        finally:
            if saved is None:
                _CATALOG["coro_tactile"].pop("encoder", None)
            else:
                _CATALOG["coro_tactile"]["encoder"] = saved

    def test_validate_catalog_missing_keys_and_bad_url(self):
        """``_validate_catalog`` rejects missing keys and non-http URLs."""
        from haptix.datasets.catalog import _CATALOG, _validate_catalog

        saved = dict(_CATALOG["touch_and_go"])
        try:
            del _CATALOG["touch_and_go"]["description"]
            with pytest.raises(RuntimeError, match="missing keys"):
                _validate_catalog()
            _CATALOG["touch_and_go"] = dict(saved)
            _CATALOG["touch_and_go"]["url"] = "ftp://example.com/x"
            with pytest.raises(RuntimeError, match="url must start with http"):
                _validate_catalog()
            _CATALOG["touch_and_go"] = dict(saved)
            _CATALOG["touch_and_go"]["size_bytes"] = 0
            with pytest.raises(RuntimeError, match="positive number"):
                _validate_catalog()
            _CATALOG["touch_and_go"] = dict(saved)
            _CATALOG["touch_and_go"]["sha256"] = "deadbeef"
            with pytest.raises(RuntimeError, match="64-char hex"):
                _validate_catalog()
        finally:
            _CATALOG["touch_and_go"] = saved

    def test_validate_catalog_encoder_not_dict(self):
        """Encoder block that is not a dict is rejected."""
        from haptix.datasets.catalog import _CATALOG, _validate_catalog

        saved = _CATALOG["coro_tactile"].get("encoder")
        try:
            _CATALOG["coro_tactile"]["encoder"] = "not-a-dict"
            with pytest.raises(RuntimeError, match="encoder must be a dict"):
                _validate_catalog()
        finally:
            if saved is None:
                _CATALOG["coro_tactile"].pop("encoder", None)
            else:
                _CATALOG["coro_tactile"]["encoder"] = saved


class TestCacheManagement:
    """Verify cache directory management."""

    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cache_info_empty(self):
        """cache_info should return empty stats when cache is empty."""
        info = cache_info(cache_dir=self.tmp / "cache")
        assert info["total_datasets"] == 0
        assert info["total_bytes"] == 0
        assert info["cache_path"] == str(self.tmp / "cache")

    def test_cached_datasets_empty(self):
        """cached_datasets should return empty list when nothing cached."""
        assert cached_datasets(cache_dir=self.tmp / "cache") == []

    def test_cache_info_with_data(self):
        """cache_info should reflect cached dataset size."""
        cache_root = self.tmp / "cache"
        (cache_root / "touch_and_go").mkdir(parents=True)
        (cache_root / "touch_and_go" / "some_file.npy").write_bytes(b"\x00" * 1024)

        info = cache_info(cache_dir=cache_root)
        assert info["total_datasets"] == 1
        assert info["total_bytes"] >= 1024

    def test_cached_datasets_with_data(self):
        """cached_datasets should list cached dataset names."""
        cache_root = self.tmp / "cache"
        (cache_root / "touch_and_go").mkdir(parents=True)
        (cache_root / "ycb_slide").mkdir(parents=True)

        cached = cached_datasets(cache_dir=cache_root)
        assert "touch_and_go" in cached
        assert "ycb_slide" in cached

    def test_clear_cache(self):
        """clear_cache should remove cached data."""
        cache_root = self.tmp / "cache"
        (cache_root / "touch_and_go").mkdir(parents=True)
        (cache_root / "touch_and_go" / "data.bin").write_bytes(b"test")

        assert cache_root.exists()
        clear_cache(cache_dir=cache_root)
        assert not cache_root.exists()


class TestDownload:
    """Verify download orchestrator (with mocked HTTP)."""

    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch("haptix.datasets.download._http_download")
    def test_download_unknown_dataset(self, mock_dl):
        """Downloading an unknown dataset should raise KeyError."""
        try:
            download_dataset("i_dont_exist", cache_dir=self.tmp / "cache")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass
        assert not mock_dl.called

    @patch("haptix.datasets.download._http_download")
    def test_download_success(self, mock_dl):
        """Download should call _http_download and store result."""
        cache_root = self.tmp / "cache"

        # Mock the download function to create a real (tiny) tar.gz archive
        def fake_download(url, dest):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                data = b"fake archive content"
                info = tarfile.TarInfo(name="data.txt")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            dest.write_bytes(buf.getvalue())

        mock_dl.side_effect = fake_download

        result = download_dataset("touch_and_go", cache_dir=cache_root)
        assert result.exists()
        assert result.name == "touch_and_go"
        # The URL for touch_and_go ends in .tar.gz, so _maybe_extract
        # extracts the archive into the dataset dir.
        assert (result / "data.txt").read_bytes() == b"fake archive content"
        mock_dl.assert_called_once()

    @patch("haptix.datasets.download._http_download")
    def test_download_idempotent(self, mock_dl):
        """Download should skip if already cached and return cached path."""
        cache_root = self.tmp / "cache"

        # Pre-populate cache
        cached_dir = cache_root / "touch_and_go"
        cached_dir.mkdir(parents=True)
        (cached_dir / "data.txt").write_bytes(b"existing cached data")

        result = download_dataset("touch_and_go", cache_dir=cache_root)
        assert result == cached_dir
        # _http_download should NOT have been called
        assert not mock_dl.called

    @patch("haptix.datasets.download._http_download")
    def test_download_force_redownload(self, mock_dl):
        """Force=True should re-download even if cached."""
        cache_root = self.tmp / "cache"

        # Pre-populate cache
        cached_dir = cache_root / "touch_and_go"
        cached_dir.mkdir(parents=True)
        (cached_dir / "data.txt").write_bytes(b"old data")

        def fake_download(url, dest):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                data = b"new data"
                info = tarfile.TarInfo(name="data.txt")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            dest.write_bytes(buf.getvalue())

        mock_dl.side_effect = fake_download

        result = download_dataset("touch_and_go", cache_dir=cache_root, force=True)
        # Force re-download should replace the old cached content
        assert result.exists()
        assert (result / "data.txt").read_bytes() == b"new data"
        mock_dl.assert_called_once()


class TestDownloadEdgePaths:
    """Failure / cleanup paths for ``download_dataset`` (no network)."""

    def setup_method(self):
        self.tmp = Path(tempfile.mkdtemp())

    def teardown_method(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _patch_catalog(self, monkeypatch, info: dict):
        """Drive downloads through a temporary catalog entry (not the shipped one)."""
        monkeypatch.setattr(
            "haptix.datasets.download.get_dataset_info",
            lambda name: {**info, "name": name},
        )

    def test_empty_cache_dir_is_cache_miss_and_cleaned(self, monkeypatch):
        """Pre-existing but empty dataset dir is removed and treated as a miss."""
        cache_root = self.tmp / "cache"
        empty = cache_root / "toy_ds"
        empty.mkdir(parents=True)
        assert empty.exists() and not any(empty.iterdir())

        payload = b"fresh-bytes"
        self._patch_catalog(
            monkeypatch,
            {
                "url": "https://example.test/toy_ds/payload.bin",
                "description": "edge",
                "size_bytes": 1,
                "sensor_type": "DIGIT",
                "modality": "imaging",
                "num_samples": 1,
                "citation": "n/a",
            },
        )

        def fake_download(url, dest):
            dest.write_bytes(payload)

        monkeypatch.setattr("haptix.datasets.download._http_download", fake_download)

        result = download_dataset("toy_ds", cache_dir=cache_root, extract=False)
        assert result == empty
        assert (result / "payload.bin").read_bytes() == payload

    def test_url_without_filename_uses_download_temp_name(self, monkeypatch):
        """Catalog URL whose path yields no filename → ``<name>.download``."""
        cache_root = self.tmp / "cache"
        self._patch_catalog(
            monkeypatch,
            {
                "url": "/",  # path yields empty filename after rstrip/split
                "description": "edge",
                "size_bytes": 1,
                "sensor_type": "DIGIT",
                "modality": "imaging",
                "num_samples": 1,
                "citation": "n/a",
            },
        )

        seen = {}

        def fake_download(url, dest):
            seen["tmp"] = dest
            dest.write_bytes(b"named-fallback")

        monkeypatch.setattr("haptix.datasets.download._http_download", fake_download)

        result = download_dataset("nofile_ds", cache_dir=cache_root, extract=False)
        assert seen["tmp"].name == ".nofile_ds.nofile_ds.download"
        assert (result / "nofile_ds.download").read_bytes() == b"named-fallback"

    def test_sha256_mismatch_cleans_temp_and_empty_dir(self, monkeypatch):
        """Pinned sha256 mismatch raises; partial temp and empty dataset dir gone."""
        import hashlib

        from haptix.io import ChecksumError

        cache_root = self.tmp / "cache"
        payload = b"wrong-payload"
        expected = "0" * 64
        self._patch_catalog(
            monkeypatch,
            {
                "url": "https://example.test/chk/payload.bin",
                "description": "edge",
                "size_bytes": 1,
                "sensor_type": "DIGIT",
                "modality": "imaging",
                "num_samples": 1,
                "citation": "n/a",
                "sha256": expected,
            },
        )

        def fake_download(url, dest):
            dest.write_bytes(payload)

        monkeypatch.setattr("haptix.datasets.download._http_download", fake_download)

        with pytest.raises(ChecksumError):
            download_dataset("chk_ds", cache_dir=cache_root)

        dataset_dir = cache_root / "chk_ds"
        tmp_path = cache_root / ".chk_ds.payload.bin"
        assert not tmp_path.exists()
        assert not dataset_dir.exists()
        # Sanity: the written payload really would not match.
        assert hashlib.sha256(payload).hexdigest() != expected

    def test_sha256_match_succeeds(self, monkeypatch):
        """Pinned sha256 that matches leaves the dataset cached."""
        import hashlib

        cache_root = self.tmp / "cache"
        payload = b"good-payload"
        digest = hashlib.sha256(payload).hexdigest()
        self._patch_catalog(
            monkeypatch,
            {
                "url": "https://example.test/ok/payload.bin",
                "description": "edge",
                "size_bytes": 1,
                "sensor_type": "DIGIT",
                "modality": "imaging",
                "num_samples": 1,
                "citation": "n/a",
                "sha256": digest,
            },
        )

        def fake_download(url, dest):
            dest.write_bytes(payload)

        monkeypatch.setattr("haptix.datasets.download._http_download", fake_download)

        result = download_dataset("ok_ds", cache_dir=cache_root, extract=False)
        assert (result / "payload.bin").read_bytes() == payload

    def test_extract_false_moves_without_extracting(self, monkeypatch):
        """``extract=False`` moves the downloaded file into the dataset dir as-is."""
        cache_root = self.tmp / "cache"
        self._patch_catalog(
            monkeypatch,
            {
                "url": "https://example.test/raw/data.bin",
                "description": "edge",
                "size_bytes": 1,
                "sensor_type": "DIGIT",
                "modality": "imaging",
                "num_samples": 1,
                "citation": "n/a",
            },
        )

        def fake_download(url, dest):
            dest.write_bytes(b"no-extract")

        monkeypatch.setattr("haptix.datasets.download._http_download", fake_download)

        result = download_dataset("raw_ds", cache_dir=cache_root, extract=False)
        assert (result / "data.bin").read_bytes() == b"no-extract"
        # No extraction side-effects — just the moved file.
        assert sorted(p.name for p in result.iterdir()) == ["data.bin"]

    def test_http_download_failure_leaves_no_partial_temp(self, monkeypatch):
        """``_http_download`` raising cleans the temp file and empty dataset dir."""
        cache_root = self.tmp / "cache"
        self._patch_catalog(
            monkeypatch,
            {
                "url": "https://example.test/fail/payload.bin",
                "description": "edge",
                "size_bytes": 1,
                "sensor_type": "DIGIT",
                "modality": "imaging",
                "num_samples": 1,
                "citation": "n/a",
            },
        )

        def fake_download(url, dest):
            dest.write_bytes(b"partial")
            raise RuntimeError("simulated download failure")

        monkeypatch.setattr("haptix.datasets.download._http_download", fake_download)

        with pytest.raises(RuntimeError, match="simulated download failure"):
            download_dataset("fail_ds", cache_dir=cache_root)

        assert not (cache_root / ".fail_ds.payload.bin").exists()
        assert not (cache_root / "fail_ds").exists()


class TestCachePath:
    """Test the cache path configuration."""

    def test_custom_cache_path(self):
        """cache_info should reflect the configurable cache path."""
        custom_path = Path("/tmp/custom_haptix_cache")
        info = cache_info(cache_dir=custom_path)
        assert info["cache_path"] == str(custom_path)
