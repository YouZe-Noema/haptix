"""
Tests for dataset catalog and download/cache utilities.

These tests cover:
  - Catalog listing and lookup
  - Cache directory management
  - Download orchestration (mocked HTTP)
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

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

        # Mock the download function to create a fake archive
        def fake_download(url, dest):
            dest.write_bytes(b"fake archive content")

        mock_dl.side_effect = fake_download

        result = download_dataset("touch_and_go", cache_dir=cache_root)
        assert result.exists()
        assert result.name == "touch_and_go"
        # Since it's not a recognized archive format, _maybe_extract
        # returns the dest path itself — so the archive stays as-is.
        # We just check the dir exists and has content.
        assert any(result.iterdir())
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
            dest.write_bytes(b"new data")

        mock_dl.side_effect = fake_download

        result = download_dataset("touch_and_go", cache_dir=cache_root, force=True)
        # Since the file is not a recognized archive, it stays as the download
        # file inside the dataset directory
        assert result.exists()
        mock_dl.assert_called_once()


class TestCachePath:
    """Test the cache path configuration."""

    def test_custom_cache_path(self):
        """cache_info should reflect the configurable cache path."""
        custom_path = Path("/tmp/custom_haptix_cache")
        info = cache_info(cache_dir=custom_path)
        assert info["cache_path"] == str(custom_path)
