"""
Dataset tools: catalog, download, and cache management.

Usage::

    >>> import haptix
    >>> haptix.list_datasets()
    ['robotouch', 'touch_and_go', 'ycb_slide']

    >>> haptix.get_dataset_info("touch_and_go")
    {'name': 'touch_and_go', 'url': ..., 'size_bytes': 58720256000, ...}

    >>> haptix.download_dataset("touch_and_go")  # downloads and caches
    PosixPath('/home/user/.haptix/cache/datasets/touch_and_go')

    >>> haptix.cached_datasets()
    ['touch_and_go']

    >>> haptix.cache_info()
    {'cache_path': '...', 'total_datasets': 1, 'total_bytes': 123456789}

    >>> haptix.clear_cache()  # removes all cached data
"""

from haptix.datasets.catalog import get_dataset_info, get_encoder_weights, list_datasets
from haptix.datasets.download import (
    ChecksumError,
    cache_info,
    cached_datasets,
    clear_cache,
    download_dataset,
    verify_checksum,
)

__all__ = [
    "ChecksumError",
    "cache_info",
    "cached_datasets",
    "clear_cache",
    "download_dataset",
    "get_dataset_info",
    "get_encoder_weights",
    "list_datasets",
    "verify_checksum",
]
