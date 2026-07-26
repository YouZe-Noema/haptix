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

from haptix.datasets.catalog import list_datasets, get_dataset_info
from haptix.datasets.download import (
    download_dataset,
    cached_datasets,
    cache_info,
    clear_cache,
)

__all__ = [
    "list_datasets",
    "get_dataset_info",
    "download_dataset",
    "cached_datasets",
    "cache_info",
    "clear_cache",
]
