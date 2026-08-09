# haptix — Tactile Data Infrastructure

from haptix.core import HaptData, Provenance, RawData, SensorMeta, Source
from haptix.datasets import (
    cache_info,
    cached_datasets,
    clear_cache,
    download_dataset,
    get_dataset_info,
    list_datasets,
    verify_checksum,
)
from haptix.io import ChecksumError, HaptFormatError, load, save
from haptix.sensors import get_sensor, list_sensors, register
from haptix.unified import CrossModalEncoder, SharedForceEncoder, UnifiedEncoder

__version__ = "0.2.0"
__all__ = [
    "ChecksumError",
    "CrossModalEncoder",
    "HaptData",
    "HaptFormatError",
    "Provenance",
    "RawData",
    "SensorMeta",
    "SharedForceEncoder",
    "Source",
    "UnifiedEncoder",
    "cache_info",
    "cached_datasets",
    "clear_cache",
    "download_dataset",
    "get_dataset_info",
    "get_sensor",
    "list_datasets",
    "list_sensors",
    "load",
    "register",
    "save",
    "verify_checksum",
]
