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
from haptix.encoders import (
    SensorEncoder,
    get_encoder,
    list_encoders,
    load_trained,
    register_encoder,
)
from haptix.io import ChecksumError, HaptFormatError, load, save
from haptix.recorder import HaptRecorder
from haptix.sensors import get_sensor, list_sensors, register
from haptix.streaming import HaptArchive, open_archive
from haptix.unified import CrossModalEncoder, SharedForceEncoder, UnifiedEncoder

__version__ = "0.2.0"
__all__ = [
    "ChecksumError",
    "CrossModalEncoder",
    "HaptArchive",
    "HaptData",
    "HaptFormatError",
    "HaptRecorder",
    "Provenance",
    "RawData",
    "SensorEncoder",
    "SensorMeta",
    "SharedForceEncoder",
    "Source",
    "UnifiedEncoder",
    "cache_info",
    "cached_datasets",
    "clear_cache",
    "download_dataset",
    "get_dataset_info",
    "get_encoder",
    "get_sensor",
    "list_datasets",
    "list_encoders",
    "list_sensors",
    "load",
    "load_trained",
    "open_archive",
    "register",
    "register_encoder",
    "save",
    "verify_checksum",
]
