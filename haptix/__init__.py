# haptix — Tactile Data Infrastructure

from haptix.browser import (
    episode_summary,
    find_hapt_files,
    frame_array,
    frame_image,
    frame_signals,
    make_gallery_dataframe,
    scan_directory,
    signal_trace,
    unified_trace,
)
from haptix.core import HaptData, Provenance, RawData, SensorMeta, Source
from haptix.datasets import (
    cache_info,
    cached_datasets,
    clear_cache,
    download_dataset,
    get_dataset_info,
    get_encoder_weights,
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
from haptix.torch_dataset import TemporalDataset, WindowedDataset
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
    "TemporalDataset",
    "UnifiedEncoder",
    "WindowedDataset",
    "cache_info",
    "cached_datasets",
    "clear_cache",
    "download_dataset",
    "episode_summary",
    "find_hapt_files",
    "frame_array",
    "frame_image",
    "frame_signals",
    "get_dataset_info",
    "get_encoder",
    "get_encoder_weights",
    "get_sensor",
    "list_datasets",
    "list_encoders",
    "list_sensors",
    "load",
    "load_trained",
    "make_gallery_dataframe",
    "open_archive",
    "register",
    "register_encoder",
    "save",
    "scan_directory",
    "signal_trace",
    "unified_trace",
    "verify_checksum",
]
