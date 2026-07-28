"""
Core data structures for .hapt format.

HaptData is the in-memory representation of a .hapt file.
It enforces the invariant that raw data is never modified.
"""

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import torch

# Lazy torch import — torch is an optional dependency
try:
    import torch as _torch
except ImportError:
    _torch = None


Modality = Literal["imaging", "dynamic", "force", "multimodal"]


@dataclass(frozen=True)
class SensorMeta:
    """Immutable sensor metadata."""

    type: str
    serial: str | None = None
    calibration_date: str | None = None
    calibration_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"type": self.type}
        if self.serial:
            d["serial"] = self.serial
        if self.calibration_date:
            d["calibration_date"] = self.calibration_date
        if self.calibration_params:
            d["calibration_params"] = self.calibration_params
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SensorMeta":
        return cls(
            type=d["type"],
            serial=d.get("serial"),
            calibration_date=d.get("calibration_date"),
            calibration_params=d.get("calibration_params", {}),
        )


@dataclass(frozen=True)
class InteractionMeta:
    """Immutable interaction parameters. This is the MANDATORY metadata that
    differentiates .hapt from generic container formats."""

    type: str  # "sliding", "pressing", "grasping", "static"
    speed_mm_s: float | None = None
    normal_force_N: float | None = None
    approach_angle_deg: float | None = None
    temperature_C: float | None = None
    humidity_pct: float | None = None

    def to_dict(self) -> dict:
        d = {"type": self.type}
        for key in [
            "speed_mm_s",
            "normal_force_N",
            "approach_angle_deg",
            "temperature_C",
            "humidity_pct",
        ]:
            v = getattr(self, key)
            if v is not None:
                d[key] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "InteractionMeta":
        return cls(
            type=d["type"],
            speed_mm_s=d.get("speed_mm_s"),
            normal_force_N=d.get("normal_force_N"),
            approach_angle_deg=d.get("approach_angle_deg"),
            temperature_C=d.get("temperature_C"),
            humidity_pct=d.get("humidity_pct"),
        )


@dataclass(frozen=True)
class Labels:
    """Immutable annotation labels."""

    material: str | None = None
    material_category: str | None = None
    object_name: str | None = None
    object_category: str | None = None
    task: str | None = None
    custom_tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {}
        if self.material:
            d["material"] = self.material
        if self.material_category:
            d["material_category"] = self.material_category
        if self.object_name:
            d["object"] = self.object_name
        if self.object_category:
            d["object_category"] = self.object_category
        if self.task:
            d["task"] = self.task
        if self.custom_tags:
            d["custom_tags"] = self.custom_tags
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Labels":
        return cls(
            material=d.get("material"),
            material_category=d.get("material_category"),
            object_name=d.get("object"),
            object_category=d.get("object_category"),
            task=d.get("task"),
            custom_tags=d.get("custom_tags", []),
        )


@dataclass(frozen=True)
class RawData:
    """Immutable raw sensor data with checksum."""

    array: np.ndarray
    checksum: str
    dtype: str
    shape: tuple

    def numpy(self) -> np.ndarray:
        """Return a read-only view of the raw data."""
        return self.array.view()

    def verify(self) -> bool:
        """Verify checksum matches data."""
        computed = hashlib.sha256(self.array.tobytes()).hexdigest()
        return computed == self.checksum

    @staticmethod
    def compute_checksum(arr: np.ndarray) -> str:
        return hashlib.sha256(arr.tobytes()).hexdigest()


@dataclass(frozen=True)
class UnifiedData:
    """Optional cross-sensor unified representation."""

    array: np.ndarray
    method: str
    source_modality: str
    target_modality: str
    is_lossy: bool
    checksum: str

    def numpy(self) -> np.ndarray:
        return self.array.view()


class _TransformDataset:
    """Wraps a Dataset to apply transforms to samples and targets.

    Only available when torch is installed.
    """

    def __init__(self, dataset, transform=None, target_transform=None):
        if _torch is None:
            raise ImportError(
                "torch is required for to_torch(). Install with: pip install 'haptix[torch]'"
            )
        self._dataset = dataset
        self._transform = transform
        self._target_transform = target_transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        items = self._dataset[idx]
        if not isinstance(items, (list, tuple)):
            items = (items,)
        items = list(items)

        # Apply transform to X (first element)
        if self._transform is not None and len(items) > 0:
            items[0] = self._transform(items[0])

        # Apply target_transform to y (second element, if present)
        if self._target_transform is not None and len(items) > 1:
            items[1] = self._target_transform(items[1])

        return tuple(items)


class HaptData:
    """Immutable in-memory representation of .hapt file contents.

    Once loaded, raw data is frozen. Use to_hapt() to write a modified copy."""

    def __init__(
        self,
        raw: RawData,
        sensor: SensorMeta,
        modality: Modality,
        sampling_rate_hz: float,
        interaction: InteractionMeta,
        labels: Labels,
        unified: UnifiedData | None = None,
        version: str = "0.1.0",
    ):
        self._raw = raw
        self._sensor = sensor
        self._modality = modality
        self._sampling_rate_hz = sampling_rate_hz
        self._interaction = interaction
        self._labels = labels
        self._unified = unified
        self._version = version

    @property
    def raw(self) -> RawData:
        return self._raw

    @property
    def sensor(self) -> SensorMeta:
        return self._sensor

    @property
    def modality(self) -> Modality:
        return self._modality

    @property
    def sampling_rate_hz(self) -> float:
        return self._sampling_rate_hz

    @property
    def interaction(self) -> InteractionMeta:
        return self._interaction

    @property
    def labels(self) -> Labels:
        return self._labels

    @property
    def unified(self) -> UnifiedData | None:
        return self._unified

    @property
    def version(self) -> str:
        return self._version

    def _to_tensor(self, arr: np.ndarray, dtype) -> "torch.Tensor":
        """Convert numpy array to torch tensor with safe type handling."""
        import torch

        if arr.dtype.kind in ("i", "u"):
            # Integer types: convert to float first then cast
            return torch.from_numpy(arr.astype(np.float64)).to(dtype)
        return torch.from_numpy(arr).to(dtype)

    def _encode_label(self, value) -> int:
        """Encode a scalar label value to an integer for classification.

        Uses deterministic hashing (SHA-256 truncation) for string values,
        not Python's salted hash(). This ensures consistent encoding across
        processes and machines — critical for reproducibility in ML.

        Handles strings, ints, floats, None.
        """
        if value is None:
            return 0
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            # Deterministic encoding via SHA-256 — NOT Python's salted hash()
            return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)
        # Fallback for unexpected types
        return int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8], 16)

    def _resolve_label_field(self, label: str):
        """Resolve a label field name to its value.

        Searches labels, interaction, and sensor metadata.
        Raises ValueError if not found.
        """
        # Check Labels dataclass fields
        label_fields = {
            "material": self._labels.material,
            "material_category": self._labels.material_category,
            "object_name": self._labels.object_name,
            "object_category": self._labels.object_category,
            "task": self._labels.task,
        }
        if label in label_fields:
            return label_fields[label]

        # Check InteractionMeta fields
        interaction_fields = {
            "speed_mm_s": self._interaction.speed_mm_s,
            "normal_force_N": self._interaction.normal_force_N,
            "approach_angle_deg": self._interaction.approach_angle_deg,
            "temperature_C": self._interaction.temperature_C,
            "humidity_pct": self._interaction.humidity_pct,
            "interaction_type": self._interaction.type,
        }
        if label in interaction_fields:
            return interaction_fields[label]

        # Check sensor type
        if label == "sensor_type":
            return self._sensor.type

        raise ValueError(
            f"Unknown label field: '{label}'. Available fields: "
            f"{list(label_fields.keys()) + list(interaction_fields.keys()) + ['sensor_type']}"
        )

    def to_torch(
        self,
        label: str | None = None,
        transform=None,
        target_transform=None,
        dtype="float32",
        batch_size: int | None = None,
        shuffle: bool = False,
        **kwargs,
    ) -> "torch.utils.data.Dataset | torch.utils.data.DataLoader":
        """Convert HaptData to a PyTorch Dataset or DataLoader.

        Returns a TensorDataset where each sample is one element along
        the first dimension of the raw data (e.g., one frame of an imaging
        sequence or one time-step of a dynamic signal).

        Parameters
        ----------
        label : str, optional
            Field name to use as target labels. Available fields:
            material, material_category, object_name, object_category,
            task, speed_mm_s, normal_force_N, approach_angle_deg,
            temperature_C, humidity_pct, interaction_type, sensor_type.
            String values are encoded as integer class indices.
        transform : callable, optional
            Function applied to each sample tensor (X).
        target_transform : callable, optional
            Function applied to each label tensor (y). Only used when
            ``label`` is specified.
        dtype : str or torch.dtype, default 'float32'
            Target dtype for the data tensor.
        batch_size : int, optional
            If set, wraps the dataset in a DataLoader with this batch size.
        shuffle : bool, default False
            Whether to shuffle the DataLoader. Only meaningful with
            ``batch_size``.
        **kwargs
            Additional keyword arguments forwarded to DataLoader
            (e.g., num_workers, pin_memory, drop_last).

        Returns
        -------
        TensorDataset or DataLoader
        """
        import torch

        # Normalize dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        # Convert raw data to tensor
        X = self._to_tensor(self._raw.array, dtype)

        # Build dataset tensors
        tensors = [X]

        if label is not None:
            value = self._resolve_label_field(label)
            encoded = self._encode_label(value)
            y = torch.full((X.shape[0], 1), encoded, dtype=torch.long)
            tensors.append(y)

        # Build base dataset
        ds = torch.utils.data.TensorDataset(*tensors)

        # Wrap with transform support if needed
        if transform is not None or target_transform is not None:
            ds = _TransformDataset(ds, transform=transform, target_transform=target_transform)

        # Optionally wrap in DataLoader
        if batch_size is not None:
            return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kwargs)

        return ds

    def __repr__(self) -> str:
        return (
            f"HaptData(sensor={self.sensor.type}, modality={self.modality}, "
            f"shape={self.raw.shape}, labels={self.labels.material})"
        )
