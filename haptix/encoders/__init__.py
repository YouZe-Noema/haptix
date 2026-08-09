"""
Per-sensor encoder registry (v0.3 roadmap: "pre-trained encoders").

Mirrors the proven :mod:`haptix.sensors` pattern: contributors drop a
module under ``haptix/encoders/`` and decorate their class with
``@register_encoder(...)``. ``get_encoder()`` lazily imports every module
in the package, returns the registered encoder for a sensor type, or a
deterministic surrogate fallback for sensor types without a registered
encoder.

The encoder registry and the adapter registry are intentionally
decoupled: an encoder may exist without a matching
:class:`~haptix.sensors.SensorAdapter` (and vice versa). A contributed
encoder's ``benchmark()`` is the evidence that a working data path
exists.

Public API: :func:`register_encoder`, :func:`get_encoder`,
:func:`list_encoders`, :class:`~haptix.encoders.base.SensorEncoder`.
"""

import importlib
import pkgutil
import sys
from typing import TYPE_CHECKING

from haptix.encoders.base import (
    SensorEncoder,
    SurrogateEncoder,
    _default_dim_for,
    _modality_for,
)

if TYPE_CHECKING:  # pragma: no cover - runtime imports are lazy
    from haptix.encoders.base import _BaseSensorEncoder

__all__ = ["SensorEncoder", "get_encoder", "list_encoders", "register_encoder"]

# Registry of available encoders, keyed by sensor type.
_registry: dict[str, type["_BaseSensorEncoder"]] = {}


def register_encoder(sensor_type: str, modality: str = "imaging"):
    """Decorator to register a per-sensor encoder class.

    Sets ``sensor_type`` and ``modality`` on the class and adds it to the
    registry, after which it is served by :func:`get_encoder` and listed
    by :func:`list_encoders`.

    Parameters
    ----------
    sensor_type : str
        Sensor family name, e.g. ``"GelSight"``, ``"CoroCapacitive"``.
    modality : str, default "imaging"
        ``"imaging"`` | ``"dynamic"`` | ``"force"`` | ``"multimodal"``.

    Examples
    --------
    >>> from haptix.encoders import register_encoder
    >>> @register_encoder("MySensor", modality="dynamic")
    ... class MySensorEncoder:
    ...     embedding_dim = 128
    ...     version = "encoders/my-sensor/v0.1"
    """

    def decorator(cls):
        cls.sensor_type = sensor_type
        cls.modality = modality
        _registry[sensor_type] = cls
        return cls

    return decorator


def _lazy_import_encoders() -> None:
    """Import every encoder module under ``haptix/encoders/``.

    Populates the registry so sensor types are discoverable without an
    explicit import — identical to ``haptix.sensors._lazy_import_adapters``.
    """
    import haptix.encoders as encoders_pkg

    for mod_info in pkgutil.iter_modules(encoders_pkg.__path__):
        if mod_info.name != "__init__":
            mod_name = f"haptix.encoders.{mod_info.name}"
            if mod_name not in sys.modules:
                importlib.import_module(mod_name)


def get_encoder(sensor_type: str) -> SensorEncoder:
    """Get (and lazily import) the encoder for a sensor type.

    Returns the best available encoder:

    - the registered encoder for *sensor_type*, if any;
    - otherwise a deterministic surrogate fallback (version tag carries
      ``/surrogate`` so callers can distinguish placeholder embeddings
      from learned ones).

    Encoders may exist without a matching :class:`SensorAdapter` — the
    two registries are intentionally decoupled. A contributed encoder's
    ``benchmark()`` is the evidence that a working data path exists.

    Never raises for unknown sensor types: the surrogate is shape-correct
    for any input.

    Parameters
    ----------
    sensor_type : str
        Sensor family name, e.g. ``"GelSight"``, ``"CoroCapacitive"``.

    Returns
    -------
    SensorEncoder
        A ready-to-encode instance with ``encode(data) -> np.ndarray
        [T, embedding_dim]``.
    """
    _lazy_import_encoders()
    if sensor_type in _registry:
        return _registry[sensor_type]()
    # Surrogate fallback: deterministic, zero learned weights, shape-correct.
    return SurrogateEncoder(
        sensor_type=sensor_type,
        embedding_dim=_default_dim_for(sensor_type),
        modality=_modality_for(sensor_type),
    )


def list_encoders() -> list[str]:
    """List sensor types with a registered encoder.

    Triggers lazy import of encoder modules so the registry is populated
    without an explicit import first.

    Returns
    -------
    list[str]
        Sensor type names, e.g. ``["GelSight", "DIGIT", ...]``.
    """
    _lazy_import_encoders()
    return list(_registry.keys())
