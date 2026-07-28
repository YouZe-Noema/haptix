"""
Sensor adapters for converting native sensor formats to .hapt.

Each adapter handles one sensor type's native file format.
New sensors can be added by implementing the SensorAdapter protocol.
"""

import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

from haptix.core import HaptData, InteractionMeta, Labels


@runtime_checkable
class SensorAdapter(Protocol):
    """Protocol for sensor format adapters."""

    sensor_type: str

    def can_load(self, path: Path) -> bool:
        """Check if this adapter can handle the given file."""
        ...

    def load(self, path: Path, interaction: InteractionMeta, labels: Labels) -> HaptData:
        """Load native sensor file and return HaptData."""
        ...


# Registry of available adapters
_registry: dict[str, type[SensorAdapter]] = {}


def register(sensor_type: str):
    """Decorator to register a sensor adapter."""

    def decorator(cls):
        _registry[sensor_type] = cls
        return cls

    return decorator


def get_sensor(sensor_type: str) -> SensorAdapter:
    """Get a sensor adapter by type.

    Triggers lazy import of adapter modules so sensor types are discoverable
    without an explicit list_sensors() call first.
    """
    # Lazy import adapters to populate registry
    _lazy_import_adapters()

    if sensor_type not in _registry:
        raise ValueError(
            f"Unknown sensor type: {sensor_type}. " f"Available: {list(_registry.keys())}"
        )
    return _registry[sensor_type]()


def _lazy_import_adapters():
    """Import all sensor adapter modules to populate the registry."""
    import importlib
    import pkgutil

    import haptix.sensors as sensors_pkg

    for _mod_info in pkgutil.iter_modules(sensors_pkg.__path__):
        if _mod_info.name != "__init__":
            _mod_name = f"haptix.sensors.{_mod_info.name}"
            if _mod_name not in sys.modules:
                importlib.import_module(_mod_name)


def list_sensors() -> list[str]:
    """List all registered sensor types."""
    _lazy_import_adapters()
    return list(_registry.keys())
