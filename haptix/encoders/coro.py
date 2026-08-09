"""
CoroCapacitive per-sensor encoder (dynamic, untrained v0.1).

Lab-CORO is a capacitive tactile array (~28 taxels) producing a
per-timestep feature vector. The v0.1 encoder is a deterministic
untrained projection: pad/truncate features -> [T, 128]. Trained
weights (v1.0+) replace the projection; the fixed 128-dim output is
the registry contract.
"""

from haptix.encoders import register_encoder
from haptix.encoders.base import _DYNAMIC_DIM, _BaseSensorEncoder


@register_encoder("CoroCapacitive", modality="dynamic")
class CoroCapacitiveEncoder(_BaseSensorEncoder):
    """Encoder for Lab-CORO capacitive arrays (dynamic, [T, 128])."""

    sensor_type = "CoroCapacitive"
    modality = "dynamic"
    embedding_dim = _DYNAMIC_DIM
    version = "encoders/coro/v0.1"
