"""
TacTip per-sensor encoder (dynamic, untrained v0.1).

TacTip is a 3D-printed optical tactile sensor. The haptix adapter
supports both marker (dynamic) and image (imaging) modes; the encoder
registry's default entry uses the dynamic marker stream, padded or
truncated to [T, 128]. Trained weights (v1.0+) replace the projection;
the fixed 128-dim output is the registry contract.
"""

from haptix.encoders import register_encoder
from haptix.encoders.base import _BaseSensorEncoder, _DYNAMIC_DIM


@register_encoder("TacTip", modality="dynamic")
class TacTipEncoder(_BaseSensorEncoder):
    """Encoder for TacTip marker streams (dynamic, [T, 128])."""

    sensor_type = "TacTip"
    modality = "dynamic"
    embedding_dim = _DYNAMIC_DIM
    version = "encoders/tactip/v0.1"
