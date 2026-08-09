"""
BioTac SP per-sensor encoder (dynamic, untrained v0.1).

BioTac SP (SynTouch) is a multimodal fingertip sensor; the haptix
adapter exposes its 19 electrodes + PDC/PAC/TDC/TAC as a dynamic
[ T, 23] stream. The v0.1 encoder is a deterministic untrained
projection: pad/truncate features -> [T, 128]. Trained weights
(v1.0+) replace the projection; the fixed 128-dim output is the
registry contract.
"""

from haptix.encoders import register_encoder
from haptix.encoders.base import _BaseSensorEncoder, _DYNAMIC_DIM


@register_encoder("BioTac_SP", modality="dynamic")
class BioTacSPEncoder(_BaseSensorEncoder):
    """Encoder for BioTac SP fingertip signals (dynamic, [T, 128])."""

    sensor_type = "BioTac_SP"
    modality = "dynamic"
    embedding_dim = _DYNAMIC_DIM
    version = "encoders/biotac/v0.1"
