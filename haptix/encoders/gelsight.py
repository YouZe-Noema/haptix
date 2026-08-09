"""
GelSight per-sensor encoder (imaging, untrained v0.1).

GelSight is a vision-based tactile sensor producing images of a
deforming elastomer. The v0.1 encoder is a deterministic untrained
projection: grayscale Lanczos resize to a 16x16 grid -> [T, 256].
Trained weights (v1.0+, per docs/encoder-registry.md) replace the
projection; the fixed 256-dim output is the registry contract.
"""

from haptix.encoders import register_encoder
from haptix.encoders.base import _BaseSensorEncoder, _IMAGING_DIM


@register_encoder("GelSight", modality="imaging")
class GelSightEncoder(_BaseSensorEncoder):
    """Encoder for GelSight tactile images (imaging, [T, 256])."""

    sensor_type = "GelSight"
    modality = "imaging"
    embedding_dim = _IMAGING_DIM
    version = "encoders/gelsight/v0.1"
