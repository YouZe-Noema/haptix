"""
DIGIT per-sensor encoder (imaging, untrained v0.1).

DIGIT (Meta AI) is a compact vision-based tactile sensor producing
grayscale or RGB images of a gel surface. The v0.1 encoder is a
deterministic untrained projection: grayscale Lanczos resize to a
16x16 grid -> [T, 256]. Trained weights (v1.0+) replace the projection;
the fixed 256-dim output is the registry contract.
"""

from haptix.encoders import register_encoder
from haptix.encoders.base import _BaseSensorEncoder, _IMAGING_DIM


@register_encoder("DIGIT", modality="imaging")
class DIGITEncoder(_BaseSensorEncoder):
    """Encoder for DIGIT tactile images (imaging, [T, 256])."""

    sensor_type = "DIGIT"
    modality = "imaging"
    embedding_dim = _IMAGING_DIM
    version = "encoders/digit/v0.1"
