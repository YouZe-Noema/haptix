"""
Unified cross-sensor representation layer for .hapt.

Provides encoders that map tactile data from different sensors
(GelSight, DIGIT, CoroCapacitive, BioTac, TacTip) into a shared
latent embedding space. Inspired by UniForce (arXiv 2602.01153).

Quick Start
-----------
>>> from haptix import load
>>> from haptix.unified import SharedForceEncoder

>>> encoder = SharedForceEncoder(embedding_dim=128)
>>> data = load("example.hapt")
>>> embedding = encoder.encode(data)
>>> embedding.array.shape  # (T, 128)
"""

from haptix.unified.encoder import _ENCODER_VERSION, SharedForceEncoder, UnifiedEncoder

__all__ = [
    "_ENCODER_VERSION",
    "SharedForceEncoder",
    "UnifiedEncoder",
]
