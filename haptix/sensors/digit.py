"""
DIGIT sensor adapter.

DIGIT (Meta) is a vision-based tactile sensor that produces
RGB images of an elastomer surface deforming under contact.

Native format: directory of PNG/JPEG frames, or .mp4 video.
This adapter handles the common DIGIT data formats.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from haptix.core import HaptData, InteractionMeta, Labels, RawData, SensorMeta
from haptix.sensors import register


@register("DIGIT")
@register("DIGIT_v2")
class DigitAdapter:
    """Adapter for DIGIT / DIGIT v2 tactile sensor data."""

    sensor_type = "DIGIT_v2"

    def can_load(self, path: Path) -> bool:
        """Check if path contains DIGIT data."""
        if path.is_dir():
            # Check for image frames
            images = sorted(path.glob("*.png")) + sorted(path.glob("*.jpg"))
            return len(images) > 0
        return path.suffix in (".mp4", ".avi")

    def load(
        self,
        path: Path,
        interaction: InteractionMeta,
        labels: Labels,
        sensor_meta: SensorMeta | None = None,
    ) -> HaptData:
        """Load DIGIT data and return HaptData.

        Args:
            path: Directory of image frames or path to .mp4 video
            interaction: Interaction metadata (REQUIRED)
            labels: Annotation labels
            sensor_meta: Optional sensor metadata override
        """
        frames = self._load_frames(path)

        if sensor_meta is None:
            sensor_meta = SensorMeta(type="DIGIT_v2")

        raw = RawData(
            array=frames,
            checksum=RawData.compute_checksum(frames),
            dtype=str(frames.dtype),
            shape=frames.shape,
        )

        return HaptData(
            raw=raw,
            sensor=sensor_meta,
            modality="imaging",
            sampling_rate_hz=self._detect_framerate(path),
            interaction=interaction,
            labels=labels,
            version="0.1.0",
        )

    def _load_frames(self, path: Path) -> np.ndarray:
        """Load frames from directory or video."""
        if path.is_dir():
            images = sorted(path.glob("*.png")) + sorted(path.glob("*.jpg"))
            if not images:
                raise FileNotFoundError(f"No image frames found in {path}")

            # Load first frame to get dimensions
            first = np.array(Image.open(images[0]))
            h, w = first.shape[:2]
            c = 1 if len(first.shape) == 2 else first.shape[2]

            frames = np.zeros((len(images), h, w, c), dtype=np.uint8)
            for i, img_path in enumerate(images):
                img = np.array(Image.open(img_path))
                if len(img.shape) == 2:
                    img = img[..., np.newaxis]
                frames[i] = img
            return frames

        # Video loading requires optional dependency
        if path.suffix in (".mp4", ".avi"):
            try:
                import cv2  # noqa: F401
            except ImportError:
                raise ImportError(
                    "Video loading requires opencv-python. " "Install with: pip install haptix[all]"
                )
            raise NotImplementedError("Video loading coming in v0.2.0")

        raise ValueError(f"Unsupported DIGIT format: {path}")

    def _detect_framerate(self, path: Path) -> float:
        """Detect or default to 60Hz for DIGIT."""
        # In practice, most DIGIT data is 60Hz.
        # TODO: extract from video metadata, or require user to specify
        return 60.0
