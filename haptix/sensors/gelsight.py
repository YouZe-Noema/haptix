"""
GelSight sensor adapter.

GelSight (MIT / Harvard / UC Berkeley) is a vision-based tactile sensor
that produces grayscale or RGB images of a reflective elastomer surface
deforming under contact.

Native format: directory of PNG, JPG, or TIFF image frames.
The sensor is typically webcam-based, producing 640×480 grayscale images
at 30 Hz, though newer variants (GelSight Mini, GelSight Wedge) may
produce RGB at different resolutions.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from haptix.core import HaptData, InteractionMeta, Labels, RawData, SensorMeta
from haptix.sensors import register

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@register("GelSight")
@register("GelSight_Mini")
@register("GelSight_Wedge")
class GelSightAdapter:
    """Adapter for GelSight tactile sensor data.

    Handles both grayscale and RGB image sequences stored as individual
    frame files in a directory. Uses 30 Hz as the default framerate
    (standard for webcam-based GelSight sensors).
    """

    sensor_type = "GelSight"

    def can_load(self, path: Path) -> bool:
        """Check if path contains GelSight data (directory of images)."""
        if not path.is_dir():
            return False
        images = self._find_images(path)
        return len(images) > 0

    def load(
        self,
        path: Path,
        interaction: InteractionMeta,
        labels: Labels,
        sensor_meta: SensorMeta | None = None,
    ) -> HaptData:
        """Load GelSight data and return HaptData.

        Args:
            path: Directory of image frames (PNG, JPG, TIFF)
            interaction: Interaction metadata (REQUIRED)
            labels: Annotation labels
            sensor_meta: Optional sensor metadata override
        """
        frames = self._load_frames(path)

        if sensor_meta is None:
            sensor_meta = SensorMeta(type="GelSight")

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
            sampling_rate_hz=self._default_framerate(),
            interaction=interaction,
            labels=labels,
            version="0.1.0",
        )

    def _find_images(self, path: Path) -> list[Path]:
        """Return sorted list of supported image files in directory."""
        images = []
        for ext in _IMAGE_EXTENSIONS:
            images.extend(path.glob(f"*{ext}"))
            images.extend(path.glob(f"*{ext.upper()}"))
        images.sort()
        return images

    def _load_frames(self, path: Path) -> np.ndarray:
        """Load all image frames from a directory into a numpy array.

        Handles both grayscale (2D) and RGB (3D) images.
        Grayscale images are expanded to [H, W, 1] for consistency
        with the imaging modality convention.
        """
        images = self._find_images(path)
        if not images:
            raise FileNotFoundError(f"No image frames found in {path}")

        # Load first frame to determine dimensions
        first = np.array(Image.open(images[0]))
        if len(first.shape) == 2:
            h, w = first.shape
            c = 1
        else:
            h, w = first.shape[:2]
            c = first.shape[2]

        frames = np.zeros((len(images), h, w, c), dtype=np.uint8)
        for i, img_path in enumerate(images):
            img = np.array(Image.open(img_path))
            if len(img.shape) == 2:
                img = img[..., np.newaxis]
            frames[i] = img

        return frames

    @staticmethod
    def _default_framerate() -> float:
        """GelSight sensors typically run at 30 Hz (webcam baseline).

        Many academic datasets record at 30 fps. Some newer variants
        can go higher, but 30 Hz is the safe default.
        """
        return 30.0
