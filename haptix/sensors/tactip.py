"""
TacTip biomimetic optical tactile sensor adapter.

TacTip (Bristol Robotics Lab / University of Bristol) is a vision-based
tactile sensor that uses a camera to track the displacement of internal
pins/markers on a soft silicone membrane. When the membrane contacts a
surface, the pins deform, and the camera captures their displacement.

Native format: directory of image frames (PNG/JPG) showing the internal
pin array, or CSV files with pre-extracted (x, y) pin positions.

The adapter supports both modes:
  - Image mode: loads frames as imaging modality [T, H, W, C]
  - Marker mode: loads CSV pin positions as dynamic modality [T, D]

References
----------
- Lepora, N. F., et al. "Tactile Superresolution and Biomimetic
  Hyperacuity." IEEE Transactions on Robotics, 2015.
- https://github.com/bristolroboticslab/tactip_toolkit
"""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from haptix.core import HaptData, InteractionMeta, Labels, RawData, SensorMeta
from haptix.sensors import register

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_DEFAULT_FRAMERATE_HZ = 30.0


@register("TacTip")
class TacTipAdapter:
    """Adapter for TacTip tactile sensor data.

    Supports both raw image frames and pre-extracted pin position CSV data.
    """

    sensor_type = "TacTip"

    def can_load(self, path: Path) -> bool:
        """Check if path contains TacTip data.

        Accepts directories with image frames or CSV files with pin positions.
        """
        if path.is_dir():
            images = self._find_images(path)
            return len(images) > 0
        if path.suffix.lower() == ".csv":
            try:
                with open(path, "r") as f:
                    first_line = f.readline().strip()
                cols = [c.strip().lower() for c in first_line.split(",")]
                # TacTip CSV typically has pin_0_x, pin_0_y, pin_1_x, pin_1_y...
                pin_cols = sum(1 for c in cols if "pin" in c)
                return pin_cols >= 10  # TacTip has 100+ pins typically
            except (OSError, UnicodeDecodeError, AttributeError):
                return False
        return False

    def load(
        self,
        path: Path,
        interaction: InteractionMeta,
        labels: Labels,
        sensor_meta: SensorMeta | None = None,
        sampling_rate_hz: float = _DEFAULT_FRAMERATE_HZ,
        mode: str = "auto",
    ) -> HaptData:
        """Load TacTip data and return HaptData.

        Parameters
        ----------
        path : Path
            Directory of images or CSV file of pin positions.
        interaction : InteractionMeta
            Interaction metadata.
        labels : Labels
            Annotation labels.
        sensor_meta : SensorMeta, optional
            Override sensor metadata. Defaults to TacTip.
        sampling_rate_hz : float, optional
            Sampling rate (default 30 Hz).
        mode : str
            "auto" (default) — detects format automatically.
            "image" — force image mode.
            "markers" — force marker CSV mode.
        """
        # --- Detect mode ---
        if mode == "auto":
            if path.is_file() and path.suffix.lower() == ".csv":
                mode = "markers"
            else:
                mode = "image"

        if mode == "markers":
            array = self._load_markers_csv(path)
            data_modality = "dynamic"
        else:
            array = self._load_images(path)
            data_modality = "imaging"

        if sensor_meta is None:
            n_pins = None
            if mode == "markers":
                n_pins = array.shape[1] // 2  # 2 coordinates per pin
            sensor_meta = SensorMeta(
                type="TacTip",
                calibration_params={
                    "n_pins": n_pins,
                    "mode": mode,
                },
            )

        raw = RawData(
            array=array,
            checksum=RawData.compute_checksum(array),
            dtype=str(array.dtype),
            shape=array.shape,
        )

        return HaptData(
            raw=raw,
            sensor=sensor_meta,
            modality=data_modality,
            sampling_rate_hz=sampling_rate_hz,
            interaction=interaction,
            labels=labels,
            version="0.1.0",
        )

    # ── Helper methods ───────────────────────────────────────────────────

    @staticmethod
    def _find_images(path: Path) -> list[Path]:
        """Find all image files in a directory, sorted by name."""
        images = []
        for ext in _IMAGE_EXTENSIONS:
            images.extend(path.glob(f"*{ext}"))
            images.extend(path.glob(f"*{ext.upper()}"))
        return sorted(images, key=lambda p: p.name)

    def _load_images(self, path: Path) -> np.ndarray:
        """Load image frames as [T, H, W, C] array."""
        if not path.is_dir():
            raise FileNotFoundError(f"Not a directory: {path}")

        images = self._find_images(path)
        if not images:
            raise FileNotFoundError(f"No image files found in {path}")

        frames = []
        for img_path in images:
            img = np.array(Image.open(img_path))
            if img.ndim == 2:
                img = img[:, :, np.newaxis]  # (H, W) → (H, W, 1)
            frames.append(img)

        # Handle mixed resolutions by using the first frame's shape
        ref_shape = frames[0].shape
        aligned = []
        for f in frames:
            if f.shape[:2] != ref_shape[:2]:
                # Resize via PIL
                pil_img = Image.fromarray(f.squeeze() if f.ndim == 3 and f.shape[2] == 1 else f)
                pil_img = pil_img.resize((ref_shape[1], ref_shape[0]))
                f = np.array(pil_img)
                if f.ndim == 2:
                    f = f[:, :, np.newaxis]
                # Channel mismatch
                if f.shape[2] != ref_shape[2]:
                    if ref_shape[2] == 1 and f.shape[2] == 3:
                        f = np.mean(f, axis=2, keepdims=True).astype(f.dtype)
                    elif ref_shape[2] == 3 and f.shape[2] == 1:
                        f = np.repeat(f, 3, axis=2)
            aligned.append(f)

        return np.stack(aligned, axis=0)

    def _load_markers_csv(self, path: Path) -> np.ndarray:
        """Load TacTip pin positions from CSV as [T, 2*N_pins] array.

        CSV format: each row is one frame, columns are alternating
        pin_0_x, pin_0_y, pin_1_x, pin_1_y, ... (or x_0, y_0, x_1, y_1, ...)
        """
        with open(path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            raise ValueError(f"Empty CSV: {path}")

        # Check if first row is a header
        first_row = [c.strip().lower() for c in rows[0]]
        pin_cols = sum(1 for c in first_row if "pin" in c or c.startswith(("x_", "y_")))
        has_header = pin_cols >= 10

        data_rows = rows[1:] if has_header else rows

        parsed = []
        for row in data_rows:
            try:
                parsed.append([float(v) for v in row])
            except (ValueError, IndexError):
                continue

        if not parsed:
            raise ValueError(f"No valid numeric data in {path}")

        return np.array(parsed, dtype=np.float32)
