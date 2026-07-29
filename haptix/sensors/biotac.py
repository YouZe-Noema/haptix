"""
BioTac biomimetic tactile sensor adapter.

The SynTouch BioTac SP is a biomimetic tactile sensor that measures:
  - 19 electrode impedances (spatially distributed across a silicone fingertip)
  - DC pressure (PDC) — static force on the fluid core
  - AC pressure (PAC) — dynamic/vibratory pressure
  - DC temperature (TDC) — thermistor temperature
  - AC temperature (TAC) — transient temperature flux

Total: 23 channels per time step.

Native format: CSV with optional header row containing column names.
Common column patterns:
  - "E1", "E2", ..., "E19", "PDC", "PAC", "TDC", "TAC"
  - "electrode_1", ..., "electrode_19", "pdc", "pac", "tdc", "tac"
  - If no header row, columns are assumed to be:
    timestamp(optional), E1..E19, PDC, PAC, TDC, TAC

The adapter auto-detects column names and handles both with and without
a timestamp column.

References
----------
- Fishel, J. A., & Loeb, G. E. (2012). "Bayesian exploration for
  intelligent identification of textures." Frontiers in neurorobotics.
- https://syntouchinc.com/biotac-sp/
"""

from pathlib import Path

import numpy as np

from haptix.core import HaptData, InteractionMeta, Labels, RawData, SensorMeta
from haptix.sensors import register

# Known BioTac channel names (case-insensitive matching)
_ELECTRODE_PATTERNS = [f"e{i}" for i in range(1, 20)] + [
    f"electrode_{i}" for i in range(1, 20)
]
_PRESSURE_PATTERNS = ["pdc", "pac", "pressure_dc", "pressure_ac"]
_TEMP_PATTERNS = ["tdc", "tac", "temperature_dc", "temperature_ac"]

# Default BioTac SP sampling rate (100 Hz is typical)
_DEFAULT_FRAMERATE_HZ = 100.0

# Expected channel count for a full BioTac SP frame
_EXPECTED_CHANNELS = 23  # 19 electrodes + PDC + PAC + TDC + TAC


@register("BioTac")
@register("BioTac_SP")
class BioTacAdapter:
    """Adapter for SynTouch BioTac tactile sensor data.

    Reads CSV files containing BioTac SP electrode impedance and
    pressure/temperature readings. Auto-detects column layout.
    """

    sensor_type = "BioTac"

    # Reduced to minimum pattern for can_load() — avoid false positives
    # by requiring a recognizable electrode column
    _MIN_ELECTRODE_COLS = 10

    def can_load(self, path: Path) -> bool:
        """Check if path contains BioTac CSV data.

        Returns True if the path is a CSV file with at least 10 columns
        matching known BioTac electrode names.
        """
        if not path.is_file():
            return False
        if path.suffix.lower() not in (".csv", ".txt", ".dat"):
            return False
        try:
            with open(path, "r") as f:
                header = f.readline().strip()
            cols = [c.strip().lower().strip('"') for c in header.split(",")]
            electrode_matches = sum(
                1 for c in cols for pat in _ELECTRODE_PATTERNS if c in pat
            )
            return electrode_matches >= self._MIN_ELECTRODE_COLS
        except Exception:
            return False

    def load(
        self,
        path: Path,
        interaction: InteractionMeta,
        labels: Labels,
        sensor_meta: SensorMeta | None = None,
        sampling_rate_hz: float = _DEFAULT_FRAMERATE_HZ,
    ) -> HaptData:
        """Load BioTac CSV data and return HaptData.

        Parameters
        ----------
        path : Path
            Path to the BioTac CSV file.
        interaction : InteractionMeta
            Interaction metadata (REQUIRED per .hapt spec).
        labels : Labels
            Annotation labels.
        sensor_meta : SensorMeta, optional
            Override sensor metadata. Defaults to BioTac_SP.
        sampling_rate_hz : float, optional
            Sampling rate in Hz (default 100 Hz).
        """
        import csv

        # --- Read the CSV file ---
        with open(path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            raise ValueError(f"Empty CSV file: {path}")

        # --- Detect header vs data ---
        first_row = [c.strip().lower().strip('"') for c in rows[0]]

        # Check if first row looks like a header (has electrode names)
        electrode_count = sum(
            1 for c in first_row for pat in _ELECTRODE_PATTERNS if c in pat
        )
        has_header = electrode_count >= self._MIN_ELECTRODE_COLS

        header: list[str] = []
        if has_header:
            header = first_row
            data_rows = rows[1:]
        else:
            # No header — assume: E1..E19, PDC, PAC, TDC, TAC
            # Check if first column looks like a timestamp (numeric with decimal)
            try:
                float(rows[0][0])
                has_timestamp = "." in rows[0][0] or len(rows) > 1
            except (ValueError, IndexError):
                has_timestamp = False
            data_rows = rows

        # --- Parse numeric data ---
        parsed = []
        for row in data_rows:
            try:
                parsed.append([float(v) for v in row])
            except (ValueError, IndexError):
                continue

        if not parsed:
            raise ValueError(f"No valid numeric rows in {path}")

        array = np.array(parsed, dtype=np.float32)

        # --- Deduplicate timestamp column if present ---
        if has_header:
            # Check if first column is a timestamp based on column name
            ts_patterns = ("time", "timestamp", "t")
            if header[0].lower().startswith(ts_patterns):
                array = array[:, 1:]  # Drop timestamp column
                header = header[1:]
        elif array.shape[1] > _EXPECTED_CHANNELS:
            # No header, but extra column → assume first is timestamp
            array = array[:, 1:]

        # --- Validate channel count ---
        n_cols = array.shape[1]
        if n_cols < 10:
            raise ValueError(
                f"Expected at least 10 sensor channels, got {n_cols}. "
                f"File: {path}"
            )

        # Build sensor metadata
        if sensor_meta is None:
            sensor_meta = SensorMeta(
                type="BioTac_SP",
                calibration_params={
                    "n_electrodes": min(n_cols, 19),
                    "n_channels": n_cols,
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
            modality="dynamic",
            sampling_rate_hz=sampling_rate_hz,
            interaction=interaction,
            labels=labels,
            version="0.1.0",
        )
