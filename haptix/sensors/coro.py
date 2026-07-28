"""
Lab-CORO capacitive tactile sensor adapter.

The Lab-CORO TactileDataset (https://github.com/Lab-CORO/TactileDataset)
provides real + simulated capacitive tactile pressure data for robotic grasping.

Native format: CSV files (Flat_Real_Abaqus.csv, Flat_Simulation_Abaqus.csv,
Curved_Real_Abaqus.csv, Curved_Simulation_Abaqus.csv) containing taxel
pressure readings grouped by a 'Path' column.

Each group (same Path value) represents a complete sensor frame from the
capacitive tactile sensor's 57-taxel array (7×8 grid with 1 padded/calibration
element). The adapter extracts pressure values and produces a [T, 57] dynamic
time-series array suitable for ML workloads.

References
----------
- De la Cruz-Sánchez, B. A., Kwiatkowski, J., & Roberge, J-P.
  "Tactile Contact Patterns for Robotic Grasping: A Dataset of Real and
  Simulated Data" (2024).
- https://github.com/Lab-CORO/TactileDataset
"""

from pathlib import Path

import numpy as np
import pandas as pd

from haptix.core import HaptData, InteractionMeta, Labels, RawData, SensorMeta
from haptix.sensors import register

# Expected number of taxels in the capacitive sensor array.
# Auto-detected from data; 57 is the default for legacy format compatibility.
_NUM_TAXELS = 57

# CSV file patterns expected in a Lab-CORO dataset directory.
# Actual filenames use version suffixes (e.g., Flat_Real_Abaqus_V1.csv).
# These patterns match the base names; the adapter finds CSV files flexibly.
_CSV_PATTERNS = [
    "Flat_Real_Abaqus.csv",
    "Flat_Simulation_Abaqus.csv",
    "Curved_Real_Abaqus.csv",
    "Curved_Simulation_Abaqus.csv",
]

# Default framerate (30 Hz is standard for capacitive sensor arrays)
_DEFAULT_FRAMERATE_HZ = 30.0

# Source types: which CSV file pattern to load.
# The adapter does flexible matching — it checks if the CSV filename
# contains the source string (case-insensitive), so "flat_real"
# matches both "Flat_Real_Abaqus_V1.csv" and "Flat_Real_Abaqus.csv".
_SOURCE_MAP = {
    "flat_real": "Flat_Real_Abaqus",
    "flat_simulation": "Flat_Simulation_Abaqus",
    "curved_real": "Curved_Real_Abaqus",
    "curved_simulation": "Curved_Simulation_Abaqus",
    # Aliases for convenience
    "real": "Flat_Real_Abaqus",
    "simulation": "Flat_Simulation_Abaqus",
    "default": "Flat_Real_Abaqus",
}


@register("CoroCapacitive")
class CoroCapacitiveAdapter:
    """Adapter for Lab-CORO capacitive tactile sensor data.

    Reads CSV pressure arrays and produces [T, D] dynamic modality data
    where D = 57 (the capacitive sensor's taxel count).

    The native dataset format:
    - CSV files with a 'Path' column grouping sensor readings
    - Each Path group contains _NUM_TAXELS rows (one complete sensor frame)
    - Remaining columns contain pressure values per taxel

    Example usage::

        from haptix.sensors import get_sensor
        from haptix.core import InteractionMeta, Labels

        adapter = get_sensor("CoroCapacitive")
        data = adapter.load(
            "/path/to/dataset/dir",
            source="flat_real",
            interaction=InteractionMeta(type="pressing", normal_force_N=3.0),
            labels=Labels(material="aluminium"),
        )
    """

    sensor_type = "CoroCapacitive"

    def can_load(self, path: Path) -> bool:
        """Check if path contains Lab-CORO capacitive sensor data.

        Returns True if the path is a directory containing at least one
        recognized CSV data file.
        """
        if not path.is_dir():
            return False
        csvs = list(path.glob("*.csv"))
        return len(csvs) > 0

    def load(
        self,
        path: Path,
        interaction: InteractionMeta,
        labels: Labels,
        sensor_meta: SensorMeta | None = None,
        source: str = "default",
        sampling_rate_hz: float = _DEFAULT_FRAMERATE_HZ,
    ) -> HaptData:
        """Load Lab-CORO capacitive tactile data and return HaptData.

        Parameters
        ----------
        path : Path
            Directory containing the dataset CSV files.
        interaction : InteractionMeta
            Interaction metadata (REQUIRED per .hapt spec).
        labels : Labels
            Annotation labels (material, object, task, etc.).
        sensor_meta : SensorMeta, optional
            Override sensor metadata. Defaults to CoroCapacitive type.
        source : str, optional
            Which data source to load. One of:
            - "flat_real"       — Flat_Real_Abaqus.csv (real indenters)
            - "flat_simulation" — Flat_Simulation_Abaqus.csv (FEA indenters)
            - "curved_real"     — Curved_Real_Abaqus.csv (real objects)
            - "curved_simulation" — Curved_Simulation_Abaqus.csv (FEA objects)
            - "real"            — alias for "flat_real"
            - "simulation"      — alias for "flat_simulation"
            - "default"         — Flat_Real_Abaqus.csv
        sampling_rate_hz : float, optional
            Override the default 30 Hz sampling rate.
        """
        import pandas as pd

        csv_pattern = _SOURCE_MAP.get(source, source)

        # Flexible filename matching: find CSV containing the pattern
        csvs = sorted(path.glob("*.csv"))
        csv_path = None
        for csv_file in csvs:
            if csv_pattern.lower() in csv_file.name.lower():
                csv_path = csv_file
                break

        if csv_path is None:
            # Fall back: pick the first CSV if no pattern match
            if not csvs:
                raise FileNotFoundError(
                    f"No CSV data files found in {path}. Looking for pattern: {csv_pattern}"
                )
            csv_path = csvs[0]

        # Read CSV and extract pressure array
        array = self._extract_pressure_array(pd.read_csv(csv_path))

        if sensor_meta is None:
            sensor_meta = SensorMeta(type="CoroCapacitive")

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

    def _extract_pressure_array(self, df: "pd.DataFrame") -> np.ndarray:
        """Extract a [T, D] pressure array from a Lab-CORO CSV DataFrame.

        The CSV format has a 'Path' column grouping taxel readings.
        Each Path group contains ``_NUM_TAXELS`` rows representing one
        complete sensor frame.

        Detection logic:
        1. If a 'Path' column exists, group rows by Path value.
        2. For each group, extract the taxel data.
        3. If no 'Path' column exists, treat each row as one frame
           and each data column as a taxel value.
        """

        # Helper: detect columns that are all-zero (likely markers/frame indices)
        def _is_data_column(col_values):
            """Return True if column appears to contain real sensor data."""
            unique = col_values.unique()
            if len(unique) <= 1:
                return False  # Constant column — likely marker
            return True

        # Remove non-data columns
        data_cols = [c for c in df.columns if c != "Path"]

        if "Path" in df.columns:
            # Group by Path, aggregating taxel readings per group
            groups = df.groupby("Path", sort=True)
            frames = []

            for _path_name, group in groups:
                # Drop the Path column for processing
                group_data = group[data_cols]

                # Data columns are already numeric from read_csv
                group_data = group_data.infer_objects(copy=False)

                if group_data.shape[0] == 1 and group_data.shape[1] >= 2:
                    # Single-row format: all taxels in columns
                    frame = group_data.values[0, :].astype(np.float32)
                elif group_data.shape[1] == 1:
                    # Single data column: each row is a taxel reading
                    frame = group_data.values.flatten().astype(np.float32)
                else:
                    # Multiple rows and columns: take mean across columns per row
                    frame = group_data.mean(axis=1).values.astype(np.float32)

                frames.append(frame)

            if not frames:
                return np.zeros((1, _NUM_TAXELS), dtype=np.float32)

            # Ensure all frames have the same length
            target_len = max(len(f) for f in frames)
            aligned = []
            for f in frames:
                if len(f) < target_len:
                    f = np.pad(f, (0, target_len - len(f)), mode="constant")
                elif len(f) > target_len:
                    f = f[:target_len]
                aligned.append(f)
            return np.stack(aligned, axis=0)

        else:
            # No Path column: each row is one frame, columns are taxel values.
            # Convert all columns to numeric, dropping non-convertible ones.
            numeric_data = df[data_cols].apply(pd.to_numeric, errors="coerce")
            numeric_data = numeric_data.dropna(axis=1, how="all")

            # Filter out constant (marker) columns
            useful_cols = [c for c in numeric_data.columns if _is_data_column(numeric_data[c])]
            if not useful_cols:
                # Fallback: use all numeric columns
                useful_cols = list(numeric_data.columns)

            result = numeric_data[useful_cols].values.astype(np.float32)
            return result
