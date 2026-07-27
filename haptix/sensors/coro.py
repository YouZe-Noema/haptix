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

from haptix.core import HaptData, RawData, SensorMeta, InteractionMeta, Labels
from haptix.sensors import register

# Expected number of taxels in the capacitive sensor array
_NUM_TAXELS = 57

# CSV file patterns expected in a Lab-CORO dataset directory
_CSV_PATTERNS = [
    "Flat_Real_Abaqus.csv",
    "Flat_Simulation_Abaqus.csv",
    "Curved_Real_Abaqus.csv",
    "Curved_Simulation_Abaqus.csv",
]

# Default framerate (30 Hz is standard for capacitive sensor arrays)
_DEFAULT_FRAMERATE_HZ = 30.0

# Source types: which CSV to load
_SOURCE_MAP = {
    "flat_real": "Flat_Real_Abaqus.csv",
    "flat_simulation": "Flat_Simulation_Abaqus.csv",
    "curved_real": "Curved_Real_Abaqus.csv",
    "curved_simulation": "Curved_Simulation_Abaqus.csv",
    # Aliases for convenience
    "real": "Flat_Real_Abaqus.csv",
    "simulation": "Flat_Simulation_Abaqus.csv",
    "default": "Flat_Real_Abaqus.csv",
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

        csv_filename = _SOURCE_MAP.get(source, source)
        csv_path = path / csv_filename

        if not csv_path.exists():
            # Fall back: find any CSV if the named one doesn't exist
            csvs = sorted(path.glob("*.csv"))
            if not csvs:
                raise FileNotFoundError(
                    f"No CSV data files found in {path}. "
                    f"Looking for: {csv_filename}"
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
        2. For each group, extract the taxel data:
           a. Drop the 'Path' column
           b. If the group has exactly _NUM_TAXELS rows, reshape
              columns to a 1D vector (or take the column mean).
           c. If the group has one row with _NUM_TAXELS+1 columns,
              treat the row directly as the taxel vector.
        3. If no 'Path' column exists, assume each row is one sample
           and the columns are taxel values. Expect _NUM_TAXELS data
           columns.
        """
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

                if group_data.shape[0] == _NUM_TAXELS:
                    # Standard format: 57 rows, one per taxel.
                    # Use column mean (or first column if only one) per row
                    # to get a 57-element vector
                    if group_data.shape[1] == 1:
                        # Single data column: each row is a taxel reading
                        frame = group_data.values.flatten().astype(np.float32)
                    else:
                        # Multiple columns: take the mean of all numeric
                        # columns per row
                        frame = group_data.mean(axis=1).values.astype(np.float32)
                elif group_data.shape[0] == 1 and group_data.shape[1] >= _NUM_TAXELS:
                    # Single-row format: all taxels in columns
                    frame = group_data.values[0, :_NUM_TAXELS].astype(np.float32)
                else:
                    # Fallback: use all values, padded/truncated to _NUM_TAXELS
                    flat = group_data.values.flatten()
                    if len(flat) >= _NUM_TAXELS:
                        frame = flat[:_NUM_TAXELS].astype(np.float32)
                    else:
                        frame = np.pad(
                            flat.astype(np.float32),
                            (0, _NUM_TAXELS - len(flat)),
                            mode="constant",
                        )

                frames.append(frame)

            if not frames:
                # Fallback: use the entire dataframe as one frame
                flat = df[data_cols].values.flatten()
                return np.zeros((1, _NUM_TAXELS), dtype=np.float32)

            return np.stack(frames, axis=0)

        else:
            # No Path column: each row is one sample, columns are taxels
            numeric_df = df[data_cols].apply(pd.to_numeric, errors="coerce")

            if numeric_df.shape[1] >= _NUM_TAXELS:
                return numeric_df.values[:, :_NUM_TAXELS].astype(np.float32)
            elif numeric_df.shape[0] >= _NUM_TAXELS:
                # Treat rows as taxels, take first _NUM_TAXELS rows
                return numeric_df.values[:_NUM_TAXELS, :].astype(np.float32).T
            else:
                # Fallback: flatten all values
                flat = numeric_df.values.flatten().astype(np.float32)
                if len(flat) >= _NUM_TAXELS:
                    return flat[:_NUM_TAXELS].reshape(1, -1)
                return np.pad(flat, (0, _NUM_TAXELS - len(flat))).reshape(1, -1)
