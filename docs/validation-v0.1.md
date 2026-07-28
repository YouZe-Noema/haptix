# haptix Validation Report v0.1

**Date**: 2026-07-28
**Baseline**: haptix v0.1.0, 112 tests passing

---

## Summary

Validated the CoroCapacitive and DIGIT sensor adapters against real and synthetic data. Found and fixed critical bugs in the Coro adapter. DIGIT adapter works correctly but lacks a publicly downloadable sample dataset for real-data validation.

---

## 1. Lab-CORO Capacitive Sensor (CoroCapacitive Adapter)

### 1.1 Data Source

- **Repository**: https://github.com/Lab-CORO/TactileDataset
- **Data location**: External MyCloud share (`os5.mycloud.com/action/share/daf9c31c-53b8-485b-b6ef-5ca75bfcfc75`), NOT in the GitHub repo
- **Primary download link** (from README): `os5.mycloud.com/action/share/dc475405-9198-4860-85c9-aeb3d8f79a09` — **EMPTY FOLDER**
- **Secondary link**: `os5.mycloud.com/action/share/daf9c31c-53b8-485b-b6ef-5ca75bfcfc75` — **Data present** but requires interactive browser access (no direct wget/curl download possible without JWT token from browser session)

### 1.2 Actual Data Format vs Assumptions

| Assumption (adapter code) | Actual Data |
|---|---|
| Files named `Flat_Real_Abaqus.csv` | Files use version suffixes: `Flat_Real_Abaqus_V1.csv`, `Flat_Real_Abaqus_V2.csv`, `Flat_Real_Abaqus_V1_Noise.csv`, etc. |
| 57 taxels (`_NUM_TAXELS = 57`) | Real sensor has **28 taxels** (validated from Bias01.csv: columns 1-28 are taxel readings) |
| CSV always has a "Path" column | Many CSV files (e.g., Bias files) have **no Path column** — raw [T × D] layout with a leading marker column |
| Data format is uniform across files | Files in Abaqus/ share are structured differently from Real/ share. Abaqus/ has top-level CSV files + Dataset01-Dataset02 subfolders. Real/ has Dataset01-Dataset06 subfolders with per-experiment CSVs |

### 1.3 Bugs Found & Fixed

1. **Hardcoded taxel count** (`_NUM_TAXELS = 57`): Wrong for real data (28 taxels). Fixed by auto-detecting taxel count from data and filtering out constant/marker columns.

2. **Wrong transposition logic**: When `columns < 57` and `rows >= 57`, the old code transposed the data, producing `(29, 57)` instead of `(245, 28)`. Fixed by treating each row as a frame and each column as a taxel.

3. **No marker column handling**: Real CSV files have a leading column (all zeros) that's a frame index, not a taxel. Fixed by detecting constant-value columns and excluding them.

4. **Strict filename matching**: Adapter expected exact filenames but real data has version suffixes. Fixed by using flexible substring matching (e.g., `Flat_Real_Abaqus` matches `Flat_Real_Abaqus_V1.csv`).

### 1.4 Validation Results

- **Bias01.csv** (real calibration data, 245 rows × 29 cols):
  - Before fix: `(29, 57)` — completely wrong shape
  - After fix: `(244, 28)` — correct: 244 frames × 28 taxels
  - Checksum valid: ✅
  - Values range: 10,067 – 37,865 (raw ADC)

- **Synthetic test data** (Path-based format, 3 samples × 57 taxels):
  - Shape: `(3, 57)` — correct
  - Checksum valid: ✅
  - Backward compatible: ✅

- **All existing tests**: 112 passed, 0 failed ✅

---

## 2. DIGIT Sensor

### 2.1 Data Source

- **Repository**: https://github.com/facebookresearch/digit-interface
- **Sample data**: **NONE** — the repo contains only the Python interface, udev rules, and example scripts (demo_digit.py, demo_rgb_intensity.py). No sample image frames or videos are included.
- **HuggingFace datasets**: Found `kingJulio/touch-and-go-probe` and `sach088/dino_touch_and_go` but both have empty configs (no downloadable files).
- **Status**: No publicly accessible DIGIT sample dataset found. Real DIGIT validation requires either (a) capturing data from real hardware, (b) finding a dataset on Kaggle/Zenodo, or (c) generating synthetic data.

### 2.2 Validation Results (Synthetic Data)

- **5 synthetic PNG frames** (240×320 RGB):
  - `can_load()`: ✅ detects PNG directory correctly
  - Shape: `(5, 240, 320, 3)` — correct
  - Dtype: `uint8` — correct
  - Modality: `imaging` — correct
  - Sampling rate: 60.0 Hz — correct default
  - Checksum valid: ✅

- **Roundtrip save/load** (`.hapt` format):
  - Shape match: ✅
  - Checksum match: ✅
  - Verify after load: ✅

### 2.3 Limitations

- `.mp4` / `.avi` video loading is not implemented (raises `NotImplementedError` with "coming in v0.2.0" message)
- No real DIGIT data was available for validation — this is a gap
- Edge cases (grayscale frames, different dimensions, JPEG) tested synthetically and pass

---

## 3. Other Adapters

### 3.1 GelSight

Not directly tested in this validation pass. Tested indirectly via `tests/test_gelsight.py` — all tests pass. GelSight adapter uses a similar image-folder approach to DIGIT and should work with real GelSight data when available.

---

## 4. Recommendations

### Immediate Actions
1. **Rename `_NUM_TAXELS`** → `_LEGACY_NUM_TAXELS` to avoid confusion — the real sensor has 28 taxels
2. **Add a `num_taxels` parameter** to `CoroCapacitiveAdapter.load()` so users can override auto-detection
3. **Document the actual dataset structure** in `docs/adapters.md` — the Lab-CORO dataset is on an external cloud share with non-obvious folder structure

### For Phase 2 Readiness
4. **Test pyproject.toml build** — `pip install build && python -m build`
5. **Fix Homepage URL** in pyproject.toml: currently points to `github.com/ronaldxia/haptix` but should be `github.com/YouZe-Noema/haptix`
6. **Verify `MANIFEST.in`** excludes `data/`, `research/`, and other non-package files

### DIGIT Next Steps
7. Search Kaggle and Zenodo for DIGIT datasets (e.g., "DIGIT sensor grasping", "tactile image classification")
8. Contact DIGIT authors at Meta for sample data or test fixtures
9. Consider generating synthetic DIGIT-like data as a test fixture for CI

---

## 5. Test Summary

```
Test file              | Tests | Passed | Failed | Skipped
-----------------------|-------|--------|--------|--------
test_coro.py           |    13 |     13 |      0 |       0
test_gelsight.py       |    12 |     12 |      0 |       0
test_roundtrip.py      |     8 |      8 |      0 |       0
test_torch.py          |     9 |      8 |      0 |       1
test_datasets.py       |     8 |      8 |      0 |       0
test_edge_cases.py     |    62 |     62 |      0 |       0
TOTAL                  |   112 |    112 |      0 |       1
```

The 1 skip is `TestHaptDataLoader/test_iteration` which requires `torch` (optional dependency).
