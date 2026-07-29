# haptix Real Data Validation Report — v0.1

> Date: 2026-07-29
> Status: Synthetic validation complete. Real sensor validation blocked by data availability.

## Summary

All three adapters (CoroCapacitive, DIGIT, GelSight) pass round-trip validation with synthetic data matching known real formats. No real sensor data could be validated due to external dataset unavailability.

## Adapter Validation Results

### 1. CoroCapacitive (Lab-CORO) — ✅ Synthetic, ⚠️ Real data unavailable

**Format expected:** CSV with `Path` column grouping 57 taxel readings per frame, or
flat CSV with one row per frame and taxel values as columns.

**Synthetic tests (132 passed):**
- `Flat_Real_Abaqus.csv` format (Path + Pressure + X, Y columns)
- One-row-per-sample format (Path + t0..t56 columns)
- can_load() detection on valid/invalid directories
- Round-trip: save → reload → checksum match
- Auto-detection of taxel count from data columns
- Edge cases: empty dir, non-CSV files, missing columns, single-row vs multi-row

**Real data attempt:**
- Lab-CORO TactileDataset on GitHub: repo contains only GUI code and README
- External dataset download (MyCloud link from README): **folder is empty** — share expired or removed
- Zenodo, Figshare searches: no mirrors found
- Authors: Berith De la Cruz Sánchez (berithcruzs@gmail.com), Jennifer Kwiatkowski, Jean-Philippe Roberge

**Recommendation:** Contact authors for dataset access, or find an alternative capacitive tactile dataset (e.g., from the RoboTouch or Touch-and-Go projects).

### 2. DIGIT — ✅ Synthetic, ⚠️ No public sample dataset found

**Format expected:** Directory of PNG/JPEG frames or .mp4 video.

**Synthetic tests (part of 132 passed):**
- Load from directory of synthetic PNG frames
- can_load() detection (image dirs vs non-image dirs)
- Round-trip save/load with checksum verification
- Sensor metadata override support
- Both `DIGIT` and `DIGIT_v2` registered

**Real data attempt:**
- No public DIGIT sample dataset found with permissive license
- `facebookresearch/digit-interface` provides sensor API but no sample data
- `facebookresearch/TACTO` is a simulator (generates synthetic data)
- HuggingFace `tactile_lightbulb` and `tactile-video-pretrain` exist but are video datasets, not DIGIT-specific
- `OpenGraphLabs-Research/ego-tactile-manipulation` contains DIGIT data but requires HuggingFace login

**Recommendation:** The DIGIT adapter works with any directory of image frames. For real validation, any set of PNG/JPEG contact images can serve as test data — they don't need to be from a physical DIGIT sensor.

### 3. GelSight — ✅ Synthetic, ⚠️ No public sample dataset found

**Format expected:** Directory of image frames (same architecture as DIGIT).

**Synthetic tests (part of 132 passed):**
- Load from directory of PNG frames
- Round-trip save/load with checksum verification
- Sensor metadata including calibration fields

**Real data attempt:**
- GelSight datasets exist in research papers but rarely published with permissive licenses
- The adapter can load any image directory — validation with any tactile-adjacent images would exercise the code path

## Test Suite Summary

```
132 tests passed (all green)
===================
Round-trip       ✓  — checksum verification, re-serialization
Edge cases       ✓  — empty dirs, corrupt files, metadata handling
Coro CSV         ✓  — Path-grouped, flat, mixed formats
DIGIT images     ✓  — PNG frames, sensor registration
GelSight images  ✓  — PNG frames, calibration fields
Torch interop    ✓  — to_torch(), batching
Datasets         ✓  — catalog, download stubs
```

## Blocked Items

| Item | Blocker | Impact |
|------|---------|--------|
| Coro real data validation | Dataset download link dead | Cannot verify taxel count auto-detection on real CSV |
| DIGIT real data validation | No public sample dataset | Cannot verify image preprocessing pipeline |
| GelSight real data validation | No public sample dataset | Same as DIGIT |
| PyPI publication (Phase 2) | No real data confidence | Premature without at least one real-sensor validation |
| End-to-end demo (Phase 3) | Requires real data | Demo needs real sensor → .hapt → PyTorch flow |

## Assumptions Not Yet Verified

1. **Coro CSV column naming** — The adapter uses flexible matching, but real CSVs may have unexpected column names for pressure data.
2. **Coro taxel count** — Assumed 57; auto-detection works from data but untested on real 57-taxel arrays.
3. **DIGIT frame naming** — Assumes standard frame naming; may break on custom naming conventions.
4. **GelSight calibration** — Calibration fields are stored but not validated against real calibration data.
5. **Sampling rate** — Default 30 Hz for Coro; actual rates may differ.
6. **Endianness / dtype** — Adapter assumes float32; real sensors may produce int16 or int8.

## Next Steps (Priority Order)

1. **Contact Lab-CORO authors** for working dataset download link — this is the fastest path to real validation
2. **Download any public DIGIT/GelSight video snippet** — even a YouTube clip of tactile data would exercise the image loading pipeline
3. **If no real data available:** proceed with Phase 2 (PyPI) using synthetic-only validation, noting the limitation in README
4. **Explore HuggingFace datasets** (`tactile_lightbulb`, `ego-tactile-manipulation`) as alternative real data sources
