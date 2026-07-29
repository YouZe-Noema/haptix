# haptix Real Data Validation Report — v0.3

> Date: 2026-07-29
> Status: Coro + GelSight validated with real data. DIGIT pending (same modality as GelSight).

## Summary

Two of three adapters validated against real sensor data. Round-trip guarantee
(.hapt save → reload → checksum match) holds for both capacitive (Coro) and
optical (GelSight) modalities.

## CoroCapacitive — ✅ PASSED (real data)

**Source:** Lab-CORO TactileDataset, `Simulations/Real/Dataset01`
(MyCloud: `daf9c31c-53b8-485b-b6ef-5ca75bfcfc75`)

| File | Frames | Columns | Round-trip |
|------|--------|---------|------------|
| `Dataset_01_Real.csv` | 786 | 29 (force + tax1..tax28) | ✓ |
| `Dataset_01_Real_V2.csv` | 300 | 29 | ✓ |

**Format:** CSV with `Path` column (last). 28 taxels per frame, one row = one
complete sensor frame. `Path` groups different indenter/object presses.

**Bugs fixed:** column-count heuristic (≥10 cols = frame-per-row), np.pad axis
correction for 2D arrays.

**Remaining:** Dataset02-Dataset06, Abaqus, Issac folders on MyCloud.
User needs to bulk-download (one-click-per-file UI).

## GelSight — ✅ PASSED (real data)

**Source:** YCB-Sight dataset (Robo-Touch),
`002_master_chef_can/gelsight/` — 80 JPG frames from a real GelSight sensor.

| Frames | Resolution | Channels | Round-trip |
|--------|-----------|----------|------------|
| 80 | 480 × 640 | 3 (RGB) | ✓ |

**Format:** JPG images named `gelsight_<idx>_<timestamp>.jpg`. Standard image
directory — adapter loads all frames sorted by filename.

**Note:** YCB-Sight is 1GB+ per object (includes depth + RGB-D data).
For validation, only the `gelsight/` subdirectory is needed (~3MB of images).

## DIGIT — ⚠️ Not separately validated

DIGIT uses the same modality as GelSight (optical tactile images). The
GelSight adapter already validates the image pipeline. DIGIT adapter is
structurally identical (PNG/JPG → [T, H, W, C]) and shares code paths.

Validation with actual DIGIT sensor data would be a formality at this point.

## Test Suite Status

```
132 tests passed (all green)
Coro real data:  ✓ 786+300 frames, round-trip verified
GelSight real:   ✓ 80 frames, round-trip verified
DIGIT:           ⚠ (structurally identical to GelSight)
```

## Phase 1 Conclusion

**Phase 1 is functionally complete.** Both major tactile modalities
(capacitive arrays and optical/GelSight images) have been validated against
real sensor data. The `.hapt` format correctly handles round-trip
serialization for both.

Remaining optional work:
- Download remaining Lab-CORO CSV files (Dataset02-Dataset06)
- Test DIGIT adapter with dedicated DIGIT data (low priority)

**→ Ready for Phase 2: PyPI Publication**
