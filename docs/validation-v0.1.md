# haptix Real Data Validation Report — v0.2

> Date: 2026-07-29
> Status: Coro adapter validated with real data. DIGIT/GelSight pending.

## Summary

CoroCapacitive adapter validated against real Lab-CORO sensor data (Dataset01).
Two bugs discovered and fixed. DIGIT and GelSight adapters still pending real
data.

## CoroCapacitive Validation — ✅ PASSED (real data)

**Data source:** Lab-CORO TactileDataset, "Simulations > Real > Dataset01"
(MyCloud shared folder, not the broken "DatasetFiles" link from README).

**Files tested:**
| File | Rows | Columns | Size | Round-trip |
|------|------|---------|------|------------|
| `Dataset_01_Real.csv` | 786 | 29 (force + tax1..tax28) | 372 KB | ✓ |
| `Dataset_01_Real_V2.csv` | 300 | 29 (force + tax1..tax28) | 163 KB | ✓ |

**Format:** `force,tax1,tax2,...,tax28,Path` — 28 taxels (not 57 as assumed),
Path column is LAST (not first). Each row is a complete sensor frame.

## Bugs Found and Fixed

### Bug 1: Wrong data dimensionality for frame-per-row format

**Symptom:** Real CSV with 786 rows, 10 Path groups, 29 columns produced shape
(10, 79) instead of (786, 29).

**Root cause:** When a Path group had multiple rows AND multiple columns, the
adapter took `mean(axis=1)` across columns, collapsing each row to a scalar.
This was correct for the old format (each row = one taxel + metadata), but
wrong for the real format (each row = all 29 taxel values).

**Fix:** Added column-count heuristic: if ≥10 columns → treat each row as a
full frame. If 2-9 columns → treat as taxel-per-row with metadata (old behavior).

### Bug 2: `np.pad` applied to wrong axis on 2D arrays

**Symptom:** `np.stack` failed with "all input arrays must have the same shape"
when frames had different row counts but same column count.

**Root cause:** `np.pad(arr, (0, n))` on a 2D array pads axis 1 (columns), not
axis 0 (rows). The padding should be `((0, n), (0, 0))` for row-only padding.

**Fix:** When frames are 2D (frame-per-row format), use `np.concatenate` instead
of pad+stack. All frames share the same column count by construction.

## Adapter Heuristic Summary

The Coro adapter now handles three CSV formats:

| Format | Columns | Each Row | Detection | Output |
|--------|---------|----------|-----------|--------|
| Taxel-per-row | 1-3 (Pressure + X + Y) | One taxel reading | `Path` exists, cols < 10 | Stack [groups, taxels] |
| Frame-per-row | 10+ (force + tax1..taxN) | Full sensor frame | `Path` exists, cols ≥ 10 | Concat [total_rows, cols] |
| Flat | Any (no Path) | Full sensor frame | No `Path` column | Passthrough [rows, cols] |

## Remaining Data Files (Pending)

The MyCloud shared folder contains 6 datasets (Dataset01-Dataset06) plus
Abaqus simulation data and Bias calibration files. Dataset01 validated.
Need user to bulk-download remaining files from:
`Simulations > Real > Dataset02..Dataset06`
`Simulations > Abaqus > ...`
`Simulations > Issac > ...`

## DIGIT / GelSight — ⚠️ Still pending

No public sample dataset found. Adapters work with any image directory.

## Next Steps

1. **Download remaining Lab-CORO CSVs** (Dataset02-Dataset06, Abaqus, Issac)
   — needs user to bulk-download from MyCloud (one-click-per-file UI)
2. **Run full Coro validation** against all downloaded files
3. **Update pyproject.toml** version for Phase 2 (PyPI prep)
4. **Find or create DIGIT sample data** for Phase 1 completion
