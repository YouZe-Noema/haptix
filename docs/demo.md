# haptix End-to-End Demo

> **Status:** ✅ Working (tested 2026-08-07)
> **Runtime:** ~12 seconds on CPU
> **Script:** `examples/end_to_end_demo.py`
> **Package:** haptix 0.2.0 on PyPI — `pip install haptix[torch] torch`

## What It Shows

The demo validates the complete haptix pipeline end-to-end:

1. **Real sensor data ingestion** — Loads GelSight frames (80 RGB images,
   480×640) and CoroCapacitive pressure arrays (786 frames, 29 columns) from
   the `research/real-data/` directory when present. Real GelSight frames are
   persisted as `.hapt` and join the training set (resized to the CNN input
   size by `MultiHaptDataset`).

2. **.hapt conversion** — Creates synthetic multi-material tactile data
   (5 materials × 5 trials × 12 frames), saves as `.hapt` files, reloads,
   and verifies checksum integrity. Synthetic frames carry **per-material
   micro-texture** (brushed streaks for metal, woven grid for fabric, grain
   stripes for wood, mottled blobs for rubber, speckled finish for plastic)
   plus per-frame noise — the texture is fixed per material so held-out
   frames are only recognizable by their surface pattern, the same signal a
   real GelSight measures.

3. **PyTorch integration** — Converts `.hapt` data into a PyTorch DataLoader
   with batching and shuffling. Each frame becomes a (C, H, W) float32 tensor
   normalized to [0, 1]. Real and synthetic `.hapt` files are mixed in one
   training loop.

4. **Training loop** — A tiny CNN (TinyTactileCNN, ~15K parameters) classifies
   tactile frames by material type (metal/plastic/fabric/wood/rubber/can)
   using cross-entropy loss and Adam optimizer.

5. **Evaluation** — Reports per-epoch loss and accuracy, plus final test set
   accuracy. The run is deterministic (`torch.manual_seed(0)`): 8 epochs
   converge to 99% training / 84% held-out accuracy on the mixed real +
   synthetic set (the real GelSight "can" class is the hardest part).

6. **Cross-sensor unified embedding (trained)** — A `CrossModalEncoder` is
   trained on paired synthetic records (GelSight-style imaging + Coro-style
   dynamic, same material labels in both). It learns CCA projections that map
   both modalities into a shared latent space where same-material records are
   close (cos ≈ 0.99) and different materials are far (cos ≈ −0.17). Real
   GelSight and Coro data embed through the trained encoder, weights
   serialize to `.npz` and reload losslessly, and the embedding survives the
   `.hapt` round-trip inside the `unified/` container path.

7. **Three storage formats** — The same data saves as a directory, a
   `.hapt.zarr` (Zstd-compressed ZipStore), or a `.hapt.zip` (stdlib DEFLATE
   archive, no extra dependencies). All three round-trip with identical
   checksums; compressed formats are ~0.75x raw size even on synthetic data.

## Why It Matters

This demo is the prerequisite for outreach to Eric Whittaker and the TouchNet
team. It demonstrates:

- **Format maturity:** `.hapt` handles both imaging (GelSight) and dynamic
  (CoroCapacitive) modalities, with verified checksum round-trips.
- **ML readiness:** Direct PyTorch DataLoader integration — no glue code.
  Researchers can go from sensor data to training loop in ~10 lines of haptix.
- **Sensor coverage:** Adapters for GelSight, DIGIT, CoroCapacitive, BioTac SP,
  and TacTip (5 sensor families).
- **Production quality:** 235 tests pass, CI green on Python 3.10/3.11/3.12,
  haptix 0.2.0 published on PyPI.
- **Shareable artifacts:** a single `.hapt.zip` file carries raw data +
  checksum + metadata + unified embedding — the "JPEG for touch" story.

## Running the Demo

```bash
# Install dependencies
pip install 'haptix[torch]' torch

# Run the demo
python examples/end_to_end_demo.py
```

Expected output:

```
====================================================================
  haptix — End-to-End Demo: Sensor Data → .hapt → Training
====================================================================
  haptix version: 0.2.0
  Registered sensors: ['CoroCapacitive', 'DIGIT_v2', 'DIGIT', ...]

  ✅ Loaded real GelSight data: (80, 480, 640, 3)
  ✅ Loaded real Coro data: (786, 29)
  ...
  Epoch  1:  loss=0.8279  train_acc=75.99%
  Epoch  2:  loss=0.3717  train_acc=94.41%
  ...
  Epoch  8:  loss=0.1496  train_acc=99.01%
====================================================================
  RESULTS
====================================================================
  Pipeline:   sensor data → .hapt → round-trip ✓ → DataLoader → CNN
  Classes:    ['can', 'fabric', 'metal', 'plastic', 'rubber', 'wood']
  Train acc:  99.01% (final epoch)
  Test acc:   84.21%
  Train time: 6.0s
  Total time: 11.2s
  .hapt spec: v0.2.0
  ✅ Demo complete — pipeline is working end-to-end.
====================================================================
```

## Architecture

```
┌─────────────────────┐
│  Real Sensor Data    │
│  (GelSight / Coro)   │
└────────┬────────────┘
         │ adapter.load()
         ▼
┌─────────────────────┐
│  .hapt file (HDF5)   │
│  • raw array         │
│  • checksum (SHA-256)│
│  • sensor metadata   │
│  • interaction labels│
└────────┬────────────┘
         │ haptix.save() / haptix.load()
         ▼
┌─────────────────────┐
│  HaptData object     │
│  • raw.numpy()       │
│  • raw.verify()      │
│  • to_torch()        │
└────────┬────────────┘
         │ DataLoader(batch_size=32)
         ▼
┌─────────────────────┐
│  TinyTactileCNN      │
│  Conv→BN→ReLU→Pool   │
│  Conv→BN→ReLU→Pool   │
│  AdaptivePool→FC      │
└─────────────────────┘
```

### TinyTactileCNN

| Layer | Input | Output | Params |
|-------|-------|--------|--------|
| Conv2d(3,16) k=3 | 3×60×80 | 16×60×80 | 448 |
| BatchNorm2d | 16 | 16 | 32 |
| MaxPool2d(2) | 16×60×80 | 16×30×40 | — |
| Conv2d(16,32) k=3 | 16×30×40 | 32×30×40 | 4,640 |
| BatchNorm2d | 32 | 32 | 64 |
| MaxPool2d(2) | 32×30×40 | 32×15×20 | — |
| AdaptiveAvgPool(8,8) | 32×15×20 | 32×8×8 | — |
| Linear | 2,048 | 5 | 10,245 |
| **Total** | | | **15,429** |

## Next Steps

After this demo is working:

1. **Replace synthetic data** with real GelSight/DIGIT frames from a
   multi-material dataset (YCB-Sight, Touch-and-Go, OPENTOUCH, etc.)
2. **Add augmentation** — random crops, rotations, brightness jitter
3. **Scale up** — more materials, deeper CNN, real-world accuracy metrics
4. **Outreach** — share demo + `.hapt.zip` sample with TouchNet team
