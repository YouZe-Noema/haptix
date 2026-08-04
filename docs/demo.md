# haptix End-to-End Demo

> **Status:** ✅ Working (tested 2026-08-04)
> **Runtime:** ~8 seconds on CPU
> **Script:** `examples/end_to_end_demo.py`
> **Package:** haptix 0.2.0 on PyPI — `pip install haptix[torch] torch`

## What It Shows

The demo validates the complete haptix pipeline end-to-end:

1. **Real sensor data ingestion** — Loads GelSight frames (80 RGB images,
   480×640) and CoroCapacitive pressure arrays (786 frames, 29 columns) from
   the `research/real-data/` directory when present.

2. **.hapt conversion** — Creates synthetic multi-material tactile data
   (5 materials × 5 trials × 12 frames), saves as `.hapt` files, reloads,
   and verifies checksum integrity.

3. **PyTorch integration** — Converts `.hapt` data into a PyTorch DataLoader
   with batching and shuffling. Each frame becomes a (C, H, W) float32 tensor
   normalized to [0, 1].

4. **Training loop** — A tiny CNN (TinyTactileCNN, ~15K parameters) classifies
   tactile frames by material type (metal/plastic/fabric/wood/rubber) using
   cross-entropy loss and Adam optimizer.

5. **Evaluation** — Reports per-epoch loss and accuracy, plus final test set
   accuracy. With synthetic data, converges to ~100% accuracy in 2-3 epochs.

6. **Cross-sensor unified embedding** — The SharedForceEncoder maps the
   sample into a shared latent space (`unified/` in the container), which
   survives the round-trip.

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
- **Production quality:** 217 tests pass, CI green on Python 3.10/3.11/3.12,
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
  haptix version: 0.1.0
  Registered sensors: ['CoroCapacitive', 'DIGIT_v2', 'DIGIT', ...]

  ✅ Loaded real GelSight data: (80, 480, 640, 3)
  ✅ Loaded real Coro data: (786, 29)
  ...
  Epoch  1:  loss=1.3654  train_acc=65.83%
  Epoch  2:  loss=0.2905  train_acc=84.17%
  Epoch  3:  loss=0.0198  train_acc=100.00%
  ...
====================================================================
  RESULTS
====================================================================
  Pipeline:   sensor data → .hapt → round-trip ✓ → DataLoader → CNN
  Test acc:   100.00%
  Total time: 4.9s
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
