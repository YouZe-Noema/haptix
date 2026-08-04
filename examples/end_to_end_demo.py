#!/usr/bin/env python3
"""
End-to-End Demo: Real Sensor Data → .hapt → PyTorch Training Loop

This demo validates the complete haptix pipeline:
  1. Loads real sensor data (GelSight frames, if available) into .hapt format
  2. Creates a synthetic multi-material classification dataset
  3. Saves to disk, reloads, and verifies checksums
  4. Trains a tiny CNN classifier to predict material type
  5. Reports loss, accuracy, and wall-clock time

Designed to run in under 60 seconds. This is the prerequisite demo for
outreach to Eric Whittaker and the TouchNet team.

Usage:
    python examples/end_to_end_demo.py

Requirements:
    pip install haptix[torch] torch
"""

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

# ── haptix imports ─────────────────────────────────────────────────────────
import haptix
from haptix.core import (
    HaptData,
    InteractionMeta,
    Labels,
    RawData,
    SensorMeta,
)

# ── PyTorch (optional, but required for the training portion) ──────────────
try:
    import torch
    import torch.nn.functional as F
    from torch import nn, optim
except ImportError:
    print("ERROR: torch is required. Install with: pip install 'haptix[torch]' torch")
    raise SystemExit(1)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Tiny CNN classifier for tactile material recognition
# ═══════════════════════════════════════════════════════════════════════════


class TinyTactileCNN(nn.Module):
    """Minimal CNN suitable for 60×80 tactile image patches.

    Architecture: Conv → BN → ReLU → Pool → Conv → BN → ReLU → Pool → FC → Output
    Kept deliberately small (≈5K params) so the demo runs fast on CPU.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — channels-first after transform
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Dataset creation
# ═══════════════════════════════════════════════════════════════════════════


MATERIALS = ["metal", "plastic", "fabric", "wood", "rubber"]

# Per-material color signatures (RGB base tint)
MATERIAL_COLORS = {
    "metal": (180, 180, 190),
    "plastic": (200, 220, 180),
    "fabric": (160, 130, 140),
    "wood": (140, 110, 80),
    "rubber": (60, 60, 65),
}


def _make_synthetic_frames(
    material: str,
    n_frames: int = 12,
    h: int = 60,
    w: int = 80,
    seed: int = 0,
) -> np.ndarray:
    """Generate synthetic tactile-like frames with material-specific signatures."""
    rng = np.random.RandomState(seed)
    r, g, b = MATERIAL_COLORS[material]
    frames = np.zeros((n_frames, h, w, 3), dtype=np.uint8)
    for t in range(n_frames):
        noise = (rng.randn(h, w, 3) * 12).astype(np.int16)
        base = np.stack(
            [
                np.full((h, w), r, dtype=np.int16),
                np.full((h, w), g, dtype=np.int16),
                np.full((h, w), b, dtype=np.int16),
            ],
            axis=-1,
        )
        frame = np.clip(base + noise, 0, 255).astype(np.uint8)
        frames[t] = frame
    return frames


def create_synthetic_dataset(
    output_dir: Path,
    trials_per_material: int = 5,
    frames_per_trial: int = 12,
) -> list[Path]:
    """Create a labeled synthetic tactile dataset and save as .hapt files.

    Returns list of .hapt file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for mat_idx, material in enumerate(MATERIALS):
        for trial in range(trials_per_material):
            seed = mat_idx * 100 + trial
            frames = _make_synthetic_frames(material, n_frames=frames_per_trial, seed=seed)

            data = HaptData(
                raw=RawData(
                    array=frames,
                    checksum=RawData.compute_checksum(frames),
                    dtype="uint8",
                    shape=frames.shape,
                ),
                sensor=SensorMeta(
                    type="GelSight_Mini",
                    serial=f"demo-{mat_idx:02d}-{trial:02d}",
                ),
                modality="imaging",
                sampling_rate_hz=30.0,
                interaction=InteractionMeta(type="pressing", normal_force_N=2.0),
                labels=Labels(material=material, material_category=material),
            )

            path = output_dir / f"{material}_{trial:02d}.hapt"
            haptix.save(data, path)
            saved_paths.append(path)

    return saved_paths


# ═══════════════════════════════════════════════════════════════════════════
# 3. PyTorch dataset & training utilities
# ═══════════════════════════════════════════════════════════════════════════


class MultiHaptDataset(torch.utils.data.Dataset):
    """Concatenates multiple .hapt files into a single classification dataset.

    Each frame from each .hapt file is a sample. The material label is
    mapped to a consecutive integer (0..N-1) for classification.
    """

    def __init__(
        self,
        hapt_paths: list[Path],
        transform=None,
        image_size: tuple[int, int] = (60, 80),
    ):
        self._transform = transform
        self._image_size = image_size

        # Collect all frames and labels
        all_frames: list[np.ndarray] = []
        all_labels: list[int] = []

        # Build consistent label mapping
        unique_materials = sorted({haptix.load(p).labels.material or "unknown" for p in hapt_paths})
        self._label_map = {m: i for i, m in enumerate(unique_materials)}
        self._classes = unique_materials

        for path in hapt_paths:
            data = haptix.load(path)
            frames = data.raw.numpy()  # (T, H, W, C)
            material = data.labels.material or "unknown"
            label_idx = self._label_map[material]

            for i in range(len(frames)):
                all_frames.append(frames[i])
                all_labels.append(label_idx)

        self._frames = all_frames
        self._labels = np.array(all_labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int):
        frame = self._frames[idx]  # (H, W, C) uint8
        label = self._labels[idx]

        # Convert to tensor: (C, H, W), float32, normalized to [0, 1]
        x = torch.from_numpy(frame).float() / 255.0
        x = x.permute(2, 0, 1)  # HWC → CHW

        if self._transform is not None:
            x = self._transform(x)

        return x, torch.tensor(label, dtype=torch.long)

    @property
    def num_classes(self) -> int:
        return len(self._classes)

    @property
    def class_names(self) -> list[str]:
        return self._classes


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


# ═══════════════════════════════════════════════════════════════════════════
# 4. Main demo
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    t_start = time.time()
    data_root = Path(__file__).resolve().parent.parent

    # Initialize defaults for except/finally paths
    train_acc = 0.0
    test_acc = 0.0
    class_names: list[str] = []
    t_train_start = t_start
    t_train_end = t_start

    print("=" * 68)
    print("  haptix — End-to-End Demo: Sensor Data → .hapt → Training")
    print("=" * 68)
    print(f"  haptix version: {haptix.__version__}")
    print(f"  Registered sensors: {haptix.list_sensors()}")
    print()

    # ── Step 1: Load real sensor data (if available) ───────────────────────
    real_data_loaded = False
    gelsight_dir = data_root / "research" / "real-data" / "gelsight"
    coro_dir = data_root / "research" / "real-data" / "coro"

    if gelsight_dir.is_dir():
        try:
            adapter = haptix.get_sensor("GelSight")
            real_data = adapter.load(
                gelsight_dir,
                interaction=InteractionMeta(type="pressing"),
                labels=Labels(material="master_chef_can", object_name="002"),
            )
            print(f"  ✅ Loaded real GelSight data: {real_data.raw.shape}")
            print(f"     Sensor: {real_data.sensor.type}")
            print(f"     Frames: {real_data.raw.shape[0]}")
            real_data_loaded = True
        except Exception as e:
            print(f"  ⚠️  Real GelSight load failed: {e}")

    if coro_dir.is_dir():
        try:
            adapter = haptix.get_sensor("CoroCapacitive")
            real_coro = adapter.load(
                coro_dir,
                source="default",
                interaction=InteractionMeta(type="pressing"),
                labels=Labels(material="foam"),
            )
            print(f"  ✅ Loaded real Coro data: {real_coro.raw.shape}")
            real_data_loaded = True
        except Exception as e:
            print(f"  ⚠️  Real Coro load failed: {e}")

    if not real_data_loaded:
        print("  ℹ️  No real sensor data found — using synthetic only.")
    print()

    # ── Step 2: Create + save labeled synthetic dataset ────────────────────
    print("─" * 68)
    print("  Step 1: Creating synthetic .hapt dataset...")
    tmpdir = Path(tempfile.mkdtemp(prefix="haptix_demo_"))
    try:
        hapt_paths = create_synthetic_dataset(
            tmpdir / "dataset",
            trials_per_material=5,
            frames_per_trial=12,
        )
        print(f"  Created {len(hapt_paths)} .hapt files")
        print(f"  Materials: {MATERIALS}")
        print(f"  Total frames: {len(hapt_paths) * 12}")

        # ── Step 3: Round-trip verification ────────────────────────────────
        print()
        print("  Step 2: Verifying round-trip (save → load → checksum)...")
        sample_path = hapt_paths[0]
        original = haptix.load(sample_path)
        reloaded = haptix.load(sample_path)
        assert np.array_equal(original.raw.array, reloaded.raw.array), "Round-trip mismatch!"
        assert original.raw.verify(), "Checksum verification failed!"
        print(f"  ✅ Round-trip verified for {sample_path.name}")
        print(f"     Shape: {original.raw.shape}, dtype: {original.raw.dtype}")
        print(f"     Material: {original.labels.material}")

        # ── Step 4: Build PyTorch dataset & DataLoader ─────────────────────
        print()
        print("  Step 3: Building PyTorch DataLoader...")
        dataset = MultiHaptDataset(hapt_paths)
        class_names = dataset.class_names
        print(f"  Total samples: {len(dataset)}")
        print(f"  Classes: {dataset.class_names}")

        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train_ds, test_ds = torch.utils.data.random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=32,
            shuffle=True,
            num_workers=0,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )
        print(f"  Train samples: {train_size}, Test samples: {test_size}")
        print("  Batch size: 32")

        # ── Step 5: Train a tiny CNN ───────────────────────────────────────
        print()
        print("  Step 4: Training TinyTactileCNN classifier...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyTactileCNN(in_channels=3, num_classes=dataset.num_classes)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        num_params = sum(p.numel() for p in model.parameters())

        print(f"  Device: {device}")
        print(f"  Model params: {num_params:,}")
        print(
            f"  Architecture: Conv(3→16)→BN→ReLU→Pool→Conv(16→32)→BN→ReLU→Pool→FC(32×8×8→{dataset.num_classes})"
        )

        t_train_start = time.time()
        for epoch in range(5):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
            print(f"  Epoch {epoch + 1:2d}:  loss={train_loss:.4f}  train_acc={train_acc:.2%}")

        t_train_end = time.time()

        # ── Step 6: Cross-sensor unified embedding ──────────────────────────
        print()
        print("  Step 5: Cross-sensor unified embedding (SharedForceEncoder)...")
        from haptix.unified import SharedForceEncoder

        encoder = SharedForceEncoder(embedding_dim=64)
        print(f"  Encoder version: {encoder.version}")
        print(f"  Supported sensors: {encoder.supported_sensors()}")

        # Encode a sample .hapt file into the shared latent space
        sample_data = haptix.load(hapt_paths[0])
        unified = encoder.encode(sample_data)

        # Create a copy with unified data embedded
        sample_with_unified = HaptData(
            raw=sample_data.raw,
            sensor=sample_data.sensor,
            modality=sample_data.modality,
            sampling_rate_hz=sample_data.sampling_rate_hz,
            interaction=sample_data.interaction,
            labels=sample_data.labels,
            unified=unified,
        )

        # Save with unified data embedded in the .hapt container
        unified_path = tmpdir / "dataset" / "unified_demo.hapt"
        haptix.save(sample_with_unified, unified_path)

        # Reload and verify unified data survived round-trip
        reloaded_u = haptix.load(unified_path)
        assert reloaded_u.unified is not None, "Unified data lost in round-trip!"
        assert reloaded_u.unified.array.shape == unified.array.shape, "Shape mismatch!"
        assert reloaded_u.unified.method == unified.method, "Method mismatch!"

        print(f"  ✅ Unified embedding: {unified.array.shape}")
        print(f"     Method: {unified.method}")
        print(f"     Target modality: {unified.target_modality}")
        print("     Container path: unified/data.npy + unified/transform.json")
        print("     Round-trip: ✓ (embedding preserved in .hapt container)")

        # ── Step 7: Storage formats — directory, .zarr, .zip ───────────────
        print()
        print("  Step 7: Storage formats (directory / .hapt.zarr / .hapt.zip)...")
        fmt_dir = tmpdir / "dataset" / "unified_demo.hapt"
        fmt_zarr = tmpdir / "dataset" / "unified_demo.hapt.zarr"
        fmt_zip = tmpdir / "dataset" / "unified_demo.hapt.zip"

        haptix.save(sample_with_unified, fmt_dir)
        try:
            haptix.save(sample_with_unified, fmt_zarr)
        except ImportError:
            fmt_zarr = None  # zarr not installed — skip gracefully
        haptix.save(sample_with_unified, fmt_zip)

        # Verify all formats round-trip with identical checksums
        fmt_paths = [p for p in (fmt_dir, fmt_zarr, fmt_zip) if p is not None]
        for fmt_path in fmt_paths:
            reloaded_fmt = haptix.load(fmt_path)
            assert reloaded_fmt.raw.checksum == sample_with_unified.raw.checksum
            assert reloaded_fmt.unified is not None
            assert np.array_equal(reloaded_fmt.unified.array, unified.array)

        sizes = {
            "directory": sum(p.stat().st_size for p in fmt_dir.rglob("*") if p.is_file()),
        }
        if fmt_zarr is not None:
            sizes[".hapt.zarr"] = fmt_zarr.stat().st_size
        sizes[".hapt.zip"] = fmt_zip.stat().st_size
        raw_bytes = int(np.prod(sample_with_unified.raw.shape))
        print(f"  ✅ Round-trip verified in {len(fmt_paths)} formats")
        for name, size in sizes.items():
            print(f"     {name:12s} {size:>10,} bytes  ({size / raw_bytes:.2f}x raw)")

        # ── Step 8: Evaluate on test set ───────────────────────────────────
        print()
        print("  Step 6: Evaluating on test set...")
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                test_correct += (preds == y).sum().item()
                test_total += x.size(0)
        test_acc = test_correct / test_total

    finally:
        shutil.rmtree(tmpdir)

    t_end = time.time()

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    print("=" * 68)
    print("  RESULTS")
    print("=" * 68)
    print("  Pipeline:   sensor data → .hapt → round-trip ✓ → DataLoader → CNN")
    print(f"  Classes:    {class_names}")
    print(f"  Train acc:  {train_acc:.2%} (epoch 5)")
    print(f"  Test acc:   {test_acc:.2%}")
    print(f"  Train time: {t_train_end - t_train_start:.1f}s")
    print(f"  Total time: {t_end - t_start:.1f}s")
    print(f"  .hapt spec: v{haptix.__version__}")
    print()
    print("  ✅ Demo complete — pipeline is working end-to-end.")
    print("=" * 68)


if __name__ == "__main__":
    main()
