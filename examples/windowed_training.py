"""Windowed episode training with haptix.WindowedDataset.

Synthetic tactile episodes -> .hapt -> WindowedDataset -> DataLoader ->
tiny MLP classifier. Runs in a few seconds on CPU.

This is the "any PyTorch framework can consume .hapt" seam: Diffusion
Policy, ACT, LeRobot, and friends all consume windowed episode datasets
with the same (X, y) / DataLoader contract shown here.

    python examples/windowed_training.py
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import haptix
from haptix.core import HaptData, InteractionMeta, Labels, RawData, SensorMeta
from haptix.io import save
from haptix.torch_dataset import WindowedDataset

WINDOW = 32
STRIDE = 16
BATCH = 16
EPOCHS = 4


def make_episode(material: str, freq_hz: float, seed: int) -> HaptData:
    """One synthetic dynamic episode: sinusoid at *freq_hz* + noise.

    Each material gets a distinct frequency so the classifier has a real
    signal to learn (like texture frequency for real tactile data).
    """
    rng = np.random.RandomState(seed)
    t = np.arange(180) / 30.0
    tone = np.sin(2 * np.pi * freq_hz * t)[:, None]
    arr = (tone + 0.05 * rng.randn(180, 16)).astype(np.float32)
    return HaptData(
        raw=RawData(
            array=arr,
            checksum=RawData.compute_checksum(arr),
            dtype="float32",
            shape=arr.shape,
        ),
        sensor=SensorMeta(type="CoroCapacitive", serial="synth-001"),
        modality="dynamic",
        sampling_rate_hz=30.0,
        interaction=InteractionMeta(type="pressing"),
        labels=Labels(material=material, task="windowed_training_demo"),
    )


class WindowMLP(nn.Module):
    """Flattened-window MLP: window [T, D] -> material class."""

    def __init__(self, t: int, d: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(t * d, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="haptix-windowed-"))
    print(f"writing synthetic episodes to {workdir}")

    # 3 materials x 2 episodes each = 6 .hapt files (one episode per file)
    specs = [
        ("cotton", 1.0, 0),
        ("cotton", 1.0, 1),
        ("nylon", 3.0, 2),
        ("nylon", 3.0, 3),
        ("leather", 6.0, 4),
        ("leather", 6.0, 5),
    ]
    paths = []
    for i, (material, freq, seed) in enumerate(specs):
        p = save(make_episode(material, freq, seed), workdir / f"ep{i}.hapt")
        paths.append(p)

    # Episode/windowed dataset: every item is a [WINDOW, 16] window that
    # never crosses an episode boundary. String labels are encoded with a
    # single dataset-wide mapping (same material -> same class everywhere).
    ds = WindowedDataset(
        paths,
        window_size=WINDOW,
        stride=STRIDE,
        label="material",
        drop_last=True,  # fixed-length batches for the DataLoader
    )
    print(f"dataset: {len(ds)} windows across {ds.n_sources} episodes")
    print(f"classes: {ds.label_classes}")

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=True,
        # num_workers=0 keeps the demo portable (spawn-based multiprocessing
        # on macOS/Windows can hang at exit with this torch/numpy combo).
        # num_workers>0 works with the fork context (or on Linux defaults).
        num_workers=0,
        drop_last=True,
    )

    model = WindowMLP(WINDOW, 16, n_classes=len(ds.label_classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, correct = 0, 0
        for X, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
            total += y.numel()
            correct += (model(X).argmax(1) == y).sum().item()
        print(f"  epoch {epoch}: loss={loss.item():.3f} acc={correct / total:.2%}")

    model.eval()
    acc = sum(
        (model(X).argmax(1) == y).float().mean().item() for X, y in loader
    ) / len(loader)
    print(f"eval accuracy: {acc:.2%}")
    print("done — the same dataset object feeds Diffusion Policy / ACT / LeRobot loops.")


if __name__ == "__main__":
    main()
