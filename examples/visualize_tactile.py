#!/usr/bin/env python3
"""
Example 3: Visualize tactile sensor data.

This example demonstrates how to:
  1. Load DIGIT frames from a .hapt file and inspect per-frame statistics
  2. Compute a contact mask (where the elastomer is pressed)
  3. Plot frame statistics over time (mean intensity, contact area, variance)
  4. Create a simple "tactile video" from frames
  5. Visualize temporal derivatives (motion within the tactile image)

Outputs are saved as PNG images in an output directory.

Usage:
    # Run with synthetic demo data:
    python examples/visualize_tactile.py

    # Point to an existing .hapt file:
    python examples/visualize_tactile.py /path/to/data.hapt

    # Specify output directory:
    python examples/visualize_tactile.py /path/to/data.hapt --output ./figures/

Requirements:
    pip install haptix[all] pillow matplotlib
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from haptix import get_sensor, load, save
from haptix.core import (
    HaptData,
    InteractionMeta,
    Labels,
    RawData,
    SensorMeta,
)

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


def compute_frame_metrics(frames: np.ndarray) -> dict:
    """Compute per-frame metrics from a (T, H, W, C) tactile video.

    Returns dict of (T,) arrays:
        mean_intensity: mean pixel value per frame
        std_intensity:  standard deviation per frame
        contact_area:   fraction of pixels below a dark threshold (contact)
        frame_diff:     mean absolute frame-to-frame difference (motion proxy)
    """
    T = frames.shape[0]
    gray = frames.mean(axis=-1).astype(np.float32)  # (T, H, W)

    mean_intensity = gray.reshape(T, -1).mean(axis=1)
    std_intensity = gray.reshape(T, -1).std(axis=1)

    # Contact area: pixels darker than 40 (elastomer depression)
    contact_area = (gray < 40).reshape(T, -1).mean(axis=1)

    # Frame-to-frame difference
    frame_diff = np.zeros(T)
    if T > 1:
        for t in range(1, T):
            frame_diff[t] = np.abs(gray[t] - gray[t - 1]).mean()

    return {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "contact_area": contact_area,
        "frame_diff": frame_diff,
    }


def render_frame_grid(frames: np.ndarray, output_path: Path, max_frames: int = 16):
    """Render frames as a grid of thumbnails.

    Saves a montage image showing the first N frames.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = min(len(frames), max_frames)
    cols = 4
    rows = int(np.ceil(T / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    for i in range(T):
        frame = frames[i]
        if frame.shape[-1] == 1:
            axes[i].imshow(frame.squeeze(), cmap="gray")
        else:
            axes[i].imshow(frame)
        axes[i].set_title(f"Frame {i}", fontsize=8)
        axes[i].axis("off")

    for i in range(T, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"DIGIT Tactile Frames (showing {T}/{len(frames)})", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved frame grid: {output_path}")


def plot_metrics(metrics: dict, output_path: Path, sampling_rate_hz: float = 60.0):
    """Plot per-frame statistics over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(metrics["mean_intensity"])
    time_sec = np.arange(T) / sampling_rate_hz

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle("Tactile Frame Metrics Over Time", fontsize=13)

    plots = [
        (axes[0, 0], time_sec, metrics["mean_intensity"], "Mean Intensity", "gray value"),
        (axes[0, 1], time_sec, metrics["std_intensity"], "Std Intensity", "gray value"),
        (axes[1, 0], time_sec, metrics["contact_area"], "Contact Area Fraction", "fraction"),
        (axes[1, 1], time_sec, metrics["frame_diff"], "Frame-to-Frame Diff", "mean abs diff"),
    ]

    for ax, t, y, title, ylabel in plots:
        ax.plot(t, y, "-", linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved metrics plot: {output_path}")


def render_contact_map(frames: np.ndarray, output_path: Path, threshold: int = 40):
    """Render an average contact map from all frames.

    The contact map shows which regions of the sensor were pressed,
    averaged across all frames. Dark regions = sustained contact.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gray = frames.mean(axis=-1)  # (T, H, W) — average channels if RGB
    avg_frame = gray.mean(axis=0)  # (H, W) — average over time
    contact_mask = avg_frame < threshold

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("Tactile Contact Analysis", fontsize=13)

    im0 = axes[0].imshow(avg_frame, cmap="gray")
    axes[0].set_title("Mean Frame (time-averaged)")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(contact_mask, cmap="Reds", interpolation="nearest")
    axes[1].set_title(f"Contact Mask (< {threshold})")
    axes[1].set_xlabel(f"Contact area: {contact_mask.mean():.1%}")
    plt.colorbar(im1, ax=axes[1])

    # Temporal variance — regions that changed most
    temporal_var = gray.std(axis=0)
    im2 = axes[2].imshow(temporal_var, cmap="hot")
    axes[2].set_title("Temporal Variance (motion)")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved contact analysis: {output_path}")


def render_tactile_differential(frames: np.ndarray, output_path: Path):
    """Render frame-difference images to show deformation over time.

    Each frame is compared to the first frame (baseline). Differences
    highlight where the elastomer changed shape.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gray = frames.mean(axis=-1).astype(np.float32)  # (T, H, W)
    baseline = gray[0]  # first frame as reference

    T = min(len(frames), 8)
    cols = 4
    rows = int(np.ceil(T / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    for i in range(T):
        diff = np.abs(gray[i] - baseline)
        axes[i].imshow(diff, cmap="viridis")
        axes[i].set_title(f"Frame {i} - Frame 0", fontsize=8)
        axes[i].axis("off")

    for i in range(T, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Differential Tactile: Deformation from Baseline", fontsize=12)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Saved differential view: {output_path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _make_demo_hapt(output_dir: Path) -> Path:
    """Create a realistic synthetic DIGIT .hapt file with a simulated press."""
    rng = np.random.RandomState(42)
    n_frames = 20
    H, W = 240, 320

    frames = np.zeros((n_frames, H, W, 3), dtype=np.uint8)

    for t in range(n_frames):
        # Base texture (like unpressed elastomer)
        frame = rng.poisson(lam=25, size=(H, W, 3)).clip(0, 255).astype(np.uint8)

        # Simulate increasing then decreasing pressure at center
        progress = t / n_frames  # 0 → 1
        press_depth = np.sin(progress * np.pi)  # 0 → 1 → 0 (press and release)

        cy, cx = H // 2, W // 2
        yy, xx = np.ogrid[:H, :W]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        radius = 60 + 20 * np.sin(progress * np.pi * 2)  # pulsing contact patch

        # Gaussian contact profile
        contact_effect = press_depth * np.exp(-(dist ** 2) / (2 * (radius / 2) ** 2))
        contact_effect = contact_effect[..., np.newaxis]

        # Darken the contacted area (elastomer depression)
        frame = (frame * (1 - 0.6 * contact_effect)).astype(np.uint8)

        # Add some temporal noise
        noise = rng.randint(-3, 4, (H, W, 3), dtype=np.int8)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        frames[t] = frame

    data = HaptData(
        raw=RawData(
            array=frames,
            checksum=RawData.compute_checksum(frames),
            dtype="uint8",
            shape=frames.shape,
        ),
        sensor=SensorMeta(type="DIGIT_v2", serial="DIGIT-2026-007"),
        modality="imaging",
        sampling_rate_hz=60.0,
        interaction=InteractionMeta(
            type="pressing",
            normal_force_N=2.5,
            speed_mm_s=10.0,
        ),
        labels=Labels(
            material="silicone_rubber",
            material_category="elastomer",
            task="pressing",
        ),
    )

    hapt_path = output_dir / "demo_press.hapt"
    save(data, hapt_path)
    return hapt_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Visualize tactile sensor data from .hapt files"
    )
    parser.add_argument("input", nargs="?", default=None,
                        help="Path to .hapt file or DIGIT frame directory")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory for visualization images")
    args = parser.parse_args()

    print("=" * 60)
    print("haptix — Tactile Data Visualization Example")
    print("=" * 60)

    # Load data
    if args.input:
        path = Path(args.input)
        if path.suffix == ".hapt" and path.is_dir():
            print(f"\nLoading .hapt file: {path}")
            data = load(path)
        else:
            print(f"\nLoading via DIGIT adapter: {path}")
            digit = get_sensor("DIGIT_v2")
            data = digit.load(
                path,
                interaction=InteractionMeta(type="pressing"),
                labels=Labels(),
            )
    else:
        print("\nNo input provided — generating synthetic DIGIT data...")
        tmp = Path(tempfile.mkdtemp(prefix="haptix_viz_"))
        hapt_path = _make_demo_hapt(tmp)
        data = load(hapt_path)

    print(f"  Sensor: {data.sensor.type}  Modality: {data.modality}")
    print(f"  Frames: {data.raw.shape[0]}  Resolution: {data.raw.shape[1]}x{data.raw.shape[2]}")
    print(f"  Channels: {data.raw.shape[3]}  dtype: {data.raw.dtype}")
    print(f"  Interaction: {data.interaction.type}  Force: {data.interaction.normal_force_N} N")
    print(f"  Labels: material={data.labels.material}  task={data.labels.task}")

    frames = data.raw.numpy()

    # Setup output
    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="haptix_viz_out_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}\n")

    # ---- Visualizations ----
    print("-" * 40)
    print("1. Frame metrics")
    print("-" * 40)
    metrics = compute_frame_metrics(frames)
    print(f"   Mean intensity range: {metrics['mean_intensity'][0]:.1f} — {metrics['mean_intensity'][-1]:.1f}")
    print(f"   Contact area range:   {metrics['contact_area'].min():.1%} — {metrics['contact_area'].max():.1%}")
    print(f"   Max frame diff:       {metrics['frame_diff'].max():.2f}")
    plot_metrics(metrics, out_dir / "frame_metrics.png", data.sampling_rate_hz)

    print("\n" + "-" * 40)
    print("2. Frame grid")
    print("-" * 40)
    render_frame_grid(frames, out_dir / "frame_grid.png")

    print("\n" + "-" * 40)
    print("3. Contact analysis (mean frame + contact mask + variance)")
    print("-" * 40)
    render_contact_map(frames, out_dir / "contact_analysis.png")

    print("\n" + "-" * 40)
    print("4. Differential view (deformation from baseline)")
    print("-" * 40)
    render_tactile_differential(frames, out_dir / "differential_view.png")

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print(f"\nOutputs saved to: {out_dir}")
    print(f"  {out_dir / 'frame_metrics.png'}")
    print(f"  {out_dir / 'frame_grid.png'}")
    print(f"  {out_dir / 'contact_analysis.png'}")
    print(f"  {out_dir / 'differential_view.png'}")


if __name__ == "__main__":
    main()
