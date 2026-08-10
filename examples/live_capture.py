#!/usr/bin/env python3
"""Live capture demo: incremental recording into .hapt.

Shows the real-time data collection toolkit (haptix.HaptRecorder):
frames arrive one at a time (here from a synthetic sensor), are appended
incrementally, and finalize into a fully valid .hapt directory on close.

Swap ``synthetic_camera_stream()`` for a real hardware source (DIGIT
camera, GelSight device, serial taxel array, ...) and the recording path
is identical.

Usage:
    python examples/live_capture.py [--frames N] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import haptix
from haptix.core import InteractionMeta, Labels, SensorMeta


def synthetic_camera_stream(n_frames: int, h: int = 60, w: int = 80, seed: int = 7):
    """Yield one GelSight-style frame per step (deterministic demo source).

    A moving gradient blob simulates a finger pressing and sliding on the
    gel — the same shape the real adapter stack produces per frame.
    """
    rng = np.random.RandomState(seed)
    base = rng.randint(120, 160, (h, w, 3), dtype=np.uint8)
    for t in range(n_frames):
        frame = base.copy()
        y = int(h * (0.3 + 0.4 * (t / max(1, n_frames))))
        x = int(w * (0.2 + 0.6 * (t / max(1, n_frames))))
        r = 8 + t % 6
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - x) ** 2 + (yy - y) ** 2 < r**2
        frame[mask] = np.clip(frame[mask].astype(np.int16) + 60, 0, 255).astype(np.uint8)
        noise = rng.randint(-8, 8, (h, w, 3)).astype(np.int16)
        yield np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--out", type=Path, default=Path("live_demo.hapt"))
    parser.add_argument("--buffer", type=int, default=16)
    args = parser.parse_args()

    print("=" * 62)
    print("  haptix — live capture demo (incremental recording)")
    print("=" * 62)

    rec = haptix.HaptRecorder(
        args.out,
        sensor=SensorMeta(type="GelSight", serial="demo-cam-01"),
        modality="imaging",
        sampling_rate_hz=30.0,
        interaction=InteractionMeta(type="sliding", speed_mm_s=40.0),
        labels=Labels(task="live_demo", material="synthetic"),
        buffer_frames=args.buffer,
    )

    t0 = time.time()
    try:
        for i, frame in enumerate(synthetic_camera_stream(args.frames)):
            rec.write_frame(frame, timestamp=i / 30.0)
            if (i + 1) % 20 == 0:
                print(f"  captured {i + 1:4d} frames (buffer={args.buffer})")
    finally:
        p = rec.close()

    print(f"  finalized in {time.time() - t0:.1f}s -> {p}")

    # Verify the result is a first-class .hapt file.
    data = haptix.load(p)
    print(f"  load: {data.raw.shape} checksum valid: {data.raw.verify()}")

    # And stream it back window-by-window (the other half of the toolkit).
    with haptix.open_archive(p) as arc:
        wins = arc.window_count(window_size=32, drop_last=True)
        print(f"  open_archive: {arc.n_frames} frames -> {wins} windows of 32")

    print("\n  ✅ Live capture pipeline works end-to-end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
