#!/usr/bin/env python3
"""
One-command hands-on tour (docs/hands-on.md sections 3–5 + 7–8).

Runs non-interactively: download the checksum-verified demo sample, build a
``.hapt`` from real GelSight frames, round-trip all three storage backends,
``.to_torch()``, trained encoder embedding, and ``WindowedDataset`` windows.

Usage:
    python examples/walkthrough.py

Scratch output stays inside a ``tempfile.TemporaryDirectory()``.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path

import numpy as np

import haptix
from haptix.core import HaptData, InteractionMeta, Labels


def _banner(n: int, title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {n}. {title}")
    print("=" * 64)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(child.relative_to(path).as_posix().encode())
            digest.update(child.read_bytes())
    else:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_demo_gelsight() -> tuple[Path, HaptData]:
    demo = haptix.download_dataset("haptix_demo_sample")
    frames = demo / "gelsight" / "002_master_chef_can"
    data = haptix.get_sensor("GelSight").load(
        frames,
        interaction=InteractionMeta(type="pressing"),
        labels=Labels(material="can", object_name="002_master_chef_can"),
    )
    return demo, data


def main() -> None:
    t0 = time.time()
    print("=" * 64)
    print("  haptix — hands-on walkthrough")
    print("=" * 64)
    print(f"  haptix version: {haptix.__version__}")
    print(f"  sensors: {haptix.list_sensors()}")

    with tempfile.TemporaryDirectory(prefix="haptix_walkthrough_") as tmp:
        scratch = Path(tmp)

        _banner(3, "Explore a .hapt file (download → GelSight → save → to_torch)")
        demo, data = _load_demo_gelsight()
        print(f"  demo cache: {demo}")
        print(f"  cache_info: {haptix.cache_info()}")
        print(f"  loaded: {data}")

        path_hapt = scratch / "my_first.hapt"
        haptix.save(data, path_hapt)
        loaded = haptix.load(path_hapt)
        print(f"  reloaded: {loaded}")
        print(f"  provenance: {loaded.provenance}")
        assert np.array_equal(loaded.raw.array, data.raw.array)

        try:
            import torch  # noqa: F401
        except ImportError:
            print("  SKIP to_torch / WindowedDataset (torch not installed)")
            torch_ok = False
        else:
            torch_ok = True
            batch = next(iter(loaded.to_torch()))
            print(f"  torch batch: {batch[0].shape}")

        _banner(4, "Encoders (registry + load_trained GelSight)")
        print(f"  list_encoders: {haptix.list_encoders()}")
        trained = haptix.load_trained("GelSight")
        emb = trained.encode(loaded)
        print(f"  embedding: {emb.shape}")
        assert emb.shape == (loaded.raw.shape[0], 256)

        if torch_ok:
            _banner(5, "Windows / episodes (WindowedDataset)")
            from haptix import WindowedDataset

            wds = WindowedDataset(path_hapt, window_size=32, stride=8)
            print(f"  windows: {len(wds)}")
            wbatch = next(iter(wds))
            print(f"  window shape: {wbatch.shape}")
            assert len(wds) == 10
            assert tuple(wbatch.shape) == (32, 480, 640, 3)

        _banner(7, "Compression modes (.hapt / .hapt.zarr / .hapt.zip)")
        path_zarr = scratch / "my_first.hapt.zarr"
        path_zip = scratch / "my_first.hapt.zip"
        haptix.save(loaded, path_zarr)
        haptix.save(loaded, path_zip)

        for label, path in (
            (".hapt", path_hapt),
            (".hapt.zarr", path_zarr),
            (".hapt.zip", path_zip),
        ):
            again = haptix.load(path)
            assert again.raw.checksum == loaded.raw.checksum
            assert again.raw.verify()
            prov = again.provenance
            file_hash = prov.file_hash if prov is not None else None
            print(
                f"  {label:12s} checksum={again.raw.checksum[:16]}… "
                f"file_hash={None if file_hash is None else file_hash[:16] + '…'} "
                f"sha256(path)={_file_sha256(path)[:16]}…"
            )

        _banner(8, "Browser (shipped — optional)")
        print("  Optional: pip install -e '.[browser]' && haptix-browser", scratch)

    elapsed = time.time() - t0
    print()
    print(f"WALKTHROUGH OK ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
