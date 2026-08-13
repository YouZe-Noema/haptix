"""
Fetch published encoder weights (Hugging Face Hub) with checksum verification.

``haptix.load_trained(sensor_type)`` auto-downloads trained weights from the
HF Hub repo ``YouZe-Noema/haptix-encoders`` (pinned in the dataset catalog,
``docs/encoder-registry.md`` §7) when no local copy exists. This module is
the fetch layer: download → SHA-256 verify → cache → return the path.

Cache location: ``~/.haptix/cache/encoders/`` (override with the
``HAPTIX_CACHE_DIR`` env var, mirroring ``haptix.datasets``). Downloads are
idempotent: a cached copy is re-verified once per fetch (a mismatch is
treated as corruption and re-downloaded), so integrity is enforced on every
load without re-hitting the network.

Zero new dependencies: ``urllib`` handles the download (HF resolve URLs
redirect to the CDN) and ``verify_checksum`` from :mod:`haptix.datasets`
provides the streaming SHA-256 check.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

from haptix.datasets.catalog import get_encoder_weights
from haptix.datasets.download import verify_checksum
from haptix.io import ChecksumError

__all__ = ["encoder_cache_dir", "fetch_trained_weights"]


def encoder_cache_dir() -> Path:
    """Default encoder weights cache directory (``~/.haptix/cache/encoders``).

    Honors the ``HAPTIX_CACHE_DIR`` env var, mirroring
    :func:`haptix.datasets.download_dataset`.
    """
    base = os.environ.get("HAPTIX_CACHE_DIR")
    if base:
        return Path(base) / "encoders"
    return Path.home() / ".haptix" / "cache" / "encoders"


def fetch_trained_weights(sensor_type: str, *, cache_dir: str | Path | None = None) -> Path | None:
    """Download + verify published weights for *sensor_type*, or None.

    Consults the dataset catalog (:func:`get_encoder_weights`); if the
    sensor has no published weights, returns ``None`` (caller decides how
    to report). Downloads to the cache dir (default
    :func:`encoder_cache_dir`), verifies the pinned SHA-256, and returns
    the cached ``.npz`` path.

    Parameters
    ----------
    sensor_type : str
        Sensor family name (e.g. ``"GelSight"``).
    cache_dir : str or Path, optional
        Override the cache directory.

    Returns
    -------
    Path | None
        Path to the verified cached weight file, or None if the catalog
        has no published weights for *sensor_type*.

    Raises
    ------
    ChecksumError
        If the downloaded file does not match the pinned SHA-256.
    OSError / urllib.error.URLError
        Propagated from the network fetch (unreachable host, HTTP error,
        private repo, ...).
    """
    info = get_encoder_weights(sensor_type)
    if info is None:
        return None
    filename = info["weights_url"].split("?")[0].rstrip("/").split("/")[-1]
    if not filename.endswith(".npz"):
        filename = f"{sensor_type}_v1.0.npz"
    cache = Path(cache_dir) if cache_dir is not None else encoder_cache_dir()
    dest = cache / filename

    # Cached copy: verify once (cheap for <1 MB weights), treat mismatch as
    # corruption and re-download rather than silently serving bad weights.
    if dest.is_file():
        try:
            verify_checksum(dest, info["weights_sha256"])
            return dest
        except ChecksumError:
            print(f"[haptix] Cached weights corrupt ({dest}); re-downloading...")
            dest.unlink()

    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / f".{filename}.part"
    print(f"[haptix] Downloading encoder weights for '{sensor_type}':")
    print(f"         {info['weights_url']}")
    print(f"         → {dest}")
    try:
        urllib.request.urlretrieve(info["weights_url"], tmp)
        verify_checksum(tmp, info["weights_sha256"])
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    shutil.move(str(tmp), str(dest))
    print(f"[haptix] Verified + cached: {dest}")
    return dest
