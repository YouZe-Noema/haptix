"""Dataset download and cache utilities.

Provides:
  - HTTP download with progress reporting
  - Local cache management under ``~/.haptix/cache/datasets/``
  - Idempotent downloads (skip if cached, re-download on demand)
  - SHA-256 integrity verification when the catalog pins a checksum
  - Cache inspection and cleanup
"""

import hashlib
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

from haptix.datasets.catalog import get_dataset_info


# Default cache location — override by setting HAPTIX_CACHE_DIR env var
def _default_cache_dir() -> Path:
    base = os.environ.get("HAPTIX_CACHE_DIR")
    if base:
        return Path(base) / "datasets"
    return Path.home() / ".haptix" / "cache" / "datasets"


DEFAULT_CACHE_DIR = _default_cache_dir()


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------


class ChecksumError(RuntimeError):
    """Raised when a downloaded archive fails SHA-256 verification."""


def _sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 digest of *path* in streaming fashion."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected_sha256: str) -> bool:
    """Return True if *path* matches *expected_sha256*.

    The expected digest is a 64-char lowercase hex string. A mismatch is
    reported via :class:`ChecksumError` — never silently ignored.
    """
    actual = _sha256_of(path)
    if actual != expected_sha256:
        raise ChecksumError(
            f"SHA-256 mismatch for {path.name}:\n"
            f"  expected: {expected_sha256}\n"
            f"  actual:   {actual}"
        )
    return True


# ---------------------------------------------------------------------------
# HTTP download helpers
# ---------------------------------------------------------------------------


def _http_download(url: str, dest: Path) -> None:
    """Download *url* to *dest* with a progress bar.

    Uses ``urllib`` from stdlib so no extra dependencies are required.
    Falls back gracefully if ``tqdm`` is not installed.
    """

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Use tqdm for a progress bar if available
        import tqdm  # noqa: F401

        _download_with_progress(url, dest)
    except ImportError:
        _download_simple(url, dest)


def _download_simple(url: str, dest: Path) -> None:
    """Download without a progress bar."""
    import urllib.request

    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


def _download_with_progress(url: str, dest: Path) -> None:
    """Download with a tqdm progress bar."""
    import urllib.request

    import tqdm

    response = urllib.request.urlopen(url)
    total = int(response.headers.get("Content-Length", 0))

    block_size = 1024 * 64  # 64 KB
    with (
        open(dest, "wb") as f,
        tqdm.tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as pbar,
    ):
        while True:
            chunk = response.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            pbar.update(len(chunk))


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------


def _maybe_extract(archive_path: Path, extract_to: Path) -> Path:
    """Extract *archive_path* into *extract_to* if it's an archive.

    If not an archive, the file is moved into *extract_to* as-is.
    Returns the final directory (usually *extract_to*).
    """
    suffix = archive_path.suffix
    extract_to.mkdir(parents=True, exist_ok=True)

    if suffix == ".tar":
        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_to)
        archive_path.unlink()
    elif suffix in (".gz", ".bz2", ".xz") and archive_path.name.endswith(
        ("tar.gz", "tar.bz2", "tar.xz")
    ):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_to)
        archive_path.unlink()
    elif suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_to)
        archive_path.unlink()
    else:
        # Not an archive — move into the dataset dir
        shutil.move(str(archive_path), str(extract_to / archive_path.name))

    return extract_to


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_dataset(
    name: str,
    cache_dir: str | Path | None = None,
    force: bool = False,
    extract: bool = True,
) -> Path:
    """Download a dataset from the catalog and cache it locally.

    Args:
        name: Dataset name from the catalog (e.g. ``"touch_and_go"``).
        cache_dir: Override the default cache directory.
        force: If True, re-download even if already cached.
        extract: If True, extract archives (``.tar.gz``, ``.zip``, etc.).

    Returns:
        Path to the cached dataset directory on disk.

    Raises:
        KeyError: If *name* is not in the catalog.
    """
    info = get_dataset_info(name)

    cache_root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    dataset_dir = cache_root / name

    # Idempotent: skip if already cached (unless force=True)
    if dataset_dir.exists() and not force:
        # Verify it's not empty
        if any(dataset_dir.iterdir()):
            return dataset_dir
        # Empty directory — treat as missing
        dataset_dir.rmdir()

    # Create the cache directory
    dataset_dir.mkdir(parents=True, exist_ok=True)

    url = info["url"]

    # Download to a temp name, then rename on success. Keep the real
    # filename (derived from the URL) so _maybe_extract can detect the
    # archive format from its suffix.
    url_filename = url.split("?")[0].rstrip("/").split("/")[-1]
    if not url_filename:
        url_filename = f"{name}.download"
    tmp_path = dataset_dir.with_name(f".{name}.{url_filename}")

    print(f"[haptix] Downloading dataset '{name}' from:")
    print(f"         {url}")
    print(f"         → {dataset_dir}")

    try:
        _http_download(url, tmp_path)

        # Verify integrity when the catalog pins a checksum.
        expected_sha256 = info.get("sha256")
        if expected_sha256:
            print(f"[haptix] Verifying SHA-256 ({expected_sha256[:12]}...)...")
            verify_checksum(tmp_path, expected_sha256)
            print("[haptix] Checksum OK ✓")

        if extract:
            # Rename to the real filename first so _maybe_extract can detect
            # the archive format, then extract into the dataset dir.
            clean_path = dataset_dir.with_name(url_filename)
            tmp_path.rename(clean_path)
            _maybe_extract(clean_path, dataset_dir)
        else:
            # Move temp file to final location without extraction
            shutil.move(str(tmp_path), str(dataset_dir / url_filename))

    except BaseException:
        # Clean up partial download on failure
        if tmp_path.exists():
            tmp_path.unlink()
        if dataset_dir.exists() and not any(dataset_dir.iterdir()):
            dataset_dir.rmdir()
        raise

    print(f"[haptix] Dataset '{name}' ready at {dataset_dir}")
    return dataset_dir


def cached_datasets(cache_dir: str | Path | None = None) -> list[str]:
    """Return a list of dataset names present in the local cache.

    Args:
        cache_dir: Override the default cache directory.
    """
    cache_root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    if not cache_root.exists():
        return []
    return sorted(p.name for p in cache_root.iterdir() if p.is_dir() and not p.name.startswith("."))


def cache_info(cache_dir: str | Path | None = None) -> dict:
    """Return summary statistics about the local cache.

    Args:
        cache_dir: Override the default cache directory.

    Returns:
        Dict with keys: ``cache_path``, ``total_datasets``, ``total_bytes``.
    """
    cache_root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    if not cache_root.exists():
        return {
            "cache_path": str(cache_root),
            "total_datasets": 0,
            "total_bytes": 0,
        }

    total_bytes = 0
    dataset_count = 0
    for entry in cache_root.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            dataset_count += 1
            total_bytes += _dir_size(entry)

    return {
        "cache_path": str(cache_root),
        "total_datasets": dataset_count,
        "total_bytes": total_bytes,
    }


def clear_cache(cache_dir: str | Path | None = None) -> None:
    """Remove the entire local dataset cache directory.

    Args:
        cache_dir: Override the default cache directory.
    """
    cache_root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    if cache_root.exists():
        shutil.rmtree(cache_root)
        print(f"[haptix] Cache cleared: {cache_root}")


def _dir_size(path: Path) -> int:
    """Recursively compute total size of *path* in bytes."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total += entry.stat().st_size
    return total
