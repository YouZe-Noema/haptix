"""Helpers that make test-time file corruption verifiably effective.

A fixed-constant overwrite (e.g. ``arr[0] = 0``) is a silent no-op when the
value is already that constant — the file bytes do not change, checksums still
match, and checksum-error tests flake. These helpers XOR so the write always
differs, then optionally assert the bytes changed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def flip_pixel_in_npy(path: Path) -> None:
    """Flip one element in a ``.npy`` file so the on-disk bytes must change.

    XOR with ``0xFF`` is never a no-op for any integer element value, so the
    tamper cannot silently leave the file identical. The file stays a valid
    ``.npy`` (unlike truncating bytes), so loaders reach the checksum check
    rather than failing on format.
    """
    before = Path(path).read_bytes()
    arr = np.load(path)
    arr.flat[0] = int(arr.flat[0]) ^ 0xFF
    np.save(path, arr)
    after = Path(path).read_bytes()
    if before == after:
        raise AssertionError(f"tamper was a no-op — file bytes unchanged at {path}")


def xor_byte(data: bytes | bytearray, index: int = -20) -> bytes:
    """Return a copy with one byte XOR ``0xFF`` (always different from input).

    Guarantees archive / streaming tampers cannot silently no-op: XOR flips
    every bit of that byte, so the returned buffer differs from ``data``.
    """
    out = bytearray(data)
    out[index] ^= 0xFF
    return bytes(out)
