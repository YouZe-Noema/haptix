"""
Tests for low-level helpers in ``haptix.datasets.download``.

Covers ``_sha256_of``, ``verify_checksum``, ``_maybe_extract``, ``_http_download``,
and ``_default_cache_dir``. No real network access — urllib is monkeypatched.
Public ``download_dataset`` / cache APIs are covered in ``tests/test_datasets.py``.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from haptix.datasets import download as dl
from haptix.io import ChecksumError


def test_sha256_of_matches_hashlib(tmp_path: Path):
    """``_sha256_of`` digest equals ``hashlib.sha256(data).hexdigest()``."""
    payload = b"haptix-checksum-fixture-" + bytes(range(64))
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert dl._sha256_of(path) == expected


def test_sha256_of_multi_chunk(tmp_path: Path):
    """Small ``chunk_size`` still yields the correct full-file digest."""
    payload = b"abcdefghij" * 20  # 200 bytes
    path = tmp_path / "chunked.bin"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert dl._sha256_of(path, chunk_size=7) == expected


def test_verify_checksum_match(tmp_path: Path):
    """Matching digest returns True."""
    path = tmp_path / "ok.bin"
    data = b"ok-payload"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    assert dl.verify_checksum(path, digest) is True


def test_verify_checksum_mismatch_raises(tmp_path: Path):
    """Mismatch raises ChecksumError naming both expected and actual digests."""
    path = tmp_path / "bad.bin"
    data = b"actual-bytes"
    path.write_bytes(data)
    actual = hashlib.sha256(data).hexdigest()
    expected = "0" * 64
    with pytest.raises(ChecksumError) as excinfo:
        dl.verify_checksum(path, expected)
    msg = str(excinfo.value)
    assert expected in msg
    assert actual in msg


def test_maybe_extract_zip(tmp_path: Path):
    """``.zip`` archive extracts into the target dir; return value is that dir."""
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("nested/hello.txt", "zip-hello")
    extract_to = tmp_path / "out_zip"
    result = dl._maybe_extract(archive, extract_to)
    assert result == extract_to
    assert (extract_to / "nested" / "hello.txt").read_text() == "zip-hello"
    assert not archive.exists()


def test_maybe_extract_tar(tmp_path: Path):
    """Plain ``.tar`` archive extracts into the target dir."""
    archive = tmp_path / "bundle.tar"
    with tarfile.open(archive, "w") as tf:
        data = b"plain-tar"
        info = tarfile.TarInfo(name="plain.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    extract_to = tmp_path / "out_plain_tar"
    result = dl._maybe_extract(archive, extract_to)
    assert result == extract_to
    assert (extract_to / "plain.txt").read_bytes() == b"plain-tar"
    assert not archive.exists()


def test_maybe_extract_tar_gz(tmp_path: Path):
    """``.tar.gz`` archive extracts into the target dir."""
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        data = b"tar-hello"
        info = tarfile.TarInfo(name="payload.txt")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    extract_to = tmp_path / "out_tar"
    result = dl._maybe_extract(archive, extract_to)
    assert result == extract_to
    assert (extract_to / "payload.txt").read_bytes() == b"tar-hello"
    assert not archive.exists()


def test_maybe_extract_non_archive_moves_as_is(tmp_path: Path):
    """Non-archive file is moved into the extract dir unchanged."""
    src = tmp_path / "raw.dat"
    src.write_bytes(b"not-an-archive")
    extract_to = tmp_path / "plain_out"
    result = dl._maybe_extract(src, extract_to)
    assert result == extract_to
    assert (extract_to / "raw.dat").read_bytes() == b"not-an-archive"
    assert not src.exists()


def test_http_download_with_tqdm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When tqdm imports, ``_http_download`` takes the progress path."""
    dest = tmp_path / "prog.bin"
    payload = b"progress-bytes" * 8
    called = {"progress": False}

    class FakeResp:
        def __init__(self):
            self.headers = {"Content-Length": str(len(payload))}
            self._buf = io.BytesIO(payload)

        def read(self, n: int = -1):
            return self._buf.read(n)

    def fake_urlopen(url):
        return FakeResp()

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            called["progress"] = True

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, n):
            pass

    fake_tqdm_mod = SimpleNamespace(tqdm=FakeTqdm)
    monkeypatch.setitem(__import__("sys").modules, "tqdm", fake_tqdm_mod)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    dl._http_download("https://example.test/file.bin", dest)
    assert called["progress"] is True
    assert dest.read_bytes() == payload


def test_http_download_without_tqdm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When tqdm import fails, ``_http_download`` falls back to ``_download_simple``.

    ``sys.modules['tqdm'] = None`` makes ``import tqdm`` raise ImportError
    (``import of tqdm halted; None in sys.modules``), which trips the handler.
    """
    import sys

    dest = tmp_path / "simple.bin"
    payload = b"simple-download-bytes"
    monkeypatch.setitem(sys.modules, "tqdm", None)

    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(payload)

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    dl._http_download("https://example.test/simple.bin", dest)
    assert dest.read_bytes() == payload


def test_default_cache_dir_honours_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``HAPTIX_CACHE_DIR`` → ``<base>/datasets``."""
    base = tmp_path / "custom_cache"
    monkeypatch.setenv("HAPTIX_CACHE_DIR", str(base))
    assert dl._default_cache_dir() == base / "datasets"


def test_default_cache_dir_fallback_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Without env, returns ``~/.haptix/cache/datasets`` under monkeypatched HOME."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("HAPTIX_CACHE_DIR", raising=False)
    monkeypatch.setenv("HOME", str(home))
    assert dl._default_cache_dir() == home / ".haptix" / "cache" / "datasets"
