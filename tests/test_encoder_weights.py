"""
Tests for published encoder weights (HF Hub wiring).

Covers the catalog encoder-weights registry (weights_url + weights_sha256),
the checksum-verified download/cache flow (haptix.encoders.weights_download),
and `load_trained()` auto-download. Network-free: downloads are served from
local ``file://`` URLs via monkeypatched catalog lookups; the real HF URLs
are never hit.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

import haptix
from haptix.datasets.catalog import get_dataset_info, get_encoder_weights
from haptix.encoders import load_trained
from haptix.encoders.weights_download import encoder_cache_dir, fetch_trained_weights
from haptix.io import ChecksumError


# ── catalog registry ──────────────────────────────────────────────────────


def test_get_encoder_weights_gelsight():
    info = get_encoder_weights("GelSight")
    assert info is not None
    assert info["weights_url"].startswith("https://huggingface.co/")
    assert "haptix-encoders" in info["weights_url"]
    assert len(info["weights_sha256"]) == 64
    int(info["weights_sha256"], 16)  # valid hex
    assert info["embedding_dim"] == 256
    assert "benchmark" in info


def test_get_encoder_weights_coro():
    info = get_encoder_weights("CoroCapacitive")
    assert info is not None
    assert info["embedding_dim"] == 128
    assert len(info["weights_sha256"]) == 64


def test_get_encoder_weights_unknown_is_none():
    assert get_encoder_weights("NoSuchSensor") is None
    assert get_encoder_weights("DIGIT") is None  # not published yet


def test_catalog_entries_pin_encoder_blocks():
    """Design doc §7: catalog dataset entries carry the encoder block."""
    coro = get_dataset_info("coro_tactile")["encoder"]
    assert coro["weights_sha256"] == get_encoder_weights("CoroCapacitive")["weights_sha256"]
    assert coro["weights_url"].endswith("CoroCapacitive_v1.0.npz")

    ycb = get_dataset_info("ycb_slide")["encoder"]
    assert ycb["weights_sha256"] == get_encoder_weights("GelSight")["weights_sha256"]
    assert ycb["weights_url"].endswith("GelSight_v1.0.npz")


@pytest.mark.skipif(
    not (
        Path(haptix.__file__).resolve().parent / "encoders" / "weights" / "GelSight_v1.0.npz"
    ).is_file(),
    reason="local gitignored weights not present (e.g. CI)",
)
def test_catalog_sha_matches_shipped_weights():
    """The pinned sha256 must match the actual shipped weight files."""
    weights_dir = Path(haptix.__file__).resolve().parent / "encoders" / "weights"
    for sensor, filename in [
        ("GelSight", "GelSight_v1.0.npz"),
        ("CoroCapacitive", "CoroCapacitive_v1.0.npz"),
    ]:
        digest = hashlib.sha256((weights_dir / filename).read_bytes()).hexdigest()
        assert digest == get_encoder_weights(sensor)["weights_sha256"], sensor


# ── fetch / cache flow ────────────────────────────────────────────────────


def _local_weights(tmp_path: Path) -> Path:
    """A tiny valid .npz weight file for download tests."""
    rng = np.random.RandomState(0)
    w = rng.randn(8, 8).astype(np.float64)
    p = tmp_path / "GelSight_v1.0.npz"
    np.savez(
        p,
        sensor_type="GelSight",
        modality="imaging",
        embedding_dim=8,
        version="encoders/gelsight/v1.0",
        trained=True,
        mean=rng.randn(8),
        W=w,
    )
    return p


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _monkeypatch_weights(monkeypatch, weights_path: Path, sha: str | None = None):
    """Point the weight-fetch layer at a local file:// weight file.

    Patches the binding inside haptix.encoders.weights_download (that module
    imports ``get_encoder_weights`` at import time), so the real HF URLs are
    never hit.
    """
    monkeypatch.setattr(
        "haptix.encoders.weights_download.get_encoder_weights",
        lambda sensor: (
            {
                "weights_url": weights_path.as_uri(),
                "weights_sha256": sha if sha is not None else _sha256(weights_path),
                "embedding_dim": 8,
                "benchmark": "test",
                "license": "MIT",
                "homepage": "file://test",
            }
            if sensor == "GelSight"
            else None
        ),
    )


def test_fetch_downloads_and_verifies(tmp_path, monkeypatch):
    src = _local_weights(tmp_path)
    _monkeypatch_weights(monkeypatch, src)
    cache = tmp_path / "cache"
    result = fetch_trained_weights("GelSight", cache_dir=cache)
    assert result is not None
    assert result == cache / "GelSight_v1.0.npz"
    assert result.is_file()
    assert _sha256(result) == _sha256(src)


def test_fetch_reuses_cache_without_redownload(tmp_path, monkeypatch):
    src = _local_weights(tmp_path)
    _monkeypatch_weights(monkeypatch, src)
    cache = tmp_path / "cache"

    import urllib.request

    calls = {"n": 0}
    real = urllib.request.urlretrieve

    def counting_retrieve(url, dest):
        calls["n"] += 1
        return real(url, dest)

    monkeypatch.setattr(urllib.request, "urlretrieve", counting_retrieve)

    first = fetch_trained_weights("GelSight", cache_dir=cache)
    second = fetch_trained_weights("GelSight", cache_dir=cache)
    assert first == second
    assert calls["n"] == 1  # second call served from cache


def test_fetch_checksum_mismatch_raises(tmp_path, monkeypatch):
    src = _local_weights(tmp_path)
    _monkeypatch_weights(monkeypatch, src, sha="0" * 64)  # wrong digest
    cache = tmp_path / "cache"
    with pytest.raises(ChecksumError):
        fetch_trained_weights("GelSight", cache_dir=cache)
    # partial download cleaned up, no bad file cached
    assert not (cache / "GelSight_v1.0.npz").exists()
    assert not list(cache.glob("*.part"))


def test_fetch_corrupt_cache_redownloads(tmp_path, monkeypatch):
    src = _local_weights(tmp_path)
    _monkeypatch_weights(monkeypatch, src)
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    dest = cache / "GelSight_v1.0.npz"
    dest.write_bytes(b"corrupt garbage")
    result = fetch_trained_weights("GelSight", cache_dir=cache)
    assert result == dest
    assert _sha256(dest) == _sha256(src)  # repaired


def test_fetch_unknown_sensor_returns_none(tmp_path, monkeypatch):
    _monkeypatch_weights(monkeypatch, _local_weights(tmp_path))
    assert fetch_trained_weights("DIGIT", cache_dir=tmp_path / "cache") is None


def test_encoder_cache_dir_default():
    assert encoder_cache_dir().name == "encoders"
    assert ".haptix" in str(encoder_cache_dir())


# ── load_trained auto-download ────────────────────────────────────────────


def test_load_trained_downloads_when_missing(tmp_path, monkeypatch):
    """load_trained() fetches published weights when no local copy exists."""
    src = _local_weights(tmp_path)
    _monkeypatch_weights(monkeypatch, src)
    empty_weights_dir = tmp_path / "no-local-weights"
    empty_weights_dir.mkdir()
    enc = load_trained(
        "GelSight",
        weights_dir=str(empty_weights_dir),
        cache_dir=tmp_path / "cache",
    )
    assert enc.trained is True
    assert enc.version == "encoders/gelsight/v1.0"
    assert enc.embedding_dim == 8
    assert (tmp_path / "cache" / "GelSight_v1.0.npz").is_file()


def test_load_trained_offline_raises_without_local(tmp_path, monkeypatch):
    _monkeypatch_weights(monkeypatch, _local_weights(tmp_path))
    empty_weights_dir = tmp_path / "no-local-weights"
    empty_weights_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_trained(
            "GelSight",
            weights_dir=str(empty_weights_dir),
            download=False,
            cache_dir=tmp_path / "cache",
        )


def test_load_trained_download_checksum_gate(tmp_path, monkeypatch):
    """A poisoned weight file must fail verification, not load."""
    src = _local_weights(tmp_path)
    _monkeypatch_weights(monkeypatch, src, sha="f" * 64)
    empty_weights_dir = tmp_path / "no-local-weights"
    empty_weights_dir.mkdir()
    with pytest.raises(ChecksumError):
        load_trained(
            "GelSight",
            weights_dir=str(empty_weights_dir),
            cache_dir=tmp_path / "cache",
        )


def test_load_trained_local_wins_over_download(tmp_path, monkeypatch):
    """A local weight file takes precedence over the download path."""
    src = _local_weights(tmp_path)
    _monkeypatch_weights(monkeypatch, src)
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    local = _local_weights(local_dir)  # same content, but local
    enc = load_trained("GelSight", weights_dir=str(local_dir), cache_dir=tmp_path / "cache")
    assert enc.trained is True
    # no cache download happened
    assert not (tmp_path / "cache").exists()
    assert _sha256(local) == _sha256(src)
