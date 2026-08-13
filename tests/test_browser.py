"""
Tests for the tactile data browser (haptix.browser).

Covers the pure library layer (haptix.browser.core) — discovery, episode
summaries, frame/trace access — which must work with core deps only (no
streamlit/plotly). The Streamlit app itself (haptix/browser/app.py) is
syntax-checked via py_compile; its logic is a thin layer over these functions.
"""

import py_compile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import haptix
from haptix.browser import (
    episode_summary,
    find_hapt_files,
    frame_array,
    frame_image,
    frame_signals,
    make_gallery_dataframe,
    scan_directory,
    signal_trace,
    unified_trace,
)
from haptix.core import (
    HaptData,
    InteractionMeta,
    Labels,
    Provenance,
    RawData,
    SensorMeta,
    Source,
    UnifiedData,
)
from haptix.io import save


def make_imaging_data(
    n_frames: int = 10,
    h: int = 24,
    w: int = 32,
    *,
    dtype=np.uint8,
    with_unified: bool = False,
    with_provenance: bool = True,
) -> HaptData:
    rng = np.random.RandomState(7)
    frames = rng.randint(0, 255, (n_frames, h, w, 3)).astype(dtype)
    data = HaptData(
        raw=RawData(
            array=frames,
            checksum=RawData.compute_checksum(frames),
            dtype=str(frames.dtype),
            shape=frames.shape,
        ),
        sensor=SensorMeta(type="GelSight", serial="GS-001"),
        modality="imaging",
        sampling_rate_hz=30.0,
        interaction=InteractionMeta(type="sliding", speed_mm_s=50.0, normal_force_N=2.0),
        labels=Labels(material="sandpaper_grit_80", task="sliding"),
        provenance=(
            Provenance(
                file_hash="abc123",
                derived_from="ycb-sight/002_master_chef_can",
                processing=[],
                is_lossy=False,
                source=Source(dataset="ycb_sight", license="CC BY-NC 4.0"),
                created="2026-08-13T00:00:00",
                created_by="haptix/0.2.0",
            )
            if with_provenance
            else None
        ),
    )
    if with_unified:
        u = rng.randn(n_frames, 8).astype(np.float32)
        data._unified = UnifiedData(
            array=u,
            method="unified/shared-force/v0.1/surrogate",
            source_modality="imaging",
            target_modality="force_latent",
            is_lossy=True,
            checksum=RawData.compute_checksum(u),
        )
    return data


def make_dynamic_data(n_frames: int = 20, n_channels: int = 12) -> HaptData:
    rng = np.random.RandomState(11)
    arr = rng.randn(n_frames, n_channels).astype(np.float32)
    return HaptData(
        raw=RawData(
            array=arr,
            checksum=RawData.compute_checksum(arr),
            dtype=str(arr.dtype),
            shape=arr.shape,
        ),
        sensor=SensorMeta(type="CoroCapacitive"),
        modality="dynamic",
        sampling_rate_hz=100.0,
        interaction=InteractionMeta(type="pressing", normal_force_N=1.5),
        labels=Labels(material="foam", object_name="cube"),
        provenance=Provenance(
            file_hash="def456",
            derived_from=None,
            processing=[],
            is_lossy=False,
            source=Source(dataset="coro_tactile"),
            created="2026-08-13T00:00:00",
            created_by="haptix/0.2.0",
        ),
    )


def write_episode(tmp: Path, name: str, data: HaptData) -> Path:
    return save(data, tmp / name)


@pytest.fixture
def gallery(tmp_path: Path):
    """A directory with imaging + dynamic episodes (all three formats)."""
    img = make_imaging_data(with_unified=True)
    dyn = make_dynamic_data()
    p_img = write_episode(tmp_path, "press.hapt", img)
    p_dyn = write_episode(tmp_path, "slide.hapt", dyn)
    p_zip = write_episode(tmp_path, "archive.hapt.zip", img)
    p_zarr = write_episode(tmp_path, "archive.hapt.zarr", dyn)
    # decoys that must NOT be discovered
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / "plain.zip").write_bytes(b"PK\x03\x04 not a hapt archive")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.hapt").mkdir()
    (tmp_path / ".hidden" / "secret.hapt" / "manifest.json").write_text("{}")
    return tmp_path, p_img, p_dyn, p_zip, p_zarr


# ── Discovery ─────────────────────────────────────────────────────────────


def test_find_hapt_files_all_formats(gallery):
    tmp, p_img, p_dyn, p_zip, p_zarr = gallery
    hits = find_hapt_files(tmp)
    expected = {str(p) for p in (p_img, p_dyn, p_zip, p_zarr)}
    assert {str(h) for h in hits} == expected
    assert all(isinstance(h, Path) for h in hits)
    # sorted by name
    assert hits == sorted(hits, key=lambda p: p.name)


def test_find_hapt_files_not_recursive(gallery):
    tmp, *_ = gallery
    nested = tmp / "nested"
    nested.mkdir()
    write_episode(nested, "deep.hapt", make_imaging_data())
    assert len(find_hapt_files(tmp, recursive=False)) == 4  # top-level only
    assert len(find_hapt_files(tmp, recursive=True)) == 5


def test_find_hapt_files_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_hapt_files(tmp_path / "does-not-exist")


def test_find_hapt_files_skips_hidden_dirs(gallery):
    tmp, *_ = gallery
    hits = find_hapt_files(tmp)
    assert not any(".hidden" in str(h) for h in hits)


def test_find_hapt_files_accepts_pathlib_and_str(gallery):
    tmp, *_ = gallery
    assert find_hapt_files(str(tmp)) == find_hapt_files(tmp)


# ── Episode summaries ─────────────────────────────────────────────────────


def test_episode_summary_imaging(gallery):
    tmp, p_img, *_ = gallery
    ep = episode_summary(p_img)
    assert ep["name"] == "press.hapt"
    assert ep["sensor"] == "GelSight"
    assert ep["serial"] == "GS-001"
    assert ep["modality"] == "imaging"
    assert ep["n_frames"] == 10
    assert ep["shape"] == (10, 24, 32, 3)
    assert ep["shape_str"] == "10×24×32×3"
    assert ep["dtype"] == "uint8"
    assert ep["sampling_rate_hz"] == 30.0
    assert ep["material"] == "sandpaper_grit_80"
    assert ep["task"] == "sliding"
    assert ep["interaction_type"] == "sliding"
    assert ep["normal_force_N"] == 2.0
    assert ep["file_hash"] == "abc123"
    assert ep["derived_from"] == "ycb-sight/002_master_chef_can"
    assert ep["is_lossy"] is False
    assert ep["created_by"] == "haptix/0.2.0"
    assert ep["source"] == {"dataset": "ycb_sight", "license": "CC BY-NC 4.0"}
    assert ep["unified"] is True
    assert ep["unified_method"] == "unified/shared-force/v0.1/surrogate"
    assert ep["unified_shape"] == (10, 8)
    assert ep["format"] == "dir"
    assert ep["size_bytes"] > 0
    # JSON-compatible (safe for st.cache_data / pandas)
    import json

    json.dumps({k: v for k, v in ep.items() if k not in ("shape", "unified_shape")})


def test_episode_summary_dynamic_and_zip_zarr(gallery):
    tmp, p_img, p_dyn, p_zip, p_zarr = gallery
    ep = episode_summary(p_dyn)
    assert ep["modality"] == "dynamic"
    assert ep["shape"] == (20, 12)
    assert ep["object_name"] == "cube"
    assert ep["sensor"] == "CoroCapacitive"

    epz = episode_summary(p_zip)
    assert epz["format"] == "zip"
    assert epz["n_frames"] == 10

    pytest.importorskip("zarr")
    epzr = episode_summary(p_zarr)
    assert epzr["format"] == "zarr"
    assert epzr["modality"] == "dynamic"


def test_episode_summary_no_provenance_no_unified(tmp_path):
    # save() always writes provenance.json; without explicit provenance it is
    # the auto-generated default (file_hash="" until a content hash is set).
    data = make_imaging_data(with_provenance=False, with_unified=False)
    p = write_episode(tmp_path, "bare.hapt", data)
    ep = episode_summary(p)
    assert ep["file_hash"] == ""
    assert ep["source"] == {}
    assert ep["unified"] is False
    assert ep["unified_shape"] is None


def test_episode_summary_timestamps(tmp_path):
    data = make_dynamic_data()
    data._timestamps_s = [i * 0.01 for i in range(data.raw.shape[0])]
    p = write_episode(tmp_path, "ts.hapt", data)
    ep = episode_summary(p)
    assert ep["timestamps"] is True
    assert ep["duration_s"] == pytest.approx(0.19)


# ── scan_directory ────────────────────────────────────────────────────────


def test_scan_directory_valid(gallery):
    tmp, *_ = gallery
    result = scan_directory(tmp)
    assert result["root"] == str(tmp)
    assert len(result["episodes"]) == 4
    assert result["errors"] == []


def test_scan_directory_tolerates_corrupt(gallery):
    tmp, *_ = gallery
    bad = tmp / "broken.hapt"
    bad.mkdir()
    (bad / "manifest.json").write_text("{not json")
    result = scan_directory(tmp)
    names = {e["name"] for e in result["episodes"]}
    assert "broken.hapt" not in names
    assert len(result["errors"]) == 1
    assert result["errors"][0]["path"].endswith("broken.hapt")
    assert "error" in result["errors"][0]


def test_scan_directory_max_episodes(gallery):
    tmp, *_ = gallery
    result = scan_directory(tmp, max_episodes=2)
    assert len(result["episodes"]) == 2


def test_scan_directory_verify_ok(gallery):
    tmp, *_ = gallery
    result = scan_directory(tmp, verify=True)
    assert len(result["episodes"]) == 4
    assert result["errors"] == []


# ── Gallery dataframe ─────────────────────────────────────────────────────


def test_make_gallery_dataframe(gallery):
    tmp, *_ = gallery
    df = make_gallery_dataframe(scan_directory(tmp))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    for col in (
        "name",
        "sensor",
        "modality",
        "frames",
        "shape",
        "material",
        "object",
        "task",
        "unified",
        "size_mb",
        "format",
        "path",
    ):
        assert col in df.columns
    assert set(df["sensor"]) == {"GelSight", "CoroCapacitive"}


def test_make_gallery_dataframe_empty():
    df = make_gallery_dataframe({"episodes": []})
    assert df.empty
    assert "name" in df.columns


# ── Frame access ──────────────────────────────────────────────────────────


def test_frame_array_imaging(gallery):
    tmp, p_img, *_ = gallery
    frame = frame_array(p_img, 3)
    assert frame.shape == (24, 32, 3)
    assert frame.dtype == np.uint8
    # matches the saved data
    data = haptix.load(p_img)
    assert np.array_equal(frame, data.raw.numpy()[3])


def test_frame_array_dynamic(gallery):
    tmp, _, p_dyn, *_ = gallery
    frame = frame_array(p_dyn, 5)
    assert frame.shape == (12,)
    data = haptix.load(p_dyn)
    assert np.array_equal(frame, data.raw.numpy()[5])


def test_frame_array_out_of_range(gallery):
    tmp, p_img, *_ = gallery
    with pytest.raises(IndexError):
        frame_array(p_img, 10)
    with pytest.raises(IndexError):
        frame_array(p_img, -1)


def test_frame_array_accepts_archive_handle(gallery):
    tmp, p_img, *_ = gallery
    import haptix

    with haptix.open_archive(p_img) as arc:
        frame = frame_array(arc, 0)
    assert frame.shape == (24, 32, 3)


def test_frame_signals(gallery):
    tmp, _, p_dyn, *_ = gallery
    sig = frame_signals(p_dyn, 0)
    assert sig.shape == (12,)
    assert sig.ndim == 1


# ── Image rendering ───────────────────────────────────────────────────────


def test_frame_image_uint8(gallery):
    tmp, p_img, *_ = gallery
    img = frame_image(p_img, 2)
    assert img.mode == "RGB"
    assert img.size == (32, 24)  # (W, H)
    # max_size downscales, preserving aspect
    small = frame_image(p_img, 2, max_size=16)
    assert max(small.size) <= 16


def test_frame_image_grayscale(tmp_path):
    arr = np.random.RandomState(3).randint(0, 255, (5, 16, 16)).astype(np.uint8)
    data = HaptData(
        raw=RawData(
            array=arr,
            checksum=RawData.compute_checksum(arr),
            dtype="uint8",
            shape=arr.shape,
        ),
        sensor=SensorMeta(type="GelSight"),
        modality="imaging",
        sampling_rate_hz=30.0,
        interaction=InteractionMeta(type="static"),
        labels=Labels(),
    )
    p = write_episode(tmp_path, "gray.hapt", data)
    img = frame_image(p, 0)
    assert img.mode == "L"
    assert img.size == (16, 16)


def test_frame_image_float_normalization(tmp_path):
    rng = np.random.RandomState(4)
    arr = rng.rand(6, 20, 20, 3).astype(np.float32) * 0.5  # [0, 0.5]
    data = HaptData(
        raw=RawData(
            array=arr,
            checksum=RawData.compute_checksum(arr),
            dtype=str(arr.dtype),
            shape=arr.shape,
        ),
        sensor=SensorMeta(type="GelSight"),
        modality="imaging",
        sampling_rate_hz=30.0,
        interaction=InteractionMeta(type="static"),
        labels=Labels(),
    )
    p = write_episode(tmp_path, "float.hapt", data)
    img = frame_image(p, 1)
    assert img.mode == "RGB"
    arr_u8 = np.asarray(img, dtype=np.uint8)
    assert arr_u8.min() == 0
    assert arr_u8.max() == 255


def test_frame_image_rejects_dynamic(gallery):
    tmp, _, p_dyn, *_ = gallery
    with pytest.raises(ValueError):
        frame_image(p_dyn, 0)


# ── Traces ────────────────────────────────────────────────────────────────


def test_signal_trace_all_channels(gallery):
    tmp, _, p_dyn, *_ = gallery
    trace = signal_trace(p_dyn)
    assert trace["t"].shape == (20,)
    assert trace["y"].shape == (20, 12)
    assert trace["channels"] == list(range(12))
    # t defaults to frame indices when no timestamps
    assert np.array_equal(trace["t"], np.arange(20))


def test_signal_trace_selected_channels(gallery):
    tmp, _, p_dyn, *_ = gallery
    trace = signal_trace(p_dyn, channels=[2, 5])
    assert trace["y"].shape == (20, 2)
    assert trace["channels"] == [2, 5]


def test_signal_trace_max_frames(gallery):
    tmp, _, p_dyn, *_ = gallery
    trace = signal_trace(p_dyn, max_frames=7)
    assert trace["t"].shape == (7,)
    assert trace["y"].shape == (7, 12)


def test_unified_trace_present_and_absent(gallery):
    tmp, p_img, p_dyn, *_ = gallery
    u = unified_trace(p_img)
    assert u is not None
    assert u.shape == (10, 8)
    assert unified_trace(p_dyn) is None


def test_unified_trace_max_frames(gallery):
    tmp, p_img, *_ = gallery
    u = unified_trace(p_img, max_frames=4)
    assert u.shape == (4, 8)


# ── App / entry point integrity ───────────────────────────────────────────


def test_app_py_compiles():
    app = Path(haptix.__file__).resolve().parent / "browser" / "app.py"
    assert app.is_file()
    py_compile.compile(str(app), doraise=True)


def test_cli_module_has_main():
    from haptix.browser import cli

    assert callable(cli.main)
    assert cli.APP_FILE.name == "app.py"


def test_browser_console_script_declared():
    """haptix-browser entry point must exist in pyproject [project.scripts]."""
    import tomllib

    pyproject = Path(haptix.__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    scripts = data.get("project", {}).get("scripts", {})
    assert scripts.get("haptix-browser") == "haptix.browser.cli:main"


def test_top_level_exports_work_without_streamlit(gallery):
    """The pure browser API is available at the haptix top level and does not
    require the streamlit extra."""
    import haptix as hx

    assert hx.find_hapt_files is find_hapt_files
    assert hx.scan_directory is scan_directory
    assert hx.episode_summary is episode_summary
    assert hx.frame_image is frame_image
    assert hx.signal_trace is signal_trace
    assert hx.unified_trace is unified_trace
    assert "scan_directory" in hx.__all__


def test_browser_core_imports_without_streamlit():
    """Importing the browser core must not pull in streamlit/plotly."""
    import sys

    import haptix.browser.core as core

    assert "streamlit" not in sys.modules
    assert "plotly" not in sys.modules
    assert core.frame_image is not None
