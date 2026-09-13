"""
Streamlit AppTest coverage for ``haptix/browser/app.py``.

Exercises the interactive browser UI (gallery, filters, episode detail,
compare view, rescan, recursive scan, empty/missing/corrupt roots) via
``streamlit.testing.v1.AppTest``. Skips entirely when the ``browser`` extra
is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit.testing.v1")

from streamlit.testing.v1 import AppTest

import haptix.browser.app as browser_app
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

APP_PATH = Path(__file__).resolve().parents[1] / "haptix" / "browser" / "app.py"


# ── Fixture builders (mirrors tests/test_browser.py style) ─────────────────


def make_imaging_data(
    *,
    sensor: str = "GelSight",
    material: str = "sandpaper_grit_80",
    n_frames: int = 8,
    h: int = 12,
    w: int = 12,
    with_unified: bool = False,
) -> HaptData:
    """Tiny RGB imaging episode for AppTest fixtures."""
    rng = np.random.RandomState(7)
    frames = rng.randint(0, 255, (n_frames, h, w, 3)).astype(np.uint8)
    data = HaptData(
        raw=RawData(
            array=frames,
            checksum=RawData.compute_checksum(frames),
            dtype=str(frames.dtype),
            shape=frames.shape,
        ),
        sensor=SensorMeta(type=sensor, serial="TEST-001"),
        modality="imaging",
        sampling_rate_hz=30.0,
        interaction=InteractionMeta(type="sliding", speed_mm_s=50.0, normal_force_N=2.0),
        labels=Labels(material=material, task="sliding", custom_tags=["fixture", "tiny"]),
        provenance=Provenance(
            file_hash="abc123",
            derived_from="fixture",
            processing=[{"name": "normalize"}, "legacy-step"],
            is_lossy=False,
            source=Source(dataset="browser_app_tests"),
            created="2026-09-12T00:00:00",
            created_by="haptix/0.2.0",
        ),
    )
    if with_unified:
        u = rng.randn(n_frames, 4).astype(np.float32)
        data._unified = UnifiedData(
            array=u,
            method="unified/shared-force/v0.1/surrogate",
            source_modality="imaging",
            target_modality="force_latent",
            is_lossy=True,
            checksum=RawData.compute_checksum(u),
        )
    return data


def make_dynamic_data(*, n_frames: int = 10, n_channels: int = 6) -> HaptData:
    """Tiny BioTac-style dynamic episode for the signal explorer path."""
    rng = np.random.RandomState(11)
    arr = rng.randn(n_frames, n_channels).astype(np.float32)
    return HaptData(
        raw=RawData(
            array=arr,
            checksum=RawData.compute_checksum(arr),
            dtype=str(arr.dtype),
            shape=arr.shape,
        ),
        sensor=SensorMeta(type="BioTac"),
        modality="dynamic",
        sampling_rate_hz=100.0,
        interaction=InteractionMeta(type="pressing", normal_force_N=1.5),
        labels=Labels(material="foam", object_name="cube"),
        provenance=Provenance(
            file_hash="def456",
            derived_from=None,
            processing=[],
            is_lossy=False,
            source=Source(dataset="browser_app_tests"),
            created="2026-09-12T00:00:00",
            created_by="haptix/0.2.0",
        ),
    )


def write_episode(tmp: Path, name: str, data: HaptData) -> Path:
    return save(data, tmp / name)


@pytest.fixture
def gallery_root(tmp_path: Path) -> Path:
    """Directory with imaging + dynamic + nested episodes for the app tests.

    Layout (4 readable episodes when recursive=True)::

        digit.hapt          — DIGIT_v2 imaging, material=rubber, with unified
        gelsight.hapt       — GelSight imaging, material=sandpaper
        biotac.hapt         — BioTac dynamic, material=foam
        nested/nested.hapt  — DIGIT_v2 imaging, material=silk (subdir only)
    """
    write_episode(
        tmp_path,
        "digit.hapt",
        make_imaging_data(sensor="DIGIT_v2", material="rubber", with_unified=True),
    )
    write_episode(
        tmp_path,
        "gelsight.hapt",
        make_imaging_data(sensor="GelSight", material="sandpaper"),
    )
    write_episode(tmp_path, "biotac.hapt", make_dynamic_data())
    nested = tmp_path / "nested"
    nested.mkdir()
    write_episode(
        nested,
        "nested.hapt",
        make_imaging_data(sensor="DIGIT_v2", material="silk"),
    )
    return tmp_path


@pytest.fixture
def app_test(gallery_root: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """AppTest pointed at ``gallery_root`` via ``HAPTIX_BROWSER_ROOT``."""
    monkeypatch.setenv("HAPTIX_BROWSER_ROOT", str(gallery_root))
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def _episode_subheader_values(at: AppTest) -> list[str]:
    return [s.value for s in at.subheader if "episode(s)" in s.value]


def _has_title(at: AppTest, text: str) -> bool:
    return any(t.value == text for t in at.title)


def _image_nodes(at: AppTest) -> list:
    """Return ``st.image`` nodes (protobuf oneof field ``imgs``).

    Streamlit 1.45's ``AppTest`` has no ``.image`` accessor; images land as
    ``UnknownElement`` nodes whose ``type`` is ``"imgs"``. ``ElementTree.get``
    is the stable lookup.
    """
    return list(at.get("imgs"))


# ── Gallery happy path ─────────────────────────────────────────────────────


class TestGalleryRender:
    """Happy-path Episode gallery rendering."""

    def test_gallery_lists_all_episodes(self, app_test: AppTest, gallery_root: Path):
        """Sidebar root → gallery title, episode count, and one dataframe row each."""
        at = app_test
        assert _has_title(at, "Episode gallery")
        assert _episode_subheader_values(at) == ["4 episode(s)"]
        assert len(at.dataframe) == 1
        df = at.dataframe[0].value
        assert len(df) == 4
        assert "path" not in df.columns
        assert set(df["name"]) == {
            "digit.hapt",
            "gelsight.hapt",
            "biotac.hapt",
            "nested.hapt",
        }
        # default root came from env
        assert at.sidebar.text_input[0].value == str(gallery_root)


class TestEpisodeDetail:
    """Episode selectbox → metadata expanders + frame / signal / unified views."""

    def test_imaging_detail_and_no_path_column(self, app_test: AppTest):
        """Default (first) episode detail renders metadata; dataframe hides path."""
        at = app_test
        df = at.dataframe[0].value
        assert "path" not in df.columns

        labels = [e.label for e in at.expander]
        for name in ("Sensor", "Interaction", "Labels", "Provenance", "Format"):
            assert name in labels

        # Select DIGIT imaging (with unified) by name order in selectbox
        names = list(df["name"])
        digit_idx = names.index("digit.hapt")
        at.selectbox(key="episode_sel").set_value(digit_idx).run()
        assert not at.exception
        assert any(s.value == "Unified representation" for s in at.subheader)
        assert len(_image_nodes(at)) >= 1

    def test_dynamic_detail_shows_channel_multiselect(self, app_test: AppTest):
        """Selecting a dynamic episode exercises the signal explorer widgets."""
        at = app_test
        names = list(at.dataframe[0].value["name"])
        dyn_idx = names.index("biotac.hapt")
        at.selectbox(key="episode_sel").set_value(dyn_idx).run()
        assert not at.exception
        channel_ms = [m for m in at.multiselect if m.label == "Channels"]
        assert len(channel_ms) == 1
        assert channel_ms[0].value  # default channels selected
        # no imaging frame for dynamic
        assert len(_image_nodes(at)) == 0


class TestFiltering:
    """Gallery material / sensor filters."""

    def test_material_filter_reduces_episode_count(self, app_test: AppTest):
        """Restricting Material to rubber leaves a single episode."""
        at = app_test
        assert _episode_subheader_values(at) == ["4 episode(s)"]
        at.multiselect(key="f_material").set_value(["rubber"]).run()
        assert not at.exception
        assert _episode_subheader_values(at) == ["1 episode(s)"]
        assert len(at.dataframe[0].value) == 1
        assert at.dataframe[0].value.iloc[0]["name"] == "digit.hapt"


class TestComparePage:
    """Compare sensors view and the <2-episodes branch."""

    def test_compare_renders_a_and_b(self, app_test: AppTest):
        """Switching View to Compare sensors shows A/B headings without error."""
        at = app_test
        at.sidebar.radio[0].set_value("Compare sensors").run()
        assert not at.exception
        assert _has_title(at, "Compare sensors")
        heads = [m.value for m in at.markdown if m.value.startswith("### A —")]
        heads_b = [m.value for m in at.markdown if m.value.startswith("### B —")]
        assert len(heads) == 1
        assert len(heads_b) == 1

    def test_compare_needs_two_episodes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """A root with a single episode hits the informational compare message."""
        write_episode(
            tmp_path,
            "only.hapt",
            make_imaging_data(sensor="DIGIT_v2", material="rubber"),
        )
        monkeypatch.setenv("HAPTIX_BROWSER_ROOT", str(tmp_path))
        at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        at.sidebar.radio[0].set_value("Compare sensors").run()
        assert not at.exception
        assert _has_title(at, "Compare sensors")
        assert any("at least two readable episodes" in i.value for i in at.info)


class TestEmptyAndMissing:
    """Empty directory and nonexistent path handling."""

    def test_empty_directory_shows_info(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """An empty root yields the No `.hapt` recordings info path."""
        monkeypatch.setenv("HAPTIX_BROWSER_ROOT", str(tmp_path))
        at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        assert not at.exception
        assert any("No `.hapt` recordings found" in i.value for i in at.info)

    def test_missing_directory_is_graceful(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """A nonexistent sidebar path must not raise an unhandled exception."""
        missing = tmp_path / "does-not-exist"
        monkeypatch.setenv("HAPTIX_BROWSER_ROOT", str(tmp_path))
        at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        at.sidebar.text_input[0].set_value(str(missing)).run()
        assert not at.exception
        # error expander from FileNotFoundError + empty-gallery info
        assert any("unreadable file" in e.label for e in at.expander)
        assert any("No `.hapt` recordings found" in i.value for i in at.info)


class TestCorruptEpisode:
    """Corrupt `.hapt` dirs are skipped; valid siblings still render."""

    def test_corrupt_skipped_with_notice(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Malformed manifest.json → errors expander; healthy episode remains."""
        write_episode(
            tmp_path,
            "ok.hapt",
            make_imaging_data(sensor="DIGIT_v2", material="rubber"),
        )
        broken = tmp_path / "broken.hapt"
        broken.mkdir()
        (broken / "manifest.json").write_text("{not json")

        monkeypatch.setenv("HAPTIX_BROWSER_ROOT", str(tmp_path))
        at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        assert not at.exception
        assert _episode_subheader_values(at) == ["1 episode(s)"]
        assert list(at.dataframe[0].value["name"]) == ["ok.hapt"]
        assert any("unreadable file" in e.label for e in at.expander)


class TestRescanAndRecursive:
    """Rescan button and Scan subdirectories checkbox."""

    def test_rescan_button_reruns(self, app_test: AppTest):
        """Clicking Rescan clears the scan cache and reruns without error."""
        at = app_test
        at.sidebar.button[0].click().run()
        assert not at.exception
        assert _has_title(at, "Episode gallery")
        assert _episode_subheader_values(at) == ["4 episode(s)"]

    def test_recursive_checkbox_hides_nested(self, app_test: AppTest):
        """Turning off Scan subdirectories drops the nested episode."""
        at = app_test
        assert "nested.hapt" in set(at.dataframe[0].value["name"])
        at.sidebar.checkbox(key="recursive").set_value(False).run()
        assert not at.exception
        assert _episode_subheader_values(at) == ["3 episode(s)"]
        names = set(at.dataframe[0].value["name"])
        assert "nested.hapt" not in names
        assert names == {"digit.hapt", "gelsight.hapt", "biotac.hapt"}

        at.sidebar.checkbox(key="recursive").set_value(True).run()
        assert not at.exception
        assert _episode_subheader_values(at) == ["4 episode(s)"]
        assert "nested.hapt" in set(at.dataframe[0].value["name"])


class TestDefaultRoot:
    """``_default_root`` reads ``HAPTIX_BROWSER_ROOT``."""

    def test_env_sets_initial_text_input(self, gallery_root: Path, monkeypatch: pytest.MonkeyPatch):
        """Env var populates the Data directory sidebar field on first run."""
        monkeypatch.setenv("HAPTIX_BROWSER_ROOT", str(gallery_root))
        at = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        assert not at.exception
        assert at.sidebar.text_input[0].value == str(gallery_root)

    def test_fallback_to_cwd_when_env_unset(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Without env / cache, ``_default_root`` returns ``Path.cwd()``."""
        monkeypatch.delenv("HAPTIX_BROWSER_ROOT", raising=False)
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(tmp_path)
        assert browser_app._default_root() == Path.cwd()

    def test_fallback_to_cache_when_present(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """When env is unset but ``~/.haptix/cache/datasets`` exists, use it."""
        monkeypatch.delenv("HAPTIX_BROWSER_ROOT", raising=False)
        home = tmp_path / "home"
        cache = home / ".haptix" / "cache" / "datasets"
        cache.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        assert browser_app._default_root() == cache


class TestSignalExplorerEdges:
    """Signal explorer empty-selection caption."""

    def test_empty_channel_selection_shows_caption(self, app_test: AppTest):
        """Clearing Channels shows the select-at-least-one caption."""
        at = app_test
        names = list(at.dataframe[0].value["name"])
        at.selectbox(key="episode_sel").set_value(names.index("biotac.hapt")).run()
        at.multiselect(key="channels").set_value([]).run()
        assert not at.exception
        assert any("Select at least one channel" in c.value for c in at.caption)
