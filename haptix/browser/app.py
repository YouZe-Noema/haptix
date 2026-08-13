"""
Streamlit app — the haptix tactile data browser.

Interactive episode gallery for ``.hapt`` recordings: browse a directory of
episodes, scrub frames, inspect metadata / labels / provenance / unified
representations, explore dynamic signals, and compare sensors side by side.

Launch (any of)::

    haptix-browser [DATA_DIR]
    python -m streamlit run haptix/browser/app.py
    streamlit run haptix/browser/app.py

The app is a thin UI layer over :mod:`haptix.browser.core` — all data access
goes through the pure library functions there (lazy archives, no full-array
materialization except for the frames the user actually looks at).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from haptix.browser import (
    frame_image,
    frame_signals,
    make_gallery_dataframe,
    scan_directory,
    signal_trace,
    unified_trace,
)

st.set_page_config(
    page_title="haptix — tactile data browser",
    page_icon="🤖",
    layout="wide",
)


# ── Session helpers ───────────────────────────────────────────────────────


def _default_root() -> Path:
    """Default browse directory: env var, haptix cache, or cwd."""
    env = os.environ.get("HAPTIX_BROWSER_ROOT")
    if env:
        return Path(env)
    cache = Path.home() / ".haptix" / "cache" / "datasets"
    if cache.exists():
        return cache
    return Path.cwd()


@st.cache_data(show_spinner="Scanning for .hapt recordings…")
def _scan(root: str, recursive: bool) -> dict:
    """Cached scan_directory result (JSON-serializable summaries)."""
    return scan_directory(root, recursive=recursive)


@st.cache_data(show_spinner=False)
def _frame_img(path: str, frame: int, max_size: int):
    """Cached PIL frame render (hashable args, picklable result)."""
    return frame_image(path, frame, max_size=max_size)


def _fmt(x) -> str:
    if x is None or x == "":
        return "—"
    if isinstance(x, float):
        return f"{x:g}"
    return str(x)


# ── Rendering helpers ─────────────────────────────────────────────────────


def _render_metadata(ep: dict) -> None:
    """Collapsible metadata blocks for one episode."""
    with st.expander("Sensor", expanded=False):
        st.markdown(
            f"**type:** {ep['sensor']}  ·  **modality:** {ep['modality']}  ·  "
            f"**serial:** {_fmt(ep['serial'])}"
        )
        st.markdown(
            f"**sampling:** {_fmt(ep['sampling_rate_hz'])} Hz  ·  "
            f"**duration:** {_fmt(ep['duration_s'])} s  ·  "
            f"**timestamps:** {'yes' if ep['timestamps'] else 'no'}"
        )
        st.markdown(
            f"**shape:** `{ep['shape_str']}`  ·  **dtype:** {ep['dtype']}  ·  "
            f"**version:** {ep['version']}"
        )
        st.markdown(f"**coordinate frame:** {_fmt(ep['coordinate_frame'])}")

    with st.expander("Interaction", expanded=False):
        st.markdown(f"**type:** {_fmt(ep['interaction_type'])}")
        st.markdown(
            f"**speed:** {_fmt(ep['speed_mm_s'])} mm/s  ·  "
            f"**force:** {_fmt(ep['normal_force_N'])} N  ·  "
            f"**angle:** {_fmt(ep['approach_angle_deg'])}°"
        )
        st.markdown(
            f"**temperature:** {_fmt(ep['temperature_C'])} °C  ·  "
            f"**humidity:** {_fmt(ep['humidity_pct'])} %"
        )

    with st.expander("Labels", expanded=False):
        st.markdown(
            f"**material:** {_fmt(ep['material'])}  ·  "
            f"**category:** {_fmt(ep['material_category'])}"
        )
        st.markdown(
            f"**object:** {_fmt(ep['object_name'])}  ·  "
            f"**object category:** {_fmt(ep['object_category'])}"
        )
        st.markdown(f"**task:** {_fmt(ep['task'])}")
        if ep["custom_tags"]:
            st.markdown(f"**tags:** {', '.join(str(t) for t in ep['custom_tags'])}")

    with st.expander("Provenance", expanded=False):
        st.markdown(f"**file hash:** `{_fmt(ep['file_hash'])}`")
        st.markdown(f"**derived from:** {_fmt(ep['derived_from'])}")
        st.markdown(
            f"**lossy:** {_fmt(ep['is_lossy'])}  ·  "
            f"**created:** {_fmt(ep['created'])}  ·  "
            f"**created by:** {_fmt(ep['created_by'])}"
        )
        src = ep["source"]
        if src:
            st.markdown("**source:** " + "  ·  ".join(f"{k}={_fmt(v)}" for k, v in src.items()))
        if ep["processing"]:
            steps = []
            for s in ep["processing"]:
                if isinstance(s, dict):
                    steps.append(s.get("name", str(s)))
                else:
                    steps.append(str(s))
            st.markdown(f"**processing:** {' → '.join(steps)}")

    with st.expander("Format", expanded=False):
        st.markdown(f"**container:** {ep['format']}  ·  **size:** {ep['size_bytes'] / 1e6:.2f} MB")
        if ep["unified"]:
            st.markdown(
                f"**unified:** yes (`{ep['unified_method']}`, shape " f"`{ep['unified_shape']}`)"
            )
        else:
            st.markdown("**unified:** no")


def _frame_view(ep: dict, key: str = "") -> None:
    """Frame scrubber + renderer (image for imaging, signal bar for dynamic)."""
    n = ep["n_frames"]
    if n <= 0:
        st.info("Empty recording (no frames).")
        return
    frame = st.slider("Frame", 0, n - 1, 0, key=f"{key}frame_slider")
    if ep["modality"] == "imaging":
        img = _frame_img(ep["path"], int(frame), max_size=480)
        st.image(img, caption=f"frame {frame} / {n - 1}", width=420)
    else:
        sig = frame_signals(ep["path"], int(frame))
        fig = go.Figure(data=[go.Scatter(y=sig.tolist(), mode="lines+markers", name="channels")])
        fig.update_layout(
            title=f"frame {frame} / {n - 1} — channel values",
            xaxis_title="channel",
            yaxis_title="value",
            height=300,
            margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)


def _signal_explorer(ep: dict, key: str = "") -> None:
    """Dynamic recordings: full-length channel traces over time."""
    trace = signal_trace(ep["path"], max_frames=5000)
    nch = trace["y"].shape[1]
    if nch == 0:
        return
    default = list(range(min(nch, 8)))
    channels = st.multiselect(
        "Channels",
        list(range(nch)),
        default=default,
        key=f"{key}channels",
        format_func=lambda c: f"ch{c}",
    )
    if not channels:
        st.caption("Select at least one channel.")
        return
    fig = go.Figure()
    for c in channels:
        fig.add_trace(go.Scatter(x=trace["t"], y=trace["y"][:, c], name=f"ch{c}", mode="lines"))
    fig.update_layout(
        title="Signal over time",
        xaxis_title="time (s)" if ep["timestamps"] else "frame",
        yaxis_title="value",
        height=320,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def _unified_view(ep: dict, key: str = "") -> None:
    """Unified cross-sensor embedding: shape/method + heatmap over time."""
    if not ep["unified"]:
        return
    st.subheader("Unified representation")
    st.caption(
        f"method `{ep['unified_method']}` · shape `{ep['unified_shape']}` · "
        "cross-sensor embedding over time"
    )
    u = unified_trace(ep["path"], max_frames=500)
    if u is None or u.shape[0] == 0:
        st.info("Unified array not readable.")
        return
    t = np.arange(u.shape[0])
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=u.T,
                x=t,
                colorscale="Viridis",
                colorbar=dict(title="value"),
            )
        ]
    )
    fig.update_layout(
        title="Unified embedding (dims × time)",
        xaxis_title="frame",
        yaxis_title="embedding dim",
        height=340,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_episode(ep: dict, key: str = "") -> None:
    """Full detail view for one episode (metadata + scrubber + explorers)."""
    left, right = st.columns([1, 2])
    with left:
        _render_metadata(ep)
    with right:
        _frame_view(ep, key=key)
    if ep["modality"] != "imaging":
        _signal_explorer(ep, key=key)
    _unified_view(ep, key=key)


# ── Pages ─────────────────────────────────────────────────────────────────


def page_gallery() -> None:
    st.title("Episode gallery")

    scan = _scan(str(st.session_state["root"]), st.session_state["recursive"])
    episodes = scan["episodes"]

    if scan["errors"]:
        with st.expander(f"⚠️ {len(scan['errors'])} unreadable file(s) skipped"):
            for err in scan["errors"]:
                st.markdown(f"**{err['path']}** — {err['error']}")

    if not episodes:
        st.info(
            f"No `.hapt` recordings found under `{scan['root']}`. Point the "
            "sidebar at a directory containing `.hapt` / `.hapt.zip` / "
            "`.hapt.zarr` files, or generate one with "
            "`python examples/end_to_end_demo.py`."
        )
        return

    # Filters
    sensors = sorted({e["sensor"] for e in episodes})
    modalities = sorted({e["modality"] for e in episodes})
    materials = sorted({e["material"] for e in episodes if e["material"]})
    f_sensor = st.multiselect("Sensor", sensors, default=sensors, key="f_sensor")
    f_modality = st.multiselect("Modality", modalities, default=modalities, key="f_modality")
    f_material = st.multiselect("Material", materials, default=[], key="f_material")
    filtered = [
        e
        for e in episodes
        if e["sensor"] in f_sensor
        and e["modality"] in f_modality
        and (not f_material or e["material"] in f_material)
    ]

    st.subheader(f"{len(filtered)} episode(s)")
    df = make_gallery_dataframe({"episodes": filtered})
    st.dataframe(
        df.drop(columns=["path"]),
        use_container_width=True,
        hide_index=True,
    )

    if filtered:
        names = [e["name"] for e in filtered]
        idx = st.selectbox(
            "Episode",
            range(len(filtered)),
            format_func=lambda i: f"{names[i]} — {filtered[i]['sensor']} · "
            f"{filtered[i]['n_frames']} frames · {filtered[i]['material'] or '—'}",
            key="episode_sel",
        )
        st.divider()
        _render_episode(filtered[idx])


def page_compare() -> None:
    st.title("Compare sensors")

    scan = _scan(str(st.session_state["root"]), st.session_state["recursive"])
    episodes = scan["episodes"]
    if len(episodes) < 2:
        st.info("Need at least two readable episodes to compare.")
        return

    paths = [e["path"] for e in episodes]
    names = [e["name"] for e in episodes]
    by_path = {e["path"]: e for e in episodes}

    a = st.selectbox(
        "Episode A",
        paths,
        index=0,
        format_func=lambda p: names[paths.index(p)],
        key="cmp_a",
    )
    b = st.selectbox(
        "Episode B",
        paths,
        index=min(1, len(paths) - 1),
        format_func=lambda p: names[paths.index(p)],
        key="cmp_b",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"### A — {by_path[a]['name']}")
        _render_episode(by_path[a], key="a_")
    with col_b:
        st.markdown(f"### B — {by_path[b]['name']}")
        _render_episode(by_path[b], key="b_")


# ── App ───────────────────────────────────────────────────────────────────


def main() -> None:
    st.sidebar.title("haptix browser")

    root_input = st.sidebar.text_input(
        "Data directory",
        value=str(_default_root()),
        key="root_input",
    )
    st.session_state["root"] = root_input
    # key="recursive" owns its session_state entry — do NOT assign to it here
    st.sidebar.checkbox("Scan subdirectories", value=True, key="recursive")
    if st.sidebar.button("Rescan", use_container_width=True):
        _scan.clear()
        st.rerun()
    st.sidebar.caption("Launch with `haptix-browser [DATA_DIR]` — see `docs/browser.md`.")

    page = st.sidebar.radio("View", ["Episode gallery", "Compare sensors"])
    if page == "Compare sensors":
        page_compare()
    else:
        page_gallery()


main()
