# Tactile Data Browser

> Interactive episode gallery for `.hapt` recordings — a Streamlit web app.
> Roadmap "Beyond" item "Tactile data browser / visualization tool"
> (decision 2026-08-10: Streamlit form factor, `haptix[browser]` extra).

The browser turns a directory of `.hapt` recordings into a browsable
collection: scan for episodes, filter by sensor / modality / material, scrub
through frames, inspect metadata / labels / provenance / unified
representations, explore dynamic signals over time, and compare two sensors
side by side.

Two layers:

| Layer | Module | Deps |
|---|---|---|
| Library (pure) | `haptix.browser` / `haptix.browser.core` | core only (numpy, Pillow) — works without streamlit |
| Web app | `haptix.browser.app` | `haptix[browser]` extra (streamlit + plotly) |

The library layer is the real API: discovery, summaries, and frame/trace
access are plain functions testable in CI and reusable from notebooks and
scripts. The Streamlit app is a thin UI over them.

---

## Install & launch

```bash
pip install "haptix[browser]"      # core + streamlit + plotly

haptix-browser                     # browse cwd (or ~/.haptix cache if present)
haptix-browser ./data              # browse a specific directory
haptix-browser ./data --server.port 8601   # extra args pass through to streamlit
```

Equivalent manual launch:

```bash
streamlit run haptix/browser/app.py        # from the repo root
python -m streamlit run haptix/browser/app.py
```

Open the printed URL (default http://localhost:8501).

**Supported containers:** `.hapt` directories, `.hapt.zip`, and `.hapt.zarr`
(any combination in one tree). The browser scans metadata only — raw arrays
are loaded lazily, one frame at a time. For very long recordings prefer
directory / zarr format: `.hapt.zip` decompresses the whole array when the
archive is opened (see [`docs/api.md`](api.md#streaming--windowing-module-haptixstreaming)).

**No data handy?** Generate a demo gallery:

```bash
python examples/end_to_end_demo.py          # writes demo .hapt files
haptix-browser ./outputs                    # (or wherever the demo wrote them)
```

---

## Using the app

### Episode gallery

- **Sidebar** — set the data directory and toggle recursive scanning.
  `Rescan` re-reads the tree (use it after adding files).
- **Filters** — narrow the table by sensor, modality, and material.
- **Table** — name, sensor, modality, frame count, shape, sampling rate,
  material / object / task labels, unified presence, size, container format.
- **Episode detail** (select a row) —
  - *metadata*: sensor, sampling, shape, interaction parameters, labels,
    **provenance** (content-addressable `file_hash`, derivation chain,
    processing steps, source, created-by), and container info;
  - *frame scrubber*: a slider walks frames one at a time —
    imaging sensors render the tactile frame as an image, dynamic sensors
    plot the frame's channel values;
  - *signal explorer* (dynamic recordings): full-length channel traces over
    time, selectable channels;
  - *unified view*: when the episode carries a `unified/` cross-sensor
    embedding, a heatmap of embedding dims over time plus the method tag.

### Compare sensors

Pick two episodes (any sensors) and scrub them side by side — handy for
visually comparing how different tactile modalities respond to the same kind
of interaction, or for sanity-checking that two recordings of the same object
look consistent.

---

## Library API

The same functionality without the web UI:

```python
import haptix

# Discover + summarize (metadata only, lazy)
haptix.find_hapt_files("./data")          # -> [Path, ...]  (.hapt, .hapt.zip, .hapt.zarr)
scan = haptix.scan_directory("./data")    # -> {"root", "episodes": [...], "errors": [...]}
ep = haptix.episode_summary("./data/press.hapt")   # one flat metadata dict
df = haptix.make_gallery_dataframe(scan)  # pandas table for display/filtering

# Frames (lazy per-frame reads)
frame = haptix.frame_array("./data/press.hapt", 12)     # np.ndarray [H, W, C] or [F]
img = haptix.frame_image("./data/press.hapt", 12)       # PIL.Image (imaging)
sig = haptix.frame_signals("./data/slide.hapt", 12)     # 1-D channel values (dynamic)

# Full traces
tr = haptix.signal_trace("./data/slide.hapt", channels=[0, 1, 2])  # {"t", "y", "channels"}
u = haptix.unified_trace("./data/press.hapt")   # [T, D] embedding or None
```

All functions accept a `str` / `Path` / already-open `HaptArchive`; paths are
opened lazily and closed automatically. `scan_directory` is tolerant — a
corrupt recording lands in `errors` instead of aborting the scan.

Full reference: [`docs/api.md`](api.md#browser-module-haptixbrowser).

---

## Notes & troubleshooting

- **Corrupt files** show up as a collapsible warning list in the gallery;
  healthy episodes still load.
- **Float / unusual-dtype frames** are normalized per-frame to uint8 for
  display; raw data is never modified.
- **`.hapt.zip` memory** — zip archives decompress in full on open (stdlib
  DEFLATE limitation). Prefer directory / zarr for large recordings.
- **Streamlit cache** — scan results and rendered frames are cached per
  session; use `Rescan` after changing files on disk.
- The browser adds **no runtime dependency to core haptix** — the library
  functions work with `pip install haptix` alone; only the web app needs the
  `[browser]` extra.
