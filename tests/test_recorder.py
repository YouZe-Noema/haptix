"""Tests for the real-time data collection toolkit (haptix.recorder).

Covers: incremental frame recording, chunk flushing, finalization into a
valid .hapt directory (load + checksum round-trip), timestamps, shape
mismatch rejection, empty recording, double close, path-exists guard,
and streaming-archive interop.
"""

import numpy as np
import pytest

from haptix.core import InteractionMeta, Labels, SensorMeta
from haptix.io import load
from haptix.recorder import HaptRecorder
from haptix.streaming import open_archive


def _make_recorder(tmp_path, name="live.hapt", buffer_frames=8, modality="imaging", **kw):
    return HaptRecorder(
        tmp_path / name,
        sensor=SensorMeta(type="DIGIT_v2", serial="cam-001"),
        modality=modality,
        sampling_rate_hz=30.0,
        interaction=InteractionMeta(type="pressing", normal_force_N=2.0),
        labels=Labels(task="live_demo", material="rubber"),
        buffer_frames=buffer_frames,
        **kw,
    )


class TestRecord:
    def test_write_frames_incrementally(self, tmp_path):
        rec = _make_recorder(tmp_path)
        try:
            assert rec.n_frames == 0
            for i in range(20):
                frame = np.full((4, 4, 3), i, dtype=np.uint8)
                n = rec.write_frame(frame)
                assert n == i + 1
            assert rec.n_frames == 20
            assert rec.is_open
        finally:
            rec.close()

    def test_close_produces_valid_hapt(self, tmp_path):
        rec = _make_recorder(tmp_path)
        frames = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(25)]
        for f in frames:
            rec.write_frame(f)
        p = rec.close()
        data = load(p)
        assert data.raw.shape == (25, 4, 4, 3)
        assert np.array_equal(data.raw.array, np.stack(frames))
        assert data.raw.verify()
        assert data.sensor.type == "DIGIT_v2"
        assert data.labels.task == "live_demo"

    def test_context_manager(self, tmp_path):
        frames = [np.full((2, 2), i, dtype=np.float32) for i in range(10)]
        with _make_recorder(tmp_path, modality="dynamic", buffer_frames=3) as rec:
            for f in frames:
                rec.write_frame(f)
        data = load(tmp_path / "live.hapt")
        assert np.array_equal(data.raw.array, np.stack(frames))

    def test_timestamps(self, tmp_path):
        rec = _make_recorder(tmp_path, buffer_frames=4)
        for i in range(12):
            rec.write_frame(np.full((3, 3), i, dtype=np.uint8), timestamp=i / 30.0)
        p = rec.close()
        data = load(p)
        assert data.timestamps_s == [i / 30.0 for i in range(12)]

    def test_timestamps_via_tuple(self, tmp_path):
        rec = _make_recorder(tmp_path, buffer_frames=4)
        for i in range(6):
            rec.write_frame((np.full((3, 3), i, dtype=np.uint8), i * 0.1))
        data = load(rec.close())
        assert data.timestamps_s == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    def test_write_batch(self, tmp_path):
        rec = _make_recorder(tmp_path, buffer_frames=2)
        arr = np.random.RandomState(0).randint(0, 255, (16, 5, 5, 3)).astype(np.uint8)
        rec.write(arr)
        data = load(rec.close())
        assert np.array_equal(data.raw.array, arr)

    def test_shape_mismatch_rejected(self, tmp_path):
        rec = _make_recorder(tmp_path)
        try:
            rec.write_frame(np.zeros((4, 4, 3), dtype=np.uint8))
            with pytest.raises(ValueError):
                rec.write_frame(np.zeros((8, 8, 3), dtype=np.uint8))
        finally:
            rec.close()

    def test_empty_recording_raises(self, tmp_path):
        rec = _make_recorder(tmp_path)
        with pytest.raises(ValueError):
            rec.close()

    def test_double_close_idempotent(self, tmp_path):
        rec = _make_recorder(tmp_path, buffer_frames=2)
        rec.write_frame(np.zeros((2, 2), dtype=np.uint8))
        p1 = rec.close()
        p2 = rec.close()
        assert p1 == p2
        assert load(p1).raw.shape == (1, 2, 2)

    def test_write_after_close_raises(self, tmp_path):
        rec = _make_recorder(tmp_path)
        rec.write_frame(np.zeros((2, 2), dtype=np.uint8))
        rec.close()
        with pytest.raises(RuntimeError):
            rec.write_frame(np.zeros((2, 2), dtype=np.uint8))

    def test_existing_path_raises(self, tmp_path):
        (tmp_path / "taken.hapt").mkdir()
        with pytest.raises(FileExistsError):
            _make_recorder(tmp_path, name="taken.hapt")

    def test_chunks_removed_after_close(self, tmp_path):
        rec = _make_recorder(tmp_path, buffer_frames=3)
        for i in range(10):
            rec.write_frame(np.zeros((2, 2), dtype=np.uint8))
        rec.close()
        assert not (tmp_path / "live.hapt" / "raw" / "chunks").exists()

    def test_flush_writes_chunks(self, tmp_path):
        rec = _make_recorder(tmp_path, buffer_frames=4)
        for i in range(10):
            rec.write_frame(np.zeros((2, 2), dtype=np.uint8))
        rec.flush()  # explicit flush leaves remaining 2 frames in a chunk
        chunks = list((tmp_path / "live.hapt" / "raw" / "chunks").glob("*.npy"))
        assert len(chunks) >= 2
        rec.close()


class TestInterop:
    def test_recorded_hapt_opens_as_archive(self, tmp_path):
        rec = _make_recorder(tmp_path, buffer_frames=5)
        for i in range(100):
            rec.write_frame(np.full((8, 8, 3), i % 255, dtype=np.uint8))
        p = rec.close()
        with open_archive(p) as arc:
            assert arc.n_frames == 100
            wins = list(arc.iter_windows(window_size=32))
            assert len(wins) == 4  # 3 full + 1 partial(4)
            assert wins[0].raw.shape == (32, 8, 8, 3)

    def test_large_recording_many_chunks(self, tmp_path):
        rec = _make_recorder(tmp_path, buffer_frames=16)
        for i in range(1000):
            rec.write_frame(np.full((4, 4), float(i), dtype=np.float32))
        p = rec.close()
        data = load(p)
        assert data.raw.shape == (1000, 4, 4)
        assert data.raw.verify()
