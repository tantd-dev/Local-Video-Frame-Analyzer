"""
Core logic tests -- verifies frame parsing, batch creation, JSON parsing,
and result management without needing an AI provider or GUI.
"""
import os
import sys
import io
import json
import tempfile
import shutil

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.timestamp import parse_frame_filename, format_duration, is_image_file
from utils.json_utils import extract_json_from_text, safe_parse_json
from core.frame_loader import discover_frames, get_frame_stats
from core.batch_manager import create_batches, get_batch_summary
from core.result_manager import (
    create_run_directory, save_batch_result, load_batch_result,
    get_batch_statuses, save_config,
)


def test_parse_frame_filename():
    """TEST: Frame filename parsing."""
    assert parse_frame_filename("frame_0001_000000.jpg") == (1, 0)
    assert parse_frame_filename("frame_0025_000215.jpg") == (25, 215)
    assert parse_frame_filename("frame_0098_000890.jpg") == (98, 890)
    assert parse_frame_filename("frame_0001_000000.png") == (1, 0)
    assert parse_frame_filename("frame_0001_000000.webp") == (1, 0)
    assert parse_frame_filename("not_a_frame.jpg") is None
    assert parse_frame_filename("random.txt") is None
    print("  ✓ parse_frame_filename")


def test_format_duration():
    """TEST: Duration formatting."""
    assert format_duration(0) == "0:00"
    assert format_duration(65) == "1:05"
    assert format_duration(907) == "15:07"
    assert format_duration(3661) == "1:01:01"
    print("  ✓ format_duration")


def test_is_image_file():
    """TEST: Image file detection."""
    assert is_image_file("test.jpg") is True
    assert is_image_file("test.JPEG") is True
    assert is_image_file("test.png") is True
    assert is_image_file("test.webp") is True
    assert is_image_file("test.txt") is False
    assert is_image_file("test.mp4") is False
    print("  ✓ is_image_file")


def test_batch_creation_98_frames():
    """TEST 1: 98 frames, batch size 10 → 10 batches (9×10 + 1×8)."""
    from core.frame_loader import FrameInfo
    frames = [
        FrameInfo(f"frame_{i:04d}_{i*9:06d}.jpg", f"/fake/{i}.jpg", i, i * 9)
        for i in range(1, 99)
    ]
    batches = create_batches(frames, 10)
    assert len(batches) == 10
    for b in batches[:9]:
        assert len(b.frames) == 10
    assert len(batches[9].frames) == 8

    summary = get_batch_summary(98, 10)
    assert summary['total_batches'] == 10
    assert summary['distribution'] == [
        {'count': 9, 'frames': 10},
        {'count': 1, 'frames': 8},
    ]
    print("  ✓ TEST 1: 98 frames / batch 10 → 9×10 + 1×8")


def test_batch_creation_100_frames():
    """TEST 2: 100 frames, batch size 10 → 10 batches (10×10)."""
    from core.frame_loader import FrameInfo
    frames = [
        FrameInfo(f"frame_{i:04d}_{i*9:06d}.jpg", f"/fake/{i}.jpg", i, i * 9)
        for i in range(1, 101)
    ]
    batches = create_batches(frames, 10)
    assert len(batches) == 10
    for b in batches:
        assert len(b.frames) == 10

    summary = get_batch_summary(100, 10)
    assert summary['total_batches'] == 10
    assert summary['distribution'] == [{'count': 10, 'frames': 10}]
    print("  ✓ TEST 2: 100 frames / batch 10 → 10×10")


def test_batch_creation_7_frames():
    """TEST 3: 7 frames, batch size 10 → 1 batch (1×7)."""
    from core.frame_loader import FrameInfo
    frames = [
        FrameInfo(f"frame_{i:04d}_{i*9:06d}.jpg", f"/fake/{i}.jpg", i, i * 9)
        for i in range(1, 8)
    ]
    batches = create_batches(frames, 10)
    assert len(batches) == 1
    assert len(batches[0].frames) == 7

    summary = get_batch_summary(7, 10)
    assert summary['total_batches'] == 1
    assert summary['distribution'] == [{'count': 1, 'frames': 7}]
    print("  ✓ TEST 3: 7 frames / batch 10 → 1×7")


def test_json_extraction():
    """TEST: JSON extraction from various AI response formats."""
    # Pure JSON
    j1 = '{"summary": "test"}'
    assert safe_parse_json(j1) == ({"summary": "test"}, None)

    # Markdown-fenced
    j2 = '```json\n{"summary": "test"}\n```'
    parsed, err = safe_parse_json(j2)
    assert parsed == {"summary": "test"}

    # With surrounding text
    j3 = 'Here is the result:\n{"summary": "test"}\nDone.'
    parsed, err = safe_parse_json(j3)
    assert parsed == {"summary": "test"}

    # Invalid
    j4 = 'This is not JSON at all'
    parsed, err = safe_parse_json(j4)
    assert parsed is None
    assert err is not None

    print("  ✓ JSON extraction & parsing")


def test_frame_discovery():
    """TEST: Frame discovery and chronological sorting."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Create fake frame files (out of order)
        filenames = [
            "frame_0003_000018.jpg",
            "frame_0001_000000.jpg",
            "frame_0002_000009.jpg",
            "not_a_frame.txt",
            "frame_0004_000027.png",
        ]
        for fn in filenames:
            with open(os.path.join(tmpdir, fn), 'wb') as f:
                f.write(b'\x00')  # minimal file

        frames = discover_frames(tmpdir)
        assert len(frames) == 4  # .txt excluded
        # Verify chronological order
        assert frames[0].timestamp_seconds == 0
        assert frames[1].timestamp_seconds == 9
        assert frames[2].timestamp_seconds == 18
        assert frames[3].timestamp_seconds == 27

        stats = get_frame_stats(frames)
        assert stats['total'] == 4
        assert stats['duration_seconds'] == 27
        assert stats['first_frame'] == "frame_0001_000000.jpg"
        assert stats['last_frame'] == "frame_0004_000027.png"

        print("  ✓ Frame discovery & sorting")
    finally:
        shutil.rmtree(tmpdir)


def test_resume_support():
    """TEST 5: Resume detects successful batches and skips them."""
    tmpdir = tempfile.mkdtemp()
    try:
        run_dir = os.path.join(tmpdir, "analysis", "run_test")
        os.makedirs(run_dir)

        # Simulate 5 successful, 1 failed, 4 pending (total 10)
        for i in range(1, 6):
            save_batch_result(run_dir, i, {
                'batch_id': i,
                'processing': {'status': 'success'},
            })
        save_batch_result(run_dir, 6, {
            'batch_id': 6,
            'processing': {'status': 'failed'},
            'error_type': 'invalid_json',
        })
        # batches 7-10 have no files (pending)

        statuses = get_batch_statuses(run_dir, 10)
        assert statuses['successful'] == [1, 2, 3, 4, 5]
        assert statuses['failed'] == [6]
        assert statuses['pending'] == [7, 8, 9, 10]

        print("  ✓ TEST 5: Resume detects successful/failed/pending batches")
    finally:
        shutil.rmtree(tmpdir)


def test_invalid_json_handling():
    """TEST 6: Invalid JSON results in error with raw response preserved."""
    result = {
        'batch_id': 4,
        'processing': {'status': 'failed'},
        'error_type': 'invalid_json',
        'raw_response': 'This is not valid JSON { broken',
        'retry_attempted': True,
    }
    tmpdir = tempfile.mkdtemp()
    try:
        save_batch_result(tmpdir, 4, result)
        loaded = load_batch_result(tmpdir, 4)
        assert loaded['processing']['status'] == 'failed'
        assert loaded['error_type'] == 'invalid_json'
        assert loaded['raw_response'] == 'This is not valid JSON { broken'
        assert loaded['retry_attempted'] is True
        print("  ✓ TEST 6: Failed batch preserves raw response")
    finally:
        shutil.rmtree(tmpdir)


def test_batch_failure_resilience():
    """TEST 4: 98 frames, one batch intentionally fails → other batches continue processing."""
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication(sys.argv)
    from core.frame_loader import FrameInfo
    from providers.base import AIProvider
    from ui.main_window import BatchWorker

    class MockProviderWithFailure(AIProvider):
        @property
        def provider_name(self) -> str:
            return "mock_resilience"

        def list_models(self) -> list[str]:
            return ["mock-vision"]

        def analyze_images(self, images, prompt, model, **kwargs) -> str:
            # Batch 3 intentionally fails with invalid non-JSON output
            if "Batch 3 " in prompt:
                return "Error: failed to analyze frame batch"
            return '{"summary": "Batch processed successfully", "scenes": []}'

        def generate_text(self, prompt, model, **kwargs) -> str:
            return '{"overall_summary": "Aggregation complete"}'

    tmpdir = tempfile.mkdtemp()
    try:
        frames = []
        for i in range(1, 99):
            path = os.path.join(tmpdir, f"frame_{i:04d}_{i*9:06d}.jpg")
            with open(path, "wb") as f:
                f.write(b"\xff\xd8\xff\xe0")  # Minimal JPEG header
            frames.append(FrameInfo(f"frame_{i:04d}_{i*9:06d}.jpg", path, i, i * 9))

        batches = create_batches(frames, 10)
        assert len(batches) == 10

        run_dir = os.path.join(tmpdir, "analysis", "run_test_fail")
        os.makedirs(run_dir, exist_ok=True)

        provider = MockProviderWithFailure("http://mock-url")
        worker = BatchWorker(
            batches=batches,
            provider=provider,
            model="mock-vision",
            user_prompt="Analyze frames",
            run_dir=run_dir,
            run_id="run_test_fail",
            total_batches=10,
            temperature=0.0,
            max_tokens=256,
            timeout=10,
            max_image_dim=None,
        )

        completed_events = []
        failed_events = []
        worker.batch_completed.connect(lambda bid, res: completed_events.append(bid))
        worker.batch_failed.connect(lambda bid, err: failed_events.append(bid))

        # Run worker synchronously
        worker.run()

        # Check results
        statuses = get_batch_statuses(run_dir, 10)
        assert statuses['failed'] == [3], f"Expected batch 3 to fail, got {statuses['failed']}"
        assert len(statuses['successful']) == 9
        assert 3 not in statuses['successful']
        assert statuses['pending'] == []
        assert len(completed_events) == 9
        assert failed_events == [3]

        print("  ✓ TEST 4: 98 frames / batch 3 fails → 9 other batches continue & complete")
    finally:
        shutil.rmtree(tmpdir)


def test_lmstudio_unavailable():
    """TEST 7: LM Studio unavailable → clear error without application crash."""
    import requests
    from providers.lmstudio import LMStudioProvider
    provider = LMStudioProvider("http://127.0.0.1:59999")
    try:
        provider.list_models()
        assert False, "Expected connection error"
    except requests.RequestException as e:
        # Provider cleanly raises RequestException which GUI catches in _refresh_models
        assert isinstance(e, requests.RequestException)
        print("  ✓ TEST 7: LM Studio unavailable → caught RequestException gracefully")


def test_ollama_unavailable():
    """TEST 8: Ollama unavailable → clear error without application crash."""
    import requests
    from providers.ollama import OllamaProvider
    provider = OllamaProvider("http://127.0.0.1:59999")
    try:
        provider.list_models()
        assert False, "Expected connection error"
    except requests.RequestException as e:
        # Provider cleanly raises RequestException which GUI catches in _refresh_models
        assert isinstance(e, requests.RequestException)
        print("  ✓ TEST 8: Ollama unavailable → caught RequestException gracefully")


def test_timing_and_final_result_ui():
    """TEST: Timing calculations & final_result.json UI integration."""
    from PySide6.QtWidgets import QApplication
    from ui.main_window import MainWindow, AggregationWorker
    from providers.base import AIProvider

    class MockTimingProvider(AIProvider):
        @property
        def provider_name(self) -> str:
            return "mock"
        def list_models(self) -> list[str]:
            return ["m"]
        def analyze_images(self, *args, **kwargs) -> str:
            return "{}"
        def generate_text(self, *args, **kwargs) -> str:
            return '{"overall_summary": "Test aggregation complete"}'

    tmpdir = tempfile.mkdtemp()
    try:
        app = QApplication.instance() or QApplication(sys.argv)
        win = MainWindow()
        win._current_run_dir = tmpdir
        win._batches_duration = 35.5
        win._agg_duration = 12.5
        win._update_timing_display()

        # Check timing labels
        assert "35.5s" in win.lbl_batches_time.text()
        assert "12.5s" in win.lbl_agg_time.text()
        assert "48.0s" in win.lbl_total_time.text()
        assert "35.5s" in win.lbl_prompt_timing.text()
        assert "12.5s" in win.lbl_prompt_timing.text()
        assert "48.0s" in win.lbl_prompt_timing.text()

        # Check AggregationWorker timing
        worker = AggregationWorker(
            run_dir=tmpdir,
            run_id="run_timing_test",
            batch_results=[{"batch_id": 1, "result": {"summary": "b1"}}],
            provider=MockTimingProvider("http://mock"),
            model="m",
            aggregation_prompt="Aggregate",
            frame_folder=tmpdir,
            frame_count=10,
            duration_seconds=90,
            batch_size=10,
            total_batches=1,
            temperature=0.0,
            max_tokens=128,
            timeout=10,
            batches_duration=35.5,
        )
        completed_results = []
        worker.completed.connect(lambda r: completed_results.append(r))
        worker.run()

        assert len(completed_results) == 1
        res = completed_results[0]
        assert "timing" in res
        assert res["timing"]["batches_duration_seconds"] == 35.5
        assert res["timing"]["aggregation_duration_seconds"] >= 0
        assert res["timing"]["total_duration_seconds"] >= 35.5

        # Check that final_result.json was saved with timing
        final_file = os.path.join(tmpdir, "final_result.json")
        assert os.path.exists(final_file)
        with open(final_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["timing"]["batches_duration_seconds"] == 35.5

        # Check final_result.json UI update
        win._update_final_result_ui()
        assert "Ready" in win.lbl_final_result_status.text()
        assert win.btn_view_final_result.isEnabled() is True
        assert win.btn_open_run_folder.isEnabled() is True

        print("  ✓ Timing & final_result.json UI tests passed")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    print("\n=== Running Core Logic & Spec Tests ===\n")
    test_parse_frame_filename()
    test_format_duration()
    test_is_image_file()
    test_batch_creation_98_frames()      # TEST 1
    test_batch_creation_100_frames()     # TEST 2
    test_batch_creation_7_frames()       # TEST 3
    test_batch_failure_resilience()      # TEST 4
    test_resume_support()                # TEST 5
    test_invalid_json_handling()         # TEST 6
    test_lmstudio_unavailable()          # TEST 7
    test_ollama_unavailable()            # TEST 8
    test_json_extraction()
    test_frame_discovery()
    test_timing_and_final_result_ui()    # New timing & UI features
    print("\n=== All tests passed ✓ ===\n")


