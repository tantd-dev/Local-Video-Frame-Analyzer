"""
Main application window.

Contains all GUI sections (frame folder, batch config, provider,
prompt, progress) and worker threads for non-blocking AI processing.
Supports both single-file and multi-file (batch) modes.
"""
import json
import logging
import os
import time
import traceback
from datetime import datetime

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot, QUrl, QTimer
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QTextEdit, QProgressBar, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QSplitter, QTabWidget, QSizePolicy, QDialog,
    QApplication, QStackedWidget, QFrame,
)

from core.frame_loader import discover_frames, get_frame_stats, FrameInfo
from core.batch_manager import create_batches, get_batch_summary, build_batch_prompt, Batch
from core.result_manager import (
    create_run_directory, save_config, save_batch_result,
    get_batch_statuses, find_existing_runs, save_final_result,
    load_all_successful_batches,
)
from core.aggregation import (
    DEFAULT_AGGREGATION_PROMPT, build_aggregation_input, estimate_token_count,
)
from core.excel_reporter import (
    append_video_result,
    check_excel_file_writable,
    create_or_load_workbook,
    save_workbook_safe,
    HAS_OPENPYXL,
)
from providers.base import AIProvider
from providers.lmstudio import LMStudioProvider
from providers.ollama import OllamaProvider
from utils.image_utils import load_and_encode_image
from utils.json_utils import safe_parse_json
from utils.timestamp import format_duration


# ---------------------------------------------------------------------------
# Default batch analysis prompt
# ---------------------------------------------------------------------------

DEFAULT_BATCH_PROMPT = """You are analyzing sequential frames extracted from the same video.
The frames are ordered chronologically.
Analyze the visual content represented by these frames.
Return ONLY valid JSON.
Use this structure:
{
  "summary": "",
  "scenes": [],
  "locations": [],
  "objects": [],
  "actions": [],
  "visible_text": [],
  "important_events": [],
  "uncertainties": []
}

Rules:
- Treat the frames as chronological.
- Only describe information supported by the images.
- Do not invent information.
- If information is uncertain, put it under "uncertainties".
- Mention important visual changes between frames.
- Return valid JSON only."""

JSON_RETRY_PROMPT = (
    "Your previous response was not valid JSON.\n\n"
    "Return ONLY valid JSON using the requested schema.\n"
    "Do not include markdown fences.\n"
    "Do not include explanations."
)


# ===================================================================
# Worker: Batch Processing
# ===================================================================

class BatchWorker(QObject):
    """Processes batches sequentially in a background thread."""

    batch_started = Signal(int)                # batch_id
    batch_completed = Signal(int, dict)        # batch_id, result_dict
    batch_failed = Signal(int, str)            # batch_id, error_message
    all_completed = Signal(dict)               # summary dict
    log_message = Signal(str)                  # log line

    def __init__(
        self,
        batches: list[Batch],
        provider: AIProvider,
        model: str,
        user_prompt: str,
        run_dir: str,
        run_id: str,
        total_batches: int,
        temperature: float,
        max_tokens: int,
        timeout: int,
        max_image_dim: int | None,
        skip_batch_ids: set[int] | None = None,
    ):
        super().__init__()
        self.batches = batches
        self.provider = provider
        self.model = model
        self.user_prompt = user_prompt
        self.run_dir = run_dir
        self.run_id = run_id
        self.total_batches = total_batches
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_image_dim = max_image_dim
        self.skip_batch_ids = skip_batch_ids or set()
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @Slot()
    def run(self):
        successful = 0
        failed = 0
        skipped = 0

        for batch in self.batches:
            if self._cancelled:
                self.log_message.emit("Processing cancelled by user.")
                break

            if batch.batch_id in self.skip_batch_ids:
                skipped += 1
                self.log_message.emit(
                    f"Skipping batch {batch.batch_id} (already successful)."
                )
                continue

            self.batch_started.emit(batch.batch_id)
            self.log_message.emit(
                f"Processing batch {batch.batch_id}/{self.total_batches} "
                f"({len(batch.frames)} frames, {batch.time_range_str})"
            )

            try:
                result = self._process_single_batch(batch)
                save_batch_result(self.run_dir, batch.batch_id, result)

                if result['processing']['status'] == 'success':
                    successful += 1
                    self.batch_completed.emit(batch.batch_id, result)
                    self.log_message.emit(
                        f"Batch {batch.batch_id} completed successfully "
                        f"({result['processing']['duration_seconds']:.1f}s)."
                    )
                else:
                    failed += 1
                    self.batch_failed.emit(
                        batch.batch_id,
                        result.get('error_type', 'unknown error'),
                    )
                    self.log_message.emit(
                        f"Batch {batch.batch_id} failed: "
                        f"{result.get('error_type', 'unknown')}"
                    )

            except Exception as e:
                failed += 1
                error_msg = f"{type(e).__name__}: {e}"
                self.log_message.emit(
                    f"Batch {batch.batch_id} exception: {error_msg}"
                )
                # Save error result
                error_result = self._make_error_result(
                    batch, 'exception', error_msg,
                )
                try:
                    save_batch_result(self.run_dir, batch.batch_id, error_result)
                except Exception:
                    pass
                self.batch_failed.emit(batch.batch_id, error_msg)

        summary = {
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'cancelled': self._cancelled,
        }
        self.all_completed.emit(summary)

    def _process_single_batch(self, batch: Batch) -> dict:
        """Process one batch: load images, call AI, parse JSON, retry if needed."""
        started_at = datetime.now()

        # Load and encode images
        images = []
        for frame in batch.frames:
            img_data = load_and_encode_image(
                frame.filepath,
                max_dimension=self.max_image_dim,
            )
            images.append(img_data)

        # Build prompt with frame metadata
        full_prompt = build_batch_prompt(batch, self.user_prompt)

        # Call provider
        raw_response = self.provider.analyze_images(
            images=images,
            prompt=full_prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

        # Try to parse JSON
        parsed, error = safe_parse_json(raw_response)

        if parsed is None:
            # Retry once with correction prompt
            self.log_message.emit(
                f"Batch {batch.batch_id}: invalid JSON, retrying..."
            )
            retry_prompt = f"{full_prompt}\n\n{JSON_RETRY_PROMPT}"
            try:
                raw_response_retry = self.provider.analyze_images(
                    images=images,
                    prompt=retry_prompt,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                parsed, error = safe_parse_json(raw_response_retry)
                if parsed is None:
                    # Both attempts failed
                    finished_at = datetime.now()
                    return {
                        'run_id': self.run_id,
                        'batch_id': batch.batch_id,
                        'total_batches': self.total_batches,
                        'frames': [
                            {'filename': f.filename, 'timestamp_seconds': f.timestamp_seconds}
                            for f in batch.frames
                        ],
                        'provider': self.provider.provider_name,
                        'model': self.model,
                        'prompt': self.user_prompt,
                        'raw_response': raw_response_retry,
                        'processing': {
                            'started_at': started_at.isoformat(),
                            'finished_at': finished_at.isoformat(),
                            'duration_seconds': round(
                                (finished_at - started_at).total_seconds(), 2
                            ),
                            'status': 'failed',
                        },
                        'error_type': 'invalid_json',
                        'retry_attempted': True,
                    }
                # Retry succeeded — use retried response
                raw_response = raw_response_retry
            except Exception as retry_err:
                finished_at = datetime.now()
                return {
                    'run_id': self.run_id,
                    'batch_id': batch.batch_id,
                    'total_batches': self.total_batches,
                    'frames': [
                        {'filename': f.filename, 'timestamp_seconds': f.timestamp_seconds}
                        for f in batch.frames
                    ],
                    'provider': self.provider.provider_name,
                    'model': self.model,
                    'prompt': self.user_prompt,
                    'raw_response': raw_response,
                    'processing': {
                        'started_at': started_at.isoformat(),
                        'finished_at': finished_at.isoformat(),
                        'duration_seconds': round(
                            (finished_at - started_at).total_seconds(), 2
                        ),
                        'status': 'failed',
                    },
                    'error_type': f'retry_failed: {retry_err}',
                    'retry_attempted': True,
                }

        finished_at = datetime.now()
        return {
            'run_id': self.run_id,
            'batch_id': batch.batch_id,
            'total_batches': self.total_batches,
            'frames': [
                {'filename': f.filename, 'timestamp_seconds': f.timestamp_seconds}
                for f in batch.frames
            ],
            'provider': self.provider.provider_name,
            'model': self.model,
            'prompt': self.user_prompt,
            'result': parsed,
            'raw_response': raw_response,
            'processing': {
                'started_at': started_at.isoformat(),
                'finished_at': finished_at.isoformat(),
                'duration_seconds': round(
                    (finished_at - started_at).total_seconds(), 2
                ),
                'status': 'success',
            },
        }

    def _make_error_result(self, batch: Batch, error_type: str, error_msg: str) -> dict:
        return {
            'run_id': self.run_id,
            'batch_id': batch.batch_id,
            'total_batches': self.total_batches,
            'frames': [
                {'filename': f.filename, 'timestamp_seconds': f.timestamp_seconds}
                for f in batch.frames
            ],
            'provider': self.provider.provider_name,
            'model': self.model,
            'prompt': self.user_prompt,
            'processing': {
                'started_at': datetime.now().isoformat(),
                'finished_at': datetime.now().isoformat(),
                'duration_seconds': 0,
                'status': 'failed',
            },
            'error_type': error_type,
            'error_message': error_msg,
        }


# ===================================================================
# Worker: Aggregation
# ===================================================================

class AggregationWorker(QObject):
    """Runs aggregation in a background thread."""

    completed = Signal(dict)    # final_result dict
    failed = Signal(str)        # error message
    log_message = Signal(str)

    def __init__(
        self,
        run_dir: str,
        run_id: str,
        batch_results: list[dict],
        provider: AIProvider,
        model: str,
        aggregation_prompt: str,
        frame_folder: str,
        frame_count: int,
        duration_seconds: int,
        batch_size: int,
        total_batches: int,
        temperature: float,
        max_tokens: int,
        timeout: int,
        batches_duration: float = 0.0,
    ):
        super().__init__()
        self.run_dir = run_dir
        self.run_id = run_id
        self.batch_results = batch_results
        self.provider = provider
        self.model = model
        self.aggregation_prompt = aggregation_prompt
        self.frame_folder = frame_folder
        self.frame_count = frame_count
        self.duration_seconds = duration_seconds
        self.batch_size = batch_size
        self.total_batches = total_batches
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.batches_duration = batches_duration

    @Slot()
    def run(self):
        started_at = datetime.now()
        try:
            self.log_message.emit("Building aggregation prompt...")
            full_prompt = build_aggregation_input(
                self.batch_results, self.aggregation_prompt
            )

            est_tokens = estimate_token_count(full_prompt)
            self.log_message.emit(
                f"Aggregation prompt: ~{est_tokens} estimated tokens."
            )

            self.log_message.emit("Sending aggregation request...")
            raw_response = self.provider.generate_text(
                prompt=full_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )

            parsed, error = safe_parse_json(raw_response)
            if parsed is None:
                # Retry once
                self.log_message.emit(
                    "Aggregation returned invalid JSON, retrying..."
                )
                retry_prompt = f"{full_prompt}\n\n{JSON_RETRY_PROMPT}"
                raw_response = self.provider.generate_text(
                    prompt=retry_prompt,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                parsed, error = safe_parse_json(raw_response)
                if parsed is None:
                    self.failed.emit(
                        f"Aggregation failed: could not parse JSON.\n"
                        f"Raw response saved.\nError: {error}"
                    )
                    return

            finished_at = datetime.now()
            agg_duration = round((finished_at - started_at).total_seconds(), 2)

            batches_dur = self.batches_duration
            if batches_dur <= 0.0:
                batches_dur = round(sum(
                    br.get('processing', {}).get('duration_seconds', 0.0)
                    for br in self.batch_results
                ), 2)

            # Count successful / failed
            successful_count = len(self.batch_results)
            failed_count = self.total_batches - successful_count

            final_result = {
                'run_id': self.run_id,
                'source': {
                    'frame_folder': self.frame_folder,
                    'frame_count': self.frame_count,
                    'duration_seconds': self.duration_seconds,
                },
                'model': {
                    'provider': self.provider.provider_name,
                    'name': self.model,
                },
                'batch_config': {
                    'batch_size': self.batch_size,
                    'total_batches': self.total_batches,
                },
                'batch_results': {
                    'successful': successful_count,
                    'failed': failed_count,
                },
                'timing': {
                    'batches_duration_seconds': batches_dur,
                    'aggregation_duration_seconds': agg_duration,
                    'total_duration_seconds': round(batches_dur + agg_duration, 2),
                },
                'final_analysis': parsed,
                'raw_response': raw_response,
            }

            save_final_result(self.run_dir, final_result)
            self.log_message.emit("Final result saved.")
            self.completed.emit(final_result)

        except Exception as e:
            self.failed.emit(f"Aggregation error: {type(e).__name__}: {e}")


# ===================================================================
# Worker: Multi-Video Processing (runs in background thread)
# ===================================================================

class MultiVideoWorker(QObject):
    """Processes multiple video folders sequentially.

    For each video folder:
      1. Discover frames
      2. Create batches and process them
      3. Run aggregation
      4. Write result row to Excel
    """

    video_started = Signal(int, str)            # video_index (0-based), video_name
    video_completed = Signal(int, str, dict)    # video_index, video_name, final_result
    video_failed = Signal(int, str, str)        # video_index, video_name, error_msg
    batch_log = Signal(str)                     # log line
    batch_progress = Signal(int, int, int)      # video_index, batch_id, total_batches
    all_videos_completed = Signal(dict)         # overall summary
    log_message = Signal(str)

    def __init__(
        self,
        video_folders: list[dict],   # list of {"name": str, "frames_path": str}
        provider: AIProvider,
        model: str,
        batch_prompt: str,
        aggregation_prompt: str,
        batch_size: int,
        temperature: float,
        max_tokens: int,
        timeout: int,
        max_image_dim: int | None,
        excel_path: str,
    ):
        super().__init__()
        self.video_folders = video_folders
        self.provider = provider
        self.model = model
        self.batch_prompt = batch_prompt
        self.aggregation_prompt = aggregation_prompt
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_image_dim = max_image_dim
        self.excel_path = excel_path
        self._cancelled = False
        self._wb = None
        self._ws = None
        self._fallback_excel_path = None

    def cancel(self):
        self._cancelled = True

    @Slot()
    def run(self):
        total_videos = len(self.video_folders)
        completed_count = 0
        failed_count = 0
        overall_start = time.time()

        # Pre-initialize or load Excel workbook
        if HAS_OPENPYXL and self.excel_path:
            try:
                self._wb, self._ws = create_or_load_workbook(self.excel_path)
            except Exception as e:
                self.log_message.emit(f"Initial Excel check warning: {e}")

        for idx, vf in enumerate(self.video_folders):
            if self._cancelled:
                self.log_message.emit("Multi-video processing cancelled by user.")
                break

            video_name = vf['name']
            frames_path = vf['frames_path']

            self.video_started.emit(idx, video_name)
            self.log_message.emit(
                f"═══ [{idx+1}/{total_videos}] Starting: {video_name} ═══"
            )

            try:
                result = self._process_single_video(idx, video_name, frames_path)
                if result is not None:
                    completed_count += 1
                    self.video_completed.emit(idx, video_name, result)
                    self.log_message.emit(
                        f"═══ [{idx+1}/{total_videos}] Completed: {video_name} ═══"
                    )
                else:
                    failed_count += 1
                    self.video_failed.emit(idx, video_name, "Processing returned no result")
            except Exception as e:
                failed_count += 1
                error_msg = f"{type(e).__name__}: {e}"
                self.log_message.emit(
                    f"═══ [{idx+1}/{total_videos}] FAILED: {video_name} — {error_msg} ═══"
                )
                self.video_failed.emit(idx, video_name, error_msg)

        # If any fallback occurred, attempt a final sync to main file
        if self._fallback_excel_path and self._wb and self.excel_path:
            self.log_message.emit(f"[Excel] Đang đồng bộ kết quả cuối cùng vào file chính: {self.excel_path}...")
            sync_ok, sync_path, _ = save_workbook_safe(
                self._wb, self.excel_path, max_retries=1, retry_delay=0.5
            )
            if sync_ok and os.path.abspath(sync_path) == os.path.abspath(self.excel_path):
                self.log_message.emit(f"[Excel] ✅ Đồng bộ thành công vào: {self.excel_path}")
                self._fallback_excel_path = None
            else:
                self.log_message.emit(
                    f"[Excel] ⚠️ File chính vẫn đang mở trong Excel. Kết quả đầy đủ nằm tại: {self._fallback_excel_path}"
                )

        overall_duration = round(time.time() - overall_start, 2)
        summary = {
            'total': total_videos,
            'completed': completed_count,
            'failed': failed_count,
            'cancelled': self._cancelled,
            'total_duration': overall_duration,
            'excel_path': self.excel_path,
            'fallback_excel_path': self._fallback_excel_path,
        }
        self.all_videos_completed.emit(summary)

    def _process_single_video(self, idx: int, video_name: str, frames_path: str) -> dict | None:
        """Process one video: frames → batches → AI → aggregation → Excel."""
        video_start = time.time()

        # 1. Discover frames
        self.log_message.emit(f"[{video_name}] Scanning frames in: {frames_path}")
        frames = discover_frames(frames_path)
        if not frames:
            self.log_message.emit(f"[{video_name}] No frames found, skipping.")
            return None

        stats = get_frame_stats(frames)
        self.log_message.emit(
            f"[{video_name}] Found {stats['total']} frames, "
            f"duration {format_duration(stats['duration_seconds'])}"
        )

        # 2. Create batches
        batches = create_batches(frames, self.batch_size)
        total_batches = len(batches)
        self.log_message.emit(
            f"[{video_name}] Created {total_batches} batches (batch_size={self.batch_size})"
        )

        # 3. Create run directory
        base = os.path.dirname(frames_path)
        run_dir, run_id = create_run_directory(base)
        self.log_message.emit(f"[{video_name}] Run directory: {run_dir}")

        # Save config
        config = {
            'frame_folder': frames_path,
            'frame_count': len(frames),
            'batch_size': self.batch_size,
            'total_batches': total_batches,
            'provider': self.provider.provider_name,
            'model': self.model,
            'base_url': getattr(self.provider, 'base_url', ''),
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'timeout': self.timeout,
            'max_image_dim': self.max_image_dim,
            'prompt': self.batch_prompt,
        }
        save_config(run_dir, config)

        # 4. Process batches
        batches_start = time.time()
        successful = 0
        failed = 0

        for batch in batches:
            if self._cancelled:
                self.log_message.emit(f"[{video_name}] Cancelled during batch processing.")
                return None

            self.batch_progress.emit(idx, batch.batch_id, total_batches)
            self.log_message.emit(
                f"[{video_name}] Batch {batch.batch_id}/{total_batches} "
                f"({len(batch.frames)} frames, {batch.time_range_str})"
            )

            try:
                result = self._process_batch(batch, run_dir, run_id, total_batches)
                save_batch_result(run_dir, batch.batch_id, result)

                if result['processing']['status'] == 'success':
                    successful += 1
                    self.log_message.emit(
                        f"[{video_name}] Batch {batch.batch_id} ✓ "
                        f"({result['processing']['duration_seconds']:.1f}s)"
                    )
                else:
                    failed += 1
                    self.log_message.emit(
                        f"[{video_name}] Batch {batch.batch_id} ✗ "
                        f"{result.get('error_type', 'unknown')}"
                    )
            except Exception as e:
                failed += 1
                self.log_message.emit(
                    f"[{video_name}] Batch {batch.batch_id} exception: {e}"
                )

        batches_duration = round(time.time() - batches_start, 2)
        self.log_message.emit(
            f"[{video_name}] Batches done: {successful} ok, {failed} failed, "
            f"time={batches_duration:.1f}s"
        )

        if successful == 0:
            self.log_message.emit(f"[{video_name}] No successful batches, skipping aggregation.")
            return None

        # 5. Run aggregation
        self.log_message.emit(f"[{video_name}] Starting aggregation...")
        agg_start = time.time()

        batch_results = load_all_successful_batches(run_dir, total_batches)
        full_prompt = build_aggregation_input(batch_results, self.aggregation_prompt)

        raw_response = self.provider.generate_text(
            prompt=full_prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

        parsed, error = safe_parse_json(raw_response)
        if parsed is None:
            # Retry once
            self.log_message.emit(f"[{video_name}] Aggregation invalid JSON, retrying...")
            retry_prompt = f"{full_prompt}\n\n{JSON_RETRY_PROMPT}"
            raw_response = self.provider.generate_text(
                prompt=retry_prompt,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            parsed, error = safe_parse_json(raw_response)
            if parsed is None:
                self.log_message.emit(
                    f"[{video_name}] Aggregation failed: could not parse JSON. Error: {error}"
                )
                return None

        agg_duration = round(time.time() - agg_start, 2)
        total_duration = round(time.time() - video_start, 2)

        final_result = {
            'run_id': run_id,
            'source': {
                'frame_folder': frames_path,
                'frame_count': stats['total'],
                'duration_seconds': stats['duration_seconds'],
            },
            'model': {
                'provider': self.provider.provider_name,
                'name': self.model,
            },
            'batch_config': {
                'batch_size': self.batch_size,
                'total_batches': total_batches,
            },
            'batch_results': {
                'successful': successful,
                'failed': failed,
            },
            'timing': {
                'batches_duration_seconds': batches_duration,
                'aggregation_duration_seconds': agg_duration,
                'total_duration_seconds': round(batches_duration + agg_duration, 2),
            },
            'final_analysis': parsed,
            'raw_response': raw_response,
        }

        save_final_result(run_dir, final_result)
        self.log_message.emit(f"[{video_name}] Final result saved.")

        # 6. Write to Excel
        final_result_path = os.path.join(run_dir, "final_result.json")
        summary_text = ""
        if isinstance(parsed, dict):
            summary_text = parsed.get('summary', '')
            if not summary_text:
                # Try nested final_analysis.summary
                fa = parsed.get('final_analysis', {})
                if isinstance(fa, dict):
                    summary_text = fa.get('summary', '')

        try:
            res = append_video_result(
                excel_path=self.excel_path,
                video_name=video_name,
                frame_count=stats['total'],
                batch_size=self.batch_size,
                total_batches=total_batches,
                successful_batches=successful,
                failed_batches=failed,
                batches_duration=batches_duration,
                aggregation_duration=agg_duration,
                total_duration=total_duration,
                final_result_path=final_result_path,
                summary_text=summary_text,
                wb=self._wb,
                ws=self._ws,
                log_fn=lambda msg: self.log_message.emit(f"[{video_name}] {msg}"),
            )
            self._wb = res['workbook']
            self._ws = res['worksheet']
            if res.get('is_fallback'):
                self._fallback_excel_path = res['saved_path']
                self.log_message.emit(
                    f"[{video_name}] ⚠️ Đã lưu dòng Excel vào file dự phòng: {res['saved_path']}\n"
                    f"  (Do file chính đang bị khóa. Hãy đóng Excel để tự động đồng bộ!)"
                )
            else:
                self.log_message.emit(f"[{video_name}] Excel row written to: {res['saved_path']}")
        except Exception as e:
            self.log_message.emit(f"[{video_name}] Excel write error: {e}")

        return final_result

    def _process_batch(self, batch: Batch, run_dir: str, run_id: str, total_batches: int) -> dict:
        """Process a single batch (synchronous, called from worker thread)."""
        started_at = datetime.now()

        images = []
        for frame in batch.frames:
            img_data = load_and_encode_image(
                frame.filepath,
                max_dimension=self.max_image_dim,
            )
            images.append(img_data)

        full_prompt = build_batch_prompt(batch, self.batch_prompt)

        raw_response = self.provider.analyze_images(
            images=images,
            prompt=full_prompt,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

        parsed, error = safe_parse_json(raw_response)

        if parsed is None:
            retry_prompt = f"{full_prompt}\n\n{JSON_RETRY_PROMPT}"
            try:
                raw_response_retry = self.provider.analyze_images(
                    images=images,
                    prompt=retry_prompt,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                parsed, error = safe_parse_json(raw_response_retry)
                if parsed is None:
                    finished_at = datetime.now()
                    return {
                        'run_id': run_id,
                        'batch_id': batch.batch_id,
                        'total_batches': total_batches,
                        'frames': [
                            {'filename': f.filename, 'timestamp_seconds': f.timestamp_seconds}
                            for f in batch.frames
                        ],
                        'provider': self.provider.provider_name,
                        'model': self.model,
                        'prompt': self.batch_prompt,
                        'raw_response': raw_response_retry,
                        'processing': {
                            'started_at': started_at.isoformat(),
                            'finished_at': finished_at.isoformat(),
                            'duration_seconds': round(
                                (finished_at - started_at).total_seconds(), 2
                            ),
                            'status': 'failed',
                        },
                        'error_type': 'invalid_json',
                        'retry_attempted': True,
                    }
                raw_response = raw_response_retry
            except Exception as retry_err:
                finished_at = datetime.now()
                return {
                    'run_id': run_id,
                    'batch_id': batch.batch_id,
                    'total_batches': total_batches,
                    'frames': [
                        {'filename': f.filename, 'timestamp_seconds': f.timestamp_seconds}
                        for f in batch.frames
                    ],
                    'provider': self.provider.provider_name,
                    'model': self.model,
                    'prompt': self.batch_prompt,
                    'raw_response': raw_response,
                    'processing': {
                        'started_at': started_at.isoformat(),
                        'finished_at': finished_at.isoformat(),
                        'duration_seconds': round(
                            (finished_at - started_at).total_seconds(), 2
                        ),
                        'status': 'failed',
                    },
                    'error_type': f'retry_failed: {retry_err}',
                    'retry_attempted': True,
                }

        finished_at = datetime.now()
        return {
            'run_id': run_id,
            'batch_id': batch.batch_id,
            'total_batches': total_batches,
            'frames': [
                {'filename': f.filename, 'timestamp_seconds': f.timestamp_seconds}
                for f in batch.frames
            ],
            'provider': self.provider.provider_name,
            'model': self.model,
            'prompt': self.batch_prompt,
            'result': parsed,
            'raw_response': raw_response,
            'processing': {
                'started_at': started_at.isoformat(),
                'finished_at': finished_at.isoformat(),
                'duration_seconds': round(
                    (finished_at - started_at).total_seconds(), 2
                ),
                'status': 'success',
            },
        }


# ===================================================================
# Main Window
# ===================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Video Frame Analyzer")
        self.setMinimumSize(1100, 800)

        # State
        self._frames: list[FrameInfo] = []
        self._frame_folder: str = ""
        self._current_run_dir: str = ""
        self._current_run_id: str = ""
        self._worker_thread: QThread | None = None
        self._worker: BatchWorker | None = None
        self._agg_thread: QThread | None = None
        self._agg_worker: AggregationWorker | None = None

        # Multi-video state
        self._multi_thread: QThread | None = None
        self._multi_worker: MultiVideoWorker | None = None
        self._multi_video_folders: list[dict] = []
        self._multi_excel_path: str = ""

        # Timing state
        self._batches_duration: float | None = None
        self._batches_start_time: float | None = None
        self._agg_duration: float | None = None
        self._agg_start_time: float | None = None
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(500)
        self._live_timer.timeout.connect(self._on_live_timer_tick)

        # Multi-video timing
        self._multi_start_time: float | None = None

        self._setup_ui()
        self._connect_signals()

    # ----- UI Construction -----

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(6)

        # === Mode Switch ===
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QHBoxLayout(mode_group)
        mode_layout.setContentsMargins(10, 6, 10, 6)

        self.btn_mode_single = QPushButton("📄 Single File")
        self.btn_mode_multi = QPushButton("📁 Multi File")

        for btn in (self.btn_mode_single, self.btn_mode_multi):
            btn.setCheckable(True)
            btn.setFixedHeight(32)
            btn.setMinimumWidth(140)

        self.btn_mode_single.setChecked(True)
        self._apply_mode_button_styles()

        mode_layout.addWidget(self.btn_mode_single)
        mode_layout.addWidget(self.btn_mode_multi)
        mode_layout.addStretch()
        main_layout.addWidget(mode_group)

        # === Stacked Widget for Mode-Specific Input ===
        self._input_stack = QStackedWidget()

        # --- Page 0: Single File Input ---
        single_page = QWidget()
        single_layout = QVBoxLayout(single_page)
        single_layout.setContentsMargins(0, 0, 0, 0)

        frame_group = QGroupBox("Frame Folder")
        frame_layout = QVBoxLayout(frame_group)

        row1 = QHBoxLayout()
        self.btn_select_folder = QPushButton("Select Frame Folder")
        self.btn_select_folder.setFixedWidth(160)
        self.lbl_folder_path = QLabel("No folder selected")
        self.lbl_folder_path.setWordWrap(True)
        row1.addWidget(self.btn_select_folder)
        row1.addWidget(self.lbl_folder_path, 1)
        frame_layout.addLayout(row1)

        info_grid = QGridLayout()
        info_grid.setHorizontalSpacing(24)
        self.lbl_frame_count = QLabel("Total frames: —")
        self.lbl_first_frame = QLabel("First: —")
        self.lbl_last_frame = QLabel("Last: —")
        self.lbl_duration = QLabel("Duration: —")
        self.lbl_avg_interval = QLabel("Avg interval: —")
        info_grid.addWidget(self.lbl_frame_count, 0, 0)
        info_grid.addWidget(self.lbl_first_frame, 0, 1)
        info_grid.addWidget(self.lbl_last_frame, 0, 2)
        info_grid.addWidget(self.lbl_duration, 1, 0)
        info_grid.addWidget(self.lbl_avg_interval, 1, 1)
        frame_layout.addLayout(info_grid)

        single_layout.addWidget(frame_group)
        self._input_stack.addWidget(single_page)

        # --- Page 1: Multi File Input ---
        multi_page = QWidget()
        multi_layout = QVBoxLayout(multi_page)
        multi_layout.setContentsMargins(0, 0, 0, 0)

        multi_group = QGroupBox("Multi-Video Folder")
        multi_group_layout = QVBoxLayout(multi_group)

        # Root folder selection
        multi_row1 = QHBoxLayout()
        self.btn_select_multi_folder = QPushButton("Select Root Folder")
        self.btn_select_multi_folder.setFixedWidth(160)
        self.lbl_multi_folder_path = QLabel("No folder selected")
        self.lbl_multi_folder_path.setWordWrap(True)
        multi_row1.addWidget(self.btn_select_multi_folder)
        multi_row1.addWidget(self.lbl_multi_folder_path, 1)
        multi_group_layout.addLayout(multi_row1)

        # Excel file selection
        multi_row2 = QHBoxLayout()
        self.btn_select_excel = QPushButton("Select Excel File")
        self.btn_select_excel.setFixedWidth(160)
        self.lbl_excel_path = QLabel("No Excel file selected")
        self.lbl_excel_path.setWordWrap(True)
        multi_row2.addWidget(self.btn_select_excel)
        multi_row2.addWidget(self.lbl_excel_path, 1)
        multi_group_layout.addLayout(multi_row2)

        # Discovered videos info
        self.lbl_multi_info = QLabel("Videos found: —")
        self.lbl_multi_info.setWordWrap(True)
        multi_group_layout.addWidget(self.lbl_multi_info)

        # Video list
        self.list_multi_videos = QListWidget()
        self.list_multi_videos.setMaximumHeight(120)
        multi_group_layout.addWidget(self.list_multi_videos)

        multi_layout.addWidget(multi_group)
        self._input_stack.addWidget(multi_page)

        main_layout.addWidget(self._input_stack)

        # --- Middle Splitter: Config | Progress ---
        splitter = QSplitter(Qt.Horizontal)

        # Left: Configuration
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Batch Config
        batch_group = QGroupBox("Batch Configuration")
        batch_layout = QGridLayout(batch_group)
        batch_layout.addWidget(QLabel("Frames per batch:"), 0, 0)
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1, 1000)
        self.spin_batch_size.setValue(10)
        batch_layout.addWidget(self.spin_batch_size, 0, 1)
        self.lbl_batch_info = QLabel("—")
        self.lbl_batch_info.setWordWrap(True)
        batch_layout.addWidget(self.lbl_batch_info, 1, 0, 1, 2)
        left_layout.addWidget(batch_group)

        # Provider
        provider_group = QGroupBox("Provider")
        provider_layout = QGridLayout(provider_group)

        provider_layout.addWidget(QLabel("Provider:"), 0, 0)
        self.combo_provider = QComboBox()
        self.combo_provider.addItems(["LM Studio", "Ollama"])
        provider_layout.addWidget(self.combo_provider, 0, 1)

        provider_layout.addWidget(QLabel("Base URL:"), 1, 0)
        self.edit_base_url = QLineEdit("http://localhost:1234")
        provider_layout.addWidget(self.edit_base_url, 1, 1)

        provider_layout.addWidget(QLabel("Model:"), 2, 0)
        model_row = QHBoxLayout()
        self.combo_model = QComboBox()
        self.combo_model.setEditable(True)
        self.combo_model.setMinimumWidth(200)
        self.btn_refresh_models = QPushButton("Refresh")
        self.btn_refresh_models.setFixedWidth(70)
        model_row.addWidget(self.combo_model, 1)
        model_row.addWidget(self.btn_refresh_models)
        provider_layout.addLayout(model_row, 2, 1)

        left_layout.addWidget(provider_group)

        # Model Parameters
        params_group = QGroupBox("Model Parameters")
        params_layout = QGridLayout(params_group)

        params_layout.addWidget(QLabel("Temperature:"), 0, 0)
        self.spin_temperature = QDoubleSpinBox()
        self.spin_temperature.setRange(0.0, 2.0)
        self.spin_temperature.setSingleStep(0.1)
        self.spin_temperature.setValue(0.0)
        self.spin_temperature.setDecimals(2)
        params_layout.addWidget(self.spin_temperature, 0, 1)

        params_layout.addWidget(QLabel("Max tokens:"), 1, 0)
        self.spin_max_tokens = QSpinBox()
        self.spin_max_tokens.setRange(256, 131072)
        self.spin_max_tokens.setValue(4096)
        self.spin_max_tokens.setSingleStep(512)
        params_layout.addWidget(self.spin_max_tokens, 1, 1)

        params_layout.addWidget(QLabel("Timeout (sec):"), 2, 0)
        self.spin_timeout = QSpinBox()
        self.spin_timeout.setRange(30, 3600)
        self.spin_timeout.setValue(300)
        self.spin_timeout.setSingleStep(30)
        params_layout.addWidget(self.spin_timeout, 2, 1)

        params_layout.addWidget(QLabel("Concurrency:"), 3, 0)
        self.spin_concurrency = QSpinBox()
        self.spin_concurrency.setRange(1, 8)
        self.spin_concurrency.setValue(1)
        params_layout.addWidget(self.spin_concurrency, 3, 1)

        params_layout.addWidget(QLabel("Max image dim:"), 4, 0)
        self.spin_max_image_dim = QSpinBox()
        self.spin_max_image_dim.setRange(0, 7680)
        self.spin_max_image_dim.setValue(1280)
        self.spin_max_image_dim.setSingleStep(128)
        self.spin_max_image_dim.setSpecialValueText("No resize")
        params_layout.addWidget(self.spin_max_image_dim, 4, 1)

        left_layout.addWidget(params_group)

        # Run Selection (only for single-file mode, but kept visible)
        run_group = QGroupBox("Run Selection (for Resume)")
        run_layout = QHBoxLayout(run_group)
        self.combo_runs = QComboBox()
        self.combo_runs.setMinimumWidth(200)
        self.btn_refresh_runs = QPushButton("Refresh")
        self.btn_refresh_runs.setFixedWidth(70)
        run_layout.addWidget(self.combo_runs, 1)
        run_layout.addWidget(self.btn_refresh_runs)
        self._run_group = run_group
        left_layout.addWidget(run_group)

        left_layout.addStretch()

        splitter.addWidget(left_widget)

        # Right: Progress
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.lbl_progress_status = QLabel("Idle")
        progress_layout.addWidget(self.lbl_progress_status)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        stats_row = QHBoxLayout()
        self.lbl_successful = QLabel("Successful: 0")
        self.lbl_failed = QLabel("Failed: 0")
        self.lbl_pending = QLabel("Pending: 0")
        stats_row.addWidget(self.lbl_successful)
        stats_row.addWidget(self.lbl_failed)
        stats_row.addWidget(self.lbl_pending)
        progress_layout.addLayout(stats_row)

        timing_row = QHBoxLayout()
        self.lbl_batches_time = QLabel("Batches Time: —")
        self.lbl_agg_time = QLabel("Final Analysis Time: —")
        self.lbl_total_time = QLabel("Total Time: —")
        self.lbl_batches_time.setStyleSheet("color: #2c3e50; font-size: 11px;")
        self.lbl_agg_time.setStyleSheet("color: #2c3e50; font-size: 11px;")
        self.lbl_total_time.setStyleSheet("color: #16a085; font-size: 11px; font-weight: bold;")
        timing_row.addWidget(self.lbl_batches_time)
        timing_row.addWidget(self.lbl_agg_time)
        timing_row.addWidget(self.lbl_total_time)
        progress_layout.addLayout(timing_row)

        self.batch_list = QListWidget()
        self.batch_list.setMinimumHeight(150)
        progress_layout.addWidget(self.batch_list)

        right_layout.addWidget(progress_group)

        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(160)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter, 1)

        # --- Prompts (Tabs) ---
        prompt_tabs = QTabWidget()

        self.txt_batch_prompt = QTextEdit()
        self.txt_batch_prompt.setPlainText(DEFAULT_BATCH_PROMPT)
        self.txt_batch_prompt.setMinimumHeight(100)
        prompt_tabs.addTab(self.txt_batch_prompt, "Batch Analysis Prompt")

        self.txt_agg_prompt = QTextEdit()
        self.txt_agg_prompt.setPlainText(DEFAULT_AGGREGATION_PROMPT)
        self.txt_agg_prompt.setMinimumHeight(100)
        prompt_tabs.addTab(self.txt_agg_prompt, "Aggregation Prompt")

        # Corner widget for prompt_tabs to show timing & final_result.json on the same row with prompts
        self.prompt_corner_widget = QWidget()
        corner_layout = QHBoxLayout(self.prompt_corner_widget)
        corner_layout.setContentsMargins(10, 2, 8, 2)
        corner_layout.setSpacing(10)

        self.lbl_prompt_timing = QLabel("⏱ Batches: — | Final Analysis: — | Total: —")
        self.lbl_prompt_timing.setStyleSheet("color: #34495e; font-size: 11px; font-weight: 500;")

        sep = QLabel("|")
        sep.setStyleSheet("color: #bdc3c7;")

        self.lbl_final_result_status = QLabel("📄 final_result.json: Not generated")
        self.lbl_final_result_status.setStyleSheet("color: #7f8c8d; font-weight: bold; font-size: 11px;")

        self.btn_view_final_result = QPushButton("View JSON")
        self.btn_view_final_result.setFixedHeight(24)
        self.btn_view_final_result.setEnabled(False)
        self.btn_view_final_result.setToolTip("View final_result.json content")

        self.btn_open_run_folder = QPushButton("Open Folder")
        self.btn_open_run_folder.setFixedHeight(24)
        self.btn_open_run_folder.setEnabled(False)
        self.btn_open_run_folder.setToolTip("Open analysis run folder in Explorer")

        corner_layout.addWidget(self.lbl_prompt_timing)
        corner_layout.addWidget(sep)
        corner_layout.addWidget(self.lbl_final_result_status)
        corner_layout.addWidget(self.btn_view_final_result)
        corner_layout.addWidget(self.btn_open_run_folder)

        prompt_tabs.setCornerWidget(self.prompt_corner_widget, Qt.TopRightCorner)

        main_layout.addWidget(prompt_tabs)

        # --- Action Buttons ---
        btn_layout = QHBoxLayout()

        # Single-file buttons
        self.btn_send_all = QPushButton("Send All")
        self.btn_resume = QPushButton("Resume")
        self.btn_force_rerun = QPushButton("Force Re-run")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_aggregate = QPushButton("Generate Final Analysis")
        self.btn_aggregate.setEnabled(False)

        # Multi-file buttons
        self.btn_multi_start = QPushButton("▶ Start Multi Processing")
        self.btn_multi_start.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; font-weight: bold; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #2ecc71; }"
            "QPushButton:disabled { background-color: #95a5a6; }"
        )
        self.btn_multi_cancel = QPushButton("Cancel")
        self.btn_multi_cancel.setEnabled(False)
        self.btn_multi_open_excel = QPushButton("Open Excel")
        self.btn_multi_open_excel.setEnabled(False)

        # Single-file action buttons
        btn_layout.addWidget(self.btn_send_all)
        btn_layout.addWidget(self.btn_resume)
        btn_layout.addWidget(self.btn_force_rerun)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_aggregate)

        # Multi-file action buttons
        btn_layout.addWidget(self.btn_multi_start)
        btn_layout.addWidget(self.btn_multi_cancel)
        btn_layout.addWidget(self.btn_multi_open_excel)

        main_layout.addLayout(btn_layout)

        # Initialize mode visibility
        self._set_mode('single')

    def _apply_mode_button_styles(self):
        """Apply visual styles to mode toggle buttons."""
        active_style = (
            "QPushButton { background-color: #2980b9; color: white; "
            "font-weight: bold; border-radius: 4px; padding: 4px 16px; }"
        )
        inactive_style = (
            "QPushButton { background-color: #ecf0f1; color: #2c3e50; "
            "border: 1px solid #bdc3c7; border-radius: 4px; padding: 4px 16px; }"
            "QPushButton:hover { background-color: #d5dbdb; }"
        )
        self.btn_mode_single.setStyleSheet(
            active_style if self.btn_mode_single.isChecked() else inactive_style
        )
        self.btn_mode_multi.setStyleSheet(
            active_style if self.btn_mode_multi.isChecked() else inactive_style
        )

    def _set_mode(self, mode: str):
        """Switch between 'single' and 'multi' mode."""
        is_single = (mode == 'single')

        self.btn_mode_single.setChecked(is_single)
        self.btn_mode_multi.setChecked(not is_single)
        self._apply_mode_button_styles()

        self._input_stack.setCurrentIndex(0 if is_single else 1)

        # Show/hide appropriate action buttons
        self.btn_send_all.setVisible(is_single)
        self.btn_resume.setVisible(is_single)
        self.btn_force_rerun.setVisible(is_single)
        self.btn_cancel.setVisible(is_single)
        self.btn_aggregate.setVisible(is_single)

        self.btn_multi_start.setVisible(not is_single)
        self.btn_multi_cancel.setVisible(not is_single)
        self.btn_multi_open_excel.setVisible(not is_single)

        # Run Selection is only for single-file mode
        self._run_group.setVisible(is_single)

        # Corner widget items for final result (single mode only)
        self.btn_view_final_result.setVisible(is_single)
        self.btn_open_run_folder.setVisible(is_single)
        self.lbl_final_result_status.setVisible(is_single)

    # ----- Signal Connections -----

    def _connect_signals(self):
        # Mode switch
        self.btn_mode_single.clicked.connect(lambda: self._set_mode('single'))
        self.btn_mode_multi.clicked.connect(lambda: self._set_mode('multi'))

        # Single-file signals
        self.btn_select_folder.clicked.connect(self._select_folder)
        self.spin_batch_size.valueChanged.connect(self._update_batch_info)
        self.combo_provider.currentIndexChanged.connect(self._on_provider_changed)
        self.btn_refresh_models.clicked.connect(self._refresh_models)
        self.btn_refresh_runs.clicked.connect(self._refresh_runs)
        self.combo_runs.currentIndexChanged.connect(self._on_run_selection_changed)
        self.btn_send_all.clicked.connect(lambda: self._start_processing('send_all'))
        self.btn_resume.clicked.connect(lambda: self._start_processing('resume'))
        self.btn_force_rerun.clicked.connect(lambda: self._start_processing('force'))
        self.btn_cancel.clicked.connect(self._cancel_processing)
        self.btn_aggregate.clicked.connect(self._start_aggregation)
        self.btn_view_final_result.clicked.connect(self._show_final_result_dialog)
        self.btn_open_run_folder.clicked.connect(self._open_current_run_folder)

        # Multi-file signals
        self.btn_select_multi_folder.clicked.connect(self._select_multi_folder)
        self.btn_select_excel.clicked.connect(self._select_excel_file)
        self.btn_multi_start.clicked.connect(self._start_multi_processing)
        self.btn_multi_cancel.clicked.connect(self._cancel_multi_processing)
        self.btn_multi_open_excel.clicked.connect(self._open_excel_file)

    # ----- Folder Selection -----

    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Frame Folder")
        if not folder:
            return

        self._frame_folder = folder
        self.lbl_folder_path.setText(folder)
        self._log(f"Selected folder: {folder}")

        try:
            self._frames = discover_frames(folder)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to scan folder:\n{e}")
            self._frames = []
            return

        stats = get_frame_stats(self._frames)
        self.lbl_frame_count.setText(f"Total frames: {stats['total']}")
        self.lbl_first_frame.setText(f"First: {stats['first_frame'] or '—'}")
        self.lbl_last_frame.setText(f"Last: {stats['last_frame'] or '—'}")
        self.lbl_duration.setText(
            f"Duration: {format_duration(stats['duration_seconds'])} "
            f"({stats['duration_seconds']}s)"
        )
        self.lbl_avg_interval.setText(
            f"Avg interval: {stats['avg_interval_seconds']}s"
        )

        self._log(f"Found {stats['total']} frames, "
                   f"duration {format_duration(stats['duration_seconds'])}")

        self._update_batch_info()
        self._refresh_runs()

    # ----- Batch Info -----

    def _update_batch_info(self):
        total = len(self._frames)
        batch_size = self.spin_batch_size.value()
        summary = get_batch_summary(total, batch_size)

        lines = [
            f"Total frames: {summary['total_frames']}",
            f"Batch size: {summary['batch_size']}",
            f"Total batches: {summary['total_batches']}",
        ]
        for d in summary['distribution']:
            lines.append(f"  {d['count']} batch{'es' if d['count'] > 1 else ''} × {d['frames']} frames")

        self.lbl_batch_info.setText("\n".join(lines))

    # ----- Provider -----

    def _on_provider_changed(self):
        provider_name = self.combo_provider.currentText()
        if provider_name == "LM Studio":
            self.edit_base_url.setText("http://localhost:1234")
        elif provider_name == "Ollama":
            self.edit_base_url.setText("http://localhost:11434")

    def _get_provider(self) -> AIProvider:
        provider_name = self.combo_provider.currentText()
        base_url = self.edit_base_url.text().strip()
        if provider_name == "LM Studio":
            return LMStudioProvider(base_url)
        elif provider_name == "Ollama":
            return OllamaProvider(base_url)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")

    def _refresh_models(self):
        try:
            provider = self._get_provider()
            models = provider.list_models()
            self.combo_model.clear()
            self.combo_model.addItems(models)
            self._log(f"Found {len(models)} model(s) from {provider.provider_name}")
            if not models:
                QMessageBox.information(
                    self, "Models",
                    "No models found. Make sure a model is loaded."
                )
        except Exception as e:
            QMessageBox.warning(
                self, "Error",
                f"Failed to list models:\n{e}\n\n"
                f"Make sure the provider is running."
            )
            self._log(f"Error listing models: {e}")

    # ----- Run Selection & Final Result UI -----

    def _refresh_runs(self):
        self.combo_runs.blockSignals(True)
        self.combo_runs.clear()
        if not self._frame_folder:
            self.combo_runs.blockSignals(False)
            self._update_final_result_ui()
            return

        base = os.path.dirname(self._frame_folder)
        runs = find_existing_runs(base)
        for r in runs:
            label = r['run_id']
            if r['config']:
                batch_size = r['config'].get('batch_size', '?')
                model = r['config'].get('model', '?')
                label += f"  (batch={batch_size}, model={model})"
            fr_path = os.path.join(r['run_dir'], "final_result.json")
            if os.path.exists(fr_path):
                label += " [final ✓]"
            self.combo_runs.addItem(label, r)

        self.combo_runs.blockSignals(False)
        if self.combo_runs.count() > 0:
            self._on_run_selection_changed()
        else:
            self._update_final_result_ui()

    def _get_selected_run(self) -> dict | None:
        idx = self.combo_runs.currentIndex()
        if idx < 0:
            return None
        return self.combo_runs.itemData(idx)

    def _on_run_selection_changed(self):
        run_info = self._get_selected_run()
        if not run_info:
            return
        self._current_run_dir = run_info['run_dir']
        self._current_run_id = run_info['run_id']

        # Estimate batches duration from existing batch results
        existing_results = load_all_successful_batches(self._current_run_dir, 1000)
        if existing_results:
            self._batches_duration = round(sum(
                b.get('processing', {}).get('duration_seconds', 0.0)
                for b in existing_results
            ), 2)
        else:
            self._batches_duration = None

        # Check if final_result.json exists
        final_path = os.path.join(self._current_run_dir, "final_result.json")
        if os.path.exists(final_path):
            try:
                with open(final_path, 'r', encoding='utf-8') as f:
                    final_data = json.load(f)
                timing = final_data.get('timing', {})
                if 'aggregation_duration_seconds' in timing:
                    self._agg_duration = timing['aggregation_duration_seconds']
                if 'batches_duration_seconds' in timing:
                    self._batches_duration = timing['batches_duration_seconds']
            except Exception:
                self._agg_duration = None
        else:
            self._agg_duration = None

        self._update_timing_display()
        self._update_final_result_ui()

    def _update_final_result_ui(self):
        """Update final_result.json display based on current run directory."""
        if not self._current_run_dir or not os.path.isdir(self._current_run_dir):
            self.lbl_final_result_status.setText("📄 final_result.json: Not generated")
            self.lbl_final_result_status.setStyleSheet("color: #7f8c8d; font-weight: bold; font-size: 11px;")
            self.lbl_final_result_status.setToolTip("No active run selected.")
            self.btn_view_final_result.setEnabled(False)
            self.btn_open_run_folder.setEnabled(False)
            return

        final_path = os.path.join(self._current_run_dir, "final_result.json")
        self.btn_open_run_folder.setEnabled(True)

        if os.path.exists(final_path):
            try:
                size_kb = os.path.getsize(final_path) / 1024.0
            except Exception:
                size_kb = 0.0

            self.lbl_final_result_status.setText(f"✓ final_result.json: Ready ({size_kb:.1f} KB)")
            self.lbl_final_result_status.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 11px;")
            self.lbl_final_result_status.setToolTip(f"Full path:\n{final_path}")
            self.btn_view_final_result.setEnabled(True)
        else:
            self.lbl_final_result_status.setText("📄 final_result.json: Not generated")
            self.lbl_final_result_status.setStyleSheet("color: #7f8c8d; font-weight: bold; font-size: 11px;")
            self.lbl_final_result_status.setToolTip(f"Not yet created in:\n{final_path}")
            self.btn_view_final_result.setEnabled(False)

    def _show_final_result_dialog(self):
        if not self._current_run_dir:
            return
        final_path = os.path.join(self._current_run_dir, "final_result.json")
        if not os.path.exists(final_path):
            QMessageBox.warning(self, "Not Found", f"File does not exist:\n{final_path}")
            return

        try:
            with open(final_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            try:
                formatted_json = json.dumps(json.loads(raw_text), indent=2, ensure_ascii=False)
            except Exception:
                formatted_json = raw_text
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to read file:\n{e}")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"final_result.json — {self._current_run_id}")
        dlg.resize(850, 600)
        vbox = QVBoxLayout(dlg)

        path_lbl = QLabel(f"<b>File:</b> {final_path}")
        path_lbl.setWordWrap(True)
        vbox.addWidget(path_lbl)

        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText(formatted_json)
        txt.setFontFamily("Consolas")
        vbox.addWidget(txt)

        btn_row = QHBoxLayout()
        btn_open_ext = QPushButton("Open in External Viewer")
        btn_copy = QPushButton("Copy JSON")
        btn_close = QPushButton("Close")

        btn_open_ext.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(final_path)))
        btn_copy.clicked.connect(lambda: (QApplication.clipboard().setText(formatted_json), QMessageBox.information(dlg, "Copied", "JSON copied to clipboard!")))
        btn_close.clicked.connect(dlg.accept)

        btn_row.addWidget(btn_open_ext)
        btn_row.addWidget(btn_copy)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        vbox.addLayout(btn_row)

        dlg.exec()

    def _open_current_run_folder(self):
        if self._current_run_dir and os.path.isdir(self._current_run_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._current_run_dir))

    # ----- Timing Helpers -----

    def _on_live_timer_tick(self):
        if self._batches_start_time is not None:
            self._batches_duration = time.time() - self._batches_start_time
            self._update_timing_display(batches_running=True)
        elif self._agg_start_time is not None:
            self._agg_duration = time.time() - self._agg_start_time
            self._update_timing_display(agg_running=True)
        elif self._multi_start_time is not None:
            # Multi-video mode: just update the total time label
            elapsed = time.time() - self._multi_start_time
            self.lbl_total_time.setText(f"Total Time: {self._format_time_sec(elapsed, is_running=True)}")

    def _format_time_sec(self, seconds: float | None, is_running: bool = False) -> str:
        if seconds is None:
            return "—"
        if seconds < 60:
            text = f"{seconds:.1f}s"
        else:
            m = int(seconds // 60)
            s = seconds % 60
            text = f"{m}m {s:04.1f}s"
        if is_running:
            text += " (running...)"
        return text

    def _update_timing_display(
        self,
        batches_running: bool = False,
        agg_running: bool = False,
    ):
        b_time_str = self._format_time_sec(self._batches_duration, is_running=batches_running)
        a_time_str = self._format_time_sec(self._agg_duration, is_running=agg_running)

        if self._batches_duration is not None or self._agg_duration is not None:
            b_val = self._batches_duration or 0.0
            a_val = self._agg_duration or 0.0
            tot_val = b_val + a_val
            tot_time_str = self._format_time_sec(tot_val, is_running=(batches_running or agg_running))
        else:
            tot_time_str = "—"

        self.lbl_batches_time.setText(f"Batches Time: {b_time_str}")
        self.lbl_agg_time.setText(f"Final Analysis Time: {a_time_str}")
        self.lbl_total_time.setText(f"Total Time: {tot_time_str}")

        self.lbl_prompt_timing.setText(
            f"⏱ Batches: {b_time_str} | Final Analysis: {a_time_str} | Total: {tot_time_str}"
        )

    # ----- Processing -----

    def _start_processing(self, mode: str):
        """Start batch processing.

        mode: 'send_all' = new run, process all batches
              'resume'   = existing run, skip successful batches
              'force'    = new run, process all batches (same as send_all)
        """
        if not self._frames:
            QMessageBox.warning(self, "Error", "No frames loaded. Select a folder first.")
            return

        model_name = self.combo_model.currentText().strip()
        if not model_name:
            QMessageBox.warning(self, "Error", "No model selected.")
            return

        batch_size = self.spin_batch_size.value()
        batches = create_batches(self._frames, batch_size)
        total_batches = len(batches)

        if total_batches == 0:
            QMessageBox.warning(self, "Error", "No batches to process.")
            return

        try:
            provider = self._get_provider()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Provider error:\n{e}")
            return

        skip_ids: set[int] = set()

        if mode == 'resume':
            run_info = self._get_selected_run()
            if not run_info:
                QMessageBox.warning(
                    self, "Error",
                    "No existing run selected for resume.\n"
                    "Select a run or use 'Send All' for a new run."
                )
                return
            run_dir = run_info['run_dir']
            run_id = run_info['run_id']
            statuses = get_batch_statuses(run_dir, total_batches)
            skip_ids = set(statuses['successful'])
            self._log(
                f"Resuming run {run_id}: "
                f"{len(skip_ids)} successful, "
                f"{len(statuses['failed'])} failed, "
                f"{len(statuses['pending'])} pending"
            )
        else:
            base = os.path.dirname(self._frame_folder)
            run_dir, run_id = create_run_directory(base)
            self._log(f"Created new run: {run_id}")

        self._current_run_dir = run_dir
        self._current_run_id = run_id

        # Reset & start timing for batches
        self._batches_start_time = time.time()
        self._batches_duration = 0.0
        self._agg_start_time = None
        self._agg_duration = None
        self._update_timing_display(batches_running=True)
        self._update_final_result_ui()
        self._live_timer.start()

        # Save config
        config = {
            'frame_folder': self._frame_folder,
            'frame_count': len(self._frames),
            'batch_size': batch_size,
            'total_batches': total_batches,
            'provider': provider.provider_name,
            'model': model_name,
            'base_url': self.edit_base_url.text().strip(),
            'temperature': self.spin_temperature.value(),
            'max_tokens': self.spin_max_tokens.value(),
            'timeout': self.spin_timeout.value(),
            'max_image_dim': self.spin_max_image_dim.value() or None,
            'prompt': self.txt_batch_prompt.toPlainText(),
        }
        save_config(run_dir, config)

        # Setup logging to file
        self._setup_file_logger(run_dir)

        # Initialize progress UI
        self._init_progress_ui(batches, skip_ids)

        max_dim = self.spin_max_image_dim.value()
        if max_dim == 0:
            max_dim = None

        # Create worker and thread
        self._worker = BatchWorker(
            batches=batches,
            provider=provider,
            model=model_name,
            user_prompt=self.txt_batch_prompt.toPlainText(),
            run_dir=run_dir,
            run_id=run_id,
            total_batches=total_batches,
            temperature=self.spin_temperature.value(),
            max_tokens=self.spin_max_tokens.value(),
            timeout=self.spin_timeout.value(),
            max_image_dim=max_dim,
            skip_batch_ids=skip_ids,
        )
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)

        self._worker_thread.started.connect(self._worker.run)
        self._worker.batch_started.connect(self._on_batch_started)
        self._worker.batch_completed.connect(self._on_batch_completed)
        self._worker.batch_failed.connect(self._on_batch_failed)
        self._worker.all_completed.connect(self._on_all_completed)
        self._worker.log_message.connect(self._log)
        self._worker.all_completed.connect(self._worker_thread.quit)

        self._set_processing_ui(True)
        self._worker_thread.start()

    def _init_progress_ui(self, batches: list[Batch], skip_ids: set[int]):
        self.batch_list.clear()
        self._batch_items: dict[int, QListWidgetItem] = {}
        self._stats = {'successful': 0, 'failed': 0, 'pending': 0, 'total': len(batches)}

        for batch in batches:
            if batch.batch_id in skip_ids:
                text = f"✓ Batch {batch.batch_id:02d} — Skipped (already successful)"
                item = QListWidgetItem(text)
                item.setForeground(Qt.darkGreen)
                self._stats['successful'] += 1
            else:
                text = f"○ Batch {batch.batch_id:02d} — Pending ({len(batch.frames)} frames)"
                item = QListWidgetItem(text)
                item.setForeground(Qt.gray)
                self._stats['pending'] += 1

            self.batch_list.addItem(item)
            self._batch_items[batch.batch_id] = item

        self._update_stats_labels()
        self.progress_bar.setRange(0, self._stats['total'])
        self.progress_bar.setValue(self._stats['successful'])

    def _update_stats_labels(self):
        self.lbl_successful.setText(f"Successful: {self._stats['successful']}")
        self.lbl_failed.setText(f"Failed: {self._stats['failed']}")
        self.lbl_pending.setText(f"Pending: {self._stats['pending']}")

    def _set_processing_ui(self, processing: bool):
        self.btn_send_all.setEnabled(not processing)
        self.btn_resume.setEnabled(not processing)
        self.btn_force_rerun.setEnabled(not processing)
        self.btn_cancel.setEnabled(processing)
        self.btn_aggregate.setEnabled(not processing)
        self.btn_select_folder.setEnabled(not processing)

    # ----- Batch Callbacks -----

    def _on_batch_started(self, batch_id: int):
        item = self._batch_items.get(batch_id)
        if item:
            item.setText(f"● Batch {batch_id:02d} — Processing...")
            item.setForeground(Qt.blue)
            self.batch_list.scrollToItem(item)

        total = self._stats['total']
        done = self._stats['successful'] + self._stats['failed']
        self.lbl_progress_status.setText(
            f"Processing batch {batch_id} / {total}"
        )
        if self._stats['pending'] > 0:
            self._stats['pending'] -= 1
            self._update_stats_labels()

    def _on_batch_completed(self, batch_id: int, result: dict):
        item = self._batch_items.get(batch_id)
        duration = result.get('processing', {}).get('duration_seconds', 0)
        if item:
            item.setText(
                f"✓ Batch {batch_id:02d} — Success ({duration:.1f}s)"
            )
            item.setForeground(Qt.darkGreen)

        self._stats['successful'] += 1
        self.progress_bar.setValue(self._stats['successful'] + self._stats['failed'])
        self._update_stats_labels()

    def _on_batch_failed(self, batch_id: int, error: str):
        item = self._batch_items.get(batch_id)
        if item:
            item.setText(f"✗ Batch {batch_id:02d} — Failed: {error}")
            item.setForeground(Qt.red)

        self._stats['failed'] += 1
        self.progress_bar.setValue(self._stats['successful'] + self._stats['failed'])
        self._update_stats_labels()

    def _on_all_completed(self, summary: dict):
        self._live_timer.stop()
        if self._batches_start_time is not None:
            self._batches_duration = round(time.time() - self._batches_start_time, 2)
            self._batches_start_time = None

        self._set_processing_ui(False)
        self._update_timing_display()
        self._update_final_result_ui()

        cancelled = summary.get('cancelled', False)
        status = "cancelled" if cancelled else "completed"
        self.lbl_progress_status.setText(
            f"Processing {status}: "
            f"{summary['successful']} successful, "
            f"{summary['failed']} failed, "
            f"{summary['skipped']} skipped"
        )
        self._log(
            f"Processing {status}: {summary['successful']} successful, "
            f"{summary['failed']} failed, {summary['skipped']} skipped"
        )

        # Enable aggregation if we have successful batches
        if summary['successful'] > 0 or summary['skipped'] > 0:
            self.btn_aggregate.setEnabled(True)

        self._refresh_runs()

    # ----- Cancel -----

    def _cancel_processing(self):
        if self._worker:
            self._worker.cancel()
            self._log("Cancel requested — waiting for current batch to finish...")
            self.btn_cancel.setEnabled(False)

    # ----- Aggregation -----

    def _start_aggregation(self):
        if not self._current_run_dir:
            # Try to use selected run
            run_info = self._get_selected_run()
            if run_info:
                self._current_run_dir = run_info['run_dir']
                self._current_run_id = run_info['run_id']
            else:
                QMessageBox.warning(self, "Error", "No run directory available.")
                return

        model_name = self.combo_model.currentText().strip()
        if not model_name:
            QMessageBox.warning(self, "Error", "No model selected.")
            return

        try:
            provider = self._get_provider()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Provider error:\n{e}")
            return

        batch_size = self.spin_batch_size.value()
        total_batches = len(create_batches(self._frames, batch_size))
        batch_results = load_all_successful_batches(
            self._current_run_dir, total_batches
        )

        if not batch_results:
            QMessageBox.warning(
                self, "Error",
                "No successful batch results found for aggregation."
            )
            return

        stats = get_frame_stats(self._frames)

        self._log(
            f"Starting aggregation with {len(batch_results)} "
            f"successful batch results..."
        )

        # Start timing for aggregation
        self._agg_start_time = time.time()
        self._agg_duration = 0.0
        self._update_timing_display(agg_running=True)
        self.lbl_final_result_status.setText("⏳ final_result.json: Generating...")
        self.lbl_final_result_status.setStyleSheet("color: #e67e22; font-weight: bold; font-size: 11px;")
        self._live_timer.start()

        self._agg_worker = AggregationWorker(
            run_dir=self._current_run_dir,
            run_id=self._current_run_id,
            batch_results=batch_results,
            provider=provider,
            model=model_name,
            aggregation_prompt=self.txt_agg_prompt.toPlainText(),
            frame_folder=self._frame_folder,
            frame_count=len(self._frames),
            duration_seconds=stats['duration_seconds'],
            batch_size=batch_size,
            total_batches=total_batches,
            temperature=self.spin_temperature.value(),
            max_tokens=self.spin_max_tokens.value(),
            timeout=self.spin_timeout.value(),
            batches_duration=self._batches_duration or 0.0,
        )
        self._agg_thread = QThread()
        self._agg_worker.moveToThread(self._agg_thread)

        self._agg_thread.started.connect(self._agg_worker.run)
        self._agg_worker.completed.connect(self._on_aggregation_completed)
        self._agg_worker.failed.connect(self._on_aggregation_failed)
        self._agg_worker.log_message.connect(self._log)
        self._agg_worker.completed.connect(self._agg_thread.quit)
        self._agg_worker.failed.connect(self._agg_thread.quit)

        self.btn_aggregate.setEnabled(False)
        self.lbl_progress_status.setText("Running aggregation...")
        self._agg_thread.start()

    def _on_aggregation_completed(self, result: dict):
        self._live_timer.stop()
        if self._agg_start_time is not None:
            self._agg_duration = round(time.time() - self._agg_start_time, 2)
            self._agg_start_time = None

        self.btn_aggregate.setEnabled(True)
        self.lbl_progress_status.setText("Aggregation completed ✓")
        self._log("Aggregation completed successfully.")
        self._update_timing_display()
        self._update_final_result_ui()
        self._refresh_runs()

        QMessageBox.information(
            self, "Aggregation Complete",
            f"Final result saved to:\n{self._current_run_dir}\\final_result.json"
        )

    def _on_aggregation_failed(self, error: str):
        self._live_timer.stop()
        if self._agg_start_time is not None:
            self._agg_duration = round(time.time() - self._agg_start_time, 2)
            self._agg_start_time = None

        self.btn_aggregate.setEnabled(True)
        self.lbl_progress_status.setText("Aggregation failed ✗")
        self._log(f"Aggregation failed: {error}")
        self._update_timing_display()
        self.lbl_final_result_status.setText("✗ final_result.json: Failed")
        self.lbl_final_result_status.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 11px;")
        QMessageBox.warning(self, "Aggregation Failed", error)

    # ===================================================================
    # Multi-File Mode
    # ===================================================================

    def _select_multi_folder(self):
        """Select root folder containing video subfolders with frames."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Root Folder (contains video subfolders)"
        )
        if not folder:
            return

        self.lbl_multi_folder_path.setText(folder)
        self._log(f"[Multi] Selected root folder: {folder}")

        # Scan for video subfolders with "frames" directories
        self._multi_video_folders = []
        self.list_multi_videos.clear()

        try:
            for entry in sorted(os.listdir(folder)):
                sub_path = os.path.join(folder, entry)
                if not os.path.isdir(sub_path):
                    continue

                frames_path = os.path.join(sub_path, "frames")
                if os.path.isdir(frames_path):
                    # Check if frames folder actually has image files
                    try:
                        frames = discover_frames(frames_path)
                        frame_count = len(frames)
                    except Exception:
                        frame_count = 0

                    if frame_count > 0:
                        self._multi_video_folders.append({
                            'name': entry,
                            'frames_path': frames_path,
                            'frame_count': frame_count,
                        })
                        item = QListWidgetItem(
                            f"📹 {entry}  ({frame_count} frames)"
                        )
                        self.list_multi_videos.addItem(item)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to scan folder:\n{e}")
            return

        count = len(self._multi_video_folders)
        total_frames = sum(vf['frame_count'] for vf in self._multi_video_folders)
        self.lbl_multi_info.setText(
            f"Videos found: {count}  |  Total frames across all videos: {total_frames}"
        )
        self._log(f"[Multi] Found {count} video folder(s) with {total_frames} total frames")

        if count == 0:
            QMessageBox.information(
                self, "No Videos Found",
                "No video subfolders with 'frames' directory found.\n\n"
                "Expected structure:\n"
                "  <selected folder>/\n"
                "    <video_name_1>/frames/\n"
                "    <video_name_2>/frames/\n"
                "    ..."
            )

    def _select_excel_file(self):
        """Select or create an Excel file for multi-video results."""
        if not HAS_OPENPYXL:
            QMessageBox.warning(
                self, "Missing Dependency",
                "openpyxl is required for Excel export.\n"
                "Install it with: pip install openpyxl"
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Excel File for Results",
            os.path.expanduser("~/multi_video_results.xlsx"),
            "Excel Files (*.xlsx);;All Files (*)",
        )
        if not path:
            return

        if not path.endswith('.xlsx'):
            path += '.xlsx'

        # Check if the file is writable / locked
        is_writable, err_msg = check_excel_file_writable(path)
        if not is_writable:
            QMessageBox.warning(
                self, "Excel File Locked / Permission Denied",
                f"Không thể chọn tệp Excel này:\n\n{err_msg}\n\n"
                f"Vui lòng đóng file Excel nếu đang mở hoặc chọn vị trí khác."
            )
            return

        self._multi_excel_path = path
        self.lbl_excel_path.setText(path)
        self._log(f"[Multi] Excel file: {path}")

        # Pre-create the workbook if it doesn't exist
        try:
            create_or_load_workbook(path)
            self._log(f"[Multi] Excel file ready.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to create/open Excel file:\n{e}")

    def _start_multi_processing(self):
        """Start processing all discovered video folders."""
        if not self._multi_video_folders:
            QMessageBox.warning(
                self, "Error",
                "No video folders loaded. Select a root folder first."
            )
            return

        if not self._multi_excel_path:
            QMessageBox.warning(
                self, "Error",
                "No Excel file selected. Please select an Excel file for results."
            )
            return

        # Validate that Excel file is not locked before starting batches
        is_writable, err_msg = check_excel_file_writable(self._multi_excel_path)
        if not is_writable:
            QMessageBox.warning(
                self, "Excel File Locked",
                f"Tệp Excel kết quả hiện không thể ghi được:\n{self._multi_excel_path}\n\n"
                f"{err_msg}\n\n"
                f"Vui lòng đóng file trong Microsoft Excel rồi nhấn Bắt đầu lại."
            )
            return

        model_name = self.combo_model.currentText().strip()
        if not model_name:
            QMessageBox.warning(self, "Error", "No model selected.")
            return

        try:
            provider = self._get_provider()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Provider error:\n{e}")
            return

        max_dim = self.spin_max_image_dim.value()
        if max_dim == 0:
            max_dim = None

        total_videos = len(self._multi_video_folders)

        # Confirm
        reply = QMessageBox.question(
            self, "Start Multi-Video Processing",
            f"Process {total_videos} video(s) with the following config?\n\n"
            f"  Model: {model_name}\n"
            f"  Batch size: {self.spin_batch_size.value()}\n"
            f"  Temperature: {self.spin_temperature.value()}\n"
            f"  Excel: {self._multi_excel_path}\n\n"
            f"This may take a long time.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        # Prepare progress UI
        self.batch_list.clear()
        self._multi_video_items: dict[int, QListWidgetItem] = {}
        for idx, vf in enumerate(self._multi_video_folders):
            item = QListWidgetItem(
                f"○ [{idx+1}/{total_videos}] {vf['name']} — Pending ({vf['frame_count']} frames)"
            )
            item.setForeground(Qt.gray)
            self.batch_list.addItem(item)
            self._multi_video_items[idx] = item

        self.progress_bar.setRange(0, total_videos)
        self.progress_bar.setValue(0)
        self.lbl_progress_status.setText(f"Multi-video: 0/{total_videos}")
        self.lbl_successful.setText("Completed: 0")
        self.lbl_failed.setText("Failed: 0")
        self.lbl_pending.setText(f"Pending: {total_videos}")

        # Timing
        self._multi_start_time = time.time()
        self._batches_duration = None
        self._agg_duration = None
        self.lbl_batches_time.setText("Batches Time: —")
        self.lbl_agg_time.setText("Final Analysis Time: —")
        self.lbl_total_time.setText("Total Time: 0.0s (running...)")
        self.lbl_prompt_timing.setText("⏱ Multi-video processing...")
        self._live_timer.start()

        # Create multi-video worker
        self._multi_worker = MultiVideoWorker(
            video_folders=self._multi_video_folders,
            provider=provider,
            model=model_name,
            batch_prompt=self.txt_batch_prompt.toPlainText(),
            aggregation_prompt=self.txt_agg_prompt.toPlainText(),
            batch_size=self.spin_batch_size.value(),
            temperature=self.spin_temperature.value(),
            max_tokens=self.spin_max_tokens.value(),
            timeout=self.spin_timeout.value(),
            max_image_dim=max_dim,
            excel_path=self._multi_excel_path,
        )
        self._multi_thread = QThread()
        self._multi_worker.moveToThread(self._multi_thread)

        self._multi_thread.started.connect(self._multi_worker.run)
        self._multi_worker.video_started.connect(self._on_multi_video_started)
        self._multi_worker.video_completed.connect(self._on_multi_video_completed)
        self._multi_worker.video_failed.connect(self._on_multi_video_failed)
        self._multi_worker.batch_progress.connect(self._on_multi_batch_progress)
        self._multi_worker.all_videos_completed.connect(self._on_multi_all_completed)
        self._multi_worker.log_message.connect(self._log)
        self._multi_worker.all_videos_completed.connect(self._multi_thread.quit)

        # Disable UI
        self._set_multi_processing_ui(True)
        self._multi_thread.start()

    def _set_multi_processing_ui(self, processing: bool):
        """Enable/disable UI elements during multi-video processing."""
        self.btn_multi_start.setEnabled(not processing)
        self.btn_multi_cancel.setEnabled(processing)
        self.btn_select_multi_folder.setEnabled(not processing)
        self.btn_select_excel.setEnabled(not processing)
        self.btn_mode_single.setEnabled(not processing)
        self.btn_mode_multi.setEnabled(not processing)

        # Also disable shared config during processing
        self.spin_batch_size.setEnabled(not processing)
        self.combo_provider.setEnabled(not processing)
        self.edit_base_url.setEnabled(not processing)
        self.combo_model.setEnabled(not processing)
        self.btn_refresh_models.setEnabled(not processing)
        self.spin_temperature.setEnabled(not processing)
        self.spin_max_tokens.setEnabled(not processing)
        self.spin_timeout.setEnabled(not processing)
        self.spin_concurrency.setEnabled(not processing)
        self.spin_max_image_dim.setEnabled(not processing)

    def _on_multi_video_started(self, idx: int, video_name: str):
        item = self._multi_video_items.get(idx)
        if item:
            total = len(self._multi_video_folders)
            item.setText(f"● [{idx+1}/{total}] {video_name} — Processing...")
            item.setForeground(Qt.blue)
            self.batch_list.scrollToItem(item)

        self.lbl_progress_status.setText(
            f"Multi-video: Processing [{idx+1}/{len(self._multi_video_folders)}] {video_name}"
        )

    def _on_multi_batch_progress(self, video_idx: int, batch_id: int, total_batches: int):
        """Update the video item text to show current batch progress."""
        item = self._multi_video_items.get(video_idx)
        if item:
            video_name = self._multi_video_folders[video_idx]['name']
            total = len(self._multi_video_folders)
            item.setText(
                f"● [{video_idx+1}/{total}] {video_name} — Batch {batch_id}/{total_batches}"
            )

    def _on_multi_video_completed(self, idx: int, video_name: str, result: dict):
        item = self._multi_video_items.get(idx)
        total = len(self._multi_video_folders)

        timing = result.get('timing', {})
        total_dur = timing.get('total_duration_seconds', 0)

        if item:
            item.setText(
                f"✓ [{idx+1}/{total}] {video_name} — Done ({total_dur:.1f}s)"
            )
            item.setForeground(Qt.darkGreen)

        completed = sum(
            1 for i in range(total)
            if self._multi_video_items.get(i) and
            self._multi_video_items[i].text().startswith("✓")
        )
        failed = sum(
            1 for i in range(total)
            if self._multi_video_items.get(i) and
            self._multi_video_items[i].text().startswith("✗")
        )
        pending = total - completed - failed

        self.progress_bar.setValue(completed + failed)
        self.lbl_successful.setText(f"Completed: {completed}")
        self.lbl_failed.setText(f"Failed: {failed}")
        self.lbl_pending.setText(f"Pending: {pending}")

    def _on_multi_video_failed(self, idx: int, video_name: str, error: str):
        item = self._multi_video_items.get(idx)
        total = len(self._multi_video_folders)

        if item:
            short_err = error[:60] + "..." if len(error) > 60 else error
            item.setText(
                f"✗ [{idx+1}/{total}] {video_name} — Failed: {short_err}"
            )
            item.setForeground(Qt.red)

        completed = sum(
            1 for i in range(total)
            if self._multi_video_items.get(i) and
            self._multi_video_items[i].text().startswith("✓")
        )
        failed = sum(
            1 for i in range(total)
            if self._multi_video_items.get(i) and
            self._multi_video_items[i].text().startswith("✗")
        )
        pending = total - completed - failed

        self.progress_bar.setValue(completed + failed)
        self.lbl_successful.setText(f"Completed: {completed}")
        self.lbl_failed.setText(f"Failed: {failed}")
        self.lbl_pending.setText(f"Pending: {pending}")

    def _on_multi_all_completed(self, summary: dict):
        self._live_timer.stop()

        total_dur = summary.get('total_duration', 0)
        self._multi_start_time = None

        self._set_multi_processing_ui(False)
        self.btn_multi_open_excel.setEnabled(True)

        cancelled = summary.get('cancelled', False)
        status = "cancelled" if cancelled else "completed"
        self.lbl_progress_status.setText(
            f"Multi-video {status}: "
            f"{summary['completed']} completed, "
            f"{summary['failed']} failed"
        )
        self.lbl_total_time.setText(f"Total Time: {self._format_time_sec(total_dur)}")
        self.lbl_prompt_timing.setText(
            f"⏱ Multi-video {status} | Total: {self._format_time_sec(total_dur)}"
        )

        self._log(
            f"Multi-video {status}: "
            f"{summary['completed']}/{summary['total']} completed, "
            f"{summary['failed']} failed, "
            f"total time: {self._format_time_sec(total_dur)}"
        )

        self._multi_last_fallback = summary.get('fallback_excel_path')

        if not cancelled:
            saved_info = f"Results saved to:\n{self._multi_excel_path}"
            if self._multi_last_fallback:
                saved_info = (
                    f"⚠️ File chính bị khóa trong lúc xử lý, kết quả đã được lưu dự phòng vào:\n"
                    f"{self._multi_last_fallback}\n\n"
                    f"Đường dẫn file chính:\n{self._multi_excel_path}"
                )

            QMessageBox.information(
                self, "Multi-Video Complete",
                f"Processing completed!\n\n"
                f"  Videos processed: {summary['completed']}/{summary['total']}\n"
                f"  Failed: {summary['failed']}\n"
                f"  Total time: {self._format_time_sec(total_dur)}\n\n"
                f"{saved_info}"
            )

    def _cancel_multi_processing(self):
        if self._multi_worker:
            self._multi_worker.cancel()
            self._log("[Multi] Cancel requested — waiting for current video to finish...")
            self.btn_multi_cancel.setEnabled(False)

    def _open_excel_file(self):
        target = self._multi_excel_path
        if hasattr(self, '_multi_last_fallback') and self._multi_last_fallback and os.path.exists(self._multi_last_fallback):
            if not target or not os.path.exists(target):
                target = self._multi_last_fallback
        if target and os.path.exists(target):
            QDesktopServices.openUrl(QUrl.fromLocalFile(target))

    # ----- Logging -----

    def _log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        self.log_text.append(line)
        # Also log to file if available
        if hasattr(self, '_file_logger') and self._file_logger:
            self._file_logger.info(message)

    def _setup_file_logger(self, run_dir: str):
        """Setup file-based logging for the current run."""
        log_path = os.path.join(run_dir, "application.log")
        self._file_logger = logging.getLogger(f"run_{id(self)}")
        self._file_logger.setLevel(logging.INFO)
        # Remove existing handlers
        self._file_logger.handlers.clear()
        handler = logging.FileHandler(log_path, encoding='utf-8')
        handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        )
        self._file_logger.addHandler(handler)
        self._file_logger.info("Application started")
