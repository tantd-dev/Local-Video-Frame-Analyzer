"""
Main application window.

Contains all GUI sections (frame folder, batch config, provider,
prompt, progress) and worker threads for non-blocking AI processing.
"""
import json
import logging
import os
import traceback
from datetime import datetime

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QTextEdit, QProgressBar, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QSplitter, QTabWidget, QSizePolicy,
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

    @Slot()
    def run(self):
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
                'final_analysis': parsed,
                'raw_response': raw_response,
            }

            save_final_result(self.run_dir, final_result)
            self.log_message.emit("Final result saved.")
            self.completed.emit(final_result)

        except Exception as e:
            self.failed.emit(f"Aggregation error: {type(e).__name__}: {e}")


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

        self._setup_ui()
        self._connect_signals()

    # ----- UI Construction -----

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(6)

        # --- Frame Folder ---
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

        main_layout.addWidget(frame_group)

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

        # Run Selection
        run_group = QGroupBox("Run Selection (for Resume)")
        run_layout = QHBoxLayout(run_group)
        self.combo_runs = QComboBox()
        self.combo_runs.setMinimumWidth(200)
        self.btn_refresh_runs = QPushButton("Refresh")
        self.btn_refresh_runs.setFixedWidth(70)
        run_layout.addWidget(self.combo_runs, 1)
        run_layout.addWidget(self.btn_refresh_runs)
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

        main_layout.addWidget(prompt_tabs)

        # --- Action Buttons ---
        btn_layout = QHBoxLayout()
        self.btn_send_all = QPushButton("Send All")
        self.btn_resume = QPushButton("Resume")
        self.btn_force_rerun = QPushButton("Force Re-run")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_aggregate = QPushButton("Generate Final Analysis")
        self.btn_aggregate.setEnabled(False)

        btn_layout.addWidget(self.btn_send_all)
        btn_layout.addWidget(self.btn_resume)
        btn_layout.addWidget(self.btn_force_rerun)
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_aggregate)

        main_layout.addLayout(btn_layout)

    # ----- Signal Connections -----

    def _connect_signals(self):
        self.btn_select_folder.clicked.connect(self._select_folder)
        self.spin_batch_size.valueChanged.connect(self._update_batch_info)
        self.combo_provider.currentIndexChanged.connect(self._on_provider_changed)
        self.btn_refresh_models.clicked.connect(self._refresh_models)
        self.btn_refresh_runs.clicked.connect(self._refresh_runs)
        self.btn_send_all.clicked.connect(lambda: self._start_processing('send_all'))
        self.btn_resume.clicked.connect(lambda: self._start_processing('resume'))
        self.btn_force_rerun.clicked.connect(lambda: self._start_processing('force'))
        self.btn_cancel.clicked.connect(self._cancel_processing)
        self.btn_aggregate.clicked.connect(self._start_aggregation)

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

    # ----- Run Selection -----

    def _refresh_runs(self):
        self.combo_runs.clear()
        if not self._frame_folder:
            return

        base = os.path.dirname(self._frame_folder)
        runs = find_existing_runs(base)
        for r in runs:
            label = r['run_id']
            if r['config']:
                batch_size = r['config'].get('batch_size', '?')
                model = r['config'].get('model', '?')
                label += f"  (batch={batch_size}, model={model})"
            self.combo_runs.addItem(label, r)

    def _get_selected_run(self) -> dict | None:
        idx = self.combo_runs.currentIndex()
        if idx < 0:
            return None
        return self.combo_runs.itemData(idx)

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
        self._set_processing_ui(False)
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
        self.btn_aggregate.setEnabled(True)
        self.lbl_progress_status.setText("Aggregation completed ✓")
        self._log("Aggregation completed successfully.")
        QMessageBox.information(
            self, "Aggregation Complete",
            f"Final result saved to:\n{self._current_run_dir}\\final_result.json"
        )

    def _on_aggregation_failed(self, error: str):
        self.btn_aggregate.setEnabled(True)
        self.lbl_progress_status.setText("Aggregation failed ✗")
        self._log(f"Aggregation failed: {error}")
        QMessageBox.warning(self, "Aggregation Failed", error)

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
