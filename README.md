# Local Video Frame Analyzer

Internal GUI tool for testing local Vision/Video AI models.  
Handles: **frame batching → AI analysis → JSON results → final aggregation**.

> This tool starts **after** frame extraction. It is read-only with respect to source frames.

---

## Requirements

- **Python 3.11+**
- **Windows 11** (primary target; should work on other OSes with PySide6 support)

## Installation

```bash
cd LocalVideoFrameAnalyzer
pip install -r requirements.txt
```

Dependencies:
- `PySide6` — Desktop GUI
- `requests` — HTTP API calls
- `Pillow` — Optional image resizing

## Running

```bash
python app.py
```

---

## Provider Setup

### LM Studio

1. Install [LM Studio](https://lmstudio.ai/)
2. Load a **vision-capable** model (e.g., LLaVA, Qwen2-VL, etc.)
3. Start the local server (default: `http://localhost:1234`)
4. In the app:
   - Provider → **LM Studio**
   - Base URL → `http://localhost:1234` (default)
   - Click **Refresh** to list loaded models
   - Select the model

### Ollama

1. Install [Ollama](https://ollama.com/)
2. Pull a vision model: `ollama pull llava` (or any multimodal model)
3. Ollama runs automatically (default: `http://localhost:11434`)
4. In the app:
   - Provider → **Ollama**
   - Base URL → `http://localhost:11434` (default)
   - Click **Refresh** to list available models
   - Select the model

---

## Workflow

### 1. Select Frame Folder

Click **Select Frame Folder** and choose a directory containing extracted frames.

Expected filename format:
```
frame_0025_000215.jpg
       ^^^^  ^^^^^^
       index timestamp (seconds)
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`

Frames are automatically sorted by timestamp (chronological order).

### 2. Configure Batches

Set **Frames per batch** (default: 10).

The display shows batch distribution:
```
Total frames: 98
Batch size: 10
Total batches: 10
  9 batches × 10 frames
  1 batch × 8 frames
```

### 3. Edit Prompt

The **Batch Analysis Prompt** tab contains the prompt sent with each batch.  
Edit it to match your analysis needs. The same prompt is used for every batch.

### 4. Configure Provider & Model

Select provider, refresh models, pick one.

### 5. Run Processing

| Button | Behavior |
|--------|----------|
| **Send All** | Creates a new run, processes all batches |
| **Resume** | Selects an existing run, skips successful batches, retries failed/pending |
| **Force Re-run** | Creates a new run, processes everything (ignores previous results) |
| **Cancel** | Stops after the current batch finishes |

### 6. Generate Final Analysis & Execution Timing

After batch processing completes, click **Generate Final Analysis**.  
This reads all successful batch results and sends them (text-only, no images) to the model for aggregation.

#### Timing Metrics & Final Result Bar
The UI continuously tracks and displays:
- **Batches Time**: Total execution duration of all batches.
- **Final Analysis Time**: Duration of the aggregation step.
- **Total Time**: Combined end-to-end processing time (Batches + Final Analysis).

These metrics and the status of `final_result.json` are displayed:
1. In the **Progress** panel below batch statistics.
2. Directly on the **Prompt Tabs bar** (alongside "Batch Analysis Prompt" and "Aggregation Prompt") with quick action buttons:
   - **View JSON**: View formatted final result directly in a dialog (with Copy & External Viewer support).
   - **Open Folder**: Open the run directory in Windows Explorer.

---

## Output Structure

```
analysis/
└── run_20260926_110000/
    ├── config.json          # Run configuration snapshot
    ├── batch_001.json       # Individual batch results
    ├── batch_002.json
    ├── ...
    ├── final_result.json    # Aggregated final analysis
    └── application.log      # Runtime log
```

The `analysis/` directory is created as a sibling to the frame folder.

### Batch JSON Format

```json
{
  "run_id": "run_20260926_110000",
  "batch_id": 1,
  "total_batches": 10,
  "frames": [
    {"filename": "frame_0001_000000.jpg", "timestamp_seconds": 0},
    {"filename": "frame_0002_000009.jpg", "timestamp_seconds": 9}
  ],
  "provider": "lmstudio",
  "model": "MODEL_NAME",
  "prompt": "...",
  "result": { ... },
  "raw_response": "...",
  "processing": {
    "started_at": "...",
    "finished_at": "...",
    "duration_seconds": 32.5,
    "status": "success"
  }
}
```

### Failed Batch

```json
{
  "processing": {"status": "failed"},
  "error_type": "invalid_json",
  "raw_response": "...",
  "retry_attempted": true
}
```

---

## Resume Support

If processing is interrupted:

1. Restart the app
2. Select the same frame folder
3. Pick the previous run from the **Run Selection** dropdown
4. Click **Resume**

The app detects successful batches and only retries failed/pending ones.

---

## Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Temperature | 0.0 | Sampling temperature |
| Max tokens | 4096 | Maximum output tokens |
| Timeout | 300s | HTTP request timeout per batch |
| Concurrency | 1 | Sequential processing (VRAM-safe) |
| Max image dim | 1280 | Resize images before sending (0 = no resize) |

---

## Error Handling

- Provider unavailable → error in GUI, no crash
- Invalid JSON → automatic retry once with correction prompt
- Batch failure → continues to next batch
- Raw responses are always preserved in batch JSON

---

## Limitations

- Concurrency > 1 is configured in the UI but processing is currently sequential
- Hierarchical aggregation (for very large batch sets) is not yet implemented
- No automatic context-size limit enforcement — relies on model/provider limits
- Frame filename must match `frame_NNNN_TTTTTT.ext` pattern exactly
