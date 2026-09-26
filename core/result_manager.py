"""
Result persistence and run directory management.

Creates run directories, saves/loads batch JSON results,
checks batch statuses for resume, and manages final results.
"""
import json
import os
from datetime import datetime
from pathlib import Path


def create_run_directory(base_path: str) -> tuple[str, str]:
    """Create a new timestamped run directory.

    The directory is created under: <base_path>/analysis/run_YYYYMMDD_HHMMSS/

    Args:
        base_path: Parent directory (typically the frame folder's parent).

    Returns:
        (run_dir_path, run_id) tuple.
    """
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(base_path, "analysis", run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, run_id


def save_config(run_dir: str, config: dict):
    """Save the run configuration to config.json."""
    path = os.path.join(run_dir, "config.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def save_batch_result(run_dir: str, batch_id: int, result: dict):
    """Save a single batch result immediately after completion."""
    filename = f"batch_{batch_id:03d}.json"
    path = os.path.join(run_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def load_batch_result(run_dir: str, batch_id: int) -> dict | None:
    """Load a batch result file. Returns None if file doesn't exist."""
    filename = f"batch_{batch_id:03d}.json"
    path = os.path.join(run_dir, filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_batch_statuses(run_dir: str, total_batches: int) -> dict:
    """Check the status of all batches in a run.

    Returns:
        dict with keys 'successful', 'failed', 'pending',
        each containing a list of batch IDs (1-indexed).
    """
    successful = []
    failed = []
    pending = []

    for i in range(1, total_batches + 1):
        result = load_batch_result(run_dir, i)
        if result is None:
            pending.append(i)
        elif result.get('processing', {}).get('status') == 'success':
            successful.append(i)
        else:
            failed.append(i)

    return {'successful': successful, 'failed': failed, 'pending': pending}


def find_existing_runs(base_path: str) -> list[dict]:
    """Find existing run directories under <base_path>/analysis/.

    Returns:
        List of dicts with 'run_id', 'run_dir', 'config' keys,
        sorted newest-first.
    """
    analysis_dir = os.path.join(base_path, "analysis")
    if not os.path.isdir(analysis_dir):
        return []

    runs = []
    for entry in sorted(os.listdir(analysis_dir), reverse=True):
        run_dir = os.path.join(analysis_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        if not entry.startswith("run_"):
            continue

        config = None
        config_path = os.path.join(run_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        runs.append({
            'run_id': entry,
            'run_dir': run_dir,
            'config': config,
        })

    return runs


def save_final_result(run_dir: str, result: dict):
    """Save the final aggregated result."""
    path = os.path.join(run_dir, "final_result.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def load_all_successful_batches(run_dir: str, total_batches: int) -> list[dict]:
    """Load all successful batch results in order."""
    results = []
    for i in range(1, total_batches + 1):
        result = load_batch_result(run_dir, i)
        if result and result.get('processing', {}).get('status') == 'success':
            results.append(result)
    return results
