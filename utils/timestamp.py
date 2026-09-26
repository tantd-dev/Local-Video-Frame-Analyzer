"""
Timestamp and frame filename parsing utilities.
"""
import re
from pathlib import Path


FRAME_PATTERN = re.compile(r'^frame_(\d+)_(\d+)$')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def parse_frame_filename(filename: str) -> tuple[int, int] | None:
    """Parse frame index and timestamp from filename.

    Expected format: frame_NNNN_TTTTTT.ext
    where NNNN is the frame index and TTTTTT is the timestamp in seconds.

    Returns:
        (frame_index, timestamp_seconds) or None if filename doesn't match.
    """
    stem = Path(filename).stem
    match = FRAME_PATTERN.match(stem)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def format_duration(seconds: int) -> str:
    """Format seconds into a human-readable HH:MM:SS or MM:SS string."""
    if seconds < 0:
        return "0:00"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def is_image_file(filename: str) -> bool:
    """Check if filename has a supported image extension."""
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS
