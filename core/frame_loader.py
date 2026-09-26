"""
Frame discovery and loading.

Scans a directory for frame images, parses their filenames to extract
frame indices and timestamps, and returns them sorted chronologically.
"""
from dataclasses import dataclass
from pathlib import Path

from utils.timestamp import parse_frame_filename, is_image_file


@dataclass
class FrameInfo:
    """Information about a single extracted frame."""
    filename: str
    filepath: str
    index: int
    timestamp_seconds: int


def discover_frames(folder_path: str) -> list[FrameInfo]:
    """Scan a folder for frame images and return them sorted by timestamp.

    Only files matching the frame_NNNN_TTTTTT.ext naming convention with
    supported image extensions (.jpg, .jpeg, .png, .webp) are included.

    Args:
        folder_path: Path to the directory containing frame images.

    Returns:
        List of FrameInfo objects sorted by timestamp_seconds (chronological).

    Raises:
        ValueError: If the path is not a valid directory.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f"Not a directory: {folder_path}")

    frames = []
    for f in folder.iterdir():
        if not f.is_file():
            continue
        if not is_image_file(f.name):
            continue
        parsed = parse_frame_filename(f.name)
        if parsed is None:
            continue
        idx, ts = parsed
        frames.append(FrameInfo(
            filename=f.name,
            filepath=str(f.resolve()),
            index=idx,
            timestamp_seconds=ts,
        ))

    # Sort chronologically by timestamp, then by index for ties
    frames.sort(key=lambda fr: (fr.timestamp_seconds, fr.index))
    return frames


def get_frame_stats(frames: list[FrameInfo]) -> dict:
    """Compute summary statistics for a list of frames.

    Returns:
        dict with keys: total, first_frame, last_frame,
        duration_seconds, avg_interval_seconds
    """
    if not frames:
        return {
            'total': 0,
            'first_frame': None,
            'last_frame': None,
            'duration_seconds': 0,
            'avg_interval_seconds': 0.0,
        }

    first = frames[0]
    last = frames[-1]
    duration = last.timestamp_seconds - first.timestamp_seconds

    avg_interval = 0.0
    if len(frames) > 1:
        avg_interval = duration / (len(frames) - 1)

    return {
        'total': len(frames),
        'first_frame': first.filename,
        'last_frame': last.filename,
        'duration_seconds': duration,
        'avg_interval_seconds': round(avg_interval, 2),
    }
