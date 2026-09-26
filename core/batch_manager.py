"""
Batch creation and management.

Divides a list of frames into sequential batches of a configurable size.
The remainder is placed into the final batch (never redistributed).
"""
import math
from dataclasses import dataclass, field

from core.frame_loader import FrameInfo


@dataclass
class Batch:
    """A single batch of frames to be analyzed together."""
    batch_id: int
    frames: list[FrameInfo] = field(default_factory=list)

    @property
    def time_range_str(self) -> str:
        if not self.frames:
            return "empty"
        start = self.frames[0].timestamp_seconds
        end = self.frames[-1].timestamp_seconds
        return f"{start}s – {end}s"


def create_batches(frames: list[FrameInfo], batch_size: int) -> list[Batch]:
    """Divide frames into sequential batches.

    The last batch may contain fewer frames than batch_size (the remainder).
    Frames are NOT redistributed — the remainder stays in the final batch.

    Args:
        frames: List of FrameInfo, must already be sorted chronologically.
        batch_size: Number of frames per batch.

    Returns:
        List of Batch objects.
    """
    if not frames or batch_size <= 0:
        return []

    total_batches = math.ceil(len(frames) / batch_size)
    batches = []

    for i in range(total_batches):
        start = i * batch_size
        end = min(start + batch_size, len(frames))
        batches.append(Batch(
            batch_id=i + 1,
            frames=frames[start:end],
        ))

    return batches


def get_batch_summary(total_frames: int, batch_size: int) -> dict:
    """Compute batch distribution information for display.

    Returns:
        dict with keys: total_frames, batch_size, total_batches, distribution
        where distribution is a list of {"count": N, "frames": M} dicts.
    """
    if total_frames == 0 or batch_size <= 0:
        return {
            'total_frames': total_frames,
            'batch_size': batch_size,
            'total_batches': 0,
            'distribution': [],
        }

    total_batches = math.ceil(total_frames / batch_size)
    full_batches = total_frames // batch_size
    remainder = total_frames % batch_size

    distribution = []
    if full_batches > 0:
        distribution.append({'count': full_batches, 'frames': batch_size})
    if remainder > 0:
        distribution.append({'count': 1, 'frames': remainder})

    return {
        'total_frames': total_frames,
        'batch_size': batch_size,
        'total_batches': total_batches,
        'distribution': distribution,
    }


def build_batch_prompt(batch: Batch, user_prompt: str) -> str:
    """Build the full prompt for a batch, prepending frame metadata.

    Args:
        batch: The batch to build a prompt for.
        user_prompt: The user's analysis prompt template.

    Returns:
        Combined prompt with frame metadata + user prompt.
    """
    lines = [f"Batch {batch.batch_id} — Frame sequence ({batch.time_range_str}):", ""]
    for i, frame in enumerate(batch.frames, 1):
        lines.append(
            f"  Image {i}: {frame.filename} (timestamp: {frame.timestamp_seconds}s)"
        )
    lines.append("")
    lines.append(user_prompt)
    return "\n".join(lines)
