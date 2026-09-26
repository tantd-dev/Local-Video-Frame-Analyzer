"""
Aggregation of batch results into a final analysis.

Reads all successful batch JSONs, builds a combined prompt,
sends it to the AI model (text-only, no images), and produces
a final unified analysis.
"""
import json


DEFAULT_AGGREGATION_PROMPT = """You are analyzing a complete video using structured analysis results \
from multiple chronological batches.

The batches are ordered by timestamp.

Based ONLY on the provided batch results, produce a final analysis.

Return ONLY valid JSON.

Use this structure:

{
  "overall_summary": "",
  "likely_location": "",
  "chronological_scenes": [],
  "important_locations": [],
  "important_objects": [],
  "actions_and_events": [],
  "visible_text": [],
  "major_changes": [],
  "uncertainties": []
}

Rules:
- Do not invent information.
- Distinguish observed information from inference.
- If different batches contain conflicting information, preserve the uncertainty.
- Use chronological information from the batches.
- Return valid JSON only."""


def build_aggregation_input(batch_results: list[dict], aggregation_prompt: str) -> str:
    """Build the full prompt for aggregation.

    Combines the aggregation prompt with all successful batch results
    in chronological order.

    Args:
        batch_results: List of successful batch result dicts, in order.
        aggregation_prompt: The aggregation prompt template.

    Returns:
        The full prompt string ready to send to the model.
    """
    parts = []

    for br in batch_results:
        batch_id = br.get('batch_id', '?')
        frames = br.get('frames', [])
        if frames:
            ts_start = frames[0].get('timestamp_seconds', '?')
            ts_end = frames[-1].get('timestamp_seconds', '?')
            ts_info = f"timestamps {ts_start}s – {ts_end}s"
        else:
            ts_info = "timestamps unknown"

        result = br.get('result', {})
        parts.append(
            f"=== Batch {batch_id} ({ts_info}) ===\n"
            f"{json.dumps(result, indent=2, ensure_ascii=False)}"
        )

    batch_text = "\n\n".join(parts)

    return f"{aggregation_prompt}\n\n--- Batch Results ---\n\n{batch_text}"


def estimate_token_count(text: str) -> int:
    """Rough estimate of token count (approximately 4 characters per token).

    This is a heuristic — actual tokenization varies by model.
    """
    return len(text) // 4
