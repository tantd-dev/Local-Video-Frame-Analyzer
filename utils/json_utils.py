"""
JSON extraction and validation utilities.

Handles AI responses that may contain markdown fences, extra text,
or other non-JSON content wrapping the actual JSON payload.
"""
import json
import re


def extract_json_from_text(text: str) -> str | None:
    """Try to extract a JSON object from text that may contain extra content.

    Handles:
    - Pure JSON
    - JSON wrapped in ```json ... ``` fences
    - JSON wrapped in ``` ... ``` fences
    - JSON embedded in surrounding text

    Returns:
        The JSON string if found, or None.
    """
    text = text.strip()

    # 1. Try direct parse
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # 2. Try markdown code fences
    fence_patterns = [
        r'```json\s*\n(.*?)\n\s*```',
        r'```\s*\n(.*?)\n\s*```',
    ]
    for pattern in fence_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

    # 3. Try to find a JSON object by matching outermost braces
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    return None


def safe_parse_json(text: str) -> tuple[dict | None, str | None]:
    """Try to parse a JSON object from AI response text.

    Returns:
        (parsed_dict, None) on success, or (None, error_message) on failure.
    """
    extracted = extract_json_from_text(text)
    if extracted is None:
        return None, "Could not find valid JSON in response"
    try:
        result = json.loads(extracted)
        if not isinstance(result, dict):
            return None, f"Expected JSON object, got {type(result).__name__}"
        return result, None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
