"""
JSON extraction and validation utilities.

Handles AI responses that may contain markdown fences, extra text,
or other non-JSON content wrapping the actual JSON payload.
Also provides repair for duplicated text and deduplication of list fields
commonly returned by local AI models.
"""
import json
import re


# Fields that contain lists of strings and should be deduplicated.
_LIST_FIELDS = {
    "scenes",
    "locations",
    "objects",
    "actions",
    "visible_text",
    "important_events",
    "uncertainties",
    "chronological_scenes",
    "important_locations",
    "important_objects",
    "actions_and_events",
    "major_changes",
}


def _normalize_text(text: str) -> str:
    """Normalize whitespace and case for deduplication comparison."""
    return " ".join(text.lower().split())


def _deduplicate_list(items: list) -> list:
    """Remove duplicate text entries from a list while preserving order.

    Only string items are checked for duplicates; non-string items
    (e.g. nested dicts) are always kept.
    """
    result = []
    seen = set()

    for item in items:
        if isinstance(item, str):
            key = _normalize_text(item)
            if key in seen:
                continue
            seen.add(key)
        result.append(item)

    return result


def clean_json_data(data):
    """Recursively deduplicate list fields in parsed JSON data.

    Walks the entire JSON tree and deduplicates any list field whose
    key is in _LIST_FIELDS.  Non-list values and unknown keys are
    left untouched.

    Args:
        data: Parsed JSON (dict, list, or scalar).

    Returns:
        The cleaned data (modified in place for dicts).
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key in _LIST_FIELDS and isinstance(value, list):
                data[key] = _deduplicate_list(value)
            else:
                data[key] = clean_json_data(value)
        return data

    if isinstance(data, list):
        return [
            clean_json_data(item) if isinstance(item, (dict, list)) else item
            for item in data
        ]

    return data


def repair_duplicated_text(text: str) -> str:
    """Attempt to repair raw AI responses with duplicated text blocks.

    Some local models produce output where a large chunk of the JSON
    response is repeated verbatim (e.g. the model "stutters" and
    emits the same block twice).  This makes the overall response
    invalid JSON even though each individual copy is fine.

    Strategy:
    1. Find the outermost ``{ ... }`` span.
    2. If that span is already valid JSON, return it as-is.
    3. Otherwise, try progressively removing a duplicated suffix
       by looking for the longest repeated tail that, once removed,
       yields valid JSON.
    4. As a last resort, try to find a *complete* valid JSON object
       starting from each ``{`` in the text.

    Returns:
        The (possibly repaired) text.  If no repair is possible the
        original text is returned unchanged.
    """
    text = text.strip()

    # Quick exit: already valid
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Locate outermost braces
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return text

    candidate = text[first_brace:last_brace + 1]

    # Already valid after trimming surrounding text?
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass

    # --- Heuristic: detect a repeated JSON block ----------------------
    # Look for two copies of a large-ish substring.  We try to split
    # the candidate at every ``}\s*{`` boundary (which is characteristic
    # of two JSON objects concatenated together) and keep the first
    # valid one.
    split_points = [m.start() for m in re.finditer(r'\}\s*\{', candidate)]
    for sp in split_points:
        fragment = candidate[:sp + 1]  # up to and including the '}'
        try:
            json.loads(fragment)
            return fragment
        except json.JSONDecodeError:
            continue

    # --- Heuristic: progressive brace-matching from the start ---------
    # Walk from the first '{' and track brace depth.  Every time depth
    # returns to 0 we have a complete object — try to parse it.
    depth = 0
    for i, ch in enumerate(candidate):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                fragment = candidate[:i + 1]
                try:
                    json.loads(fragment)
                    return fragment
                except json.JSONDecodeError:
                    break  # first balanced span didn't parse, give up

    return text


def _extract_from_fences(text: str) -> str | None:
    """Extract JSON from markdown code fences.

    Tries multiple fence patterns with flexible whitespace handling.
    Prefers ```json fences over bare ``` fences.  Within each fence
    type, returns the first block that parses as valid JSON.
    """
    # Patterns ordered by specificity: ```json first, then bare ```.
    # Each pattern allows optional whitespace (including no newline)
    # between the fence marker and the content, and allows the
    # closing ``` to appear on the same line or after whitespace.
    fence_patterns = [
        r'```json\s*([\s\S]*?)\s*```',      # ```json ... ```
        r'```\s*([\s\S]*?)\s*```',           # ``` ... ```
    ]
    for pattern in fence_patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            candidate = match.group(1).strip()
            if not candidate:
                continue
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                # This fence block didn't contain valid JSON;
                # try the next match for this pattern.
                continue
    return None


def _extract_by_brace_depth(text: str) -> str | None:
    """Find the first complete JSON object or array via brace/bracket depth tracking.

    Scans through the text character-by-character, tracking brace ``{}``
    and bracket ``[]`` depth while respecting JSON string literals
    (skipping escaped characters inside strings).  Returns the first
    balanced span that parses as valid JSON.

    This is more reliable than ``text[text.find('{'):text.rfind('}')+1]``
    because it correctly handles JSON embedded between other text that
    may also contain braces.
    """
    # Find all positions where a JSON value could start
    openers = []
    for i, ch in enumerate(text):
        if ch in ('{', '['):
            openers.append(i)

    for start in openers:
        open_char = text[start]
        close_char = '}' if open_char == '{' else ']'
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            ch = text[i]

            if escape_next:
                escape_next = False
                continue

            if in_string:
                if ch == '\\':
                    escape_next = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break  # This opener failed; try the next one.

    return None


def extract_json_from_text(text: str) -> str | None:
    """Try to extract a JSON object from text that may contain extra content.

    Handles:
    - Pure JSON
    - JSON wrapped in ```json ... ``` or ``` ... ``` fences
    - JSON embedded in surrounding conversational text
    - JSON arrays (``[...]``) as well as objects (``{...}``)
    - Duplicated / repeated JSON blocks (via repair)

    The extraction pipeline:
    1. Direct parse of the full text.
    2. Markdown code-fence extraction (prefers ```json).
    3. Brace/bracket-depth balanced extraction (first complete
       JSON value in the text).
    4. Duplicated-text repair as a last resort.

    Returns:
        The JSON string if found, or None.
    """
    text = text.strip()

    # 1. Try direct parse — the response is already pure JSON.
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # 2. Try markdown code fences (handles the very common
    #    "Here is the result:\n```json\n{...}\n```\nDone." pattern).
    result = _extract_from_fences(text)
    if result is not None:
        return result

    # 3. Depth-balanced extraction — walk the text to find the first
    #    complete JSON object ``{...}`` or array ``[...]``.
    result = _extract_by_brace_depth(text)
    if result is not None:
        return result

    # 4. Try to repair duplicated text blocks (model "stuttering").
    repaired = repair_duplicated_text(text)
    if repaired != text:
        try:
            json.loads(repaired)
            return repaired
        except json.JSONDecodeError:
            pass

    return None


def safe_parse_json(text: str) -> tuple[dict | None, str | None]:
    """Try to parse a JSON object from AI response text.

    Applies text repair (for duplicated blocks) and extraction,
    then deduplicates known list fields in the result.

    Returns:
        (parsed_dict, None) on success, or (None, error_message) on failure.
    """
    # First try repair in case of duplicated raw text
    repaired = repair_duplicated_text(text)

    extracted = extract_json_from_text(repaired)
    if extracted is None:
        return None, "Could not find valid JSON in response"
    try:
        result = json.loads(extracted)
        if not isinstance(result, dict):
            return None, f"Expected JSON object, got {type(result).__name__}"
        # Deduplicate list fields before returning
        result = clean_json_data(result)
        return result, None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"
