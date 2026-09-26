"""
Image loading and encoding utilities.

Handles loading images from disk, optional resizing, and base64 encoding
for transmission to AI providers.
"""
import base64
import io
from pathlib import Path

from PIL import Image


MEDIA_TYPE_MAP = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.webp': 'image/webp',
}

# PIL save format names
SAVE_FORMAT_MAP = {
    '.jpg': 'JPEG',
    '.jpeg': 'JPEG',
    '.png': 'PNG',
    '.webp': 'WEBP',
}


def load_and_encode_image(
    filepath: str,
    max_dimension: int | None = None,
) -> dict:
    """Load an image file, optionally resize it, and return base64-encoded data.

    Args:
        filepath: Path to the image file.
        max_dimension: If set and > 0, resize images so the largest dimension
                       does not exceed this value. Original file is never modified.

    Returns:
        dict with keys: 'base64', 'media_type'
    """
    ext = Path(filepath).suffix.lower()
    media_type = MEDIA_TYPE_MAP.get(ext, 'image/jpeg')
    save_format = SAVE_FORMAT_MAP.get(ext, 'JPEG')

    if max_dimension and max_dimension > 0:
        img = Image.open(filepath)
        if max(img.size) > max_dimension:
            img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
            # Convert RGBA to RGB for JPEG
            if save_format == 'JPEG' and img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            buffer = io.BytesIO()
            quality = 85 if save_format == 'JPEG' else None
            save_kwargs = {'format': save_format}
            if quality is not None:
                save_kwargs['quality'] = quality
            img.save(buffer, **save_kwargs)
            b64 = base64.b64encode(buffer.getvalue()).decode('ascii')
            return {'base64': b64, 'media_type': media_type}

    # No resize needed — read raw bytes
    with open(filepath, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')

    return {'base64': b64, 'media_type': media_type}
