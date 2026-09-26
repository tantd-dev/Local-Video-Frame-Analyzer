"""
LM Studio provider — uses the OpenAI-compatible API.

Default endpoint: http://localhost:1234/v1
"""
import requests

from providers.base import AIProvider


class LMStudioProvider(AIProvider):
    """AI provider for LM Studio's OpenAI-compatible local server."""

    @property
    def provider_name(self) -> str:
        return "lmstudio"

    def list_models(self) -> list[str]:
        resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m['id'] for m in data.get('data', [])]

    def analyze_images(
        self,
        images: list[dict],
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 300,
    ) -> str:
        # Build multimodal content array: text first, then images
        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img['media_type']};base64,{img['base64']}"
                }
            })

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content']

    def generate_text(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 300,
    ) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data['choices'][0]['message']['content']
