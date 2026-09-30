"""
Ollama provider — uses Ollama's native HTTP API.

Default endpoint: http://localhost:11434
"""
import requests

from providers.base import AIProvider


class OllamaProvider(AIProvider):
    """AI provider for Ollama's local server."""

    @property
    def provider_name(self) -> str:
        return "ollama"

    def list_models(self) -> list[str]:
        resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m['name'] for m in data.get('models', [])]

    def analyze_images(
        self,
        images: list[dict],
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 300,
    ) -> str:
        # Ollama expects images as a list of base64 strings in the message
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img['base64'] for img in images],
                }
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=timeout,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            try:
                err_data = resp.json()
                err_msg = err_data.get('error', '')
                if err_msg:
                    raise requests.HTTPError(
                        f"Ollama ({resp.status_code}): {err_msg}",
                        response=resp,
                    ) from e
            except requests.HTTPError:
                raise
            except Exception:
                pass
            raise

        data = resp.json()
        msg = data.get('message', {})
        content = msg.get('content')
        if not content:
            content = msg.get('thinking') or ""
        return content

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
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=timeout,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            try:
                err_data = resp.json()
                err_msg = err_data.get('error', '')
                if err_msg:
                    raise requests.HTTPError(
                        f"Ollama ({resp.status_code}): {err_msg}",
                        response=resp,
                    ) from e
            except requests.HTTPError:
                raise
            except Exception:
                pass
            raise

        data = resp.json()
        msg = data.get('message', {})
        content = msg.get('content')
        if not content:
            content = msg.get('thinking') or ""
        return content
