"""
Abstract base class for AI providers.

All provider implementations (LM Studio, Ollama, future providers)
must inherit from AIProvider and implement its interface.
"""
from abc import ABC, abstractmethod


class AIProvider(ABC):
    """Base class for AI model providers."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return a short identifier for this provider (e.g. 'lmstudio', 'ollama')."""
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return a list of available model names/IDs from the provider.

        Raises:
            requests.RequestException: If the provider is unreachable.
        """
        ...

    @abstractmethod
    def analyze_images(
        self,
        images: list[dict],
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 300,
    ) -> str:
        """Send images with a prompt to the vision model.

        Args:
            images: List of dicts, each with 'base64' (str) and 'media_type' (str).
            prompt: The text prompt to send alongside images.
            model: The model name/ID to use.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            timeout: Request timeout in seconds.

        Returns:
            The raw text response from the model.

        Raises:
            requests.RequestException: On HTTP/connection errors.
        """
        ...

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 300,
    ) -> str:
        """Send a text-only prompt to the model (used for aggregation).

        Args:
            prompt: The text prompt.
            model: The model name/ID to use.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            timeout: Request timeout in seconds.

        Returns:
            The raw text response from the model.

        Raises:
            requests.RequestException: On HTTP/connection errors.
        """
        ...
