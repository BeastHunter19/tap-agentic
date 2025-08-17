"""Model factory helpers.

This module exposes a small factory that reads the configured API key from
the environment (using the `AI_API_KEY_FILE` path) and initializes the
LangChain chat model. The factory is synchronous so it can be called at
startup; the actual network activity occurs inside the model implementation
when the model is first used.
"""

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def _read_api_key_from_file(path: str) -> str:
    """Read and return a trimmed API key from the provided file path.

    Args:
        path: Filesystem path to the file that contains the API key.

    Raises:
        FileNotFoundError: If the file does not exist.

    Returns:
        The API key as a stripped string.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().strip()


def create_model() -> BaseChatModel:
    """Create and return the chat model instance.

    The function expects the environment variable `AI_API_KEY_FILE` to point
    to a file containing the API key.

    Returns:
        An initialized `BaseChatModel` instance ready for use.
    """
    api_key_file = os.getenv("AI_API_KEY_FILE")
    if not api_key_file:
        raise RuntimeError("AI_API_KEY_FILE environment variable is not set")

    api_key = _read_api_key_from_file(api_key_file)

    return init_chat_model("google_genai:gemini-2.5-flash", api_key=api_key)
