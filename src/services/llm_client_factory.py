"""LLM Client Factory — builds the correct OpenAI-compatible client for a given provider.

Supported providers:
  - "open_ai"   : OpenAI cloud API (default)
  - "self_host" : Self-hosted vLLM endpoint (OpenAI-compatible)

Usage::

    client, model = build_llm_client("self_host")
    response = client.chat.completions.create(model=model, messages=[...])
"""
import os
from openai import OpenAI

# ---------------------------------------------------------------------------
# Provider constants
# ---------------------------------------------------------------------------

PROVIDER_OPEN_AI = "open_ai"
PROVIDER_SELF_HOST = "self_host"

_VALID_PROVIDERS = {PROVIDER_OPEN_AI, PROVIDER_SELF_HOST}


def build_llm_client(provider: str = PROVIDER_OPEN_AI) -> tuple[OpenAI, str]:
    """Return an (OpenAI client, model name) pair for the requested provider.

    Args:
        provider: Either ``"open_ai"`` (default) or ``"self_host"``.

    Returns:
        A tuple of ``(client, model_name)`` ready for
        ``client.chat.completions.create(model=model_name, ...)``.

    Raises:
        ValueError: If *provider* is not a recognised value.
    """
    if provider not in _VALID_PROVIDERS:
        raise ValueError(
            f"Unknown llm_provider '{provider}'. Valid values: {sorted(_VALID_PROVIDERS)}"
        )

    if provider == PROVIDER_SELF_HOST:
        base_url = os.getenv("SELF_HOST_LLM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("SELF_HOST_LLM_API_KEY", "EMPTY")
        model = os.getenv("SELF_HOST_LLM_MODEL", "Qwen3-14B-AWQ-INT4")
        client = OpenAI(base_url=base_url, api_key=api_key)
        return client, model

    # Default: open_ai
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
    client = OpenAI(api_key=api_key)
    return client, model
