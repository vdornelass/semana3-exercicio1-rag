import os
from dotenv import load_dotenv

load_dotenv()

_client = None


def _is_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def run_config() -> dict:
    """Retorna config com callback do Langfuse se as chaves estiverem no .env."""
    global _client
    if not _is_enabled():
        return {}

    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    _client = get_client()
    return {"callbacks": [CallbackHandler()]}


def flush() -> None:
    """Envia traces pendentes ao Langfuse (no-op se não estiver configurado)."""
    if _client is not None:
        _client.flush()
