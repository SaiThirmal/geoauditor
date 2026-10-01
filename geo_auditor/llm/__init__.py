"""Free LLM access: every provider from awesome-free-llm-apis, pooled with
round-robin + failover so the free tiers stack instead of capping out."""
import os
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader (KEY=value lines) so API keys can live in a file
    without adding a dependency. Real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value[:1] in ("'", '"'):
            value = value[1:].split(value[0], 1)[0]
        else:
            value = value.split(" #", 1)[0].split("\t#", 1)[0].strip()
            if value.startswith("#"):
                value = ""
        if value:
            os.environ.setdefault(key.strip(), value)


_load_dotenv()

from .pool import FreeLLMPool, LLMResponse  # noqa: E402
from .providers import PROVIDERS, Provider, enabled_providers  # noqa: E402

__all__ = ["FreeLLMPool", "LLMResponse", "PROVIDERS", "Provider", "enabled_providers"]
