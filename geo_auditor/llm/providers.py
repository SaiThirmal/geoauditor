"""Registry of LLM APIs with permanent free tiers.

Source: https://github.com/mnfst/awesome-free-llm-apis (data.json, last
updated 2026-08-21). Every provider here is called through its
OpenAI-compatible /chat/completions endpoint, so one tiny HTTP client covers
all of them - no per-vendor SDKs.

Three providers need no key at all (LLM7.io, OVHcloud, Kilo Code), so the
pool works out of the box. Every other provider joins the pool as soon as its
API key env var is set; the more keys you add, the more free quota the pool
can draw on. See .env.example for the env var names and signup links.

Per-provider overrides (env):
  GEO_LLM_<ID>_MODEL   use a different model, e.g. GEO_LLM_GROQ_MODEL=qwen/qwen3.6-27b
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    base_url: str  # OpenAI-compatible base; "/chat/completions" is appended
    default_model: str
    signup_url: str
    free_tier: str  # human-readable limits, copied from the source list
    key_env: Optional[str] = None  # None = keyless provider
    key_optional: bool = False  # keyless, but a key raises limits if present
    extra_env: tuple[str, ...] = ()  # other env vars substituted into base_url

    @property
    def model(self) -> str:
        return os.environ.get(f"GEO_LLM_{self.id.upper()}_MODEL", self.default_model)

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.key_env) if self.key_env else None

    @property
    def configured(self) -> bool:
        """True when the provider can be called right now."""
        if any(not os.environ.get(e) for e in self.extra_env):
            return False
        if self.key_env is None or self.key_optional:
            return True
        return bool(self.api_key)

    @property
    def endpoint(self) -> str:
        url = self.base_url
        for env in self.extra_env:
            url = url.replace("{" + env + "}", os.environ.get(env, ""))
        return url.rstrip("/") + "/chat/completions"


# Ordered keyless first, then keyed providers roughly by free-tier generosity.
PROVIDERS: list[Provider] = [
    # --- keyless ---
    Provider("llm7", "LLM7.io", "https://api.llm7.io/v1", "gpt-oss:20b",
             "https://token.llm7.io", "10 RPM, 60 req/hr anonymous; a free token raises limits",
             key_env="LLM7_API_KEY", key_optional=True),
    Provider("ovh", "OVHcloud AI Endpoints", "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1", "gpt-oss-120b",
             "https://www.ovhcloud.com/en/public-cloud/ai-endpoints/catalog/", "2 RPM per IP per model, anonymous",
             key_env="OVH_AI_ENDPOINTS_ACCESS_TOKEN", key_optional=True),
    Provider("kilo", "Kilo Code", "https://api.kilo.ai/api/gateway", "kilo-auto/free",
             "https://app.kilo.ai/profile", "200 req/hr on :free models, no key needed",
             key_env="KILO_API_KEY", key_optional=True),
    # --- free key required ---
    Provider("groq", "Groq", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b",
             "https://console.groq.com/keys", "30 RPM, 1,000 RPD", key_env="GROQ_API_KEY"),
    Provider("gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash",
             "https://aistudio.google.com/app/apikey", "Free tier; prompts may be used to improve Google products",
             key_env="GEMINI_API_KEY"),
    Provider("mistral", "Mistral AI", "https://api.mistral.ai/v1", "mistral-small-2603",
             "https://console.mistral.ai/api-keys", "$10/month credits in free mode", key_env="MISTRAL_API_KEY"),
    Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "openai/gpt-oss-20b:free",
             "https://openrouter.ai/keys", "20 RPM, 50 RPD on :free models", key_env="OPENROUTER_API_KEY"),
    Provider("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1", "meta/llama-3.3-70b-instruct",
             "https://build.nvidia.com/explore/discover", "Free with Developer Program; rate-limited per model",
             key_env="NVIDIA_API_KEY"),
    Provider("cloudflare", "Cloudflare Workers AI",
             "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
             "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
             "https://dash.cloudflare.com/profile/api-tokens", "10,000 Neurons/day",
             key_env="CLOUDFLARE_API_TOKEN", extra_env=("CLOUDFLARE_ACCOUNT_ID",)),
    Provider("ollama", "Ollama Cloud", "https://ollama.com/v1", "gpt-oss:20b",
             "https://ollama.com/settings/keys", "Session limits reset every 5h, weekly limits every 7d",
             key_env="OLLAMA_API_KEY"),
    Provider("cohere", "Cohere", "https://api.cohere.ai/compatibility/v1", "command-a-03-2025",
             "https://dashboard.cohere.com/api-keys", "1,000 calls/month, non-commercial only",
             key_env="COHERE_API_KEY"),
    Provider("zai", "Z AI (Zhipu AI)", "https://open.bigmodel.cn/api/paas/v4", "glm-4.5-flash",
             "https://open.bigmodel.cn/usercenter/apikeys", "Permanent free flash models", key_env="ZAI_API_KEY"),
    Provider("huggingface", "Hugging Face", "https://router.huggingface.co/v1", "meta-llama/Llama-3.1-8B-Instruct",
             "https://huggingface.co/settings/tokens", "$0.10/month inference credits", key_env="HF_TOKEN"),
    Provider("modelscope", "ModelScope", "https://api-inference.modelscope.cn/v1", "Qwen/Qwen3.5-27B",
             "https://modelscope.cn/my/myaccesstoken", "Free API-Inference (needs Alibaba Cloud binding)",
             key_env="MODELSCOPE_API_KEY"),
    Provider("siliconflow", "SiliconFlow", "https://api.siliconflow.cn/v1", "Qwen/Qwen3-8B",
             "https://cloud.siliconflow.cn/account/ak", "Permanently free models (ID verification required)",
             key_env="SILICONFLOW_API_KEY"),
    Provider("aion", "Aion Labs", "https://api.aionlabs.ai/v1", "aion-labs/aion-3.0-mini",
             "https://www.aionlabs.ai/app/api-keys/", "15 RPM, 20K tokens/day", key_env="AION_API_KEY"),
]


def enabled_providers() -> list[Provider]:
    """Providers that can be called now, optionally narrowed by
    GEO_LLM_PROVIDERS=llm7,groq,... (comma-separated ids)."""
    allow = {p.strip() for p in os.environ.get("GEO_LLM_PROVIDERS", "").split(",") if p.strip()}
    return [p for p in PROVIDERS if p.configured and (not allow or p.id in allow)]
