"""Live AI spot check (opt-in): ask real LLMs about the business and record
whether they mention it.

This is the "build next #1" item from the README, done the careful way:
- Answers come from free LLM APIs pooled in geo_auditor/llm (no paid keys;
  three providers need no key at all).
- Results are a separate panel, NOT part of the 0-100 score. LLM answers
  vary run to run, so they're shown as dated, raw evidence of what a few
  models said right now - never blended into the deterministic score.
- Caveat stated in the report: these are open-weight / API models answering
  from training data without web search, a proxy for ChatGPT/Perplexity
  answers rather than the same thing.
"""
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from ..llm import FreeLLMPool, LLMResponse

SYSTEM_PROMPT = "You are a helpful assistant answering a user's question the way an AI search engine would. Be concise and name specific businesses where relevant."


@dataclass
class SpotAnswer:
    provider_name: str
    model: str
    text: str
    mentioned: bool
    matched: list[str]
    excerpt: str


@dataclass
class SpotPrompt:
    kind: str  # "discovery" | "entity"
    prompt: str
    answers: list[SpotAnswer] = field(default_factory=list)

    @property
    def mention_count(self) -> int:
        return sum(a.mentioned for a in self.answers)


@dataclass
class SpotCheckResult:
    ran_at: str
    providers_available: int
    prompts: list[SpotPrompt] = field(default_factory=list)
    error: Optional[str] = None


def _match_terms(kind: str, business_name: str, url: str) -> list[str]:
    """Discovery prompts count a mention of the name or domain. Entity
    prompts already contain the name, so echoing it proves nothing - there
    only the correct domain counts as "the model knows who you are"."""
    domain = urlparse(url).netloc.lower().removeprefix("www.")
    terms = [domain] if kind == "entity" else [business_name.strip(), domain]
    return [t for t in dict.fromkeys(terms) if t and len(t) >= 3]


def _find(text: str, terms: list[str]) -> tuple[list[str], str]:
    matched, excerpt = [], ""
    for term in terms:
        m = re.search(re.escape(term), text, re.I)
        if m:
            matched.append(term)
            if not excerpt:
                s, e = max(0, m.start() - 120), min(len(text), m.end() + 120)
                excerpt = ("…" if s else "") + text[s:e].strip() + ("…" if e < len(text) else "")
    return matched, excerpt


def build_prompts(business_name: str, category: Optional[str], location: Optional[str]) -> list[tuple[str, str]]:
    prompts = []
    if category:
        where = f" in {location}" if location else ""
        prompts.append(("discovery", f"What are the best {category} businesses{where}? List specific names and websites."))
    where = f" in {location}" if location else ""
    prompts.append(("entity", f"What do you know about {business_name}{where}? What do they offer and what is their website?"))
    return prompts


def run(url: str, business_name: str, category: Optional[str], location: Optional[str],
        engines_per_prompt: int = 3, pool: Optional[FreeLLMPool] = None) -> SpotCheckResult:
    pool = pool or FreeLLMPool()
    result = SpotCheckResult(
        ran_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        providers_available=len(pool.providers),
    )
    if not pool.providers:
        result.error = "No free LLM providers are enabled (check GEO_LLM_PROVIDERS)."
        return result

    for kind, prompt in build_prompts(business_name, category, location):
        terms = _match_terms(kind, business_name, url)
        sp = SpotPrompt(kind=kind, prompt=prompt)
        responses: list[LLMResponse] = pool.ask_many(prompt, n=engines_per_prompt, system=SYSTEM_PROMPT)
        for r in responses:
            matched, excerpt = _find(r.text or "", terms)
            sp.answers.append(SpotAnswer(r.provider_name, r.model, r.text or "", bool(matched), matched, excerpt))
        result.prompts.append(sp)

    if not any(p.answers for p in result.prompts):
        result.error = "Every free provider failed or was rate-limited - try again in a minute, or add more free API keys (.env.example)."
    return result
