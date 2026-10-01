"""`python -m geo_auditor.llm` - show which free providers are active.
`python -m geo_auditor.llm --probe` - send one tiny request to each.
`python -m geo_auditor.llm "your prompt"` - answer via the failover pool."""
import argparse

from . import PROVIDERS, FreeLLMPool
from .pool import call_provider


def main() -> None:
    parser = argparse.ArgumentParser(description="Free LLM provider pool")
    parser.add_argument("prompt", nargs="?", help="Prompt to answer through the pool")
    parser.add_argument("--probe", action="store_true", help="Call every active provider once")
    args = parser.parse_args()

    if args.prompt:
        resp = FreeLLMPool().complete(args.prompt)
        print(f"[{resp.provider_name} / {resp.model}]" if resp.ok else f"ERROR: {resp.error}")
        if resp.ok:
            print(resp.text)
        return

    active = 0
    for p in PROVIDERS:
        status = "active" if p.configured else f"inactive - set {p.key_env}" + (
            "".join(f" + {e}" for e in p.extra_env))
        active += p.configured
        print(f"{p.id:12} {p.name:24} {p.model:42} {status}")
        if not p.configured:
            print(f"{'':12} free key: {p.signup_url}  ({p.free_tier})")
        elif args.probe:
            resp, _ = call_provider(p, [{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=20)
            print(f"{'':12} probe: {'OK ' + str(resp.latency_s) + 's' if resp.ok else resp.error}")
    print(f"\n{active}/{len(PROVIDERS)} providers active. Add keys to .env (see .env.example) to pool more free quota.")


if __name__ == "__main__":
    main()
