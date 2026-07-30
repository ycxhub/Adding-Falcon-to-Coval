#!/usr/bin/env python3
# Copyright 2026 — TTFA multi-trial harness (Coval-aligned)
"""Run ≥10 perceived-TTFA trials against a provider adapter module.

Provider module must define:

    async def synthesize(text: str) -> tuple[bytes, float, float, int]:
        '''Return (pcm16_mono, t0_monotonic, first_chunk_monotonic, sample_rate).'''

API key and connection setup belong inside that module (read from env).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import random
import sys
from pathlib import Path

# Allow `python scripts/run_latency_test.py` from the skill root or scripts/.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ttfa import compute_ttfa, first_audible_offset_ms, summarize  # noqa: E402

# First 10 Coval tts-v1 prompts (A1–A10). Full list in reference.md.
DEFAULT_PROMPTS: list[tuple[str, str]] = [
    (
        "A1",
        "Your order #ORD-24589 shipped via UPS tracking 1Z999AA1234567890 "
        "and will arrive between 2:30 PM and 4:45 PM tomorrow.",
    ),
    (
        "A2",
        "There's a slight delay with your $347.89 order, "
        "but we expect it to ship by Friday afternoon.",
    ),
    (
        "A3",
        'Order status update: your items are now in "Quality Check" '
        "and should ship within 24-48 hours.",
    ),
    (
        "A4",
        "Hi Ms. Garcia, your appointment with Dr. Peterson is scheduled "
        "for Tuesday, March 5th at 10:30 AM.",
    ),
    (
        "A5",
        "I can reschedule you to either 11:15 AM on Wednesday or, "
        "let me check, 2:45 PM on Thursday.",
    ),
    (
        "A6",
        "Please arrive 10 minutes early and bring your insurance card and a photo ID.",
    ),
    (
        "A7",
        "Let's fix your Wi-Fi: unplug your router for exactly 30 seconds, "
        "then wait for the green light.",
    ),
    (
        "A8",
        "Your device is downloading update version 12.4.1, "
        "about 250 MB remaining, roughly 8 minutes left.",
    ),
    (
        "A9",
        "I'm seeing error code E-1047 here, which usually means, "
        "actually, let me guide you through the solution.",
    ),
    (
        "A10",
        "For security, please confirm the last 4 digits of the card "
        "ending in 8429 used for your recent purchase.",
    ),
]


def _load_synthesize(module_path: Path):
    spec = importlib.util.spec_from_file_location("provider_client", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load provider module: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "synthesize"):
        raise SystemExit(f"{module_path} must define async synthesize(text) -> ...")
    return mod.synthesize


async def _one_trial(synthesize, text: str) -> dict:
    pcm, t0, first_at, sample_rate = await synthesize(text)
    remainder = len(pcm) % 2
    if remainder:
        pcm = pcm[: len(pcm) - remainder]

    arrival_ms = (first_at - t0) * 1000.0
    silence_ms = first_audible_offset_ms(pcm, sample_rate) if pcm else None
    ttfa_ms = compute_ttfa(t0, first_at, silence_ms if silence_ms is not None else 0.0)
    return {
        "arrival_ms": arrival_ms,
        "silence_ms": silence_ms,
        "ttfa_ms": ttfa_ms,
        "audio_bytes": len(pcm),
        "sample_rate": sample_rate,
        "silence_degraded": silence_ms is None,
    }


async def _run(args: argparse.Namespace) -> dict:
    synthesize = _load_synthesize(Path(args.provider_module))
    prompts = list(DEFAULT_PROMPTS)
    if args.fixed_prompt:
        prompts = [("FIXED", args.fixed_prompt)] * args.trials
    else:
        rng = random.Random(args.seed)
        if args.trials <= len(prompts):
            prompts = rng.sample(prompts, args.trials)
        else:
            # Cycle with shuffle for longer runs.
            bag: list[tuple[str, str]] = []
            while len(bag) < args.trials:
                chunk = prompts[:]
                rng.shuffle(chunk)
                bag.extend(chunk)
            prompts = bag[: args.trials]

    trials: list[dict] = []
    failures: list[dict] = []

    # Warmup — discarded
    if not args.no_warmup:
        try:
            await synthesize(prompts[0][1])
            print("warmup: ok", flush=True)
        except Exception as exc:
            print(f"warmup: failed ({exc}); continuing", flush=True)

    for i, (pid, text) in enumerate(prompts, start=1):
        try:
            row = await _one_trial(synthesize, text)
            row.update({"trial": i, "prompt_id": pid, "error": None})
            trials.append(row)
            print(
                f"trial {i}/{args.trials}: ttfa={row['ttfa_ms']:.1f} ms "
                f"(arrival={row['arrival_ms']:.1f}, silence={row['silence_ms']})",
                flush=True,
            )
        except Exception as exc:
            failures.append({"trial": i, "prompt_id": pid, "error": str(exc)})
            print(f"trial {i}/{args.trials}: FAIL {exc}", flush=True)

    if len(trials) < 10:
        print(
            f"WARNING: only {len(trials)} successful trials (need ≥ 10)",
            file=sys.stderr,
        )

    ttfa_vals = [t["ttfa_ms"] for t in trials]
    arrival_vals = [t["arrival_ms"] for t in trials]
    silence_vals = [t["silence_ms"] for t in trials if t["silence_ms"] is not None]

    result = {
        "provider": args.provider,
        "model": args.model,
        "voice": args.voice,
        "trials_ok": len(trials),
        "trials_failed": len(failures),
        "methodology": "coval-perceived-ttfa",
        "ttfa_ms": summarize(ttfa_vals) if ttfa_vals else None,
        "arrival_ms": summarize(arrival_vals) if arrival_vals else None,
        "silence_ms": summarize(silence_vals) if silence_vals else None,
        "trials": trials,
        "failures": failures,
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Coval-aligned perceived TTFA latency test")
    p.add_argument("--provider-module", required=True, help="Path to adapter .py")
    p.add_argument("--provider", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--voice", required=True)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fixed-prompt", default=None)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--out", default=None, help="Write full JSON results here")
    args = p.parse_args()

    if args.trials < 10:
        raise SystemExit("--trials must be >= 10")

    result = asyncio.run(_run(args))
    print()
    print(f"Provider / model / voice: {args.provider} / {args.model} / {args.voice}")
    print(f"Trials: {result['trials_ok']} ok / {result['trials_failed']} failed")
    if result["ttfa_ms"]:
        s = result["ttfa_ms"]
        print("Perceived TTFA (ms)")
        print(f"  median: {s['median']:.2f}")
        print(f"  mean:   {s['mean']:.2f}")
        print(f"  min:    {s['min']:.2f}")
        print(f"  max:    {s['max']:.2f}")
    if result.get("arrival_ms") and result.get("silence_ms"):
        print("Breakdown (ms) — median")
        print(f"  arrival: {result['arrival_ms']['median']:.2f}")
        print(f"  leading silence: {result['silence_ms']['median']:.2f}")

    if args.out:
        out = Path(args.out)
        out.write_text(json.dumps(result, indent=2))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
