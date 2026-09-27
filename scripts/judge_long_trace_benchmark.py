"""Compare short and synthetic long traces with one GPU-resident MLX model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from judge_token_probabilities import ROOT, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--query-id", default="p2")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--long-spans", type=int, default=32)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "judge_long_trace_comparison.json")
    args = parser.parse_args()
    if args.rounds < 1 or args.long_spans < 1:
        parser.error("--rounds and --long-spans must both be positive")

    import mlx.core as mx
    from mlx_lm import load

    mx.set_default_device(mx.gpu)
    load_start = time.perf_counter()
    model, tokenizer = load(args.model)
    model_load_seconds = time.perf_counter() - load_start
    scenarios = {}
    for name, span_count in (("short", 0), ("long", args.long_spans)):
        print(f"\n{name} trace: {span_count} synthetic context spans", flush=True)
        scenarios[name] = run_benchmark(
            args.model, args.query_id, args.rounds,
            synthetic_spans=span_count,
            loaded_model=(model, tokenizer),
            keep_prefix_warm=True,
        )
        report = scenarios[name]
        print(f"shared prefix: {report['shared_prefix_tokens']} tokens; "
              f"warm-cache speedup: {report['summary']['speedup']:.2f}x", flush=True)

    short, long = scenarios["short"], scenarios["long"]
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "Single process and one GPU-resident model; each scenario prefilled "
                  "its shared cache once after warm-up and kept it alive across timed rounds. "
                  "Per-question cache copies are timed. The long trace adds synthetic "
                  "trace-shaped context to the same recorded p2 case.",
        "model_load_seconds": model_load_seconds,
        "scenarios": scenarios,
        "comparison": {
            "shared_prefix_token_ratio": (
                long["shared_prefix_tokens"] / short["shared_prefix_tokens"]),
            "short_warm_cache_speedup": short["summary"]["speedup"],
            "long_warm_cache_speedup": long["summary"]["speedup"],
            "speedup_ratio_long_over_short": (
                long["summary"]["speedup"] / short["summary"]["speedup"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"\nresult: {args.output}")
    for name, report in scenarios.items():
        s = report["summary"]
        print(f"{name}: {report['shared_prefix_tokens']} shared tokens, "
              f"{s['cold_median_seconds']:.2f}s cold, "
              f"{s['cached_median_seconds_warm']:.2f}s warm cached, "
              f"{s['speedup']:.2f}x; one-time prefill "
              f"{report['shared_prefill_seconds_once']:.2f}s")


if __name__ == "__main__":
    main()
