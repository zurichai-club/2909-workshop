"""Measure YES/NO next-token probabilities with and without a shared MLX KV cache."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import time

from judge_cache_benchmark import QUESTIONS, RUBRIC, load_evidence, longest_shared_prefix

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))


def add_synthetic_trace_spans(evidence: dict, count: int) -> dict:
    """Add deterministic trace-shaped context without changing the p2 facts.

    These spans model the repeated prompt, tool, and validation payloads often
    present in an agent trace. They are synthetic and are only for timing.
    """
    if count < 0:
        raise ValueError("Synthetic span count cannot be negative")
    if not count:
        return evidence
    evidence = copy.deepcopy(evidence)
    phases = ("request intake", "tool planning", "forecast lookup", "tool response",
              "answer draft", "answer review")
    for index in range(count):
        evidence.setdefault("synthetic_trace_spans", []).append({
            "span_id": f"synthetic-context-{index:03d}",
            "name": f"agent.{phases[index % len(phases)].replace(' ', '_')}",
            "sequence": index + 1,
            "attributes": {
                "user_request_snapshot": (
                    "The user asks whether an umbrella is needed in Berlin, Germany "
                    "tomorrow and explicitly asks for the temperature in Fahrenheit. "
                    "This is a recorded workshop case, not a live weather request."
                ),
                "agent_context": (
                    "The agent should use the weather tool result as its source, "
                    "distinguish tool arguments from tool output, and avoid claiming "
                    "that an illustrative fixture is a live observation. The response "
                    "should answer the umbrella question, report precipitation clearly, "
                    "and respect the requested temperature units."
                ),
                "recorded_observation": (
                    "The get_forecast call for Berlin, Germany returned a maximum "
                    "temperature of 14 degrees Celsius and a precipitation probability "
                    "of 70 percent. The tool returned a normal result, not an error."
                ),
                "quality_note": (
                    "A final answer in Celsius would miss the user's Fahrenheit "
                    "requirement even if its precipitation advice and tool call are "
                    "otherwise grounded in the recorded result."
                ),
            },
        })
    return evidence


def run_benchmark(model_name: str, query_id: str, rounds: int, *,
                  synthetic_spans: int = 0, loaded_model=None,
                  keep_prefix_warm: bool = False) -> dict:
    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
    except ImportError as exc:
        raise SystemExit("Install the optional dependency: uv sync --extra eval --extra mlx") from exc

    mx.set_default_device(mx.gpu)
    evidence, trace_id = load_evidence(query_id)
    evidence = add_synthetic_trace_spans(evidence, synthetic_spans)
    evidence_text = json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2)
    if loaded_model is None:
        load_start = time.perf_counter()
        model, tokenizer = load(model_name)
        model_load_seconds = time.perf_counter() - load_start
    else:
        model, tokenizer = loaded_model
        model_load_seconds = 0.0

    label_tokens = {label: tokenizer.encode(label, add_special_tokens=False)
                    for label in ("YES", "NO")}
    if any(len(tokens) != 1 for tokens in label_tokens.values()):
        raise SystemExit(f"Expected one token per answer label; got {label_tokens}. "
                         "Sequence scoring is required for this tokenizer.")
    yes_id, no_id = label_tokens["YES"][0], label_tokens["NO"][0]
    if yes_id == no_id:
        raise SystemExit("YES and NO resolved to the same token")

    prompts = []
    for _, question, _ in QUESTIONS:
        messages = [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"EVIDENCE:\n{evidence_text}\n\nQUESTION: {question}\nAnswer:"},
        ]
        prompts.append(list(tokenizer.apply_chat_template(messages, add_generation_prompt=True)))
    shared = longest_shared_prefix(prompts)
    if len(shared) < 32 or any(len(prompt) <= len(shared) for prompt in prompts):
        raise SystemExit("The tokenized prompts do not share a useful prefix")
    tails = [prompt[len(shared):] for prompt in prompts]

    def evaluate(tokens: list[int], cache) -> dict:
        logits = model(mx.array(tokens, dtype=mx.int32)[None], cache=cache)[0, -1, :]
        # Float32 keeps the normalization stable even when model logits use less precision.
        logits = logits.astype(mx.float32)
        log_normalizer = mx.logsumexp(logits)
        pair_normalizer = mx.logsumexp(mx.stack([logits[yes_id], logits[no_id]]))
        raw_yes = mx.exp(logits[yes_id] - log_normalizer)
        raw_no = mx.exp(logits[no_id] - log_normalizer)
        pair_yes = mx.exp(logits[yes_id] - pair_normalizer)
        pair_no = mx.exp(logits[no_id] - pair_normalizer)
        mx.eval(raw_yes, raw_no, pair_yes, pair_no, [item.state for item in cache])
        raw_yes, raw_no = float(raw_yes.item()), float(raw_no.item())
        pair_yes, pair_no = float(pair_yes.item()), float(pair_no.item())
        return {
            "raw_yes_pct": 100 * raw_yes,
            "raw_no_pct": 100 * raw_no,
            "other_token_pct": 100 * (1 - raw_yes - raw_no),
            "yes_pct_given_yes_or_no": 100 * pair_yes,
            "no_pct_given_yes_or_no": 100 * pair_no,
            "choice": "YES" if pair_yes >= pair_no else "NO",
        }

    # Compile the model and scoring operations before either timed path.
    evaluate(prompts[0], make_prompt_cache(model))
    mx.synchronize()

    shared_base = None
    shared_prefill_seconds = 0.0
    shared_cache_bytes = 0
    active_memory_after_prefill_gb = None
    if keep_prefix_warm:
        start = time.perf_counter()
        shared_base = make_prompt_cache(model)
        model(mx.array(shared, dtype=mx.int32)[None], cache=shared_base)
        mx.eval([item.state for item in shared_base])
        mx.synchronize()
        shared_prefill_seconds = time.perf_counter() - start
        if shared_base[0].offset != len(shared):
            raise RuntimeError("Shared prefix cache has an unexpected offset")
        shared_cache_bytes = sum(item.nbytes for item in shared_base)
        active_memory_after_prefill_gb = mx.get_active_memory() / 1e9

    def cold_round() -> dict:
        rows = []
        start = time.perf_counter()
        for (name, question, expected), prompt in zip(QUESTIONS, prompts):
            tic = time.perf_counter()
            scores = evaluate(prompt, make_prompt_cache(model))
            rows.append({"id": name, "question": question, "expected": expected,
                         **scores, "seconds": time.perf_counter() - tic})
        return {"total_seconds": time.perf_counter() - start, "prefill_seconds": 0.0,
                "rows": rows}

    def cached_round() -> dict:
        start = time.perf_counter()
        base = shared_base if keep_prefix_warm else make_prompt_cache(model)
        if not keep_prefix_warm:
            model(mx.array(shared, dtype=mx.int32)[None], cache=base)
            mx.eval([item.state for item in base])
            prefill_seconds = time.perf_counter() - start
        else:
            prefill_seconds = 0.0
        if base[0].offset != len(shared):
            raise RuntimeError(f"Prefix cache has {base[0].offset} tokens, expected {len(shared)}")
        rows = []
        for (name, question, expected), tail in zip(QUESTIONS, tails):
            tic = time.perf_counter()
            scores = evaluate(tail, copy.deepcopy(base))
            rows.append({"id": name, "question": question, "expected": expected,
                         **scores, "seconds": time.perf_counter() - tic})
            if base[0].offset != len(shared):
                raise RuntimeError("Shared prefix cache was modified by a judgment")
        return {"total_seconds": time.perf_counter() - start,
                "prefill_seconds": prefill_seconds, "rows": rows}

    runs = []
    for round_number in range(rounds):
        order = ["cold", "cached"] if round_number % 2 == 0 else ["cached", "cold"]
        result = {"round": round_number + 1, "order": order}
        for mode in order:
            result[mode] = cold_round() if mode == "cold" else cached_round()
            print(f"round {round_number + 1} {mode}: {result[mode]['total_seconds']:.2f}s",
                  flush=True)
        runs.append(result)

    cold_times = [run["cold"]["total_seconds"] for run in runs]
    cached_times = [run["cached"]["total_seconds"] for run in runs]
    last_cold, last_cached = runs[-1]["cold"]["rows"], runs[-1]["cached"]["rows"]
    max_delta = max(abs(a["yes_pct_given_yes_or_no"] - b["yes_pct_given_yes_or_no"])
                    for run in runs for a, b in zip(run["cold"]["rows"],
                                                    run["cached"]["rows"]))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name, "source_query_id": query_id, "source_trace_id": trace_id,
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "mlx_version": importlib.metadata.version("mlx"),
        "platform": platform.platform(), "model_load_seconds": model_load_seconds,
        "peak_process_memory_gb": mx.get_peak_memory() / 1e9,
        "execution_device": str(mx.default_device()),
        "synthetic_trace_spans": synthetic_spans,
        "shared_cache_kept_warm_across_rounds": keep_prefix_warm,
        "shared_prefill_seconds_once": shared_prefill_seconds,
        "shared_cache_gb": shared_cache_bytes / 1e9,
        "active_memory_after_prefill_gb": active_memory_after_prefill_gb,
        "label_token_ids": {key: value[0] for key, value in label_tokens.items()},
        "shared_prefix_tokens": len(shared), "full_prompt_tokens": [len(p) for p in prompts],
        "tail_tokens": [len(t) for t in tails], "rounds": runs,
        "summary": {
            "cold_median_seconds": statistics.median(cold_times),
            "cached_median_seconds_including_prefill": (
                statistics.median(cached_times) if not keep_prefix_warm else None),
            "cached_median_seconds_warm": (
                statistics.median(cached_times) if keep_prefix_warm else None),
            "cached_median_seconds_amortized_prefill": (
                statistics.median(cached_times) + shared_prefill_seconds / rounds
                if keep_prefix_warm else statistics.median(cached_times)),
            "cached_prefill_median_seconds": statistics.median(
                run["cached"]["prefill_seconds"] for run in runs),
            "speedup": statistics.median(cold_times) / statistics.median(cached_times),
            "speedup_amortized_prefill": (
                statistics.median(cold_times) /
                (statistics.median(cached_times) + shared_prefill_seconds / rounds)),
            "first_cached_round_seconds_including_one_time_prefill": (
                runs[0]["cached"]["total_seconds"] + shared_prefill_seconds
                if keep_prefix_warm else runs[0]["cached"]["total_seconds"]),
            "cold_correct": sum(row["choice"] == row["expected"] for row in last_cold),
            "cached_correct": sum(row["choice"] == row["expected"] for row in last_cached),
            "max_cold_cached_yes_pct_delta_points": max_delta,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--query-id", default="p2")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--synthetic-spans", type=int, default=0,
                        help="Add deterministic trace-shaped context for timing")
    parser.add_argument("--keep-prefix-warm", action="store_true",
                        help="Prefill once before timing and reuse the cache across rounds")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "judge_token_probabilities.json")
    args = parser.parse_args()
    if args.rounds < 1 or args.synthetic_spans < 0:
        parser.error("--rounds must be positive and --synthetic-spans nonnegative")
    report = run_benchmark(args.model, args.query_id, args.rounds,
                           synthetic_spans=args.synthetic_spans,
                           keep_prefix_warm=args.keep_prefix_warm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("\nQuestion                              raw YES %  raw NO %  YES|labels %  "
          "NO|labels %  other %  choice")
    for row in report["rounds"][-1]["cached"]["rows"]:
        print(f"{row['id']:<37} {row['raw_yes_pct']:9.4f}  "
              f"{row['raw_no_pct']:8.4f}  "
              f"{row['yes_pct_given_yes_or_no']:12.4f}  "
              f"{row['no_pct_given_yes_or_no']:11.4f}  "
              f"{row['other_token_pct']:7.4f}  {row['choice']}")
    print("Raw and other percentages cover the full next-token vocabulary; "
          "YES|labels and NO|labels are normalized over the two answer tokens.")
    print(f"result: {args.output}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
