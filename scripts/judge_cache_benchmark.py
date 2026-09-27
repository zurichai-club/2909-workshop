"""Benchmark ten local yes/no trace judgments with and without a shared KV prefix.

The cold and cached paths use the same fully tokenized prompts and greedy
decoding. Only the cached path prefills their longest shared token prefix once.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

QUESTIONS = [
    ("requested_fahrenheit", "Did the user explicitly request Fahrenheit?", "YES"),
    ("answered_fahrenheit", "Did the final answer give the temperature in Fahrenheit?", "NO"),
    ("called_forecast", "Did the agent call get_forecast?", "YES"),
    ("city_berlin", "Was the forecast tool called for Berlin, Germany?", "YES"),
    ("temperature_14c", "Did the tool return a maximum temperature of 14 °C?", "YES"),
    ("rain_over_90", "Did the tool return a precipitation probability above 90%?", "NO"),
    ("answer_70_percent", "Did the final answer mention a 70% precipitation probability?", "YES"),
    ("live_source", "Was the tool output from a live Open-Meteo call?", "NO"),
    ("satisfied_units", "Did the final answer satisfy the user's requested temperature units?", "NO"),
    ("tool_error", "Did the forecast tool return an error?", "NO"),
]

RUBRIC = (
    "You are a strict evaluator of one weather-agent trace. Answer the QUESTION "
    "using only the EVIDENCE. Return exactly YES or NO, with no explanation. "
    "Distinguish the user's request, the tool call, the tool result, and the "
    "final answer. A Celsius answer does not satisfy a Fahrenheit request. "
    "A recorded illustrative fixture is not a live API call. If the evidence "
    "does not support a yes, answer NO."
)


def load_evidence(query_id: str) -> tuple[dict, str]:
    batch_path = ROOT / "batch.jsonl"
    if not batch_path.exists():
        raise SystemExit("Run scripts/production_batch.py first to create batch.jsonl")
    rows = [json.loads(line) for line in batch_path.read_text().splitlines()]
    row = next((item for item in rows if item.get("query_id") == query_id), None)
    if row is None:
        raise SystemExit(f"Query {query_id!r} was not found in batch.jsonl")
    spans = []
    trace_path = ROOT / "traces.jsonl"
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            span = json.loads(line)
            if span.get("trace_id") == row["trace_id"]:
                attributes = span.get("attributes", {})
                spans.append({"name": span["name"],
                              "input": attributes.get("langfuse.observation.input"),
                              "output": attributes.get("langfuse.observation.output")})
    evidence = {
        "user_question": row["input"],
        "tool_calls": row["tool_calls"],
        "final_answer": row["answer"],
        "trace_spans": spans,
    }
    return evidence, row["trace_id"]


def longest_shared_prefix(sequences: list[list[int]]) -> list[int]:
    if not sequences:
        raise ValueError("No prompts")
    prefix = sequences[0]
    for sequence in sequences[1:]:
        stop = next((i for i, (a, b) in enumerate(zip(prefix, sequence)) if a != b),
                    min(len(prefix), len(sequence)))
        prefix = prefix[:stop]
    return prefix


def label_of(output: str) -> str:
    match = re.match(r"^\s*(YES|NO)\b", output, re.IGNORECASE)
    return match.group(1).upper() if match else "INVALID"


def run_benchmark(model_name: str, query_id: str, rounds: int, max_tokens: int) -> dict:
    try:
        import mlx.core as mx
        from mlx_lm import generate, load
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
    except ImportError as exc:
        raise SystemExit("Install the optional dependency: uv sync --extra eval --extra mlx") from exc

    evidence, trace_id = load_evidence(query_id)
    evidence_text = json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2)
    load_start = time.perf_counter()
    try:
        model, tokenizer = load(model_name)
    except Exception as exc:
        raise SystemExit(f"Could not load MLX model {model_name}: {exc}") from exc
    model_load_seconds = time.perf_counter() - load_start

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

    # Compile kernels before timing either path.
    generate(model, tokenizer, prompt=prompts[0], max_tokens=2)

    def cold_round():
        rows = []
        start = time.perf_counter()
        for (name, question, expected), prompt in zip(QUESTIONS, prompts):
            tic = time.perf_counter()
            output = generate(model, tokenizer, prompt=prompt,
                              prompt_cache=make_prompt_cache(model), max_tokens=max_tokens)
            rows.append({"id": name, "question": question, "expected": expected,
                         "output": output.strip(), "label": label_of(output),
                         "seconds": time.perf_counter() - tic})
        return {"total_seconds": time.perf_counter() - start, "prefill_seconds": 0.0,
                "rows": rows}

    def cached_round():
        start = time.perf_counter()
        base = make_prompt_cache(model)
        for _ in generate_step(mx.array(shared), model, max_tokens=0, prompt_cache=base):
            pass
        prefill_seconds = time.perf_counter() - start
        if base[0].offset != len(shared):
            raise RuntimeError(f"Prefix cache has {base[0].offset} tokens, expected {len(shared)}")
        rows = []
        for (name, question, expected), tail in zip(QUESTIONS, tails):
            tic = time.perf_counter()
            output = generate(model, tokenizer, prompt=tail,
                              prompt_cache=copy.deepcopy(base), max_tokens=max_tokens)
            rows.append({"id": name, "question": question, "expected": expected,
                         "output": output.strip(), "label": label_of(output),
                         "seconds": time.perf_counter() - tic})
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
            print(f"round {round_number + 1} {mode}: {result[mode]['total_seconds']:.2f}s", flush=True)
        runs.append(result)

    cold_times = [run["cold"]["total_seconds"] for run in runs]
    cached_times = [run["cached"]["total_seconds"] for run in runs]
    cold_labels = [row["label"] for row in runs[-1]["cold"]["rows"]]
    cached_labels = [row["label"] for row in runs[-1]["cached"]["rows"]]
    expected = [item[2] for item in QUESTIONS]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name, "source_query_id": query_id, "source_trace_id": trace_id,
        "mlx_lm_version": importlib.metadata.version("mlx-lm"),
        "mlx_version": importlib.metadata.version("mlx"),
        "platform": platform.platform(), "model_load_seconds": model_load_seconds,
        "peak_process_memory_gb": mx.get_peak_memory() / 1e9,
        "shared_prefix_tokens": len(shared), "full_prompt_tokens": [len(p) for p in prompts],
        "tail_tokens": [len(t) for t in tails], "max_output_tokens": max_tokens,
        "rounds": runs,
        "summary": {
            "cold_median_seconds": statistics.median(cold_times),
            "cached_median_seconds_including_prefill": statistics.median(cached_times),
            "cached_prefill_median_seconds": statistics.median(
                run["cached"]["prefill_seconds"] for run in runs),
            "cold_question_median_seconds": statistics.median(
                row["seconds"] for run in runs for row in run["cold"]["rows"]),
            "cached_question_median_seconds_including_copy": statistics.median(
                row["seconds"] for run in runs for row in run["cached"]["rows"]),
            "speedup": statistics.median(cold_times) / statistics.median(cached_times),
            "cold_correct": sum(a == b for a, b in zip(cold_labels, expected)),
            "cached_correct": sum(a == b for a, b in zip(cached_labels, expected)),
            "labels_identical": cold_labels == cached_labels,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--query-id", default="p2")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "judge_kv_cache.json")
    args = parser.parse_args()
    if args.rounds < 1 or args.max_tokens < 2:
        parser.error("--rounds must be positive and --max-tokens must be at least 2")
    report = run_benchmark(args.model, args.query_id, args.rounds, args.max_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"result: {args.output}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
