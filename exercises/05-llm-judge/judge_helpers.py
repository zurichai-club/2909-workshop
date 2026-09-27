"""Boilerplate for judge.py: load the trace, pad it, print and save the results."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).parent


def load_trace_evidence(query_id):
    """Collect what the judge sees for one production run: question, tool calls, answer and spans."""
    batch_file = HERE / "batch.jsonl"
    if not batch_file.exists():
        sys.exit("Run production_batch.py first to create batch.jsonl")

    row = None
    for line in batch_file.read_text().splitlines():
        candidate = json.loads(line)
        if candidate["query_id"] == query_id:
            row = candidate
    if row is None:
        sys.exit(f"No run with query id {query_id} in batch.jsonl")

    spans = []
    for line in (HERE / "traces.jsonl").read_text().splitlines():
        span = json.loads(line)
        if span["trace_id"] == row["trace_id"]:
            spans.append({
                "name": span["name"],
                "input": span["attributes"].get("langfuse.observation.input"),
                "output": span["attributes"].get("langfuse.observation.output"),
            })

    return {
        "user_question": row["input"],
        "tool_calls": row["tool_calls"],
        "final_answer": row["answer"],
        "trace_spans": spans,
    }


def add_synthetic_spans(evidence, count):
    """Make the trace longer with repeated, made-up context spans. Only used for timing.

    The facts stay the same as p2, so the expected answers do not change.
    """
    phases = ["request_intake", "tool_planning", "forecast_lookup", "tool_response", "answer_draft", "answer_review"]
    padding = []
    for index in range(count):
        padding.append({
            "span_id": f"synthetic-context-{index:03d}",
            "name": "agent." + phases[index % len(phases)],
            "sequence": index + 1,
            "attributes": {
                "user_request_snapshot": (
                    "The user asks whether an umbrella is needed in Berlin, Germany "
                    "tomorrow and explicitly asks for the temperature in Fahrenheit. "
                    "This is a recorded workshop case, not a live weather request."),
                "agent_context": (
                    "The agent should use the weather tool result as its source, "
                    "distinguish tool arguments from tool output, and avoid claiming "
                    "that an illustrative fixture is a live observation. The response "
                    "should answer the umbrella question, report precipitation clearly, "
                    "and respect the requested temperature units."),
                "recorded_observation": (
                    "The get_forecast call for Berlin, Germany returned a maximum "
                    "temperature of 14 degrees Celsius and a precipitation probability "
                    "of 70 percent. The tool returned a normal result, not an error."),
                "quality_note": (
                    "A final answer in Celsius would miss the user's Fahrenheit "
                    "requirement even if its precipitation advice and tool call are "
                    "otherwise grounded in the recorded result."),
            },
        })
    if padding:
        evidence["synthetic_trace_spans"] = padding
    return evidence


def shared_prefix_length(prompts):
    """How many tokens, from the start, are identical in every prompt."""
    length = 0
    shortest = min(len(prompt) for prompt in prompts)
    while length < shortest:
        token = prompts[0][length]
        for prompt in prompts:
            if prompt[length] != token:
                return length
        length += 1
    return length


def single_token_id(tokenizer, text):
    """The token id for a label such as "YES". The judge needs each label to be exactly one token."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) != 1:
        sys.exit(f"{text!r} is {len(tokens)} tokens for this tokenizer; judge.py needs exactly one")
    return tokens[0]


def print_and_save(report, output_path):
    print(f"\n{'question':<22} {'expected':>8} {'YES %':>7} {'NO %':>7} {'other %':>8} {'YES vs NO %':>12} choice")
    for row in report["questions"]:
        print(f"{row['id']:<22} {row['expected']:>8} {row['yes_pct']:7.2f} {row['no_pct']:7.2f} "
              f"{row['other_pct']:8.2f} {row['yes_vs_no_pct']:12.2f} {row['choice']}")
    print("YES % and NO % are over the whole vocabulary; 'YES vs NO %' only compares the two labels.\n")

    for name, value in report["timing_seconds"].items():
        print(f"{name:<40} {value:8.2f} s")
    print(f"speedup once the prefix is cached: {report['speedup_warm_cache']:.1f}x")
    print(f"correct: {report['correct']}/{len(report['questions'])}")

    largest_difference = 0
    for row in report["questions"]:
        largest_difference = max(largest_difference, abs(row["yes_vs_no_pct"] - row["cold_yes_vs_no_pct"]))
    print(f"largest cold vs cached difference: {largest_difference:.5f} percentage points")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"report: {output_path}")
