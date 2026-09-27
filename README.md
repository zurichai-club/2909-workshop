# Closing the Loop: weather agent workshop

This is a separate repository inside `workshop/`. It copies and adapts the
parent project's `example_agent` interface. The parent project, proxy and
telemetry packages are not needed here. The workshop agent has **one** tool,
`get_forecast(city, date)`, which calls Open-Meteo for live forecasts. Recorded,
illustrative data and a deterministic model make every exercise runnable without
API credentials.

## Setup

Requirements: Python 3.11+, `uv`, Docker Compose for Langfuse, and enough memory
for Langfuse's Postgres, ClickHouse, Redis, MinIO, web and worker services.

```bash
cd workshop
uv sync --extra eval
python3 scripts/setup_env.py
docker compose up -d
uv run --env-file .env python scripts/check_setup.py
```

`setup_env.py` creates an ignored `.env` once. Langfuse will be at
<http://127.0.0.1:3000>. Sign in as `workshop@example.com` with the
`LANGFUSE_INIT_USER_PASSWORD` value from `.env`. The project and OTLP keys are
created automatically. The first startup can take a few minutes. Compose
binds only the Langfuse UI and MinIO media endpoint to localhost; database
ports stay private to its Docker network.

## Run

```bash
uv run --env-file .env python example_agent/example.py
uv run --env-file .env python example_agent/cli.py "Do I need an umbrella in Berlin tomorrow?"
uv run --env-file .env python scripts/eval_golden.py --deepeval
uv run --env-file .env python scripts/mutate.py
uv run --env-file .env python scripts/production_batch.py
uv run --env-file .env python scripts/capture_failure.py p2
```

Run `uv run python -m unittest discover -s tests -v` to verify trace
parentage and the authenticated OTLP export request locally.

Every run writes local `traces.jsonl` and exports OTLP spans to Langfuse when
`.env` is loaded. In Langfuse, open **Traces**, search the trace ID printed by
the CLI or batch, then inspect the root agent span, two model spans and tool
span. The batch index in `batch.jsonl` maps query IDs to trace IDs. `p2` is a
deliberate failure: the user requested Fahrenheit but the mock answer says °C.
Captured questions and tool outputs are included in traces; use synthetic
workshop inputs only.

For real model calls and current forecasts, set `OPENAI_API_KEY` and
`OPENAI_MODEL` in your environment, then run:

```bash
MODEL_MODE=openai uv run --env-file .env python example_agent/cli.py --model openai --live "Do I need an umbrella in Berlin tomorrow?"
```

An OpenAI-compatible local gateway also works with `OPENAI_BASE_URL`; its
model must support tool calling. Live Open-Meteo needs network access. The
recorded fixture is intentionally illustrative and should never be used as
weather advice.

## Exercises

1. [Golden cases](exercises/01-golden.md) — `ex1-start`
2. [Mutation testing](exercises/02-mutation.md) — `ex2-start`
3. [Observability](exercises/03-observability.md) — `ex3-start`
4. [Close the loop](exercises/04-close-loop.md) — `ex4-start`
5. [Advanced LLM judge and KV cache](exercises/05-advanced-judge.md) — optional `ex5-start`

Each tag is a checkpoint in this nested repository. The `solutions` branch
adds a unit assertion and a Fahrenheit fix. Return to `main` after exploring
it. For example, `git checkout ex3-start` starts the tracing exercise.

## How this maps to the talk

The offline loop is `fixtures/golden.json` → `scripts/eval_golden.py` →
`scripts/mutate.py`. The online loop is `scripts/production_batch.py` →
Langfuse trace inspection. `scripts/capture_failure.py` brings a confirmed
online failure back into the golden set. DeepEval reports tool correctness;
plain Python assertions check exact arguments and answer properties.

The local stack follows [Langfuse's Docker Compose deployment](https://langfuse.com/self-hosting/deployment/docker-compose)
and sends authenticated [OTLP/HTTP spans](https://langfuse.com/integrations/native/opentelemetry).
The tool uses the [Open-Meteo forecast](https://open-meteo.com/en/docs) and
[geocoding](https://open-meteo.com/en/docs/geocoding-api) APIs.

The optional fifth exercise scores the YES and NO next-token probabilities for
ten trace questions on a local MLX model. It benchmarks full prompt processing
against a precomputed shared prefix, while keeping the original four exercises
and their checkpoint tags unchanged. Install it with
`uv sync --extra eval --extra mlx`; it requires Apple Silicon and a working
Metal device.
The [local 7B token probability benchmark](results/judge_token_probabilities.json)
measured 10.57 s cold versus 1.34 s with cache reuse for ten judgments
(7.89×), with virtually identical percentages. The report provides both raw
full-vocabulary probabilities and YES/NO normalized percentages.

To stop without deleting traces: `docker compose stop`. To restart:
`docker compose up -d`. Only use `docker compose down -v` when you intend to
delete this workshop's saved Langfuse data.
