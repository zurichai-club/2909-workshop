# Closing the Loop: weather agent workshop

A tiny weather agent with **one** tool, `get_forecast(city, date)`, and five
exercises about evaluating it: golden sets, mutation testing, traces, turning a
production failure into a test, and (optional) an LLM judge.

Each exercise is a **self-contained folder** in `exercises/`. It has its own copy
of the agent (`weather_agent/`), its own data and its own README. Just `cd` into
the folder, no git checkouts. Each folder starts where the previous exercise
ended, and each README ends with a solution.

| # | Folder | You learn |
|---|---|---|
| 1 | [`01-golden-set`](exercises/01-golden-set) | Write golden cases and check tool calls, arguments and answers |
| 2 | [`02-mutation-testing`](exercises/02-mutation-testing) | Break the agent on purpose to find gaps in the tests |
| 3 | [`03-observability`](exercises/03-observability) | Find a failure in Langfuse traces |
| 4 | [`04-close-the-loop`](exercises/04-close-the-loop) | Turn that trace into a golden case, then fix the agent |
| 5 | [`05-llm-judge`](exercises/05-llm-judge) | Optional: a local LLM judge, sped up with a shared KV cache |

By default the agent uses a **mock model** (a few rules in
`weather_agent/agent.py`) and **recorded forecasts**, so every run is repeatable
and needs no API key.

## Setup

Requirements: Python 3.11+, `uv`, and Docker Compose for Langfuse (exercise 3).

```bash
cd workshop
uv sync --extra eval
python3 scripts/setup_env.py      # creates an ignored .env with local passwords and keys
docker compose up -d              # Langfuse at http://127.0.0.1:3000
uv run --env-file .env python scripts/check_setup.py
```

Sign in to Langfuse as `workshop@example.com` with the
`LANGFUSE_INIT_USER_PASSWORD` value from `.env`. The first start can take a few
minutes. Only the Langfuse UI and the MinIO media endpoint are exposed, both
bound to localhost.

Try the agent:

```bash
cd exercises/01-golden-set
uv run python ask.py "Do I need an umbrella in Berlin tomorrow?"
```

## A real model and live weather

Set `OPENAI_API_KEY` and optionally `OPENAI_MODEL` (default `gpt-4o-mini`), then:

```bash
uv run python ask.py --model openai --live "Do I need an umbrella in Berlin tomorrow?"
```

An OpenAI-compatible gateway also works: set `OPENAI_BASE_URL`. Its model must
support tool calling. `--live` calls the [Open-Meteo](https://open-meteo.com/en/docs)
forecast and [geocoding](https://open-meteo.com/en/docs/geocoding-api) APIs. The
recorded data is illustrative only. Never use it as weather advice.

## Traces

Every run appends spans to `traces.jsonl` in the exercise folder. When `.env` is
loaded (`uv run --env-file ../../.env ...`), spans are also sent to Langfuse over
[OTLP/HTTP](https://langfuse.com/integrations/native/opentelemetry). Traces
contain the questions and tool outputs, so use only synthetic workshop inputs.

## For maintainers

```bash
uv run python -m unittest discover -s tests -v
```

The tests run the starting point of every exercise folder, and check trace
parentage and the authenticated OTLP export request. The agent is copied on
purpose, so a change to it must be made in every folder that needs it.

To stop Langfuse without deleting traces: `docker compose stop`. Only run
`docker compose down -v` if you want to delete this workshop's Langfuse data.
The stack follows [Langfuse's Docker Compose deployment](https://langfuse.com/self-hosting/deployment/docker-compose).
