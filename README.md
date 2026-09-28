# Workshop: Agents Lifecycle explained

## Install

Requirements: Python 3.11+, [`uv`](https://docs.astral.sh/uv/), and Docker Compose.

```bash
cd workshop
uv sync --extra eval
docker compose up -d              # Langfuse at http://127.0.0.1:3000 (the first start takes a few minutes)
uv run python scripts/check_setup.py
```

Sign in to Langfuse as `workshop@example.com` with the password `workshop`.

For exercise 5, set an OpenAI key:

```bash
export OPENAI_API_KEY=...
```

For the local LLM judge in `extras/` (Apple Silicon only, ~4.3 GB model download):

```bash
uv sync --extra eval --extra mlx
```

