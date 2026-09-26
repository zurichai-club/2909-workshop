"""Run one weather question; use --live for real Open-Meteo data."""

import argparse
from weather_agent.agent import build_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="+", help="Weather question")
    parser.add_argument("--live", action="store_true", help="Use live Open-Meteo instead of recorded data")
    parser.add_argument("--model", choices=["mock", "openai"], default="mock")
    args = parser.parse_args()
    result = build_agent(mode=args.model, fixture=not args.live).run(" ".join(args.question))
    print(result.answer)
    print(f"trace_id={result.trace_id}")


if __name__ == "__main__":
    main()
