"""Ask the weather agent one question.

    python ask.py "Do I need an umbrella in Berlin tomorrow?"
"""

import argparse

from weather_agent.agent import run_agent

parser = argparse.ArgumentParser()
parser.add_argument("question")
parser.add_argument("--live", action="store_true", help="Use live Open-Meteo instead of recorded data")
parser.add_argument("--model", choices=["mock", "openai"], default="mock")
args = parser.parse_args()

result = run_agent(args.question, model=args.model, live=args.live)
print(result.answer)
print(f"trace_id={result.trace_id}")
