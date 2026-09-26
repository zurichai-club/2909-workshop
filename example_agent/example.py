"""Quickstart examples for the copied and adapted weather agent."""

from weather_agent.agent import ask, build_agent

QUESTIONS = [
    "Do I need an umbrella in Berlin tomorrow?",
    "Will it rain in Zurich tomorrow?",
    "Will it rain in Berlin three weeks from now?",
]


def main() -> None:
    agent = build_agent()
    for i, question in enumerate(QUESTIONS, start=1):
        print(f"\n=== Q{i}: {question} ===")
        print(ask(agent, question))


if __name__ == "__main__":
    main()
