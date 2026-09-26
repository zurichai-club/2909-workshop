"""Prompt mutation: remove the units instruction and see whether goldens notice."""

from eval_golden import evaluate_cases


def main():
    print("Baseline:")
    baseline = evaluate_cases()
    print("\nMutant: remove the units instruction")
    mutant = evaluate_cases(mutation="remove_units")
    killed = not baseline and bool(mutant)
    print(f"Mutation score: {int(killed)}/1; {'killed' if killed else 'survived'}")
    return bool(baseline)


if __name__ == "__main__":
    raise SystemExit(main())
