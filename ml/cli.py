"""
ml/cli.py — Command-Line Interface for the Policy Optimizer
===========================================================

Exposes the ML pipeline as a ``rabies-optimize`` console command (configured
in ``pyproject.toml``) with three subcommands::

    rabies-optimize generate --n 300 --out data.csv
    rabies-optimize train    --data data.csv --out surrogate.joblib
    rabies-optimize optimize --model surrogate.joblib --budget 0.6

``generate`` and ``train`` can also be chained implicitly: ``optimize`` will
train on a fresh dataset if no ``--model`` is supplied.

Run ``rabies-optimize <subcommand> -h`` for the full option list.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from ml.data import generate_training_data
from ml.optimizer import PolicyOptimizer
from ml.surrogate import PolicySurrogate


def _cmd_generate(args: argparse.Namespace) -> None:
    generate_training_data(n_samples=args.n, random_seed=args.seed, save_path=args.out)
    print(f"Generated {args.n} runs -> {args.out}")


def _cmd_train(args: argparse.Namespace) -> None:
    data = pd.read_csv(args.data)
    surrogate = PolicySurrogate().fit(data, cv=args.cv)
    surrogate.save(args.out)
    print(f"Trained surrogate on {len(data)} rows -> {args.out}")
    for outcome, score in surrogate.cv_scores.items():
        print(f"  CV R^2 [{outcome}]: {score:.3f}")
    print("\nFeature importances:")
    print(surrogate.feature_importances().round(3).to_string())


def _cmd_optimize(args: argparse.Namespace) -> None:
    if args.model:
        surrogate = PolicySurrogate.load(args.model)
    else:
        print(f"No --model given; generating {args.n} runs and training...")
        data = generate_training_data(n_samples=args.n, random_seed=args.seed)
        surrogate = PolicySurrogate().fit(data)

    optimizer = PolicyOptimizer(surrogate, target=args.target)
    result = optimizer.optimize(budget=args.budget)

    print(
        f"\n=== Optimal policy under budget={args.budget:.2f} "
        f"(minimising {args.target}) ==="
    )
    for lever, value in result.best_policy.items():
        print(f"  {lever:<24} {value:.3f}")
    print(f"  {'-' * 32}")
    print(f"  predicted {args.target:<14} {result.predicted_deaths:.1f}")
    print(
        f"  normalised cost          {result.cost:.3f} "
        f"(budget {result.budget * optimizer.max_cost:.3f})"
    )

    print("\nCost / deaths Pareto frontier (cheapest first):")
    frontier = result.pareto[list(result.best_policy) + ["cost", "predicted_deaths"]]
    print(frontier.round(3).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rabies-optimize",
        description="ML-driven rabies intervention policy optimizer.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate ABM training data.")
    g.add_argument("--n", type=int, default=300, help="Number of simulation runs.")
    g.add_argument("--out", default="ml_training_data.csv", help="Output CSV path.")
    g.add_argument("--seed", type=int, default=42, help="Random seed.")
    g.set_defaults(func=_cmd_generate)

    t = sub.add_parser("train", help="Train the surrogate on a dataset.")
    t.add_argument("--data", required=True, help="Training CSV (from 'generate').")
    t.add_argument("--out", default="surrogate.joblib", help="Output model path.")
    t.add_argument(
        "--cv", type=int, default=5, help="Cross-validation folds (0 to skip)."
    )
    t.set_defaults(func=_cmd_train)

    o = sub.add_parser("optimize", help="Find the best budget-feasible policy.")
    o.add_argument(
        "--model", help="Trained surrogate (.joblib). Trains fresh if omitted."
    )
    o.add_argument("--budget", type=float, default=1.0, help="Budget in [0, 1].")
    o.add_argument("--target", default="human_deaths", help="Outcome to minimise.")
    o.add_argument("--n", type=int, default=300, help="Runs if training fresh.")
    o.add_argument(
        "--seed", type=int, default=42, help="Random seed if training fresh."
    )
    o.set_defaults(func=_cmd_optimize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
