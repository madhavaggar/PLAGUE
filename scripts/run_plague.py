#!/usr/bin/env python3
"""Run a single PLAGUE attack against one target model on HarmBench.

This reproduces the canonical PLAGUE configuration reported in the paper
(Primer -> Planner -> Finisher with a GOAT finisher). Results, including the
StrongReject / HarmBench / JailbreakBench ASR, are written to
``results/<project>/plague_<N>/``.

Example
-------
    python scripts/run_plague.py --target-model o3-2025-04-16 --num-goals 200
"""

import argparse
import os
import sys

# Make the `plague` package importable without an editable install.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from datasets import load_from_disk  # noqa: E402

from plague import Plague, PlagueConfig  # noqa: E402

DEFAULT_DATASET = os.path.join(REPO_ROOT, "data", "harmbench_dataset")


def build_config(args: argparse.Namespace) -> PlagueConfig:
    """Build the canonical PLAGUE config used for the main paper results."""
    return PlagueConfig(
        target_model=args.target_model,
        attacker_model=args.attacker_model,
        evaluator_model=args.evaluator_model,
        summariser_model=args.summariser_model,
        embedding_model=args.embedding_model,
        project=args.project,
        max_rounds=args.max_rounds,
        use_strategy_library=args.use_strategy_library,
        dataset="harmbench",
        verbose=args.verbose,
        max_workers=args.max_workers,
        max_tries=args.max_tries,
        primer_steps=args.primer_steps,
        finisher_type=args.finisher_type,
        use_iterative_prompting=args.use_iterative_prompting,
        use_planner=args.use_planner,
        max_search_results=args.max_search_results,
        use_actor_plan=args.use_actor_plan,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Models (defaults match the canonical PLAGUE run from the paper).
    parser.add_argument("--target-model", default="o3-2025-04-16")
    parser.add_argument("--attacker-model", default="deepseek-ai/DeepSeek-R1")
    parser.add_argument("--evaluator-model", default="Qwen/Qwen3-235B-A22B-fp8-tput")
    parser.add_argument("--summariser-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    # Attack hyper-parameters.
    parser.add_argument("--project", default="plague")
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--max-tries", type=int, default=2)
    parser.add_argument("--primer-steps", type=int, default=1)
    parser.add_argument("--max-search-results", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--finisher-type", default="goat", choices=["normal", "goat"])
    parser.add_argument("--use-strategy-library", action="store_true")
    parser.add_argument("--no-planner", dest="use_planner", action="store_false")
    parser.add_argument(
        "--no-iterative-prompting",
        dest="use_iterative_prompting",
        action="store_false",
    )
    parser.add_argument("--use-actor-plan", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    # Data.
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument("--num-goals", type=int, default=200)
    parser.set_defaults(use_planner=True, use_iterative_prompting=True)
    args = parser.parse_args()

    config = build_config(args)

    dataset = load_from_disk(args.dataset_path)
    goals = [row["prompt"] for row in dataset["train"]][: args.num_goals]

    print(f"Running PLAGUE against {args.target_model} on {len(goals)} goals...")
    attack = Plague(config)
    attack.optimize(goals)
    print(f"Done. Results written under results/{args.project}/")


if __name__ == "__main__":
    main()
