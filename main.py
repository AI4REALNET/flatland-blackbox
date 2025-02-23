import argparse
import json

from flatland_blackbox.compute_results import aggregate_experiment_results
from flatland_blackbox.run_experiments import (
    run_experiments,
    run_single_experiment,
    write_results_to_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Flatland experiments, single solves, or training."
    )
    parser.add_argument(
        "--mode",
        choices=["solve", "train", "experiments"],
        default="experiments",
        help="Mode to run. 'solve' runs a single map; 'train' runs training; 'experiments' runs multiple configurations of training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--width", type=int, default=30, help="Environment width.")
    parser.add_argument("--height", type=int, default=30, help="Environment height.")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents.")
    parser.add_argument(
        "--max_cities", type=int, default=2, help="Max number of cities."
    )

    # For solve mode: select solver (pp or cbs)
    parser.add_argument(
        "--solver",
        choices=["pp", "cbs"],
        default="pp",
        help="Which solver to use in solve mode.",
    )

    # Training parameters (for train mode)
    parser.add_argument("--iters", type=int, default=300, help="Training iterations.")
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate for training."
    )
    parser.add_argument(
        "--lam", type=float, default=3.0, help="Lambda for differentiable solver."
    )

    # Experiment parameters (for experiments mode)
    parser.add_argument(
        "--num_seeds", type=int, default=1, help="Number of seeds to run."
    )
    parser.add_argument("--start_seed", type=int, default=42, help="Starting seed.")
    parser.add_argument(
        "--num_agents_list",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16],
        help="List of agent counts to test.",
    )
    parser.add_argument(
        "--max_cities_list",
        type=int,
        nargs="+",
        default=[2, 3, 5],
        help="List of max city counts to test.",
    )
    parser.add_argument(
        "--width_list",
        type=int,
        nargs="+",
        default=[30],
        help="List of map widths to test.",
    )
    parser.add_argument(
        "--height_list",
        type=int,
        nargs="+",
        default=[30],
        help="List of map heights to test.",
    )

    # Output file for experiments
    parser.add_argument(
        "--output_csv",
        type=str,
        default="outputs/experiment_results.csv",
        help="Output CSV file for experiment results.",
    )
    # Optional config file for experiments
    parser.add_argument(
        "--config",
        type=str,
        default="run_configs.json",
        help="Path to JSON file with experiment configurations (overrides command-line arguments).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "experiments" and args.config:
        try:
            with open(args.config, "r") as f:
                config_data = json.load(f)
            for key, value in config_data.items():
                setattr(args, key, value)
            print(f"Loaded experiment config from {args.config}")
        except Exception as e:
            print(f"Failed to load config file {args.config}: {e}")

    # Convert args to a dictionary of keyword arguments.
    kwargs = vars(args)

    if args.mode == "experiments":
        results = run_experiments(max_workers=12, **kwargs)
        csv_file_name = kwargs["output_csv"]
        write_results_to_csv(results, csv_file_name)
        aggregate_experiment_results(
            csv_file_name, out_csv="outputs/aggregated_results.csv"
        )
    elif args.mode == "solve":
        run_single_experiment(**kwargs)
    elif args.mode == "train":
        kwargs["solver"] = "trained"
        run_single_experiment(**kwargs)
    else:
        print("Unknown mode.")


if __name__ == "__main__":
    main()
