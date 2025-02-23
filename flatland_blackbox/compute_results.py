import argparse
import os

import pandas as pd


def aggregate_experiment_results(csv_file, out_csv="outputs/aggregated_results.csv"):
    """
    Reads experiment results, computes aggregate statistics (min, mean, max)
    for each method grouped by num_agents, prints the statistics, and writes the results
    to a new CSV file.

    Args:
        csv_file (str): Path to the CSV file with the raw experiment results.
        out_csv (str, optional): Path to save the aggregated CSV. Defaults to "outputs/aggregated_results.csv".

    Returns:
        None
    """
    # Read the raw CSV file
    df = pd.read_csv(csv_file)

    # Ensure the output directory exists
    out_dir = os.path.dirname(out_csv)
    os.makedirs(out_dir, exist_ok=True)

    # Group the results by number of agents
    grouped = df.groupby("num_agents")
    num_agents = sorted(df["num_agents"].unique())

    # Define methods and initialize a list for aggregated rows.
    methods = ["flowtime_pp", "flowtime_cbs", "flowtime_trained_pp"]
    aggregated_rows = []

    # Print header for aligned output
    header = f"{'num_agents':>10} | " + " | ".join(f"{m:>30}" for m in methods)
    print(header)
    print("-" * len(header))

    for n in num_agents:
        subdf = grouped.get_group(n)
        row_stats = {"num_agents": n}
        stats_print = []
        for m in methods:
            m_mean = subdf[m].mean()
            m_min = subdf[m].min()
            m_max = subdf[m].max()
            row_stats[f"{m}_mean"] = m_mean
            row_stats[f"{m}_min"] = m_min
            row_stats[f"{m}_max"] = m_max
            stats_print.append(f"{m_mean:6.2f} (min:{m_min:6.2f}, max:{m_max:6.2f})")
        print(f"{n:>10} | " + " | ".join(stats_print))
        aggregated_rows.append(row_stats)

    # Create a DataFrame from the aggregated rows and save to CSV
    agg_df = pd.DataFrame(aggregated_rows)
    agg_df.to_csv(out_csv, index=False)
    print(f"\nAggregated results saved to {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results from a CSV file and save the statistics."
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="outputs/experiment_results.csv",
        help="Path to the CSV file containing raw experiment results.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="outputs/aggregated_results.csv",
        help="Path to save the aggregated results CSV. Defaults to outputs/aggregated_results.csv.",
    )
    args = parser.parse_args()
    aggregate_experiment_results(args.csv_file, args.out_csv)


if __name__ == "__main__":
    main()
