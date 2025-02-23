import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # use non-interactive backend
import matplotlib.pyplot as plt


def plot_experiment_results(csv_file, out_dir="outputs/plots"):
    """
    Reads experiment results from a CSV file, computes summary statistics (min, mean, max),
    prints the statistics with aligned formatting, and generates a line plot with asymmetric error bars
    for flow times per method for each number of agents configuration.

    Args:
        csv_file (str): Path to the CSV file containing experiment results.
        out_dir (str, optional): Directory to save the generated plot. Defaults to "outputs/plots".

    Returns:
        None
    """
    # Read CSV file and ensure output directory exists.
    df = pd.read_csv(csv_file)
    os.makedirs(out_dir, exist_ok=True)

    # Group by num_agents and get sorted unique values.
    grouped = df.groupby("num_agents")
    num_agents = sorted(df["num_agents"].unique())

    methods = ["flowtime_pp", "flowtime_cbs", "flowtime_trained_pp"]
    labels = ["PP", "CBS", "Trained PP"]

    # Dictionaries to store statistics per method
    means = {m: [] for m in methods}
    mins = {m: [] for m in methods}
    maxs = {m: [] for m in methods}

    # Print header for statistics with alignment
    header = f"{'num_agents':>10} | " + " | ".join(f"{label:>15}" for label in labels)
    print(header)
    print("-" * len(header))

    for n in num_agents:
        subdf = grouped.get_group(n)
        stat_strs = []
        for m, label in zip(methods, labels):
            m_val = subdf[m].mean()
            min_val = subdf[m].min()
            max_val = subdf[m].max()
            means[m].append(m_val)
            mins[m].append(min_val)
            maxs[m].append(max_val)
            # Format: mean (min, max)
            stat_strs.append(f"{m_val:6.2f} ({min_val:6.2f}, {max_val:6.2f})")
        print(f"{n:>10} | " + " | ".join(stat_strs))

    # Now create a line plot with asymmetric error bars.
    plt.figure(figsize=(10, 6))
    x = np.array(num_agents)  # discrete x-axis

    markers = {"flowtime_pp": "o", "flowtime_cbs": "s", "flowtime_trained_pp": "D"}
    colors = {
        "flowtime_pp": "blue",
        "flowtime_cbs": "green",
        "flowtime_trained_pp": "red",
    }

    for m in methods:
        mean_vals = np.array(means[m])
        min_vals = np.array(mins[m])
        max_vals = np.array(maxs[m])
        # Calculate asymmetric errors: lower error and upper error.
        lower_err = mean_vals - min_vals
        upper_err = max_vals - mean_vals
        yerr = np.vstack((lower_err, upper_err))

        plt.errorbar(
            x,
            mean_vals,
            yerr=yerr,
            marker=markers[m],
            color=colors[m],
            linestyle="-",
            label=m,
            capsize=5,
            linewidth=2,
        )

    plt.xlabel("Number of Agents")
    plt.ylabel("Flow Time")
    plt.title("Experiment Results: Flow Time vs. Number of Agents")
    plt.xticks(x)  # ensure x-axis is discrete
    plt.legend()
    plt.grid(True, axis="y", alpha=0.6)

    out_path = os.path.join(out_dir, "flow_time_vs_num_agents.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot experiment results from a CSV file as a line graph with asymmetric error bars."
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="outputs/experiment_results.csv",
        help="Path to the CSV file containing experiment results. Defaults to outputs/experiment_results.csv.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/plots",
        help="Directory to save the generated plots. Defaults to outputs/plots.",
    )
    args = parser.parse_args()
    plot_experiment_results(args.csv_file, args.out_dir)


if __name__ == "__main__":
    main()
