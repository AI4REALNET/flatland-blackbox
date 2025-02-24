import argparse
import os

import pandas as pd


def aggregate_experiment_results(
    csv_file,
    out_csv_overall="outputs/aggregated_overall.csv",
    out_csv_filtered="outputs/aggregated_filtered.csv",
):
    """
    Reads experiment results, computes overall mean flow times for each method (PP, CBS, Trained PP)
    grouped by num_agents, and also computes the mean flow times for only the cases where PP flow time
    is different from CBS flow time.

    The function prints the results in aligned tables and saves them to two separate CSV files.

    Args:
        csv_file (str): Path to the CSV file with raw experiment results.
        out_csv_overall (str, optional): Path to save the overall aggregated CSV. Defaults to "outputs/aggregated_overall.csv".
        out_csv_filtered (str, optional): Path to save the filtered aggregated CSV. Defaults to "outputs/aggregated_filtered.csv".

    Returns:
        None
    """
    # Read the raw CSV file.
    df = pd.read_csv(csv_file)

    # Ensure output directories exist.
    out_dir_overall = os.path.dirname(out_csv_overall)
    out_dir_filtered = os.path.dirname(out_csv_filtered)
    os.makedirs(out_dir_overall, exist_ok=True)
    os.makedirs(out_dir_filtered, exist_ok=True)

    # Group by num_agents.
    grouped = df.groupby("num_agents")
    num_agents = sorted(df["num_agents"].unique())

    methods = ["flowtime_pp", "flowtime_cbs", "flowtime_trained_pp"]

    overall_rows = []
    filtered_rows = []

    # Print header for overall statistics.
    overall_header = f"{'num_agents':>10} | " + " | ".join(
        f"{m+'_mean':>20}" for m in methods
    )
    print("\nOverall Mean Flow Times:")
    print(overall_header)
    print("-" * len(overall_header))

    # Overall statistics.
    for n in num_agents:
        subdf = grouped.get_group(n)
        row = {"num_agents": n}
        stats_print = []
        for m in methods:
            m_mean = subdf[m].mean()
            row[m + "_mean"] = m_mean
            stats_print.append(f"{m_mean:20.2f}")
        print(f"{n:>10} | " + " | ".join(stats_print))
        overall_rows.append(row)

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out_csv_overall, index=False)
    print(f"\nOverall aggregated results saved to {out_csv_overall}")

    # Now filtered statistics: only consider rows where flowtime_pp != flowtime_cbs.
    df_filtered = df[df["flowtime_pp"] != df["flowtime_cbs"]]
    if df_filtered.empty:
        print("\nNo filtered cases found (where PP flow time differs from CBS).")
    else:
        filtered_grouped = df_filtered.groupby("num_agents")
        print("\nFiltered Mean Flow Times (only when PP != CBS):")
        filtered_header = f"{'num_agents':>10} | " + " | ".join(
            f"{m+'_mean':>20}" for m in methods
        )
        print(filtered_header)
        print("-" * len(filtered_header))

        for n in num_agents:
            if n in filtered_grouped.groups:
                subdf = filtered_grouped.get_group(n)
                row = {"num_agents": n}
                stats_print = []
                for m in methods:
                    m_mean = subdf[m].mean()
                    row[m + "_mean"] = m_mean
                    stats_print.append(f"{m_mean:20.2f}")
                print(f"{n:>10} | " + " | ".join(stats_print))
                filtered_rows.append(row)
        filtered_df = pd.DataFrame(filtered_rows)
        filtered_df.to_csv(out_csv_filtered, index=False)
        print(f"\nFiltered aggregated results saved to {out_csv_filtered}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate experiment results from a CSV file and save aggregate statistics."
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="outputs/experiment_results.csv",
        help="Path to the CSV file with raw experiment results.",
    )
    parser.add_argument(
        "--out_csv_overall",
        type=str,
        default="outputs/aggregated_overall.csv",
        help="Output CSV file for overall aggregated results.",
    )
    parser.add_argument(
        "--out_csv_filtered",
        type=str,
        default="outputs/aggregated_filtered.csv",
        help="Output CSV file for filtered aggregated results (only when PP != CBS).",
    )
    args = parser.parse_args()
    aggregate_experiment_results(
        args.csv_file, args.out_csv_overall, args.out_csv_filtered
    )


if __name__ == "__main__":
    main()
