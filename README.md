# Flatland Blackbox

Flatland Blackbox is a multi-agent path planning framework for the Flatland train environment. The repository implements two solvers, Conflict-Based Search (CBS) and Prioritized Planning (PP) (with optional training via learned edge multipliers). It also provides scripts for running experiments over various environment configurations and for visualizing the results.

## Features

- **Multi-agent Path Planning:** Implements both optimal (CBS) and suboptimal (PP) solvers.
- **Learnable Heuristics:** The PP solver can be trained using a differentiable framework to adjust edge costs.
- **Experiments Framework:** Run experiments over multiple configurations (varying number of agents, map size, etc.) and save aggregated results.
- **Visualization Tools:** Generate plots from experiment results.

## Requirements

- Python 3.12
- See `requirements.txt` for additional package dependencies (e.g., `numpy`, `pandas`, `matplotlib`, etc.)

## Installation


**Create and Activate a Conda Environment:**

```bash
conda create --name flatland-blackbox python=3.12
conda activate flatland-blackbox
```

**Install Requirements**

```bash
pip install -r requirements.txt
```

## Testing

To check that the instalation went smoothly run all tests with:

```bash
python -m pytest
```

## Running the Experiments

The main entry point for running experiments, training, or a single solve is `main.py`. It accepts various command-line arguments to select the mode and configure parameters. For detailed usage information, run:

```bash
python main.py --help
```

**Running from the Command Line**

To run a set of experiments across multiple instances, according to parameters set in `run_configs.json`: 

```bash
python main.py --mode experiments
```

Or to run training on a single configuration:

```bash
python main.py --mode train --seed 42 --num_agents 4 --width 30 --height 30 --iters 300 --lr 0.01 --lam 3.0
```

Or simply run a solver (pp or cbs) on a specific instance:

```bash
python main.py --mode solve --solver pp --seed 42 --num_agents 4 --width 30 --height 30
```

When a single instance is run, the resulting paths can be vizualized under `/outputs`, under a the respective folder with the name of the solver used: `/pp`, `/cbs` and `/trained`.

## Plotting Experiment Results

After running experiments, you can generate summary plots with the provided script `plot_results.py`.

```bash
python plot_results.py --csv_file outputs/experiment_results.csv --out_dir outputs/plots
```