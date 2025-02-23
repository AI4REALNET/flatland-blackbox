import csv
import os
from itertools import product

import networkx as nx
from flatland.graphs.graph_utils import RailEnvGraph
from tqdm import tqdm

from flatland_blackbox.solvers.cbs import CBSSolver
from flatland_blackbox.solvers.pp import PrioritizedPlanningSolver
from flatland_blackbox.train import train_and_apply_weights
from flatland_blackbox.utils import (
    NoSolutionError,
    add_proxy_nodes,
    check_no_collisions,
    filter_proxy_nodes,
    get_col,
    get_rail_subgraph,
    get_row,
    initialize_environment,
    plot_agent_subgraphs,
    print_agents_start,
    shift_overlapping_edts,
)


def run_single_experiment(plot=True, **kwargs):
    """Runs a single experiment for a given environment configuration.

    Modes:
      - "pp" or "cbs": Run the specified solver on a graph with all edge costs set to 1.
      - "trained": Run CBS and PP on the cost=1 graph to get reference solutions, then train
        edge multipliers and run PP with the updated costs.

    The function performs environment setup, agent sorting (with overlapping EDT shifting),
    solver execution, and collision filtering.

    Args:
        **kwargs: A dictionary containing keys:
          - seed (int): Random seed.
          - width (int): Environment width.
          - height (int): Environment height.
          - num_agents (int): Number of agents.
          - max_cities (int): Maximum number of cities.
          - solver (str): One of "pp", "cbs", or "trained".
          - iters (int): Training iterations (used if solver == "trained").
          - lr (float): Learning rate for training.
          - lam (float): Lambda parameter for the differentiable solver.

    Returns:
        tuple: A triple (flow_time_pp, flow_time_cbs, flow_time_fpp) where each is an integer
               representing the total flow time for the PP, CBS, and final (trained) PP plans respectively.
               If no solution is found, returns (None, None, None).
    """
    seed = kwargs["seed"]
    width = kwargs["width"]
    height = kwargs["height"]
    num_agents = kwargs["num_agents"]
    max_cities = kwargs["max_cities"]
    solver_type = kwargs["solver"]
    iters_ = kwargs["iters"]
    lr_ = kwargs["lr"]
    lam_ = kwargs["lam"]

    env = initialize_environment(
        seed=seed,
        width=width,
        height=height,
        num_agents=num_agents,
        max_num_cities=max_cities,
    )
    agents = env.agents
    rail_env_graph = RailEnvGraph(env)
    G_rail = rail_env_graph.graph_rail_grid()  # Or use reduce_simple_paths()
    nx_rail_graph = get_rail_subgraph(G_rail)
    solver_graph = add_proxy_nodes(nx_rail_graph, agents)

    # Sort agents by earliest departure time and shift overlapping EDTs.
    sorted_agents = sorted(agents, key=lambda a: a.earliest_departure)
    sorted_agents_shifted = shift_overlapping_edts(sorted_agents)
    print_agents_start(sorted_agents_shifted)

    # Initialize solution variables.
    pp_plan, cbs_plan, final_plan, G_rail_updated = None, None, None, None

    if solver_type in ("pp", "cbs"):
        print(f"Running {solver_type}...")
        # Run PP or CBS on cost=1 edges
        final_plan = run_solver(solver_graph, sorted_agents_shifted, solver_type)
        final_plan = filter_proxy_nodes(final_plan)
        check_no_collisions(final_plan)
        print_agent_paths(final_plan, solver_type)
        print("Flow time: ", compute_flowtime(final_plan))
        generate_and_plot_agent_subgraphs(env, G_rail, final_plan, solver_type)
        return None, None, None

    elif solver_type == "trained":
        try:
            print("Running cbs...")
            cbs_plan = run_solver(solver_graph, sorted_agents_shifted, "cbs")
        except NoSolutionError as e:
            print(f"[WARNING] Skipping run because CBS did not find a solution: {e}")
            return None, None, None

        print("Running pp...")
        # Run PP on cost=1 edges
        pp_plan = run_solver(solver_graph, sorted_agents_shifted, "pp")

        G_rail_updated, final_plan = train_and_apply_weights(
            solver_graph,
            sorted_agents_shifted,
            cbs_plan,
            iters=iters_,
            lr=lr_,
            lam=lam_,
        )
        # Filter the proxy nodes out
        pp_plan = filter_proxy_nodes(pp_plan)
        cbs_plan = filter_proxy_nodes(cbs_plan)
        final_plan = filter_proxy_nodes(final_plan)

        # Check for collisions.
        check_no_collisions(pp_plan)
        check_no_collisions(cbs_plan)
        check_no_collisions(final_plan)

        # Compute flow times.
        flow_time_pp = compute_flowtime(pp_plan)
        flow_time_cbs = compute_flowtime(cbs_plan)
        flow_time_fpp = compute_flowtime(final_plan)

        # Print paths and flow time results.
        print_agent_paths(pp_plan, solver_type)
        print_agent_paths(cbs_plan, solver_type)
        print_agent_paths(final_plan, solver_type)

        print(
            f"\n### Flow times: pp: {flow_time_pp}  cbs: {flow_time_cbs}  pp final: {flow_time_fpp}"
        )

        assert flow_time_pp >= flow_time_cbs, "PP flow time is lower than CBS flow time"
        assert (
            flow_time_fpp >= flow_time_cbs
        ), "Learned PP flow time is lower than CBS flow time"

        if flow_time_pp < flow_time_fpp:
            print("[WARNING] Final pp flow time is higher than original PP flow time.")

        if plot:
            generate_and_plot_agent_subgraphs(env, G_rail, pp_plan, "pp")
            generate_and_plot_agent_subgraphs(env, G_rail, cbs_plan, "cbs")
            generate_and_plot_agent_subgraphs(
                env, G_rail_updated, final_plan, "trained"
            )

        return flow_time_pp, flow_time_cbs, flow_time_fpp


def run_experiments(**kwargs):
    """Runs a series of experiments over multiple environment configurations.

    For each configuration, it computes:
      - The cost=1 CBS plan (reference).
      - The cost=1 PP plan (reference).
      - The final trained PP plan (with learned edge costs).
    Flow time is computed for each configuration.

    Args:
        **kwargs: A dictionary with experiment parameters including:
            start_seed (int): Starting seed.
            num_seeds (int): Number of seeds to run.
            num_agents_list (list[int]): List of agent counts to test.
            max_cities_list (list[int]): List of maximum city counts to test.
            width_list (list[int]): List of map widths to test.
            height_list (list[int]): List of map heights to test.
            iters (int): Training iterations.
            lr (float): Learning rate.
            lam (float): Lambda parameter.

    Returns:
        list[dict]: A list of result dictionaries for each experiment configuration.
    """
    results = []
    seeds = range(kwargs["start_seed"], kwargs["start_seed"] + kwargs["num_seeds"])

    all_configs = list(
        product(
            seeds,
            kwargs["num_agents_list"],
            kwargs["max_cities_list"],
            kwargs["width_list"],
            kwargs["height_list"],
        )
    )
    total = len(all_configs)

    for seed, num_agents, max_cities, width, height in tqdm(
        all_configs, total=total, desc="Experiments"
    ):
        print(
            f"Config: seed={seed}, num_agents={num_agents}, max_cities={max_cities}, width={width}, height={height}"
        )

        flow_time_pp, flow_time_cbs, flow_time_fpp = run_single_experiment(
            plot=False,
            seed=seed,
            width=width,
            height=height,
            num_agents=num_agents,
            max_cities=max_cities,
            solver="trained",
            iters=kwargs["iters"],
            lr=kwargs["lr"],
            lam=kwargs["lam"],
        )
        result = {
            "seed": seed,
            "num_agents": num_agents,
            "max_cities": max_cities,
            "width": width,
            "height": height,
            "flowtime_pp": flow_time_pp,
            "flowtime_cbs": flow_time_cbs,
            "flowtime_trained_pp": flow_time_fpp,
        }
        results.append(result)
    return results


def compute_flowtime(plan_dict):
    """Computes the total flow time of a plan.

    Flow time is defined as the sum of the durations of each agent's path.

    Args:
        plan_dict (dict): Mapping from agent_id to a list of (node, time) tuples.

    Returns:
        int or None: The total flow time if paths exist; otherwise, None.
    """
    durations = []
    for path in plan_dict.values():
        if path:
            start_time = path[0][1]
            end_time = path[-1][1]
            durations.append(end_time - start_time)
    flow_time = sum(durations) if durations else None
    return flow_time


def run_solver(graph, agents, solver_type):
    """Runs the specified solver on the given graph.

    Args:
        graph (nx.Graph): The graph to solve on.
        agents (list): List of agent objects.
        solver_type (str): The solver to run, either "pp" or "cbs".

    Returns:
        dict: Mapping from agent_id to a list of (node, time) tuples.

    Raises:
        NoSolutionError: If the solver cannot find a solution.
    """
    if solver_type == "pp":
        solver = PrioritizedPlanningSolver(graph)
    elif solver_type == "cbs":
        solver = CBSSolver(graph)
    else:
        raise ValueError(f"Unknown solver_type: {solver_type}")

    try:
        plan = solver.solve(agents)
    except NoSolutionError as e:
        raise NoSolutionError(f"No solution found by {solver_type}: {e}") from e

    return plan


def print_agent_paths(plan, solver_name):
    """Prints a summary of each agent's path.

    Args:
        plan (dict): Mapping from agent_id to a list of (node, time) tuples.
        solver_name (str): Identifier for the solver (e.g., "pp", "cbs", "trained").

    Raises:
        AssertionError: If any agent's times are not strictly increasing.
    """
    print(f"### {solver_name} path: ")
    for agent_id, path in plan.items():
        coords = [((int(get_row(n)), int(get_col(n))), t) for n, t in path]
        assert all(
            path[i + 1][1] > path[i][1] for i in range(len(path) - 1)
        ), f"Agent {agent_id} has non‐strictly‐increasing times!"
        first_timestep = coords[0][1]
        final_timestep = coords[-1][1]
        path_duration = final_timestep - first_timestep
        print(
            f"Agent {agent_id}: duration={path_duration} ({first_timestep}->{final_timestep}), coords={coords}"
        )


def generate_and_plot_agent_subgraphs(env, G_rail, solution, solver_type):
    """Generates and plots subgraphs for each agent's path.

    Args:
        env (Environment): The Flatland environment.
        G_rail (nx.Graph): The rail subgraph.
        solution (dict): Mapping from agent_id to a list of (node, time) tuples.
        solver_type (str): Identifier for the solver (used in naming output files).
    """
    G_paths_subgraphs = {}
    for a_id, path in solution.items():
        only_nodes = [p[0] for p in path]
        G_sub = nx.induced_subgraph(G_rail, only_nodes)
        G_paths_subgraphs[a_id] = G_sub

    print(f"Generating {solver_type} plots ...")
    plot_agent_subgraphs(
        env, G_paths_subgraphs, save_fig_folder=f"outputs/{solver_type}"
    )


def write_results_to_csv(results, filename):
    """Writes experiment results to a CSV file.

    Args:
        results (list[dict]): List of result dictionaries.
        filename (str): Path to the output CSV file.
    """
    if not results:
        print("No results to write.")
        return
    keys = list(results[0].keys())
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to {filename}")
