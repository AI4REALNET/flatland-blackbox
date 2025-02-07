import argparse

import networkx as nx
from flatland.graphs.graph_utils import RailEnvGraph

from flatland_blackbox.solvers.cbs import CBSSolver
from flatland_blackbox.solvers.pp import PrioritizedPlanningSolver
from flatland_blackbox.utils.graph_utils import (
    add_proxy_nodes_for_agents,
    get_rail_subgraph,
)
from flatland_blackbox.utils.run_utils import (
    get_agents_start_end,
    initialize_environment,
    plot_agent_subgraphs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solve Flatland instance with PP or CBS solver."
    )
    parser.add_argument(
        "--solver",
        choices=["pp", "cbs"],
        default="pp",
        help="Which solver to run: 'pp' or 'cbs'.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for environment creation."
    )
    parser.add_argument("--width", type=int, default=30, help="Environment width.")
    parser.add_argument("--height", type=int, default=30, help="Environment height.")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents.")
    return parser.parse_args()


def main():
    args = parse_args()
    env, _ = initialize_environment(
        seed=args.seed, width=args.width, height=args.height, num_agents=args.num_agents
    )
    env.reset()
    agents = env.agents

    rail_env_graph = RailEnvGraph(env)
    # G2 = rail_env_graph.graph_rail_grid()
    G5 = rail_env_graph.reduce_simple_paths()
    list_starts, list_ends = get_agents_start_end(G5, agents)

    print("Agent starts:", list_starts)
    print("Agent ends:", list_ends)

    nx_rail_graph = get_rail_subgraph(G5)
    solver_rail_graph = add_proxy_nodes_for_agents(nx_rail_graph, agents)

    if args.solver == "pp":
        solver = PrioritizedPlanningSolver(solver_rail_graph)
    else:
        solver = CBSSolver(solver_rail_graph)

    # Returns dict: agent.handle -> [(RailNode, time), ...]
    solution_dict = solver.solve(agents)

    G_paths_subgraphs = []

    for i, agent in enumerate(agents):
        agent_path = solution_dict[agent.handle]

        # Filter out 'proxy' nodes (direction == -1)
        filtered_path = [(node, t) for (node, t) in agent_path if node.direction != -1]
        if not filtered_path:
            print(f"[WARNING] Agent {agent.handle} path empty after removing proxies.")
            continue

        # Convert to ((row, col, dir), time)
        grid_path = [
            ((node.row, node.col, node.direction), t) for (node, t) in filtered_path
        ]

        # Basic checks: start, end
        path_start = grid_path[0][0][:2]  # (row, col)
        path_end = grid_path[-1][0][:2]  # (row, col)

        # Compare with old start/end logic
        assert (
            path_start == list_starts[i][:2]
        ), f"Mismatch start for agent {agent.handle}: {path_start} vs {list_starts[i][:2]}"

        valid_end_positions = {(r, c) for (r, c, _) in list_ends[i]}
        assert (
            path_end in valid_end_positions
        ), f"Mismatch end for agent {agent.handle}: {path_end} not in {valid_end_positions}"

        print(f"Agent {agent.handle} path length: {len(grid_path)}")

        # for ((r,c,d), t) in grid_path:
        #     print(f"  Node=({int(r)},{int(c)},{int(d)}), time={t}")

        # Build subgraph for visualization
        only_nodes = [pos_time[0] for pos_time in grid_path]  # extract (row, col, dir)
        G_sub = nx.induced_subgraph(G5, only_nodes)
        G_paths_subgraphs.append(G_sub)

    plot_agent_subgraphs(env, G_paths_subgraphs, agents)


if __name__ == "__main__":
    main()
