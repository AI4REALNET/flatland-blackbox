import argparse

import networkx as nx
import numpy as np
import torch
import torch.optim as optim
from flatland.graphs.graph_utils import RailEnvGraph

from flatland_blackbox.models import DifferentiableSolver, EdgeWeightParam
from flatland_blackbox.solvers.cbs import CBSSolver
from flatland_blackbox.solvers.pp import PrioritizedPlanningSolver
from flatland_blackbox.utils.graph_utils import (
    add_proxy_nodes_for_agents,
    check_no_collisions,
    get_rail_subgraph,
    preprocess_departure_times,
)
from flatland_blackbox.utils.run_utils import (
    initialize_environment,
    plot_agent_subgraphs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run or train on a Flatland environment."
    )
    parser.add_argument(
        "--mode",
        choices=["solve", "train"],
        default="solve",
        help="Which mode to run: either 'solve' or 'train'.",
    )
    parser.add_argument(
        "--solver",
        choices=["pp", "cbs"],
        default="pp",
        help="Which solver to run in 'solve' mode.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--width", type=int, default=30, help="Environment width.")
    parser.add_argument("--height", type=int, default=30, help="Environment height.")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents.")
    parser.add_argument(
        "--max_cities", type=int, default=3, help="Max number of cities."
    )
    parser.add_argument(
        "--iters", type=int, default=50, help="Training iterations if 'train' mode."
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate for training."
    )
    parser.add_argument(
        "--lam", type=float, default=3.0, help="Lambda for differentiable solver."
    )

    return parser.parse_args()


################################################################################
# HELPER LOGIC FOR TRAINING
################################################################################


def plan_usage(plan_dict, edge_to_idx):
    E = max(edge_to_idx.values()) + 1
    usage = np.zeros(E, dtype=np.float32)
    for _, path in plan_dict.items():
        for i in range(len(path) - 1):
            n0, _ = path[i]
            n1, _ = path[i + 1]
            if (n0, n1) in edge_to_idx:
                usage[edge_to_idx[(n0, n1)]] = 1.0
    return usage


def cbs_expert_plan(G, agents, edge_to_idx):
    H = G.copy()
    for u, v in H.edges():
        H[u][v]["l"] = 1.0
    plan_dict = CBSSolver(H).solve(agents)
    usage_vec = plan_usage(plan_dict, edge_to_idx)
    return plan_dict, usage_vec


def pp_plan_dict(G, agents):
    return PrioritizedPlanningSolver(G).solve(agents)


def pp_solver_fn(w_np, G, edge_to_idx, agents):
    H = G.copy()
    for u, v in H.edges():
        idx = edge_to_idx[(u, v)]
        cost = max(w_np[idx], 1e-6)
        H[u][v]["l"] = cost
    plan_dict = pp_plan_dict(H, agents)
    return plan_usage(plan_dict, edge_to_idx)


def train_on_flatland_env(G, agents, iters=50, lr=0.01, lam=3.0):
    edgelist = sorted(G.edges())
    edge_to_idx = {e: i for i, e in enumerate(edgelist)}

    cbs_plan, cbs_usage = cbs_expert_plan(G, agents, edge_to_idx)

    model = EdgeWeightParam(len(edgelist))
    opt = optim.Adam(model.parameters(), lr=lr)

    def solver_forward(w_np):
        return pp_solver_fn(w_np, G, edge_to_idx, agents)

    expert_t = torch.from_numpy(cbs_usage).float()

    for step in range(iters):
        opt.zero_grad()
        w = model()
        plan_tensor = DifferentiableSolver.apply(w, solver_forward, lam)
        loss = torch.sum(torch.abs(plan_tensor - expert_t))
        loss.backward()
        opt.step()

        if step % 5 == 0:
            print(f"Train step={step}, loss={loss.item():.3f}")

    final_w = model().detach().cpu().numpy()
    return final_w, cbs_plan


################################################################################
# MAIN LOGIC
################################################################################


def main():
    args = parse_args()

    env, renderer = initialize_environment(
        seed=args.seed,
        width=args.width,
        height=args.height,
        num_agents=args.num_agents,
        max_num_cities=args.max_cities,
    )
    env.reset()
    agents = env.agents

    rail_env_graph = RailEnvGraph(env)

    # G_rail = rail_env_graph.graph_rail_grid()
    G_rail = rail_env_graph.reduce_simple_paths()  # edge weights in "l"
    # Keep only 'rail' nodes and 'dir' edges
    nx_rail_graph = get_rail_subgraph(G_rail)
    solver_rail_graph = add_proxy_nodes_for_agents(nx_rail_graph, agents)

    sorted_agents_edt_original = sorted(agents, key=lambda a: a.earliest_departure)
    # Shift overlapping edts
    sorted_agents_edt = preprocess_departure_times(sorted_agents_edt_original)

    for agent in sorted_agents_edt:
        start_rc = tuple(map(int, agent.initial_position))
        end_rc = tuple(map(int, agent.target))
        print(
            f"Agent {agent.handle} start: {start_rc}"
            f" end: {end_rc} edt: {agent.earliest_departure}"
        )

    if args.mode == "solve":
        if args.solver == "pp":
            print("Planning using PP ...")
            solver = PrioritizedPlanningSolver(solver_rail_graph)
        else:
            print("Planning using CBS ...")
            solver = CBSSolver(solver_rail_graph)

        solution_dict = solver.solve(sorted_agents_edt)

        # Filter proxy start node with dir=-1
        solution_dict_filtered = {
            agent_id: [(n, t) for (n, t) in agent_path if n.direction != -1]
            for agent_id, agent_path in solution_dict.items()
        }

        G_paths_subgraphs = []

        for agent_id, path in solution_dict_filtered.items():
            agent_path = solution_dict[agent.handle]

            assert all(
                path[i + 1][1] > path[i][1] for i in range(len(path) - 1)
            ), f"Agent {agent_id} has non‐strictly‐increasing times!"

            coords = [((int(n.row), int(n.col)), t) for n, t in path]
            path_duration = coords[-1][1] - agent.earliest_departure
            print(f"Agent {agent_id}: duration={path_duration}, coords={coords}")

            # Generate sub-graph
            only_nodes = [p[0] for p in path]
            G_sub = nx.induced_subgraph(G_rail, only_nodes)
            G_paths_subgraphs.append(G_sub)

        check_no_collisions(solution_dict_filtered)

        plot_agent_subgraphs(
            env, G_paths_subgraphs, agents, save_fig_folder=f"outputs/{args.solver}"
        )

    else:
        print("=== TRAIN MODE: Using CBS as expert, training PP weights ===")

        # For training, we use the entire Nx graph in its current state (with cost=1).
        # Then we run cbs_expert_plan => use blackbox approach => get final weights
        final_w, cbs_plan = train_on_flatland_env(
            solver_rail_graph, agents, iters=args.iters, lr=args.lr, lam=args.lam
        )

        # Apply the final weights
        H = solver_rail_graph.copy()
        edgelist = sorted(H.edges())
        edge_to_idx = {e: i for i, e in enumerate(edgelist)}

        for u, v in H.edges():
            idx = edge_to_idx[(u, v)]
            cost = max(final_w[idx], 1e-6)
            H[u][v]["l"] = cost

        # Run PP with updated weights
        solver_updated = PrioritizedPlanningSolver(H)
        new_plan = solver_updated.solve(agents)

        # check_no_collisions(new_plan)

        G_paths_subgraphs = []
        for agent in agents:
            agent_path = new_plan[agent.handle]
            filtered_path = [(n, t) for (n, t) in agent_path if n.direction != -1]
            if not filtered_path:
                print(
                    f"[WARNING] Agent {agent.handle} path empty after removing proxies."
                )
                continue
            only_nodes = [p[0] for p in filtered_path]
            G_sub = nx.induced_subgraph(G_rail, only_nodes)
            G_paths_subgraphs.append(G_sub)

        print("=== Expert CBS plan ===")
        for ag, path in cbs_plan.items():
            coords = [(int(n.row), int(n.col)) for n, _ in path]
            print(f"Agent {ag}: length={len(path)}, coords={coords}")

        print("=== Updated PP plan (trained) ===")
        for ag, path in new_plan.items():
            coords = [(int(n.row), int(n.col)) for n, _ in path]
            print(f"Agent {ag}: length={len(path)}, coords={coords}")

        plot_agent_subgraphs(
            env, G_paths_subgraphs, agents, save_fig_folder="outputs/trained_pp"
        )


if __name__ == "__main__":
    main()
