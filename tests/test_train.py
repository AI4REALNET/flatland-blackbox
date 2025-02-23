from test_utils import MockAgent

from flatland_blackbox.solvers.cbs import CBSSolver
from flatland_blackbox.solvers.pp import PrioritizedPlanningSolver
from flatland_blackbox.train import (
    train_and_apply_weights,
)
from flatland_blackbox.utils import (
    add_proxy_nodes,
    filter_proxy_nodes,
    get_col,
    get_rail_subgraph,
    get_row,
    visualize_graph_weights,
)


def test_learned_weights_suboptimal(two_trains_suboptimal_graph):
    """
    Trains edge weights so that the final PP solution (with updated costs)
    matches the optimal CBS plan for the 2 trains scenario.
    Then checks final path lengths against the known optimal solution (CBS).
    """
    # Build the base rail subgraph from the fixture (cost=1 since "l"=1)
    # G = get_rail_subgraph(two_trains_suboptimal_graph)

    a0 = MockAgent(0, (4, 2), (0, 1))
    a1 = MockAgent(1, (1, 0), (4, 1))
    agents = [a0, a1]

    # Get the rail subgraph from your test graph.
    planning_env = get_rail_subgraph(two_trains_suboptimal_graph)
    # Add proxy nodes for the agents.
    solver_graph = add_proxy_nodes(planning_env, agents)

    cbs_plan = CBSSolver(solver_graph).solve(agents)
    pp_plan_orig = PrioritizedPlanningSolver(solver_graph).solve(agents)

    solver_graph_updated, pp_plan_trained = train_and_apply_weights(
        solver_graph, agents, cbs_plan, iters=50, lr=0.01, lam=3.0
    )

    # Only filter out proxy ndoes CBS plan after training
    cbs_plan_filtered = filter_proxy_nodes(cbs_plan)
    pp_plan_orig_filtered = filter_proxy_nodes(pp_plan_orig)

    pp_plan_trained_filtered = filter_proxy_nodes(pp_plan_trained)

    assert (
        0 in pp_plan_trained_filtered and 1 in pp_plan_trained_filtered
    ), "Missing agent keys in final plan."

    path0 = pp_plan_trained_filtered[0]
    path1 = pp_plan_trained_filtered[1]
    assert len(path0) > 0 and len(path1) > 0, "One or both final paths are empty."

    print("\n=== Original PP Plan (weights=1) ===")
    for ag, path in pp_plan_orig_filtered.items():
        coords = [(get_row(n), get_col(n)) for n, _ in path]
        print(f"Agent {ag}, length={len(path)}, coords={coords}")

    print("\n=== CBS Expert Plan (weights=1) ===")
    for ag, path in cbs_plan_filtered.items():
        coords = [(get_row(n), get_col(n)) for n, _ in path]
        print(f"Agent {ag}, length={len(path)}, coords={coords}")

    print("\n=== Updated PP Plan (learned weights) ===")
    for ag_id, path in pp_plan_trained_filtered.items():
        coords = [(get_row(n), get_col(n)) for n, _ in path]
        print(f"Agent {ag_id}, length={len(path)}, coords={coords}")

    # Final node checks
    agent0_end = path0[-1][0]
    agent1_end = path1[-1][0]
    assert (get_row(agent0_end), get_col(agent0_end)) == (0, 1), "Agent 0 not at (0,1)"
    assert (get_row(agent1_end), get_col(agent1_end)) == (4, 1), "Agent 1 not at (4,1)"

    # Compare final path lengths
    assert len(path0) == 8, f"Agent 0 path length mismatch"
    assert len(path1) == 5, f"Agent 1 path length mismatch"

    flow_time = len(path0) + len(path1)
    makespan = max(len(path0), len(path1))
    assert flow_time == 13, "Flow time mismatch"
    assert makespan == 8, "Makespan mismatch"

    show_graphs = False
    if show_graphs:
        visualize_graph_weights(solver_graph_updated, "Updated Weights")
