from test_utils import MockAgent

from flatland_blackbox.solvers.cbs import CBSSolver
from flatland_blackbox.solvers.pp import PrioritizedPlanningSolver
from flatland_blackbox.train import (
    apply_learned_weights,
    train_and_apply_weights,
)
from flatland_blackbox.utils.graph_utils import (
    get_rail_subgraph,
    visualize_graph_weights,
)


def test_learned_weights_suboptimal(two_trains_suboptimal_graph):
    """
    Trains edge weights so that the final PP solution (with updated costs)
    matches the optimal CBS plan for the 2 trains scenario.
    Then checks final path lengths against the known optimal solution (CBS).
    """
    # Build the base rail subgraph from the fixture (cost=1 since "l"=1)
    G = get_rail_subgraph(two_trains_suboptimal_graph)

    a0 = MockAgent(0, (4, 2), 0, (0, 1))
    a1 = MockAgent(1, (1, 0), 0, (4, 1))
    agents = [a0, a1]

    cbs_plan = CBSSolver(G).solve(agents)
    pp_plan_orig = PrioritizedPlanningSolver(G).solve(agents)

    G_updated, final_pp_solution = train_and_apply_weights(
        G, agents, cbs_plan, iters=50, lr=0.01, lam=3.0
    )

    assert (
        0 in final_pp_solution and 1 in final_pp_solution
    ), "Missing agent keys in final plan."
    path0 = final_pp_solution[0]
    path1 = final_pp_solution[1]
    assert len(path0) > 0 and len(path1) > 0, "One or both final paths are empty."

    print("\n=== Original PP Plan (weights=1) ===")
    for ag, path in pp_plan_orig.items():
        coords = [(n.row, n.col) for n, _ in path]
        print(f"Agent {ag}, length={len(path)}, coords={coords}")

    print("\n=== CBS Expert Plan (weights=1) ===")
    for ag, path in cbs_plan.items():
        coords = [(n.row, n.col) for n, _ in path]
        print(f"Agent {ag}, length={len(path)}, coords={coords}")

    print("\n=== Updated PP Plan (learned weights) ===")
    for ag_id, path in final_pp_solution.items():
        coords = [(n.row, n.col) for n, _ in path]
        print(f"Agent {ag_id}, length={len(path)}, coords={coords}")

    # Final node checks
    agent0_end = path0[-1][0]
    agent1_end = path1[-1][0]
    assert (agent0_end.row, agent0_end.col) == (0, 1), "Agent 0 not at (0,1)"
    assert (agent1_end.row, agent1_end.col) == (4, 1), "Agent 1 not at (4,1)"

    # Compare final path lengths
    assert len(path0) == 8, f"Agent 0 path length mismatch"
    assert len(path1) == 5, f"Agent 1 path length mismatch"

    flow_time = len(path0) + len(path1)
    makespan = max(len(path0), len(path1))
    assert flow_time == 13, "Flow time mismatch"
    assert makespan == 8, "Makespan mismatch"

    show_graphs = False
    if show_graphs:
        # G_updated = apply_learned_weights(G, final_w)
        visualize_graph_weights(G_updated, "Updated Weights")
