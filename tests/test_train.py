import matplotlib.pyplot as plt
from test_utils import MockAgent

from flatland_blackbox.train import index_edges, pp_plan_dict, train_on_environment
from flatland_blackbox.utils.graph_utils import get_rail_subgraph, visualize_graph


def test_learned_weights_suboptimal(two_trains_suboptimal_graph):
    """
    Trains edge weights so PP solution (with updated costs)
    matches the CBS plan for the 2 trains scenario.
    Then checks final path lengths against the known "CBS" solution.
    """
    G = get_rail_subgraph(two_trains_suboptimal_graph)

    a0 = MockAgent(0, (4, 2), 0, (0, 1))
    a1 = MockAgent(1, (1, 0), 0, (4, 1))
    agents = [a0, a1]

    final_weights, cbs_plan = train_on_environment(
        G, agents, iters=50, lr=0.01, lam=3.0
    )

    # Apply final weights to a copy of G
    G_updated = G.copy()
    edgelist, edge_to_idx = index_edges(G)
    for u, v in G_updated.edges():
        idx = edge_to_idx[(u, v)]
        cost = max(final_weights[idx], 1e-6)
        G_updated[u][v]["l"] = cost

    # Run PP on updated weights
    final_pp_solution = pp_plan_dict(G_updated, agents)

    # Basic checks
    assert 0 in final_pp_solution and 1 in final_pp_solution
    path0 = final_pp_solution[0]
    path1 = final_pp_solution[1]
    assert len(path0) > 0 and len(path1) > 0

    # Agents final node checks
    agent0_end = path0[-1][0]
    agent1_end = path1[-1][0]
    assert (agent0_end.row, agent0_end.col) == (0, 1), "Agent0 not at (0,1)"
    assert (agent1_end.row, agent1_end.col) == (4, 1), "Agent1 not at (4,1)"

    # Compare final path lengths
    expected_len0, expected_len1 = 8, 5
    assert len(path0) == expected_len0, f"Agent0 path length mismatch"
    assert len(path1) == expected_len1, f"Agent1 path length mismatch"

    flow_time = len(path0) + len(path1)
    makespan = max(len(path0), len(path1))
    expected_flow, expected_makespan = 13, 8
    assert flow_time == expected_flow, "Flow time mismatch"
    assert makespan == expected_makespan, "Makespan mismatch"

    print("\n=== CBS Expert Plan (weights=1) ===")
    for ag, path in cbs_plan.items():
        coords = [(n.row, n.col) for n, _ in path]
        print(f"Agent{ag}, length={len(path)}, coords={coords}")

    print("\n=== Updated PP Plan (learned weights) ===")
    for ag, path in enumerate([path0, path1]):
        coords = [(n.row, n.col) for n, _ in path]
        print(f"Agent{ag}, length={len(path)}, coords={coords}")

    # Visualize the final graph
    # fig, axes = plt.subplots(1, 2, figsize=(8,6))
    # visualize_graph(G, "Original Graph (all edges=1)", ax=axes[0])
    # visualize_graph(G_updated, "Updated Weights", ax=axes[1])
    # plt.show()
