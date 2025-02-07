import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from flatland_blackbox.models import DifferentiableSolver, EdgeWeightParam
from flatland_blackbox.solvers.cbs import CBSSolver
from flatland_blackbox.solvers.pp import PrioritizedPlanningSolver
from flatland_blackbox.utils.graph_utils import visualize_graph
from tests.test_utils import MockAgent


def two_trains_suboptimal_graph():
    G = nx.DiGraph()
    nodes = [
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 2, 0),
        (1, 3, 0),
        (2, 1, 0),
        (2, 3, 0),
        (3, 1, 0),
        (3, 2, 0),
        (3, 3, 0),
        (4, 1, 0),
        (4, 2, 0),
    ]

    for n in nodes:
        G.add_node(n, type="rail")

    edges = [
        ((0, 1, 0), (1, 1, 0)),
        ((1, 0, 0), (1, 1, 0)),
        ((1, 1, 0), (1, 2, 0)),
        ((1, 2, 0), (1, 3, 0)),
        ((1, 3, 0), (2, 3, 0)),
        ((2, 3, 0), (3, 3, 0)),
        ((1, 1, 0), (2, 1, 0)),
        ((2, 1, 0), (3, 1, 0)),
        ((3, 1, 0), (4, 1, 0)),
        ((3, 1, 0), (3, 2, 0)),
        ((3, 2, 0), (4, 2, 0)),
        ((3, 2, 0), (3, 3, 0)),
    ]

    for u, v in edges:
        G.add_edge(u, v, type="dir", l=1)
        G.add_edge(v, u, type="dir", l=1)

    return G


def index_edges(G):
    edgelist = sorted(G.edges())
    edge_to_idx = {}

    for i, e in enumerate(edgelist):
        edge_to_idx[e] = i

    return edgelist, edge_to_idx


def cbs_expert_plan(G, agents, edge_to_idx):
    H = G.copy()
    for u, v in H.edges():
        H[u][v]["l"] = 1.0
    plan = CBSSolver(H).solve(agents)
    usage = plan_usage(plan, edge_to_idx)
    return plan, usage


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


def pp_plan_dict(G, agents):
    return PrioritizedPlanningSolver(G).solve(agents)


def pp_solver_fn(w_np, G, edge_to_idx, agents):
    H = G.copy()

    for u, v in H.edges():
        idx = edge_to_idx[(u, v)]
        cost = max(w_np[idx], 1e-6)
        H[u][v]["l"] = cost

    plan = pp_plan_dict(H, agents)
    return plan_usage(plan, edge_to_idx)


def train_on_environment(G, agents, iters=50, lr=1e-2, lam=3.0):
    edgelist, edge_to_idx = index_edges(G)
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
            print(f"Step={step}, loss={loss.item():.3f}")

    final_w = model().detach().cpu().numpy()
    return final_w, cbs_plan


def run_training():
    G = two_trains_suboptimal_graph()

    a0 = MockAgent(0, (4, 2), 0, (0, 1))
    a1 = MockAgent(1, (1, 0), 0, (4, 1))
    agents = [a0, a1]

    # Original PP plan
    for u, v in G.edges():
        G[u][v]["l"] = 1.0

    orig_plan = pp_plan_dict(G, agents)
    print("\n=== Original PP Plan ===")

    for ag, path in orig_plan.items():
        coords = [(n.row, n.col) for n, _ in path]
        print(f"Agent {ag}, length={len(path)}, coords={coords}")

    print("\n=== Training... ===")
    w_final, cbs_plan = train_on_environment(G, agents)

    # Apply final
    H = G.copy()
    for u, v in H.edges():
        idx = index_edges(G)[1][(u, v)]
        cost = max(w_final[idx], 1e-6)
        H[u][v]["l"] = cost

    new_plan = pp_plan_dict(H, agents)

    print("\n=== CBS Expert Plan (cost=1) ===")
    for ag, path in cbs_plan.items():
        coords = [(n.row, n.col) for n, _ in path]
        print(f"Agent {ag}, length={len(path)}, coords={coords}")

    print("\n=== Updated PP Plan (learned weights) ===")
    for ag, path in new_plan.items():
        coords = [(n.row, n.col) for n, _ in path]
        print(f"Agent {ag}, length={len(path)}, coords={coords}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    visualize_graph(G, "Original Weights", ax=axes[0])
    visualize_graph(H, "Updated Weights", ax=axes[1])
    plt.show()


if __name__ == "__main__":
    run_training()
