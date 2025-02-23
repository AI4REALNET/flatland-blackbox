import numpy as np
import torch
import torch.optim as optim

from flatland_blackbox.models import DifferentiableSolver, EdgeWeightParam
from flatland_blackbox.solvers.pp import PrioritizedPlanningSolver
from flatland_blackbox.utils import NoSolutionError, is_proxy_node


def index_edges(G):
    """Returns a sorted list of edges and a dictionary mapping each edge to its index.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        tuple: A tuple (edgelist, edge_to_idx) where edgelist is a sorted list of edges,
               and edge_to_idx is a dict mapping (u, v) tuples to an integer index.
    """
    edgelist = sorted(G.edges())
    edge_to_idx = {e: i for i, e in enumerate(edgelist)}
    return edgelist, edge_to_idx


def plan_usage(plan_dict, edge_to_idx):
    """Builds a usage array indicating the frequency of each edge in the plan.

    The usage array has shape [num_edges], where each element is incremented by 1.0
    every time the corresponding edge appears consecutively in any agent's path.

    Args:
        plan_dict (dict): Mapping from agent_id to a list of (node, time) tuples.
        edge_to_idx (dict): Mapping from (u, v) tuples to edge indices.

    Returns:
        np.ndarray: An array of float values representing edge usage.
    """
    E = max(edge_to_idx.values()) + 1
    usage = np.zeros(E, dtype=np.float32)
    for _, path in plan_dict.items():
        for i in range(len(path) - 1):
            (n0, _), (n1, _) = path[i], path[i + 1]
            if (n0, n1) in edge_to_idx:
                usage[edge_to_idx[(n0, n1)]] += 1.0
    return usage


def update_learned_costs(G, multipliers, edge_to_idx):
    """
    Returns a copy of G where for each edge (u, v) the 'learned_l' attribute is set as follows:
      - If either endpoint is a proxy node, use the original cost ("l").
      - Otherwise, set learned_l = original_cost * max(multipliers[idx], 0)
    """
    H_new = G.copy()
    for u, v in H_new.edges():
        orig_cost = G[u][v]["l"]
        if is_proxy_node(u) or is_proxy_node(v):
            H_new[u][v]["learned_l"] = orig_cost
        else:
            idx = edge_to_idx[(u, v)]
            m = max(multipliers[idx], 0)
            H_new[u][v]["learned_l"] = orig_cost * m
    return H_new


def pp_solver_fn(w_np, base_graph, edge_to_idx, agents):
    """
    Helper function used during training to compute the PP plan usage.
    It applies learned multipliers to update the graph's 'learned_l' values,
    then runs the PP solver and returns the usage array computed from the resulting plan.
    If no valid path is found, it returns a zero usage vector.
    """
    H = update_learned_costs(base_graph, w_np, edge_to_idx)
    try:
        plan = PrioritizedPlanningSolver(H).solve(agents)
    except NoSolutionError as e:
        print("  No solution found with current weights; returning zero usage vector")
        # Return a zero usage vector (length = number of edges)
        E = max(edge_to_idx.values()) + 1
        return np.zeros(E, dtype=np.float32)
    return plan_usage(plan, edge_to_idx)


def train_and_apply_weights(
    solver_graph, agents, cbs_plan, iters=100, lr=0.01, lam=3.0
):
    """Trains edge multipliers to mimic the CBS edge usage and applies them to obtain an updated PP plan.

    Args:
        solver_graph (nx.Graph): The original graph.
        agents (list): List of agent objects.
        cbs_plan (dict): The cost=1 CBS plan (reference).
        iters (int): Number of training iterations.
        lr (float): Learning rate for Adam optimizer.
        lam (float): Lambda parameter for the differentiable solver.

    Returns:
        tuple: (solver_graph_updated, pp_plan_trained) where:
          - solver_graph_updated is the graph updated with the best learned multipliers.
          - pp_plan_trained is the PP plan computed on that updated graph.
    """
    # Get the edge index mapping.
    _, edge_to_idx = index_edges(solver_graph)
    # Compute the expert plan usage from the CBS plan.
    cbs_usage = plan_usage(cbs_plan, edge_to_idx)

    # Prepare training: re-index edges and initialize the weight parameter.
    edgelist, edge_to_idx = index_edges(solver_graph)
    model = EdgeWeightParam(len(edgelist))
    opt = optim.Adam(model.parameters(), lr=lr)

    def solver_forward(w_np):
        return pp_solver_fn(w_np, solver_graph, edge_to_idx, agents)

    expert_t = torch.from_numpy(cbs_usage).float()

    best_loss = float("inf")
    best_w = None

    print("Running training...")

    for step in range(iters):
        opt.zero_grad()
        w = model()
        plan_usage_arr = DifferentiableSolver.apply(w, solver_forward, lam)
        loss = torch.sum(torch.abs(plan_usage_arr - expert_t))
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"  Train step={step}, loss={loss.item():.3f}")
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_w = model().detach().cpu().numpy().copy()

    # Update the solver graph using the best learned multipliers.
    solver_graph_updated = update_learned_costs(solver_graph, best_w, edge_to_idx)
    pp_plan_trained = PrioritizedPlanningSolver(solver_graph_updated).solve(agents)

    return solver_graph_updated, pp_plan_trained
