import numpy as np
import torch
import torch.optim as optim

from flatland_blackbox.models import DifferentiableSolver, EdgeWeightParam
from flatland_blackbox.solvers.pp import PrioritizedPlanningSolver
from flatland_blackbox.utils import NoSolutionError, is_proxy_node


def index_edges(G):
    """Returns a sorted list of edges and an index mapping for a given graph.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        tuple: A tuple (edgelist, edge_to_idx) where:
            - edgelist (list): Sorted list of edges.
            - edge_to_idx (dict): Dictionary mapping each edge (u, v) to its integer index.
    """
    edgelist = sorted(G.edges())
    edge_to_idx = {e: i for i, e in enumerate(edgelist)}
    return edgelist, edge_to_idx


def plan_usage(plan_dict, edge_to_idx):
    """Computes an edge usage array from a plan.

    The usage array has length equal to the number of edges and each element is
    incremented by 1.0 for each time an edge is used consecutively in any agent's path.

    Args:
        plan_dict (dict): Mapping from agent_id to a list of (node, time) tuples.
        edge_to_idx (dict): Mapping from (u, v) tuples to edge indices.

    Returns:
        np.ndarray: A float array representing the frequency of usage for each edge.
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
    """Updates the graph's edge costs using learned multipliers.

    For each edge (u, v) in the graph copy, the 'learned_l' attribute is set as follows:
      - If either endpoint is a proxy node, use the original cost ("l").
      - Otherwise, set learned_l = original_cost * max(multipliers[idx], 0).

    Args:
        G (nx.Graph): The original graph.
        multipliers (np.ndarray): Learned multipliers for each edge.
        edge_to_idx (dict): Mapping from (u, v) tuples to edge indices.

    Returns:
        nx.Graph: A copy of G with updated 'learned_l' attributes on each edge.
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
    """Computes the PP plan usage using updated learned weights.

    This function updates the graph's edge costs (setting 'learned_l') according to
    the provided multipliers, runs the PP solver on the updated graph, and returns an
    edge usage array derived from the computed plan. If no valid plan is found, a zero
    usage vector is returned.

    Args:
        w_np (np.ndarray): Current learned multipliers as a NumPy array.
        base_graph (nx.Graph): The base graph (with original edge costs in 'l').
        edge_to_idx (dict): Mapping from (u, v) tuples to edge indices.
        agents (list): List of agent objects.

    Returns:
        np.ndarray: Edge usage array computed from the PP plan.
    """
    H = update_learned_costs(base_graph, w_np, edge_to_idx)
    try:
        plan = PrioritizedPlanningSolver(H).solve(agents)
    except NoSolutionError as e:
        # print("  No solution found with current weights; returning zero usage vector")
        E = max(edge_to_idx.values()) + 1
        return np.zeros(E, dtype=np.float32)
    return plan_usage(plan, edge_to_idx)


def train_and_apply_weights(
    solver_graph, agents, cbs_plan, iters=100, lr=0.01, lam=3.0
):
    """Trains edge multipliers to mimic CBS edge usage and computes an updated PP plan.

    This function uses a differentiable solver to train edge weight multipliers so that the PP plan
    usage matches that of the CBS reference plan. After training, it applies the best learned multipliers
    to update the graph and computes the final PP plan.

    Args:
        solver_graph (nx.Graph): The original graph.
        agents (list): List of agent objects.
        cbs_plan (dict): The cost=1 CBS plan (reference).
        iters (int, optional): Number of training iterations. Defaults to 100.
        lr (float, optional): Learning rate for the Adam optimizer. Defaults to 0.01.
        lam (float, optional): Lambda parameter for finite differences in the differentiable solver. Defaults to 3.0.

    Returns:
        tuple: A tuple (solver_graph_updated, pp_plan_trained) where:
            - solver_graph_updated (nx.Graph): The graph updated with the best learned multipliers.
            - pp_plan_trained (dict): The PP plan computed on the updated graph.
    """
    # Compute edge index mapping and expert plan usage.
    _, edge_to_idx = index_edges(solver_graph)
    cbs_usage = plan_usage(cbs_plan, edge_to_idx)

    # Prepare training by re-indexing edges and initializing the weight parameter.
    edgelist, edge_to_idx = index_edges(solver_graph)
    model = EdgeWeightParam(len(edgelist))
    opt = optim.Adam(model.parameters(), lr=lr)

    def solver_forward(w_np):
        return pp_solver_fn(w_np, solver_graph, edge_to_idx, agents)

    expert_t = torch.from_numpy(cbs_usage).float()

    best_loss = float("inf")
    best_w = None

    for step in range(iters):
        opt.zero_grad()
        w = model()
        plan_usage_arr = DifferentiableSolver.apply(w, solver_forward, lam)
        loss = torch.sum(torch.abs(plan_usage_arr - expert_t))
        loss.backward()
        opt.step()
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_w = model().detach().cpu().numpy().copy()

    # Update the graph using the best learned multipliers and compute the final PP plan.
    solver_graph_updated = update_learned_costs(solver_graph, best_w, edge_to_idx)
    pp_plan_trained = PrioritizedPlanningSolver(solver_graph_updated).solve(agents)

    return solver_graph_updated, pp_plan_trained
