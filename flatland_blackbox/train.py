import numpy as np
import torch
import torch.optim as optim

from flatland_blackbox.models import DifferentiableSolver, EdgeWeightParam
from flatland_blackbox.solvers.pp import PrioritizedPlanningSolver


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
        plan_dict (dict): Mapping from agent_id to a list of (RailNode, time) tuples.
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


def apply_learned_weights(G, multipliers):
    """Applies learned multipliers to update the cost of each edge in a copy of G.

    The updated cost (stored in the 'learned_l' attribute) is computed as:
      learned_l = original_cost * multiplier

    Args:
        G (nx.Graph): The original graph (with cost in 'l').
        multipliers (np.ndarray): Learned multipliers for each edge.

    Returns:
        nx.Graph: A copy of G with updated 'learned_l' attribute on each edge.
    """
    H_new = G.copy()
    _, edge_to_idx = index_edges(G)
    for u, v in H_new.edges():
        idx = edge_to_idx[(u, v)]
        orig_cost = G[u][v]["l"]
        m = max(multipliers[idx], 0)
        # store learned cost in 'learned_l'
        H_new[u][v]["learned_l"] = orig_cost * m
    return H_new


def pp_solver_fn(w_np, base_graph, edge_to_idx, agents):
    """Helper function used during training to compute the PP plan usage.

    It applies the learned multipliers (w_np) to the base graph and then runs the PP solver,
    returning the usage array computed from the resulting plan.

    Args:
        w_np (np.ndarray): Learned multipliers.
        base_graph (nx.Graph): The base graph (with original costs in 'l').
        edge_to_idx (dict): Mapping from (u,v) tuples to edge indices.
        agents (list): List of agent objects.

    Returns:
        np.ndarray: Usage array computed from the PP plan on the updated graph.
    """
    H = base_graph.copy()
    for u, v in H.edges():
        idx = edge_to_idx[(u, v)]
        orig_cost = H[u][v]["l"]
        H[u][v]["learned_l"] = orig_cost * max(w_np[idx], 0)
    plan = PrioritizedPlanningSolver(H).solve(agents)
    return plan_usage(plan, edge_to_idx)


def train_and_apply_weights(G, agents, cbs_plan, iters=100, lr=0.01, lam=3.0):
    """Trains edge multipliers to mimic the CBS edge usage and applies them to obtain an updated PP plan.
    Args:
        G (nx.Graph): The original graph.
        agents (list): List of agent objects.
        cbs_plan (dict): The cost=1 CBS plan (reference).
        iters (int): Number of training iterations.
        lr (float): Learning rate for Adam optimizer.
        lam (float): Lambda parameter for the differentiable solver.

    Returns:
        tuple: A tuple (final_w, pp_plan_trained) where:
          - final_w (np.ndarray): The learned multipliers.
          - pp_plan_trained (dict): The PP plan computed on the updated graph.
    """
    _, edge_to_idx = index_edges(G)
    cbs_usage = plan_usage(cbs_plan, edge_to_idx)

    # Train multipliers
    edgelist, edge_to_idx = index_edges(G)
    model = EdgeWeightParam(len(edgelist))
    opt = optim.Adam(model.parameters(), lr=lr)

    def solver_forward(w_np):
        return pp_solver_fn(w_np, G, edge_to_idx, agents)

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
        if step % 10 == 0:
            print(f"  Train step={step}, loss={loss.item():.3f}")
        # Update best model if current loss is lower
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_w = model().detach().cpu().numpy().copy()

    # Learned weights
    # final_w = model().detach().cpu().numpy()
    # # # Apply best learned multipliers multipliers to new graph with 'learned_l'
    # G_updated = apply_learned_weights(G, final_w)
    # pp_plan_trained = PrioritizedPlanningSolver(G_updated).solve(agents)

    # Apply the best learned multipliers to update the graph.
    G_updated = apply_learned_weights(G, best_w)
    pp_plan_trained = PrioritizedPlanningSolver(G_updated).solve(agents)

    return G_updated, pp_plan_trained
