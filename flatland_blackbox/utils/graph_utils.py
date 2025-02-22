from collections import namedtuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from flatland.envs.rail_env import RailEnv

RailNode = namedtuple("RailNode", ["row", "col", "direction"])


def filter_proxy_nodes(plan):
    """
    Returns an updated plan with all entries where node.direction is -1 removed.

    Args:
        plan (dict): Dictionary where keys map to lists of (node, value) tuples.

    Returns:
        dict: A new plan dictionary with filtered tuples.
    """
    return {
        key: [(node, t) for (node, t) in path if node.direction != -1]
        for key, path in plan.items()
    }


def get_rail_subgraph(nx_graph):
    """
    Extract and return only the 'rail' nodes and 'dir' edges
    from the original Flatland graph.
    """
    rail_nodes = [
        n for n, data in nx_graph.nodes(data=True) if data.get("type") == "rail"
    ]
    rail_edges = []
    for u, v, data in nx_graph.edges(data=True):
        if data.get("type") == "dir" and u in rail_nodes and v in rail_nodes:
            rail_edges.append((u, v))

    return nx_graph.edge_subgraph(rail_edges).copy()


def shift_overlapping_edts(sorted_agents, min_time_between_dep=1):
    """
    Adjust earliest departure times to prevent multiple agents
    from starting in the same (row, col) too close in time.
    """
    start_times = {}  # Maps (row, col) -> last assigned EDT

    for agent in sorted_agents:
        start_pos = agent.initial_position
        current_edt = getattr(agent, "earliest_departure", 0)
        # If this cell is already occupied up to `start_times[start_pos]`
        # we shift the agent's EDT by at least `min_time_between_dep`.
        if start_pos in start_times and start_times[start_pos] >= current_edt:
            # push it so that agent starts at least min_time_between_dep
            current_edt = start_times[start_pos] + min_time_between_dep

        agent.earliest_departure = current_edt
        start_times[start_pos] = current_edt

    return sorted_agents


def true_distance_heuristic(nx_graph, goal_node):
    """
    Returns a dict: node -> distance, where distance is the minimal cost
    from each node *to* goal_node in the original directed graph.
    We do so by reversing the graph and running a single Dijkstra from goal_node.
    """
    # Reverse the graph so we can do a single source Dijkstra from goal_node
    # to all other nodes in that reversed perspective,
    # which corresponds to "from node to goal" in the original.
    reversed_graph = nx.reverse(nx_graph, copy=True)

    # Custom edge weight function that uses learned weights
    # "learned_l" over original weights "l"
    def edge_weight(u, v, d):
        # If 'learned_l' exists, use it; otherwise, use 'l'
        return d.get("learned_l", d["l"])

    # Run Dijkstra in the reversed graph
    dist_map = nx.single_source_dijkstra_path_length(
        reversed_graph, goal_node, weight=edge_weight
    )

    dist = {}
    for node in nx_graph.nodes():
        dist[node] = dist_map.get(node, float("inf"))

    return dist


def add_proxy_nodes_for_agents(graph, agents, cost=0, copy_graph=True):
    if copy_graph:
        G = graph.copy()
    else:
        G = graph

    for agent in agents:
        add_single_agent_proxy(G, agent, cost)

    return G


def add_single_agent_proxy(G, agent, cost=0):
    (start_r, start_c) = agent.initial_position
    (end_r, end_c) = agent.target

    proxy_start = (start_r, start_c, -1)
    proxy_end = (end_r, end_c, -1)

    G.add_node(proxy_start, type="proxy")
    G.add_node(proxy_end, type="proxy")

    s_nodes = get_rail_nodes_in_cell(G, start_r, start_c)
    for n in s_nodes:
        G.add_edge(proxy_start, n, l=cost)

    e_nodes = get_rail_nodes_in_cell(G, end_r, end_c)
    for n in e_nodes:
        G.add_edge(n, proxy_end, l=cost)


def get_rail_nodes_in_cell(G, row, col):
    return [
        n
        for n in G.nodes()
        if (n[0] == row and n[1] == col and G.nodes[n].get("type") == "rail")
    ]


def decide_node(nx_graph, row, col, agent_id):
    """
    If exactly one node matches (row, col), use it.
    If multiple nodes match (row, col), look for direction=-1.
    If none is found, or no direction=-1, throw error.
    """
    matching_nodes = [n for n in nx_graph.nodes() if (n[0], n[1]) == (row, col)]
    if len(matching_nodes) == 0:
        raise ValueError(f"Agent {agent_id}: no node found at row={row}, col={col}")

    if len(matching_nodes) == 1:
        # Exactly one node > use it
        single_n = matching_nodes[0]
        # print(f"[DEBUG] Agent {agent_id}: single node {single_n}")
        return RailNode(*single_n)
    else:
        # multiple nodes > look for direction=-1
        candidates = [n for n in matching_nodes if n[2] == -1]
        if not candidates:
            raise ValueError(
                f"Agent {agent_id}: multiple nodes at (row={row}, col={col}) but none with direction=-1"
            )
        # if len(candidates) > 1:
        #     print(f"[WARNING] Agent {agent_id}: multiple direction=-1 matches, picking first. {candidates}")
        chosen = candidates[0]
        # print(f"[DEBUG] Agent {agent_id}: multiple matches => picking {chosen}")
        return chosen


def visualize_graph_weights(G, title, scale=True):
    """
    Visualize the rail graph with "learned_l" edge weights and plot it directly.

    If scale is True, node positions are computed using a force-directed layout
    (with initial positions based on the grid) so that the edge lengths in the
    plot reflect the "learned_l" values.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute grid positions (initial positions)
    initial_pos = {}
    for node in G.nodes():
        row, col, _ = node
        initial_pos[node] = (col, -row)  # x=col, y=-row

    # 2. Compute positions using spring_layout if scaling is desired.
    #    Use a custom weight function that inverts the learned_l value.
    if scale:

        def inv_weight(u, v, d):
            # Avoid division by zero: if learned_l is 0, return a default weight.
            return 1.0 / d["learned_l"] if d["learned_l"] > 0 else 1.0

        pos = nx.spring_layout(
            G, pos=initial_pos, weight=inv_weight, iterations=100, k=1
        )
    else:
        pos = initial_pos

    # Create a label dict that shows only (row, col)
    node_labels = {node: f"({node[0]},{node[1]})" for node in G.nodes()}

    # Determine node colors based on node type
    node_colors = []
    for n in G.nodes():
        node_type = G.nodes[n].get("type", "rail")
        node_colors.append("lightsteelblue" if node_type == "rail" else "blue")

    # Get edge weights from "learned_l" attribute and format labels
    edge_dict = nx.get_edge_attributes(G, "learned_l")
    edge_labels = {e: f"{val:.2f}" for e, val in edge_dict.items()}

    nx.draw_networkx_edges(G, pos=pos, arrows=True, arrowstyle="-|>", ax=ax)
    nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors, node_size=400, ax=ax)
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=8, ax=ax)
    nx.draw_networkx_edge_labels(
        G, pos=pos, edge_labels=edge_labels, font_size=6, ax=ax
    )

    ax.set_title(title)
    ax.axis("equal")
    plt.show()


def check_no_collisions(paths):
    """
    paths: dict of agent_id -> list of (RailNode, timestep)
    Example of paths:
      {
          0: [(RailNode(row=14, col=16, direction=-1), 0),
              (RailNode(row=14, col=16, direction=3), 1),
              ...],
          1: [...],
          ...
      }
    Raises AssertionError if:
      - Two agents overlap in the same cell at the same time (vertex collision).
      - Two agents do an edge swap: (A goes (r1,c1)->(r2,c2)) at time t,
        while B goes (r2,c2)->(r1,c1)) at the same time t.
    """

    # 1) Check vertex collisions
    seen_positions = {}  # key: (row, col, time) -> agent_id
    for agent_id, path in paths.items():
        for node, t in path:
            # Skip checking if this node is a proxy node.
            if node.direction == -1:
                continue
            pos_time = (node.row, node.col, t)
            if pos_time in seen_positions:
                other_agent = seen_positions[pos_time]
                raise AssertionError(
                    f"Collision detected! Agent {agent_id} and agent {other_agent} "
                    f"both at row={node.row}, col={node.col} at time={t}"
                )
            seen_positions[pos_time] = agent_id

    # 2) Build transitions for edge-swap collision check
    # We'll store each move as ((r1,c1,t1),(r2,c2,t2)) => agent_id,
    # ignoring moves where time does not strictly increase.
    transitions = {}  # key: ((r1,c1,t1),(r2,c2,t2)) -> agent_id

    for agent_id, path in paths.items():
        for i in range(len(path) - 1):
            (node1, t1) = path[i]
            (node2, t2) = path[i + 1]
            # Only consider moves that progress in time
            if t2 > t1:
                move_key = ((node1.row, node1.col, t1), (node2.row, node2.col, t2))
                # If we see the reverse move, it's a collision
                reverse_key = ((node2.row, node2.col, t1), (node1.row, node1.col, t2))

                if reverse_key in transitions:
                    other_agent = transitions[reverse_key]
                    raise AssertionError(
                        f"Edge-swap collision detected! Agent {agent_id} and agent {other_agent} "
                        f"swap positions: {move_key} <-> {reverse_key}"
                    )

                transitions[move_key] = agent_id

    # If we get here, no collisions found


# Addapted from flatland.graphs.graph_utils  plotGraphEnvlatland() function
def plotGraphEnv(
    G,
    env: RailEnv,
    aImg,
    space=0.3,
    figsize=(8, 8),
    dpi=100,
    show_labels=(),
    show_edges=("dir"),
    show_edge_weights=False,
    show_nodes="all",
    node_colors=None,
    edge_colors=None,
    alpha_img=0.2,
    node_size=300,
    lvHighlight=None,
    arrowsize=10,
):

    # NESW directions in xy coords
    xyDir = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

    # Rotate the xyDir 90 deg to create visual offsets for the rail nodes,
    # eg for a N dir node, the offset needs to be E of the grid node.
    xy2 = np.array([xyDir[(i + 1) % 4, :] for i in range(4)])

    if figsize is not None:
        plt.figure(figsize=figsize, dpi=dpi)

    rows, cols = env.rail.grid.shape
    plt.imshow(aImg, extent=(-0.5, cols - 0.5, 0.5 - rows, 0.5), alpha=alpha_img)

    if show_nodes == "all":
        nodelist = G.nodes()
    else:
        nodelist = [n for n, d in G.nodes(data=True) if d["type"] in show_nodes]

    if node_colors is None:
        node_colors = {"grid": "red", "rail": "lightblue"}

    if edge_colors is None:
        edge_colors = {"grid": "gray", "hold": "blue", "dir": "green"}

    edgelist = [(u, v) for u, v, d in G.edges(data=True) if d["type"] in show_edges]
    dnDat = G.nodes(data=True)
    deDat = {(u, v): d for u, v, d in G.edges(data=True) if d["type"] in show_edges}

    dnxyPos = {
        n: (
            n[1] if len(n) == 2 else n[1] - space * xy2[n[2], 0],
            -n[0] if len(n) == 2 else -n[0] - space * xy2[n[2], 1],
        )
        for n in G.nodes()
    }

    nx.draw(
        G,
        labels={n: str(n) for n, d in G.nodes(data=True) if d["type"] in show_labels},
        node_color=[node_colors[dnDat[n]["type"]] for n in nodelist],
        pos=dnxyPos,
        edgelist=edgelist,
        edge_color=[edge_colors[deDat[(u, v)]["type"]] for u, v in edgelist],
        nodelist=nodelist,
        node_size=node_size,
        arrowsize=arrowsize,
        font_size=8,
    )

    if show_edge_weights:
        labels = nx.get_edge_attributes(G, "learned_l")
        formatted_labels = {
            (u, v): f"{l:.2f}" for (u, v), l in labels.items() if l != 1
        }
        nx.draw_networkx_edge_labels(
            G,
            dnxyPos,
            edge_labels=formatted_labels,
            font_size=5,
            bbox=dict(facecolor="none", edgecolor="none", pad=0),
        )

    # plot initial, target positions
    rcStarts = np.array([agent.initial_position for agent in env.agents])
    xyStarts = np.matmul(rcStarts, [[0, -1], [1, 0]])
    rcTargs = np.array([agent.target for agent in env.agents])
    xyTargs = np.matmul(rcTargs, [[0, -1], [1, 0]])

    # Cyan Square for starts, Red Triangle for targets
    plt.scatter(*xyStarts.T, s=200, marker="s", facecolor="cyan", edgecolor="black")
    plt.scatter(*xyTargs.T, s=200, marker="^", facecolor="red", edgecolor="black")

    # make dict of list of initial, target pos
    dlIPos = {}
    dlTPos = {}
    for agent in env.agents:
        liAgent = dlIPos.get(agent.initial_position, []) + [agent.handle]
        dlIPos[agent.initial_position] = liAgent

        liAgent = dlTPos.get(agent.target, []) + [agent.handle]
        dlTPos[agent.target] = liAgent

    # Write the agent numbers for each initial, target pos
    for rcPos, liAgent in dlIPos.items():
        plt.annotate(",".join(map(str, liAgent)), (rcPos[1], -rcPos[0] + 0.4))

    for rcPos, liAgent in dlTPos.items():
        plt.annotate(",".join(map(str, liAgent)), (rcPos[1], -rcPos[0] - 0.6))
        # plt.annotate(str(iAgent), (rcPos[1]+i*0.3, -rcPos[0]-0.6))

    if lvHighlight is not None:
        xyV = np.matmul(np.array(lvHighlight), [[0, -1], [1, 0]])
        plt.scatter(*xyV.T, s=900, marker="s", facecolor="none", edgecolor="red")
