from collections import namedtuple

import matplotlib.pyplot as plt
import networkx as nx

RailNode = namedtuple("RailNode", ["row", "col", "direction"])


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


def true_distance_heuristic(nx_graph, goal_node):
    """
    Return a dict: node -> distance
    Using a shortest-path approach on 'l' edge weight in nx_graph.
    """
    dist = {}
    for node in nx_graph.nodes():
        try:
            d = nx.shortest_path_length(
                nx_graph, source=node, target=goal_node, weight="l"
            )
            dist[node] = d
        except nx.NetworkXNoPath:
            dist[node] = float("inf")

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


def visualize_graph(G, title="Graph", ax=None):
    """
    Visualize the rail graph in a given Axes (if ax is provided)
    or create a new figure/axes if ax=None.
    """
    create_new_figure = ax is None

    if create_new_figure:
        fig, ax = plt.subplots(figsize=(8, 6))

    pos = {}
    for node in G.nodes():
        row, col, direction = node
        pos[node] = (col, -row)  # x=col, y=-row

    # Create a label dict that shows only (row,col)
    node_labels = {node: f"({node[0]},{node[1]})" for node in G.nodes()}

    # Node color
    node_colors = []
    for n in G.nodes():
        node_type = G.nodes[n].get("type", "rail")
        node_colors.append("lightsteelblue" if node_type == "rail" else "blue")

    # Edge weights
    edge_dict = nx.get_edge_attributes(G, "l")
    edge_labels = {e: f"{val:.2f}" for e, val in edge_dict.items()}

    # Scale thickness by weight (clamp min=0.1)
    widths = []
    for u, v in G.edges():
        w = G[u][v]["l"]
        widths.append(max(0.1, w * 2))

    nx.draw_networkx_edges(
        G, pos=pos, width=widths, arrows=True, arrowstyle="-|>", ax=ax
    )
    nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors, node_size=400, ax=ax)
    nx.draw_networkx_labels(G, pos=pos, labels=node_labels, font_size=8, ax=ax)
    nx.draw_networkx_edge_labels(
        G, pos=pos, edge_labels=edge_labels, font_size=6, ax=ax
    )

    ax.set_title(title)
    ax.axis("equal")


def visualize_graph_with_lengths(G, title="Graph (length ~ weight)"):
    """
    Visualize the graph so that edges with higher 'l'
    become longer in the final 2D layout.
    Uses Kamada-Kawai layout, interpreting 'l' as the
    distance/weight in shortest-path computations.
    """
    pos = nx.kamada_kawai_layout(G, weight="l")

    # Node color
    node_colors = []
    for n in G.nodes():
        node_type = G.nodes[n].get("type", "rail")
        node_colors.append("green" if node_type == "rail" else "blue")

    # Format edge labels to show cost with 2 decimals
    edge_labels = {e: f"{G[e[0]][e[1]]['l']:.2f}" for e in G.edges()}

    plt.figure(figsize=(8, 6))

    nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors, node_size=600)
    nx.draw_networkx_labels(G, pos=pos, font_size=8)

    nx.draw_networkx_edges(G, pos=pos, arrows=False)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=6)

    plt.title(title)
    plt.axis("equal")
    plt.show()
