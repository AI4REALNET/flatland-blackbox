import glob
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.graphs.graph_utils import plotGraphEnv
from flatland.utils.rendertools import RenderTool


class NoSolutionError(Exception):
    """Raised when a solver fails to find any solution paths."""

    pass


# Helper functions for raw node tuples.
def get_row(node):
    """Returns the row of the node."""
    return node[0]


def get_col(node):
    """Returns the column of the node."""
    return node[1]


def get_direction(node):
    """Returns the direction of the node."""
    return node[2]


def normalize_node(node):
    """
    Normalizes a node tuple by ignoring the agent ID if present.

    Returns:
        tuple: The node as a 3-tuple.
    """
    return node[:3] if len(node) == 4 else node


def is_proxy_node(node):
    """
    Returns True if the node is a proxy node.

    A node is considered a proxy if its direction is -1.
    """
    return get_direction(node) == -1


def filter_proxy_nodes(plan):
    """
    Removes proxy nodes from a plan.

    Args:
        plan (dict): Mapping of agent_id to list of (node, time) tuples.

    Returns:
        dict: The filtered plan.
    """
    return {
        key: [(node, t) for (node, t) in path if not is_proxy_node(node)]
        for key, path in plan.items()
    }


def print_proxy_nodes(G):
    """
    Prints all proxy nodes in the graph.

    Args:
        G (nx.Graph): The input graph.
    """
    proxy_nodes = [
        (n, data) for n, data in G.nodes(data=True) if data.get("type") == "proxy"
    ]
    if proxy_nodes:
        print("Proxy nodes found:")
        for n, data in proxy_nodes:
            print(n, data)
    else:
        print("No proxy nodes found.")


def get_rail_subgraph(nx_graph):
    """
    Returns the subgraph containing only 'rail' nodes and 'dir' edges.

    Args:
        nx_graph (nx.Graph): The original Flatland graph.

    Returns:
        nx.Graph: The rail subgraph.
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
    Adjusts earliest departure times (EDTs) to avoid overlap in the same cell.

    Args:
        sorted_agents (list): Agents sorted by their EDT.
        min_time_between_dep (int): Minimum time difference required.

    Returns:
        list: Agents with updated EDTs.
    """
    start_times = {}
    for agent in sorted_agents:
        start_pos = agent.initial_position
        current_edt = getattr(agent, "earliest_departure", 0)
        if start_pos in start_times and start_times[start_pos] >= current_edt:
            current_edt = start_times[start_pos] + min_time_between_dep
        agent.earliest_departure = current_edt
        start_times[start_pos] = current_edt
    return sorted_agents


def true_distance_heuristic(nx_graph, goal_node):
    """
    Computes the minimal cost from each node to the goal using Dijkstra's algorithm on the reversed graph.

    Args:
        nx_graph (nx.Graph): The original directed graph.
        goal_node (tuple): The goal node.

    Returns:
        dict: Mapping from node to minimal cost.
    """
    reversed_graph = nx.reverse(nx_graph, copy=True)

    def edge_weight(u, v, d):
        return d.get("learned_l", d["l"])

    dist_map = nx.single_source_dijkstra_path_length(
        reversed_graph, goal_node, weight=edge_weight
    )
    dist = {}
    for node in nx_graph.nodes():
        dist[node] = dist_map.get(node, float("inf"))
    return dist


def add_proxy_nodes(G, agents, cost=0.0):
    """
    Adds proxy nodes for each agent to the graph.

    Args:
        G (nx.Graph): The original graph.
        agents (list): List of agent objects.
        cost (float): Edge cost for proxy connections.

    Returns:
        nx.Graph: A copy of G with proxy nodes added.
    """
    H = G.copy()
    for agent in agents:
        add_single_agent_proxy(H, agent, cost)
    return H


def add_single_agent_proxy(G, agent, cost=0.0):
    """
    Adds a start proxy node for the agent and a shared goal proxy node if not present.

    Args:
        G (nx.Graph): The graph.
        agent: An agent object with attributes initial_position, target, and handle.
        cost (float): Edge cost for proxy connections.
    """
    start_r, start_c = map(np.int64, agent.initial_position)
    end_r, end_c = map(np.int64, agent.target)
    agent_id = np.int64(agent.handle)
    proxy_start = (start_r, start_c, -1, agent_id)
    G.add_node(proxy_start, type="proxy", agent_id=agent_id)
    s_nodes = get_rail_nodes_in_cell(G, start_r, start_c)
    for n in s_nodes:
        G.add_edge(proxy_start, n, l=cost)
    proxy_end = (end_r, end_c, -1)
    if proxy_end not in G:
        G.add_node(proxy_end, type="proxy")
        e_nodes = get_rail_nodes_in_cell(G, end_r, end_c)
        for n in e_nodes:
            G.add_edge(n, proxy_end, l=cost)


def get_rail_nodes_in_cell(G, row, col):
    """
    Returns all rail nodes in the specified cell.

    Args:
        G (nx.Graph): The graph.
        row (int): Row index.
        col (int): Column index.

    Returns:
        list: List of rail nodes in the cell.
    """
    return [
        n
        for n in G.nodes()
        if n[0] == row and n[1] == col and G.nodes[n].get("type") == "rail"
    ]


def get_start_proxy_node(nx_graph, row, col, agent_id):
    """
    Retrieves the start proxy node for a given agent at (row, col).

    Args:
        nx_graph (nx.Graph): The graph.
        row (int): Start row.
        col (int): Start column.
        agent_id (int): The agent's unique identifier.

    Returns:
        tuple: The start proxy node.

    Raises:
        ValueError: If no start proxy is found.
    """
    matching_nodes = [n for n in nx_graph.nodes() if n[0] == row and n[1] == col]
    agent_proxies = [
        n
        for n in matching_nodes
        if nx_graph.nodes[n].get("type") == "proxy"
        and nx_graph.nodes[n].get("agent_id") == agent_id
    ]
    if agent_proxies:
        return agent_proxies[0]
    else:
        raise ValueError(
            f"Agent {agent_id}: no start proxy found at row={row}, col={col}"
        )


def get_goal_proxy_node(nx_graph, row, col):
    """
    Retrieves the shared goal proxy node at (row, col).

    Args:
        nx_graph (nx.Graph): The graph.
        row (int): Goal row.
        col (int): Goal column.

    Returns:
        tuple: The goal proxy node.

    Raises:
        ValueError: If no shared goal proxy is found.
    """
    matching_nodes = [n for n in nx_graph.nodes() if n[0] == row and n[1] == col]
    shared_proxies = [
        n
        for n in matching_nodes
        if nx_graph.nodes[n].get("type") == "proxy"
        and "agent_id" not in nx_graph.nodes[n]
    ]
    if shared_proxies:
        return shared_proxies[0]
    else:
        raise ValueError(f"No shared goal proxy found at row={row}, col={col}")


def visualize_graph_weights(G, title, scale=True):
    """
    Visualizes the rail graph with edge weights ("learned_l").

    Args:
        G (nx.Graph): The rail graph.
        title (str): Title for the plot.
        scale (bool): If True, use a spring layout to scale node positions.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    initial_pos = {}
    for node in G.nodes():
        row, col, _ = node
        initial_pos[node] = (col, -row)
    if scale:

        def inv_weight(u, v, d):
            return 1.0 / d["learned_l"] if d["learned_l"] > 0 else 1.0

        pos = nx.spring_layout(
            G, pos=initial_pos, weight=inv_weight, iterations=100, k=1
        )
    else:
        pos = initial_pos
    node_labels = {node: f"({node[0]},{node[1]})" for node in G.nodes()}
    node_colors = [
        "lightsteelblue" if G.nodes[n].get("type") == "rail" else "blue"
        for n in G.nodes()
    ]
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
    Checks a set of agent paths for collisions.

    Args:
        paths (dict): Mapping from agent_id to list of (node, timestep) tuples.

    Raises:
        AssertionError: If a vertex or edge-swap collision is detected.
    """
    seen_positions = {}
    for agent_id, path in paths.items():
        for node, t in path:
            if is_proxy_node(node):
                continue
            pos_time = (get_row(node), get_col(node), t)
            if pos_time in seen_positions:
                other_agent = seen_positions[pos_time]
                raise AssertionError(
                    f"Collision detected! Agent {agent_id} and agent {other_agent} "
                    f"both at row={get_row(node)}, col={get_col(node)} at time={t}"
                )
            seen_positions[pos_time] = agent_id

    transitions = {}
    for agent_id, path in paths.items():
        for i in range(len(path) - 1):
            (node1, t1) = path[i]
            (node2, t2) = path[i + 1]
            if t2 > t1:
                move_key = (
                    (get_row(node1), get_col(node1), t1),
                    (get_row(node2), get_col(node2), t2),
                )
                reverse_key = (
                    (get_row(node2), get_col(node2), t1),
                    (get_row(node1), get_col(node1), t2),
                )
                if reverse_key in transitions:
                    other_agent = transitions[reverse_key]
                    raise AssertionError(
                        f"Edge-swap collision detected! Agent {agent_id} and agent {other_agent} "
                        f"swap positions: {move_key} <-> {reverse_key}"
                    )
                transitions[move_key] = agent_id


def initialize_environment(
    seed=42, width=30, height=30, num_agents=2, max_num_cities=3
):
    """
    Initializes a Flatland RailEnv with the given parameters.

    Args:
        seed (int): Random seed.
        width (int): Environment width.
        height (int): Environment height.
        num_agents (int): Number of agents.
        max_num_cities (int): Maximum number of cities.

    Returns:
        RailEnv: The initialized environment.
    """
    env = RailEnv(
        width=width,
        height=height,
        rail_generator=sparse_rail_generator(
            max_num_cities=max_num_cities,
            grid_mode=False,
            max_rails_between_cities=4,
            max_rail_pairs_in_city=2,
            seed=seed,
        ),
        line_generator=sparse_line_generator(seed=seed),
        obs_builder_object=DummyObservationBuilder(),
        number_of_agents=num_agents,
    )
    return env


def plot_agent_subgraphs(env, G_paths_subgraphs, save_fig_folder, agent_titles):
    """
    Plots each agent's subgraph over the environment background image.

    Clears previous PNG files in the output folder and saves new plots.

    Args:
        env (RailEnv): The Flatland environment.
        G_paths_subgraphs (dict): Mapping from agent_id to agent subgraphs.
        save_fig_folder (str): Directory to save the plots.
    """
    render_tool = RenderTool(env, show_debug=True)
    render_tool.render_env(
        show_rowcols=True, show_inactive_agents=False, show_observations=False
    )
    aImg = render_tool.get_image()
    png_files = glob.glob(os.path.join(save_fig_folder, "*.png"))
    for file in png_files:
        os.remove(file)
    for agent_id, Gpath in G_paths_subgraphs.items():
        plt.figure(figsize=(8, 8))
        plotGraphEnv(
            Gpath,
            env,
            aImg,
            figsize=(8, 8),
            node_size=8,
            space=0.1,
            node_colors={"rail": "blue", "grid": "red"},
            edge_colors={"hold": "gray", "dir": "green"},
            show_nodes=("rail", "grid"),
            show_edges=("dir",),
            show_labels=(),
            show_edge_weights=True,
            alpha_img=0.7,
        )
        plt.title(agent_titles[agent_id])
        plt.savefig(f"{save_fig_folder}/path_agent_{agent_id}.png", dpi=300)
        plt.close("all")


def print_agents_start(agents):
    """
    Prints each agent's start position, target, and earliest departure time.

    Args:
        agents (list): List of agent objects.
    """
    for agent in agents:
        start_rc = tuple(map(int, agent.initial_position))
        end_rc = tuple(map(int, agent.target))
        print(
            f"Agent {agent.handle} start: {start_rc} end: {end_rc} earliest departure: {agent.earliest_departure}"
        )
