from heapq import heappop, heappush
from math import inf

from flatland_blackbox.utils import (
    NoSolutionError,
    get_col,
    get_direction,
    get_goal_proxy_node,
    get_row,
    get_start_proxy_node,
    normalize_node,
    true_distance_heuristic,
)


class ReservationManager:
    """
    Manages time-step reservations to avoid collisions among agents.

    The reservation table (occupant_table) stores the agent_id that occupies a given cell at a given time.
    """

    def __init__(self):
        """Initializes an empty reservation table."""
        # occupant_table[((row, col), time)] = agent_id (int), or None if unoccupied
        self.occupant_table = {}

    def block_path(self, agent_id, path):
        """
        Reserves all cells along the given path for an agent.

        Args:
            agent_id (int): The identifier of the agent.
            path (list[tuple]): List of (node, time) tuples representing the agent's path.
        """
        for node, t in path:
            r, c = get_row(node), get_col(node)
            self.occupant_table[((r, c), t)] = agent_id

    def is_blocked(self, agent_id, pre_r, pre_c, r, c, t):
        """
        Checks if occupying cell (r, c) at time t for the given agent conflicts with another agent.

        This function checks for:
          1. Vertex conflict: The cell is occupied by a different agent.
          2. Edge-swap conflict: A direct swap with another agent occurs.

        Args:
            agent_id (int): The agent attempting to occupy the cell.
            pre_r (int): Row of the previous cell.
            pre_c (int): Column of the previous cell.
            r (int): Row of the cell to occupy.
            c (int): Column of the cell to occupy.
            t (int): Time step for occupancy.

        Returns:
            bool: True if the cell is blocked by a different agent, False otherwise.
        """
        t_int = int(t)
        # Vertex conflict: check if cell is occupied by a different agent.
        occupant = self.occupant_table.get(((r, c), t_int), None)
        if occupant is not None and occupant != agent_id:
            return True

        # Edge conflict: check for direct swap conflicts.
        if t_int > 0:
            occupant_rc_tminus = self.occupant_table.get(((r, c), t_int - 1), None)
            occupant_prec_t = self.occupant_table.get(((pre_r, pre_c), t_int), None)
            if (
                occupant_rc_tminus is not None
                and occupant_rc_tminus == occupant_prec_t
                and occupant_rc_tminus != agent_id
            ):
                return True

        return False


class PrioritizedPlanningSolver:
    """
    Prioritized Planning (PP) solver for multi-agent path planning.

    Agents are planned sequentially in the given order. Each agent's path is computed using
    a time-augmented A* search. The search uses the original edge cost ("l") for computing
    arrival times (for reservations) and the learned edge cost ("learned_l"), if available,
    for guiding the search. Proxy nodes are used for the agent start (which are unique per agent)
    and for the shared goal.
    """

    def __init__(self, nx_graph):
        """
        Initializes the PP solver with the given environment graph.

        Args:
            nx_graph (nx.Graph): The environment graph containing rail nodes and edges
                with attribute "l" for the original cost.
        """
        self.nx_graph = nx_graph
        self.res_manager = ReservationManager()
        self.agent_data = (
            {}
        )  # {agent_id: {"start_node", "goal_node", "dist_map", "earliest_departure"}}

    def solve(self, agents):
        """
        Plans paths for the given agents sequentially.

        For each agent, the start proxy, goal proxy, and a heuristic distance map (via true_distance_heuristic)
        are computed. Then each agent is planned in turn using a time-augmented A* search, and the path is
        reserved in the reservation table.

        Args:
            agents (list): List of agent objects. Each agent must have:
                - initial_position (tuple): (row, col) of the start.
                - target (tuple): (row, col) of the goal.
                - handle (int): Unique agent identifier.
                - earliest_departure (int, optional): Earliest departure time.

        Returns:
            dict: Mapping from agent_id to a list of (node, time) tuples representing the computed path.

        Raises:
            NoSolutionError: If no path can be found for any agent.
        """
        # Precompute agent-specific data.
        for agent in agents:
            row_s, col_s = agent.initial_position
            row_g, col_g = agent.target
            start_node = get_start_proxy_node(self.nx_graph, row_s, col_s, agent.handle)
            goal_node = get_goal_proxy_node(self.nx_graph, row_g, col_g)
            dist_map = true_distance_heuristic(self.nx_graph, goal_node)
            assert None not in dist_map.values(), "Found None in distance map"
            self.agent_data[agent.handle] = {
                "start_node": start_node,
                "goal_node": goal_node,
                "dist_map": dist_map,
                "earliest_departure": getattr(agent, "earliest_departure", 0),
            }

        # Plan each agent in sequence.
        solution = {}
        for agent in agents:
            path = self._compute_best_path_for_agent(agent)
            if not path:
                raise NoSolutionError(f"No path found for agent {agent.handle}")
            self.res_manager.block_path(agent.handle, path)
            solution[agent.handle] = path

        return solution

    def _compute_best_path_for_agent(self, agent):
        """
        Computes the best path for a single agent using time-augmented A*.

        Args:
            agent: The agent object.

        Returns:
            list: A list of (node, time) tuples representing the path, or None if no path is found.
        """
        data = self.agent_data[agent.handle]
        return self._cooperative_a_star(
            agent_id=agent.handle,
            start_node=data["start_node"],
            goal_node=data["goal_node"],
            dist_map=data["dist_map"],
            earliest_departure=data["earliest_departure"],
        )

    def _cooperative_a_star(
        self, agent_id, start_node, goal_node, dist_map, earliest_departure
    ):
        """
        Performs a time-augmented A* search for a single agent.

        The search uses the original cost ("l") to compute arrival times for occupancy
        and the learned cost ("learned_l") (if available) to update the search cost (g-value).

        Args:
            agent_id (int): The unique identifier of the agent.
            start_node (tuple): The agent's start proxy node.
            goal_node (tuple): The shared goal proxy node.
            dist_map (dict): A mapping from normalized nodes to heuristic distances.
            earliest_departure (int): The agent's earliest departure time.

        Returns:
            list: A list of (node, time) tuples representing the computed path, or None if not found.
        """
        open_list = []
        visited = {}

        start_occ_t = int(earliest_departure)
        s_key = normalize_node(start_node)
        h_val = dist_map.get(s_key, inf)
        start_g = 0.0
        start_f = start_g + h_val

        heappush(open_list, (start_f, start_g, start_occ_t, start_node, None))

        while open_list:
            f_val, g_val, occ_t, node, parent = heappop(open_list)

            # Check goal: compare row and column only.
            if (get_row(node), get_col(node)) == (
                get_row(goal_node),
                get_col(goal_node),
            ):
                return self._reconstruct_path((node, occ_t, parent))

            if (node, occ_t) in visited and visited[(node, occ_t)] <= g_val:
                continue
            visited[(node, occ_t)] = g_val

            # Expand successor moves.
            for succ_node, new_g, new_occ_time in self._get_successors(
                agent_id, node, occ_t, g_val
            ):
                if (succ_node, new_occ_time) in visited and visited[
                    (succ_node, new_occ_time)
                ] <= new_g:
                    continue
                s2_key = normalize_node(succ_node)
                succ_h = dist_map.get(s2_key, inf)
                new_f = new_g + succ_h
                heappush(
                    open_list,
                    (new_f, new_g, new_occ_time, succ_node, (node, occ_t, parent)),
                )

        return None

    def _get_successors(self, agent_id, node, occ_t, g_val):
        """
        Generates successor states for a given node in the PP search.

        For each neighbor, arrival time is computed using the original edge cost ("l"),
        while the search cost is updated with the learned cost ("learned_l"). Additionally,
        a wait action is generated. If the agent is in its own start proxy, waiting is allowed
        unconditionally.

        Args:
            agent_id (int): The agent identifier.
            node (tuple): The current node.
            occ_t (int): The current occupancy time.
            g_val (float): The current cost value.

        Returns:
            list: A list of successor tuples (succ_node, new_g, arrival_time).
        """
        successors = []
        if node in self.nx_graph:
            for nbr in self.nx_graph.neighbors(node):
                edge_data = self.nx_graph.get_edge_data(node, nbr)
                orig_cost = edge_data.get("l", 1.0)
                learned_cost = float(edge_data.get("learned_l", orig_cost))
                arrival_time = int(occ_t + orig_cost)
                new_g = g_val + learned_cost
                nr, nc, nd = normalize_node(nbr)
                blocked = self.res_manager.is_blocked(
                    agent_id, get_row(node), get_col(node), nr, nc, arrival_time
                )
                if not blocked:
                    successors.append((nbr, new_g, arrival_time))
        # Expand wait action.
        wait_t = int(occ_t + 1)
        wait_cost = 1.0
        if (
            get_direction(node) == -1
            and self.nx_graph.nodes[node].get("agent_id", None) == agent_id
        ):
            # Agent is in its own start proxy: allow waiting unconditionally.
            successors.append((node, g_val + wait_cost, wait_t))
        else:
            if not self.res_manager.is_blocked(
                agent_id,
                get_row(node),
                get_col(node),
                get_row(node),
                get_col(node),
                wait_t,
            ):
                successors.append((node, g_val + wait_cost, wait_t))
        return successors

    def _reconstruct_path(self, final_state):
        """
        Reconstructs the path from the final state of the search.

        Args:
            final_state (tuple): A tuple (node, time, parent) from the low-level search.

        Returns:
            list: A list of (node, time) tuples representing the reconstructed path.
        """
        path = []
        curr = final_state
        while curr:
            node, occ_t, parent = curr
            path.append((node, int(occ_t)))
            curr = parent
        return list(reversed(path))
