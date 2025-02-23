from heapq import heappop, heappush
from math import inf

from flatland_blackbox.utils import (
    NoSolutionError,
    get_col,
    get_goal_proxy_node,
    get_row,
    get_start_proxy_node,
    normalize_node,
    true_distance_heuristic,
)


class ReservationManager:
    """
    Manages time-step reservations for collision avoidance,
    storing occupant agent_id in occupant_table.
    """

    def __init__(self):
        # occupant_table[((row, col), time)] = agent_id (int), or None if unoccupied
        self.occupant_table = {}

    def block_path(self, agent_id, path):
        """
        Once we finalize a path for agent_id, mark occupant=agent_id at each step.
        """
        for node, t in path:
            r, c = get_row(node), get_col(node)
            self.occupant_table[((r, c), t)] = agent_id

    def is_blocked(self, agent_id, pre_r, pre_c, r, c, t):
        """
        Checks if occupying (r,c) at time t by agent_id is blocked due to:
          1) Vertex conflict: occupant_table[((r,c), t)] has a different agent.
          2) Edge-swap conflict: occupant_table[((r,c), t-1)] == occupant_table[((pre_r,pre_c), t)]
             for some occupant != agent_id.
        """
        t_int = int(t)
        # Vertex conflict
        occupant = self.occupant_table.get(((r, c), t_int), None)
        if occupant is not None and occupant != agent_id:
            return True

        # Edge conflict check (direct swap)
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
    Prioritized Planning (PP) planning agents in sequence.
    Supports a second edge attribute 'learned_l' for expansions priority,
    and 'l' (the original cost) for occupant times.
    If 'learned_l' is not found, fallback to 'l'.
    """

    def __init__(self, nx_graph):
        self.nx_graph = nx_graph
        self.res_manager = ReservationManager()
        self.agent_data = {}

    def solve(self, agents):
        """
        Plans each agent in sequence.
        Priority is the order of agents in the list.
        Returns {agent.handle -> [(RailNode, time), ...]}.
        """
        # Store agent data
        for agent in agents:
            row_s, col_s = agent.initial_position
            row_g, col_g = agent.target
            start_node = get_start_proxy_node(
                self.nx_graph, row_s, col_s, agent.handle
            )  # (row, col, -1, agent_id)
            goal_node = get_goal_proxy_node(
                self.nx_graph, row_g, col_g
            )  # (end_r, end_c, -1)

            # If "learned_l" exists uses it over original weights "l"
            dist_map = true_distance_heuristic(self.nx_graph, goal_node)
            assert None not in dist_map.values(), "Found None in distance map"

            self.agent_data[agent.handle] = {
                "start_node": start_node,
                "goal_node": goal_node,
                "dist_map": dist_map,
                "earliest_departure": getattr(agent, "earliest_departure", 0),
            }

        # Plan each agent in turn
        solution = {}
        for agent in agents:
            path = self._compute_best_path_for_agent(agent)
            if not path:
                raise NoSolutionError(f"No path found for agent {agent.handle}")
            # Block path
            self.res_manager.block_path(agent.handle, path)
            solution[agent.handle] = path

        return solution

    def _compute_best_path_for_agent(self, agent):
        data = self.agent_data[agent.handle]
        path = self._cooperative_a_star(
            agent_id=agent.handle,
            start_node=data["start_node"],
            goal_node=data["goal_node"],
            dist_map=data["dist_map"],
            earliest_departure=data["earliest_departure"],
        )
        return path

    def _cooperative_a_star(
        self, agent_id, start_node, goal_node, dist_map, earliest_departure
    ):
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

            # Check goal: compare row and col only.
            if (get_row(node), get_col(node)) == (
                get_row(goal_node),
                get_col(goal_node),
            ):
                return self._reconstruct_path((node, occ_t, parent))

            if (node, occ_t) in visited and visited[(node, occ_t)] <= g_val:
                continue
            visited[(node, occ_t)] = g_val

            for succ_node, new_g, new_occ_time in self._get_successors(
                agent_id, node, occ_t, g_val
            ):
                if (succ_node, new_occ_time) in visited and visited[
                    (succ_node, new_occ_time)
                ] <= new_g:
                    continue
                s2_key = normalize_node(succ_node)
                succ_h = dist_map.get(s2_key, inf)
                f_val_succ = new_g + succ_h
                heappush(
                    open_list,
                    (f_val_succ, new_g, new_occ_time, succ_node, (node, occ_t, parent)),
                )

        return None

    def _get_successors(self, agent_id, node, occ_t, g_val):
        successors = []
        # Use the full node as key.
        if node in self.nx_graph:
            for nbr in self.nx_graph.neighbors(node):
                edge_data = self.nx_graph.get_edge_data(node, nbr)
                orig_cost = edge_data.get("l", 1.0)
                learned_cost = float(edge_data.get("learned_l", orig_cost))
                arrival_time = int(occ_t + orig_cost)
                new_g = g_val + learned_cost
                nr, nc, nd = normalize_node(
                    nbr
                )  # nbr might be 3 or 4 elements; we care about row, col, direction.
                blocked = self.res_manager.is_blocked(
                    agent_id, get_row(node), get_col(node), nr, nc, arrival_time
                )
                if not blocked:
                    successors.append((nbr, new_g, arrival_time))

        # Wait action.
        wait_t = int(occ_t + 1)
        wait_learned = 1.0
        if not self.res_manager.is_blocked(
            agent_id, get_row(node), get_col(node), get_row(node), get_col(node), wait_t
        ):
            successors.append((node, g_val + wait_learned, wait_t))

        return successors

    def _reconstruct_path(self, final_state):
        path = []
        curr = final_state
        while curr:
            node, occ_t, parent = curr
            path.append((node, int(occ_t)))  # store occupant_time as int
            curr = parent
        return list(reversed(path))
