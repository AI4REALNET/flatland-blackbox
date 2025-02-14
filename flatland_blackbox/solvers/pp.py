from heapq import heappop, heappush
from math import inf

from flatland_blackbox.utils.graph_utils import (
    RailNode,
    decide_node,
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
            r, c = node.row, node.col
            self.occupant_table[((r, c), t)] = agent_id

    def is_blocked(self, agent_id, pre_r, pre_c, r, c, t):
        """
        Checks if occupying (r,c) at time t by agent_id is blocked due to:
          1) Vertex conflict: occupant_table[((r,c), t)] has a different agent.
          2) Edge-swap conflict: occupant_table[((r,c), t-1)] == occupant_table[((pre_r,pre_c), t)]
             for some occupant != agent_id.
        """
        # 1) Vertex conflict
        occupant = self.occupant_table.get(((r, c), t), None)
        if occupant is not None and occupant != agent_id:
            return True

        # 2) Edge conflict check (direct swap)
        if t > 0:
            occupant_rc_tminus = self.occupant_table.get(((r, c), t - 1), None)
            occupant_prec_t = self.occupant_table.get(((pre_r, pre_c), t), None)
            if (
                occupant_rc_tminus is not None
                and occupant_rc_tminus == occupant_prec_t
                and occupant_rc_tminus != agent_id
            ):
                # Another agent is moving (r,c)@t-1 -> (pre_r,pre_c)@t
                # while we want (pre_r,pre_c)@t-1 -> (r,c)@t => swap conflict
                return True

        return False


class PrioritizedPlanningSolver:
    """
    Prioritized Planning (PP) plansagents in sequence,
    reserving their paths to avoid collisions:
      - plan agent 0, block path
      - plan agent 1, block path
      - ...
    """

    def __init__(self, nx_graph):
        self.nx_graph = nx_graph
        self.res_manager = ReservationManager()
        self.agent_data = {}

    def solve(self, agents, priority_order=None):
        """
        Plans each agent in sequence based on given priority order.
        Returns {agent.handle -> [(RailNode, time), ...]}.
        """
        # Store agent data
        for agent in agents:
            row_s, col_s = agent.initial_position
            row_g, col_g = agent.target

            start_tuple = decide_node(self.nx_graph, row_s, col_s, agent.handle)
            goal_tuple = decide_node(self.nx_graph, row_g, col_g, agent.handle)

            dist_map = true_distance_heuristic(self.nx_graph, goal_tuple)
            assert None not in dist_map.values(), "Found None in distance map"

            self.agent_data[agent.handle] = {
                "start_node": RailNode(*start_tuple),
                "goal_node": RailNode(*goal_tuple),
                "dist_map": dist_map,
                "earliest_departure": getattr(agent, "earliest_departure", 0),
            }

        # Order of planning
        if priority_order is None:
            planning_order = sorted(agents, key=lambda a: a.handle)
        else:
            handle_to_prio = {h: i for i, h in enumerate(priority_order)}
            planning_order = sorted(agents, key=lambda a: handle_to_prio[a.handle])

        # Plan each agent in turn
        solution = {}
        for agent in planning_order:
            path = self._compute_best_path_for_agent(agent)
            if not path:
                raise ValueError(f"No path found for agent {agent.handle}")
            # Block path
            self.res_manager.block_path(agent.handle, path)
            solution[agent.handle] = path

        return solution

    def _compute_best_path_for_agent(self, agent):
        # Retrieve data
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
        """
        Time-augmented A* with earliest_departure.
          - We start expansions at time=earliest_departure
        """
        open_list = []
        visited = {}

        # cost to reach start is earliest_departure, so we treat g_val= earliest_departure
        # f_val = g_val + h_val
        s_key = (start_node.row, start_node.col, start_node.direction)
        h_val = dist_map.get(s_key, inf)
        start_f = earliest_departure + h_val

        # push => (f_val, g_val, node, time, parent)
        heappush(
            open_list,
            (start_f, earliest_departure, start_node, earliest_departure, None),
        )

        while open_list:
            f_val, g_val, node, t, parent = heappop(open_list)

            # Goal check
            if (node.row, node.col) == (goal_node.row, goal_node.col):
                return self._reconstruct_path((node, t, parent))

            if (node, t) in visited and visited[(node, t)] <= g_val:
                continue
            visited[(node, t)] = g_val

            # Expand successors
            for succ_node, succ_time in self._get_successors(agent_id, node, t):
                new_g_val = g_val + (succ_time - t)  # increment by time delta
                if (succ_node, succ_time) in visited and visited[
                    (succ_node, succ_time)
                ] <= new_g_val:
                    continue

                s2_key = (succ_node.row, succ_node.col, succ_node.direction)
                succ_h_val = dist_map.get(s2_key, inf)
                if succ_h_val == inf:
                    continue

                new_f = new_g_val + succ_h_val
                heappush(
                    open_list,
                    (new_f, new_g_val, succ_node, succ_time, (node, t, parent)),
                )

        return None

    def _get_successors(self, agent_id, node, current_time):
        """
        Move + Wait expansions. We skip times < earliest_departure in practice by starting at earliest_departure.
        """
        successors = []
        # Move expansions
        if (node.row, node.col, node.direction) in self.nx_graph:
            for nbr in self.nx_graph.neighbors((node.row, node.col, node.direction)):
                edge_data = self.nx_graph.get_edge_data(
                    (node.row, node.col, node.direction), nbr
                )
                cost = edge_data.get("l", 1)
                arrival_t = current_time + cost

                nr, nc, nd = nbr
                blocked = self.res_manager.is_blocked(
                    agent_id, node.row, node.col, nr, nc, arrival_t
                )
                if not blocked:
                    successors.append((RailNode(nr, nc, nd), arrival_t))

        # Wait in place
        wait_t = current_time + 1
        if not self.res_manager.is_blocked(
            agent_id, node.row, node.col, node.row, node.col, wait_t
        ):
            successors.append((node, wait_t))

        return successors

    def _reconstruct_path(self, final_state):
        path = []
        current = final_state
        while current:
            node, t, parent = current
            path.append((node, t))
            current = parent
        return list(reversed(path))
