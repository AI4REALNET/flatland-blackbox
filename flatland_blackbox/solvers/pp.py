from heapq import heappop, heappush

from flatland_blackbox.utils.graph_utils import (
    RailNode,
    decide_node,
    true_distance_heuristic,
)


class ReservationManager:
    """
    Manages time-step reservations (row, col) for collision avoidance.
    """

    def __init__(self):
        # Reservation table stores ((row, col), time)
        self.reservation_table = set()

    def block_path(self, path):
        """
        Block the reservation table for each step in the path.
        path: list of ((RailNode), time)
        """
        for i in range(len(path)):
            node, t = path[i]
            self.reservation_table.add(((node.row, node.col), t))

            if i < len(path) - 1:
                next_node, next_t = path[i + 1]
                self.reservation_table.add(((next_node.row, next_node.col), next_t))
                # Block head-on swap (edge conflict)
                self.reservation_table.add(((node.row, node.col), next_t))
                self.reservation_table.add(((next_node.row, next_node.col), t))

    def is_blocked(self, row, col, time):
        return ((row, col), time) in self.reservation_table


class PrioritizedPlanningSolver:
    """
    Prioritized Planning (PP) plansagents in sequence,
    reserving their paths to avoid collisions.
    """

    def __init__(self, nx_graph):
        """
        Args:
            nx_graph (nx.DiGraph): The full (or rail-subgraph) for pathfinding.
        """
        self.nx_graph = nx_graph
        self.res_manager = ReservationManager()

    def solve(self, agents):
        """
        Plans each agent in sequence (i.e., prioritized).

        Returns:
            dict: {agent.handle -> [(RailNode, time), ...]}
        """
        solution = {}
        for agent in agents:
            path = self._compute_best_path_for_agent(agent)
            if not path:
                raise ValueError(f"No path found for agent {agent.handle}.")

            self.res_manager.block_path(path)
            solution[agent.handle] = path

            # print(f"[DEBUG] After agent {i} -> reservation table size ="
            #       f" {len(self.res_manager.reservation_table)}")

        return solution

    def _compute_best_path_for_agent(self, agent):
        """
        Decide which node in the graph to use as the start and goal node,
        based on how many (row, col) matches exist. If there's multiple,
        we specifically look for direction=-1.
        """
        row_s, col_s = agent.initial_position
        row_g, col_g = agent.target

        # Decide the start and goal node
        start_tuple = decide_node(self.nx_graph, row_s, col_s, agent.handle)
        goal_tuple = decide_node(self.nx_graph, row_g, col_g, agent.handle)

        # Precompute distances to the goal
        static_dist = true_distance_heuristic(self.nx_graph, goal_tuple)
        assert None not in static_dist.values(), "Found None in distance map"

        # Convert them to RailNode
        start_node = RailNode(*start_tuple)
        goal_node = RailNode(*goal_tuple)

        path = self._cooperative_a_star(start_node, goal_node, static_dist)
        return path

    def _cooperative_a_star(self, start_node, goal_node, static_dist):
        """
        Time-augmented A*:
          - g_val = time so far
          - h_val = static_dist[node]
          - f_val = g_val + h_val
        Returns: [(RailNode, time), ...] or None if not found.
        """
        open_list = []  # (f_val, g_val, RailNode, time, parent)
        visited = {}

        # Convert start_node to a tuple if it isn't one, for indexing
        s_key = (start_node.row, start_node.col, start_node.direction)
        start_f = static_dist.get(s_key, float("inf"))
        heappush(open_list, (start_f, 0, start_node, 0, None))

        while open_list:
            # print("[DEBUG]: ", open_list)
            f_val, g_val, node, t, parent = heappop(open_list)

            # If we've reached the goal
            if (node.row, node.col) == (goal_node.row, goal_node.col):
                return self._reconstruct_path((node, t, parent))

            if (node, t) in visited and visited[(node, t)] <= g_val:
                continue
            visited[(node, t)] = g_val

            # Expand successors
            for succ_node, succ_time in self._get_successors(node, t):
                # Check reservation
                if self.res_manager.is_blocked(succ_node.row, succ_node.col, succ_time):
                    continue

                new_g_val = g_val + (succ_time - t)
                if (succ_node, succ_time) in visited and visited[
                    (succ_node, succ_time)
                ] <= new_g_val:
                    continue

                s2_key = (succ_node.row, succ_node.col, succ_node.direction)
                h_val = static_dist.get(s2_key, float("inf"))
                if h_val == float("inf"):
                    continue
                new_f = new_g_val + h_val

                heappush(
                    open_list,
                    (new_f, new_g_val, succ_node, succ_time, (node, t, parent)),
                )
        return None

    def _get_successors(self, node, current_time):
        """
        Return list of (RailNode, arrival_time) from 'node' at 'current_time'.
        Move + wait expansions.
        """
        successors = []
        # Move to neighbors
        if (node.row, node.col, node.direction) in self.nx_graph:
            for neighbor in self.nx_graph.neighbors(
                (node.row, node.col, node.direction)
            ):
                edge_data = self.nx_graph.get_edge_data(
                    (node.row, node.col, node.direction), neighbor
                )
                cost = edge_data.get("l", 1)
                # print(cost)
                arrival_t = current_time + cost

                succ_node = RailNode(*neighbor)
                successors.append((succ_node, arrival_t))

        # Wait in place
        wait_t = current_time + 1
        successors.append((node, wait_t))  # same node, but time+1

        return successors

    def _reconstruct_path(self, final_state):
        """
        final_state = (RailNode, time, parent).
        Return path = [(RailNode, time), ...], reversed from the chain.
        """
        path = []
        current = final_state
        while current:
            node, t, parent = current
            path.append((node, t))
            current = parent
        return list(reversed(path))
