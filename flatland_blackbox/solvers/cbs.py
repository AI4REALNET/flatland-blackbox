import copy
from heapq import heappop, heappush
from itertools import combinations

from flatland_blackbox.utils.graph_utils import (
    RailNode,
    decide_node,
    true_distance_heuristic,
)


class Constraints:
    """
    Tracks vertex and edge constraints in a CBS framework.
    """

    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_vertex_constraint(self, time, location):
        self.vertex_constraints.add(VertexConstraint(time, location))

    def add_edge_constraint(self, time, loc1, loc2):
        self.edge_constraints.add(EdgeConstraint(time, loc1, loc2))

    def is_vertex_constrained(self, time, location):
        return VertexConstraint(time, location) in self.vertex_constraints

    def is_edge_constrained(self, time, loc1, loc2):
        return EdgeConstraint(time, loc1, loc2) in self.edge_constraints


class VertexConstraint:
    def __init__(self, time, location):
        self.time = time
        self.location = location  # (row, col)

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return hash((self.time, self.location))


class EdgeConstraint:
    def __init__(self, time, loc1, loc2):
        self.time = time
        self.loc1 = loc1  # (row, col)
        self.loc2 = loc2  # (row, col)

    def __eq__(self, other):
        return (
            self.time == other.time
            and self.loc1 == other.loc1
            and self.loc2 == other.loc2
        )

    def __hash__(self):
        return hash((self.time, self.loc1, self.loc2))


class HighLevelNode:
    """
    A node in the high-level CBS search tree.
    """

    def __init__(self, solution, constraints, cost):
        # solution: dict {agent_id: [(RailNode, time), ...]}
        self.solution = solution
        self.constraints = constraints
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


class CBSSolver:
    """
    A Conflict-Based Search (CBS) solver that uses
    proxy start/goal nodes of the form (row, col, -1).
    """

    def __init__(self, nx_graph):
        """
        Args:
            nx_graph (nx.DiGraph): The entire or rail-subgraph (with proxy nodes) for pathfinding.
        """
        self.nx_graph = nx_graph
        self.open_list = []
        self.MAX_TIME = 2 * (self.nx_graph.number_of_nodes())

    def solve(self, agents):
        """
        Returns dict: agent_id -> [(RailNode, time), ...]
        """
        # Root node: no constraints, plan each agent individually
        root_constraints = Constraints()
        root_solution = {}
        for agent in agents:
            path = self.low_level_search(agent, root_constraints)
            if not path:
                raise ValueError(f"No path for agent {agent.handle}")
            root_solution[agent.handle] = path

        root_cost = self.compute_solution_cost(root_solution)
        root_node = HighLevelNode(root_solution, root_constraints, root_cost)
        heappush(self.open_list, root_node)

        # Main CBS loop
        while self.open_list:
            current = heappop(self.open_list)
            conflict = self.detect_conflict(current.solution)
            if conflict is None:
                # Conflict-free solution
                return current.solution

            # Generate child nodes with constraints
            for agent_id, constraint_info in self.generate_constraints(conflict):
                child_node = copy.deepcopy(current)
                # Add constraint
                if constraint_info["type"] == "vertex":
                    child_node.constraints.add_vertex_constraint(
                        constraint_info["time"], constraint_info["location"]
                    )
                else:
                    child_node.constraints.add_edge_constraint(
                        constraint_info["time"],
                        constraint_info["loc1"],
                        constraint_info["loc2"],
                    )

                # Replan path for the agent that got new constraint
                new_path = self.low_level_search(
                    agents[agent_id], child_node.constraints
                )
                if not new_path:
                    # No feasible path => skip
                    continue

                child_node.solution[agent_id] = new_path
                child_node.cost = self.compute_solution_cost(child_node.solution)
                heappush(self.open_list, child_node)

        raise ValueError("No solution found by CBS.")

    def low_level_search(self, agent, constraints):
        """
        A* for a single agent subject to constraints.
        Returns: [(RailNode, time), ...] or None if no path found.

        We assume the graph already has two proxy nodes for this agent:
          - Start proxy = (start_row, start_col, -1), type="proxy"
          - Goal proxy  = (goal_row,  goal_col,  -1), type="proxy"
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

        # A* with constraints
        path = self._cbs_a_star(start_node, goal_node, static_dist, constraints)
        return path

    def _cbs_a_star(self, start_node, goal_node, dist_dict, constraints):
        """
        The time-augmented A* used by CBS low-level search.
        constraints: Constraints object for vertex/edge constraints.
        """
        open_list = []
        visited = {}

        # Start f
        s_key = (start_node.row, start_node.col, start_node.direction)
        start_f = dist_dict.get(s_key, float("inf"))
        heappush(open_list, (start_f, 0.0, start_node, 0.0, None))

        while open_list:
            f_val, g_val, node, t, parent = heappop(open_list)

            # No solution if time is too large
            if t > self.MAX_TIME:
                continue

            if (node.row, node.col) == (goal_node.row, goal_node.col):
                return self._reconstruct_path((node, t, parent))

            if (node, t) in visited and visited[(node, t)] <= g_val:
                continue
            visited[(node, t)] = g_val

            raw_node = (node.row, node.col, node.direction)
            if raw_node in self.nx_graph:
                for nbr in self.nx_graph.neighbors(raw_node):
                    edge_data = self.nx_graph.get_edge_data(raw_node, nbr)
                    cost = edge_data.get("l", 1)
                    arrival_t = t + cost

                    nbr_node = RailNode(*nbr)

                    # Vertex constraint
                    if constraints.is_vertex_constrained(
                        arrival_t, (nbr_node.row, nbr_node.col)
                    ):
                        continue
                    # Edge constraint
                    if constraints.is_edge_constrained(
                        t, (node.row, node.col), (nbr_node.row, nbr_node.col)
                    ):
                        continue

                    new_g = g_val + cost
                    s2_key = (nbr_node.row, nbr_node.col, nbr_node.direction)
                    h_val = dist_dict.get(s2_key, float("inf"))
                    if h_val == float("inf"):
                        continue
                    new_f = new_g + h_val

                    if (nbr_node, arrival_t) in visited and visited[
                        (nbr_node, arrival_t)
                    ] <= new_g:
                        continue

                    heappush(
                        open_list,
                        (new_f, new_g, nbr_node, arrival_t, (node, t, parent)),
                    )

            # Expand wait
            wait_t = t + 1
            if not constraints.is_vertex_constrained(wait_t, (node.row, node.col)):
                new_g = g_val + 1
                s2_key = (node.row, node.col, node.direction)
                h_val = dist_dict.get(s2_key, float("inf"))
                if h_val == float("inf"):
                    continue
                new_f = new_g + h_val

                if (node, wait_t) not in visited or visited[(node, wait_t)] > new_g:
                    heappush(open_list, (new_f, new_g, node, wait_t, (node, t, parent)))

        return None

    def detect_conflict(self, solution):
        """
        Return the first conflict or None if conflict-free.
        solution: dict agent_id -> [(RailNode, time), ...]
        """
        if not solution:
            return None

        max_len = max(len(path) for path in solution.values())
        for t in range(max_len):
            for a1, a2 in combinations(solution.keys(), 2):
                path1 = solution[a1]
                path2 = solution[a2]
                # Vertex conflict
                if t < len(path1) and t < len(path2):
                    if (
                        path1[t][0].row == path2[t][0].row
                        and path1[t][0].col == path2[t][0].col
                    ):
                        # same (row, col)
                        return {
                            "type": "vertex",
                            "time": path1[t][1],
                            "agents": (a1, a2),
                            "location": (path1[t][0].row, path1[t][0].col),
                        }
                # Edge conflict (swap)
                if (t + 1 < len(path1)) and (t + 1 < len(path2)):
                    n1_t, n1_t1 = path1[t], path1[t + 1]
                    n2_t, n2_t1 = path2[t], path2[t + 1]
                    # If we have a swap:
                    if (
                        n1_t[0].row == n2_t1[0].row
                        and n1_t[0].col == n2_t1[0].col
                        and n1_t1[0].row == n2_t[0].row
                        and n1_t1[0].col == n2_t[0].col
                    ):
                        return {
                            "type": "edge",
                            "time": n1_t[1],
                            "agents": (a1, a2),
                            "loc1": (n1_t[0].row, n1_t[0].col),
                            "loc2": (n1_t1[0].row, n1_t1[0].col),
                        }
        return None

    def generate_constraints(self, conflict):
        """
        For each conflict, produce constraints for each of the conflicting agents.
        Yields (agent_id, constraint_info).
        """
        a1, a2 = conflict["agents"]
        if conflict["type"] == "vertex":
            # Each agent gets a vertex constraint
            c_info_1 = {
                "type": "vertex",
                "time": conflict["time"],
                "location": conflict["location"],
            }
            c_info_2 = dict(c_info_1)
            yield (a1, c_info_1)
            yield (a2, c_info_2)

        elif conflict["type"] == "edge":
            # Each agent gets an edge constraint
            c_info_1 = {
                "type": "edge",
                "time": conflict["time"],
                "loc1": conflict["loc1"],
                "loc2": conflict["loc2"],
            }
            c_info_2 = {
                "type": "edge",
                "time": conflict["time"],
                "loc1": conflict["loc2"],
                "loc2": conflict["loc1"],
            }
            yield (a1, c_info_1)
            yield (a2, c_info_2)

    def compute_solution_cost(self, solution):
        """
        E.g. sum of path lengths for all agents.
        """
        return sum(len(path) for path in solution.values())

    def _reconstruct_path(self, final_state):
        """
        Rebuilds path from (RailNode, time, parent).
        Returns: [(RailNode, time), ...]
        """
        path = []
        cur = final_state
        while cur:
            node, t, parent = cur
            path.append((node, t))
            cur = parent
        return list(reversed(path))
