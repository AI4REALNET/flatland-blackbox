import copy
from heapq import heappop, heappush
from itertools import combinations

from flatland_blackbox.utils.graph_utils import (
    RailNode,
    decide_node,
    true_distance_heuristic,
)


class Constraints:
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
    def __init__(self, solution, constraints, cost):
        # solution: dict {agent_id: [(RailNode, time), ...]}
        self.solution = solution
        self.constraints = constraints
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


class CBSSolver:
    """
    A Conflict-Based Search (CBS) solver.
    """

    def __init__(self, nx_graph):
        self.nx_graph = nx_graph
        self.open_list = []
        self.MAX_TIME = self.nx_graph.number_of_nodes()
        self.agent_data = (
            {}
        )  # store {agent_id: {"start_node", "goal_node", "dist_map"}}
        self.plan_cache = {}  # key: (agent_id, constraints_key) -> path

    def solve(self, agents, max_high_level_expansions=10000):
        """
        Returns dict: agent_id -> [(RailNode, time), ...]
        1) For each agent, decide start/goal proxies, compute dist_map once.
        2) Build root constraints, plan each agent's path => root node.
        3) CBS search with detect_conflict and generate_constraints
        """
        # 1) Precompute agent data
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

        # 2) Build root constraints (none) and root solution
        root_constraints = Constraints()
        root_solution = {}
        for agent in agents:
            path = self._cbs_a_star(agent_id=agent.handle, constraints=root_constraints)
            if not path:
                raise ValueError(f"No path found for agent {agent.handle}")
            root_solution[agent.handle] = path

        root_cost = self.compute_solution_cost(root_solution)
        root_node = HighLevelNode(root_solution, root_constraints, root_cost)
        heappush(self.open_list, root_node)

        # 3) Main CBS loop
        expansions_count = 0
        while self.open_list:
            expansions_count += 1
            if expansions_count > max_high_level_expansions:
                raise ValueError("CBS high-level search exceeded expansion limit.")

            current = heappop(self.open_list)
            conflict = self.detect_conflict(current.solution)
            if conflict is None:
                return current.solution

            for agent_id, constraint_info in self.generate_constraints(conflict):
                child_node = copy.deepcopy(current)
                ctype = constraint_info["type"]
                if ctype == "vertex":
                    child_node.constraints.add_vertex_constraint(
                        constraint_info["time"], constraint_info["location"]
                    )
                else:  # edge conflict
                    child_node.constraints.add_edge_constraint(
                        constraint_info["time"],
                        constraint_info["loc1"],
                        constraint_info["loc2"],
                    )

                # Replan for that agent
                new_path = self._cbs_a_star(
                    agent_id=agent_id, constraints=child_node.constraints
                )
                if not new_path:
                    continue
                child_node.solution[agent_id] = new_path
                child_node.cost = self.compute_solution_cost(child_node.solution)
                heappush(self.open_list, child_node)

        raise ValueError("No solution found by CBS")

    def _cbs_a_star(self, agent_id, constraints):
        # Make a hashable key for these constraints
        constraints_key = self._constraints_to_key(constraints)

        # If we have seen exactly this (agent, constraints) combo reuse it
        if (agent_id, constraints_key) in self.plan_cache:
            return self.plan_cache[(agent_id, constraints_key)]

        data = self.agent_data[agent_id]
        start_node = data["start_node"]
        goal_node = data["goal_node"]
        dist_dict = data["dist_map"]
        edt = data["earliest_departure"]

        open_list = []
        visited = {}

        # Compute an f-value that includes starting at time edt
        s_key = (start_node.row, start_node.col, start_node.direction)
        start_h = dist_dict.get(s_key, float("inf"))
        # g_val = edt; f_val = g_val + h_val
        start_f = edt + start_h

        heappush(open_list, (start_f, float(edt), start_node, float(edt), None))

        shortest_dist = dist_dict.get(s_key, float("inf"))
        time_bound = edt + (shortest_dist * 3)

        while open_list:
            f_val, g_val, node, t, parent = heappop(open_list)

            # if t > time_bound:
            #     continue

            # Goal check: same cell as goal, ignoring direction
            if (node.row, node.col) == (goal_node.row, goal_node.col):
                path = self._reconstruct_path((node, t, parent))
                # If found path, store in cache *before* returning
                self.plan_cache[(agent_id, constraints_key)] = path
                return path

            if (node, t) in visited and visited[(node, t)] <= g_val:
                continue
            visited[(node, t)] = g_val

            # Expand neighbors
            raw_node = (node.row, node.col, node.direction)
            if raw_node in self.nx_graph:
                for nbr in self.nx_graph.neighbors(raw_node):
                    cost = self.nx_graph.get_edge_data(raw_node, nbr).get("l", 1)
                    arrival = t + cost
                    nbr_node = RailNode(*nbr)

                    # Constraints
                    if constraints.is_vertex_constrained(
                        arrival, (nbr_node.row, nbr_node.col)
                    ):
                        continue
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

                    if (nbr_node, arrival) not in visited or visited[
                        (nbr_node, arrival)
                    ] > new_g:
                        heappush(
                            open_list,
                            (new_f, new_g, nbr_node, arrival, (node, t, parent)),
                        )

            # Wait action
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

        # If no path is found, store None in the cache too
        self.plan_cache[(agent_id, constraints_key)] = None
        return None

    def detect_conflict(self, solution):
        if not solution:
            return None

        # 1) Find the max time among all agents
        max_time = 0
        for path in solution.values():
            if path:
                max_time = max(max_time, path[-1][1])

        # 2) For t = 0..max_time, gather each agent’s position
        for t in range(int(max_time) + 1):
            # Dictionary: agent_id -> (row, col) at time t
            positions = {}
            # Also track "previous position" to check edge swaps
            prev_positions = {}

            for agent_id, path in solution.items():
                # Find the position at time t (if any)
                # We can walk the path or do a simple search:
                # If path = [(node, time), (node, time), ...], find the largest index i
                # s.t. path[i].time <= t. That node is the agent's location *during* [ time[i], time[i+1] ).
                # A quick way is a reverse search or a binary search. For simplicity:
                pos_idx = None
                for i in range(len(path)):
                    if path[i][1] == t:
                        # Exactly at time t
                        pos_idx = i
                    elif path[i][1] > t:
                        # This agent’s time jumped past t
                        # So the position is the i-1 node, if i>0
                        pos_idx = i - 1
                        break

                if pos_idx is not None and 0 <= pos_idx < len(path):
                    node_t, real_t = path[pos_idx]
                    if node_t is not None:
                        positions[agent_id] = (node_t.row, node_t.col)

                # For the "previous position" at time t-1 (for edge-swap checks)
                if t > 0:
                    # same logic for time = t-1
                    pos_idx2 = None
                    for i in range(len(path)):
                        if path[i][1] == t - 1:
                            pos_idx2 = i
                        elif path[i][1] > t - 1:
                            pos_idx2 = i - 1
                            break
                    if pos_idx2 is not None and 0 <= pos_idx2 < len(path):
                        node_t2, real_t2 = path[pos_idx2]
                        if node_t2 is not None:
                            prev_positions[agent_id] = (node_t2.row, node_t2.col)

            # Now check for vertex conflicts
            agent_list = list(positions.keys())
            for i in range(len(agent_list)):
                a1 = agent_list[i]
                for j in range(i + 1, len(agent_list)):
                    a2 = agent_list[j]
                    if positions[a1] == positions[a2]:
                        return {
                            "type": "vertex",
                            "time": t,
                            "agents": (a1, a2),
                            "location": positions[a1],
                        }

            # Check for edge conflicts (a1 moves from X to Y while a2 moves from Y to X)
            if t > 0:
                for a1, a2 in combinations(prev_positions.keys(), 2):
                    if a1 not in positions or a2 not in positions:
                        continue
                    # a1 was at prev_positions[a1] at time t-1, and is at positions[a1] at time t
                    # a2 was at prev_positions[a2] at time t-1, and is at positions[a2] at time t
                    if (
                        prev_positions[a1] == positions[a2]
                        and prev_positions[a2] == positions[a1]
                    ):
                        return {
                            "type": "edge",
                            "time": t - 1,  # The swap started at t-1
                            "agents": (a1, a2),
                            "loc1": prev_positions[a1],
                            "loc2": prev_positions[a2],
                        }
        return None

    def generate_constraints(self, conflict):
        a1, a2 = conflict["agents"]
        ctype = conflict["type"]
        if ctype == "vertex":
            c_info = {
                "type": "vertex",
                "time": conflict["time"],
                "location": conflict["location"],
            }
            yield (a1, c_info)
            yield (a2, dict(c_info))
        else:
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
        return sum(len(path) for path in solution.values())

    def _reconstruct_path(self, final_state):
        path = []
        curr = final_state
        while curr:
            node, t, parent = curr
            path.append((node, t))
            curr = parent
        return list(reversed(path))

    def _constraints_to_key(self, constraints):
        """
        Convert all constraints into an immutable tuple
        so we can store them in a dict for caching.
        """
        vertex_key = frozenset(
            (vc.time, vc.location) for vc in constraints.vertex_constraints
        )
        edge_key = frozenset(
            (ec.time, ec.loc1, ec.loc2) for ec in constraints.edge_constraints
        )
        return (vertex_key, edge_key)
