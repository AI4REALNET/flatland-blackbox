from copy import deepcopy
from heapq import heappop, heappush
from math import inf

from flatland_blackbox.utils import (
    NoSolutionError,
    get_col,
    get_goal_proxy_node,
    get_row,
    get_start_proxy_node,
    is_proxy_node,
    normalize_node,
    true_distance_heuristic,
)

# Portions of this code were adapted and heavily modified from multi_agent_path_planning,
# available at https://github.com/atb033/multi_agent_path_planning/.

class Constraints:
    """Container for vertex and edge constraints used in CBS.

    Attributes:
        vertex_constraints (set): A set of VertexConstraint objects.
        edge_constraints (set): A set of EdgeConstraint objects.
    """

    def __init__(self):
        """Initializes an empty set of vertex and edge constraints."""
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_vertex_constraint(self, time, location):
        """Adds a vertex constraint.

        Args:
            time (int): The time at which the constraint applies.
            location (tuple): The (row, col) of the constrained cell.
        """
        self.vertex_constraints.add(VertexConstraint(time, location))

    def add_edge_constraint(self, time, loc1, loc2):
        """Adds an edge constraint.

        Args:
            time (int): The time at which the constraint applies.
            loc1 (tuple): The (row, col) of the source cell.
            loc2 (tuple): The (row, col) of the destination cell.
        """
        self.edge_constraints.add(EdgeConstraint(time, loc1, loc2))

    def is_vertex_constrained(self, time, location):
        """Checks if a vertex is constrained at a given time.

        Args:
            time (int): The time step.
            location (tuple): The (row, col) location.

        Returns:
            bool: True if a vertex constraint exists, False otherwise.
        """
        return VertexConstraint(time, location) in self.vertex_constraints

    def is_edge_constrained(self, time, loc1, loc2):
        """Checks if an edge is constrained at a given time.

        Args:
            time (int): The time step.
            loc1 (tuple): The (row, col) of the source.
            loc2 (tuple): The (row, col) of the destination.

        Returns:
            bool: True if an edge constraint exists, False otherwise.
        """
        return EdgeConstraint(time, loc1, loc2) in self.edge_constraints


class VertexConstraint:
    """Represents a vertex constraint at a specific time.

    Attributes:
        time (int): The time at which the constraint applies.
        location (tuple): The (row, col) location of the vertex.
    """

    def __init__(self, time, location):
        """Initializes a VertexConstraint.

        Args:
            time (int): The time step.
            location (tuple): The (row, col) location.
        """
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return hash((self.time, self.location))


class EdgeConstraint:
    """Represents an edge constraint at a specific time.

    Attributes:
        time (int): The time at which the constraint applies.
        loc1 (tuple): The (row, col) of the source vertex.
        loc2 (tuple): The (row, col) of the destination vertex.
    """

    def __init__(self, time, loc1, loc2):
        """Initializes an EdgeConstraint.

        Args:
            time (int): The time step.
            loc1 (tuple): The source cell (row, col).
            loc2 (tuple): The destination cell (row, col).
        """
        self.time = time
        self.loc1 = loc1
        self.loc2 = loc2

    def __eq__(self, other):
        return (
            self.time == other.time
            and self.loc1 == other.loc1
            and self.loc2 == other.loc2
        )

    def __hash__(self):
        return hash((self.time, self.loc1, self.loc2))


class HighLevelNode:
    """Represents a high-level node in CBS.

    Attributes:
        solution (dict): Mapping from agent_id to a list of (node, time) tuples.
        constraints (Constraints): The set of constraints at this high-level node.
        cost (float): The total cost of the current solution.
    """

    def __init__(self, solution, constraints, cost):
        """Initializes a HighLevelNode.

        Args:
            solution (dict): A mapping from agent IDs to their planned paths.
            constraints (Constraints): The constraints imposed at this node.
            cost (float): The cost of the solution.
        """
        self.solution = solution
        self.constraints = constraints
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


class CBSSolver:
    """A Conflict-Based Search (CBS) solver.

    This solver plans paths for multiple agents by first planning individual paths
    and then resolving conflicts via high-level search over constraint sets.

    Attributes:
        nx_graph (nx.Graph): The graph representing the environment.
        vanish_at_goal (bool): If True, agents vanish after reaching the goal (reducing conflicts).
        open_list (list): The high-level open list (priority queue).
        agent_data (dict): A mapping from agent_id to a dict containing:
            - start_node: The agent’s start proxy node.
            - goal_node: The shared goal proxy node.
            - dist_map: The heuristic distance map computed from the goal.
            - earliest_departure: The agent's earliest departure time.
        plan_cache (dict): Cache of low-level plans, keyed by (agent_id, constraints_key).
    """

    def __init__(self, nx_graph, vanish_at_goal=True):
        """Initializes the CBS solver.

        Args:
            nx_graph (nx.Graph): The graph representing the environment.
            vanish_at_goal (bool, optional): Whether agents vanish upon reaching their goal.
                Defaults to True.
        """
        self.nx_graph = nx_graph
        self.vanish_at_goal = vanish_at_goal
        self.open_list = []
        self.agent_data = {}
        self.plan_cache = {}

    def solve(self, agents, max_high_level_expansions=50_000):
        """Plans paths for the given agents using CBS.

        This method precomputes necessary agent data, builds the root high-level node,
        and then performs a high-level search over constraint sets until a conflict-free
        solution is found or the expansion limit is exceeded.

        Args:
            agents (list): List of agent objects.
            max_high_level_expansions (int, optional): Maximum allowed high-level expansions.
                Defaults to 30_000.

        Returns:
            dict: Mapping from agent_id to a list of (node, time) tuples representing the plan.

        Raises:
            NoSolutionError: If no conflict-free solution is found within the expansion limit.
        """
        # Precompute agent data.
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

        # Build root constraints (empty) and root solution.
        root_constraints = Constraints()
        root_solution = {}
        for agent in agents:
            path = self._cbs_a_star(agent_id=agent.handle, constraints=root_constraints)
            if not path:
                raise NoSolutionError(f"No path found for agent {agent.handle}")
            root_solution[agent.handle] = path

        root_cost = self.compute_solution_cost(root_solution)
        root_node = HighLevelNode(root_solution, root_constraints, root_cost)
        heappush(self.open_list, root_node)

        # High-level CBS loop.
        expansions_count = 0
        while self.open_list:
            expansions_count += 1
            # if expansions_count % 1000 == 0:
            #     print(
            #         f"  CBS High-level expansions: {expansions_count}, open list size: {len(self.open_list)}"
            #     )
            if expansions_count > max_high_level_expansions:
                raise NoSolutionError(
                    f"CBS high-level search exceeded expansion limit ({max_high_level_expansions})."
                )
            current = heappop(self.open_list)
            conflict = self.detect_conflict(current.solution)
            if conflict is None:
                return current.solution
            for agent_id, constraint_info in self.generate_constraints(conflict):
                child_node = deepcopy(current)
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
                new_path = self._cbs_a_star(
                    agent_id=agent_id, constraints=child_node.constraints
                )
                if not new_path:
                    continue
                child_node.solution[agent_id] = new_path
                child_node.cost = self.compute_solution_cost(child_node.solution)
                heappush(self.open_list, child_node)

        raise NoSolutionError("No solution found by CBS")

    def _cbs_a_star(self, agent_id, constraints):
        """Low-level search for an individual agent under given constraints.

        This method uses a time-augmented A* search to compute a path for a single agent,
        considering vertex and edge constraints.

        Args:
            agent_id: The identifier for the agent.
            constraints (Constraints): The current set of constraints.

        Returns:
            list: A list of (node, time) tuples representing the path, or None if no path is found.
        """
        constraints_key = self._constraints_to_key(constraints)
        if (agent_id, constraints_key) in self.plan_cache:
            return self.plan_cache[(agent_id, constraints_key)]
        data = self.agent_data[agent_id]
        start_node = data["start_node"]
        goal_node = data["goal_node"]
        dist_dict = data["dist_map"]
        edt = data["earliest_departure"]
        open_list = []
        visited = {}
        start_occ_time = int(edt)
        s_key = normalize_node(start_node)
        start_h = dist_dict.get(s_key, inf)
        start_f = edt + start_h
        start_g = 0.0
        heappush(open_list, (start_f, start_g, start_node, start_occ_time, None))
        while open_list:
            f_val, g_val, node, t, parent = heappop(open_list)
            if (get_row(node), get_col(node)) == (
                get_row(goal_node),
                get_col(goal_node),
            ):
                path = self._reconstruct_path((node, t, parent))
                self.plan_cache[(agent_id, constraints_key)] = path
                return path
            if (node, t) in visited and visited[(node, t)] <= g_val:
                continue
            visited[(node, t)] = g_val
            if node in self.nx_graph:
                for nbr in self.nx_graph.neighbors(node):
                    cost = self.nx_graph.get_edge_data(node, nbr).get("l", 1)
                    arrival = int(t + cost)
                    if constraints.is_vertex_constrained(
                        arrival, (get_row(nbr), get_col(nbr))
                    ):
                        continue
                    if constraints.is_edge_constrained(
                        t, (get_row(node), get_col(node)), (get_row(nbr), get_col(nbr))
                    ):
                        continue
                    new_g = g_val + cost
                    s2_key = normalize_node(nbr)
                    h_val = dist_dict.get(s2_key, inf)
                    if h_val == inf:
                        continue
                    new_f = new_g + h_val
                    if (nbr, arrival) not in visited or visited[(nbr, arrival)] > new_g:
                        heappush(
                            open_list, (new_f, new_g, nbr, arrival, (node, t, parent))
                        )
            wait_t = int(t + 1)
            wait_cost = 1.0
            if (
                is_proxy_node(node)
                and self.nx_graph.nodes[node].get("agent_id", None) == agent_id
            ):
                new_g = g_val + wait_cost
                s2_key = normalize_node(node)
                h_val = dist_dict.get(s2_key, inf)
                if h_val != inf:
                    new_f = new_g + h_val
                    heappush(open_list, (new_f, new_g, node, wait_t, (node, t, parent)))
            else:
                if not constraints.is_vertex_constrained(
                    wait_t, (get_row(node), get_col(node))
                ):
                    new_g = g_val + wait_cost
                    s2_key = normalize_node(node)
                    h_val = dist_dict.get(s2_key, inf)
                    if h_val != inf:
                        new_f = new_g + h_val
                        heappush(
                            open_list, (new_f, new_g, node, wait_t, (node, t, parent))
                        )
        self.plan_cache[(agent_id, constraints_key)] = None
        return None

    def detect_conflict(self, solution):
        """Detects conflicts among the agents' paths.

        This method checks for vertex conflicts (multiple agents occupying the same cell at the same time)
        and edge conflicts (agents swapping positions in one time step).

        Args:
            solution (dict): Mapping from agent_id to a list of (node, time) tuples.

        Returns:
            dict or None: A conflict dictionary with keys "type", "time", "agents", and "location"
            if a conflict is found; otherwise, None.
        """
        global_min_time = min(
            self.agent_data[agent_id]["earliest_departure"] for agent_id in solution
        )
        max_time = 0
        for agent_id, path in solution.items():
            if path:
                t_last = int(path[-1][1])
                max_time = max(max_time, t_last)
        # Vertex Conflict Check.
        for t in range(int(global_min_time), max_time + 1):
            pos_to_agents = {}
            for agent_id, path in solution.items():
                edt = self.agent_data[agent_id]["earliest_departure"]
                if t < edt:
                    continue
                if self.vanish_at_goal and t > int(path[-1][1]):
                    continue
                node = None
                for n, t_val in path:
                    if int(t_val) == t:
                        # Skip if node is a proxy.
                        if is_proxy_node(n):
                            node = None
                            break
                        node = n
                        break
                if node is None:
                    continue
                pos = (get_row(node), get_col(node))
                pos_to_agents.setdefault(pos, []).append(agent_id)
                if len(pos_to_agents[pos]) > 1:
                    return {
                        "type": "vertex",
                        "time": t,
                        "agents": tuple(pos_to_agents[pos]),
                        "location": pos,
                    }
        # Edge Conflict Check.
        moves = {}
        for agent_id, path in solution.items():
            edt = self.agent_data[agent_id]["earliest_departure"]
            for i in range(len(path) - 1):
                (node1, t1) = path[i]
                (node2, t2) = path[i + 1]
                if t2 > t1 and t1 >= edt:
                    if self.vanish_at_goal and t2 > int(path[-1][1]):
                        continue
                    key = (
                        int(t1),
                        (get_row(node1), get_col(node1)),
                        (get_row(node2), get_col(node2)),
                    )
                    moves.setdefault(key, []).append(agent_id)
        for (t, pos_from, pos_to), agents in moves.items():
            reverse_key = (t, pos_to, pos_from)
            if reverse_key in moves:
                for a1 in agents:
                    for a2 in moves[reverse_key]:
                        if a1 != a2:
                            return {
                                "type": "edge",
                                "time": t,
                                "agents": (a1, a2),
                                "loc1": pos_from,
                                "loc2": pos_to,
                            }
        return None

    def generate_constraints(self, conflict):
        """Generates new constraints from a detected conflict.

        For a vertex conflict, two branches are generated (one for each conflicting agent).
        For an edge conflict, each agent is constrained on the corresponding edge transition.

        Args:
            conflict (dict): A conflict dictionary with keys "type", "time", "agents", etc.

        Yields:
            tuple: A tuple (agent_id, constraint_info) for each generated constraint.
        """
        a1, a2 = conflict["agents"]
        ctype = conflict["type"]
        if ctype == "vertex":
            yield (
                a1,
                {
                    "type": "vertex",
                    "time": conflict["time"],
                    "location": conflict["location"],
                },
            )
            yield (
                a2,
                {
                    "type": "vertex",
                    "time": conflict["time"],
                    "location": conflict["location"],
                },
            )
        else:
            yield (
                a1,
                {
                    "type": "edge",
                    "time": conflict["time"],
                    "loc1": conflict["loc1"],
                    "loc2": conflict["loc2"],
                },
            )
            yield (
                a2,
                {
                    "type": "edge",
                    "time": conflict["time"],
                    "loc1": conflict["loc2"],
                    "loc2": conflict["loc1"],
                },
            )

    def compute_solution_cost(self, solution):
        """Computes the total flow time cost of a solution.

        The flow time for each agent is defined as the difference between the final
        and the initial time (ignoring proxy nodes).

        Args:
            solution (dict): Mapping from agent_id to a list of (node, time) tuples.

        Returns:
            int: The total flow time cost.
        """
        total = 0
        for path in solution.values():
            filtered = [(n, t) for (n, t) in path if not is_proxy_node(n)]
            if filtered:
                total += filtered[-1][1] - filtered[0][1]
        return total

    def _reconstruct_path(self, final_state):
        """Reconstructs a path from a final state tuple.

        Args:
            final_state (tuple): A tuple (node, time, parent) from the low-level search.

        Returns:
            list: A list of (node, time) tuples representing the path.
        """
        path = []
        curr = final_state
        while curr:
            node, t, parent = curr
            path.append((node, int(t)))
            curr = parent
        return list(reversed(path))

    def _constraints_to_key(self, constraints):
        """Converts constraints to a hashable key.

        Args:
            constraints (Constraints): The constraint set.

        Returns:
            tuple: A tuple of frozensets representing the constraints.
        """
        vertex_key = frozenset(
            (vc.time, vc.location) for vc in constraints.vertex_constraints
        )
        edge_key = frozenset(
            (ec.time, ec.loc1, ec.loc2) for ec in constraints.edge_constraints
        )
        return (vertex_key, edge_key)
