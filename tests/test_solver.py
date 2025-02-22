import pytest
from matplotlib import pyplot as plt
from test_utils import MockAgent

from flatland_blackbox.solvers.cbs import CBSSolver
from flatland_blackbox.solvers.pp import PrioritizedPlanningSolver
from flatland_blackbox.utils.graph_utils import get_rail_subgraph


@pytest.mark.parametrize("solver_cls", [PrioritizedPlanningSolver, CBSSolver])
def test_single_agent_tiny(tiny_test_graph, solver_cls):
    """
    Single agent traveling from (0,0) to (0,1) in a tiny graph.
    """
    planning_env = get_rail_subgraph(tiny_test_graph)
    solver = solver_cls(planning_env)

    a0 = MockAgent(0, (0, 0), 0, (0, 1))
    agents = [a0]

    solution_dict = solver.solve(agents)

    path = solution_dict[a0.handle]
    assert len(path) > 0, "Path should not be empty"
    last_node, last_time = path[-1]
    assert (last_node[0], last_node[1]) == (0, 1), "Wrong final cell"


@pytest.mark.parametrize("solver_cls", [PrioritizedPlanningSolver, CBSSolver])
def test_two_agents_cross(two_by_two_cross_graph, solver_cls):
    """
    A 2x2 cross scenario ensuring one agent must wait or otherwise avoid collision.
    """
    planning_env = get_rail_subgraph(two_by_two_cross_graph)
    solver = solver_cls(planning_env)

    a0 = MockAgent(0, (0, 0), 0, (1, 1))
    a1 = MockAgent(1, (1, 0), 0, (0, 1))
    agents = [a0, a1]

    solution_dict = solver.solve(agents)

    pathA = solution_dict[0]
    pathB = solution_dict[1]

    assert len(pathA) > 0, "Agent A's path shouldn't be empty"
    assert len(pathB) > 0, "Agent B's path shouldn't be empty"

    endA, timeA = pathA[-1]
    assert (endA[0], endA[1]) == (1, 1), "Agent A not at (1,1)"

    endB, timeB = pathB[-1]
    assert (endB[0], endB[1]) == (0, 1), "Agent B not at (0,1)"


@pytest.mark.parametrize("solver_cls", [PrioritizedPlanningSolver, CBSSolver])
def test_passing_node_scenario(passing_node_graph, solver_cls, stay_at_goal=False):
    """
    Scenario:

    1) Agent 0: start=(2,1,0)=s1 -> goal=(2,4,0)=t1
    2) Agent 1: start=(2,5,0)=s2 -> goal=(2,2,0)=t2

    There's a top node at (1,3,0) that can let them pass.
    If they don't vanish at goal, PP should fail, CBS succeed.
    """

    G_rail = get_rail_subgraph(passing_node_graph)
    solver = solver_cls(G_rail)

    a0 = MockAgent(0, (2, 1), 0, (2, 4))
    a1 = MockAgent(1, (2, 5), 0, (2, 2))
    agents = [a0, a1]

    # Attempt to plan
    try:
        solution = solver.solve(
            agents
        )  # or .schedule_agents_in_order(agents) if PP uses that
    except ValueError as e:
        # If no solution found, we can check if it's the PP solver that fails
        if (solver_cls is PrioritizedPlanningSolver) and stay_at_goal:
            print("PP failed to find a solution (expected).")
            return  # test passes if PP fails
        elif solver_cls is CBSSolver:
            # If CBS fails, that's unexpected
            pytest.fail("CBS unexpectedly failed on passing node scenario.")

    # For PP, we expect no solution if trains remain at their goal position
    if solver_cls is PrioritizedPlanningSolver and stay_at_goal:
        pytest.fail(
            "PP found a solution, but was expected to fail if goals remain blocked."
        )

    pathA = solution[0]
    pathB = solution[1]
    assert len(pathA) > 0, "Agent 0 path is empty?"
    assert len(pathB) > 0, "Agent 1 path is empty?"

    # Check final node is correct
    assert (pathA[-1][0].row, pathA[-1][0].col) == (2, 4), "Agent0 didn't end at (2,4)"
    assert (pathB[-1][0].row, pathB[-1][0].col) == (2, 2), "Agent1 didn't end at (2,2)"

    print("CBS succeeded with a detour. Great!")


@pytest.mark.parametrize("solver_cls", [PrioritizedPlanningSolver, CBSSolver])
def test_two_trains_suboptimal_scenario(two_trains_suboptimal_graph, solver_cls):
    a0 = MockAgent(0, (4, 2), 0, (0, 1))
    a1 = MockAgent(1, (1, 0), 0, (4, 1))
    agents = [a0, a1]

    rail_subgraph = get_rail_subgraph(two_trains_suboptimal_graph)
    solver = solver_cls(rail_subgraph)

    try:
        solution = solver.solve(agents)
    except ValueError as e:
        pytest.fail(f"{solver_cls.__name__} raised ValueError: {e}")

    # Basic solution presence checks
    assert 0 in solution, "No path returned for agent 0"
    assert 1 in solution, "No path returned for agent 1"
    path0 = solution[0]
    path1 = solution[1]
    assert path0, "Agent0 path is empty"
    assert path1, "Agent1 path is empty"

    # Final node checks
    final0 = path0[-1][0]
    final1 = path1[-1][0]
    assert (final0.row, final0.col) == (0, 1), f"Agent0 didn't end at {(0,1)}"
    assert (final1.row, final1.col) == (4, 1), f"Agent1 didn't end at {(4,1)}"

    path0_coords = [(n.row, n.col) for n, _ in path0]
    path1_coords = [(n.row, n.col) for n, _ in path1]
    print(f"Agent0 path coords: {path0_coords}")
    print(f"Agent1 path coords: {path1_coords}")

    # Flow time & makespan
    flow_time = len(path0) + len(path1)
    makespan = max(len(path0), len(path1))

    # Separate checks for PP and CBS expected paths
    if solver_cls is PrioritizedPlanningSolver:
        expected_len0, expected_len1 = 6, 9
        expected_flow, expected_makespan = 15, 9
    else:
        expected_len0, expected_len1 = 8, 5
        expected_flow, expected_makespan = 13, 8

    assert (
        len(path0) == expected_len0
    ), f"Agent0 path length mismatch for {solver_cls.__name__}"
    assert (
        len(path1) == expected_len1
    ), f"Agent1 path length mismatch for {solver_cls.__name__}"

    assert flow_time == expected_flow, f"Flow time mismatch for {solver_cls.__name__}"
    assert makespan == expected_makespan, f"Makespan mismatch for {solver_cls.__name__}"

    print(f"Flow time: {flow_time}, Makespan: {makespan}")
