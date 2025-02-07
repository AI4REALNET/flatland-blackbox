import time

import matplotlib.pyplot as plt
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.graphs.graph_utils import plotGraphEnv
from flatland.utils import jupyter_utils
from flatland.utils.rendertools import RenderTool


def initialize_environment(seed=42, width=30, height=30, num_agents=2):
    env = RailEnv(
        width=width,
        height=height,
        rail_generator=sparse_rail_generator(
            max_num_cities=3,
            grid_mode=False,
            max_rails_between_cities=4,
            max_rail_pairs_in_city=2,
            seed=seed,
        ),
        line_generator=sparse_line_generator(seed=seed),
        obs_builder_object=DummyObservationBuilder(),
        number_of_agents=num_agents,
    )
    env._max_episode_steps = int(15 * (env.width + env.height))

    renderer = RenderTool(env, screen_height=1200, screen_width=1600, show_debug=True)
    return env, renderer


def execute_simulation(env, renderer, action_sequences):
    max_steps = max(len(seq) for seq in action_sequences.values())

    for step in range(max_steps):
        actions = {}
        for agent_id, seq in action_sequences.items():
            if step < len(seq):
                actions[agent_id] = seq[step]
            else:
                actions[agent_id] = 0  # default to 'do nothing' action

        obs, rewards, dones, info = env.step(actions)
        renderer.render_env(show=True, show_observations=False, show_predictions=False)

        if all(dones.values()):
            print(f"All agents completed their tasks in {step + 1} steps.")
            break

        time.sleep(0.5)

    renderer.close_window()


def get_agents_start_end(graph, agents):
    list_starts = []
    list_ends = []

    for agent in agents:
        start_grid = agent.initial_position
        start_dir = agent.initial_direction
        official_start_node = (*start_grid, start_dir)
        official_start_node = tuple(int(element) for element in official_start_node)
        list_starts.append(official_start_node)

        # Potential end nodes are neighbors of the agent's target cell
        end_dirs = [
            n
            for n in graph.neighbors(agent.target)
            if graph.nodes[n].get("type") == "rail"
        ]
        list_ends.append(end_dirs)
        # print(f"Possible end nodes for agent {agent.handle}: {end_dirs}")
        # print(f"Departure-arrival timeframe: [{agent.earliest_departure}-{agent.latest_arrival}]")
        # print("Speed counter value:", agent.speed_counter.max_count)

    return list_starts, list_ends


def plot_agent_subgraphs(env, G_paths_subgraphs, agents):
    """
    Takes the environment, a list of subgraphs (one per agent),
    and the agents list, then does the jupyter canvas logic
    and calls plotGraphEnv for each subgraph.
    """
    env_canvas = jupyter_utils.EnvCanvas(env, behaviour=None)
    env_canvas.show()
    aImg = env_canvas.oRT.get_image()

    for i, Gpath in enumerate(G_paths_subgraphs):
        plotGraphEnv(
            Gpath,
            env,
            aImg,
            figsize=(8, 8),
            node_size=10,
            space=0.1,
            node_colors={"rail": "blue", "grid": "red"},
            edge_colors={"hold": "gray", "dir": "green"},
            show_nodes=("rail", "grid"),
            show_edges=("dir"),
            show_labels=(),
            show_edge_weights=True,
            alpha_img=0.7,
        )
        plt.title(f"Agent {agents[i].handle} path")
        plt.show()
