import glob
import os

import matplotlib.pyplot as plt
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool

# from flatland.graphs.graph_utils import plotGraphEnv
from flatland_blackbox.utils.graph_utils import plotGraphEnv


def initialize_environment(
    seed=42, width=30, height=30, num_agents=2, max_num_cities=3
):
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
    env.reset(random_seed=seed)
    return env


def plot_agent_subgraphs(env, G_paths_subgraphs, save_fig_folder):
    """
    Plots each agent's subgraph over the environment background image.

    This function renders the environment using a RenderTool instance to obtain a background
    image. It then overlays each agent's subgraph on this image using plotGraphEnv.
    It clears out any previous PNG files in that folder and saves the new figures.

    Args:
        env: The Flatland environment.
        G_paths_subgraphs (dict): Dictionary mapping agent IDs to their subgraphs.
    """
    # Create a RenderTool instance and render the environment to get the background image.
    render_tool = RenderTool(env, show_debug=False)
    render_tool.render_env(
        show_rowcols=True, show_inactive_agents=False, show_observations=False
    )
    aImg = render_tool.get_image()

    # Remove any previous PNG files in the folder.
    png_files = glob.glob(os.path.join(save_fig_folder, "*.png"))
    for file in png_files:
        os.remove(file)

    # Plot each agent's subgraph.
    for agent_id, Gpath in G_paths_subgraphs.items():
        plt.figure(figsize=(8, 8))
        plotGraphEnv(
            Gpath,
            env,
            aImg,
            figsize=(8, 8),
            dpi=100,
            node_size=8,
            space=0.1,
            node_colors={"rail": "blue", "grid": "red"},
            edge_colors={"hold": "gray", "dir": "green"},
            show_nodes=("rail", "grid"),
            show_edges=("dir"),
            show_labels=(),
            show_edge_weights=True,
            alpha_img=0.7,
        )
        plt.title(f"Agent {agent_id} path")
        plt.savefig(f"{save_fig_folder}/path_agent_{agent_id}.png", dpi="figure")
        plt.close("all")


def print_agents_start(agents):
    for agent in agents:
        start_rc = tuple(map(int, agent.initial_position))
        end_rc = tuple(map(int, agent.target))
        print(
            f"Agent {agent.handle} start: {start_rc}"
            f" end: {end_rc} edt: {agent.earliest_departure}"
        )
