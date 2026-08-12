import os
import xml.etree.ElementTree as ET

import jax
import mujoco
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jnp

from jaxgcrl.envs.multi_path_layout import (
    MULTI_PATH_MAZE,
    MULTI_PATH_TASKS,
    SUBGOAL_A,
    SUBGOAL_B,
    cell_to_xy,
    find_marker_cells,
)

# This is based on original Ant environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/ant.py
# Maze creation dapted from: https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/locomotion/maze_env.py

RESET = R = "r"
GOAL = G = "g"


U_MAZE = [
    [1, 1, 1, 1, 1],
    [1, R, G, G, 1],
    [1, 1, 1, G, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]

U_MAZE_EVAL = [
    [1, 1, 1, 1, 1],
    [1, R, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, G, G, G, 1],
    [1, 1, 1, 1, 1],
]


BIG_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, G, 1, 1, G, G, 1],
    [1, G, G, 1, G, G, G, 1],
    [1, 1, G, G, G, 1, 1, 1],
    [1, G, G, 1, G, G, G, 1],
    [1, G, 1, G, G, 1, G, 1],
    [1, G, G, G, 1, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

BIG_MAZE_EVAL = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, 0, 1, 1, G, G, 1],
    [1, 0, 0, 1, 0, G, G, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, G, 0, 1, G, 1],
    [1, 0, G, G, 1, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

HARDEST_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, R, G, G, G, 1, G, G, G, G, G, 1],
    [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
    [1, G, G, G, G, G, G, 1, G, G, G, 1],
    [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
    [1, G, G, 1, G, 1, G, G, G, G, G, 1],
    [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
    [1, G, G, 1, G, G, G, 1, G, G, G, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


MAZE_HEIGHT = 0.5

MAZE_LAYOUTS = {
    "u_maze": U_MAZE,
    "u_maze_eval": U_MAZE_EVAL,
    "big_maze": BIG_MAZE,
    "big_maze_eval": BIG_MAZE_EVAL,
    "hardest_maze": HARDEST_MAZE,
    "multi_path": MULTI_PATH_MAZE,
}


def find_starts(structure, size_scaling):
    starts = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == RESET:
                starts.append([i * size_scaling, j * size_scaling])

    return jnp.array(starts)


def find_goals(structure, size_scaling):
    goals = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == GOAL:
                goals.append([i * size_scaling, j * size_scaling])

    return jnp.array(goals)


def find_subgoals(structure, size_scaling):
    """Return xy positions for path-A then path-B subgoals (may be empty)."""
    subgoals = []
    for marker in (SUBGOAL_A, SUBGOAL_B):
        for i, j in find_marker_cells(structure, marker):
            subgoals.append([i * size_scaling, j * size_scaling])
    if not subgoals:
        return jnp.zeros((0, 2))
    return jnp.array(subgoals)


# Create a xml with maze and a list of possible goal positions
def make_maze(maze_layout_name, maze_size_scaling):
    if maze_layout_name not in MAZE_LAYOUTS:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")
    maze_layout = MAZE_LAYOUTS[maze_layout_name]

    xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "simple_maze.xml")

    possible_starts = find_starts(maze_layout, maze_size_scaling)
    possible_goals = find_goals(maze_layout, maze_size_scaling)
    possible_subgoals = find_subgoals(maze_layout, maze_size_scaling)

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            struct = maze_layout[i][j]
            if struct == 1:
                ET.SubElement(
                    worldbody,
                    "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f"
                    % (
                        i * maze_size_scaling,
                        j * maze_size_scaling,
                        MAZE_HEIGHT / 2 * maze_size_scaling,
                    ),
                    size="%f %f %f"
                    % (
                        0.5 * maze_size_scaling,
                        0.5 * maze_size_scaling,
                        MAZE_HEIGHT / 2 * maze_size_scaling,
                    ),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    # Visual-only markers for path subgoals (do not collide / affect dynamics).
    # Sized to nearly fill the 1-cell corridor so an agent cannot slip past
    # without intersecting the subgoal volume (visit checks can use the same radius).
    subgoal_radius = 0.0
    subgoal_visuals = (
        (SUBGOAL_A, "subgoal_a", "1.0 0.5 0.05 1.0"),
        (SUBGOAL_B, "subgoal_b", "0.58 0.40 0.74 1.0"),
    )
    has_subgoals = any(find_marker_cells(maze_layout, marker) for marker, _, _ in subgoal_visuals)
    if has_subgoals:
        subgoal_radius = 0.45 * maze_size_scaling
    for marker, name, rgba in subgoal_visuals:
        for i, j in find_marker_cells(maze_layout, marker):
            ET.SubElement(
                worldbody,
                "geom",
                name=name,
                pos="%f %f %f"
                % (
                    i * maze_size_scaling,
                    j * maze_size_scaling,
                    subgoal_radius,
                ),
                size="%f" % subgoal_radius,
                type="sphere",
                contype="0",
                conaffinity="0",
                rgba=rgba,
            )

    tree = tree.getroot()
    xml_string = ET.tostring(tree)

    return xml_string, possible_starts, possible_goals, possible_subgoals, subgoal_radius


class SimpleMaze(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        backend="generalized",
        maze_layout_name="u_maze",
        maze_size_scaling=4.0,
        task_name=None,
        **kwargs,
    ):
        xml_string, possible_starts, possible_goals, possible_subgoals, subgoal_radius = make_maze(
            maze_layout_name, maze_size_scaling
        )

        sys = mjcf.loads(xml_string)
        self.maze_layout_name = maze_layout_name
        self.maze_size_scaling = maze_size_scaling
        self.maze_layout = MAZE_LAYOUTS[maze_layout_name]
        self.possible_starts = possible_starts
        self.possible_goals = possible_goals
        self.possible_subgoals = possible_subgoals
        self.subgoal_radius = subgoal_radius
        self.subgoal_reach_thresh = float(subgoal_radius)
        self.task_name = task_name
        self._task = None
        if task_name is not None:
            if maze_layout_name != "multi_path":
                raise ValueError("task_name is only supported for multi_path maze layouts")
            if task_name not in MULTI_PATH_TASKS:
                raise ValueError(f"Unknown multi_path task: {task_name}")
            self._task = MULTI_PATH_TASKS[task_name]

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        if backend == "positional":
            # TODO: does the same actuator strength work as in spring
            sys = sys.replace(actuator=sys.actuator.replace(gear=200 * jnp.ones_like(sys.actuator.gear)))

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        self.state_dim = 4
        self.goal_indices = jnp.array([0, 1])
        self.goal_reach_thresh = 0.5

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng1, (self.sys.qd_size(),))

        # set the start and target q, qd
        start = self._random_start(rng2)
        q = q.at[:2].set(start)

        target = self._random_target(rng3)
        q = q.at[-2:].set(target)

        qd = qd.at[-2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        return state

    # Todo rename seed to traj_id
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = 0.0

        obs = self._get_obs(pipeline_state)
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        dist = jnp.linalg.norm(obs[:2] - obs[-2:])
        success = jnp.array(dist < self.goal_reach_thresh, dtype=float)
        success_easy = jnp.array(dist < 2.0, dtype=float)
        reward = -dist + healthy_reward - ctrl_cost - contact_cost
        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy,
        )
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]

        target_pos = pipeline_state.x.pos[-1][:2]

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        return jnp.concatenate([qpos] + [qvel] + [target_pos])

    def _random_target(self, rng: jax.Array) -> jax.Array:
        """Returns a random target location chosen from possibilities specified in the maze layout."""
        if self._task is not None:
            return jnp.array(cell_to_xy(self._task["goal"], self.maze_size_scaling))
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return jnp.array(self.possible_goals[idx])[0]

    def _random_start(self, rng: jax.Array) -> jax.Array:
        if self._task is not None:
            return jnp.array(cell_to_xy(self._task["start"], self.maze_size_scaling))
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_starts))
        return jnp.array(self.possible_starts[idx])[0]
