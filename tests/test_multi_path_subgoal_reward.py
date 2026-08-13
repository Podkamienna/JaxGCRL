"""Tests for multi-path subgoal-gated goal rewards."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("brax")

from jaxgcrl.envs.simple_maze import SimpleMaze, gated_path_ok, gated_subgoal_step_bonus  # noqa: E402


def test_gated_path_ok_modes():
    assert float(gated_path_ok("either", 0.0, 0.0)) == 0.0
    assert float(gated_path_ok("either", 1.0, 0.0)) == 1.0
    assert float(gated_path_ok("either", 0.0, 1.0)) == 1.0
    assert float(gated_path_ok("only_a", 1.0, 0.0)) == 1.0
    assert float(gated_path_ok("only_a", 0.0, 1.0)) == 0.0
    assert float(gated_path_ok("only_b", 0.0, 1.0)) == 1.0
    assert float(gated_path_ok("only_b", 1.0, 0.0)) == 0.0


def test_gated_subgoal_step_bonus():
    assert float(gated_subgoal_step_bonus("either", 1.0, 0.0, 1.0)) == 1.0
    assert float(gated_subgoal_step_bonus("either", 0.0, 1.0, 1.0)) == 1.0
    assert float(gated_subgoal_step_bonus("only_a", 0.0, 1.0, 1.0)) == 0.0
    assert float(gated_subgoal_step_bonus("only_a", 1.0, 0.0, 1.0)) == 1.0


def _teleport(env: SimpleMaze, state, xy):
    q = state.pipeline_state.q.at[:2].set(jnp.asarray(xy, dtype=jnp.float32))
    qd = state.pipeline_state.qd * 0.0
    ps = env.pipeline_init(q, qd)
    return state.replace(pipeline_state=ps, obs=env._get_obs(ps))


def test_small_maze_default_cell_size_is_half():
    from jaxgcrl.utils.env import create_env

    small = create_env("simple_multi_path_small", backend="spring")
    full = create_env("simple_multi_path", backend="spring")
    assert small.maze_size_scaling == 2.0
    assert full.maze_size_scaling == 4.0
    assert small.subgoal_reach_thresh == pytest.approx(0.9)


def _make_env(mode: str) -> SimpleMaze:
    return SimpleMaze(
        backend="spring",
        maze_layout_name="multi_path",
        task_name="south_to_north",
        subgoal_reward_mode=mode,
        goal_bonus=10.0,
        terminate_on_success=True,
        ctrl_cost_weight=0.0,
    )


def test_only_a_requires_west_subgoal_before_goal():
    env = _make_env("only_a")
    state = env.reset(jax.random.PRNGKey(0))
    goal = state.pipeline_state.q[-2:]
    sg_a = env.possible_subgoals[0]
    sg_b = env.possible_subgoals[1]

    # Reach goal without visiting A: ungated success, gated failure.
    state = _teleport(env, state, goal)
    state = env.step(state, jnp.zeros(env.action_size))
    assert float(state.metrics["success_ungated"]) == 1.0
    assert float(state.metrics["success"]) == 0.0
    assert float(state.metrics["reward_goal"]) == 0.0
    assert float(state.done) == 0.0

    # Visit B only, then goal: still gated failure for only_a.
    state = env.reset(jax.random.PRNGKey(1))
    goal = state.pipeline_state.q[-2:]
    state = _teleport(env, state, sg_b)
    state = env.step(state, jnp.zeros(env.action_size))
    assert float(state.info["visited_b"]) == 1.0
    assert float(state.info["visited_a"]) == 0.0
    state = _teleport(env, state, goal)
    state = env.step(state, jnp.zeros(env.action_size))
    assert float(state.metrics["success"]) == 0.0

    # Visit A then goal: gated success + bonus + terminate.
    state = env.reset(jax.random.PRNGKey(2))
    goal = state.pipeline_state.q[-2:]
    state = _teleport(env, state, sg_a)
    state = env.step(state, jnp.zeros(env.action_size))
    assert float(state.info["visited_a"]) == 1.0
    state = _teleport(env, state, goal)
    state = env.step(state, jnp.zeros(env.action_size))
    assert float(state.metrics["success"]) == 1.0
    assert float(state.metrics["reward_goal"]) == 10.0
    assert float(state.done) == 1.0


def test_either_accepts_a_or_b():
    env = _make_env("either")
    state = env.reset(jax.random.PRNGKey(3))
    goal = state.pipeline_state.q[-2:]
    sg_b = env.possible_subgoals[1]

    state = _teleport(env, state, sg_b)
    state = env.step(state, jnp.zeros(env.action_size))
    state = _teleport(env, state, goal)
    state = env.step(state, jnp.zeros(env.action_size))
    assert float(state.metrics["success"]) == 1.0
    assert float(state.metrics["reward_goal"]) == 10.0
