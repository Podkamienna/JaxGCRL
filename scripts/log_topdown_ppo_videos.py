"""Render finished PPO checkpoints as OGBench-style top-down videos and log to W&B."""

from __future__ import annotations

from pathlib import Path

import jax
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

import wandb
from jaxgcrl.envs.simple_maze import SimpleMaze
from jaxgcrl.utils.env import _topdown_video

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("mqq3ln0y", "run_mp-ppo-either-s0_s_0", "either", 0),
    ("be3sowfy", "run_mp-ppo-either-s1_s_1", "either", 1),
    ("hptdxnla", "run_mp-ppo-either-s2_s_2", "either", 2),
    ("452iie7b", "run_mp-ppo-only_a-s0_s_0", "only_a", 0),
    ("fu1fyjrv", "run_mp-ppo-only_a-s1_s_1", "only_a", 1),
    ("yra3csnu", "run_mp-ppo-only_a-s2_s_2", "only_a", 2),
]


def _rollout(env, policy, seed: int = 1, length: int = 1000):
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_policy = jax.jit(policy)
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    state = jit_reset(subkey)
    states = []
    for _ in range(length):
        states.append(state.pipeline_state)
        key, subkey = jax.random.split(key)
        action, _ = jit_policy(state.obs[None], subkey)
        state = jit_step(state, action[0])
    return states


def main() -> None:
    for run_id, folder, mode, seed in RUNS:
        ckpt = ROOT / "runs" / folder / "ckpt" / "final"
        env = SimpleMaze(
            backend="spring",
            maze_layout_name="multi_path",
            task_name="south_to_north",
            subgoal_reward_mode=mode,
            goal_bonus=10.0,
            terminate_on_success=True,
        )
        ppo_net = ppo_networks.make_ppo_networks(
            env.observation_size,
            env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
        )
        make_policy = ppo_networks.make_inference_fn(ppo_net)
        params = model.load_params(str(ckpt))
        policy = make_policy(params, deterministic=True)
        states = _rollout(env, policy, seed=1)
        out = ROOT / "runs" / folder / f"{folder}_topdown.mp4"
        video = _topdown_video(env, states, str(out))
        run = wandb.init(project="jaxgcrl-multipath", id=run_id, resume="must")
        run.log({"render": video})
        print(f"logged topdown video {out} -> {run_id}")
        wandb.finish()


if __name__ == "__main__":
    main()
