"""9-job arm_grasp grid: paper CRL hparams × {baseline, rotational A, neural A} × 3 seeds.

Common W&B tag (every run): ARM_GRASP_PAPER_HP_TAG
Does not import jaxgcrl.
"""

from __future__ import annotations

from typing import Any, Dict, List

# Unique to this grid only. Filter W&B with tags = this string.
ARM_GRASP_PAPER_HP_TAG = "arm-grasp-paper-hp-2026-08-19"
WANDB_PROJECT = "jaxgcrl"

# JaxGCRL paper Table 2 (algorithm). Arm grasp episode length follows the env (100),
# not Table 2's locomotion default of 1000.
PAPER_CRL: Dict[str, Any] = {
    "env": "arm_grasp",
    "total_env_steps": 50_000_000,
    "episode_length": 100,
    "num_envs": 1024,
    "num_eval_envs": 256,
    "num_evals": 200,
    "batch_size": 256,
    "discounting": 0.99,
    "action_repeat": 1,
    "unroll_length": 62,
    "min_replay_size": 1000,
    "max_replay_size": 10000,
    "train_step_multiplier": 1,
    "policy_lr": 6e-4,
    "critic_lr": 3e-4,
    "h_dim": 256,
    "n_hidden": 2,
    "repr_dim": 64,
    "contrastive_loss_fn": "sym_infonce",
    "energy_fn": "l2",
    "logsumexp_penalty_coeff": 0.1,
}

ARMS: List[Dict[str, Any]] = [
    {
        "arm": "baseline",
        "use_A": False,
        "neural_A": False,
        "rotational_A": False,
        "rotational_with_Q": False,
    },
    {
        # Same A path as the crl_use_A_test grid that looked better on arm_grasp.
        "arm": "rotA",
        "use_A": True,
        "neural_A": False,
        "rotational_A": True,
        "rotational_with_Q": True,
        "fixed_rope_A": True,
        "randomly_initialized_A": True,
    },
    {
        "arm": "neuralA",
        "use_A": True,
        "neural_A": True,
        "neural_A_depth": 2,
        "rotational_A": False,
        "rotational_with_Q": False,
    },
]

SEEDS = [0, 1, 2]


def cells() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in SEEDS:
            cell = {**PAPER_CRL, **arm, "seed": seed}
            cell["wandb_project_name"] = WANDB_PROJECT
            cell["wandb_tags"] = ARM_GRASP_PAPER_HP_TAG
            cell["wandb_group"] = f"{ARM_GRASP_PAPER_HP_TAG}-{arm['arm']}"
            cell["exp_name"] = f"crl_arm_grasp_{arm['arm']}_seed{seed}"
            out.append(cell)
    return out


def cli_args(cell: Dict[str, Any]) -> List[str]:
    """Flags for `python run.py crl ...`."""
    args = [
        "run.py",
        "crl",
        "--env",
        str(cell["env"]),
        "--seed",
        str(cell["seed"]),
        "--total-env-steps",
        str(cell["total_env_steps"]),
        "--episode-length",
        str(cell["episode_length"]),
        "--num-envs",
        str(cell["num_envs"]),
        "--num-eval-envs",
        str(cell["num_eval_envs"]),
        "--num-evals",
        str(cell["num_evals"]),
        "--batch-size",
        str(cell["batch_size"]),
        "--discounting",
        str(cell["discounting"]),
        "--action-repeat",
        str(cell["action_repeat"]),
        "--unroll-length",
        str(cell["unroll_length"]),
        "--min-replay-size",
        str(cell["min_replay_size"]),
        "--max-replay-size",
        str(cell["max_replay_size"]),
        "--train-step-multiplier",
        str(cell["train_step_multiplier"]),
        "--policy-lr",
        str(cell["policy_lr"]),
        "--critic-lr",
        str(cell["critic_lr"]),
        "--h-dim",
        str(cell["h_dim"]),
        "--n-hidden",
        str(cell["n_hidden"]),
        "--repr-dim",
        str(cell["repr_dim"]),
        "--contrastive-loss-fn",
        str(cell["contrastive_loss_fn"]),
        "--energy-fn",
        str(cell["energy_fn"]),
        "--logsumexp-penalty-coeff",
        str(cell["logsumexp_penalty_coeff"]),
        "--wandb-project-name",
        str(cell["wandb_project_name"]),
        "--wandb-group",
        str(cell["wandb_group"]),
        "--wandb-tags",
        str(cell["wandb_tags"]),
        "--exp-name",
        str(cell["exp_name"]),
        "--log-wandb",
    ]
    if cell["use_A"]:
        args.append("--use-A")
    if cell["neural_A"]:
        args.append("--neural-A")
    else:
        args.append("--no-neural-A")
    if cell["rotational_A"]:
        args.append("--rotational-A")
    else:
        args.append("--no-rotational-A")
    if cell.get("rotational_with_Q"):
        args.append("--rotational-with-Q")
    else:
        args.append("--no-rotational-with-Q")
    if cell.get("neural_A_depth") is not None:
        args.extend(["--neural-A-depth", str(cell["neural_A_depth"])])
    if cell.get("fixed_rope_A"):
        args.append("--fixed-rope-A")
    if cell.get("randomly_initialized_A") and cell["use_A"] and not cell["neural_A"]:
        args.append("--randomly-initialized-A")
    return args


if __name__ == "__main__":
    import json
    import sys

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else None
    all_cells = cells()
    if idx is None:
        print(len(all_cells))
    else:
        print(json.dumps(cli_args(all_cells[idx])))
