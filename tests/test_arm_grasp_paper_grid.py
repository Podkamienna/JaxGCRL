"""Lock the 9-cell arm_grasp paper-hp grid (no jax)."""

import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "arm_grasp_paper_hp.py"
    spec = importlib.util.spec_from_file_location("arm_grasp_paper_hp", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_mod = _load()
ARM_GRASP_PAPER_HP_TAG = _mod.ARM_GRASP_PAPER_HP_TAG
PAPER_CRL = _mod.PAPER_CRL
cells = _mod.cells
cli_args = _mod.cli_args


def test_nine_jobs_three_arms_three_seeds():
    c = cells()
    assert len(c) == 9
    arms = sorted({x["arm"] for x in c})
    seeds = sorted({x["seed"] for x in c})
    assert arms == ["baseline", "neuralA", "rotA"]
    assert seeds == [0, 1, 2]


def test_common_wandb_tag_on_every_cell():
    for cell in cells():
        assert cell["wandb_tags"] == ARM_GRASP_PAPER_HP_TAG
        args = cli_args(cell)
        assert ARM_GRASP_PAPER_HP_TAG in args
        assert "--wandb-tags" in args
        assert args[args.index("--wandb-tags") + 1] == ARM_GRASP_PAPER_HP_TAG


def test_paper_hparams_shared():
    for cell in cells():
        assert cell["env"] == "arm_grasp"
        assert cell["total_env_steps"] == PAPER_CRL["total_env_steps"]
        assert cell["num_envs"] == 1024
        assert cell["policy_lr"] == 6e-4
        assert cell["contrastive_loss_fn"] == "sym_infonce"
        assert cell["energy_fn"] == "l2"
        assert cell["episode_length"] == 100
        n = cell["num_envs"] * (cell["episode_length"] - 1)
        assert n % cell["batch_size"] == 0


def test_arms_are_exclusive_A_modes():
    by_arm = {c["arm"]: c for c in cells() if c["seed"] == 0}
    b, r, n = by_arm["baseline"], by_arm["rotA"], by_arm["neuralA"]
    assert b["use_A"] is False
    assert r["use_A"] is True and r["neural_A"] is False and r["rotational_A"] is True
    assert n["use_A"] is True and n["neural_A"] is True and n["rotational_A"] is False
    assert n["neural_A_depth"] == 2
    rot_args = cli_args(r)
    neu_args = cli_args(n)
    assert "--no-neural-A" in rot_args and "--use-A" in rot_args
    assert "--neural-A" in neu_args and "--no-rotational-A" in neu_args
    assert "--use-A" not in cli_args(b)


def test_exp_names_unique():
    names = [c["exp_name"] for c in cells()]
    assert len(names) == len(set(names))
