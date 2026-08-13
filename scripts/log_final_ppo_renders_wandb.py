"""Re-log the latest brax HTML render onto each finished PPO W&B run."""

from __future__ import annotations

from pathlib import Path

import wandb

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ("mqq3ln0y", "run_mp-ppo-either-s0_s_0", "mp-ppo-either-s0"),
    ("be3sowfy", "run_mp-ppo-either-s1_s_1", "mp-ppo-either-s1"),
    ("hptdxnla", "run_mp-ppo-either-s2_s_2", "mp-ppo-either-s2"),
    ("452iie7b", "run_mp-ppo-only_a-s0_s_0", "mp-ppo-only_a-s0"),
    ("fu1fyjrv", "run_mp-ppo-only_a-s1_s_1", "mp-ppo-only_a-s1"),
    ("yra3csnu", "run_mp-ppo-only_a-s2_s_2", "mp-ppo-only_a-s2"),
]


def latest_html(folder: str, exp_name: str) -> Path:
    htmls = sorted((ROOT / "runs" / folder).glob(f"{exp_name}_*.html"), key=lambda p: p.stat().st_mtime)
    if not htmls:
        raise FileNotFoundError(f"no html in {folder}")
    return htmls[-1]


def main() -> None:
    for run_id, folder, exp_name in RUNS:
        path = latest_html(folder, exp_name)
        run = wandb.init(
            project="jaxgcrl-multipath",
            id=run_id,
            resume="must",
        )
        run.log({"final_policy": wandb.Html(path.read_text())})
        print(f"logged {path.name} -> {run_id}")
        wandb.finish()


if __name__ == "__main__":
    main()
