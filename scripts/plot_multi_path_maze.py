#!/usr/bin/env python3
"""Render top-down PNG views of the circular multi-path maze for each task."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, FancyBboxPatch, Patch


def _load_layout_module():
    """Load layout constants without importing jaxgcrl (avoids brax/flax deps)."""
    layout_path = Path(__file__).resolve().parents[1] / "jaxgcrl" / "envs" / "multi_path_layout.py"
    spec = importlib.util.spec_from_file_location("multi_path_layout", layout_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_layout = _load_layout_module()
MULTI_PATH_MAZE = _layout.MULTI_PATH_MAZE
MULTI_PATH_TASKS = _layout.MULTI_PATH_TASKS
SUBGOAL_A = _layout.SUBGOAL_A
SUBGOAL_B = _layout.SUBGOAL_B
is_wall = _layout.is_wall


def _grid_array(maze):
    rows, cols = len(maze), len(maze[0])
    grid = np.zeros((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            grid[i, j] = 1.0 if is_wall(maze[i][j]) else 0.0
    return grid


def plot_task(maze, task_name: str, start, goal, out_path: Path):
    grid = _grid_array(maze)
    rows, cols = grid.shape

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    cmap = ListedColormap(["#f4f1e8", "#5c4033"])
    ax.imshow(grid, cmap=cmap, origin="upper", vmin=0, vmax=1)

    # Light ring highlight on open cells.
    for i in range(rows):
        for j in range(cols):
            if not is_wall(maze[i][j]):
                ax.add_patch(
                    FancyBboxPatch(
                        (j - 0.48, i - 0.48),
                        0.96,
                        0.96,
                        boxstyle="round,pad=0.02,rounding_size=0.15",
                        linewidth=0.0,
                        facecolor="#d9e7f5",
                        alpha=0.35,
                    )
                )

    sub_a = next(((i, j) for i, row in enumerate(maze) for j, v in enumerate(row) if v == SUBGOAL_A), None)
    sub_b = next(((i, j) for i, row in enumerate(maze) for j, v in enumerate(row) if v == SUBGOAL_B), None)

    def mark(cell, color, label, marker="o", size=220):
        i, j = cell
        ax.scatter([j], [i], s=size, c=color, marker=marker, edgecolors="black", linewidths=1.2, zorder=5)
        ax.text(j, i - 0.62, label, ha="center", va="top", fontsize=10, fontweight="bold")

    mark(start, "#2ca02c", "start", marker="s")
    mark(goal, "#d62728", "goal", marker="*")
    if sub_a is not None:
        mark(sub_a, "#ff7f0e", "subgoal A\n(west path)", marker="D", size=180)
    if sub_b is not None:
        mark(sub_b, "#9467bd", "subgoal B\n(east path)", marker="D", size=180)

    # Sketch the two canonical south→north routes when relevant.
    west = [(7, 4), (7, 1), (4, 1), (1, 1), (1, 4)]
    east = [(7, 4), (7, 7), (4, 7), (1, 7), (1, 4)]
    if task_name == "south_to_north":
        for path, color in ((west, "#ff7f0e"), (east, "#9467bd")):
            ys, xs = zip(*path)
            ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.75, zorder=3)

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_title(f"multi_path · {task_name}", fontsize=13)
    ax.set_aspect("equal")

    legend = [
        Patch(facecolor="#5c4033", edgecolor="black", label="wall"),
        Patch(facecolor="#f4f1e8", edgecolor="black", label="corridor"),
        Circle((0, 0), radius=0.1, facecolor="#2ca02c", edgecolor="black", label="start"),
        Circle((0, 0), radius=0.1, facecolor="#d62728", edgecolor="black", label="goal"),
        Circle((0, 0), radius=0.1, facecolor="#ff7f0e", edgecolor="black", label="subgoal A"),
        Circle((0, 0), radius=0.1, facecolor="#9467bd", edgecolor="black", label="subgoal B"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("imgs/multi_path"),
        help="Directory for PNG outputs",
    )
    args = parser.parse_args()

    overview = args.out_dir / "multi_path_layout.png"
    # Overview uses the default south→north markers from the map itself.
    plot_task(
        MULTI_PATH_MAZE,
        "layout",
        start=(7, 4),
        goal=(1, 4),
        out_path=overview,
    )

    for task_name, task in MULTI_PATH_TASKS.items():
        plot_task(
            MULTI_PATH_MAZE,
            task_name,
            start=task["start"],
            goal=task["goal"],
            out_path=args.out_dir / f"multi_path_{task_name}.png",
        )


if __name__ == "__main__":
    main()
