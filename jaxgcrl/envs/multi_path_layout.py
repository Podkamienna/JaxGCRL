"""Circular multi-path point-maze layout and named tasks.

Topology: a ring corridor around a solid center. From the south start to the
north goal there are exactly two routes (west vs east); one subgoal sits on
each route. Named tasks fix (start, goal) pairs on the same ring for
visualization and later reward experiments.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# Cell markers (match simple_maze / D4RL-style maze maps).
RESET = R = "r"
GOAL = G = "g"
SUBGOAL_A = A = "a"  # west-path subgoal
SUBGOAL_B = B = "b"  # east-path subgoal

Cell = int | str
Coord = Tuple[int, int]

# Ring / circular corridor. Walls=1, open=0, R/G/A/B as above.
# Indexing is (row, col) with row increasing southward (same as other JaxGCRL mazes).
MULTI_PATH_MAZE: List[List[Cell]] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, G, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, A, 1, 1, 1, 1, 1, B, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, R, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Fixed (start, goal) tasks on the ring. Subgoals stay at A/B (west/east midpoints).
MULTI_PATH_TASKS: Dict[str, Dict[str, Coord]] = {
    "south_to_north": {"start": (7, 4), "goal": (1, 4)},
    "south_to_northeast": {"start": (7, 4), "goal": (1, 7)},
    "southwest_to_north": {"start": (7, 1), "goal": (1, 4)},
    "southeast_to_northwest": {"start": (7, 7), "goal": (1, 1)},
    "south_to_northwest": {"start": (7, 4), "goal": (1, 1)},
}

# Tighter ring (7x7) for easier exploration while keeping two disjoint paths.
MULTI_PATH_SMALL_MAZE: List[List[Cell]] = [
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, G, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, A, 1, 1, 1, B, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, R, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
]

MULTI_PATH_SMALL_TASKS: Dict[str, Dict[str, Coord]] = {
    "south_to_north": {"start": (5, 3), "goal": (1, 3)},
    "south_to_northeast": {"start": (5, 3), "goal": (1, 5)},
    "southwest_to_north": {"start": (5, 1), "goal": (1, 3)},
    "southeast_to_northwest": {"start": (5, 5), "goal": (1, 1)},
    "south_to_northwest": {"start": (5, 3), "goal": (1, 1)},
}

MULTI_PATH_LAYOUT_NAMES = ("multi_path", "multi_path_small")

# Start ~ Unif(south corridor), goal ~ Unif(north corridor); not a fixed cell pair.
RANDOM_NS_TASK = "random_ns"


def tasks_for_layout(maze_layout_name: str) -> Dict[str, Dict[str, Coord]]:
    if maze_layout_name == "multi_path":
        return MULTI_PATH_TASKS
    if maze_layout_name == "multi_path_small":
        return MULTI_PATH_SMALL_TASKS
    raise ValueError(f"Not a multi_path layout: {maze_layout_name}")


def find_marker_cells(structure: Sequence[Sequence[Cell]], marker: Cell) -> List[Coord]:
    cells: List[Coord] = []
    for i, row in enumerate(structure):
        for j, value in enumerate(row):
            if value == marker:
                cells.append((i, j))
    return cells


def cell_to_xy(cell: Coord, size_scaling: float) -> Tuple[float, float]:
    """Map maze (row, col) to env xy used by SimpleMaze (row→x, col→y)."""
    i, j = cell
    return (i * size_scaling, j * size_scaling)


def horizontal_side_rows(maze: Sequence[Sequence[Cell]]) -> Tuple[int, int]:
    """North and south rows of the ring (wide open corridors, no A/B subgoals)."""
    nrows = len(maze)
    ncols = len(maze[0])
    h_rows = [
        i
        for i in range(1, nrows - 1)
        if sum(1 for j in range(ncols) if is_open(maze[i][j])) >= 3
    ]
    if len(h_rows) < 2:
        raise ValueError("maze has no north/south open corridors")
    return h_rows[0], h_rows[-1]


def north_south_open_cells(maze: Sequence[Sequence[Cell]]) -> Tuple[List[Coord], List[Coord]]:
    """Open cells on the north then south sides (the sides that do not host A/B)."""
    north_row, south_row = horizontal_side_rows(maze)
    north = [(north_row, j) for j in range(len(maze[0])) if is_open(maze[north_row][j])]
    south = [(south_row, j) for j in range(len(maze[0])) if is_open(maze[south_row][j])]
    return north, south


def side_corridor_xy_range(
    cells: Sequence[Coord], size_scaling: float
) -> Tuple[float, float, float]:
    """(x, y_min, y_max) spanning cell centers along a horizontal corridor."""
    rows = {i for i, _ in cells}
    cols = [j for _, j in cells]
    if len(rows) != 1 or not cols:
        raise ValueError("side corridor cells must share one row")
    (row,) = rows
    x = row * size_scaling
    return x, min(cols) * size_scaling, max(cols) * size_scaling


def is_wall(value: Cell) -> bool:
    return value == 1


def is_open(value: Cell) -> bool:
    return not is_wall(value)
