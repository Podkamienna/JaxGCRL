"""Pure-Python checks for the circular multi-path maze (no brax/jax required)."""

import importlib.util
from collections import deque
from pathlib import Path


def _load_layout_module():
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
find_marker_cells = _layout.find_marker_cells
is_open = _layout.is_open
is_wall = _layout.is_wall


def _neighbors(maze, cell):
    i, j = cell
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni, nj = i + di, j + dj
        if 0 <= ni < len(maze) and 0 <= nj < len(maze[0]) and is_open(maze[ni][nj]):
            yield (ni, nj)


def _shortest_path(maze, start, goal):
    queue = deque([(start, [start])])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for nxt in _neighbors(maze, node):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
    return None


def _blocked(maze, blocked_cells):
    blocked = set(blocked_cells)
    return [
        [1 if (i, j) in blocked or is_wall(maze[i][j]) else maze[i][j] for j in range(len(maze[0]))]
        for i in range(len(maze))
    ]


def test_multi_path_has_exactly_two_subgoals():
    assert len(find_marker_cells(MULTI_PATH_MAZE, SUBGOAL_A)) == 1
    assert len(find_marker_cells(MULTI_PATH_MAZE, SUBGOAL_B)) == 1


def test_path_constraint_modes_match_experiment_intent():
    """either = A∨B; only_a / only_b require the named path (pure-Python mirror)."""

    def path_ok(mode, a, b):
        if mode == "either":
            return max(a, b)
        if mode == "only_a":
            return a
        if mode == "only_b":
            return b
        raise ValueError(mode)

    assert path_ok("either", 0, 0) == 0
    assert path_ok("either", 1, 0) == 1
    assert path_ok("either", 0, 1) == 1
    assert path_ok("only_a", 1, 0) == 1
    assert path_ok("only_a", 0, 1) == 0
    assert path_ok("only_b", 0, 1) == 1
    assert path_ok("only_b", 1, 0) == 0


def test_south_to_north_has_two_disjoint_routes_via_subgoals():
    start = MULTI_PATH_TASKS["south_to_north"]["start"]
    goal = MULTI_PATH_TASKS["south_to_north"]["goal"]
    sub_a = find_marker_cells(MULTI_PATH_MAZE, SUBGOAL_A)[0]
    sub_b = find_marker_cells(MULTI_PATH_MAZE, SUBGOAL_B)[0]

    via_a = _shortest_path(MULTI_PATH_MAZE, start, goal)
    assert via_a is not None

    # With west subgoal cell blocked, east path still reaches the goal and uses B.
    no_west = _blocked(MULTI_PATH_MAZE, [sub_a])
    path_east = _shortest_path(no_west, start, goal)
    assert path_east is not None
    assert sub_b in path_east

    # With east subgoal cell blocked, west path still reaches the goal and uses A.
    no_east = _blocked(MULTI_PATH_MAZE, [sub_b])
    path_west = _shortest_path(no_east, start, goal)
    assert path_west is not None
    assert sub_a in path_west

    # Blocking both subgoal cells disconnects start from goal.
    no_both = _blocked(MULTI_PATH_MAZE, [sub_a, sub_b])
    assert _shortest_path(no_both, start, goal) is None


def test_all_named_tasks_are_solvable():
    for name, task in MULTI_PATH_TASKS.items():
        path = _shortest_path(MULTI_PATH_MAZE, task["start"], task["goal"])
        assert path is not None, name
        assert path[0] == task["start"]
        assert path[-1] == task["goal"]
