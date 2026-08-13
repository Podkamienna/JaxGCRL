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
MULTI_PATH_SMALL_MAZE = _layout.MULTI_PATH_SMALL_MAZE
MULTI_PATH_TASKS = _layout.MULTI_PATH_TASKS
MULTI_PATH_SMALL_TASKS = _layout.MULTI_PATH_SMALL_TASKS
SUBGOAL_A = _layout.SUBGOAL_A
SUBGOAL_B = _layout.SUBGOAL_B
find_marker_cells = _layout.find_marker_cells
is_open = _layout.is_open
is_wall = _layout.is_wall
north_south_open_cells = _layout.north_south_open_cells
side_corridor_xy_range = _layout.side_corridor_xy_range


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


def test_simple_multi_path_is_dispatched_without_maze_substring():
    """create_env used to match only names containing 'maze', which misses this env."""
    assert "maze" not in "simple_multi_path"
    assert "maze" not in "simple_multi_path_small"
    env_py = Path(__file__).resolve().parents[1] / "jaxgcrl" / "utils" / "env.py"
    text = env_py.read_text()
    assert 'env_name.startswith("simple_multi_path")' in text
    assert '"simple_multi_path_small"' in text


def test_small_south_to_north_has_two_disjoint_routes():
    start = MULTI_PATH_SMALL_TASKS["south_to_north"]["start"]
    goal = MULTI_PATH_SMALL_TASKS["south_to_north"]["goal"]
    sub_a = find_marker_cells(MULTI_PATH_SMALL_MAZE, SUBGOAL_A)[0]
    sub_b = find_marker_cells(MULTI_PATH_SMALL_MAZE, SUBGOAL_B)[0]
    assert _shortest_path(MULTI_PATH_SMALL_MAZE, start, goal) is not None
    path_east = _shortest_path(_blocked(MULTI_PATH_SMALL_MAZE, [sub_a]), start, goal)
    path_west = _shortest_path(_blocked(MULTI_PATH_SMALL_MAZE, [sub_b]), start, goal)
    assert path_east is not None and sub_b in path_east
    assert path_west is not None and sub_a in path_west
    assert _shortest_path(_blocked(MULTI_PATH_SMALL_MAZE, [sub_a, sub_b]), start, goal) is None


def test_first_visit_bonus_follows_path_mode():
    def bonus(mode, first_a, first_b, amount=1.0):
        if mode == "either":
            return (first_a + first_b) * amount
        if mode == "only_a":
            return first_a * amount
        if mode == "only_b":
            return first_b * amount
        raise ValueError(mode)

    assert bonus("either", 1, 0) == 1.0
    assert bonus("either", 1, 1) == 2.0
    assert bonus("only_a", 1, 0) == 1.0
    assert bonus("only_a", 0, 1) == 0.0


def test_ns_sides_exclude_subgoals_and_stay_disconnected_if_both_blocked():
    for maze in (MULTI_PATH_MAZE, MULTI_PATH_SMALL_MAZE):
        north, south = north_south_open_cells(maze)
        sub_a = find_marker_cells(maze, SUBGOAL_A)[0]
        sub_b = find_marker_cells(maze, SUBGOAL_B)[0]
        assert north and south
        assert sub_a not in north and sub_a not in south
        assert sub_b not in north and sub_b not in south
        assert len({i for i, _ in north}) == 1
        assert len({i for i, _ in south}) == 1
        assert north[0][0] < south[0][0]
        no_both = _blocked(maze, [sub_a, sub_b])
        for start in south:
            for goal in north:
                assert _shortest_path(no_both, start, goal) is None
                assert _shortest_path(maze, start, goal) is not None


def test_side_corridor_xy_range_is_horizontal():
    north, south = north_south_open_cells(MULTI_PATH_SMALL_MAZE)
    sx, sy_lo, sy_hi = side_corridor_xy_range(south, 2.0)
    gx, gy_lo, gy_hi = side_corridor_xy_range(north, 2.0)
    assert sx > gx
    assert sy_lo < sy_hi
    assert gy_lo < gy_hi


def test_all_named_tasks_are_solvable():
    for name, task in MULTI_PATH_TASKS.items():
        path = _shortest_path(MULTI_PATH_MAZE, task["start"], task["goal"])
        assert path is not None, name
        assert path[0] == task["start"]
        assert path[-1] == task["goal"]
