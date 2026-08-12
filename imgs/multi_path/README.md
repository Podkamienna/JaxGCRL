# Multi-path point maze

Circular / ring corridor for `simple_multi_path`. From the south start to the
north goal there are two routes (west and east); **subgoal A** sits on the west
path and **subgoal B** on the east path.

Generate / refresh these PNGs:

```bash
python scripts/plot_multi_path_maze.py
```

| File | Task |
| --- | --- |
| `multi_path_layout.png` | Default layout markers |
| `multi_path_south_to_north.png` | Canonical two-path task |
| `multi_path_south_to_northeast.png` | South → northeast |
| `multi_path_southwest_to_north.png` | Southwest → north |
| `multi_path_southeast_to_northwest.png` | Southeast → northwest |
| `multi_path_south_to_northwest.png` | South → northwest |

## Brax-instantiated views

Top-down / 3D plots drawn from the real MuJoCo geoms after
`SimpleMaze(maze_layout_name="multi_path", task_name=...).reset()` live in
`brax_instantiated/`.
