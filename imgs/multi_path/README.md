# Multi-path point maze

Circular / ring corridor for `simple_multi_path`. From the south start to the
north goal there are two routes (west and east); **subgoal A** sits on the west
path and **subgoal B** on the east path.

## Schematic layout PNGs

Generate / refresh schematic grid plots:

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

## MuJoCo / brax-instantiated renders

`brax_instantiated/` contains real `mujoco.Renderer` RGB frames after
`SimpleMaze(maze_layout_name="multi_path", task_name=...).reset()`:

- `*_render.png` — oblique free-camera overview
- `*_render_topdown.png` — near top-down free-camera view

These are the same renderer path as brax `image.render_array` / MuJoCo offscreen
frames (not matplotlib geom sketches).
