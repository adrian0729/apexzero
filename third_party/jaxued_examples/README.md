# Vendored JaxUED example (unmodified)

`maze_plr.py` is copied byte-for-byte from DramaCow/jaxued at commit
`0f8f1284677375b889e4f13a32c9617cd009f8c4` (`examples/maze_plr.py`). JaxUED is licensed
Apache-2.0 (per its PyPI package metadata). It is the reference DR / PLR / ACCEL maze
training code for the paper's external check (E5): we run it at its own defaults and
audit its checkpoints. Run it through `scripts/run_jaxued_maze.py`, which applies a JAX
compatibility shim (`jax.tree_map`) and forces wandb offline, without editing this file.
