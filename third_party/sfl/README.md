# Vendored SFL MiniGrid code (unmodified)

These files are copied byte-for-byte from `amacrutherford/sampling-for-learnability` (Rutherford
et al. 2024, "No Regrets: Investigating and Improving Regret Approximations for Curriculum
Discovery", arXiv 2408.15099) at commit `8b14e2ec07b38dc33a9df476b84faa2f3ca5fd3b` (21 Oct 2025),
keeping their paths. SFL is licensed Apache-2.0 (`LICENSE`, from the same commit), the same
licence as JaxUED. They are the reference SFL code for the paper's external check (maze E5,
`docs/prereg/maze-e5.md`): we run SFL's MiniGrid maze script at its own config defaults and audit
its checkpoints. Run it through `scripts/run_sfl_maze.py`, which applies a compatibility shim
without editing these files (see that script's docstring).

Downloaded on 27 Oct 2026 from raw.githubusercontent.com at the commit above (the download the
user approved that day), with the tree listing from the GitHub API for the blob hashes. Each
file's git blob SHA-1 was computed with `git hash-object` and matched the tree before it was
copied here; `apexzero.maze.sfl.check_vendored` checks them again before every E5 stage.

| Path | Bytes | Git blob SHA-1 |
|---|---|---|
| `LICENSE` | 11,357 | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` |
| `sfl/__init__.py` | 21 | `b3c06d488393abb3b3829e5590d42409c995b4cf` |
| `sfl/train/__init__.py` | 0 | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` |
| `sfl/train/minigrid_sfl.py` | 33,866 | `6436f5f659fa5fd0992c3ce479054eacb0353fdd` |
| `sfl/train/config/minigrid-sfl.yaml` | 466 | `41a016967485abcc8001b24ec91ddca783e54e6e` |
| `sfl/train/config/learning/ippo-minigrid.yaml` | 423 | `633e1793c235cce1f3b046075d7f14a1d68ac67d` |
| `sfl/train/config/env/minigrid.yaml` | 247 | `8df39e2c6881321897748e2450b51d43f3ad5b5c` |

**Not vendored.** `minigrid_sfl.py` imports two modules of the repository that the approval did
not cover (it named `sfl/train/common/network.py`, which `minigrid_sfl.py` does not import):

| Path | Bytes | Git blob SHA-1 | Imported for |
|---|---|---|---|
| `sfl/train/minigrid_plr.py` | 38,737 | `9537a794b7df09c76bf816215b366b03eb10b682` | `ActorCritic` (the network) and `evaluate_rnn` |
| `sfl/train/train_utils/__init__.py` | 44 | `b5437dc2e9f5ed3a6700b5414702fe5ba7902394` | `save_params` |
| `sfl/train/train_utils/common.py` | 6,308 | `be4f1f26619dc302aae1a23882ed287c93ebc01c` | (presumably `save_params`) |

The shim replaces `save_params` with E5's checkpoint writer, so `train_utils` is never needed.
`minigrid_plr.py` is: SFL's runs cannot start until it is vendored, and its download needs the
user's OK. Its own imports are not known (it was not downloaded).

**DEV-E5-2, how E5 gets checkpoints without editing these files.** `minigrid_sfl.py` saves only
its parameters, overwriting one file every `NUM_UPDATES / EVAL_FREQ / NUM_CHECKPOINTS` refresh
blocks, and never returns the learnable set D from its jitted step. With
`learning.NUM_CHECKPOINTS` equal to the number of blocks, the script calls `save_params` after
every block, and E5's writer keeps the 16 selected updates. D is captured by giving the module a
view of `jax` whose `jit` of `get_learnability_set` also sends its outputs to an `io_callback`;
the parameters are bit-identical with and without it (`tests/test_maze_e5.py`, on the CPU).
