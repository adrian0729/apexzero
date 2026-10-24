"""Worker pools for the long stages of scripts/costate.py (``POOLED``): probes and own after
the tag and the nulls before it, one unit per network, and decoy-k before the tag, one unit
per candidate k, world and validation seed (``gates.decoy_k``).

``run(fn, units, workers, state)`` returns ``[fn(state, unit) for unit in units]``. With one
worker (or one unit) it is that loop in this process: the stage's serial path. Otherwise the
units run in ``workers`` spawned processes and are gathered in the units' order. Either way
the stage then writes and stamps its outputs from this process, after every unit has
returned, so the two paths write the same outputs:

- **The same numbers.** A unit's result depends only on the unit and on ``state``, never on
  the worker or the order. Each unit seeds its own streams from its own ids. ``state`` (the
  stage's world, say) is pickled once into a temporary file that every worker loads when it
  starts, so its arrays are the parent's bit for bit (and a large state never sits in a
  spawn pipe: a worker that dies at start fails the run instead of blocking the parent).
  Each worker runs with the parent's thread pins (``THREAD_ENV``: XLA's intra-op threads
  and the BLAS libraries', one thread on a pooled stage; inherited at spawn and checked),
  since floating-point summation order can depend on the thread count (Accelerate's does at
  2 threads), and with x64 as the parent has it.
- **Spawned, not forked**, as step 2's ``gates.map_keys`` (JAX is not fork-safe). A worker
  allows trained networks only where the parent does (``critic``, ``probes`` and
  ``network.unlock_trained``, which scripts/costate.py calls after its freeze check): it
  mirrors the parent and decides nothing. A unit carries everything it reads, or reads it
  from ``state``: the parent loads the checkpoints and checks their pins, so a worker never
  opens a checkpoint or the record.
- **A failure fails the run.** The first unit that raises, or a worker that dies (out of
  memory, say), stops the pool: the pending units are cancelled and the running ones ended,
  and ``WorkerFailed`` names the unit (a dead worker's: every unit left unfinished). Nothing
  is returned, so the stage writes nothing.

A function of scripts/costate.py itself (a script loaded by path, not an importable module)
reaches a worker as a ``ScriptFn``, its file and name, which the worker loads once.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import pickle
import sys
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path
from typing import Any

POOLED = ("probes", "own", "nulls", "decoy-k")   # the stages with --workers
BLAS_THREADS = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS")        # Accelerate (the Mac), OpenBLAS, OpenMP, MKL
THREAD_ENV = ("XLA_FLAGS",) + BLAS_THREADS   # scripts/costate.py pins them before numpy loads
LOCKS = ("apexzero.costate.critic", "apexzero.costate.probes", "apexzero.fields.network")


class WorkerFailed(RuntimeError):
    """A unit raised in a worker process, or a worker died: the run returns nothing."""


def default_workers(threads: int, cores: int | None = None) -> int:
    """The default pool: the machine's cores over each worker's threads (``--threads``), less
    one for the parent and the system; at least 1. 5 on the 12-core Mac at 2 threads."""
    cores = int(cores or os.cpu_count() or 1)
    return max(1, cores // max(1, int(threads)) - 1)


def thread_env() -> dict[str, str | None]:
    """This process' thread pins (``THREAD_ENV``), which every worker must share."""
    return {k: os.environ.get(k) for k in THREAD_ENV}


def _x64_now() -> bool | None:
    """Whether x64 is on here (None when JAX was never imported: the worker leaves it)."""
    jax = sys.modules.get("jax")
    return None if jax is None else bool(jax.config.jax_enable_x64)


def _locks_now() -> dict[str, bool]:
    """{module: trained networks allowed} here (``LOCKS``; a module not imported is locked)."""
    return {m: bool(getattr(sys.modules.get(m), "_TRAINED_OK", False)) for m in LOCKS}


# ------------------------------------------------------------------------- script functions


@dataclasses.dataclass(frozen=True)
class ScriptFn:
    """A function of a script (``path``: a file such as scripts/costate.py, not an importable
    module) by ``name``, callable in a spawned worker: the worker loads the script once (the
    module already loaded from that file if there is one, as spawn's re-run of the main
    script), its directory on sys.path as when it runs."""

    path: str
    name: str

    def __call__(self, *args, **kwargs):
        return getattr(_script(self.path), self.name)(*args, **kwargs)


_SCRIPTS: dict[str, Any] = {}


def _file_of(mod) -> str | None:
    f = getattr(mod, "__file__", None)
    return str(Path(f).resolve()) if f else None


def _script(path: str):
    if path not in _SCRIPTS:
        mod = next((m for m in list(sys.modules.values()) if _file_of(m) == path), None)
        if mod is None:
            d = str(Path(path).parent)
            if d not in sys.path:
                sys.path.insert(0, d)
            spec = importlib.util.spec_from_file_location(f"_worker_{Path(path).stem}", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
        _SCRIPTS[path] = mod
    return _SCRIPTS[path]


def _ref(fn: Callable) -> Callable:
    """``fn`` as a spawned worker can unpickle it: itself when its module is importable by name
    (apexzero's), else a ``ScriptFn``. Only a module-level function qualifies."""
    if isinstance(fn, ScriptFn):
        return fn
    if getattr(fn, "__qualname__", None) != getattr(fn, "__name__", None) or not hasattr(
            fn, "__code__"):
        raise TypeError(f"{fn!r}: a worker runs a module-level function only")
    if str(fn.__module__).startswith("apexzero."):
        return fn
    return ScriptFn(str(Path(fn.__code__.co_filename).resolve()), fn.__name__)


# ------------------------------------------------------------------------- the worker side


_WORKER: dict = {}


def _start(path: str, x64: bool | None, locks: dict[str, bool],
           env: dict[str, str | None]) -> None:
    """A worker's initializer: the parent's thread pins checked, x64 as the parent has it,
    trained networks allowed where the parent allows them, and the shared state loaded from
    the parent's file (``path``)."""
    bad = {k: (os.environ.get(k), v) for k, v in env.items() if os.environ.get(k) != v}
    if bad:
        raise RuntimeError(f"a worker's thread pins differ from its parent's: {bad}")
    if x64 is not None:
        import jax

        jax.config.update("jax_enable_x64", bool(x64))
    for m, ok in locks.items():
        if ok:
            importlib.import_module(m).unlock_trained()
    with open(path, "rb") as fh:
        _WORKER["state"] = pickle.load(fh)


def _call(fn: Callable, unit):
    return fn(_WORKER.get("state"), unit)


def _stop(ex: ProcessPoolExecutor) -> None:
    """Cancel the pending units and end the running ones (a failed run returns nothing)."""
    procs = list((getattr(ex, "_processes", None) or {}).values())
    for p in procs:
        if p.is_alive():
            p.terminate()
    ex.shutdown(wait=True, cancel_futures=True)


# ------------------------------------------------------------------------- the pool


def _raise_first(names: Sequence[str], futs: Sequence) -> None:
    """``WorkerFailed`` for the first unit (in the units' order) that raised, if any. A worker
    that dies breaks the pool, and every unit not yet done then fails with
    ``BrokenProcessPool``: those are named together, since which of them was running in the
    dead worker is not known."""
    errs = [(name, f.exception()) for name, f in zip(names, futs, strict=True)
            if f.done() and not f.cancelled() and f.exception() is not None]
    if not errs:
        return
    own = [(name, e) for name, e in errs if not isinstance(e, BrokenProcessPool)]
    if own:
        name, e = own[0]
        raise WorkerFailed(f"unit {name} failed in a worker: {type(e).__name__}: {e}") from e
    e = errs[0][1]
    raise WorkerFailed(f"a worker died with units {', '.join(n for n, _ in errs)} unfinished: "
                       f"{type(e).__name__}: {e}") from e


def run(fn: Callable, units: Sequence, workers: int = 1, state=None,
        names: Sequence[str] | None = None) -> list:
    """``[fn(state, unit) for unit in units]``, in this process when ``workers`` (capped at
    the number of units) is 1, else in that many spawned processes (module docstring), the
    results in the units' order. ``names`` label the units in a failure (default: their
    index). In a worker, the first unit to raise stops the pool and raises ``WorkerFailed``
    (``_raise_first``; the serial path raises the unit's own exception)."""
    units = list(units)
    n = max(1, min(int(workers), len(units)))
    if n == 1:
        return [fn(state, u) for u in units]
    names = [str(i) for i in range(len(units))] if names is None else [str(s) for s in names]
    ref = _ref(fn)
    fd, path = tempfile.mkstemp(prefix="costate-workers-", suffix=".pkl")
    try:
        with os.fdopen(fd, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        ex = ProcessPoolExecutor(max_workers=n, mp_context=get_context("spawn"),
                                 initializer=_start,
                                 initargs=(path, _x64_now(), _locks_now(), thread_env()))
        try:
            futs = [ex.submit(_call, ref, u) for u in units]
            wait(futs, return_when=FIRST_EXCEPTION)
            _raise_first(names, futs)
            out = [f.result() for f in futs]
        except BaseException:
            _stop(ex)
            raise
        ex.shutdown(wait=True)
    finally:
        Path(path).unlink(missing_ok=True)
    return out
