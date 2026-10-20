"""Paths, column tables, JSON, hashes and stage logs shared by the psychophysics modules.

Tables are plain column dicts (``dict[str, np.ndarray]``, equal-length 1-d columns) saved as
``.npz`` without pickle, so they need no dependency beyond numpy (pandas is not a declared
dependency of the repo) and they hash reproducibly. String columns are fixed-width unicode
arrays. Every stage writes under ``outputs/psychophysics/`` (gitignored) and logs to
``runs/psychophysics-<stage>.log``, which the terminal dashboard lists on its own.

This module is implemented in the skeleton (not left to a builder) because every other
module writes through it.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Self

import numpy as np

from apexzero.interp.constants import LOG_EVERY, ROOT_SEED, STREAM_OFFSET, STREAMS

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs" / "psychophysics"
MANIFEST_PATH = OUT / "manifest.json"
SUPPORT_DIR = OUT / "support"
STIMULI_DIR = OUT / "stimuli"
KEYS_DIR = OUT / "keys"
KEY_KINDS = ("nominal", "backoff", "phase", "null", "anchored", "viability", "socp", "actuation")
CONTROLS_DIR = OUT / "controls"
ROLLOUTS_DIR = OUT / "rollouts"
PROBES_DIR = OUT / "probes"
ANALYSIS_DIR = OUT / "analysis"
RUNS_DIR = REPO / "runs"
SEALED_DIR = REPO / "outputs" / "sealed"   # never read (lockbox)

Table = dict[str, np.ndarray]


def key_dir(kind: str) -> Path:
    """``outputs/psychophysics/keys/<kind>/`` for one of ``KEY_KINDS``."""
    if kind not in KEY_KINDS:
        raise ValueError(f"unknown key kind {kind!r}; expected one of {KEY_KINDS}")
    return KEYS_DIR / kind


def rollout_path(agent: str, timesteps: int) -> Path:
    """``outputs/psychophysics/rollouts/<agent>/<timesteps:012d>.npz`` (one file per snapshot)."""
    return ROLLOUTS_DIR / agent / f"{int(timesteps):012d}.npz"


# ------------------------------------------------------------------------- tables


def check_table(table: Table) -> int:
    """Number of rows; raises if the columns are not equal-length 1-d arrays."""
    n = None
    for k, v in table.items():
        a = np.asarray(v)
        if a.ndim != 1:
            raise ValueError(f"column {k!r} is {a.ndim}-d; tables hold 1-d columns only")
        if n is None:
            n = len(a)
        elif len(a) != n:
            raise ValueError(f"column {k!r} has {len(a)} rows, expected {n}")
    return n or 0


def _table_cols(table: Table) -> dict[str, np.ndarray]:
    check_table(table)
    cols = {k: np.asarray(v) for k, v in table.items()}
    for k, v in cols.items():
        if v.dtype == object:
            raise TypeError(f"column {k!r} has dtype object; use str, float, int or bool")
    return cols


def save_table(path: str | Path, table: Table) -> Path:
    """Write a column table to ``path`` (.npz, no pickle), atomically."""
    cols = _table_cols(table)
    return _atomic(Path(path), lambda fh: np.savez(fh, **cols))


def table_bytes(table: Table) -> bytes:
    """``save_table``'s file content in memory, with its checks (a run that writes several
    files serialises them all before writing the first)."""
    import io as _io

    cols = _table_cols(table)
    buf = _io.BytesIO()
    np.savez(buf, **cols)
    return buf.getvalue()


def write_bytes(path: str | Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` atomically."""
    return _atomic(Path(path), lambda fh: fh.write(data))


def load_table(path: str | Path) -> Table:
    with np.load(Path(path), allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def concat_tables(tables: list[Table]) -> Table:
    """Row-wise concatenation of tables with identical columns."""
    if not tables:
        return {}
    cols = list(tables[0])
    for t in tables[1:]:
        if list(t) != cols:
            raise ValueError("tables have different columns")
    return {k: np.concatenate([np.asarray(t[k]) for t in tables]) for k in cols}


def select(table: Table, mask: np.ndarray) -> Table:
    return {k: np.asarray(v)[mask] for k, v in table.items()}


# ------------------------------------------------------------------------- JSON


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None if np.isnan(x) else ("inf" if x > 0 else "-inf")
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "__dataclass_fields__"):
        return _jsonable({k: getattr(x, k) for k in x.__dataclass_fields__})
    if hasattr(x, "_asdict"):
        return _jsonable(x._asdict())
    return x


def write_json(path: str | Path, obj: Any) -> Path:
    """Atomic, sorted-key JSON (dataclasses, NamedTuples and numpy values converted;
    non-finite floats become null / "inf" / "-inf")."""
    text = json_bytes(obj)
    return _atomic(Path(path), lambda fh: fh.write(text))


def json_bytes(obj: Any) -> bytes:
    """``write_json``'s file content in memory."""
    return json.dumps(_jsonable(obj), indent=1, sort_keys=True).encode()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _umask() -> int:
    old = os.umask(0)
    os.umask(old)
    return old


_FILE_MODE = 0o666 & ~_umask()       # the mode open() gives a new file (0644 under umask 022)


def _atomic(path: Path, write) -> Path:
    """Write via a temp file and ``os.replace``. ``mkstemp`` creates the temp file 0600, so
    it is given the mode a plain ``open`` would (0666 less the umask) before the replace:
    outputs and runs/activity.md stay readable as before."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            write(fh)
        os.chmod(tmp, _FILE_MODE)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


# ------------------------------------------------------------------------- hashes and seeds


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_arrays(*arrays: np.ndarray) -> str:
    """Hash of array contents (dtype, shape and bytes), independent of file containers."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(obj), sort_keys=True).encode()).hexdigest()


def code_hash(roots: tuple[str, ...] = ("src/apexzero", "scripts/psychophysics.py")) -> str:
    """Hash of every .py file under ``roots`` (repo-relative path and bytes), for the manifest."""
    h = hashlib.sha256()
    files: list[Path] = []
    for r in roots:
        p = REPO / r
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    for f in files:
        h.update(str(f.relative_to(REPO)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def stream_key(name: str):
    """PRNG key of a named seed stream (constants.STREAMS), disjoint from the noise keys."""
    import jax

    return jax.random.fold_in(jax.random.PRNGKey(ROOT_SEED), STREAM_OFFSET + STREAMS[name])


def stream_rng(name: str) -> np.random.Generator:
    """NumPy generator of a named seed stream."""
    return np.random.default_rng([ROOT_SEED, STREAM_OFFSET + STREAMS[name]])


def assert_x64_off() -> None:
    """Rollouts run in float32: ``replay`` and ``check`` switch x64 on globally, so the harness
    asserts it is off (run keys and rollouts in separate processes, or inside
    ``jax.enable_x64(False)``)."""
    import jax

    if jax.config.jax_enable_x64:
        raise RuntimeError("jax_enable_x64 is on; psychophysics rollouts must run in float32")


# ------------------------------------------------------------------------- stage logs


class StageLog:
    """``runs/psychophysics-<stage>.log`` with a ``done/total`` line at least once a minute,
    and a one-line note in ``runs/activity.md`` while the stage runs (protocol 10).

        with StageLog("keys", total=74) as log:
            for ...:
                log.step(note="R13.1/T180 A1")
    """

    def __init__(self, stage: str, total: int, every: float = LOG_EVERY, note: str = ""):
        self.stage, self.total, self.every = stage, int(total), every
        self.done = 0
        self.path = RUNS_DIR / f"psychophysics-{stage}.log"
        self._last = 0.0
        self._t0 = 0.0
        self._note = f"psychophysics {stage} [pid {os.getpid()}]"   # one line per process
        self._what = note
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._beat: threading.Thread | None = None
        self._last_note = ""

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()
        self._last = self._t0
        self.write(f"start {self.stage}: 0/{self.total}" + (f" ({self._what})" if self._what else ""))
        self._activity(f"running, pid {os.getpid()}" + (f", {self._what}" if self._what else ""))
        if self.every > 0:
            # A single item (a key solve, a compile) can take longer than ``every``: a heartbeat
            # thread repeats the done/total line so the log never goes quiet for a minute.
            self._beat = threading.Thread(target=self._heartbeat, daemon=True)
            self._beat.start()
        return self

    def _heartbeat(self) -> None:
        while not self._stop.wait(min(self.every, 5.0)):
            with self._lock:
                due = time.time() - self._last >= self.every
            if due:
                self._emit(self._last_note)

    def write(self, line: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.path, "a") as fh:
            fh.write(f"{stamp} {line}\n")

    def _emit(self, note: str) -> None:
        with self._lock:
            self._last = time.time()
            line = f"{self.done}/{self.total}" + (f" {note}" if note else "")
        self.write(line)

    def step(self, n: int = 1, note: str = "") -> None:
        with self._lock:
            self.done += n
            if note:
                self._last_note = note
            due = time.time() - self._last >= self.every or self.done >= self.total
        if due:
            self._emit(note)

    def set_total(self, total: int) -> None:
        """Change the total once it is known (e.g. after a resume check)."""
        with self._lock:
            self.total = int(total)

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._beat is not None:
            self._beat.join(timeout=10.0)
        mins = (time.time() - self._t0) / 60.0
        status = "failed" if exc_type else "done"
        self.write(f"{status} {self.stage}: {self.done}/{self.total} in {mins:.1f} min"
                   + (f" ({exc_type.__name__}: {exc})" if exc_type else ""))
        self._activity(None)

    def _activity(self, text: str | None) -> None:
        """Add, replace or remove this stage's one-line note in runs/activity.md (under an
        exclusive lock, since several stages and other agents edit the file)."""
        import fcntl

        notes = RUNS_DIR / "activity.md"
        tag = f"  {self._note}: "        # the process's own line (visible, no markup)
        notes.parent.mkdir(parents=True, exist_ok=True)
        with open(notes.parent / ".activity.lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                lines = notes.read_text().splitlines() if notes.exists() else []
                lines = [ln for ln in lines if tag not in ln]
                if text is not None:
                    lines.append(f"- {time.strftime('%H:%M')}{tag}{text}")
                _atomic(notes, lambda fh: fh.write(("\n".join(lines) + "\n").encode()))
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)
