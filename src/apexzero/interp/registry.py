"""The agents of protocol 8 and ruling R18: ids, runs, kinds, training variants and seeds.

Pure data and helpers, with no JAX and no other ``interp`` import, so every module that names
agents (support, stimuli, rollout, agents, analysis and scripts/psychophysics.py) reads them
here instead of hard-coding id lists. ``agents`` re-exports ``AgentSpec``, ``AGENTS`` and the
helpers; ``support`` re-exports ``GrammarSpec`` and ``GRAMMARS``.

Generalists come in three variants, one per training grammar (v1 base, v2 70/30 mix, v3 50/50
mix with alpha = 2). Seed 0 of each variant (G1, G2, G3) carries the pre-registered decisions
(protocol 8). Ruling R18 adds seeds 1 and 2 of each variant (G1s1, G1s2, G2s1, G2s2, G3s1,
G3s2) as prospective agents: trained by scripts/gpu_queue_replicates.sh with exactly the seed-0
runs' arguments (only the seed differs), never evaluated on a protocol stimulus before the
tag. A replicate gets the same tiers, keys and anchored keys as its seed-0 agent
(``seed0_of``), its own regimes and R_clip, and runs the finals and the last three checkpoints
only (P6's checkpoint history stays on seed 0). Its decisions form the replication family: a
variant-level claim needs the same decision in at least 2 of the variant's 3 seeds
(``variant_seeds``), otherwise it is seed-specific. Seeds 3 and 4 of each variant (same queue)
only measure the seed spread of R-dev completion; they are not agents of this study.

Specialists are the Hockenheim recipe's three seeds (S1 seed 1, S2 seed 2, S3 seed 3, as
their config.json records); they have no variant.
"""

from __future__ import annotations

from dataclasses import dataclass

# ------------------------------------------------------------------------- grammars


@dataclass(frozen=True)
class GrammarSpec:
    """A training grammar as scripts/train_dr.py builds it: ``fit_grammar`` on the 12 R_DEV
    circuits, with a per-strip mixture of ``tilted(g, alpha)`` (gap_scale 0.8) at
    ``tilt_fraction`` (the ``generate_bank`` mixture)."""

    name: str              # 'base' (v1), 'v2mix', 'v3mix'
    tilt_fraction: float   # share of strips drawn from the tilted grammar
    tilt_alpha: float      # tilted corner weights (1 / r_min) ** alpha


# The order fixes support.training_strips' seed streams (the grammar's index): never reorder.
GRAMMARS = {
    "base": GrammarSpec("base", 0.0, 1.0),     # v1: G1 (dr-pilot-200M); also l_typ and v_typ
    "v2mix": GrammarSpec("v2mix", 0.3, 1.0),   # v2: G2 (dr-pilot-v2-tilt30)
    "v3mix": GrammarSpec("v3mix", 0.5, 2.0),   # v3: G3 (dr-pilot-v3-tilt50a2)
}

# ------------------------------------------------------------------------- agents


@dataclass(frozen=True)
class AgentSpec:
    """One trained agent of protocol 8 (and ruling R18)."""

    agent_id: str                 # 'S1'..'S3', 'G1'..'G3', R18's 'G1s1'..'G3s2'
    run: str                      # repo-relative run directory
    kind: str                     # 'specialist' | 'generalist'
    pilot_informed: bool          # seen in the pilots (secondary family)
    prospective_after: float      # snapshots with timesteps > this are prospective (inf: none)
    final_file: str               # 'agent.pkl' (S1) or 'ckpt' (last ckpt/*.pkl)
    done_file: str                # file whose presence marks the run finished
    grammar: GrammarSpec | None   # training grammar (generalists)
    variant: str = ""             # generalists: 'v1' | 'v2' | 'v3' (one per grammar); '' else
    seed: int = 0                 # training seed (the run's config.json); 0 for G1-G3
    hardware: str = ""            # where it trained, as R18 records it ('' where not stated)


# Ruling R18: hardware differs between seed 0 and the replicates; it is recorded, not corrected.
SEED0_HARDWARE = "Mac CPU"
REPLICATE_HARDWARE = "GPU (NVIDIA L40S, scripts/gpu_queue_replicates.sh)"


def _gen(agent_id: str, run: str, grammar: str, variant: str, seed: int,
         pilot_informed: bool = False, prospective_after: float = 0.0) -> AgentSpec:
    return AgentSpec(agent_id, run, "generalist", pilot_informed, prospective_after, "ckpt",
                     "eval_rdev.json", GRAMMARS[grammar], variant, seed,
                     SEED0_HARDWARE if seed == 0 else REPLICATE_HARDWARE)


AGENTS = {
    "S1": AgentSpec("S1", "runs/hockenheim-specialist-mlp-200M", "specialist", True,
                    float("inf"), "agent.pkl", "agent.pkl", None, "", 1),
    "S2": AgentSpec("S2", "runs/hockenheim-specialist-mlp-200M-s2", "specialist", False,
                    122.3e6, "ckpt", "agent.pkl", None, "", 2),
    "S3": AgentSpec("S3", "runs/hockenheim-specialist-mlp-200M-s3", "specialist", False,
                    0.0, "ckpt", "agent.pkl", None, "", 3),
    "G1": _gen("G1", "runs/dr-pilot-200M", "base", "v1", 0, True, float("inf")),
    "G2": _gen("G2", "runs/dr-pilot-v2-tilt30", "v2mix", "v2", 0, True, float("inf")),
    "G3": _gen("G3", "runs/dr-pilot-v3-tilt50a2", "v3mix", "v3", 0),
    # Ruling R18: seeds 1 and 2 of each variant, prospective (never pilot-informed).
    "G1s1": _gen("G1s1", "runs/dr-pilot-200M-s1", "base", "v1", 1),
    "G1s2": _gen("G1s2", "runs/dr-pilot-200M-s2", "base", "v1", 2),
    "G2s1": _gen("G2s1", "runs/dr-pilot-v2-tilt30-s1", "v2mix", "v2", 1),
    "G2s2": _gen("G2s2", "runs/dr-pilot-v2-tilt30-s2", "v2mix", "v2", 2),
    "G3s1": _gen("G3s1", "runs/dr-pilot-v3-tilt50a2-s1", "v3mix", "v3", 1),
    "G3s2": _gen("G3s2", "runs/dr-pilot-v3-tilt50a2-s2", "v3mix", "v3", 2),
}

# ------------------------------------------------------------------------- helpers


def _spec(agent: str) -> AgentSpec | None:
    return AGENTS.get(agent)


def specialists() -> tuple[str, ...]:
    """The specialists, S1, S2, S3."""
    return tuple(a for a, s in AGENTS.items() if s.kind == "specialist")


def generalists() -> tuple[str, ...]:
    """Every generalist: the seed-0 agents (G1, G2, G3), then R18's replicates by variant
    and seed."""
    return tuple(a for a, s in AGENTS.items() if s.kind == "generalist")


def seed0_generalists() -> tuple[str, ...]:
    """The pre-registered generalists G1, G2, G3 (seed 0 of each variant): the primary and
    secondary decisions, P1c's labels, P6's checkpoint history and P8's crash units."""
    return tuple(a for a in generalists() if AGENTS[a].seed == 0)


def replicates(variant: str | None = None) -> tuple[str, ...]:
    """R18's replicate generalists (seed != 0), of one variant when ``variant`` is given."""
    return tuple(a for a in generalists() if AGENTS[a].seed != 0
                 and (variant is None or AGENTS[a].variant == variant))


def variants() -> tuple[str, ...]:
    """The generalist variants, in the order of their seed-0 agents: ('v1', 'v2', 'v3')."""
    return tuple(AGENTS[a].variant for a in seed0_generalists())


def variant_of(agent: str) -> str:
    """An agent's variant ('' for specialists, controls and unknown ids)."""
    s = _spec(agent)
    return s.variant if s is not None else ""


def seed_of(agent: str) -> int:
    """An agent's training seed (-1 for unknown ids)."""
    s = _spec(agent)
    return s.seed if s is not None else -1


def variant_seeds(variant: str) -> tuple[str, ...]:
    """A variant's agents in seed order: its seed-0 agent, then its replicates."""
    out = [a for a in generalists() if AGENTS[a].variant == variant]
    return tuple(sorted(out, key=lambda a: AGENTS[a].seed))


def is_generalist(agent: str) -> bool:
    s = _spec(agent)
    return s is not None and s.kind == "generalist"


def is_specialist(agent: str) -> bool:
    s = _spec(agent)
    return s is not None and s.kind == "specialist"


def is_replicate(agent: str) -> bool:
    """R18: a generalist of seed 1 or 2 (never G1, G2, G3, a specialist or a control)."""
    return is_generalist(agent) and AGENTS[agent].seed != 0


def seed0_of(agent: str) -> str:
    """The seed-0 agent of a generalist's variant (G1 for G1s1); any other id itself. A
    replicate gets the tiers, keys and anchored keys of this agent (R18)."""
    if not is_replicate(agent):
        return agent
    v = AGENTS[agent].variant
    return next(a for a in seed0_generalists() if AGENTS[a].variant == v)


def agent_grammar() -> dict[str, str]:
    """Each generalist's own training grammar name (replicates share their variant's)."""
    return {a: AGENTS[a].grammar.name for a in generalists() if AGENTS[a].grammar is not None}


def runs_cell(runners: tuple[str, ...] | list[str], agent: str) -> bool:
    """Whether ``agent`` runs a cell whose ``Cell.runners`` is ``runners`` (empty: every
    agent). R18: a replicate runs every cell its seed-0 agent runs, so tier D cells built from
    S1's state (``stimuli.RUNNERS_FROM_S1``) reach the replicates without changing the
    design."""
    return not runners or agent in runners or seed0_of(agent) in runners
