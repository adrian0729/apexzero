"""Compatibility shims for third-party reference code.

jaxued 0.0.1 (the reference UED implementation we diff against and reuse) calls
``jax.tree_map``, which recent JAX releases removed in favour of ``jax.tree.map``.
Call ``jaxued_compat()`` before importing jaxued; it restores the alias without changing
behaviour, so the reference code runs unmodified.
"""

import jax


def jaxued_compat() -> None:
    if not hasattr(jax, "tree_map"):
        jax.tree_map = jax.tree.map  # type: ignore[attr-defined]


jaxued_compat()
