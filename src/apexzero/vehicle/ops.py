"""Math backends so the vehicle ODE is written once and compiled twice.

``JAX`` drives the vectorised RL simulator; ``CASADI`` builds the symbolic dynamics for
the minimum-lap-time optimal-control problem. Every non-smooth operation in the model
goes through the smooth helpers below, so both backends see exactly the same function
and IPOPT gets continuous derivatives.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Ops:
    name: str
    sin: Callable
    cos: Callable
    tan: Callable
    atan: Callable
    tanh: Callable
    sqrt: Callable
    stack: Callable[[list], Any]
    index: Callable[[Any, int], Any]
    where: Callable[[Any, Any, Any], Any]

    def smax(self, a, b, eps=1e-2):
        """Smooth max: 0.5 * (a + b + sqrt((a - b)^2 + eps^2))."""
        return 0.5 * (a + b + self.sqrt((a - b) ** 2 + eps**2))

    def smin(self, a, b, eps=1e-2):
        return 0.5 * (a + b - self.sqrt((a - b) ** 2 + eps**2))

    def sabs(self, x, eps=1e-3):
        return self.sqrt(x**2 + eps**2)

    def softsat(self, x, limit):
        """Smooth saturation to +/-limit that is close to the identity well inside it."""
        r = x / limit
        return x / (1.0 + r**8) ** 0.125

    def smoothstep(self, x):
        """C1 step from 0 (x <= 0) to 1 (x >= 1), via a clamped cubic."""
        t = self.smin(self.smax(x, 0.0, 1e-3), 1.0, 1e-3)
        return t * t * (3.0 - 2.0 * t)

    def relu(self, x, eps=2e-2):
        """C1 ReLU that is exactly 0 for x <= 0: quadratic on [0, eps], then x - eps/2."""
        quad = x * x / (2.0 * eps)
        return self.where(x <= 0.0, 0.0 * x, self.where(x < eps, quad, x - 0.5 * eps))


def _jax_ops() -> Ops:
    import jax.numpy as jnp

    return Ops(
        name="jax", sin=jnp.sin, cos=jnp.cos, tan=jnp.tan, atan=jnp.arctan, tanh=jnp.tanh,
        sqrt=jnp.sqrt, stack=lambda xs: jnp.stack(jnp.broadcast_arrays(*xs), axis=-1),
        index=lambda x, i: x[..., i], where=jnp.where,
    )


def _casadi_ops() -> Ops:
    import casadi as ca

    return Ops(
        name="casadi", sin=ca.sin, cos=ca.cos, tan=ca.tan, atan=ca.atan, tanh=ca.tanh,
        sqrt=ca.sqrt, stack=lambda xs: ca.vertcat(*xs), index=lambda x, i: x[i],
        where=lambda c, a, b: ca.if_else(c, a, b),
    )


class _Lazy:
    def __init__(self, factory):
        self._factory = factory
        self._ops = None

    def __getattr__(self, item):
        if self._ops is None:
            self._ops = self._factory()
        return getattr(self._ops, item)


JAX: Ops = _Lazy(_jax_ops)  # type: ignore[assignment]
CASADI: Ops = _Lazy(_casadi_ops)  # type: ignore[assignment]
