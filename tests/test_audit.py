import numpy as np

from apexzero.data.audit import lateral_shift, reference_curve


def test_reference_curve_and_lateral_shift_on_a_circle():
    t = np.linspace(0, 20, 75)                       # 3.7 Hz, like the feed
    xy = np.stack([100 * np.cos(0.3 * t), 100 * np.sin(0.3 * t)], axis=1)
    curve = reference_curve(t, xy)
    r = np.hypot(curve[:, 0], curve[:, 1])
    assert np.max(np.abs(r - 100)) < 0.05          # spline, not chords (sagitta would be ~1 m)
    shifted = lateral_shift(t, xy, 2.0)             # +2 m to the left of travel = inward here
    assert np.allclose(np.hypot(shifted[:, 0], shifted[:, 1]), 98.0, atol=1e-3)
