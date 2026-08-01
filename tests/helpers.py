"""Shared checks for the MathSlate test-suite."""

from __future__ import annotations

import numpy as np

from mathslate.core.sampling import SampleResult

__all__ = ["crossing_points", "finite_fraction"]


def crossing_points(
    sample: SampleResult, points: tuple[float, ...], *, axis: str = "x"
) -> list[float]:
    """Points that a drawn line segment jumps straight across.

    A discontinuity that is handled correctly has a NaN between the samples on
    either side of it, so no segment spans it. Anything returned here is a
    spurious connecting line — exactly what PRD acceptance criterion 1 forbids.

    ``axis`` says which coordinate the breakpoints are expressed in. For an
    explicit curve x *is* the parameter, so the default is right. For a
    parametric or polar curve the breakpoints are values of t, and testing them
    against x would ask an unrelated question — x is not monotone there, and
    ``x = 1.57`` on the curve has nothing to do with the pole at ``t = 1.57``.
    """
    x, y = sample.x, sample.y
    drawn = np.isfinite(x[:-1]) & np.isfinite(x[1:]) & np.isfinite(y[:-1]) & np.isfinite(y[1:])
    along = x if axis == "x" else sample.t
    assert along is not None, "the sample has no parameter array"
    drawn &= np.isfinite(along[:-1]) & np.isfinite(along[1:])
    left, right = along[:-1], along[1:]
    offenders: list[float] = []
    for point in points:
        spans = drawn & (np.minimum(left, right) < point) & (point < np.maximum(left, right))
        if spans.any():
            offenders.append(point)
    return offenders


def finite_fraction(sample: SampleResult) -> float:
    if sample.n_points == 0:
        return 0.0
    return sample.finite_count / sample.n_points
