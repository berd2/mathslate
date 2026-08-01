"""Error types for MathSlate.

Every error carries a message written for a learner, not for a library author.
"""

from __future__ import annotations

__all__ = [
    "MathSlateError",
    "AmbiguousAxisError",
    "UnsupportedInputError",
    "NotYetImplementedError",
    "SamplingError",
]


class MathSlateError(Exception):
    """Base class for every MathSlate error."""


class AmbiguousAxisError(MathSlateError):
    """Raised when symbol-to-axis binding cannot be resolved (PRD 5.2 rule 4).

    The PRD says the system should *ask* rather than fail silently. In a
    library context the question is delivered as this exception's message.
    """

    def __init__(self, question: str, candidates: tuple[str, ...]) -> None:
        super().__init__(question)
        self.question: str = question
        self.candidates: tuple[str, ...] = candidates


class UnsupportedInputError(MathSlateError):
    """The input does not match any row of the plot() dispatch contract."""


class NotYetImplementedError(MathSlateError):
    """A documented API that is scheduled for a later milestone."""


class SamplingError(MathSlateError):
    """The expression could not be evaluated numerically at all."""
