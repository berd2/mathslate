"""Printing that survives a non-UTF-8 console.

MathSlate's inference report and π tick labels are nicer with real Unicode,
and notebooks render it happily. A Windows terminal on a legacy code page
(cp949, cp1252, …) raises ``UnicodeEncodeError`` instead — which would turn a
learner's very first ``plot()`` into a traceback. So: keep the Unicode, and
transliterate only when the destination cannot take it.
"""

from __future__ import annotations

import sys
from typing import Final, TextIO

__all__ = ["to_ascii", "encodable", "safe_print"]

_REPLACEMENTS: Final[tuple[tuple[str, str], ...]] = (
    ("∈", "in"),        # ∈
    ("·", "-"),         # ·
    ("—", "--"),        # —
    ("–", "-"),         # –
    ("→", "->"),        # →
    ("π", "pi"),        # π
    ("θ", "theta"),     # θ
    ("φ", "phi"),       # φ
    ("≤", "<="),        # ≤
    ("≥", ">="),        # ≥
    ("×", "x"),         # ×
    ("∞", "inf"),       # ∞
)


def to_ascii(text: str) -> str:
    """Replace MathSlate's decorative Unicode with plain ASCII equivalents."""
    for source, target in _REPLACEMENTS:
        text = text.replace(source, target)
    return text.encode("ascii", "replace").decode("ascii")


def encodable(text: str, stream: TextIO) -> bool:
    """Can ``stream`` represent ``text`` without loss?"""
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return True
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def safe_print(text: str, stream: TextIO | None = None) -> None:
    """``print`` that transliterates rather than raising or printing mojibake.

    Checking up front matters: a console configured with ``errors='replace'``
    does not raise, it silently prints question marks.
    """
    target = stream if stream is not None else sys.stdout
    if not encodable(text, target):
        text = to_ascii(text)
    try:
        print(text, file=target)
    except UnicodeEncodeError:
        print(to_ascii(text), file=target)
