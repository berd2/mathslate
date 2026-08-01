"""A learner's first plot() must never end in a UnicodeEncodeError."""

from __future__ import annotations

import io
from typing import Any

import pytest

from mathslate import floor, plot, sin, tan, x
from mathslate._text import encodable, safe_print, to_ascii


class LegacyConsole(io.StringIO):
    """A stream that behaves like a terminal on a narrow code page."""

    encoding = "ascii"

    def write(self, text: str) -> int:
        text.encode(self.encoding)  # raises exactly where a real console would
        return super().write(text)


class UnicodeConsole(io.StringIO):
    """A stream that can take anything, like a notebook."""

    encoding = "utf-8"


class TestTransliteration:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [("x ∈ [-1, 1]", "x in [-1, 1]"), ("a — b", "a -- b"), ("π/2", "pi/2"),
         ("  · note", "  - note")],
    )
    def test_the_decorative_characters_have_ascii_twins(
        self, source: str, expected: str
    ) -> None:
        assert to_ascii(source) == expected

    def test_encodability_is_checked_against_the_stream(self) -> None:
        assert encodable("plain", LegacyConsole()) is True
        assert encodable("a — b", LegacyConsole()) is False


class TestSafePrint:
    def test_it_does_not_raise_on_a_legacy_console(self) -> None:
        stream = LegacyConsole()
        safe_print("curve | x ∈ [-10, 10] — 200 samples", stream)
        assert "in [-10, 10] -- 200" in stream.getvalue()

    def test_it_keeps_unicode_when_the_stream_can_take_it(self) -> None:
        stream = UnicodeConsole()
        safe_print("x ∈ ℝ", stream)
        assert "∈" in stream.getvalue()


class TestPlotOutputIsSafe:
    @pytest.mark.parametrize("expr", [sin(x), tan(x), floor(x)], ids=lambda e: str(e))
    def test_the_report_and_the_code_both_survive(
        self, expr: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys

        console = LegacyConsole()
        monkeypatch.setattr(sys, "stdout", console)
        plot(expr, verbose=True).show_python()
        assert console.getvalue()
