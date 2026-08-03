"""MathSlate as a tool an outside agent calls — computed answers, not recalled ones."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mathslate.ai import tools
from mathslate.errors import UnsupportedInputError


class TestTheAnswersAreComputed:
    def test_roots_come_back_exact_with_their_method(self) -> None:
        """The point of the whole module: an agent can tell a solved answer
        from a sampled one instead of treating both as fact."""
        result = tools.call("mathslate_analyze", {"expression": "x**3 - 3*x"})
        assert result["roots"]["values"] == pytest.approx(
            [-1.7320508075688772, 0.0, 1.7320508075688772]
        )
        assert result["roots"]["exact"] == ["-sqrt(3)", "0", "sqrt(3)"]
        assert result["roots"]["approximate"] is False
        assert "solveset" in result["roots"]["method"]
        assert result["any_approximate"] is False
        assert result["symmetry"] == "odd"

    def test_an_approximate_answer_says_so(self) -> None:
        """`x - cos(x)` is a ConditionSet to solveset, so the root is sampled."""
        result = tools.call("mathslate_analyze", {"expression": "x - cos(x)"})
        assert result["roots"]["approximate"] is True
        assert result["any_approximate"] is True
        assert result["roots"]["values"][0] == pytest.approx(0.739085, abs=1e-5)

    def test_a_window_is_honoured(self) -> None:
        result = tools.call(
            "mathslate_analyze", {"expression": "sin(x)", "range": ["x", 0, 3.5]}
        )
        assert result["window"] == pytest.approx([0.0, 3.5])
        assert result["roots"]["count"] == 2

    def test_a_table_returns_the_numbers(self) -> None:
        result = tools.call(
            "mathslate_table",
            {"expression": "x**2", "range": ["x", 0, 3], "rows": 4},
        )
        assert result["headers"] == ["x", "x**2"]
        assert [row[0] for row in result["rows"]] == pytest.approx([0.0, 1.0, 2.0, 3.0])
        assert [row[1] for row in result["rows"]] == pytest.approx([0.0, 1.0, 4.0, 9.0])

    def test_an_undefined_point_is_null_not_missing(self) -> None:
        """JSON has no NaN. Dropping the row instead would silently misalign
        the column against its inputs."""
        result = tools.call(
            "mathslate_table", {"expression": "1/x", "range": ["x", -1, 1], "rows": 3}
        )
        assert result["rows"][1] == [0.0, None]

    def test_a_plot_returns_what_can_be_said_about_it(self) -> None:
        result = tools.call("mathslate_plot", {"expression": "sin(x)/x"})
        assert result["kind"] == "curve"
        assert "411 samples" in result["summary"]
        assert any("singularities" in note for note in result["notes"])
        assert "import plotly" in result["python"]


class TestArgumentsAreUntrusted:
    """A tool call has exactly the trust of a model-written suggestion: none."""

    @pytest.mark.parametrize(
        "arguments",
        [
            {"expression": "__import__('os').system('echo pwned')"},
            {"expression": "solve(\"__import__('os').system('echo pwned')\")"},
            {"expression": "().__class__.__base__.__subclasses__()"},
            {"expression": "sin(x)) or __import__('os').system('echo pwned') or (1"},
            {"expression": "sin(x)", "range": ["x) or __import__('os') or (0", 0, 1]},
            {"expression": "sin(x)", "range": ["x", "0) or __import__('os') or (0", 1]},
        ],
    )
    def test_an_injection_is_refused_rather_than_run(
        self, arguments: dict[str, Any], tmp_path: Path
    ) -> None:
        marker = tmp_path / "pwned"
        assert not marker.exists()
        result = tools.call("mathslate_analyze", arguments)
        assert "error" in result
        assert not marker.exists()

    def test_a_kind_may_not_carry_source(self) -> None:
        result = tools.call(
            "mathslate_plot",
            {"expression": "x*y", "kind": "contour'); __import__('os').system('id'); ('"},
        )
        assert "error" in result and "plain name" in result["error"]

    def test_a_runaway_computation_is_stopped_by_the_budget(self) -> None:
        result = tools.call("mathslate_analyze", {"expression": "9**9**9"})
        assert "error" in result and "budget" in result["error"]

    def test_a_malformed_range_is_refused(self) -> None:
        assert "error" in tools.call(
            "mathslate_analyze", {"expression": "sin(x)", "range": ["x", 5, 5]}
        )
        assert "error" in tools.call(
            "mathslate_analyze", {"expression": "sin(x)", "range": ["x", 0]}
        )


class TestTheAgentGetsSomethingItCanActOn:
    def test_a_bad_expression_is_a_message_not_an_exception(self) -> None:
        """The caller is an agent mid-conversation: a message it can read and
        correct beats an exception it has to be shielded from."""
        result = tools.call("mathslate_analyze", {"expression": "@@@"})
        assert "error" in result
        assert isinstance(result["error"], str)

    def test_a_missing_expression_says_what_was_wanted(self) -> None:
        result = tools.call("mathslate_analyze", {})
        assert "expression must be a non-empty string" in result["error"]

    def test_an_unknown_tool_is_a_programming_error_not_a_reply(self) -> None:
        with pytest.raises(UnsupportedInputError, match="unknown tool"):
            tools.call("mathslate_teleport", {})

    def test_every_result_is_json_serialisable(self) -> None:
        """It crosses a wire to a model; anything numpy or sympy would not."""
        for name, arguments in (
            ("mathslate_analyze", {"expression": "sin(x)/x"}),
            ("mathslate_table", {"expression": "sin(x)", "rows": 5}),
            ("mathslate_plot", {"expression": "x*y"}),
        ):
            json.dumps(tools.call(name, arguments))


class TestTheSchemas:
    def test_both_dialects_describe_the_same_tools(self) -> None:
        anthropic = tools.tool_schemas("anthropic")
        openai = tools.tool_schemas("openai")
        assert [t["name"] for t in anthropic] == list(tools.tool_names())
        assert [t["function"]["name"] for t in openai] == list(tools.tool_names())
        assert anthropic[0]["input_schema"] == openai[0]["function"]["parameters"]

    def test_they_are_json_serialisable(self) -> None:
        json.dumps(tools.tool_schemas("anthropic"))
        json.dumps(tools.tool_schemas("openai"))

    def test_every_tool_requires_an_expression(self) -> None:
        for tool in tools.TOOLS:
            assert tool.parameters["required"] == ["expression"]

    def test_an_unknown_dialect_is_refused(self) -> None:
        with pytest.raises(UnsupportedInputError, match="unknown dialect"):
            tools.tool_schemas("cohere")


def test_importing_tools_costs_no_provider_sdk() -> None:
    """The core promise: nothing here reaches for a provider package."""
    import subprocess
    import sys

    code = (
        "import sys; import mathslate.ai.tools; "
        "print(any(m in sys.modules for m in ('anthropic', 'openai', 'google')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
