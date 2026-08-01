"""Model-written code is visible and restricted by default."""

from __future__ import annotations

import os

import pytest

from mathslate.ai import Suggestion
from mathslate.ai import suggest as _suggest
from mathslate.core._budget import SymbolicTimeout
from mathslate.errors import UnsupportedInputError


def _suggestion(code: str) -> Suggestion:
    return Suggestion(code=code, question="q", provider="test", model="test")


class TestRestrictedExecution:
    def test_ordinary_mathslate_code_runs(self) -> None:
        scope = _suggestion("result = plot(sin(x), verbose=False)").run()
        assert scope["result"].plan.kind == "curve"

    def test_dataset_fit_method_is_allowed(self) -> None:
        scope = _suggestion(
            "a, b = symbols('a b')\n"
            "data = dataset({'x': [0, 1, 2], 'y': [1, 3, 5]})\n"
            "result = data.fit(a*x + b)"
        ).run()
        assert len(scope["result"].parameters) == 2

    @pytest.mark.parametrize(
        "code,word",
        [
            ("import os\nos.system('echo unsafe')", "Import"),
            ("open('stolen.txt', 'w')", "open"),
            ("result = x.__class__", "private"),
            ("for item in [1]:\n    print(item)", "For"),
            ("result = (lambda: 1)()", "indirect call"),
        ],
    )
    def test_dangerous_or_unbounded_constructs_are_refused(
        self, code: str, word: str
    ) -> None:
        with pytest.raises(UnsupportedInputError, match=word):
            _suggestion(code).run()

    def test_a_supplied_namespace_cannot_restore_builtins(self) -> None:
        scope = {"__builtins__": __builtins__, "os": os}
        with pytest.raises(UnsupportedInputError, match="system"):
            _suggestion("os.system('echo unsafe')").run(scope)

    def test_a_supplied_module_cannot_expose_environment_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-readable")
        with pytest.raises(UnsupportedInputError, match="environ"):
            _suggestion("secret = os.environ['OPENAI_API_KEY']").run({"os": os})

    def test_invalid_python_is_reported_as_input_error(self) -> None:
        with pytest.raises(UnsupportedInputError, match="valid Python"):
            _suggestion("plot(").validate()

    def test_unsafe_execution_is_an_explicit_escape_hatch(self) -> None:
        scope = _suggestion("import math\nresult = math.sqrt(9)").run(unsafe=True)
        assert scope["result"] == 3.0


class TestRestrictedDatasetCannotReadTheFilesystem:
    """`dataset()` is in the allowlist because literal data is legitimate;
    the CSV-path form of the same call is a file-read primitive and must not
    be reachable from restricted execution."""

    def test_a_path_is_refused(self, tmp_path: "os.PathLike[str]") -> None:
        secret = os.path.join(str(tmp_path), "secret.csv")
        with open(secret, "w", encoding="utf-8") as handle:
            handle.write("x,y\n1,999\n2,888\n")
        code = f"leaked = dataset({secret!r})"
        with pytest.raises(UnsupportedInputError, match="restricted execution"):
            _suggestion(code).run()

    def test_literal_data_still_works(self) -> None:
        scope = _suggestion(
            "data = dataset({'x': [0, 1, 2], 'y': [1, 3, 5]})\n"
            "result = data.describe()"
        ).run()
        assert "result" in scope


class TestRestrictedExecutionHasATimeBudget:
    """The AST allowlist bounds what a suggestion can *name*, not what a
    legal expression costs to run: ``9**9**9`` is three Constant/BinOp nodes
    and passes validation, yet computing it exhausts memory. ``run()`` must
    still bound it — but proving that with the real expression would leave a
    multi-second, memory-hungry thread running loose in the test session, so
    the timeout is exercised at the mechanism boundary instead.
    """

    def test_validation_alone_does_not_catch_an_expression_bomb(self) -> None:
        """Documents the gap the execution budget exists to cover."""
        _suggest._validate_code("x = 9**9**9")  # does not raise

    def test_a_timeout_becomes_an_input_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_within_budget(operation: object, *args: object, **kwargs: object) -> None:
            raise SymbolicTimeout("simulated")

        monkeypatch.setattr(_suggest, "within_budget", fake_within_budget)
        with pytest.raises(UnsupportedInputError, match="execution budget"):
            _suggestion("x = 1").run()

    def test_ordinary_code_is_unaffected(self) -> None:
        scope = _suggestion("result = plot(sin(x), verbose=False)").run()
        assert scope["result"].plan.kind == "curve"
