"""Model-written code is visible and restricted by default."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mathslate.ai import Suggestion
from mathslate.ai import sandbox as _sandbox
from mathslate.errors import UnsupportedInputError


def _suggestion(code: str) -> Suggestion:
    return Suggestion(code=code, question="q", provider="test", model="test")


class TestRestrictedExecution:
    def test_ordinary_mathslate_code_runs(self) -> None:
        scope = _suggestion("result = plot(sin(x), verbose=False)").run()
        assert scope["result"].plan.kind == "curve"
        assert "expand" not in scope

    def test_a_final_plot_expression_is_returned_as_the_result(self) -> None:
        scope = _suggestion("plot(tan(x), verbose=False)").run()
        assert scope["result"].plan.kind == "curve"
        assert set(scope) == {"result"}

    def test_run_shows_the_visible_code_by_default(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _suggestion("result = sin(x)").run()
        assert "result = sin(x)" in capsys.readouterr().out

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

    @pytest.mark.parametrize(
        "code",
        [
            # `sympify` on a string `eval`s it, so a string argument to any
            # allowlisted callable that sympifies its input is arbitrary code —
            # the allowlist waves it through as a single Constant node.
            "solve(\"__import__('os').system('echo pwned')\")",
            "plot(\"__import__('os')\")",
            "analyze(\"().__class__.__base__.__subclasses__()\")",
            "integrate(\"__import__('os')\")",
            "Eq(\"().__class__\", 0)",
            # sympify recurses into containers, so a string buried in a range
            # tuple or a nested call is evaluated too.
            "plot([sin(x)], (x, \"__import__('os')\", 10))",
            "dataset({'x': solve(\"__import__('os')\")})",
        ],
    )
    def test_a_string_that_sympify_would_evaluate_is_refused(self, code: str) -> None:
        """The restricted allowlist checks what code *names*; a string literal
        that reaches `sympify` is code the allowlist never sees."""
        with pytest.raises(UnsupportedInputError, match="evaluate"):
            _suggestion(code).run()

    @pytest.mark.parametrize(
        "code",
        [
            # symbols()/Symbol() read a string as a name, never sympify it.
            "result = symbols('a b')",
            "result = Symbol('theta')",
            # dataset() reads string dict keys as column names, never sympifies.
            "result = dataset({'x': [0, 1, 2], 'y': [1.0, 3.0, 5.0]})",
            # the keywords named in _STRING_KEYWORDS read a label or a choice.
            "result = plot(sin(x), title='My Plot', kind='scatter', verbose=False)",
            "result = limit(1/x, x, 0, dir='+')",
            "a = slider(1, 3, name='a', label='amplitude')",
            "d = dataset({'t': [0, 1, 2], 'v': [1, 3, 5]})\n"
            "a, b = symbols('a b')\n"
            "result = d.fit(a*t + b, x='t', y='v')",
        ],
    )
    def test_strings_that_cannot_reach_sympify_are_allowed(self, code: str) -> None:
        _suggestion(code).run(show_code=False)

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

    def test_a_supplied_object_cannot_be_called_through_an_allowed_method(self) -> None:
        class Capability:
            def python(self) -> None:
                raise AssertionError("restricted code reached a caller capability")

        with pytest.raises(UnsupportedInputError, match="does not accept a namespace"):
            _suggestion("victim.python()").run({"victim": Capability()})

    def test_invalid_python_is_reported_as_input_error(self) -> None:
        with pytest.raises(UnsupportedInputError, match="valid Python"):
            _suggestion("plot(").validate()

    def test_unsafe_execution_is_an_explicit_escape_hatch(self) -> None:
        scope = _suggestion("import math\nresult = math.sqrt(9)").run(unsafe=True)
        assert scope["result"] == 3.0


class TestStringsThatReachSympifyByAnotherRoad:
    """A string need not be a positional literal to be evaluated as code."""

    @pytest.mark.parametrize(
        "code",
        [
            # keyword arguments are sympified too: these ran their string
            "series(x, x, x0=\"__import__('os').getpid()\")",
            "limit(x, x, z0=\"__import__('os').getpid()\")",
            "solve(x, x, domain=\"__import__('os').getpid()\")",
        ],
    )
    def test_a_string_keyword_outside_the_allowlist_is_refused(self, code: str) -> None:
        with pytest.raises(UnsupportedInputError, match="evaluate"):
            _suggestion(code).validate()

    @pytest.mark.parametrize(
        "expression",
        [
            "solve(d.names)",
            "plot(d.names[0])",
            "sin(d.names[0])",
            "Matrix([d.names])",
            "a = slider(1, 3)\nresult = a + d.names[0]",
        ],
    )
    def test_a_string_made_at_run_time_is_not_evaluated(
        self, expression: str, tmp_path: "os.PathLike[str]"
    ) -> None:
        """Column names are strings the AST never sees as literals."""
        marker = os.path.join(os.fspath(tmp_path), "ran")
        payload = f"__import__('pathlib').Path({marker!r}).write_text('x')"
        code = f"d = dataset({{{payload!r}: [1, 2]}})\n{expression}"
        with pytest.raises(UnsupportedInputError):
            _suggestion(code).run(show_code=False)
        assert not os.path.exists(marker)

    def test_the_restricted_process_does_not_inherit_provider_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-must-not-leak")
        monkeypatch.setenv("GEMINI_API_KEY", "must-not-leak")
        environment = _sandbox._child_environment()
        assert "ANTHROPIC_API_KEY" not in environment
        assert "GEMINI_API_KEY" not in environment
        assert "PATH" in environment

    @pytest.mark.parametrize(
        "name",
        [
            "PGPASSWORD",
            "GITHUB_PAT",
            "DATABASE_URL",
            "REDIS_URL",
            "AWS_ACCESS_KEY_ID",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ],
    )
    def test_common_credential_names_are_not_inherited(
        self, name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Credentials have no universal name shape, so filtering is allowlist-based."""
        monkeypatch.setenv(name, "must-not-leak")
        assert name not in _sandbox._child_environment()

    def test_numeric_runtime_settings_are_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
        assert _sandbox._child_environment()["OPENBLAS_NUM_THREADS"] == "2"

    def test_the_child_imports_the_callers_packages_after_startup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Source paths work without giving Python a startup-code hook."""
        root = str(_sandbox._mathslate_root())
        assert _sandbox._child_import_paths()[0] == root

        marker = tmp_path / "sitecustomize-ran"
        (tmp_path / "sitecustomize.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('ran', encoding='utf-8')\n",
            encoding="utf-8",
        )
        actual_paths = _sandbox._child_import_paths()
        monkeypatch.setattr(
            _sandbox,
            "_child_import_paths",
            lambda: (str(tmp_path), *actual_paths),
        )
        # A caller may have started with this setting, but the restricted child
        # must not inherit it: Python imports sitecustomize from PYTHONPATH before
        # the `-c` bootstrap can install the validated runner.
        monkeypatch.setenv("PYTHONPATH", str(tmp_path))
        monkeypatch.chdir(tmp_path.parent)

        assigned, _ = _sandbox._run_restricted("result = 1")

        assert assigned["result"] == 1
        assert "PYTHONPATH" not in _sandbox._child_environment()
        assert not marker.exists()

    def test_a_reply_naming_an_arbitrary_callable_is_not_unpickled(self) -> None:
        import io
        import pickle

        class Evil:
            def __reduce__(self) -> tuple[object, tuple[str]]:
                return (os.system, ("echo pwned",))

        with pytest.raises(pickle.UnpicklingError, match="not allowed"):
            _sandbox._ResultUnpickler(io.BytesIO(pickle.dumps(Evil()))).load()


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
    still bound it.

    Now that the work happens in a separate process, the real expression can
    be used: the bomb is confined to a child that is killed on the deadline,
    so nothing memory-hungry is left loose in the test session. That was the
    only reason the timeout used to be checked at the mechanism boundary.
    """

    def test_validation_alone_does_not_catch_an_expression_bomb(self) -> None:
        """Documents the gap the execution budget exists to cover."""
        _sandbox._validate_code("x = 9**9**9")  # does not raise

    def test_the_real_bomb_is_stopped_and_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole path, with a short budget so the test stays quick.

        Patching `_run_restricted` to raise the very error then asserted made
        this a tautology: it proved that a function which raises X propagates
        X, and would have passed against a `run()` with no budget at all.
        """
        monkeypatch.setattr(_sandbox, "_RUN_BUDGET", 3.0)
        with pytest.raises(UnsupportedInputError, match="execution budget"):
            _suggestion("x = 9**9**99").run()

    def test_a_host_that_cannot_start_a_process_says_so(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pyodide has no fork, and a hardened container may refuse. `run()`
        promises `UnsupportedInputError`, not a raw OSError from subprocess."""

        def refuse(*args: object, **kwargs: object) -> None:
            raise OSError("process creation is not permitted here")

        monkeypatch.setattr(_sandbox.subprocess, "run", refuse)
        with pytest.raises(UnsupportedInputError, match="separate Python process"):
            _suggestion("x = 1").run()

    def test_an_empty_namespace_is_refused_rather_than_quietly_dropped(self) -> None:
        """`run({})` used to be filled in place. Across a process boundary it
        cannot be, and returning a different mapping without a word would
        leave the caller reading their own empty dict."""
        with pytest.raises(UnsupportedInputError, match="does not accept a namespace"):
            _suggestion("x = 1").run({})

    def test_ordinary_code_is_unaffected(self) -> None:
        scope = _suggestion("result = plot(sin(x), verbose=False)").run()
        assert scope["result"].plan.kind == "curve"
