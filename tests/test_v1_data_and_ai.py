"""PRD 7 v1.0 — `dataset()`, statistics, linear algebra, the AI layer, worksheets.

Everything before this milestone starts from an expression. Real work usually
starts from measurements, and `Dataset.fit` is where the two meet: you write
the model as you would on paper and get **the same expression back with its
parameters filled in**, still SymPy, so `diff`, `analyze` and `plot` all work
on the result. `TestTheBridge` is the class that matters here; the rest is
plumbing that gets you to it.

`TestTheCoreStaysOffline` is the other one to read. PRD §8 lists "AI dependency
blocked on school or corporate networks" as a risk whose mitigation is that the
core works fully offline and the assistant ships as extras. That is a property
of the import graph, and properties of import graphs rot silently, so it is
asserted rather than intended.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
import sympy as sp

import mathslate as ms
from mathslate import Matrix, analyze, dataset, exp, plot, sin, table, x
from mathslate.classroom import Worksheet, worksheet
from mathslate.errors import UnsupportedInputError

a, b = sp.symbols("a b", real=True)


class TestTheBridge:
    """`fit()` is the point of v1.0: data in, symbolic model out."""

    def test_a_straight_line_is_solved_exactly(self) -> None:
        data = dataset({"x": [0, 1, 2, 3], "y": [1.0, 3.0, 5.0, 7.0]})
        found = data.fit(a * x + b)
        assert float(found.parameters[a]) == pytest.approx(2.0)
        assert float(found.parameters[b]) == pytest.approx(1.0)
        assert found.r_squared == pytest.approx(1.0)

    def test_the_result_is_still_a_sympy_expression(self) -> None:
        """Which is the whole bridge: everything else in MathSlate works on it."""
        data = dataset({"x": [0, 1, 2, 3], "y": [1.0, 3.0, 5.0, 7.0]})
        fitted = data.fit(a * x + b).expr
        assert isinstance(fitted, sp.Expr)
        assert float(sp.diff(fitted, x)) == pytest.approx(2.0)
        assert plot(fitted, verbose=False).plan.kind == "curve"
        assert analyze(fitted).roots.values == pytest.approx((-0.5,))

    def test_linear_in_the_parameters_is_not_linear_in_x(self) -> None:
        """a*x**2 + b has an exact least-squares answer; the fit must use it."""
        xs = np.linspace(-2.0, 2.0, 9)
        data = dataset({"x": xs, "y": 3.0 * xs**2 - 1.0})
        found = data.fit(a * x**2 + b)
        assert float(found.parameters[a]) == pytest.approx(3.0, rel=1e-9)
        assert float(found.parameters[b]) == pytest.approx(-1.0, abs=1e-9)

    def test_a_non_linear_model_is_fitted_by_iteration(self) -> None:
        xs = np.linspace(0.0, 2.0, 21)
        data = dataset({"x": xs, "y": 2.0 * np.exp(0.7 * xs)})
        found = data.fit(a * exp(b * x))
        assert float(found.parameters[a]) == pytest.approx(2.0, rel=1e-3)
        assert float(found.parameters[b]) == pytest.approx(0.7, rel=1e-3)

    def test_the_residuals_come_back_so_the_fit_can_be_judged(self) -> None:
        data = dataset({"x": [0, 1, 2, 3], "y": [1.0, 3.1, 4.9, 7.2]})
        found = data.fit(a * x + b)
        assert found.residuals.shape == (4,)
        assert 0.9 < (found.r_squared or 0.0) <= 1.0

    def test_constant_data_reports_r_squared_as_undefined(self) -> None:
        """R² compares against "predict the mean"; there is nothing to beat."""
        data = dataset({"x": [0, 1, 2, 3], "y": [5.0, 5.0, 5.0, 5.0]})
        assert data.fit(a * x + b).r_squared is None

    def test_a_model_with_no_parameters_is_refused(self) -> None:
        data = dataset({"x": [0, 1], "y": [0.0, 1.0]})
        with pytest.raises(UnsupportedInputError, match="no free parameters"):
            data.fit(sin(x))

    def test_too_few_rows_for_the_parameters_is_refused(self) -> None:
        data = dataset({"x": [0.0], "y": [1.0]})
        with pytest.raises(UnsupportedInputError, match="cannot determine"):
            data.fit(a * x + b)

    def test_rank_deficient_linear_parameters_are_refused(self) -> None:
        """All x=0 leaves the slope unknowable; an arbitrary zero is not a fit."""
        data = dataset({"x": [0.0, 0.0, 0.0], "y": [1.0, 2.0, 3.0]})
        with pytest.raises(UnsupportedInputError, match="rank-deficient"):
            data.fit(a * x + b)

    def test_rank_deficient_nonlinear_parameters_are_refused(self) -> None:
        """Only the product a*b is visible, so a and b cannot be separated."""
        data = dataset({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
        with pytest.raises(UnsupportedInputError, match="rank-deficient"):
            data.fit(a * b * x)

    def test_non_finite_rows_are_dropped_not_propagated(self) -> None:
        data = dataset({"x": [0, 1, 2, 3], "y": [1.0, float("nan"), 5.0, 7.0]})
        assert float(data.fit(a * x + b).parameters[a]) == pytest.approx(2.0)


class TestBuildingOne:
    def test_from_a_mapping(self) -> None:
        data = dataset({"t": [0, 1], "v": [2.0, 4.0]})
        assert data.names == ("t", "v") and len(data) == 2

    def test_from_a_two_dimensional_array(self) -> None:
        data = dataset(np.arange(6.0).reshape(3, 2), columns=["p", "q"])
        assert data.names == ("p", "q") and len(data) == 3

    def test_from_a_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "readings.csv"
        path.write_text("x,y\n0,1\n1,3\n2,5\n", encoding="utf-8")
        data = dataset(str(path))
        assert data.names == ("x", "y")
        assert data["y"].tolist() == [1.0, 3.0, 5.0]

    def test_a_csv_cell_that_is_not_a_number_becomes_a_blank(self, tmp_path: Path) -> None:
        """A hand-exported sheet has a note in it; refusing the file helps nobody."""
        path = tmp_path / "messy.csv"
        path.write_text("x,y\n0,1\n1,check this\n2,5\n", encoding="utf-8")
        assert np.isnan(dataset(str(path))["y"][1])

    def test_ragged_columns_are_refused(self) -> None:
        with pytest.raises(UnsupportedInputError, match="same length"):
            dataset({"x": [1, 2, 3], "y": [1, 2]})

    def test_an_empty_dataset_is_refused_at_construction(self) -> None:
        with pytest.raises(UnsupportedInputError, match="at least one row"):
            dataset({"x": [], "y": []})

    def test_an_unknown_column_names_the_ones_there_are(self) -> None:
        with pytest.raises(UnsupportedInputError, match="x, y"):
            dataset({"x": [1.0], "y": [2.0]})["z"]

    def test_describe_covers_every_column(self) -> None:
        text = dataset({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}).describe()
        assert "mean" in text and "x" in text and "y" in text

    def test_the_numpy_hatch(self) -> None:
        names, values = dataset({"x": [1.0, 2.0], "y": [3.0, 4.0]}).numpy
        assert names == ("x", "y") and values.shape == (2, 2)


class TestCsvEncodings:
    """A spreadsheet exported on a Korean or Western Windows box is not UTF-8.

    Excel writes cp949 in Korea and cp1252 in much of Europe, and `read_bytes`
    then `decode("utf-8")` met that with a raw `UnicodeDecodeError` from deep in
    the stdlib — no file name, no suggestion, and no hint that an `encoding=`
    argument exists. For the audience in PRD 2.1 that is the end of the session.
    """

    HEADER = "시간,측정값\n"
    ROWS = "0,1\n1,3\n2,5\n"

    def _write(self, path: Path, encoding: str) -> Path:
        path.write_bytes((self.HEADER + self.ROWS).encode(encoding))
        return path

    @pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp949"])
    def test_it_is_read_without_being_told_the_encoding(
        self, tmp_path: Path, encoding: str
    ) -> None:
        path = self._write(tmp_path / f"{encoding}.csv", encoding)
        data = dataset(str(path))
        assert data.names == ("시간", "측정값")
        assert data["측정값"].tolist() == [1.0, 3.0, 5.0]

    def test_a_bom_does_not_become_part_of_the_first_column_name(
        self, tmp_path: Path
    ) -> None:
        """Excel's BOM turned column `x` into `\\ufeffx`, which then had no data."""
        path = tmp_path / "bom.csv"
        path.write_bytes("x,y\n0,1\n".encode("utf-8-sig"))
        assert dataset(str(path)).names == ("x", "y")

    def test_an_explicit_encoding_is_honoured(self, tmp_path: Path) -> None:
        path = self._write(tmp_path / "korean.csv", "cp949")
        assert dataset(str(path), encoding="cp949").names == ("시간", "측정값")

    def test_the_wrong_explicit_encoding_says_what_to_do(self, tmp_path: Path) -> None:
        """Naming an encoding turns off the guessing, so it must fail usefully."""
        path = self._write(tmp_path / "korean.csv", "cp949")
        with pytest.raises(UnsupportedInputError) as caught:
            dataset(str(path), encoding="utf-8")
        message = str(caught.value)
        assert "korean.csv" in message
        assert "utf-8" in message and "encoding=" in message
        assert "cp949" not in message, "it was told not to guess; it must not claim it did"

    def test_a_binary_file_is_refused_rather_than_mangled(self, tmp_path: Path) -> None:
        """`latin-1` decodes any byte, so this used to be silently accepted.

        Handed a PNG, the reader produced a one-column dataset named `\\x89PNG`
        — no error anywhere, and a plausible-looking object holding nothing.
        """
        path = tmp_path / "picture.csv"
        path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe\xfd")
        with pytest.raises(UnsupportedInputError, match="not a text file"):
            dataset(str(path))

    def test_the_binary_refusal_names_the_utf16_escape(self, tmp_path: Path) -> None:
        """UTF-16 legitimately contains NUL bytes, so it must be reachable."""
        path = tmp_path / "wide.csv"
        path.write_bytes("x,y\n0,1\n1,3\n".encode("utf-16"))
        with pytest.raises(UnsupportedInputError, match="encoding='utf-16'"):
            dataset(str(path))
        data = dataset(str(path), encoding="utf-16")
        assert data.names == ("x", "y") and data["y"].tolist() == [1.0, 3.0]


class TestPlottingData:
    def test_a_dataset_plots_first_column_across(self) -> None:
        data = dataset({"x": [0, 1, 2], "y": [1.0, 2.0, 3.0], "z": [3.0, 2.0, 1.0]})
        plan = plot(data, verbose=False).plan
        assert plan.kind == "data"
        assert [s.name for s in plan.series] == ["y", "z"]

    def test_the_axis_titles_come_from_the_column_names(self) -> None:
        data = dataset({"time": [0, 1], "speed": [1.0, 2.0]})
        figure = plot(data, verbose=False).plotly
        assert figure.layout.xaxis.title.text == "time"
        assert figure.layout.yaxis.title.text == "speed"

    def test_a_histogram(self) -> None:
        result = plot([1.0, 2.0, 2.0, 3.0], kind="hist", verbose=False)
        assert result.plan.kind == "hist"
        assert isinstance(result.plotly.data[0], go.Histogram)

    def test_a_box_plot_per_column(self) -> None:
        data = dataset({"a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 9.0]})
        result = plot(data, kind="box", verbose=False)
        assert [type(t).__name__ for t in result.plotly.data] == ["Box", "Box"]

    def test_histograms_are_overlaid_not_stacked(self) -> None:
        """Stacking answers a question about the sum, which nobody asked."""
        data = dataset({"a": [1.0, 2.0], "b": [1.0, 2.0]})
        assert plot(data, kind="hist", verbose=False).plotly.layout.barmode == "overlay"


class TestLinearAlgebra:
    def test_a_matrix_is_drawn_as_what_it_does(self) -> None:
        result = plot(Matrix([[2, 1], [1, 3]]), verbose=False)
        assert result.plan.kind == "linalg"
        before, after = result.plotly.data[0], result.plotly.data[1]
        assert list(before.x) == [0.0, 1.0, 1.0, 0.0, 0.0]
        # The unit square's corner (1,1) goes to (3,4) under this matrix.
        assert (float(after.x[2]), float(after.y[2])) == (3.0, 4.0)

    def test_the_eigenvectors_are_drawn(self) -> None:
        plan = plot(Matrix([[2, 1], [1, 3]]), verbose=False).plan
        assert len(plan.eigen) == 2
        assert sorted(round(v, 3) for v, _ in plan.eigen) == [1.382, 3.618]

    def test_a_rotation_says_it_has_no_real_eigenvectors(self) -> None:
        result = plot(Matrix([[0, -1], [1, 0]]), verbose=False)
        assert result.plan.eigen == []
        assert any("turns every direction" in n for n in result.notes)

    def test_the_determinant_is_reported_with_its_meaning(self) -> None:
        notes = plot(Matrix([[2, 0], [0, 3]]), verbose=False).notes
        assert any("areas are scaled by 6" in n for n in notes)

    def test_a_flip_is_called_a_flip(self) -> None:
        notes = plot(Matrix([[1, 0], [0, -1]]), verbose=False).notes
        assert any("orientation is flipped" in n for n in notes)

    def test_the_aspect_is_locked(self) -> None:
        """Otherwise the distortion you see is the plot's, not the matrix's."""
        figure = plot(Matrix([[2, 0], [0, 1]]), verbose=False).plotly
        assert figure.layout.yaxis.scaleanchor == "x"

    def test_a_matrix_that_is_not_2x2_says_so(self) -> None:
        with pytest.raises(UnsupportedInputError, match="2x2"):
            plot(Matrix([[1, 2, 3], [4, 5, 6]]), verbose=False)


class TestShowPythonForV1Kinds:
    CASES = (
        ("dataset", lambda: plot(dataset({"x": [0, 1, 2], "y": [1.0, 2.0, 3.0]}), verbose=False)),
        ("hist", lambda: plot([1.0, 2.0, 2.0, 3.0], kind="hist", verbose=False)),
        ("box", lambda: plot([1.0, 2.0, 2.0, 3.0], kind="box", verbose=False)),
        ("linalg", lambda: plot(Matrix([[2, 1], [1, 3]]), verbose=False)),
    )

    @pytest.mark.parametrize(("label", "build"), CASES, ids=[c[0] for c in CASES])
    def test_it_runs_and_draws_the_same_traces(
        self, label: str, build: object, headless_show: list[object]
    ) -> None:
        result = build()  # type: ignore[operator]
        namespace: dict[str, object] = {}
        exec(compile(result.python(), "<c>", "exec"), namespace)  # noqa: S102
        emitted = namespace["fig"]
        assert [type(t).__name__ for t in emitted.data] == [  # type: ignore[union-attr]
            type(t).__name__ for t in result.plotly.data
        ]

    def test_a_multi_column_dataset_emits_every_column(self) -> None:
        """One series was emitted where there were two, until it was checked."""
        data = dataset({"x": [0, 1], "y": [1.0, 2.0], "z": [3.0, 4.0]})
        result = plot(data, verbose=False)
        namespace: dict[str, object] = {}
        exec(compile(result.python(), "<c>", "exec"), namespace)  # noqa: S102
        assert len(namespace["fig"].data) == 2  # type: ignore[union-attr]

    def test_the_matrix_is_emitted_as_a_product_not_as_its_answer(self) -> None:
        """`M @ square` is the whole idea; the eight resulting numbers hide it."""
        assert "M @ square" in plot(Matrix([[2, 1], [1, 3]]), verbose=False).python()

    def test_matrix_legend_visibility_survives_show_python(
        self, headless_show: list[object]
    ) -> None:
        result = plot(
            Matrix([[2, 1], [1, 3]]), show_legend=False, verbose=False
        )
        namespace: dict[str, object] = {}
        exec(compile(result.python(), "<c>", "exec"), namespace)  # noqa: S102
        assert result.plotly.layout.showlegend is False
        assert namespace["fig"].layout.showlegend is False  # type: ignore[union-attr]

    @pytest.mark.parametrize(
        "build",
        [
            lambda: plot(
                [1.0, 2.0, 2.0, 3.0],
                kind="hist",
                title="Distribution",
                verbose=False,
            ),
            lambda: plot(
                [1.0, 2.0, 2.0, 3.0],
                kind="box",
                title="Distribution",
                verbose=False,
            ),
            lambda: plot(
                Matrix([[2, 1], [1, 3]]),
                title="Distribution",
                verbose=False,
            ),
        ],
        ids=["hist", "box", "linalg"],
    )
    def test_v1_titles_survive_show_python(
        self, build: object, headless_show: list[object]
    ) -> None:
        result = build()  # type: ignore[operator]
        namespace: dict[str, object] = {}
        exec(compile(result.python(), "<c>", "exec"), namespace)  # noqa: S102
        assert namespace["fig"].layout.title.text == "Distribution"  # type: ignore[union-attr]

    def test_a_large_histogram_keeps_every_observation(
        self, headless_show: list[object]
    ) -> None:
        values = np.arange(5000.0)
        result = plot(values, kind="hist", verbose=False)
        namespace: dict[str, object] = {}
        exec(compile(result.python(), "<c>", "exec"), namespace)  # noqa: S102
        emitted = np.asarray(namespace["fig"].data[0].x, dtype=float)  # type: ignore[union-attr]
        assert emitted.size == values.size
        assert np.array_equal(emitted, values)


class TestTheCoreStaysOffline:
    """PRD §8's mitigation, asserted rather than intended."""

    def test_importing_mathslate_does_not_reach_the_ai_layer(self) -> None:
        code = (
            "import sys, mathslate; "
            "print([m for m in sys.modules "
            "if m.startswith(('anthropic', 'openai', 'google.genai')) "
            "or m == 'mathslate.ai'])"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "[]"

    def test_importing_the_ai_layer_does_not_import_any_sdk(self) -> None:
        """The SDKs are imported inside the call, so this stays free."""
        code = (
            "import sys, mathslate.ai; "
            "print([m for m in sys.modules "
            "if m.startswith(('anthropic', 'openai', 'google'))])"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "[]"

    def test_no_ai_package_is_a_runtime_dependency(self) -> None:
        # `read_text()` alone decodes with the *locale* encoding, so this blew
        # up with a UnicodeDecodeError on any machine whose default is not
        # UTF-8 (cp949, cp1252) the moment pyproject.toml grew a non-ASCII
        # character — which the comments in its dependency table have.
        text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        runtime = text.split("dependencies = [", 1)[1].split("]", 1)[0]
        for package in ("anthropic", "openai", "google-genai"):
            assert package not in runtime


class TestTheAiLayer:
    def test_it_says_exactly_what_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On a machine with nothing installed — which is what the message is
        for, and the only state in which it names the packages to install.

        Both halves have to be forced. Reading whatever the developer's
        environment happens to hold made this fail wherever an SDK was
        installed (the message then names the missing *key*, correctly), and
        on a machine with a real key in the environment `ask()` would not
        raise at all: it would reach `backend.complete()` and bill a live
        request to whoever ran the tests.
        """
        from mathslate.ai import ask
        from mathslate.ai.providers import PROVIDERS, Provider

        monkeypatch.setattr(Provider, "installed", lambda self: False)
        for provider in PROVIDERS:
            monkeypatch.delenv(provider.env_var, raising=False)

        with pytest.raises(UnsupportedInputError) as caught:
            ask("plot sine")
        message = str(caught.value)
        assert "works offline" in message
        assert "anthropic" in message and "openai" in message and "google-genai" in message

    def test_an_unknown_provider_lists_the_known_ones(self) -> None:
        from mathslate.ai import resolve_provider

        with pytest.raises(UnsupportedInputError, match="claude, openai, gemini"):
            resolve_provider("clippy")

    def test_the_pip_name_is_not_assumed_to_be_the_import_path(self) -> None:
        from mathslate.ai import PROVIDERS

        gemini = next(p for p in PROVIDERS if p.name == "gemini")
        assert gemini.package == "google.genai"
        assert gemini.pip_name == "google-genai"

    def test_an_empty_question_is_refused_before_any_network_call(self) -> None:
        from mathslate.ai import ask

        with pytest.raises(UnsupportedInputError, match="needs a question"):
            ask("   ")

    def test_an_explicit_key_needs_no_environment_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mathslate.ai.providers import Provider, resolve_provider

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(Provider, "installed", lambda self: True)
        assert resolve_provider("openai", api_key="sk-explicit").name == "openai"

    def test_ask_passes_the_explicit_key_through_resolution_and_build(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mathslate.ai.suggest as suggest
        from mathslate.ai.providers import Provider

        seen: dict[str, object] = {}

        class FakeBackend:
            def complete(self, system: str, question: str, model: str) -> str:
                return "```python\nplot(sin(x))\n```"

        def build(key: str | None) -> FakeBackend:
            seen["built_with"] = key
            return FakeBackend()

        provider = Provider(
            name="openai",
            package="openai",
            pip_name="openai",
            env_var="OPENAI_API_KEY",
            default_model="test-model",
            build=build,
        )

        def resolve(name: str | None, *, api_key: str | None = None) -> Provider:
            seen["resolved_name"] = name
            seen["resolved_key"] = api_key
            return provider

        monkeypatch.setattr(suggest, "resolve_provider", resolve)
        answer = suggest.ask(
            "plot sine", provider="openai", api_key="sk-explicit"
        )
        assert answer.code == "plot(sin(x))"
        assert seen == {
            "resolved_name": "openai",
            "resolved_key": "sk-explicit",
            "built_with": "sk-explicit",
        }

    def test_configure_accepts_provider_and_key_in_the_same_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mathslate.ai.suggest as suggest
        from mathslate.ai.providers import Provider

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(Provider, "installed", lambda self: True)
        for name in ("provider", "model", "api_key"):
            monkeypatch.setitem(suggest._DEFAULTS, name, None)

        suggest.configure(provider="openai", api_key="sk-explicit")

        assert suggest._DEFAULTS["provider"] == "openai"
        assert suggest._DEFAULTS["api_key"] == "sk-explicit"

    def test_the_prompt_is_built_from_the_real_dispatch_contract(self) -> None:
        """So a new plot kind cannot be added without the assistant hearing."""
        from mathslate.ai import system_prompt
        from mathslate.core.dispatch import KINDS

        prompt = system_prompt()
        for kind in KINDS:
            assert kind in prompt

    def test_a_suggestion_holds_code_and_does_not_run_it(self) -> None:
        """A model writing Python that the library runs unseen is the one thing
        a learner cannot check."""
        from mathslate.ai import Suggestion

        suggestion = Suggestion(
            code="raise AssertionError('this must not run on its own')",
            question="q",
            provider="test",
            model="test",
        )
        assert "must not run" in suggestion.code  # constructed, never executed

    def test_run_is_explicit_and_brings_mathslate_with_it(self) -> None:
        from mathslate.ai import Suggestion

        scope = Suggestion(
            code="result = plot(sin(x), verbose=False)",
            question="q",
            provider="test",
            model="test",
        ).run()
        assert scope["result"].plan.kind == "curve"  # type: ignore[union-attr]

    def test_it_pulls_the_code_out_of_a_fenced_reply(self) -> None:
        from mathslate.ai.suggest import _split

        code, commentary = _split("Here you go.\n\n```python\nplot(sin(x))\n```\nDone.")
        assert code == "plot(sin(x))"
        assert "Here you go." in commentary

    def test_an_unfenced_reply_is_still_usable(self) -> None:
        from mathslate.ai.suggest import _split

        assert _split("plot(sin(x))")[0] == "plot(sin(x))"


class TestWorksheets:
    def _page(self) -> Worksheet:
        return worksheet(
            [
                "Where does sin(x)/x go at zero?",
                ("The graph", plot(sin(x) / x, verbose=False)),
                ("The numbers", table(sin(x) / x, (x, -1, 1), rows=5)),
                ("What it is", analyze(sin(x) / x)),
            ],
            title="Limits",
        )

    def test_it_holds_everything_it_was_given(self) -> None:
        assert len(self._page()) == 4

    def test_the_page_stands_alone(self) -> None:
        page = self._page().html()
        assert re.search(r'<script[^>]*\ssrc="', page) is None
        assert len(page) > 1_000_000

    def test_plotly_is_embedded_once_however_many_figures(self) -> None:
        """Ten plots must not mean a 40 MB file."""
        one = worksheet([plot(sin(x), verbose=False)]).html()
        three = worksheet(
            [plot(sin(x), verbose=False) for _ in range(3)]
        ).html()
        # Counting "Plotly.newPlot" would count the bundle's own source too.
        assert three.count("plotly-graph-div") == 3
        assert len(three) < len(one) * 1.5

    def test_the_headings_and_title_appear(self) -> None:
        page = self._page().html()
        assert "<h1>Limits</h1>" in page
        assert "<h2>The graph</h2>" in page

    def test_text_is_escaped_not_interpreted(self) -> None:
        page = worksheet(["compare a < b and a > b"]).html()
        assert "&lt; b" in page and "<p>" in page

    def test_a_text_item_can_have_a_heading(self) -> None:
        page = worksheet([("Introduction", "Read this first")]).html()
        assert "<h2>Introduction</h2>" in page
        assert "<p>Read this first</p>" in page

    def test_it_writes_the_file(self, tmp_path: Path) -> None:
        destination = self._page().save(tmp_path / "handout.html")
        assert destination.exists() and destination.stat().st_size > 1_000_000

    def test_an_interactive_plot_survives_the_trip(self) -> None:
        """The reason §11.8 chose frames over a widget: email has no kernel."""
        from mathslate import slider
        from mathslate.ui import interact

        slider_a = slider(-2, 2, default=0, name="worksheet_a")
        try:
            page = worksheet([plot(slider_a * sin(x), verbose=False)]).html()
            assert "addFrames" in page
        finally:
            interact.release_all()

    def test_an_empty_worksheet_says_so(self) -> None:
        with pytest.raises(UnsupportedInputError, match="nothing to hand out"):
            worksheet([]).html()

    def test_it_refuses_what_it_cannot_render(self) -> None:
        with pytest.raises(UnsupportedInputError, match="does not know what to do"):
            worksheet([{"not": "renderable"}])
