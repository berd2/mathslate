"""Retrying with the error as evidence, and choosing a model from the data."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from mathslate import dataset
from mathslate.ai.fitting import fit_evidence, suggest_model
from mathslate.ai.providers import Provider
from mathslate.ai.suggest import Suggestion
from mathslate.errors import UnsupportedInputError


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    class Backend:
        def complete(self, system: str, question: str, model: str) -> str:
            seen["system"] = system
            seen["question"] = question
            return seen.get("reply", "Fixed it.\n```python\nplot(sin(x))\n```")

    provider = Provider(
        name="claude", package="sys", pip_name="anthropic",
        env_var="ANTHROPIC_API_KEY", default_model="model",
        build=lambda key: Backend(),
    )
    import mathslate.ai.suggest as suggest
    from mathslate.ai import credentials as credentials_module

    monkeypatch.setattr(
        suggest, "resolve_provider", lambda name, *, api_key=None: provider
    )
    # Mocking resolve_provider does not stop _resolve_request from reaching
    # load_credential first — .repair() below omits provider/model, so
    # without this a developer's real saved credential decides model_name
    # (via saved.model) ahead of the fake provider's own default, on a
    # machine with one stored through assistant()'s "remember" option.
    monkeypatch.setattr(credentials_module, "load_credential", lambda *_: None)
    return seen


class TestRepair:
    def test_a_refused_draft_comes_back_corrected(self, spy: dict[str, Any]) -> None:
        draft = Suggestion(
            code="import os", question="plot sine", provider="p", model="m"
        )
        assert draft.repair(verbose=False).code == "plot(sin(x))"

    def test_the_refusal_itself_is_the_evidence(self, spy: dict[str, Any]) -> None:
        """The finding behind the SageMath-agent results: a model gets closer on
        the second try when it is shown what actually happened."""
        Suggestion(
            code="import os", question="plot sine", provider="p", model="m"
        ).repair(verbose=False)
        assert "restricted execution does not allow" in spy["question"]
        assert "import os" in spy["question"]

    def test_the_original_question_is_carried_so_the_fix_still_answers_it(
        self, spy: dict[str, Any]
    ) -> None:
        repaired = Suggestion(
            code="import os", question="plot the tangent", provider="p", model="m"
        ).repair(verbose=False)
        assert "plot the tangent" in spy["question"]
        assert "still answer the original question" in spy["system"]
        assert repaired.question == "plot the tangent"

    def test_a_runtime_failure_can_be_passed_in(self, spy: dict[str, Any]) -> None:
        """Some code only fails once it runs, so the error comes from the caller."""
        draft = Suggestion(
            code="analyze(x*y)", question="describe it", provider="p", model="m"
        )
        draft.validate()  # passes: nothing static is wrong with it
        try:
            draft.run(show_code=False)
        except UnsupportedInputError as failure:
            draft.repair(failure, verbose=False)
        else:  # pragma: no cover - the call above is expected to fail
            pytest.fail("expected the ambiguous-axis failure")
        assert "AmbiguousAxisError" in spy["question"]

    def test_working_code_has_nothing_to_repair(self, spy: dict[str, Any]) -> None:
        good = Suggestion(
            code="plot(sin(x))", question="q", provider="p", model="m"
        )
        with pytest.raises(UnsupportedInputError, match="nothing to repair"):
            good.repair()

    def test_it_returns_a_draft_rather_than_running_it(
        self, spy: dict[str, Any]
    ) -> None:
        """The loop is deliberately not closed: retrying is the part worth
        automating, skipping the look is not."""
        repaired = Suggestion(
            code="import os", question="q", provider="p", model="m"
        ).repair(verbose=False)
        assert isinstance(repaired, Suggestion)


class TestFitEvidence:
    @pytest.mark.parametrize(
        "family,values,expected",
        [
            ("linear", lambda xs: 2 * xs + 1, "y ~ x  (linear/polynomial)"),
            ("exponential", lambda xs: 2 * np.exp(0.7 * xs), "log(y) ~ x  (exponential)"),
            ("power", lambda xs: 3 * xs**2, "log(y) ~ log(x)  (power law)"),
            ("logarithmic", lambda xs: 2 * np.log(xs) + 1, "y ~ log(x)  (logarithmic)"),
        ],
    )
    def test_the_transform_that_straightens_it_names_the_family(
        self, family: str, values: Any, expected: str
    ) -> None:
        """This is the whole argument for one model over another, and it is a
        measurement rather than an opinion."""
        xs = np.linspace(1.0, 5.0, 20)
        evidence = fit_evidence(dataset({"x": xs, "y": values(xs)}))
        best = max(
            (k for k, v in evidence["straightness"].items() if v is not None),
            key=lambda k: evidence["straightness"][k]["r"],
        )
        assert best == expected
        assert evidence["straightness"][best]["r"] == pytest.approx(1.0, abs=1e-3)

    def test_a_transform_that_drops_rows_says_how_many_it_kept(self) -> None:
        """`log` drops every non-positive value, so on data crossing zero the
        exponential figure describes the positive tail. A correlation without
        its row count would offer that tail's shape as the dataset's."""
        xs = np.linspace(1.0, 5.0, 20)
        evidence = fit_evidence(dataset({"x": xs, "y": 2 * xs - 6}))
        exponential = evidence["straightness"]["log(y) ~ x  (exponential)"]
        assert exponential["rows_used"] < exponential["of"] == 20
        assert evidence["y_all_positive"] is False
        # The honest answer is still there: the data really is a line.
        assert evidence["straightness"]["y ~ x  (linear/polynomial)"] == {
            "r": 1.0, "rows_used": 20, "of": 20
        }

    def test_a_constant_column_measures_nothing_rather_than_nan(self) -> None:
        evidence = fit_evidence(dataset({"x": [1.0, 2.0, 3.0, 4.0], "y": [5.0] * 4}))
        assert evidence["straightness"]["y ~ x  (linear/polynomial)"] is None

    def test_too_few_rows_is_refused_rather_than_correlated(self) -> None:
        with pytest.raises(UnsupportedInputError, match="too few"):
            fit_evidence(dataset({"x": [1.0, 2.0], "y": [1.0, 2.0]}))

    def test_the_evidence_is_json_serialisable(self) -> None:
        xs = np.linspace(1.0, 5.0, 10)
        json.dumps(fit_evidence(dataset({"x": xs, "y": xs**2})))

    def test_columns_can_be_named(self) -> None:
        data = dataset({"t": [1.0, 2.0, 3.0, 4.0], "v": [2.0, 4.0, 6.0, 8.0]})
        assert fit_evidence(data, x="t", y="v")["columns"] == {"x": "t", "y": "v"}
        with pytest.raises(UnsupportedInputError, match="not a column"):
            fit_evidence(data, x="nope")

    def test_something_that_is_not_a_dataset_is_refused(self) -> None:
        with pytest.raises(UnsupportedInputError, match="build one with dataset"):
            fit_evidence([1.0, 2.0, 3.0, 4.0])


class TestSuggestModel:
    def test_it_returns_runnable_fit_code(self, spy: dict[str, Any]) -> None:
        spy["reply"] = (
            "Exponential.\n```python\na, b = symbols('a b')\ndata.fit(a*exp(b*x))\n```"
        )
        xs = np.linspace(1.0, 5.0, 20)
        suggestion = suggest_model(
            dataset({"x": xs, "y": 2 * np.exp(0.7 * xs)}), verbose=False
        )
        assert "data.fit(a*exp(b*x))" in suggestion.code

    def test_the_measurements_are_what_is_sent(self, spy: dict[str, Any]) -> None:
        xs = np.linspace(1.0, 5.0, 20)
        suggest_model(dataset({"x": xs, "y": 2 * np.exp(0.7 * xs)}), verbose=False)
        assert "straightness" in spy["question"]
        assert "rows_used" in spy["question"]

    def test_the_model_is_told_to_check_the_row_count(self, spy: dict[str, Any]) -> None:
        xs = np.linspace(1.0, 5.0, 20)
        suggest_model(dataset({"x": xs, "y": xs + 1}), verbose=False)
        assert '"rows_used" against "of"' in spy["system"]

    def test_the_model_may_not_state_a_fit_it_has_not_run(
        self, spy: dict[str, Any]
    ) -> None:
        xs = np.linspace(1.0, 5.0, 20)
        suggest_model(dataset({"x": xs, "y": xs + 1}), verbose=False)
        assert "Do not state an r-squared" in spy["system"]

    def test_the_dataset_variable_name_is_used_in_the_code(
        self, spy: dict[str, Any]
    ) -> None:
        """The code runs in the caller's namespace, so it has to name the
        dataset the way the caller does."""
        xs = np.linspace(1.0, 5.0, 20)
        suggest_model(dataset({"x": xs, "y": xs + 1}), name="readings", verbose=False)
        assert "`readings`" in spy["question"]
        assert "readings.fit(...)" in spy["system"]

    def test_a_name_that_is_not_a_variable_is_refused(self) -> None:
        xs = np.linspace(1.0, 5.0, 20)
        with pytest.raises(UnsupportedInputError, match="name must be"):
            suggest_model(dataset({"x": xs, "y": xs}), name="data; import os")
