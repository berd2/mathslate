"""The non-linear fit must be damped, and must not throw away a good answer.

Undamped Gauss–Newton takes the full linearised step. For ``a/(x + b)`` that
put ``b`` on the far side of the pole on the first iteration; the search never
came back, and 200 iterations later a transient rank-deficiency raised and
discarded the best point it had found. Damping fixes both: a bad linearisation
now costs one iteration instead of the fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from mathslate import dataset, exp, log, sin, symbols, x
from mathslate.errors import UnsupportedInputError

a, b, c = symbols("a b c", real=True)
GRID = np.linspace(0.1, 5.0, 300)


def _fit(values: np.ndarray, model, **kwargs):
    return dataset({"x": GRID, "y": values}).fit(model, **kwargs)


class TestModelsThatNeedDamping:
    @pytest.mark.parametrize(
        ("label", "values", "model", "expected"),
        [
            ("pole", 1.0 / (GRID + 0.5) + 0.2, a / (x + b) + c, {"a": 1.0, "b": 0.5, "c": 0.2}),
            ("exponential", 2.5 * np.exp(0.7 * GRID), a * exp(b * x), {"a": 2.5, "b": 0.7}),
            ("sine", 2.0 * np.sin(1.3 * GRID), a * sin(b * x), {"a": 2.0, "b": 1.3}),
            ("logarithm", 1.5 * np.log(GRID + 2.0), a * log(x + b), {"a": 1.5, "b": 2.0}),
            ("offset decay", 3.0 * np.exp(-0.4 * GRID) + 1.0, a * exp(b * x) + c,
             {"a": 3.0, "b": -0.4, "c": 1.0}),
        ],
    )
    def test_it_recovers_the_true_parameters(
        self, label: str, values: np.ndarray, model, expected: dict[str, float]
    ) -> None:
        fit = _fit(values, model)
        found = {symbol.name: value for symbol, value in fit.parameters.items()}
        for name, want in expected.items():
            assert found[name] == pytest.approx(want, rel=1e-3, abs=1e-6), label
        assert fit.r_squared is not None and fit.r_squared > 0.999

    def test_a_pole_model_no_longer_raises(self) -> None:
        """It used to die with 'rank-deficient' after wandering past the pole."""
        fit = _fit(1.0 / (GRID + 0.5) + 0.2, a / (x + b) + c)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-6)

    def test_noise_does_not_derail_it(self) -> None:
        rng = np.random.default_rng(0)
        noisy = 2.5 * np.exp(0.7 * GRID) * (1.0 + 0.01 * rng.standard_normal(GRID.size))
        fit = _fit(noisy, a * exp(b * x))
        assert fit.r_squared is not None and fit.r_squared > 0.99

    def test_a_deliberately_bad_guess_still_converges(self) -> None:
        fit = _fit(2.5 * np.exp(0.7 * GRID), a * exp(b * x), guess={a: 50.0, b: -3.0})
        assert fit.r_squared is not None and fit.r_squared > 0.99

    def test_a_candidate_may_not_discard_rows_outside_its_domain(self) -> None:
        xs = np.linspace(0.0, 4.0, 9)
        ys = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        fit = dataset({"x": xs, "y": ys}).fit(a * log(x + b) + c)

        assert np.isfinite(fit.residuals).all()
        assert fit.residuals.size == xs.size


class TestWhatStillFails_AndShould:
    def test_structurally_unidentifiable_parameters_are_refused(self) -> None:
        """Only the product a*b moves the curve, so neither can be reported."""
        data = dataset({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
        with pytest.raises(UnsupportedInputError, match="rank-deficient"):
            data.fit(a * b * x)

    def test_a_linear_model_with_no_information_is_refused(self) -> None:
        data = dataset({"x": [0.0, 0.0, 0.0], "y": [1.0, 2.0, 3.0]})
        with pytest.raises(UnsupportedInputError, match="rank-deficient"):
            data.fit(a * x + b)

    def test_too_few_rows_is_refused(self) -> None:
        data = dataset({"x": [1.0], "y": [2.0]})
        with pytest.raises(UnsupportedInputError, match="cannot determine"):
            data.fit(a * exp(b * x))


class TestItIsNotSlow:
    def test_the_model_is_compiled_once_not_per_iteration(self) -> None:
        """The loop used to `subs` and `lambdify` on every step of every gradient."""
        import time

        start = time.perf_counter()
        _fit(2.5 * np.exp(0.7 * GRID), a * exp(b * x) + c)
        assert time.perf_counter() - start < 2.0


class TestLinearModelsAreUntouched:
    @pytest.mark.parametrize(
        ("values", "model", "expected"),
        [
            (3.0 * GRID + 2.0, a * x + b, {"a": 3.0, "b": 2.0}),
            (2.0 * GRID**2 - GRID + 4.0, a * x**2 + b * x + c, {"a": 2.0, "b": -1.0, "c": 4.0}),
        ],
    )
    def test_exact_least_squares_still_applies(self, values, model, expected) -> None:
        found = {s.name: v for s, v in _fit(values, model).parameters.items()}
        for name, want in expected.items():
            assert found[name] == pytest.approx(want, abs=1e-8)

    @pytest.mark.parametrize("model", [a * x**0.5, a * log(x), a / x])
    def test_a_model_undefined_on_input_rows_is_explained(self, model) -> None:
        with pytest.raises(UnsupportedInputError, match="model is undefined"):
            dataset({"x": [-1.0, 0.0, 1.0], "y": [1.0, 1.0, 1.0]}).fit(model)


class TestDataInputEdges:
    def test_a_single_column_is_not_fitted_against_itself(self) -> None:
        data = dataset({"v": np.linspace(1.0, 5.0, 20)})
        with pytest.raises(UnsupportedInputError, match="recover the identity"):
            data.fit(a * x + b)

    def test_naming_one_column_twice_is_refused(self) -> None:
        data = dataset({"p": [1.0, 2.0, 3.0], "q": [2.0, 4.0, 6.0]})
        with pytest.raises(UnsupportedInputError, match="recover the identity"):
            data.fit(a * x + b, x="p", y="p")

    def test_a_header_only_file_is_a_mathslate_error(self, tmp_path) -> None:
        path = tmp_path / "blank.csv"
        path.write_text("\n", encoding="utf-8")
        with pytest.raises(UnsupportedInputError, match="no header row"):
            dataset(path)

    def test_duplicate_headers_keep_every_column(self, tmp_path) -> None:
        """They used to collide as dict keys and one column vanished."""
        path = tmp_path / "dup.csv"
        path.write_text("a,a,b\n1,2,3\n4,5,6\n", encoding="utf-8")
        data = dataset(path)
        assert data.names == ("a", "a_1", "b")
        assert list(data["a_1"]) == [2.0, 5.0]

    def test_generated_header_names_cannot_collide_with_real_ones(
        self, tmp_path
    ) -> None:
        path = tmp_path / "colliding.csv"
        path.write_text("x,x,x_1\n1,2,3\n", encoding="utf-8")

        data = dataset(path)

        assert data.names == ("x", "x_1", "x_1_1")
        assert data.numpy[1].tolist() == [[1.0, 2.0, 3.0]]

    def test_a_1d_array_requires_exactly_one_column_name(self) -> None:
        with pytest.raises(UnsupportedInputError, match="exactly one"):
            dataset(np.array([1.0, 2.0]), columns=["a", "b"])


class TestTheDatasetIsReallyFrozen:
    def test_the_callers_array_is_not_captured(self) -> None:
        source = np.array([1.0, 2.0, 3.0])
        data = dataset({"v": source, "w": [4.0, 5.0, 6.0]})
        source[0] = 99.0
        assert data["v"][0] == 1.0

    def test_a_column_cannot_be_written_through(self) -> None:
        data = dataset({"v": [1.0, 2.0], "w": [3.0, 4.0]})
        with pytest.raises(ValueError):
            data["v"][0] = 5.0

    def test_columns_cannot_be_added_or_removed(self) -> None:
        data = dataset({"v": [1.0, 2.0], "w": [3.0, 4.0]})
        with pytest.raises(TypeError):
            data.columns["z"] = np.array([1.0, 2.0])  # type: ignore[index]

    def test_the_escape_hatch_still_gives_a_usable_copy(self) -> None:
        data = dataset({"v": [1.0, 2.0], "w": [3.0, 4.0]})
        names, values = data.numpy
        values[0, 0] = 99.0  # a copy, so this is allowed and harmless
        assert names == ("v", "w") and data["v"][0] == 1.0
