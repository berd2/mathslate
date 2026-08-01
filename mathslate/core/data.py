"""``dataset()`` — the bridge from symbolic to data (PRD 7, v1.0).

Everything before this milestone starts from an expression. Real work usually
starts from measurements, and the interesting question is how the two meet.

That meeting point is :meth:`Dataset.fit`. You write the model the way you
would write it on paper — ``a*x + b``, ``a*exp(b*x)`` — and get back **the same
expression with its parameters filled in**, still a SymPy object. So the result
of fitting data can be differentiated, solved, analysed and plotted by every
part of MathSlate that came before, without conversion and without leaving the
symbolic world. That is the bridge; the rest of this module is the plumbing
that gets you to it.

Pure Python, NumPy and SymPy. No Plotly, no frontend (PRD 6.1).
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from ..errors import UnsupportedInputError
from ._failure import EVALUATION_FAILURE
from .binding import free_symbols_of, sort_by_convention

__all__ = [
    "Dataset",
    "FitResult",
    "load_dataset",
    "read_csv",
    "FALLBACK_ENCODINGS",
]


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

Array = NDArray[np.float64]

#: Levenberg–Marquardt iterations for the non-linear fit. Enough for the
#: models a learner writes; the linear cases never reach the loop at all.
_MAX_STEPS: int = 200


@dataclass(frozen=True)
class FitResult:
    """A fitted model, still symbolic (PRD 4's escape-hatch rule).

    ``expr`` is the model with its parameters substituted, so it is an ordinary
    SymPy expression: ``plot(fit.expr)``, ``diff(fit.expr, x)`` and
    ``analyze(fit.expr)`` all work on it unchanged.
    """

    expr: sp.Expr
    model: sp.Expr
    parameters: dict[sp.Symbol, float]
    symbol: sp.Symbol
    residuals: Array
    #: Coefficient of determination. ``None`` when the target is constant, where
    #: R² is not defined rather than being 1 or 0.
    r_squared: float | None

    @property
    def sympy(self) -> sp.Expr:
        return self.expr

    def describe(self) -> str:
        found = ", ".join(
            f"{symbol.name} = {value:.6g}" for symbol, value in self.parameters.items()
        )
        quality = "R² undefined (the data is constant)"
        if self.r_squared is not None:
            quality = f"R² = {self.r_squared:.4f}"
        return f"{sp.sstr(self.model)}  ->  {found}   [{quality}]"

    def __repr__(self) -> str:
        return f"<FitResult {sp.sstr(self.expr)}>"


@dataclass(frozen=True)
class Dataset:
    """Named columns of numbers, and the doors from them into SymPy."""

    #: Read-only after construction; see ``__post_init__``.
    columns: Mapping[str, Array]

    # -- construction ------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.columns:
            raise UnsupportedInputError("a dataset needs at least one column.")
        sizes = {name: values.size for name, values in self.columns.items()}
        if len(set(sizes.values())) > 1:
            listed = ", ".join(f"{name}={size}" for name, size in sizes.items())
            raise UnsupportedInputError(
                f"every column must be the same length; got {listed}."
            )
        if next(iter(sizes.values())) == 0:
            raise UnsupportedInputError("a dataset needs at least one row.")

        # `frozen=True` only stops the *attribute* being rebound; the dict and
        # its arrays were still writable, so the checks above could be voided a
        # line later. Take a copy — the caller's array stays theirs — and hand
        # back views nobody can write through.
        frozen: dict[str, Array] = {}
        for name, values in self.columns.items():
            copied = np.array(values, dtype=np.float64, copy=True)
            copied.setflags(write=False)
            frozen[name] = copied
        object.__setattr__(self, "columns", MappingProxyType(frozen))

    # -- shape -------------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.columns)

    def __len__(self) -> int:
        return int(next(iter(self.columns.values())).size)

    def __getitem__(self, name: str) -> Array:
        if name not in self.columns:
            available = ", ".join(self.names)
            raise UnsupportedInputError(
                f"no column named {name!r}; this dataset has: {available}."
            )
        return self.columns[name]

    def __contains__(self, name: object) -> bool:
        return name in self.columns

    # -- escape hatches ----------------------------------------------------

    @property
    def numpy(self) -> tuple[tuple[str, ...], Array]:
        """``(names, values)`` with values shaped ``(rows, columns)``."""
        return self.names, np.column_stack([self.columns[n] for n in self.names])

    # -- statistics --------------------------------------------------------

    def summary(self, name: str) -> dict[str, float]:
        """The five-number summary plus mean and standard deviation."""
        finite = self[name][np.isfinite(self[name])]
        if finite.size == 0:
            return {}
        quartiles = np.percentile(finite, [25.0, 50.0, 75.0])
        return {
            "count": float(finite.size),
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "min": float(finite.min()),
            "25%": float(quartiles[0]),
            "50%": float(quartiles[1]),
            "75%": float(quartiles[2]),
            "max": float(finite.max()),
        }

    def describe(self) -> str:
        """Every column's summary, as aligned text."""
        keys = ("count", "mean", "std", "min", "25%", "50%", "75%", "max")
        rows = [(name, self.summary(name)) for name in self.names]
        width = max(len(n) for n in self.names)
        header = "".rjust(width) + "".join(k.rjust(12) for k in keys)
        lines = [header]
        for name, stats in rows:
            if not stats:
                lines.append(name.rjust(width) + "  (no finite values)")
                continue
            lines.append(
                name.rjust(width) + "".join(f"{stats[k]:12.6g}" for k in keys)
            )
        return "\n".join(lines)

    # -- the bridge --------------------------------------------------------

    def fit(
        self,
        model: sp.Expr,
        x: str | None = None,
        y: str | None = None,
        symbol: sp.Symbol | None = None,
        guess: Mapping[sp.Symbol, float] | None = None,
    ) -> FitResult:
        """Fit ``model`` to two columns and return it **still symbolic**.

        ``model`` is written the way it would be on paper — ``a*x + b`` — where
        one symbol is the independent variable and the rest are the parameters
        to solve for. The variable is the one named by ``symbol``, or the one
        matching the ``x`` column's name, or the most axis-like by PRD 5.2's
        convention.

        A model linear in its parameters is solved exactly by least squares.
        Anything else is refined by damped least squares from ``guess`` (or 1.0),
        and the residuals come back either way so the fit can be judged rather
        than trusted.
        """
        names = self.names
        x_name = x or names[0]
        y_name = y or (names[1] if len(names) > 1 else names[0])
        if x_name == y_name:
            # Fitting a column against itself always "succeeds" with R² = 1 and
            # means nothing. It is never what was intended, so say so.
            raise UnsupportedInputError(
                f"the independent and dependent columns are both {x_name!r}, so "
                "the fit would just recover the identity. "
                + (
                    f"This dataset has only one column; give fit() two, or name "
                    f"them: fit(model, x='...', y='...')."
                    if len(names) == 1
                    else f"Name them: fit(model, x=..., y=...) from {', '.join(names)}."
                )
            )
        inputs, targets = self[x_name], self[y_name]

        variable = _fit_variable(model, symbol, x_name)
        parameters = sort_by_convention(model.free_symbols - {variable})
        if not parameters:
            raise UnsupportedInputError(
                f"{sp.sstr(model)} has no free parameters to fit; every symbol "
                f"other than {variable.name} would have to be solved for."
            )

        keep = np.isfinite(inputs) & np.isfinite(targets)
        if int(np.count_nonzero(keep)) < len(parameters):
            raise UnsupportedInputError(
                f"{int(np.count_nonzero(keep))} usable rows cannot determine "
                f"{len(parameters)} parameters."
            )
        inputs, targets = inputs[keep], targets[keep]

        values = _solve_fit(model, variable, parameters, inputs, targets, guess)
        fitted = model.subs({s: sp.Float(v) for s, v in values.items()})
        predicted = _evaluate(fitted, variable, inputs)
        invalid = int(np.count_nonzero(~np.isfinite(predicted)))
        if invalid:
            raise UnsupportedInputError(
                f"the fitted model is undefined for {invalid} of the "
                f"{inputs.size} usable rows; try a model whose domain covers "
                "the data."
            )
        residuals = targets - predicted
        return FitResult(
            expr=fitted,
            model=model,
            parameters=values,
            symbol=variable,
            residuals=residuals,
            r_squared=_r_squared(targets, residuals),
        )

    def _repr_html_(self) -> str:
        """The summary as a table, for a notebook cell.

        Without this a dataset displayed as `<Dataset 6 rows: x, y>` — true,
        and useless next to a library whose other objects all draw themselves.
        """
        keys = ("count", "mean", "std", "min", "25%", "50%", "75%", "max")
        head = "".join(f"<th style='padding:2px 10px'>{key}</th>" for key in keys)
        body = ""
        for name in self.names:
            stats = self.summary(name)
            cells = (
                "".join(
                    f"<td style='padding:2px 10px;text-align:right'>"
                    f"<code>{stats[key]:.6g}</code></td>"
                    for key in keys
                )
                if stats
                else f"<td colspan='{len(keys)}' style='opacity:0.7'>no finite values</td>"
            )
            body += (
                f"<tr><th style='padding:2px 10px;text-align:left'>"
                f"{_escape(name)}</th>{cells}</tr>"
            )
        return (
            f"<div style='opacity:0.6;font-size:0.9em;padding:2px 8px'>"
            f"dataset · {len(self)} rows</div>"
            f"<table><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>"
        )

    def __repr__(self) -> str:
        return f"<Dataset {len(self)} rows: {', '.join(self.names)}>"


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


def _fit_variable(
    model: sp.Expr, symbol: sp.Symbol | None, x_name: str
) -> sp.Symbol:
    free = free_symbols_of([model])
    if symbol is not None:
        if symbol not in free:
            raise UnsupportedInputError(
                f"{symbol.name} is not in {sp.sstr(model)}."
            )
        return symbol
    named = [s for s in free if s.name == x_name]
    if named:
        return named[0]
    if not free:
        raise UnsupportedInputError(f"{sp.sstr(model)} has no symbols to fit.")
    return sort_by_convention(free)[0]


def _solve_fit(
    model: sp.Expr,
    variable: sp.Symbol,
    parameters: Sequence[sp.Symbol],
    inputs: Array,
    targets: Array,
    guess: Mapping[sp.Symbol, float] | None,
) -> dict[sp.Symbol, float]:
    """Least squares where the model allows it, damped iteration where it does not."""
    basis = _linear_basis(model, variable, parameters)
    if basis is not None:
        design = np.column_stack(
            [_evaluate(term, variable, inputs) for term in basis[:-1]]
        )
        offset = _evaluate(basis[-1], variable, inputs)
        usable = np.isfinite(offset) & np.isfinite(design).all(axis=1)
        invalid = int(inputs.size - np.count_nonzero(usable))
        if invalid:
            raise UnsupportedInputError(
                f"the model is undefined for {invalid} of the {inputs.size} "
                "usable rows, so its parameters cannot be fitted."
            )
        solution, _residuals, rank, _singular = np.linalg.lstsq(
            design, targets - offset, rcond=None
        )
        if int(rank) < len(parameters):
            raise UnsupportedInputError(
                "the model parameters cannot be determined independently from "
                "these rows; the fit is rank-deficient."
            )
        return {symbol: float(value) for symbol, value in zip(parameters, solution)}
    return _levenberg_marquardt(model, variable, parameters, inputs, targets, guess)


def _linear_basis(
    model: sp.Expr, variable: sp.Symbol, parameters: Sequence[sp.Symbol]
) -> list[sp.Expr] | None:
    """``[f1, ..., fn, c]`` when ``model == sum(p_i * f_i) + c``, else ``None``.

    Linearity *in the parameters* is what matters, not in the variable:
    ``a*x**2 + b*x + c`` is linear here and has an exact answer, while
    ``a*exp(b*x)`` is not and needs iteration.
    """
    terms: list[sp.Expr] = []
    remainder = sp.expand(model)
    for parameter in parameters:
        coefficient = remainder.coeff(parameter, 1)
        if coefficient.has(*parameters):
            return None
        terms.append(coefficient)
        remainder = sp.expand(remainder - parameter * coefficient)
    if remainder.has(*parameters):
        return None
    terms.append(remainder)
    return terms


def _levenberg_marquardt(
    model: sp.Expr,
    variable: sp.Symbol,
    parameters: Sequence[sp.Symbol],
    inputs: Array,
    targets: Array,
    guess: Mapping[sp.Symbol, float] | None,
) -> dict[sp.Symbol, float]:
    """Damped least squares. The damping is the whole point.

    Undamped Gauss–Newton takes the full linearised step, which for a model
    with a pole in it — ``a/(x + b)`` — routinely throws ``b`` straight past
    the pole on the first iteration and never comes back. Damping interpolates
    between that step and a short gradient step, and only keeps a step that
    actually lowered the cost, so a bad linearisation costs an iteration
    instead of the fit.

    The model and its derivatives are lambdified **once**, with the parameters
    as arguments, rather than re-substituted and re-compiled every iteration.
    """
    evaluate = _compile(model, variable, parameters)
    gradients = [_compile(sp.diff(model, p), variable, parameters) for p in parameters]

    # Damping keeps a step from bolting; it cannot rescue a start in the wrong
    # basin. `a*exp(b*x)` from b = -3 descends monotonically into a flat region
    # and reports the flat region. So try a handful of deterministic starts and
    # keep the best — each run is a few milliseconds now that the model is
    # compiled once.
    spread = _cost_variance(targets)
    starts = _starting_points(parameters, targets, guess)
    best: np.ndarray | None = None
    best_cost = math.inf
    deficient = 0
    for start in starts:
        try:
            found, cost = _descend(
                evaluate, gradients, inputs, targets, start, parameters
            )
        except _RankDeficient:
            # Deficient *here* may just be this starting point; only deficient
            # everywhere means the parameters genuinely cannot be separated.
            deficient += 1
            continue
        if cost < best_cost:
            best_cost, best = cost, found
        if best_cost <= 1e-12 * max(spread, 1.0):
            break  # already exact; further starts cannot improve on it

    if deficient == len(starts):
        raise UnsupportedInputError(
            "the model parameters cannot be determined independently from "
            "these rows; the fit Jacobian is rank-deficient."
        )
    if best is None or not np.isfinite(best_cost):
        raise UnsupportedInputError(
            "the nonlinear fit produced no finite prediction; try a different "
            "model or starting guess."
        )
    return {symbol: float(value) for symbol, value in zip(parameters, best)}


class _RankDeficient(Exception):
    """The Jacobian could not separate the parameters at this starting point."""


def _cost_variance(targets: Array) -> float:
    finite = targets[np.isfinite(targets)]
    if finite.size < 2:
        return 1.0
    return float(np.sum((finite - finite.mean()) ** 2))


def _starting_points(
    parameters: Sequence[sp.Symbol],
    targets: Array,
    guess: Mapping[sp.Symbol, float] | None,
) -> list[np.ndarray]:
    """Deterministic starts, the caller's first.

    Nothing clever: a couple of magnitudes drawn from the data and both signs
    of the leading parameter, which between them cover growth, decay and the
    ordinary "all ones" case that most textbook models want.
    """
    count = len(parameters)
    finite = targets[np.isfinite(targets)]
    scale = float(np.median(np.abs(finite))) if finite.size else 1.0
    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0

    starts: list[np.ndarray] = []
    if guess:
        starts.append(
            np.array([float(guess.get(s, 1.0)) for s in parameters], dtype=np.float64)
        )
    for lead in (1.0, scale, -scale):
        for rest in (1.0, -1.0, 0.1):
            point = np.full(count, rest, dtype=np.float64)
            point[0] = lead
            starts.append(point)

    unique: list[np.ndarray] = []
    for point in starts:
        if not any(np.allclose(point, seen) for seen in unique):
            unique.append(point)
    return unique


def _descend(
    evaluate: "_Compiled",
    gradients: Sequence["_Compiled"],
    inputs: Array,
    targets: Array,
    start: np.ndarray,
    parameters: Sequence[sp.Symbol],
) -> tuple[np.ndarray, float]:
    """One damped descent from ``start``. Returns ``(parameters, cost)``."""
    current = start.copy()
    best = current.copy()
    best_cost = _cost(targets - evaluate(inputs, current))
    damping = 1e-3

    for iteration in range(_MAX_STEPS):
        residual = targets - evaluate(inputs, current)
        jacobian = np.column_stack([g(inputs, current) for g in gradients])
        usable = np.isfinite(residual) & np.isfinite(jacobian).all(axis=1)
        if int(np.count_nonzero(usable)) < len(parameters):
            break

        rows, values = jacobian[usable], residual[usable]
        normal = rows.T @ rows
        gradient = rows.T @ values
        if not np.isfinite(normal).all() or not np.isfinite(gradient).all():
            break
        if iteration == 0 and int(np.linalg.matrix_rank(rows)) < len(parameters):
            # Rank-deficient before a single step. In `a*b*x` only the product
            # moves the curve, so infinitely many (a, b) fit equally well and
            # returning one of them would be a lie — but that verdict belongs
            # to the caller, which knows whether *every* start said the same.
            #
            # Later iterations are never structural: they are only where the
            # search happens to be standing, and the damping term already keeps
            # the normal equations solvable. Raising there used to throw away a
            # perfectly good fit the iteration had already found.
            raise _RankDeficient

        scale = np.diag(np.maximum(np.diag(normal), 1e-12))
        try:
            step = np.linalg.solve(normal + damping * scale, gradient)
        except np.linalg.LinAlgError:
            damping *= 10.0
            if damping > 1e12:
                break
            continue
        if not np.isfinite(step).all():
            break

        candidate = current + step
        cost = _cost(targets - evaluate(inputs, candidate))
        if np.isfinite(cost) and cost < best_cost:
            best_cost, best = cost, candidate.copy()
            current = candidate
            damping = max(damping / 10.0, 1e-12)
            if float(np.abs(step).max()) < 1e-12 * max(float(np.abs(current).max()), 1.0):
                break
        else:
            # The step made things worse: trust the linearisation less and
            # retry from where we were.
            damping *= 10.0
            if damping > 1e12:
                break

    return best, best_cost


def _cost(residual: Array) -> float:
    if not np.isfinite(residual).all():
        return math.inf
    return float(np.sum(residual**2))


def _compile(
    expr: sp.Expr, variable: sp.Symbol, parameters: Sequence[sp.Symbol]
) -> "_Compiled":
    return _Compiled(expr, variable, parameters)


class _Compiled:
    """``f(inputs, parameter_values) -> array``, lambdified once."""

    def __init__(
        self, expr: sp.Expr, variable: sp.Symbol, parameters: Sequence[sp.Symbol]
    ) -> None:
        self._function = sp.lambdify((variable, *parameters), expr, modules="numpy")

    def __call__(self, inputs: Array, values: Array) -> Array:
        with np.errstate(all="ignore"):
            try:
                out = np.asarray(self._function(inputs, *values), dtype=np.float64)
            except EVALUATION_FAILURE:  # an undefined point is not an error
                return np.full(inputs.shape, np.nan, dtype=np.float64)
        if out.shape != inputs.shape:
            out = np.broadcast_to(out, inputs.shape)
        return np.asarray(out, dtype=np.float64)


def _evaluate(expr: sp.Expr, symbol: sp.Symbol, values: Array) -> Array:
    function = sp.lambdify(symbol, expr, modules="numpy")
    with np.errstate(all="ignore"):
        try:
            out = np.asarray(function(values), dtype=np.float64)
        except EVALUATION_FAILURE:  # fall back point by point
            out = np.array(
                [_scalar(function, float(v)) for v in values], dtype=np.float64
            )
    if out.shape != values.shape:
        out = np.broadcast_to(out, values.shape)
    return np.asarray(out, dtype=np.float64)


def _scalar(function: object, value: float) -> float:
    try:
        return float(function(value))  # type: ignore[operator]
    except EVALUATION_FAILURE:
        return float("nan")


def _r_squared(targets: Array, residuals: Array) -> float | None:
    finite = np.isfinite(targets) & np.isfinite(residuals)
    if int(np.count_nonzero(finite)) < 2:
        return None
    spread = float(np.sum((targets[finite] - targets[finite].mean()) ** 2))
    if spread <= 0.0:
        # Every target is the same number. R² compares against "predict the
        # mean", and there is nothing to be better than.
        return None
    return float(1.0 - np.sum(residuals[finite] ** 2) / spread)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def load_dataset(
    source: object,
    columns: Sequence[str] | None = None,
    encoding: str | None = None,
) -> Dataset:
    """Build a :class:`Dataset` from a mapping, a CSV path, or arrays."""
    if isinstance(source, Dataset):
        return source
    if isinstance(source, Mapping):
        return Dataset({str(k): _column(v) for k, v in source.items()})
    if isinstance(source, (str, Path)) and _looks_like_path(source):
        return read_csv(source, encoding=encoding)

    array = np.asarray(source, dtype=np.float64)
    if array.ndim == 1:
        names = tuple(columns or ("value",))
        if len(names) != 1:
            raise UnsupportedInputError(
                f"{len(names)} column names for a 1-D dataset; exactly one is required."
            )
        return Dataset({names[0]: array})
    if array.ndim == 2:
        given = tuple(columns or tuple(f"c{i}" for i in range(array.shape[1])))
        if len(given) != array.shape[1]:
            raise UnsupportedInputError(
                f"{len(given)} column names for {array.shape[1]} columns."
            )
        return Dataset({name: array[:, i] for i, name in enumerate(given)})
    raise UnsupportedInputError(
        "a dataset is built from a mapping of columns, a CSV path, or a 1-D or "
        f"2-D array; got {array.ndim} dimensions."
    )


def _looks_like_path(source: str | Path) -> bool:
    text = str(source)
    return text.endswith((".csv", ".tsv", ".txt")) or Path(text).exists()


#: Tried in order when no encoding is given. UTF-8 first because it is the
#: right answer; the rest are what spreadsheets on a non-English system
#: actually write. Excel on a Korean install exports cp949 by default, and a
#: raw ``UnicodeDecodeError`` from three frames down is not a usable answer to
#: "why will my file not open".
FALLBACK_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "cp949", "cp1252", "latin-1")


def read_csv(
    path: str | Path, delimiter: str | None = None, encoding: str | None = None
) -> Dataset:
    """Read a CSV whose first row is the column names.

    Non-numeric cells become ``NaN`` rather than an error: a spreadsheet
    exported by hand routinely has a blank or a note in it, and refusing the
    whole file over one cell helps nobody.

    ``encoding`` is tried alone when given. Otherwise UTF-8 is tried first and
    then the encodings spreadsheets on a non-English system write — see
    :data:`FALLBACK_ENCODINGS`. ``latin-1`` is last and decodes any byte, so a
    file that is not text at all would be accepted as mojibake rather than
    refused; a NUL byte anywhere in the file is taken as the evidence that it is
    binary. UTF-16 has to be named explicitly for that reason.
    """
    location = Path(path)
    if not location.exists():
        raise UnsupportedInputError(f"no such file: {location}")
    text = _decode(location, encoding).splitlines()
    if not text:
        raise UnsupportedInputError(f"{location} is empty.")
    if delimiter is None:
        delimiter = "\t" if location.suffix == ".tsv" else ","

    reader = csv.reader(text, delimiter=delimiter)
    header = next(reader, [])
    names = _header_names(header, location)
    gathered: list[list[float]] = [[] for _ in names]
    for row in reader:
        if not row:
            continue
        for index in range(len(names)):
            cell = row[index].strip() if index < len(row) else ""
            gathered[index].append(_number(cell))
    if not gathered[0]:
        raise UnsupportedInputError(f"{location} has a header but no rows.")
    return Dataset(
        {name: np.array(values, dtype=np.float64) for name, values in zip(names, gathered)}
    )


def _decode(location: Path, encoding: str | None) -> str:
    """Read the file as text, trying the fallbacks unless one was named."""
    raw = location.read_bytes()
    # `latin-1` is the last fallback and decodes *any* byte, which means the
    # walk below can never conclude "this is not a text file". Handed a PNG,
    # it produced a one-column dataset named `\x89PNG` — accepted, plausible
    # looking, and entirely meaningless. A NUL byte is the cheap discriminator:
    # no text CSV in any of these encodings contains one, and essentially every
    # binary format does. (UTF-16 is the honourable exception, and it is named
    # in the message rather than guessed, since guessing it would misread the
    # single-byte encodings that share its byte patterns.)
    # The guard applies only while guessing: it is the *fallback chain* that
    # cannot fail, and an explicitly named encoding is the escape this message
    # points at, so blocking that too would advertise a door and then bolt it.
    if encoding is None and b"\x00" in raw:
        raise UnsupportedInputError(
            f"{location} is not a text file — it contains NUL bytes, which a CSV "
            "does not. If it is really UTF-16 or UTF-32, say so: "
            "read_csv(..., encoding='utf-16')."
        )
    attempts = (encoding,) if encoding else FALLBACK_ENCODINGS
    for candidate in attempts:
        try:
            return raw.decode(candidate)
        except (UnicodeDecodeError, LookupError):
            continue
    listed = ", ".join(str(a) for a in attempts)
    raise UnsupportedInputError(
        f"{location} is not text in any encoding tried ({listed}). "
        "Pass the right one as read_csv(..., encoding=...), or re-save the "
        "file as UTF-8."
    )


def _header_names(header: Sequence[str], location: Path) -> list[str]:
    """Column names from the first row: never empty, never duplicated.

    A duplicate used to silently cost a column, because the names became dict
    keys and the last one won — a three-column file quietly became two.
    Spreadsheets export repeated headers often enough that failing is worse
    than disambiguating, so the repeats are numbered instead.
    """
    if not header:
        raise UnsupportedInputError(
            f"{location} has no header row; the first line must name the columns."
        )
    names: list[str] = []
    used: set[str] = set()
    next_suffix: dict[str, int] = {}
    for index, raw in enumerate(header):
        base = raw.strip() or f"c{index}"
        suffix = next_suffix.get(base, 0)
        name = base if suffix == 0 else f"{base}_{suffix}"
        while name in used:
            suffix += 1
            name = f"{base}_{suffix}"
        used.add(name)
        next_suffix[base] = suffix + 1
        names.append(name)
    return names


def _number(cell: str) -> float:
    try:
        return float(cell)
    except ValueError:
        return float("nan")


def _column(values: Iterable[object]) -> Array:
    return np.asarray(list(values), dtype=np.float64)
