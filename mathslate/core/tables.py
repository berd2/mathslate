"""``table()`` — the same function, read as numbers instead of a picture.

A graph shows shape; a table shows value. A learner checking whether their
hand-worked answer is right needs the second, and reaches for it at exactly the
moment a plot stops being enough.

The rules are the ones the rest of MathSlate already follows: the axis is
chosen by PRD 5.2, points outside the real domain are blank rather than wrong,
and the escape hatches are there — ``.numpy`` gives the arrays and ``.sympy``
the expressions.

Pure Python and SymPy. No Plotly, no frontend (PRD 6.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from ..errors import UnsupportedInputError
from ._source import expr_source, import_block
from .sampling import NumericFunction

__all__ = ["Column", "Table", "tabulate", "DEFAULT_ROWS"]

Array = NDArray[np.float64]

#: Rows drawn when the caller does not say. Odd, so a symmetric range has a
#: row exactly at its centre — which is usually the interesting one.
DEFAULT_ROWS: int = 11


@dataclass(frozen=True)
class Column:
    """One expression's worth of values."""

    name: str
    values: Array
    expr: sp.Expr | None = None


@dataclass(frozen=True)
class Table:
    """A table of values, and the doors back out to NumPy and SymPy."""

    symbol: sp.Symbol
    inputs: Array
    columns: tuple[Column, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    # -- escape hatches ----------------------------------------------------

    @property
    def numpy(self) -> tuple[Array, Array]:
        """``(inputs, values)``, values shaped ``(rows, columns)``."""
        return self.inputs, np.column_stack([c.values for c in self.columns])

    @property
    def sympy(self) -> sp.Expr | tuple[sp.Expr, ...] | None:
        exprs = tuple(c.expr for c in self.columns if c.expr is not None)
        if not exprs:
            return None
        return exprs[0] if len(exprs) == 1 else exprs

    # -- reading it --------------------------------------------------------

    def headers(self) -> tuple[str, ...]:
        return (self.symbol.name, *(c.name for c in self.columns))

    def rows(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            (float(value), *(float(c.values[index]) for c in self.columns))
            for index, value in enumerate(self.inputs)
        )

    def text(self) -> str:
        """The table as aligned plain text."""
        headers = self.headers()
        body = [[_cell(v) for v in row] for row in self.rows()]
        widths = [
            max(len(headers[i]), *(len(r[i]) for r in body)) if body else len(headers[i])
            for i in range(len(headers))
        ]
        lines = ["  ".join(h.rjust(w) for h, w in zip(headers, widths))]
        lines.append("  ".join("-" * w for w in widths))
        lines += ["  ".join(c.rjust(w) for c, w in zip(row, widths)) for row in body]
        lines += [f"note: {note}" for note in self.notes]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        head = "".join(f"<th style='padding:2px 10px'>{_escape(h)}</th>" for h in self.headers())
        body = "".join(
            "<tr>"
            + "".join(
                f"<td style='padding:2px 10px;text-align:right'>"
                f"<code>{_escape(_cell(v))}</code></td>"
                for v in row
            )
            + "</tr>"
            for row in self.rows()
        )
        notes = "".join(
            f"<div style='padding:2px 8px;opacity:0.7'>{_escape(n)}</div>"
            for n in self.notes
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{notes}"

    def python(self) -> str:
        """The plain NumPy that builds the same table (PRD 5.5)."""
        name = self.symbol.name
        needed: set[str] = set()
        lines: list[str] = [f"{name} = sp.symbols({name!r}, real=True)"]
        for index, column in enumerate(self.columns):
            if column.expr is None:
                continue
            source, extra = expr_source(column.expr, {name})
            needed |= extra
            lines.append(f"expr{index} = {source}")
            lines.append(f"fn{index} = sp.lambdify({name}, expr{index}, 'numpy')")
        lines += [
            "",
            f"xs = np.linspace({_number(self.inputs[0])}, "
            f"{_number(self.inputs[-1])}, {self.inputs.size})",
        ]
        for index, column in enumerate(self.columns):
            if column.expr is None:
                continue
            lines.append(
                f"col{index} = np.broadcast_to(np.asarray(fn{index}(xs), dtype=float), xs.shape)"
            )
        wanted = [i for i, c in enumerate(self.columns) if c.expr is not None]
        lines += [
            "",
            f"for row in zip(xs, {', '.join(f'col{i}' for i in wanted)}):",
            "    print(*(f'{value:12.6g}' for value in row))",
        ]
        header = [
            "# Equivalent code — this runs exactly as printed.",
            import_block(needed, plotly=False),
            "",
        ]
        return "\n".join(header + lines).rstrip() + "\n"

    def __str__(self) -> str:
        return self.text()

    def __repr__(self) -> str:
        return f"<Table {self.symbol.name}: {len(self.inputs)} rows x {len(self.columns)}>"


def tabulate(
    exprs: Sequence[sp.Expr],
    symbol: sp.Symbol,
    span: tuple[float, float],
    rows: int = DEFAULT_ROWS,
    labels: Sequence[str] | None = None,
) -> Table:
    """Evaluate every expression at ``rows`` evenly spaced points."""
    if rows < 2:
        raise UnsupportedInputError(f"a table needs at least 2 rows; got {rows!r}.")
    lo, hi = float(span[0]), float(span[1])
    if not hi > lo:
        raise UnsupportedInputError(f"empty range: ({lo}, {hi}).")

    inputs = np.linspace(lo, hi, rows, dtype=np.float64)
    columns: list[Column] = []
    blanks = 0
    for index, expr in enumerate(exprs):
        values = NumericFunction(expr, symbol)(inputs)
        blanks += int(np.count_nonzero(~np.isfinite(values)))
        name = labels[index] if labels and index < len(labels) else sp.sstr(expr)
        columns.append(Column(name=name, values=values, expr=expr))

    notes: list[str] = []
    if blanks:
        # Blank, not zero and not an error: the function has no value there,
        # and a table that prints one would be lying about the mathematics.
        notes.append(
            f"{blanks} cell{'s are' if blanks > 1 else ' is'} blank — the "
            "expression is not a real number there."
        )
    return Table(symbol=symbol, inputs=inputs, columns=tuple(columns), notes=tuple(notes))


def _cell(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return f"{value:.6g}"


def _number(value: float) -> str:
    return repr(round(float(value), 12))


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
