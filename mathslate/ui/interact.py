"""``slider()`` and the parameter it binds (PRD 5.7, 6.2).

    a = slider(-5, 5, default=1)
    plot(a*sin(x))

The PRD's requirement is that **the user never writes a callback binding**, and
that it works in marimo, in Jupyter/Colab and in a plain script alike.

Two things have to be true for that one line to work.

**A slider must be usable inside a SymPy expression.** ``Slider`` is not a
``Symbol`` — it carries a range, a step and a current value — but it answers
``_sympy_()`` with its symbol, which is the hook SymPy's ``sympify`` looks for.
So ``a*sin(x)`` builds an ordinary expression in an ordinary symbol, and every
part of MathSlate downstream stays unaware that a widget was involved.

**``plot()`` must recognise the symbol as a parameter, not an axis.** That is
PRD 5.2 rule 2, and it happens through ``core.binding``'s registry: creating a
slider binds its symbol to a value there, and dispatch folds bound symbols into
``parameters=`` automatically. The registry holds plain floats, so ``core``
never learns what a widget is.

See :func:`values` for why the default control is Plotly's own, and
:meth:`Slider.widget` for the native ones.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterator

import sympy as sp

from ..core import binding
from ..errors import UnsupportedInputError
from .adapters import Frontend, detect_frontend

__all__ = [
    "Slider",
    "slider",
    "sliders_in",
    "release",
    "release_all",
    "DEFAULT_STEPS",
]

#: Frames drawn when a slider does not say how finely to divide its range.
#: Enough to read as continuous, few enough to stay a small HTML file.
DEFAULT_STEPS: int = 21

_COUNTER: Iterator[int] = itertools.count(1)

#: ``symbol -> Slider``. The float half of this lives in ``core.binding``;
#: this half is what remembers there was a widget behind the number.
_REGISTRY: dict[sp.Symbol, "Slider"] = {}


class Slider:
    """A named number the reader can vary, and the symbol standing for it."""

    def __init__(
        self,
        start: float,
        stop: float,
        step: float | None = None,
        *,
        default: float | None = None,
        name: str | None = None,
        label: str | None = None,
    ) -> None:
        start, stop = float(start), float(stop)
        if not stop > start:
            raise UnsupportedInputError(
                f"a slider needs stop greater than start; got ({start}, {stop})."
            )
        if step is not None and not float(step) > 0:
            raise UnsupportedInputError(f"a slider step must be positive; got {step!r}.")

        self.start: float = start
        self.stop: float = stop
        self.step: float | None = None if step is None else float(step)
        self.name: str = name or f"p{next(_COUNTER)}"
        self.label: str = label or self.name
        #: An ordinary SymPy symbol. Everything downstream sees only this.
        self.symbol: sp.Symbol = sp.Symbol(self.name, real=True)

        chosen = self.start if default is None else float(default)
        if not self.start <= chosen <= self.stop:
            raise UnsupportedInputError(
                f"default={chosen} is outside the slider's range "
                f"[{self.start}, {self.stop}]."
            )
        self._value: float = chosen
        binding.bind_parameter(self.symbol, chosen)
        _REGISTRY[self.symbol] = self

    # -- the value ---------------------------------------------------------

    @property
    def value(self) -> float:
        """Where the slider currently sits. Setting it re-binds the symbol."""
        return self._value

    @value.setter
    def value(self, new: float) -> None:
        chosen = float(new)
        if not self.start <= chosen <= self.stop:
            raise UnsupportedInputError(
                f"value={chosen} is outside the slider's range "
                f"[{self.start}, {self.stop}]."
            )
        if _REGISTRY.get(self.symbol) is not self:
            raise UnsupportedInputError(
                f"slider {self.name!r} is no longer active because another "
                "slider with that name replaced it."
            )
        self._value = chosen
        binding.bind_parameter(self.symbol, self._value)

    def values(self) -> tuple[float, ...]:
        """The positions a frame is drawn at.

        MathSlate's default control is Plotly's own slider, built from
        pre-computed frames. PRD 6.2 lists that as the fallback for "other /
        static export", but it is the better *default* for three reasons: it
        behaves identically in all three environments, it needs no frontend
        package (acceptance criterion 6), and it survives being written to a
        single HTML file, which is what PRD 3's teacher hands out.

        The cost is that the positions are decided in advance rather than
        recomputed on demand, which is what :meth:`widget` is for.
        """
        if self.step is not None:
            count = int(math.floor((self.stop - self.start) / self.step + 1e-12))
            positions = [
                self.start + index * self.step for index in range(count + 1)
            ]
            if not math.isclose(positions[-1], self.stop, rel_tol=1e-12, abs_tol=1e-12):
                positions.append(self.stop)
            else:
                positions[-1] = self.stop
            return tuple(positions)
        span = self.stop - self.start
        return tuple(
            self.start + span * index / (DEFAULT_STEPS - 1) for index in range(DEFAULT_STEPS)
        )

    def nearest(self, value: float) -> float:
        """The drawn position closest to ``value``."""
        return min(self.values(), key=lambda candidate: abs(candidate - value))

    # -- SymPy interop -----------------------------------------------------

    def _sympy_(self) -> sp.Symbol:
        """What ``sympify`` calls. This is the whole trick."""
        return self.symbol

    # SymPy handles `a * sin(x)` by sympifying us through `_sympy_`. Plain
    # numbers cannot: `a * 2` asks int, int declines, and Python gives up. So
    # the arithmetic that a reader will actually type is delegated explicitly.
    def __mul__(self, other: Any) -> sp.Expr:
        return self.symbol * sp.sympify(other)

    def __rmul__(self, other: Any) -> sp.Expr:
        return sp.sympify(other) * self.symbol

    def __add__(self, other: Any) -> sp.Expr:
        return self.symbol + sp.sympify(other)

    def __radd__(self, other: Any) -> sp.Expr:
        return sp.sympify(other) + self.symbol

    def __sub__(self, other: Any) -> sp.Expr:
        return self.symbol - sp.sympify(other)

    def __rsub__(self, other: Any) -> sp.Expr:
        return sp.sympify(other) - self.symbol

    def __truediv__(self, other: Any) -> sp.Expr:
        return self.symbol / sp.sympify(other)

    def __rtruediv__(self, other: Any) -> sp.Expr:
        return sp.sympify(other) / self.symbol

    def __pow__(self, other: Any) -> sp.Expr:
        return self.symbol ** sp.sympify(other)

    def __rpow__(self, other: Any) -> sp.Expr:
        return sp.sympify(other) ** self.symbol

    def __neg__(self) -> sp.Expr:
        return -self.symbol

    # -- native widgets ----------------------------------------------------

    def widget(self) -> Any:
        """The host's own slider, for live recomputation instead of frames.

        PRD 6.2's table: ``mo.ui.slider`` in marimo, ``ipywidgets`` in Jupyter
        and Colab. Neither is a dependency — this raises if the host's package
        is absent, and the Plotly control (:meth:`values`) needs neither.
        """
        frontend = detect_frontend()
        if frontend is Frontend.MARIMO:
            return self._marimo_widget()
        if frontend in {Frontend.JUPYTER, Frontend.COLAB}:
            return self._ipywidget()
        raise UnsupportedInputError(
            "no notebook frontend is running, so there is no native widget to "
            "build. The Plotly slider that plot() draws works here already."
        )

    def _marimo_widget(self) -> Any:
        import marimo as mo

        return mo.ui.slider(
            start=self.start,
            stop=self.stop,
            step=self.step or (self.stop - self.start) / (DEFAULT_STEPS - 1),
            value=self.value,
            label=self.label,
        )

    def _ipywidget(self) -> Any:
        try:
            import ipywidgets
        except ImportError as error:  # pragma: no cover - depends on the host
            raise UnsupportedInputError(
                "ipywidgets is not installed. Either `pip install mathslate[jupyter]` "
                "or use the Plotly slider plot() already draws, which needs nothing."
            ) from error

        return ipywidgets.FloatSlider(
            min=self.start,
            max=self.stop,
            step=self.step or (self.stop - self.start) / (DEFAULT_STEPS - 1),
            value=self.value,
            description=self.label,
        )

    # -- display -----------------------------------------------------------

    def __repr__(self) -> str:
        step = "" if self.step is None else f", step={self.step:g}"
        return (
            f"<slider {self.label}: {self.start:g}..{self.stop:g}{step} "
            f"at {self.value:g}>"
        )


def slider(
    start: float,
    stop: float,
    step: float | None = None,
    *,
    default: float | None = None,
    name: str | None = None,
    label: str | None = None,
) -> Slider:
    """A parameter the reader can vary (PRD 5.7).

    ``name`` is the symbol's name, and defaults to ``p1``, ``p2``, … A slider
    cannot see the variable you assign it to, so pass ``name="a"`` when you
    want the control and the emitted code to say ``a``.

    Examples
    --------
    >>> from mathslate import slider, plot, sin, x
    >>> a = slider(-5, 5, default=1, name="a")
    >>> _ = plot(a*sin(x), verbose=False)
    """
    return Slider(start, stop, step, default=default, name=name, label=label)


def release(*sliders: Slider) -> None:
    """Stop treating these sliders' symbols as parameters."""
    for instance in sliders:
        if _REGISTRY.get(instance.symbol) is instance:
            _REGISTRY.pop(instance.symbol, None)
            binding.unbind_parameter(instance.symbol)


def release_all() -> None:
    """Forget every slider in the session.

    A slider binds its symbol until something releases it — that is what lets
    ``plot(a*sin(x))`` know that ``a`` is a parameter without being told again
    each time. The binding therefore outlives the cell that made it, which is
    right in a notebook and wrong in a test, so this exists to draw the line.
    """
    for instance in list(_REGISTRY.values()):
        release(instance)


def sliders_in(*exprs: sp.Expr) -> tuple[Slider, ...]:
    """Every live slider whose symbol appears free in ``exprs``, in order."""
    free: set[sp.Symbol] = set()
    for expr in exprs:
        if isinstance(expr, sp.Basic):
            free |= {s for s in expr.free_symbols if isinstance(s, sp.Symbol)}
    return tuple(s for symbol, s in _REGISTRY.items() if symbol in free)
