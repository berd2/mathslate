"""Build the guided tour notebooks from one list of cells.

There are two tour notebooks — ``mathslate_tour.py`` for marimo and
``mathslate_tour.ipynb`` for Jupyter — and they have to be the same notebook or
they are worse than useless. So neither is written by hand: the cells live here
once, this script emits the marimo file, and ``marimo export ipynb`` derives the
Jupyter one from that.

The marimo cell signatures (``def _(a, b):`` … ``return (c,)``) are computed by
reading each cell with the same AST rules marimo uses, rather than being typed
out and hoped over. Getting one wrong is a dataflow error the reader would meet
instead of a graph.

Run it from the project root::

    python examples/build_tour.py
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from textwrap import dedent, indent

HERE = Path(__file__).resolve().parent
MARIMO_FILE = HERE / "mathslate_3d_tour.py"
JUPYTER_FILE = HERE / "mathslate_3d_tour.ipynb"

MARIMO_VERSION = "0.23.15"


def md(text: str) -> str:
    """A prose cell."""
    body = dedent(text).strip()
    return f'mo.md(\n    r"""\n{indent(body, "    ")}\n    """\n)'


# --------------------------------------------------------------------------
# the tour
# --------------------------------------------------------------------------

CELLS: list[str] = [
    """
    import marimo as mo
    import numpy as np
    """,
    md(
        """
        # MathSlate 3D Plotting Tour

        This notebook is an advanced exploration of 3D plotting with MathSlate.
        It covers surfaces, contours, implicit 3D curves, space curves, and interactive 3D plots.
        """
    ),
    """
    from mathslate import (
        plot, polar, analyze, table, slider, animate, dataset, show_python,
        set_verbose, get_verbose, frontend_report,
        sin, cos, tan, exp, log, sqrt, Abs, floor, sign, pi, Eq, Matrix,
        diff, integrate, limit, solve, simplify, symbols, Rational,
        x, y, z, t, n, k, theta,
    )
    from mathslate.ui import release, release_all
    import mathslate

    print(f"MathSlate {mathslate.__version__} · {frontend_report()}")
    """,
    md(
        """
        ## 1. Simple Surfaces

        Two free symbols create a 3D surface by default.
        """
    ),
    "plot(x**2 - y**2, (x, -5, 5), (y, -5, 5), title='Saddle')",
    "plot(x * exp(-(x**2 + y**2)), (x, -3, 3), (y, -3, 3), title='Gaussian Peak and Dip')",
    "plot(sin(x) + cos(y), (x, -10, 10), (y, -10, 10), title='Sin-Cos Surface')",
    md(
        """
        You can explicitly set the range for the two variables.
        """
    ),
    "plot(sin(sqrt(x**2 + y**2)), (x, -10, 10), (y, -10, 10), title='Ripples')",
    md(
        """
        ## 2. Contours

        The same objects can be drawn flat with `kind=\"contour\"`.
        """
    ),
    "plot(x**2 - y**2, kind='contour', title='Saddle (Contour)')",
    "plot(sin(sqrt(x**2 + y**2)), (x, -10, 10), (y, -10, 10), kind='contour', title='Ripples (Contour)')",
    md(
        """
        ## 3. Parametric Surfaces

        A three-component tuple with two shared symbols forms a parametric surface.
        For example, a Torus:
        """
    ),
    "u, v = symbols('u v', real=True)",
    """
    R, r = 3, 1
    plot(
        ((R + r*cos(v))*cos(u), (R + r*cos(v))*sin(u), r*sin(v)),
        (u, 0, 2*pi), (v, 0, 2*pi),
        title='Torus'
    )
    """,
    md(
        """
        And a Moebius strip:
        """
    ),
    """
    plot(
        ((1 + v/2 * cos(u/2)) * cos(u), (1 + v/2 * cos(u/2)) * sin(u), v/2 * sin(u/2)),
        (u, 0, 2*pi), (v, -1, 1),
        title='Moebius Strip'
    )
    """,
    md(
        """
        ## 4. Space Curves

        A three-component tuple with one shared symbol forms a 3D space curve.
        """
    ),
    "plot((sin(t), cos(t), t), (t, 0, 20), title='3D Spiral')",
    "plot((cos(t)*(3 + cos(5*t)), sin(t)*(3 + cos(5*t)), sin(5*t)), (t, 0, 2*pi), title='Trefoil Knot-like')",
    md(
        """
        ## 5. Interactive 3D Surfaces

        We can use sliders to interact with 3D surfaces!
        Let's add a frequency multiplier to the ripple effect.
        """
    ),
    """
    release_all()
    freq = slider(0.5, 3.0, default=1.0, name='freq')
    plot(sin(freq * sqrt(x**2 + y**2)), (x, -10, 10), (y, -10, 10), title='Interactive Ripples')
    """,
    md(
        """
        Let's animate a phase shift on the ripple:
        """
    ),
    """
    phase = slider(0, 2*pi, default=0, name='phase')
    animate(sin(sqrt(x**2 + y**2) - phase), (x, -10, 10), (y, -10, 10), title='Animated Ripples')
    """
]


# --------------------------------------------------------------------------
# marimo cell signatures, computed rather than typed
# --------------------------------------------------------------------------


def analyse(code: str) -> tuple[set[str], set[str]]:
    """``(defined, referenced)`` for one cell, marimo's rules in miniature."""
    tree = ast.parse(dedent(code))
    defined: set[str] = set()
    referenced: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                referenced.add(node.id)
            else:
                defined.add(node.id)
        elif isinstance(node, ast.alias):
            defined.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    return defined, referenced


def check_order(parsed: list[tuple[set[str], set[str]]]) -> None:
    """No cell may use a name a later cell defines.

    marimo would resolve a forward reference from its graph, but the Jupyter
    export runs top to bottom and would raise `NameError`. Requiring one order
    to satisfy both is what keeps the two notebooks the same notebook.
    """
    first_defined: dict[str, int] = {}
    for index, (defined, _) in enumerate(parsed):
        for name in defined:
            first_defined.setdefault(name, index)

    for index, (_, referenced) in enumerate(parsed):
        for name in sorted(referenced):
            where = first_defined.get(name)
            if where is not None and where > index:
                raise SystemExit(
                    f"cell {index} uses {name!r}, which cell {where} defines. "
                    "Move the definition earlier — the Jupyter twin runs in "
                    "file order."
                )


def build_marimo() -> str:
    parsed = [analyse(cell) for cell in CELLS]
    check_order(parsed)

    available: dict[str, int] = {}
    parameters: list[list[str]] = []
    for index, (defined, referenced) in enumerate(parsed):
        parameters.append(sorted(name for name in referenced if name in available))
        for name in defined:
            available.setdefault(name, index)

    # A name is returned only if a later cell asks for it.
    returns: list[list[str]] = []
    for index, (defined, _) in enumerate(parsed):
        wanted = {
            name
            for name in defined
            if any(name in parsed[later][1] for later in range(index + 1, len(parsed)))
            and available.get(name) == index
        }
        returns.append(sorted(wanted))

    lines = [
        "# Generated by examples/build_tour.py — edit the cells there, not here.",
        "import marimo",
        "",
        f'__generated_with = "{MARIMO_VERSION}"',
        'app = marimo.App(width="medium")',
        "",
    ]
    for cell, params, produced in zip(CELLS, parameters, returns):
        signature = ", ".join(params)
        lines.append("")
        lines.append("@app.cell")
        lines.append(f"def _({signature}):")
        lines.append(indent(dedent(cell).strip(), "    "))
        if produced:
            body = ", ".join(produced)
            lines.append(f"    return ({body},)" if len(produced) == 1 else f"    return ({body})")
        else:
            lines.append("    return")
        lines.append("")
    lines += ["", 'if __name__ == "__main__":', "    app.run()", ""]
    return "\n".join(lines)


def main() -> int:
    MARIMO_FILE.write_text(build_marimo(), encoding="utf-8")
    print(f"wrote {MARIMO_FILE.relative_to(HERE.parent)} ({len(CELLS)} cells)")

    completed = subprocess.run(
        [
            sys.executable, "-m", "marimo", "export", "ipynb",
            str(MARIMO_FILE), "-o", str(JUPYTER_FILE), "--sort", "top-down",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        print(completed.stdout, completed.stderr, sep="\n")
        return completed.returncode

    _name_the_kernel()
    print(f"wrote {JUPYTER_FILE.relative_to(HERE.parent)}")
    return 0


def _name_the_kernel() -> None:
    """Give the exported notebook a kernelspec.

    ``marimo export ipynb`` leaves the metadata empty, and a notebook with no
    kernelspec makes Jupyter stop and ask which kernel to use before the reader
    has seen anything. Naming the ordinary Python 3 kernel removes the prompt.
    """
    import json

    notebook = json.loads(JUPYTER_FILE.read_text(encoding="utf-8"))
    notebook.setdefault("metadata", {}).update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        }
    )
    JUPYTER_FILE.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
