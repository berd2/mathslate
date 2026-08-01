"""``ask()`` — a question in, MathSlate code out.

The system prompt is built from the dispatch contract itself rather than being
prose written alongside it, so the two cannot drift: if a row is added to
:data:`mathslate.core.dispatch.KINDS`, the assistant is told about it in the
same commit.

PRD goal 5 is "the API is easy for an AI to generate correctly". This module is
where that stops being an aspiration and becomes a measurement — the prompt is
short because the API is small, and that is the whole argument for keeping the
API small.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .._text import safe_print
from ..core._budget import SymbolicTimeout, within_budget
from ..errors import UnsupportedInputError
from .providers import Backend, resolve_provider

__all__ = ["Suggestion", "ask", "configure", "configured", "forget", "system_prompt"]

_FENCE = re.compile(r"```(?:python)?\n(?P<body>.*?)```", re.DOTALL)

#: Set by :func:`configure`; ``ask()`` falls back to auto-detection.
_DEFAULTS: dict[str, Any] = {"provider": None, "model": None, "api_key": None}

# Names the assistant is invited to use and that are safe to expose without the
# rest of Python's builtins or module system. In particular, ``lambdify`` is not
# here: its ``modules=`` argument can import arbitrary modules.
_SAFE_CALLS: frozenset[str] = frozenset(
    {
        "Abs", "Eq", "Float", "Integer", "Matrix", "Piecewise", "Rational",
        "Sum", "Symbol", "acos", "analyze", "animate", "apart", "asin",
        "atan", "binomial", "cancel", "cbrt", "ceiling", "cos", "cosh",
        "cot", "csc", "dataset", "diff", "exp", "expand", "factor",
        "factorial", "floor", "gcd", "integrate", "limit", "log", "nsimplify",
        "nsolve", "plot", "polar", "root", "sec", "series", "show_python",
        "sign", "simplify", "sin", "sinh", "slider", "solve", "solveset",
        "sqrt", "symbols", "table", "tan", "tanh", "together", "trigsimp",
    }
)
_SAFE_METHODS: frozenset[str] = frozenset(
    {
        "analyze", "describe", "fit", "headers", "numpy", "provenance", "python",
        "rows", "show_python", "summary", "sympy", "table", "text",
    }
)
_SAFE_ATTRIBUTES: frozenset[str] = _SAFE_METHODS | frozenset(
    {
        "approximate", "asymptotes", "breakpoints", "coefficients",
        "decreasing", "discontinuities", "figure", "increasing", "inflections",
        "maxima", "minima", "names", "notes", "parameters", "periodicity",
        "plan", "plotly", "r_squared", "residuals", "roots", "symmetry", "value",
    }
)
_SAFE_BUILTINS: dict[str, Any] = {
    "False": False,
    "None": None,
    "True": True,
    "print": print,
}
_MAX_SAFE_CODE_CHARS = 20_000
_MAX_SAFE_AST_NODES = 400
#: Wall-clock budget for one restricted execution, via the same mechanism
#: :mod:`mathslate.core._budget` gives symbolic calls. The AST allowlist bounds
#: *what* restricted code can name, not what a legal expression costs to run —
#: ``9**9**9`` is three ``Constant``/``BinOp`` nodes and passes validation, but
#: computing it exhausts memory. This is the backstop for that gap.
_RUN_BUDGET: Final[float] = 10.0
_ALLOWED_AST_NODES: tuple[type[ast.AST], ...] = (
    ast.Module,
    ast.Expr,
    ast.Assign,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.keyword,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.operator,
    ast.unaryop,
    ast.boolop,
    ast.cmpop,
)


def configure(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> None:
    """Fix the provider, model or key for the rest of the session.

    Every argument is optional and ``None`` means "leave as it was", so a
    notebook can set the provider once in its first cell.
    """
    current_provider = _DEFAULTS["provider"]
    current_key = _DEFAULTS["api_key"]
    next_key = current_key if api_key is None else api_key
    if provider is not None:
        switching_provider = (
            current_provider is not None and provider != current_provider
        )
        if switching_provider and api_key is None:
            # An explicit key belongs to the provider it was configured with.
            # Never carry it across a provider switch. If the new provider has
            # an environment credential it can be selected safely; otherwise
            # the caller must supply the matching key alongside the provider.
            resolve_provider(provider)
            next_key = None
        else:
            resolve_provider(
                provider, api_key=next_key
            )  # fail now, not at the first question
        _DEFAULTS["provider"] = provider
    if model is not None:
        _DEFAULTS["model"] = model
    if api_key is not None or (provider is not None and next_key is None):
        _DEFAULTS["api_key"] = next_key


def forget() -> None:
    """Drop the configured provider, model and key.

    ``configure()`` treats ``None`` as "leave as it was", which is what makes
    it convenient to call twice — and which left no way to take a key back out
    of the session once it was in. This is that way.
    """
    _DEFAULTS.update({"provider": None, "model": None, "api_key": None})


def configured() -> dict[str, Any]:
    """What :func:`configure` is currently holding. The key is never shown."""
    key = _DEFAULTS["api_key"]
    return {
        "provider": _DEFAULTS["provider"],
        "model": _DEFAULTS["model"],
        "api_key": None if key is None else "set (hidden)",
    }


@dataclass(frozen=True)
class Suggestion:
    """Code a model wrote, which you read before it runs.

    Nothing here executes on its own. :meth:`run` exists and is one line, but
    it is a line *you* type — a model writing Python and the library running it
    unseen is the one thing a learner cannot check, and checking is the point.
    """

    code: str
    question: str
    provider: str
    model: str
    #: The reply with the code fence removed — the model's own commentary.
    commentary: str = ""
    raw: str = field(default="", repr=False)

    def show(self) -> str:
        """Print the code and return it, as ``show_python()`` does."""
        safe_print(self.code)
        return self.code

    def validate(self) -> None:
        """Refuse code outside MathSlate's small expression-oriented policy.

        This blocks prompt-injected imports, file/network access and Python
        introspection. It is a policy guard, not an operating-system sandbox;
        callers needing arbitrary Python must opt into :meth:`run`'s unsafe
        mode explicitly.
        """
        _validate_code(self.code)

    def run(
        self,
        namespace: dict[str, Any] | None = None,
        *,
        unsafe: bool = False,
    ) -> dict[str, Any]:
        """Execute the visible code after validating it.

        The default namespace contains only the documented MathSlate/SymPy
        names and a minimal builtin set, and runs under a wall-clock budget
        (see :data:`_RUN_BUDGET`) so a legal-looking but runaway expression
        cannot hang the caller. Pass ``unsafe=True`` to recover normal Python
        execution, including imports, filesystem access and no time limit;
        never use that mode for untrusted model output.
        """
        scope: dict[str, Any] = namespace if namespace is not None else {}
        if unsafe:
            if "plot" not in scope:
                exec("from mathslate import *", scope)  # noqa: S102
            exec(compile(self.code, "<mathslate.ai>", "exec"), scope)  # noqa: S102
            return scope

        self.validate()
        scope.update(_safe_exports())
        # Never let a supplied namespace restore unrestricted builtins.
        scope["__builtins__"] = dict(_SAFE_BUILTINS)
        compiled = compile(self.code, "<mathslate.ai>", "exec")
        try:
            within_budget(exec, compiled, scope, seconds=_RUN_BUDGET)  # noqa: S102
        except SymbolicTimeout as error:
            raise UnsupportedInputError(
                f"AI suggestion exceeded its {_RUN_BUDGET:g}s execution budget "
                "and was stopped — it likely does unbounded work such as a huge "
                "power or a large repetition."
            ) from error
        return scope

    def _repr_html_(self) -> str:
        escaped = (
            self.code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        return (
            f"<div style='opacity:0.6;font-size:0.9em'>{self.provider} · "
            f"{self.model} — read it before running it</div>"
            f"<pre style='padding:8px'><code>{escaped}</code></pre>"
        )

    def __str__(self) -> str:
        return self.code


def _safe_exports() -> dict[str, Any]:
    """The explicit execution namespace; importing MathSlate stays lazy."""
    import mathslate

    constants = {"E", "I", "oo", "pi", "x", "y", "z", "t", "n", "k", "theta"}
    exports = {name: getattr(mathslate, name) for name in _SAFE_CALLS | constants}
    exports["dataset"] = _safe_dataset
    return exports


def _safe_dataset(
    source: Any, columns: Any = None, encoding: Any = None
) -> Any:
    """``dataset()``, refusing a path so restricted code cannot read files.

    The real ``dataset()`` accepts a CSV path as ``source`` — exactly the
    file-read primitive :meth:`Suggestion.validate` exists to keep out of
    restricted execution. Literal data (a mapping, arrays, or an existing
    ``Dataset``) still works; only "load this path" is refused.
    """
    import mathslate

    if isinstance(source, (str, Path)):
        raise UnsupportedInputError(
            "dataset() may not load from a path in restricted execution; pass "
            "literal data instead, e.g. dataset({'x': [...], 'y': [...]})."
        )
    return mathslate.dataset(source, columns, encoding)


def _validate_code(code: str) -> None:
    """Validate one suggestion using an allowlist, never a porous denylist."""
    if len(code) > _MAX_SAFE_CODE_CHARS:
        raise UnsupportedInputError(
            "AI suggestion is too large for restricted execution; inspect it and "
            "use run(unsafe=True) only if you trust it."
        )
    try:
        tree = ast.parse(code, filename="<mathslate.ai>", mode="exec")
    except SyntaxError as error:
        raise UnsupportedInputError(
            f"AI suggestion is not valid Python: {error.msg}."
        ) from error

    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_SAFE_AST_NODES:
        raise UnsupportedInputError(
            "AI suggestion is too complex for restricted execution."
        )

    for node in nodes:
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise UnsupportedInputError(
                f"AI suggestion uses {type(node).__name__}, which restricted "
                "execution does not allow."
            )
        if isinstance(node, ast.Assign):
            if not all(_safe_assignment_target(target) for target in node.targets):
                raise UnsupportedInputError(
                    "AI suggestion may assign only to ordinary variable names."
                )
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise UnsupportedInputError(
                    "AI suggestion may not inspect private or dunder attributes."
                )
            if node.attr not in _SAFE_ATTRIBUTES:
                raise UnsupportedInputError(
                    f"AI suggestion may not access attribute {node.attr!r} "
                    "in restricted mode."
                )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                allowed = node.func.id in _SAFE_CALLS or node.func.id in _SAFE_BUILTINS
                if not allowed:
                    raise UnsupportedInputError(
                        f"AI suggestion may not call {node.func.id!r} in restricted mode."
                    )
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in _SAFE_METHODS:
                    raise UnsupportedInputError(
                        f"AI suggestion may not call method {node.func.attr!r} "
                        "in restricted mode."
                    )
            else:
                raise UnsupportedInputError(
                    "AI suggestion uses an indirect call that restricted execution "
                    "cannot verify."
                )


def _safe_assignment_target(target: ast.expr) -> bool:
    if isinstance(target, ast.Name):
        return not target.id.startswith("_")
    if isinstance(target, (ast.Tuple, ast.List)):
        return all(_safe_assignment_target(item) for item in target.elts)
    return False


def system_prompt() -> str:
    """What the model is told about MathSlate. Built from the real contract."""
    from ..core.dispatch import KINDS

    return textwrap.dedent(
        f"""\
        You write short Python using the MathSlate library, and nothing else.

        MathSlate wraps SymPy, NumPy and Plotly behind one function, `plot()`,
        which infers what was meant. Assume `from mathslate import *` has run:
        `plot`, `polar`, `analyze`, `table`, `slider`, `animate`, `dataset`,
        `show_python`, the SymPy functions (`sin`, `cos`, `exp`, `sqrt`, `log`,
        `floor`, `Abs`, `diff`, `integrate`, `solve`, `Eq`, `Matrix`, ...) and
        the symbols `x, y, z, t, n, k, theta` all already exist.

        The dispatch contract — one rule: a `list` means several things
        together, a `tuple` means one vector-valued object.

          plot(sin(x)/x)                      a 2D curve
          plot([sin(x), cos(x)])              two curves overlaid
          plot((cos(t), sin(t)))              one parametric curve
          plot((cos(t), sin(t), t))           a 3D space curve
          plot(x*y)                           a surface
          plot(x*y, kind="contour")           the same, flat
          plot(Eq(x**2 + y**2, 4))            an implicit curve
          polar(1 + cos(t))                   r = f(theta)
          plot(Matrix([[2, 1], [1, 3]]))      a matrix as a transformation
          plot(sin(x), (x, 0, 6.28))          an explicit range
          plot([1.0, 2.0, 3.0], kind="hist")  a histogram

        Other entry points:
          analyze(expr)      roots, extrema, asymptotes, symmetry, period
          table(expr)        the same function as a table of values
          a = slider(-5, 5)  then plot(a*sin(x)) — no callback needed
          dataset({{...}})     then .fit(a*x + b) to fit a symbolic model

        Plot kinds that can come back: {", ".join(KINDS)}.

        Rules for your answer:
        - Reply with ONE ```python fenced block and at most two short sentences
          outside it.
        - Do not import anything. Do not define symbols that already exist.
        - Do not invent MathSlate functions. If MathSlate cannot do it, say so
          in one sentence and give the plain SymPy or Plotly that can.
        - Prefer the shortest call that answers the question. MathSlate's
          defaults are good; do not pass arguments that only restate them.
        """
    )


def ask(
    question: str,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    verbose: bool = False,
) -> Suggestion:
    """Turn a question into MathSlate code. Nothing is executed.

    Examples
    --------
    >>> from mathslate.ai import ask                       # doctest: +SKIP
    >>> print(ask("plot the tangent over one period").code)  # doctest: +SKIP
    plot(tan(x), (x, -1.5, 1.5))
    """
    if not question.strip():
        raise UnsupportedInputError("ask() needs a question.")

    effective_key = api_key or _DEFAULTS["api_key"]
    # A per-call key does not inherit a session provider: the key may belong to
    # another service. With several SDKs installed, resolve_provider() will ask
    # the caller to name the provider instead of exposing the credential.
    provider_name = provider
    if provider_name is None and api_key is None:
        provider_name = _DEFAULTS["provider"]
    chosen = resolve_provider(
        provider_name, api_key=effective_key
    )
    model_name = model or _DEFAULTS["model"] or chosen.default_model
    backend: Backend = chosen.build(effective_key)

    reply = backend.complete(system_prompt(), question, model_name)
    code, commentary = _split(reply)
    suggestion = Suggestion(
        code=code,
        question=question,
        provider=chosen.name,
        model=model_name,
        commentary=commentary,
        raw=reply,
    )
    if verbose:
        safe_print(f"# {chosen.name} · {model_name}")
        safe_print(code)
    return suggestion


def _split(reply: str) -> tuple[str, str]:
    """Separate the fenced code from the model's prose.

    A model that ignores the fence instruction still has to be usable, so an
    unfenced reply is treated as code rather than thrown away — but only after
    the fenced case, which is what the prompt asks for.
    """
    matches = _FENCE.findall(reply)
    if matches:
        code = "\n\n".join(block.strip() for block in matches)
        commentary = _FENCE.sub("", reply).strip()
        return code, commentary
    return reply.strip(), ""
