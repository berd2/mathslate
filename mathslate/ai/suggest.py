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
import contextlib
import io
import pickle
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from .._text import safe_print
from ..core._budget import set_budget
from ..errors import UnsupportedInputError
from .providers import Backend, Provider, resolve_provider

__all__ = [
    "check_connection",
    "RunResult",
    "Suggestion",
    "ask",
    "configure",
    "configured",
    "forget",
    "system_prompt",
]

_FENCE = re.compile(r"```(?:python)?\n(?P<body>.*?)```", re.DOTALL)

#: Set by :func:`configure`; ``ask()`` falls back to auto-detection.
_DEFAULTS: dict[str, Any] = {
    "provider": None,
    "model": None,
    "api_key": None,
    "remembered": False,
}

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
#: The allowlisted callables that take a string and do *not* hand it to
#: ``sympify``. ``symbols``/``Symbol`` read it as a name; ``dataset`` reads it
#: as column names and data. Every other allowlisted callable sympifies its
#: positional arguments, and ``sympify`` on a string ``eval``\ s that string as
#: Python — ``solve("__import__('os').system(...)")`` is arbitrary code, not an
#: equation. So a string literal is refused everywhere except the positions
#: these three make safe (and keyword values, which nothing sympifies).
_STRING_CALLABLES: frozenset[str] = frozenset({"symbols", "Symbol", "dataset"})
#: A name ``symbols()``/``Symbol()`` will accept — belt and suspenders, since a
#: name is never evaluated, but it keeps even that string to identifier shapes.
_SYMBOL_NAME = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*(\s*[ ,:]\s*[A-Za-z_][A-Za-z0-9_]*)*\s*$")
#: Wall-clock budget for one restricted execution. The AST allowlist bounds
#: *what* restricted code can name, not what a legal expression costs to run —
#: ``9**9**9`` is three ``Constant``/``BinOp`` nodes and passes validation, but
#: computing it exhausts memory. Restricted code therefore runs in a separate
#: Python process which can be terminated without injecting an exception into
#: NumPy, SymPy or Plotly on the caller's thread.
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


class RunResult(dict[str, Any]):
    """Values produced by :meth:`Suggestion.run`.

    The isolated executor deliberately keeps its MathSlate/SymPy helper names
    private.  This mapping therefore contains only names written by the
    suggestion.  When the visible code ends with an expression (as generated
    plot requests normally do), its value is available as ``result`` and is
    displayed directly in a notebook.
    """

    def _ipython_display_(self) -> None:
        """Render a final plot/value instead of the implementation mapping."""
        if "result" not in self:
            return
        try:
            from IPython.display import display
        except ImportError:
            return
        display(self["result"])


def configure(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    *,
    remember: bool = False,
) -> None:
    """Fix the provider, model or key for the rest of the session.

    Every argument is optional and ``None`` means "leave as it was", so a
    notebook can set the provider once in its first cell. ``remember=True``
    stores the key in the operating-system credential manager, never in the
    project or notebook.
    """
    current_provider = _DEFAULTS["provider"]
    current_key = _DEFAULTS["api_key"]
    switching_provider = (
        provider is not None
        and current_provider is not None
        and provider != current_provider
    )
    # A session key is bound to its provider. On a switch, only an explicitly
    # supplied key or one saved for the new provider may be considered.
    next_key = (
        None if switching_provider and api_key is None
        else current_key if api_key is None
        else api_key
    )
    loaded_saved = False
    if provider is not None and next_key is None:
        from .credentials import load_credential

        saved = load_credential(provider)
        next_key = saved.api_key if saved is not None else None
        loaded_saved = saved is not None
    if provider is not None:
        if switching_provider and api_key is None:
            # An explicit key belongs to the provider it was configured with.
            # Never carry it across a provider switch. If the new provider has
            # an environment credential it can be selected safely; otherwise
            # the caller must supply the matching key alongside the provider.
            resolve_provider(provider, api_key=next_key)
        else:
            resolve_provider(
                provider, api_key=next_key
            )  # fail now, not at the first question
        _DEFAULTS["provider"] = provider
    if model is not None:
        _DEFAULTS["model"] = model
    if api_key is not None or loaded_saved or (provider is not None and next_key is None):
        _DEFAULTS["api_key"] = next_key
        _DEFAULTS["remembered"] = loaded_saved
    if remember:
        from .credentials import save_credential

        chosen = provider or _DEFAULTS["provider"]
        key = api_key or _DEFAULTS["api_key"]
        if not chosen or not key:
            raise UnsupportedInputError(
                "remember=True needs both a provider and an API key."
            )
        save_credential(chosen, key, model or _DEFAULTS["model"])
        _DEFAULTS["remembered"] = True


def forget(*, persistent: bool = False, provider: str | None = None) -> None:
    """Drop the configured provider, model and key.

    ``configure()`` treats ``None`` as "leave as it was", which is what makes
    it convenient to call twice — and which left no way to take a key back out
    of the session once it was in. This is that way.
    """
    chosen = provider or _DEFAULTS["provider"]
    if persistent:
        from .credentials import delete_credential

        delete_credential(chosen)
    _DEFAULTS.update(
        {"provider": None, "model": None, "api_key": None, "remembered": False}
    )


def configured() -> dict[str, Any]:
    """What :func:`configure` is currently holding. The key is never shown."""
    key = _DEFAULTS["api_key"]
    if key is None:
        from .credentials import load_credential

        saved = load_credential(_DEFAULTS["provider"])
        if saved is not None:
            return {
                "provider": saved.provider,
                "model": _DEFAULTS["model"] or saved.model,
                "api_key": "saved securely",
            }
    return {
        "provider": _DEFAULTS["provider"],
        "model": _DEFAULTS["model"],
        "api_key": (
            None
            if key is None
            else "saved securely" if _DEFAULTS["remembered"] else "set (hidden)"
        ),
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
        show_code: bool = True,
    ) -> RunResult | dict[str, Any]:
        """Execute the visible code after validating it.

        The default namespace contains only the documented MathSlate/SymPy
        names and a minimal builtin set. Restricted code runs in a separate
        process under a wall-clock budget (see :data:`_RUN_BUDGET`), so a
        legal-looking but runaway expression cannot hang or corrupt the caller.
        Its return value contains only values the suggestion created, never the
        internal MathSlate/SymPy execution namespace. A final bare expression
        such as ``plot(tan(x))`` is returned as ``result`` and displayed in a
        notebook. By default it also prints the visible code first, so
        ``ask("...").run()`` retains the review step; pass ``show_code=False``
        only when the code is already visible in another interface.
        A supplied namespace is deliberately available only in ``unsafe`` mode:
        its objects could carry capabilities that an allowed method name could
        invoke. Pass ``unsafe=True`` to recover normal Python execution,
        including imports, filesystem access, a supplied namespace and no time
        limit; never use that mode for untrusted model output.
        """
        if show_code:
            self.show()
        scope: dict[str, Any] = namespace if namespace is not None else {}
        if unsafe:
            if "plot" not in scope:
                exec("from mathslate import *", scope)  # noqa: S102
            exec(compile(self.code, "<mathslate.ai>", "exec"), scope)  # noqa: S102
            return scope

        self.validate()
        if namespace is not None:
            # `is not None`, not a truthiness test: an empty dict is still a
            # caller asking to be given the results in *their* object, and
            # restricted execution can no longer do that — the names come back
            # from another process, so nothing is filled in place. Accepting
            # `{}` silently would leave that caller reading an empty mapping
            # and finding a KeyError where the answer used to be.
            raise UnsupportedInputError(
                "restricted execution does not accept a namespace because its "
                "objects may carry capabilities, and its results come back from "
                "a separate process rather than being written into yours — read "
                "them from the returned mapping. Use run(namespace, unsafe=True) "
                "only if you trust both the code and the namespace."
            )

        assigned, output = _run_restricted(self.code)
        if output:
            safe_print(output, end="")
        return RunResult(assigned)

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


def _safe_scope() -> dict[str, Any]:
    """A new restricted global scope, never mixed with caller-owned values."""
    scope = _safe_exports()
    scope["__builtins__"] = dict(_SAFE_BUILTINS)
    return scope


def _assigned_names(code: str) -> tuple[str, ...]:
    """Names whose values need to travel back from the isolated process."""
    names: list[str] = []

    def add(target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                add(item)

    for statement in ast.parse(code, filename="<mathslate.ai>", mode="exec").body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                add(target)
    return tuple(dict.fromkeys(names))


def _run_restricted(code: str) -> tuple[dict[str, Any], str]:
    """Run validated code out-of-process and return its assigned values/output."""
    request = pickle.dumps((code, _assigned_names(code)))
    command = (
        "from mathslate.ai.suggest import _restricted_process_entry as run; "
        "import sys; sys.stdout.buffer.write(run(sys.stdin.buffer.read()))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", command],
            input=request,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_RUN_BUDGET,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise UnsupportedInputError(
            f"AI suggestion exceeded its {_RUN_BUDGET:g}s execution budget and "
            "was stopped in an isolated process."
        ) from error
    except (OSError, ValueError) as error:
        # Not every host can start a process at all — Pyodide/JupyterLite has
        # no `fork`, and a locked-down container may refuse. That is a fact
        # about the environment, not about the suggestion, so say so instead
        # of letting a raw OSError out of a method whose contract is
        # `UnsupportedInputError`.
        raise UnsupportedInputError(
            "restricted execution needs to start a separate Python process, "
            f"which this environment does not allow ({error}). Read the code "
            "and use run(unsafe=True) if you trust it."
        ) from error

    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise UnsupportedInputError(
            "restricted AI execution failed in its isolated process"
            + (f": {detail}" if detail else ".")
        )
    try:
        status, payload = pickle.loads(completed.stdout)
    except (EOFError, pickle.UnpicklingError, ValueError, TypeError) as error:
        raise UnsupportedInputError(
            "restricted AI execution returned an unreadable result."
        ) from error
    if status == "error":
        raise UnsupportedInputError(f"AI suggestion failed: {payload}")
    return payload


def _restricted_process_entry(request: bytes) -> bytes:
    """Subprocess entry point. Kept importable so it needs no notebook state."""
    try:
        code, assigned_names = pickle.loads(request)
        # The parent owns the only deadline for this process. Starting the
        # thread-based symbolic budget here would reintroduce async exception
        # injection into its native dependencies, while this process can be
        # terminated safely as a whole.
        set_budget(None)
        scope = _safe_scope()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            value = _execute_and_capture_final_expression(code, scope)
        assigned = {name: scope[name] for name in assigned_names if name in scope}
        if value is not _NO_FINAL_EXPRESSION:
            assigned["result"] = value
        response: tuple[str, Any] = ("ok", (assigned, output.getvalue()))
        # Serialize before returning so unpicklable results become a useful
        # restricted-execution error rather than a broken pipe.
        return pickle.dumps(response)
    except BaseException as error:  # noqa: BLE001 - crosses a process boundary
        return pickle.dumps(("error", f"{type(error).__name__}: {error}"))


_NO_FINAL_EXPRESSION = object()


def _execute_and_capture_final_expression(code: str, scope: dict[str, Any]) -> Any:
    """Execute *code*, evaluating its final expression as the user result.

    ``exec`` intentionally discards an expression's value.  That makes a
    generated one-line ``plot(...)`` appear to do nothing once it has run in
    the isolated process.  Splitting only the final ``ast.Expr`` retains normal
    statement semantics while allowing the caller to receive and display it.
    """
    tree = ast.parse(code, filename="<mathslate.ai>", mode="exec")
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        exec(compile(tree, "<mathslate.ai>", "exec"), scope)  # noqa: S102
        return _NO_FINAL_EXPRESSION

    statements = tree.body[:-1]
    if statements:
        prefix = ast.Module(body=statements, type_ignores=[])
        exec(compile(ast.fix_missing_locations(prefix), "<mathslate.ai>", "exec"), scope)  # noqa: S102
    final = ast.Expression(tree.body[-1].value)
    return eval(compile(ast.fix_missing_locations(final), "<mathslate.ai>", "eval"), scope)  # noqa: S307


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

    safe_strings = _sympify_safe_string_literals(tree)
    for node in nodes:
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise UnsupportedInputError(
                f"AI suggestion uses {type(node).__name__}, which restricted "
                "execution does not allow."
            )
        if isinstance(node, ast.Constant):
            # SymPy's ``sympify`` evaluates a string as a Python expression, and
            # recurses into a list/tuple/dict to do the same to each element —
            # so a string literal anywhere an allowlisted callable might sympify
            # it is arbitrary code that the allowlist above waves straight
            # through (it is one ``Constant`` node). Everything else here checks
            # what code *names*; this checks the one value that is itself code.
            if isinstance(node.value, str) and id(node) not in safe_strings:
                raise UnsupportedInputError(
                    "AI suggestion has a string literal in a position SymPy "
                    "would evaluate as code; write the expression itself, e.g. "
                    'sin(x) rather than "sin(x)".'
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


def _sympify_safe_string_literals(tree: ast.AST) -> frozenset[int]:
    """The ``id()``\\ s of string literals that cannot reach ``sympify``.

    A string is safe in exactly two shapes: as a keyword-argument value (no
    allowlisted callable sympifies one), or as a *literal* inside the positional
    arguments of the three callables that read strings as names or data rather
    than as expressions (:data:`_STRING_CALLABLES`). "Literal" is the load-
    bearing word: ``dataset({"x": [1, 2]})`` is safe, but the ``"..."`` in
    ``dataset({"x": solve("...")})`` sits inside a *call*, would be sympified
    there, and must stay refused — so the walk into a container stops at the
    first thing that is not itself a literal.
    """
    safe: set[int] = set()

    def _is_str(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, str)

    def _allow_literal(node: ast.AST) -> None:
        """Mark every string reachable through pure literal containers only."""
        if _is_str(node):
            safe.add(id(node))
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if key is not None:
                    _allow_literal(key)
            for value in node.values:
                _allow_literal(value)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for element in node.elts:
                _allow_literal(element)
        # Anything else — a Call, a Name, a BinOp — is not descended into, so a
        # string buried inside it stays outside `safe` and is refused.

    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and _is_str(node.value):
            safe.add(id(node.value))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "dataset":
                for arg in node.args:
                    _allow_literal(arg)
            elif node.func.id in _STRING_CALLABLES:  # symbols / Symbol
                for arg in node.args:
                    if _is_str(arg) and _SYMBOL_NAME.match(arg.value):
                        safe.add(id(arg))
    return frozenset(safe)


def _safe_assignment_target(target: ast.expr) -> bool:
    if isinstance(target, ast.Name):
        return not target.id.startswith("_")
    if isinstance(target, (ast.Tuple, ast.List)):
        return all(_safe_assignment_target(item) for item in target.elts)
    return False


def contract_reference() -> str:
    """The API description both the writing and the diagnosing prompts share.

    Built from :data:`mathslate.core.dispatch.KINDS` rather than written out
    beside it, so a kind cannot be added without every prompt learning about it
    in the same commit — which is the point §22 of the manual makes about this
    module, and applies as much to explaining a mistake as to writing code.
    """
    from ..core.dispatch import KINDS

    return textwrap.dedent(
        f"""\
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

        A range is always the triple `(symbol, lo, hi)` — `plot(sin(x), (x, 0,
        6.28))`, never `plot(sin(x), 0, 6.28)`.

        Other entry points:
          analyze(expr)      roots, extrema, asymptotes, symmetry, period
          table(expr)        the same function as a table of values
          a = slider(-5, 5)  then plot(a*sin(x)) — no callback needed
          dataset({{...}})     then .fit(a*x + b) to fit a symbolic model

        Plot kinds that can come back: {", ".join(KINDS)}.
        """
    )


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


def _complete(
    system: str,
    question: str,
    provider: str | None,
    model: str | None,
    api_key: str | None,
) -> tuple[str, Provider, str]:
    """Resolve a provider, send one message, and return ``(reply, provider, model)``.

    Every entry point that talks to a model goes through here, so credential
    resolution — a per-call key not inheriting a session provider, a saved key
    being found for the named one, the key never reaching an error message —
    is decided in one place rather than re-implemented per feature.
    """
    effective_key = api_key or _DEFAULTS["api_key"]
    # A per-call key does not inherit a session provider: the key may belong to
    # another service. With several SDKs installed, resolve_provider() will ask
    # the caller to name the provider instead of exposing the credential.
    provider_name = provider
    if provider_name is None and api_key is None:
        provider_name = _DEFAULTS["provider"]
    saved = None
    if effective_key is None and api_key is None:
        from .credentials import load_credential

        saved = load_credential(provider_name)
        if saved is not None:
            provider_name = saved.provider
            effective_key = saved.api_key
    chosen = resolve_provider(provider_name, api_key=effective_key)
    model_name = (
        model
        or _DEFAULTS["model"]
        or (saved.model if saved is not None else None)
        or chosen.default_model
    )
    backend: Backend = chosen.build(effective_key)

    try:
        reply = backend.complete(system, question, model_name)
    except Exception as exc:
        detail = str(exc).strip()
        if effective_key:
            detail = detail.replace(effective_key, "[hidden]")
        raise UnsupportedInputError(
            f"{chosen.name.title()} did not complete the request "
            f"({type(exc).__name__}: {detail or 'no detail'}). "
            "Check the API key, model access, quota, and network, then retry."
        ) from exc
    if not reply or not reply.strip():
        raise UnsupportedInputError(
            f"provider {chosen.name!r} returned no text. Retry the request, "
            "check the provider status, or choose another model."
        )
    return reply, chosen, model_name


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

    reply, chosen, model_name = _complete(
        system_prompt(), question, provider, model, api_key
    )
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


def check_connection(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Check a provider credential without asking it a MathSlate question.

    Saved credentials are resolved exactly as they are by :func:`ask`.  The
    probe sends only a fixed, non-mathematical message and returns a concise
    local status instead of treating the provider reply as generated code.
    """
    effective_key = api_key or _DEFAULTS["api_key"]
    provider_name = provider
    if provider_name is None and api_key is None:
        provider_name = _DEFAULTS["provider"]
    saved = None
    if effective_key is None and api_key is None:
        from .credentials import load_credential

        saved = load_credential(provider_name)
        if saved is not None:
            provider_name = saved.provider
            effective_key = saved.api_key
    chosen = resolve_provider(provider_name, api_key=effective_key)
    model_name = (
        model
        or _DEFAULTS["model"]
        or (saved.model if saved is not None else None)
        or chosen.default_model
    )
    backend: Backend = chosen.build(effective_key)
    try:
        reply = backend.complete(
            "You are an API connection check. Reply with exactly OK.",
            "Reply exactly OK.",
            model_name,
        )
    except Exception as exc:
        detail = str(exc).strip()
        if effective_key:
            detail = detail.replace(effective_key, "[hidden]")
        raise UnsupportedInputError(
            f"{chosen.name.title()} did not complete the connection check "
            f"({type(exc).__name__}: {detail or 'no detail'}). "
            "Check the API key, model access, quota, and network, then retry."
        ) from exc
    if not reply or not reply.strip():
        raise UnsupportedInputError(
            f"provider {chosen.name!r} returned no text for the connection check. "
            "Check the API key, model access, quota, and network, then retry."
        )
    return f"{chosen.name.title()} connection is working ({model_name})."


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
