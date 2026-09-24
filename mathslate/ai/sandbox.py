"""Restricted execution for model-written MathSlate code.

Two layers, because neither is enough alone. :func:`_validate_code` checks the
source against an allowlist (names, attributes, AST node types, and where a
string literal may stand). :class:`_StringGuard` then checks the values each
allowlisted callable actually receives, because a string made at run time
never appears in the source as a literal. The validated code runs in a separate
process with no credential environment variables and a wall-clock budget, and
its reply is unpickled against an allowlist of globals.

This is a policy guard, not an operating-system sandbox: the process runs as
the same user.
"""

from __future__ import annotations

import ast
import contextlib
import io
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Final


from ..core._budget import set_budget
from ..errors import UnsupportedInputError


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
#: these three make safe and the keywords :data:`_STRING_KEYWORDS` names.
_STRING_CALLABLES: frozenset[str] = frozenset({"symbols", "Symbol", "dataset"})
#: The keyword arguments that may carry a string, per callable (or method).
#: Nothing else may: a keyword reaches ``sympify`` as readily as a positional
#: argument does — ``series(x, x, x0="...")`` and ``limit(x, x, z0="...")``
#: both evaluate their string — so the only safe policy is to name the keywords
#: known to read a string as a label, a choice or a name, and refuse the rest.
_STRING_KEYWORDS: Final[dict[str, frozenset[str]]] = {
    "plot": frozenset({"kind", "label", "title", "yscale"}),
    "polar": frozenset({"kind", "label", "title", "yscale"}),
    "animate": frozenset({"kind", "label", "title", "yscale"}),
    "table": frozenset({"label"}),
    "slider": frozenset({"name", "label"}),
    "limit": frozenset({"dir"}),
    "series": frozenset({"dir"}),
    "dataset": frozenset({"columns", "encoding"}),
}
#: The same, for methods reached by attribute (``data.fit(model, x="t")``).
_METHOD_STRING_KEYWORDS: Final[dict[str, frozenset[str]]] = {
    "fit": frozenset({"x", "y"}),
}
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


def _safe_exports() -> dict[str, Any]:
    """The explicit execution namespace; importing MathSlate stays lazy."""
    import mathslate

    constants = {"E", "I", "oo", "pi", "x", "y", "z", "t", "n", "k", "theta"}
    exports: dict[str, Any] = {name: getattr(mathslate, name) for name in constants}
    for name in _SAFE_CALLS:
        target = _safe_dataset if name == "dataset" else getattr(mathslate, name)
        exports[name] = _StringGuard(name, target)
    return exports


class _StringGuard:
    """An allowlisted callable that refuses strings it would evaluate as code.

    :func:`_validate_code` can only see string *literals*. A string made at run
    time — a dataset's column names, a result's ``.text()`` — is a ``Name`` or
    an ``Attribute`` in the tree and passes it, yet ``sympify`` evaluates it
    just the same. This checks the values themselves, at the call, so the
    policy no longer depends on the analysis seeing every string's origin.
    """

    def __init__(self, name: str, target: Any) -> None:
        self._name = name
        self._target = target

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._name in _STRING_CALLABLES:
            positional_ok = self._name == "dataset" or all(
                not isinstance(a, str) or _SYMBOL_NAME.match(a) for a in args
            )
        else:
            positional_ok = not any(_holds_string(a) for a in args)
        allowed = _STRING_KEYWORDS.get(self._name, frozenset())
        keywords_ok = all(
            key in allowed or not _holds_string(value) for key, value in kwargs.items()
        )
        if not (positional_ok and keywords_ok):
            raise UnsupportedInputError(
                f"{self._name}() was given a string where SymPy would evaluate "
                "it as code; restricted execution passes expressions, not text."
            )
        return self._target(*args, **kwargs)

    def __repr__(self) -> str:
        return repr(self._target)


def _holds_string(value: Any, depth: int = 0) -> bool:
    """Whether ``value`` is, or plainly contains, a ``str``."""
    if isinstance(value, str):
        return True
    if depth > 8:
        # Deeper than any legitimate argument; refuse rather than recurse.
        return True
    if isinstance(value, dict):
        return any(
            _holds_string(k, depth + 1) or _holds_string(v, depth + 1)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_holds_string(item, depth + 1) for item in value)
    return False


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
        "from mathslate.ai.sandbox import _restricted_process_entry as run; "
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
            env=_child_environment(),
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
        status, payload = _ResultUnpickler(io.BytesIO(completed.stdout)).load()
    except (EOFError, pickle.UnpicklingError, ValueError, TypeError) as error:
        raise UnsupportedInputError(
            "restricted AI execution returned an unreadable result."
        ) from error
    if status == "error":
        raise UnsupportedInputError(f"AI suggestion failed: {payload}")
    return payload


#: The small, non-secret part of the caller's environment that the child may
#: need to start Python and load numeric libraries. A denylist cannot support
#: the promise that credentials are absent: ordinary names such as
#: ``PGPASSWORD``, ``GITHUB_PAT`` and ``DATABASE_URL`` do not share a reliable
#: spelling. Keep this an allowlist instead, and add a name only when restricted
#: MathSlate execution demonstrably needs it.
_CHILD_ENV_NAMES: Final[frozenset[str]] = frozenset(
    {
        # Python/process startup on Windows and POSIX.
        "COMSPEC",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "LANG",
        "PATH",
        "PATHEXT",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
        "USERPROFILE",
        "WINDIR",
        # Deterministic text/hash behaviour explicitly selected by the host.
        "PYTHONHASHSEED",
        "PYTHONIOENCODING",
        "PYTHONUTF8",
        # Thread ceilings used by NumPy and its common BLAS backends.
        "BLIS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    }
)
_CHILD_ENV_PREFIXES: Final[tuple[str, ...]] = ("LC_",)


def _child_environment() -> dict[str, str]:
    """A minimal process environment containing no caller credentials."""
    environment: dict[str, str] = {}
    for name, value in os.environ.items():
        normalized = name.upper()
        if normalized in _CHILD_ENV_NAMES or normalized.startswith(
            _CHILD_ENV_PREFIXES
        ):
            environment[name] = value
    return environment


#: What a restricted result may be rebuilt from: classes of these packages,
#: and the two reconstructors their pickles are known to call. Any other
#: global — ``os.system``, ``builtins.eval``, ``sympy.sympify`` — is refused.
_RESULT_MODULES: Final[tuple[str, ...]] = ("mathslate", "sympy", "numpy", "mpmath")
_RESULT_BUILTINS: Final[frozenset[str]] = frozenset(
    {"complex", "set", "frozenset", "slice", "range", "bytearray", "object"}
)
_RESULT_FUNCTIONS: Final[frozenset[tuple[str, str]]] = frozenset(
    {
        ("mathslate.result", "_rebuild_plot_result"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "_reconstruct"),
    }
)


class _ResultUnpickler(pickle.Unpickler):
    """Unpickle a restricted process's reply without trusting its globals.

    Hardening, not isolation: the child runs as the same user, so code running
    there can already act as that user. This only stops a reply from naming an
    arbitrary callable for the notebook's own process to invoke.
    """

    def find_class(self, module: str, name: str) -> Any:
        if module == "builtins":
            if name in _RESULT_BUILTINS:
                return super().find_class(module, name)
        elif (module, name) in _RESULT_FUNCTIONS:
            return super().find_class(module, name)
        elif module.split(".")[0] in _RESULT_MODULES:
            found = super().find_class(module, name)
            if isinstance(found, type):
                return found
        raise pickle.UnpicklingError(
            f"restricted result refers to {module}.{name}, which is not allowed."
        )


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

    A string is safe in exactly two shapes: as the value of a keyword named in
    :data:`_STRING_KEYWORDS` / :data:`_METHOD_STRING_KEYWORDS`, or as a
    *literal* inside the positional arguments of the three callables that read
    strings as names or data rather than as expressions
    (:data:`_STRING_CALLABLES`). "Literal" is the load-
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
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg in _string_keywords(node.func):
                _allow_literal(keyword.value)
        if isinstance(node.func, ast.Name):
            if node.func.id == "dataset":
                for arg in node.args:
                    _allow_literal(arg)
            elif node.func.id in _STRING_CALLABLES:  # symbols / Symbol
                for arg in node.args:
                    if _is_str(arg) and _SYMBOL_NAME.match(arg.value):
                        safe.add(id(arg))
    return frozenset(safe)


def _string_keywords(func: ast.expr) -> frozenset[str]:
    """The keywords of this call that may carry a string literal."""
    if isinstance(func, ast.Name):
        return _STRING_KEYWORDS.get(func.id, frozenset())
    if isinstance(func, ast.Attribute):
        return _METHOD_STRING_KEYWORDS.get(func.attr, frozenset())
    return frozenset()


def _safe_assignment_target(target: ast.expr) -> bool:
    if isinstance(target, ast.Name):
        return not target.id.startswith("_")
    if isinstance(target, (ast.Tuple, ast.List)):
        return all(_safe_assignment_target(item) for item in target.elts)
    return False
