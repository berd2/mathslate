"""Natural language → MathSlate code (PRD 6.1, 7 v1.0).

    from mathslate.ai import ask
    print(ask("plot the tangent over one period").code)

Three rules, and the first two are absolute.

**The core never imports this package.** PRD §8 lists "AI dependency blocked on
school or corporate networks" as a risk whose mitigation is that the core must
work fully offline and the AI ships as extras. ``import mathslate`` does not
reach this module, and this module's provider libraries are imported lazily
inside the adapter that needs them, so nothing here can make a plain install
heavier or a school network a problem. ``tests/test_ai.py`` asserts it.

**It returns code; it does not run it.** A model writing Python and the library
executing it unseen is not something a learner can check, and checking is the
whole point of this project. :class:`Suggestion` holds the source and prints it.
An explicit :meth:`Suggestion.run` validates a small MathSlate-oriented AST,
uses restricted builtins, and runs under a wall-clock budget so an
allowlisted-but-runaway expression cannot hang the caller; unrestricted Python
requires ``unsafe=True``.

**No provider is bundled.** :func:`ask` resolves one at call time from what is
installed and configured, and says exactly what to install when it finds
nothing. Resolving PRD §9 open decision 4: the layer lives in the core package
as extras, not as a separate distribution — it is a few hundred lines and a
separate package would cost more in version skew than it saves.
"""

from __future__ import annotations

from .providers import (
    PROVIDERS,
    Provider,
    available_providers,
    resolve_provider,
)
from .suggest import (
    RunResult,
    Suggestion,
    ask,
    check_connection,
    configure,
    configured,
    forget,
    system_prompt,
)
from .ui import AssistantPanel, assistant

__all__ = [
    "ask",
    "check_connection",
    "assistant",
    "AssistantPanel",
    "configure",
    "configured",
    "forget",
    "RunResult",
    "Suggestion",
    "Provider",
    "PROVIDERS",
    "available_providers",
    "resolve_provider",
    "system_prompt",
]
