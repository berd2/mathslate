"""Frontend adapters (PRD 6.2). Importing this pulls in no frontend package."""

from __future__ import annotations

from .adapters import Frontend, detect_frontend, frontend_report
from .interact import Slider, release, release_all, sliders_in

__all__ = [
    "Frontend",
    "detect_frontend",
    "frontend_report",
    "Slider",
    "release",
    "release_all",
    "sliders_in",
]
