"""Print the version in this checkout, or check a release tag against it.

    python scripts/release_version.py                  # -> 0.1.6
    python scripts/release_version.py --check-tag v0.1.6

The version is read from ``mathslate/__init__.py`` next to this script, not by
importing ``mathslate``: an import finds whichever copy is installed, which
need not be this checkout, and needs every dependency present. Works from any
directory.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

INIT = Path(__file__).resolve().parents[1] / "mathslate" / "__init__.py"
_VERSION = re.compile(r'^__version__ = "(?P<version>[^"]+)"$', re.MULTILINE)


def checkout_version() -> str:
    match = _VERSION.search(INIT.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"no `__version__ = \"...\"` line in {INIT}")
    return match["version"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check-tag",
        metavar="TAG",
        help="fail unless TAG is exactly 'v' followed by the checkout's version",
    )
    args = parser.parse_args(argv)
    version = checkout_version()
    if args.check_tag is None:
        print(version)
        return 0
    expected = f"v{version}"
    if args.check_tag != expected:
        print(
            f"tag {args.check_tag!r} does not match the package version: "
            f"mathslate/__init__.py says {version!r}, so the tag must be {expected!r}.",
            file=sys.stderr,
        )
        return 1
    print(f"tag {args.check_tag} matches mathslate {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
