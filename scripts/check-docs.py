#!/usr/bin/env python3
"""Pre-CI documentation validation — build docs and check for warnings.

Usage
-----

.. code-block:: bash

    python scripts/check-docs.py          # build docs with --keep-going
    python scripts/check-docs.py --quick  # skip sphinx build

Exit 0 when all checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_DIR = REPO_ROOT / "doc"


def _run(
    cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> tuple[bool, str]:
    """Run a command and return ``(success, combined_stdout_stderr)``."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=cwd,
            env=env,
        )
    except FileNotFoundError:
        return False, f"Command not found: {cmd[0]}"

    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    return result.returncode == 0, output


def check_sphinx() -> tuple[bool, str]:
    """Run ``sphinx-build -b html -W --keep-going`` in the doc directory."""
    cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
        "-d",
        "_build/doctrees",
        "-W",
        "--keep-going",
        ".",
        "_build/html",
    ]
    env = {**os.environ}
    return _run(cmd, cwd=DOC_DIR, env=env)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-CI documentation validation.",
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Skip sphinx build (fast check only).",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    failed = 0

    if args.quick:
        print("check-docs: Skipping Sphinx build (--quick).")
    else:
        print("check-docs: Building documentation...")
        print(
            "  Command: cd doc && python -m sphinx -b html "
            "-d _build/doctrees -W --keep-going . _build/html"
        )
        ok, output = check_sphinx()
        if ok:
            print("  Sphinx build: OK")
        else:
            failed += 1
            print(f"  Sphinx build: FAILED", file=sys.stderr)
            for line in output.splitlines():
                stripped = line.strip()
                if stripped:
                    print(f"    {stripped}", file=sys.stderr)

    if failed:
        print(f"\n{failed} check(s) failed.", file=sys.stderr)
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
