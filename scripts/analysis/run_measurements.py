#!/usr/bin/env python3
"""Run selected measurement modules from a JSON configuration.

Configuration keys are module names (breadth, distance, frontier, impact,
novelty). Values are command-line argument mappings. Keys use underscores;
they are converted to ``--kebab-case`` options. Lists expand to multiple
values, true booleans become flags, and false/null values are omitted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


MODULES = {
    "breadth": "scripts.analysis.measurements.breadth",
    "distance": "scripts.analysis.measurements.distance",
    "frontier": "scripts.analysis.measurements.frontier",
    "impact": "scripts.analysis.measurements.impact",
    "novelty": "scripts.analysis.measurements.novelty",
}


def command(module: str, options: dict[str, Any]) -> list[str]:
    output = [sys.executable, "-m", MODULES[module]]
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            output.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, list):
            output.append(flag)
            output.extend(str(item) for item in value)
        else:
            output.extend([flag, str(value)])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", nargs="*", choices=tuple(MODULES), default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected = args.only or [name for name in MODULES if name in config]
    for name in selected:
        options = config.get(name)
        if not isinstance(options, dict):
            raise ValueError(f"Missing object-valued configuration for {name}")
        cmd = command(name, options)
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
