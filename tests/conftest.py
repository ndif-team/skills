from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from docblocks import REPO_ROOT

REPORT_PATH = REPO_ROOT / "tests" / ".report.json"

# Collected by test_skills.py, written out at session end.
RESULTS: list[dict] = []


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run code blocks marked `<!-- test: slow -->`",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "docs: executes code blocks from a skill markdown file")


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency of the suite
        return False
    return torch.cuda.is_available()


def ndif_host() -> str | None:
    return os.environ.get("NDIF_HOST") or None


def pytest_sessionfinish(session, exitstatus):
    if not RESULTS:
        return
    REPORT_PATH.write_text(json.dumps(RESULTS, indent=2, default=str))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not RESULTS:
        return

    by_file: dict[str, dict[str, int]] = {}
    for entry in RESULTS:
        counts = by_file.setdefault(entry["file"], {"ran": 0, "skipped": 0, "compiled": 0})
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1

    terminalreporter.write_sep("=", "skill code blocks")
    width = max(len(name) for name in by_file)
    for name, counts in sorted(by_file.items()):
        terminalreporter.write_line(
            f"{name:<{width}}  ran={counts['ran']:<3} "
            f"compiled-only={counts['compiled']:<3} skipped={counts['skipped']}"
        )
    total_ran = sum(c["ran"] for c in by_file.values())
    total_skipped = sum(c["skipped"] for c in by_file.values())
    total_compiled = sum(c["compiled"] for c in by_file.values())
    terminalreporter.write_line(
        f"{'TOTAL':<{width}}  ran={total_ran:<3} "
        f"compiled-only={total_compiled:<3} skipped={total_skipped}"
    )
