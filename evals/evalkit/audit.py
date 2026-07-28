"""Validate the eval suite itself, without calling any model.

Every code/debug task carries a hand-written ``reference_solution``. The audit
runs each one through the real runner and asserts it passes. That catches:

- a verifier that no correct solution can satisfy
- a task whose canonical answer broke on a dependency upgrade (this is exactly
  how the transformers 4 -> 5 tuple change was caught)
- a prompt that asks for something the API cannot do

Run it after touching tasks, after upgrading nnsight or transformers, and before
spending money on a grid.

    python -m evalkit.audit            # every task with a reference
    python -m evalkit.audit -k source  # only ids matching a substring
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from .registry import TaskKind, all_tasks, load_all
from .runner import run_task


def audit(pattern: str | None = None, workers: int = 1) -> int:
    load_all()
    tasks = [t for t in all_tasks() if t.kind is not TaskKind.MCQ]
    if pattern:
        tasks = [t for t in tasks if pattern in t.id]
    tasks.sort(key=lambda t: t.id)

    with_reference = [t for t in tasks if t.reference_solution.strip()]
    missing = [t for t in tasks if not t.reference_solution.strip()]

    print(f"auditing {len(with_reference)} task(s); {len(missing)} without a reference\n")

    def run_one(task):
        started = time.time()
        outcome = run_task(task, f"```python\n{task.reference_solution}\n```")
        return task, outcome, time.time() - started

    failures = []
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_one, with_reference))
    else:
        results = [run_one(task) for task in with_reference]

    for task, outcome, elapsed in results:
        mark = "pass" if outcome.passed else "FAIL"
        print(f"[{mark}] {task.id:<46} {elapsed:5.1f}s")
        if not outcome.passed:
            failures.append((task, outcome))

    if missing:
        print("\nno reference solution:")
        for task in missing:
            print(f"  {task.id}")

    if failures:
        print(f"\n{len(failures)} failing task(s):\n")
        for task, outcome in failures:
            print(f"--- {task.id}: {outcome.reason or outcome.error_type}")
            tail = (outcome.stderr or outcome.stdout or "").strip().splitlines()
            for line in tail[-12:]:
                print(f"    {line}")
            print()

    print(f"\n{len(with_reference) - len(failures)}/{len(with_reference)} references pass")
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-k", "--pattern", help="only tasks whose id contains this")
    parser.add_argument("-j", "--workers", type=int, default=1, help="parallel subprocesses")
    args = parser.parse_args(argv)
    return audit(args.pattern, args.workers)


if __name__ == "__main__":
    sys.exit(main())
