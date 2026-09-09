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

from collections import Counter

from .registry import TaskKind, all_tasks, load_all
from .runner import run_task

# An MCQ set has to be unanswerable without reading it. Two shortcuts get
# measured here because both were live in the first version of this suite: the
# keyed answer was the longest choice in 29/32 questions and letter B in 29/32,
# so either heuristic alone scored 91% — indistinguishable from the strongest
# real condition, and the reason the no-resources baseline looked so good.
MAX_LONGEST_SHARE = 0.40      # chance is 0.25
MAX_POSITION_SHARE = 0.40     # chance is 0.25
MAX_LENGTH_SPREAD = 1.6       # longest choice vs shortest, within one question


def audit_mcq_bias(verbose: bool = True) -> int:
    load_all()
    mcqs = [t for t in all_tasks() if t.kind is TaskKind.MCQ]
    if not mcqs:
        return 0

    lengths = {t.id: [len(c) for c in t.choices] for t in mcqs}
    longest = [t for t in mcqs if lengths[t.id][t.correct_index] == max(lengths[t.id])]
    positions = Counter(t.correct_index for t in mcqs)
    spread = {
        t.id: max(lengths[t.id]) / max(1, min(lengths[t.id])) for t in mcqs
    }
    wide = sorted(
        (t for t in mcqs if spread[t.id] > MAX_LENGTH_SPREAD),
        key=lambda t: -spread[t.id],
    )

    longest_share = len(longest) / len(mcqs)
    worst_position, worst_count = positions.most_common(1)[0]
    position_share = worst_count / len(mcqs)

    problems = []
    if longest_share > MAX_LONGEST_SHARE:
        problems.append(
            f"the keyed answer is the longest choice in {len(longest)}/{len(mcqs)} "
            f"({longest_share:.0%}); ceiling is {MAX_LONGEST_SHARE:.0%}"
        )
    if position_share > MAX_POSITION_SHARE:
        problems.append(
            f"the keyed answer is {chr(65 + worst_position)} in {worst_count}/{len(mcqs)} "
            f"({position_share:.0%}); ceiling is {MAX_POSITION_SHARE:.0%}"
        )
    if wide:
        problems.append(f"{len(wide)} question(s) exceed a {MAX_LENGTH_SPREAD}x length spread")

    if verbose:
        print(f"{len(mcqs)} MCQs")
        print(f"  keyed answer is longest : {len(longest)}/{len(mcqs)} ({longest_share:.0%})")
        print(f"  position distribution   : "
              + ", ".join(f"{chr(65 + i)}={positions.get(i, 0)}" for i in range(4)))
        print(f"  widest length spread    : {max(spread.values()):.2f}x")
        if longest:
            print("\n  correct answer is longest in:")
            for t in longest:
                print(f"    {t.id:<44} lens={lengths[t.id]}")
        if wide:
            print("\n  over the spread ceiling:")
            for t in wide:
                print(f"    {t.id:<44} {spread[t.id]:.2f}x lens={lengths[t.id]}")

    if problems:
        print("\nFAIL:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nno exploitable answer-shape bias")
    return 0


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
    parser.add_argument("--mcq-bias", action="store_true",
                        help="check MCQ answer shape instead of running references")
    args = parser.parse_args(argv)
    if args.mcq_bias:
        return audit_mcq_bias()
    return audit(args.pattern, args.workers)


if __name__ == "__main__":
    sys.exit(main())
