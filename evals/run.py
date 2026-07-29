#!/usr/bin/env python3
"""Run the grid: tasks x resource conditions x models x repeats.

Results are appended to a JSONL file, one record per run, so a sweep can be
interrupted and resumed without repeating completed cells.

On a Claude subscription nothing here is billed — the `claude` CLI authenticates
with your existing login, and the dollar figures it reports are the API-EQUIVALENT
price of the tokens, not a charge. Budget with --max-tokens; --max-cost is the
same ceiling expressed in that proxy currency.

    # what would run, and how much usage it should take
    python run.py --dry-run

    # a cheap smoke sweep
    python run.py --conditions none skills --tasks basic_01_trace_and_save --repeats 1

    # the full grid, resumable, with a usage ceiling
    python run.py --models sonnet opus --repeats 3 --max-tokens 100_000_000 --output results/grid.jsonl
    python run.py --resume --output results/grid.jsonl      # continue where it stopped

Metrics recorded per run: pass/fail, wall-clock for the agent and for execution,
tokens by class, API-equivalent dollars, turns, which resource files were read,
which skills fired, and a failure class for anything that did not pass.

Account-level failures (usage window, expired login) stop the sweep instead of
being recorded as task failures — otherwise a limit hit mid-run would silently
turn into a wall of zeros in the results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evalkit.conditions import CONDITIONS, DEFAULT_GRID, get_condition  # noqa: E402
from evalkit.providers import get_provider  # noqa: E402
from evalkit.registry import Difficulty, TaskKind, load_all, select  # noqa: E402
from evalkit.runner import run_task  # noqa: E402

# Rough per-task cost, measured on the smoke runs. Only used by --dry-run.
# Rough per-task usage, measured on the smoke runs. `cost` here is the
# API-EQUIVALENT price the CLI reports; on a Claude subscription nothing is
# billed against it, so treat it as a proxy for how much usage a sweep burns.
# `tokens` is the number that actually matters against a subscription window.
COST_HINTS = {"none": 0.005, "static": 0.12, "agentic": 0.11}
TOKEN_HINTS = {"none": 2_000, "static": 60_000, "agentic": 55_000}


def preflight() -> None:
    """Fail before spending anything if the environment cannot run a task.

    Learned the hard way: nnsight is installed editable from a shared checkout,
    and a branch switch there (0.8 -> a feature branch with a different public
    API) turned every code task into an ImportError. MCQs kept passing, so the
    sweep looked like a catastrophic model regression rather than a broken
    import. One import check up front distinguishes the two in a second.
    """
    try:
        import nnsight
        from nnsight import TransformersModel  # noqa: F401
    except ImportError as exc:
        source = getattr(__import__("nnsight"), "__file__", "?") if "nnsight" in sys.modules else "?"
        raise SystemExit(
            f"preflight failed: {exc}\n"
            f"nnsight resolves to {source}\n"
            "Code tasks cannot run. If nnsight is an editable install, check which "
            "branch that checkout is on — the eval suite targets 0.8."
        ) from exc

    print(f"nnsight {nnsight.__version__} from {nnsight.__file__}")


def cell_key(record: dict) -> tuple:
    return (record["model"], record["condition"], record["task_id"], record["repeat"])


def load_done(path: Path) -> set[tuple]:
    done: set[tuple] = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("passed") is not None or record.get("agent", {}).get("ok"):
            done.add(cell_key(record))
    return done


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_GRID,
                        help=f"any of: {', '.join(CONDITIONS)}")
    parser.add_argument("--models", nargs="+", default=["sonnet"])
    parser.add_argument("--provider", default="claude-code", choices=["claude-code", "anthropic"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--tasks", nargs="+", help="explicit task ids")
    parser.add_argument("--kinds", nargs="+", choices=[k.value for k in TaskKind])
    parser.add_argument("--difficulties", nargs="+", choices=[d.value for d in Difficulty])
    parser.add_argument("--tags", nargs="+")
    parser.add_argument("--output", default="results/grid.jsonl")
    parser.add_argument("--resume", action="store_true", help="skip cells already in the output file")
    parser.add_argument("--dry-run", action="store_true", help="list the grid and estimate cost")
    parser.add_argument("--max-cost", type=float,
                        help="stop once API-equivalent cost reaches this (proxy, not billed on a subscription)")
    parser.add_argument("--max-tokens", type=int,
                        help="stop once this many tokens have been used (the real budget on a subscription)")
    parser.add_argument("--timeout", type=int, default=900, help="per-agent-call timeout")
    args = parser.parse_args(argv)

    if not args.dry_run:
        preflight()

    load_all()
    tasks = select(
        ids=args.tasks,
        kinds=[TaskKind(k) for k in args.kinds] if args.kinds else None,
        difficulties=[Difficulty(d) for d in args.difficulties] if args.difficulties else None,
        tags=args.tags,
    )
    if not tasks:
        print("no tasks matched", file=sys.stderr)
        return 1

    conditions = [get_condition(name) for name in args.conditions]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(output) if args.resume else set()

    cells = [
        (model, condition, task, repeat)
        for model in args.models
        for condition in conditions
        for task in tasks
        for repeat in range(args.repeats)
    ]
    pending = [
        cell for cell in cells
        if (cell[0], cell[1].name, cell[2].id, cell[3]) not in done
    ]

    estimate = sum(
        COST_HINTS.get(cell[1].name, COST_HINTS.get(cell[1].mode, 0.1)) for cell in pending
    )
    token_estimate = sum(
        TOKEN_HINTS.get(cell[1].name, TOKEN_HINTS.get(cell[1].mode, 50_000)) for cell in pending
    )
    print(f"{len(tasks)} tasks x {len(conditions)} conditions x {len(args.models)} models "
          f"x {args.repeats} repeats = {len(cells)} cells")
    if done:
        print(f"{len(cells) - len(pending)} already done, {len(pending)} to run")
    print(f"rough usage estimate: {token_estimate:,} tokens "
          f"(~${estimate:.2f} API-equivalent; not billed on a Claude subscription)\n")

    if args.dry_run:
        for model in args.models:
            for condition in conditions:
                print(f"  {model:<8} {condition.name:<16} {condition.mode:<8} "
                      f"{len(tasks)} tasks x {args.repeats}")
        return 0

    providers = {model: get_provider(args.provider, model, timeout=args.timeout) for model in args.models}

    spent = 0.0
    tokens_used = 0
    passed = 0
    started_all = time.time()

    with output.open("a") as handle:
        for index, (model, condition, task, repeat) in enumerate(pending, start=1):
            if args.max_cost is not None and spent >= args.max_cost:
                print(f"\nstopping: ${spent:.2f} API-equivalent reached the ${args.max_cost:.2f} ceiling")
                break
            if args.max_tokens is not None and tokens_used >= args.max_tokens:
                print(f"\nstopping: {tokens_used:,} tokens reached the {args.max_tokens:,} ceiling")
                break

            try:
                response = providers[model].ask(task.user_prompt(), condition)
            except Exception as exc:  # noqa: BLE001 — recorded, never fatal to the sweep
                response = None
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            else:
                error = "" if response.ok else response.error

            if response is not None and getattr(response, "error_kind", "") == "limit":
                print(
                    f"\nSTOPPING — the account, not the task, failed:\n  {response.error[:300]}\n\n"
                    "This cell was NOT recorded, so it will be retried. Resume with:\n"
                    f"  python run.py --resume --output {output} ...\n"
                )
                break

            if response is not None and response.ok:
                outcome = run_task(task, response.text)
            else:
                outcome = None

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "provider": args.provider,
                "model": model,
                "condition": condition.name,
                "condition_mode": condition.mode,
                "repeat": repeat,
                "task_id": task.id,
                "kind": task.kind.value,
                "difficulty": task.difficulty.value,
                "tags": task.tags,
                "passed": bool(outcome.passed) if outcome else False,
                "agent": response.to_dict() if response else {"ok": False, "error": error},
                "outcome": outcome.to_dict() if outcome else None,
                "agent_error": error,
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()

            if response is not None:
                spent += response.cost_usd
                tokens_used += response.total_tokens
            passed += int(record["passed"])

            mark = "PASS" if record["passed"] else "fail"
            detail = ""
            if outcome and not outcome.passed:
                detail = f" [{outcome.error_type}]"
            elif error:
                detail = f" [agent: {error[:40]}]"
            print(
                f"{index:>4}/{len(pending)} {mark} {model:<7} {condition.name:<15} "
                f"{task.id:<44} {tokens_used / 1000:6.0f}k{detail}"
            )

    elapsed = time.time() - started_all
    attempted = min(len(pending), index if pending else 0)
    print(
        f"\n{passed}/{attempted} passed  |  {tokens_used:,} tokens  "
        f"|  ${spent:.2f} API-equivalent  |  {elapsed / 60:.1f} min"
    )
    print(f"records appended to {output}")
    print(f"report with:  python report.py {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
