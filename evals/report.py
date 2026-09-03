#!/usr/bin/env python3
"""Turn a results JSONL into the numbers you actually want.

    python report.py results/grid.jsonl
    python report.py results/grid.jsonl --by tag --markdown report.md

Headline table is per condition: pass rate with a Wilson interval (agents are
stochastic, so a bare percentage over a handful of runs is noise), plus the two
efficiency numbers that decide whether a resource is worth its context —
**tokens per solve** and **API-equivalent dollars per solve** (on a subscription
the dollars are a proxy for usage, not a charge).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves sensibly at 0/N and N/N, unlike normal-approx."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def load(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarize(records: list[dict], key) -> dict[str, dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        for value in key(record):
            groups[value].append(record)

    summary = {}
    for name, rows in groups.items():
        total = len(rows)
        passed = sum(1 for r in rows if r.get("passed"))
        tokens = sum(r.get("agent", {}).get("total_tokens", 0) for r in rows)
        cost = sum(r.get("agent", {}).get("cost_usd", 0.0) for r in rows)
        latencies = [r.get("agent", {}).get("duration_seconds", 0.0) for r in rows]
        turns = [r.get("agent", {}).get("num_turns", 0) for r in rows]
        reads = [len(r.get("agent", {}).get("files_read", []) or []) for r in rows]
        skill_hits = sum(1 for r in rows if r.get("agent", {}).get("skills_used"))
        low, high = wilson(passed, total)
        summary[name] = {
            "n": total,
            "passed": passed,
            "rate": passed / total if total else 0.0,
            "ci": (low, high),
            "tokens": tokens,
            "cost": cost,
            "tokens_per_solve": tokens / passed if passed else float("inf"),
            "cost_per_solve": cost / passed if passed else float("inf"),
            "median_latency": statistics.median(latencies) if latencies else 0.0,
            "mean_turns": statistics.mean(turns) if turns else 0.0,
            "mean_reads": statistics.mean(reads) if reads else 0.0,
            "skill_use_rate": skill_hits / total if total else 0.0,
        }
    return summary


def format_table(summary: dict[str, dict], label: str, order: list[str] | None = None) -> str:
    names = order or sorted(summary, key=lambda n: -summary[n]["rate"])
    names = [n for n in names if n in summary]
    width = max([len(label)] + [len(n) for n in names])
    lines = [
        f"| {label:<{width}} | pass | 95% CI | n | tok/solve | $-equiv/solve | med s | turns | reads |",
        f"|{'-' * (width + 2)}|------|--------|---|-----------|---------------|-------|-------|-------|",
    ]
    for name in names:
        s = summary[name]
        tps = "—" if s["tokens_per_solve"] == float("inf") else f"{s['tokens_per_solve']:,.0f}"
        cps = "—" if s["cost_per_solve"] == float("inf") else f"${s['cost_per_solve']:.3f}"
        lines.append(
            f"| {name:<{width}} | {s['rate']:.0%} | {s['ci'][0]:.0%}-{s['ci'][1]:.0%} "
            f"| {s['n']} | {tps} | {cps} | {s['median_latency']:.0f} "
            f"| {s['mean_turns']:.1f} | {s['mean_reads']:.1f} |"
        )
    return "\n".join(lines)


def failure_taxonomy(records: list[dict]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        if record.get("passed"):
            continue
        outcome = record.get("outcome") or {}
        label = outcome.get("error_type") or "AgentError"
        counts[label] += 1
    if not counts:
        return "_no failures_"
    lines = ["| failure | count |", "|---------|-------|"]
    for label, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {label} | {count} |")
    return "\n".join(lines)


def resource_usage(records: list[dict]) -> str:
    """Which files agents actually opened, per condition — does routing work?"""
    per_condition: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        condition = record["condition"]
        for path in record.get("agent", {}).get("files_read", []) or []:
            per_condition[condition][Path(path).name] += 1
        for skill in record.get("agent", {}).get("skills_used", []) or []:
            per_condition[condition][f"skill:{skill}"] += 1
    if not per_condition:
        return "_no tool use recorded (static conditions only)_"
    lines = []
    for condition in sorted(per_condition):
        top = sorted(per_condition[condition].items(), key=lambda kv: -kv[1])[:8]
        listed = ", ".join(f"{name} ({count})" for name, count in top)
        lines.append(f"- **{condition}**: {listed}")
    return "\n".join(lines)


def build_report(records: list[dict], condition_order: list[str] | None) -> str:
    if not records:
        return "no records"

    parts = [
        "# nnsight resource evaluation",
        "",
        f"{len(records)} runs · "
        f"{len({r['task_id'] for r in records})} tasks · "
        f"{len({r['condition'] for r in records})} conditions · "
        f"{len({r['model'] for r in records})} model(s) · "
        f"{sum(r.get('agent', {}).get('total_tokens', 0) for r in records):,} tokens "
        f"(${sum(r.get('agent', {}).get('cost_usd', 0) for r in records):.2f} API-equivalent)",
        "",
        "## By resource condition",
        "",
        format_table(summarize(records, lambda r: [r["condition"]]), "condition", condition_order),
        "",
        "## By task kind",
        "",
        format_table(summarize(records, lambda r: [r["kind"]]), "kind"),
        "",
        "## By difficulty",
        "",
        format_table(
            summarize(records, lambda r: [r["difficulty"]]),
            "difficulty",
            ["basic", "intermediate", "advanced"],
        ),
        "",
        "## By model",
        "",
        format_table(summarize(records, lambda r: [r["model"]]), "model"),
        "",
        "## Condition x kind",
        "",
        format_table(
            summarize(records, lambda r: [f"{r['condition']} / {r['kind']}"]), "cell"
        ),
        "",
        "## Failures",
        "",
        failure_taxonomy(records),
        "",
        "## What the agents opened",
        "",
        resource_usage(records),
        "",
    ]
    return "\n".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--by", choices=["tag"], help="add a per-tag breakdown")
    parser.add_argument("--markdown", type=Path, help="also write the report to a file")
    parser.add_argument("--order", nargs="+", help="condition display order")
    args = parser.parse_args(argv)

    records = load(args.results)
    report = build_report(records, args.order)

    if args.by == "tag":
        report += "\n## By tag\n\n" + format_table(
            summarize(records, lambda r: r.get("tags") or ["untagged"]), "tag"
        ) + "\n"

    print(report)
    if args.markdown:
        args.markdown.write_text(report)
        print(f"\nwritten to {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
