"""Print the verification table from the last test run (`tests/.report.json`)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPORT = Path(__file__).with_name(".report.json")


def main() -> int:
    if not REPORT.exists():
        print("no report — run `make test` first", file=sys.stderr)
        return 1

    entries = json.loads(REPORT.read_text())
    per_file: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    reasons: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        per_file[entry["file"]][entry["status"]] += 1
        if entry["status"] == "skipped" and entry["detail"]:
            reasons[entry["file"]].add(entry["detail"])

    width = max(len(name) for name in per_file)
    print(f"{'file':<{width}}  {'ran':>4} {'compiled':>9} {'skipped':>8}  why skipped")
    print("-" * (width + 40))
    for name in sorted(per_file):
        counts = per_file[name]
        print(
            f"{name:<{width}}  {counts['ran']:>4} {counts['compiled']:>9} "
            f"{counts['skipped']:>8}  {'; '.join(sorted(reasons[name]))}"
        )

    totals = defaultdict(int)
    for counts in per_file.values():
        for status, n in counts.items():
            totals[status] += n
    print("-" * (width + 40))
    print(f"{'TOTAL':<{width}}  {totals['ran']:>4} {totals['compiled']:>9} {totals['skipped']:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
