"""
Deprecated pattern checker for detecting pre-NNsight-0.5 patterns.
"""

import re
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class Severity(str, Enum):
    """Severity level of deprecated pattern."""
    ERROR = "error"      # Will definitely fail
    WARNING = "warning"  # May cause issues
    INFO = "info"        # Style/best practice


@dataclass
class DeprecatedPattern:
    """A deprecated pattern found in code."""
    pattern_name: str
    pattern: str
    reason: str
    replacement: str
    severity: Severity
    line_number: Optional[int] = None
    matched_text: Optional[str] = None


# Deprecated patterns in NNsight 0.5
DEPRECATED_PATTERNS = [
    {
        "name": "grad_before_backward",
        "pattern": r"\.grad\.save\(\).*?\.backward\(\)",
        "reason": "In NNsight 0.5, gradients must be accessed inside .backward() context",
        "replacement": "with metric.backward():\n    grad = tensor.grad.save()",
        "severity": Severity.ERROR,
        "flags": re.DOTALL
    },
    {
        "name": "cross_prompt_no_barrier",
        "pattern": r"with\s+tracer\.invoke\([^)]+\):.*?(\w+)\s*=\s*[^=].*?\.output.*?with\s+tracer\.invoke\([^)]+\):.*?\1",
        "reason": "Cross-prompt interventions require barrier() in NNsight 0.5",
        "replacement": "barrier = tracer.barrier(N)\n# ... barrier() calls in each invoke",
        "severity": Severity.WARNING,
        "flags": re.DOTALL
    },
    {
        "name": "nnsight_apply",
        "pattern": r"nnsight\.apply\s*\(",
        "reason": "nnsight.apply() is deprecated, use direct function calls",
        "replacement": "function(tensor)",
        "severity": Severity.ERROR,
        "flags": 0
    },
    {
        "name": "nnsight_cond",
        "pattern": r"nnsight\.cond\s*\(",
        "reason": "nnsight.cond() is deprecated, use standard if/else",
        "replacement": "if condition: ... else: ...",
        "severity": Severity.ERROR,
        "flags": 0
    },
    {
        "name": "nnsight_iter",
        "pattern": r"nnsight\.iter\s*\(",
        "reason": "nnsight.iter() is deprecated, use standard for loops",
        "replacement": "for item in iterable: ...",
        "severity": Severity.ERROR,
        "flags": 0
    },
    {
        "name": "nnsight_log",
        "pattern": r"nnsight\.log\s*\(",
        "reason": "nnsight.log() is deprecated, use standard print()",
        "replacement": "print(...)",
        "severity": Severity.WARNING,
        "flags": 0
    },
    {
        "name": "nnsight_stop",
        "pattern": r"nnsight\.stop\s*\(",
        "reason": "nnsight.stop() is deprecated, use tracer.stop() or breakpoint()",
        "replacement": "tracer.stop() or breakpoint()",
        "severity": Severity.ERROR,
        "flags": 0
    },
    {
        "name": "nnsight_list",
        "pattern": r"nnsight\.list\s*\(",
        "reason": "nnsight.list() is deprecated, use native Python list()",
        "replacement": "list()",
        "severity": Severity.WARNING,
        "flags": 0
    },
    {
        "name": "nnsight_dict",
        "pattern": r"nnsight\.dict\s*\(",
        "reason": "nnsight.dict() is deprecated, use native Python dict()",
        "replacement": "dict()",
        "severity": Severity.WARNING,
        "flags": 0
    },
    {
        "name": "backward_no_context",
        "pattern": r"\.backward\(\s*\)(?!\s*:)",
        "reason": "For gradient interventions, use 'with tensor.backward():' context",
        "replacement": "with tensor.backward():\n    grad = tensor.grad.save()",
        "severity": Severity.INFO,  # Not always wrong, just for intervention
        "flags": 0
    },
    {
        "name": "requires_grad_method",
        "pattern": r"\.requires_grad_\s*\(\s*True\s*\)",
        "reason": "Prefer .requires_grad = True assignment over method call",
        "replacement": "tensor.requires_grad = True",
        "severity": Severity.INFO,
        "flags": 0
    },
    {
        "name": "iter_slice_all",
        "pattern": r"tracer\.iter\[\s*:\s*\]",
        "reason": "For all iterations, tracer.all() is clearer than tracer.iter[:]",
        "replacement": "tracer.all()",
        "severity": Severity.INFO,
        "flags": 0
    },
    {
        "name": "nnsight_save_function",
        "pattern": r"nnsight\.save\s*\(",
        "reason": "nnsight.save() is not documented, use .save() method instead",
        "replacement": "tensor.save()",
        "severity": Severity.WARNING,
        "flags": 0
    },
]


class DeprecatedPatternChecker:
    """Checks code for deprecated NNsight patterns."""

    def __init__(self, include_info: bool = False):
        """
        Initialize the checker.

        Args:
            include_info: Whether to include INFO-level patterns (style suggestions)
        """
        self.include_info = include_info

    def check(self, code: str) -> List[DeprecatedPattern]:
        """
        Check code for deprecated patterns.

        Args:
            code: The code to check

        Returns:
            List of deprecated patterns found
        """
        findings = []

        for pattern_def in DEPRECATED_PATTERNS:
            severity = pattern_def["severity"]

            # Skip INFO patterns if not requested
            if severity == Severity.INFO and not self.include_info:
                continue

            flags = pattern_def.get("flags", 0)
            matches = re.finditer(pattern_def["pattern"], code, flags)

            for match in matches:
                # Calculate line number
                line_num = code[:match.start()].count('\n') + 1

                findings.append(DeprecatedPattern(
                    pattern_name=pattern_def["name"],
                    pattern=pattern_def["pattern"],
                    reason=pattern_def["reason"],
                    replacement=pattern_def["replacement"],
                    severity=severity,
                    line_number=line_num,
                    matched_text=match.group()[:100]  # Truncate long matches
                ))

        return findings

    def check_file(self, filepath: str) -> List[DeprecatedPattern]:
        """Check a file for deprecated patterns."""
        with open(filepath) as f:
            return self.check(f.read())

    def has_errors(self, code: str) -> bool:
        """Check if code has any ERROR-severity deprecated patterns."""
        findings = self.check(code)
        return any(f.severity == Severity.ERROR for f in findings)

    def has_warnings(self, code: str) -> bool:
        """Check if code has any WARNING-severity deprecated patterns."""
        findings = self.check(code)
        return any(f.severity == Severity.WARNING for f in findings)

    def summary(self, findings: List[DeprecatedPattern]) -> str:
        """Generate a human-readable summary of findings."""
        if not findings:
            return "No deprecated patterns found."

        lines = ["Deprecated patterns found:\n"]

        # Group by severity
        by_severity = {s: [] for s in Severity}
        for f in findings:
            by_severity[f.severity].append(f)

        for severity in [Severity.ERROR, Severity.WARNING, Severity.INFO]:
            items = by_severity[severity]
            if items:
                lines.append(f"\n{severity.value.upper()}S ({len(items)}):")
                for f in items:
                    lines.append(f"  Line {f.line_number}: {f.pattern_name}")
                    lines.append(f"    Reason: {f.reason}")
                    lines.append(f"    Use instead: {f.replacement}")

        return "\n".join(lines)


def check_deprecated(code: str, include_info: bool = False) -> List[DeprecatedPattern]:
    """Convenience function for checking deprecated patterns."""
    checker = DeprecatedPatternChecker(include_info=include_info)
    return checker.check(code)


# CLI interface
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Check code for deprecated NNsight patterns")
    parser.add_argument("--code", "-c", help="Code string to check")
    parser.add_argument("--file", "-f", help="File containing code to check")
    parser.add_argument("--include-info", "-i", action="store_true",
                        help="Include INFO-level patterns (style suggestions)")

    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            code = f.read()
    elif args.code:
        code = args.code
    else:
        print("Error: Must provide --code or --file")
        sys.exit(1)

    checker = DeprecatedPatternChecker(include_info=args.include_info)
    findings = checker.check(code)

    print(checker.summary(findings))

    # Exit with error if any ERROR-level patterns found
    has_errors = any(f.severity == Severity.ERROR for f in findings)
    sys.exit(1 if has_errors else 0)
