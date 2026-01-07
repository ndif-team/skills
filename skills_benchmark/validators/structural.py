"""
Structural validator for checking required/forbidden patterns in generated code.
"""

import re
import ast
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from schema import ValidationRules, ForbiddenPattern


@dataclass
class StructuralResult:
    """Result of structural validation."""
    must_include_results: Dict[str, bool] = field(default_factory=dict)
    must_not_include_results: Dict[str, Tuple[bool, str]] = field(default_factory=dict)
    syntax_valid: bool = True
    syntax_error: Optional[str] = None

    @property
    def must_include_score(self) -> float:
        """Fraction of required patterns found."""
        if not self.must_include_results:
            return 1.0
        found = sum(1 for v in self.must_include_results.values() if v)
        return found / len(self.must_include_results)

    @property
    def has_forbidden_patterns(self) -> bool:
        """Whether any forbidden patterns were found."""
        return any(found for found, _ in self.must_not_include_results.values())

    @property
    def all_required_found(self) -> bool:
        """Whether all required patterns were found."""
        return all(self.must_include_results.values())

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = []

        if not self.syntax_valid:
            lines.append(f"Syntax Error: {self.syntax_error}")
            return "\n".join(lines)

        # Required patterns
        if self.must_include_results:
            lines.append("Required patterns:")
            for pattern, found in self.must_include_results.items():
                status = "FOUND" if found else "MISSING"
                lines.append(f"  [{status}] {pattern}")

        # Forbidden patterns
        if self.must_not_include_results:
            lines.append("\nForbidden patterns:")
            for pattern, (found, reason) in self.must_not_include_results.items():
                if found:
                    lines.append(f"  [VIOLATION] {pattern}")
                    lines.append(f"              Reason: {reason}")

        # Score
        lines.append(f"\nScore: {self.must_include_score:.1%} required patterns found")
        if self.has_forbidden_patterns:
            lines.append("WARNING: Forbidden patterns detected!")

        return "\n".join(lines)


class StructuralValidator:
    """Validates code structure against required/forbidden patterns."""

    def __init__(self):
        pass

    def validate(self, code: str, rules: ValidationRules) -> StructuralResult:
        """
        Validate code against the given rules.

        Args:
            code: The generated code to validate
            rules: Validation rules specifying required/forbidden patterns

        Returns:
            StructuralResult with validation details
        """
        result = StructuralResult()

        # Check syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            result.syntax_valid = False
            result.syntax_error = str(e)
            return result

        # Check required patterns
        for pattern in rules.must_include:
            result.must_include_results[pattern] = self._pattern_found(code, pattern)

        # Check forbidden patterns
        for forbidden in rules.must_not_include:
            found = self._pattern_found(code, forbidden.pattern)
            result.must_not_include_results[forbidden.pattern] = (found, forbidden.reason)

        return result

    def _pattern_found(self, code: str, pattern: str) -> bool:
        """
        Check if a pattern is found in the code.

        Supports both literal string matching and regex patterns.
        """
        # First try literal match
        if pattern in code:
            return True

        # Then try regex
        try:
            if re.search(pattern, code, re.MULTILINE | re.DOTALL):
                return True
        except re.error:
            # Invalid regex, treat as literal only
            pass

        return False

    def validate_from_file(self, code: str, query_path: Path) -> StructuralResult:
        """
        Validate code against rules from a query YAML file.

        Args:
            code: The generated code to validate
            query_path: Path to the query YAML file

        Returns:
            StructuralResult with validation details
        """
        from schema import BenchmarkQuery

        query = BenchmarkQuery.from_yaml(query_path)
        return self.validate(code, query.validation)


def validate_code(code: str, rules: ValidationRules) -> StructuralResult:
    """Convenience function for one-off validation."""
    validator = StructuralValidator()
    return validator.validate(code, rules)


# CLI interface
if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Validate code against structural rules")
    parser.add_argument("--code", "-c", help="Code string to validate")
    parser.add_argument("--file", "-f", help="File containing code to validate")
    parser.add_argument("--query", "-q", help="Query YAML file with validation rules")

    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            code = f.read()
    elif args.code:
        code = args.code
    else:
        print("Error: Must provide --code or --file")
        sys.exit(1)

    if args.query:
        validator = StructuralValidator()
        result = validator.validate_from_file(code, Path(args.query))
    else:
        # Default minimal rules
        rules = ValidationRules(must_include=["model.trace", ".save()"])
        result = validate_code(code, rules)

    print(result.summary())
    sys.exit(0 if result.all_required_found and not result.has_forbidden_patterns else 1)
