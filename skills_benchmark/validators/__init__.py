"""
Validators for benchmark code analysis.
"""

from .structural import StructuralValidator, StructuralResult
from .deprecated import DeprecatedPatternChecker, DeprecatedPattern

__all__ = [
    "StructuralValidator",
    "StructuralResult",
    "DeprecatedPatternChecker",
    "DeprecatedPattern"
]
