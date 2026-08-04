"""
Schema definitions for benchmark queries and results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal
from enum import Enum
import yaml
from pathlib import Path


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Skill(str, Enum):
    NNSIGHT_BASICS = "nnsight-basics"
    LOGIT_LENS = "logit-lens"
    ACTIVATION_PATCHING = "activation-patching"
    ATTRIBUTION_PATCHING = "attribution-patching"
    CAUSAL_TRACING = "causal-tracing"
    MODEL_STEERING = "model-steering"


@dataclass
class ForbiddenPattern:
    """A pattern that should not appear in the solution."""
    pattern: str
    reason: str


@dataclass
class ExpectedOutput:
    """Expected characteristics of the output."""
    type: str  # "tensor", "list", "dict", "float", etc.
    shape: Optional[List[int]] = None
    value_range: Optional[tuple] = None  # (min, max)


@dataclass
class ValidationRules:
    """Rules for validating generated code."""
    must_include: List[str] = field(default_factory=list)
    must_not_include: List[ForbiddenPattern] = field(default_factory=list)
    expected_output: Optional[ExpectedOutput] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "ValidationRules":
        must_not = []
        for item in data.get("must_not_include", []):
            if isinstance(item, dict):
                must_not.append(ForbiddenPattern(**item))
            else:
                must_not.append(ForbiddenPattern(pattern=item, reason="Forbidden pattern"))

        expected = None
        if "expected_output" in data:
            expected = ExpectedOutput(**data["expected_output"])

        return cls(
            must_include=data.get("must_include", []),
            must_not_include=must_not,
            expected_output=expected
        )


@dataclass
class BenchmarkQuery:
    """A single benchmark query."""
    id: str
    skill: Skill
    difficulty: Difficulty
    title: str
    query: str
    expected_concepts: List[str]
    validation: ValidationRules
    reference_solution: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchmarkQuery":
        """Load a query from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            id=data["id"],
            skill=Skill(data["skill"]),
            difficulty=Difficulty(data["difficulty"]),
            title=data["title"],
            query=data["query"],
            expected_concepts=data.get("expected_concepts", []),
            validation=ValidationRules.from_dict(data.get("validation", {})),
            reference_solution=data.get("reference_solution"),
            tags=data.get("tags", [])
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "skill": self.skill.value,
            "difficulty": self.difficulty.value,
            "title": self.title,
            "query": self.query,
            "expected_concepts": self.expected_concepts,
            "tags": self.tags
        }


@dataclass
class ValidationResult:
    """Result of validating a single piece of generated code."""
    must_include_results: Dict[str, bool]  # pattern -> found
    must_not_include_results: Dict[str, bool]  # pattern -> found (True = violation)
    deprecated_patterns: List[Dict[str, Any]]  # List of deprecated pattern findings

    @property
    def must_include_score(self) -> float:
        """Fraction of required patterns found."""
        if not self.must_include_results:
            return 1.0
        return sum(self.must_include_results.values()) / len(self.must_include_results)

    @property
    def has_forbidden_patterns(self) -> bool:
        """Whether any forbidden patterns were found."""
        return any(self.must_not_include_results.values())

    @property
    def has_deprecated_patterns(self) -> bool:
        """Whether any deprecated patterns were found."""
        return len(self.deprecated_patterns) > 0


@dataclass
class ExecutionResult:
    """Result of executing generated code."""
    executed: bool
    error_message: Optional[str] = None
    output: Optional[Any] = None
    output_type: Optional[str] = None
    output_shape: Optional[List[int]] = None
    execution_time_ms: Optional[float] = None


@dataclass
class RunMetrics:
    """Metrics for a single run of a query."""
    executes: bool
    produces_output: bool
    uses_correct_api: float  # 0-1
    avoids_deprecated: bool
    output_correct: float  # 0-1
    output_shape_match: bool
    forward_passes: Optional[int] = None

    @property
    def score(self) -> float:
        """Compute composite score."""
        return (
            0.25 * float(self.executes) +
            0.20 * self.uses_correct_api +
            0.15 * float(self.avoids_deprecated) +
            0.40 * self.output_correct
        )


@dataclass
class QueryRun:
    """A single run of a benchmark query."""
    generated_code: str
    validation_result: ValidationResult
    execution_result: ExecutionResult
    metrics: RunMetrics

    def to_dict(self) -> Dict:
        return {
            "generated_code": self.generated_code,
            "execution_time_ms": self.execution_result.execution_time_ms,
            "metrics": {
                "executes": self.metrics.executes,
                "uses_correct_api": self.metrics.uses_correct_api,
                "avoids_deprecated": self.metrics.avoids_deprecated,
                "output_correct": self.metrics.output_correct,
                "forward_passes": self.metrics.forward_passes
            },
            "score": self.metrics.score
        }


@dataclass
class QueryResult:
    """Aggregated results for a single query across multiple runs."""
    query_id: str
    runs: List[QueryRun]

    @property
    def mean_score(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.metrics.score for r in self.runs) / len(self.runs)

    @property
    def std_score(self) -> float:
        if len(self.runs) < 2:
            return 0.0
        mean = self.mean_score
        variance = sum((r.metrics.score - mean) ** 2 for r in self.runs) / len(self.runs)
        return variance ** 0.5

    def to_dict(self) -> Dict:
        return {
            "query_id": self.query_id,
            "runs": [r.to_dict() for r in self.runs],
            "aggregate": {
                "mean_score": self.mean_score,
                "std_score": self.std_score
            }
        }


@dataclass
class BenchmarkMetadata:
    """Metadata for a benchmark run."""
    agent: str
    with_skills: bool
    timestamp: str
    nnsight_version: str
    num_runs_per_query: int = 3
    model: Optional[str] = None
    claude_version: Optional[str] = None


@dataclass
class BenchmarkResults:
    """Complete results of a benchmark run."""
    metadata: BenchmarkMetadata
    results: List[QueryResult]

    @property
    def overall_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.mean_score for r in self.results) / len(self.results)

    def score_by_difficulty(self) -> Dict[str, float]:
        """Compute average score by difficulty level."""
        # This requires access to query metadata - simplified version
        return {}

    def score_by_skill(self) -> Dict[str, float]:
        """Compute average score by skill."""
        # This requires access to query metadata - simplified version
        return {}

    def to_dict(self) -> Dict:
        metadata_dict = {
            "agent": self.metadata.agent,
            "with_skills": self.metadata.with_skills,
            "timestamp": self.metadata.timestamp,
            "nnsight_version": self.metadata.nnsight_version,
            "num_runs_per_query": self.metadata.num_runs_per_query,
        }
        if self.metadata.model:
            metadata_dict["model"] = self.metadata.model
        if self.metadata.claude_version:
            metadata_dict["claude_version"] = self.metadata.claude_version
        return {
            "metadata": metadata_dict,
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "overall_score": self.overall_score,
                "num_queries": len(self.results)
            }
        }


def load_all_queries(benchmark_dir: Path) -> List[BenchmarkQuery]:
    """Load all benchmark queries from the queries directory."""
    queries = []
    queries_dir = benchmark_dir / "queries"

    for difficulty in ["easy", "medium", "hard"]:
        difficulty_dir = queries_dir / difficulty
        if difficulty_dir.exists():
            for yaml_file in difficulty_dir.glob("*.yaml"):
                try:
                    queries.append(BenchmarkQuery.from_yaml(yaml_file))
                except Exception as e:
                    print(f"Warning: Failed to load {yaml_file}: {e}")

    return queries


def load_queries_by_filter(
    benchmark_dir: Path,
    difficulty: Optional[Difficulty] = None,
    skill: Optional[Skill] = None,
    tags: Optional[List[str]] = None
) -> List[BenchmarkQuery]:
    """Load queries matching the specified filters."""
    all_queries = load_all_queries(benchmark_dir)

    filtered = all_queries

    if difficulty:
        filtered = [q for q in filtered if q.difficulty == difficulty]

    if skill:
        filtered = [q for q in filtered if q.skill == skill]

    if tags:
        filtered = [q for q in filtered if any(t in q.tags for t in tags)]

    return filtered
