"""
Base benchmark runner interface.

This module provides the infrastructure for running benchmarks against
different AI agents (Claude Code, Codex, etc.).
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from schema import (
    BenchmarkQuery, BenchmarkResults, BenchmarkMetadata,
    QueryResult, QueryRun, ValidationResult, ExecutionResult,
    RunMetrics, load_all_queries, load_queries_by_filter,
    Difficulty, Skill
)
from validators import StructuralValidator, DeprecatedPatternChecker


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""
    agent: str = "claude-code"
    with_skills: bool = True
    num_runs: int = 3
    difficulty: Optional[Difficulty] = None
    skill: Optional[Skill] = None
    tags: Optional[List[str]] = None
    output_dir: Path = Path("results")
    timeout_seconds: int = 120


class BenchmarkRunner(ABC):
    """
    Abstract base class for benchmark runners.

    Subclasses should implement the `generate_code` method for their
    specific AI agent.
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self.structural_validator = StructuralValidator()
        self.deprecated_checker = DeprecatedPatternChecker()

    @abstractmethod
    def generate_code(self, query: BenchmarkQuery) -> str:
        """
        Generate code for the given query using the AI agent.

        Args:
            query: The benchmark query to solve

        Returns:
            Generated code string
        """
        pass

    def validate_code(self, code: str, query: BenchmarkQuery) -> ValidationResult:
        """Validate generated code against query rules."""
        structural_result = self.structural_validator.validate(code, query.validation)
        deprecated_findings = self.deprecated_checker.check(code)

        return ValidationResult(
            must_include_results=structural_result.must_include_results,
            must_not_include_results={
                p: found for p, (found, _) in structural_result.must_not_include_results.items()
            },
            deprecated_patterns=[
                {
                    "name": f.pattern_name,
                    "reason": f.reason,
                    "line": f.line_number,
                    "severity": f.severity.value
                }
                for f in deprecated_findings
            ]
        )

    def execute_code(self, code: str) -> ExecutionResult:
        """
        Execute generated code and capture results.

        Note: This is a simplified implementation. In practice, you'd want
        to run this in a sandboxed environment with GPU access.
        """
        start_time = time.time()

        try:
            # Create a restricted namespace for execution
            namespace: Dict[str, Any] = {}

            # This is a placeholder - real execution would need:
            # - GPU environment
            # - nnsight/torch installed
            # - Sandboxing for safety
            # - Output capture

            # For now, just check syntax
            compile(code, "<generated>", "exec")

            execution_time = (time.time() - start_time) * 1000

            return ExecutionResult(
                executed=True,
                error_message=None,
                output=None,  # Would be populated by actual execution
                output_type=None,
                output_shape=None,
                execution_time_ms=execution_time
            )

        except SyntaxError as e:
            return ExecutionResult(
                executed=False,
                error_message=f"Syntax error: {e}",
                execution_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return ExecutionResult(
                executed=False,
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )

    def compute_metrics(
        self,
        validation: ValidationResult,
        execution: ExecutionResult,
        query: BenchmarkQuery
    ) -> RunMetrics:
        """Compute metrics for a single run."""
        return RunMetrics(
            executes=execution.executed,
            produces_output=execution.output is not None,
            uses_correct_api=validation.must_include_score,
            avoids_deprecated=not validation.has_deprecated_patterns,
            output_correct=1.0 if execution.executed else 0.0,  # Simplified
            output_shape_match=True,  # Would need actual execution to verify
            forward_passes=None  # Would need instrumentation
        )

    def run_query(self, query: BenchmarkQuery) -> QueryResult:
        """Run a single query multiple times."""
        runs = []

        for run_idx in range(self.config.num_runs):
            print(f"  Run {run_idx + 1}/{self.config.num_runs}...")

            # Generate code
            code = self.generate_code(query)

            # Validate
            validation = self.validate_code(code, query)

            # Execute (simplified)
            execution = self.execute_code(code)

            # Compute metrics
            metrics = self.compute_metrics(validation, execution, query)

            runs.append(QueryRun(
                generated_code=code,
                validation_result=validation,
                execution_result=execution,
                metrics=metrics
            ))

        return QueryResult(query_id=query.id, runs=runs)

    def run_benchmark(self, benchmark_dir: Path) -> BenchmarkResults:
        """Run the full benchmark."""
        # Load queries based on config
        queries = load_queries_by_filter(
            benchmark_dir,
            difficulty=self.config.difficulty,
            skill=self.config.skill,
            tags=self.config.tags
        )

        print(f"Running benchmark with {len(queries)} queries...")
        print(f"Agent: {self.config.agent}")
        print(f"Skills: {'enabled' if self.config.with_skills else 'disabled'}")
        print(f"Runs per query: {self.config.num_runs}")
        print()

        results = []
        for i, query in enumerate(queries):
            print(f"Query {i + 1}/{len(queries)}: {query.id} ({query.difficulty.value})")
            result = self.run_query(query)
            results.append(result)
            print(f"  Score: {result.mean_score:.2%}")

        metadata = BenchmarkMetadata(
            agent=self.config.agent,
            with_skills=self.config.with_skills,
            timestamp=datetime.now().isoformat(),
            nnsight_version="0.5.0",
            num_runs_per_query=self.config.num_runs
        )

        return BenchmarkResults(metadata=metadata, results=results)

    def save_results(self, results: BenchmarkResults, filepath: Path):
        """Save results to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"Results saved to {filepath}")


class MockRunner(BenchmarkRunner):
    """
    Mock runner for testing the benchmark infrastructure.

    Returns the reference solution if available, otherwise a placeholder.
    """

    def generate_code(self, query: BenchmarkQuery) -> str:
        if query.reference_solution:
            return query.reference_solution
        return f"# Placeholder for {query.id}\npass"


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run NNsight skills benchmark")
    parser.add_argument("--agent", default="mock",
                        help="Agent to use (mock, claude-code, codex)")
    parser.add_argument("--with-skills", action="store_true", default=True)
    parser.add_argument("--no-skills", action="store_true")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of runs per query")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"],
                        help="Filter by difficulty")
    parser.add_argument("--skill", help="Filter by skill name")
    parser.add_argument("--output", "-o", default="results/benchmark.json",
                        help="Output file path")

    args = parser.parse_args()

    config = RunConfig(
        agent=args.agent,
        with_skills=not args.no_skills,
        num_runs=args.num_runs,
        difficulty=Difficulty(args.difficulty) if args.difficulty else None,
        skill=Skill(args.skill) if args.skill else None,
    )

    # Use mock runner for testing
    runner = MockRunner(config)

    benchmark_dir = Path(__file__).parent.parent
    results = runner.run_benchmark(benchmark_dir)

    output_path = Path(args.output)
    runner.save_results(results, output_path)

    print(f"\nOverall score: {results.overall_score:.2%}")
