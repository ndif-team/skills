"""
Results analysis and comparison for benchmark runs.

This module provides tools for:
- Comparing with-skills vs without-skills results
- Generating reports by difficulty and skill
- Statistical analysis of improvements
"""

import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent))
from schema import BenchmarkQuery, load_all_queries


@dataclass
class ComparisonResult:
    """Result of comparing two benchmark runs."""
    with_skills_score: float
    without_skills_score: float
    improvement: float
    improvement_pct: float

    # Per-difficulty breakdown
    by_difficulty: Dict[str, Tuple[float, float, float]]  # (with, without, improvement)

    # Per-skill breakdown
    by_skill: Dict[str, Tuple[float, float, float]]

    # Individual query improvements
    query_improvements: Dict[str, float]


def load_results(filepath: Path) -> Dict:
    """Load benchmark results from JSON file."""
    with open(filepath) as f:
        return json.load(f)


def get_query_scores(results: Dict) -> Dict[str, float]:
    """Extract query ID -> mean score mapping from results."""
    scores = {}
    for query_result in results.get("results", []):
        query_id = query_result["query_id"]
        # Use aggregate mean_score if available
        if "aggregate" in query_result:
            scores[query_id] = query_result["aggregate"]["mean_score"]
        elif "runs" in query_result and query_result["runs"]:
            # Calculate mean from runs
            run_scores = [r.get("score", 0) for r in query_result["runs"]]
            scores[query_id] = sum(run_scores) / len(run_scores)
    return scores


def compare_results(
    with_skills_path: Path,
    without_skills_path: Path,
    queries: Optional[List[BenchmarkQuery]] = None
) -> ComparisonResult:
    """
    Compare benchmark results with and without skills.

    Args:
        with_skills_path: Path to results with skills enabled
        without_skills_path: Path to results without skills
        queries: Optional list of queries for metadata (difficulty, skill)

    Returns:
        ComparisonResult with detailed comparison
    """
    with_results = load_results(with_skills_path)
    without_results = load_results(without_skills_path)

    with_scores = get_query_scores(with_results)
    without_scores = get_query_scores(without_results)

    # Build query metadata lookup
    query_meta = {}
    if queries:
        for q in queries:
            query_meta[q.id] = {"difficulty": q.difficulty.value, "skill": q.skill.value}

    # Calculate overall scores
    common_queries = set(with_scores.keys()) & set(without_scores.keys())

    if not common_queries:
        raise ValueError("No common queries found between the two result sets")

    with_overall = sum(with_scores[q] for q in common_queries) / len(common_queries)
    without_overall = sum(without_scores[q] for q in common_queries) / len(common_queries)

    improvement = with_overall - without_overall
    improvement_pct = (improvement / without_overall * 100) if without_overall > 0 else 0

    # Per-query improvements
    query_improvements = {
        q: with_scores[q] - without_scores[q]
        for q in common_queries
    }

    # Per-difficulty breakdown
    by_difficulty: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for q in common_queries:
        if q in query_meta:
            diff = query_meta[q]["difficulty"]
            by_difficulty[diff].append((with_scores[q], without_scores[q]))

    difficulty_summary = {}
    for diff, scores in by_difficulty.items():
        with_avg = sum(s[0] for s in scores) / len(scores)
        without_avg = sum(s[1] for s in scores) / len(scores)
        difficulty_summary[diff] = (with_avg, without_avg, with_avg - without_avg)

    # Per-skill breakdown
    by_skill: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for q in common_queries:
        if q in query_meta:
            skill = query_meta[q]["skill"]
            by_skill[skill].append((with_scores[q], without_scores[q]))

    skill_summary = {}
    for skill, scores in by_skill.items():
        with_avg = sum(s[0] for s in scores) / len(scores)
        without_avg = sum(s[1] for s in scores) / len(scores)
        skill_summary[skill] = (with_avg, without_avg, with_avg - without_avg)

    return ComparisonResult(
        with_skills_score=with_overall,
        without_skills_score=without_overall,
        improvement=improvement,
        improvement_pct=improvement_pct,
        by_difficulty=difficulty_summary,
        by_skill=skill_summary,
        query_improvements=query_improvements
    )


def generate_report(comparison: ComparisonResult) -> str:
    """Generate a human-readable report from comparison results."""
    lines = [
        "=" * 60,
        "NNsight Skills Benchmark Comparison Report",
        "=" * 60,
        "",
        "OVERALL RESULTS",
        "-" * 40,
        f"With Skills:    {comparison.with_skills_score:.2%}",
        f"Without Skills: {comparison.without_skills_score:.2%}",
        f"Improvement:    {comparison.improvement:+.2%} ({comparison.improvement_pct:+.1f}%)",
        "",
    ]

    if comparison.by_difficulty:
        lines.extend([
            "BY DIFFICULTY",
            "-" * 40,
        ])
        for diff in ["easy", "medium", "hard"]:
            if diff in comparison.by_difficulty:
                with_s, without_s, imp = comparison.by_difficulty[diff]
                lines.append(f"  {diff.capitalize():8} | "
                           f"With: {with_s:.2%} | "
                           f"Without: {without_s:.2%} | "
                           f"Δ: {imp:+.2%}")
        lines.append("")

    if comparison.by_skill:
        lines.extend([
            "BY SKILL",
            "-" * 40,
        ])
        for skill, (with_s, without_s, imp) in sorted(comparison.by_skill.items()):
            skill_short = skill[:20]
            lines.append(f"  {skill_short:20} | "
                        f"With: {with_s:.2%} | "
                        f"Without: {without_s:.2%} | "
                        f"Δ: {imp:+.2%}")
        lines.append("")

    # Top improvements
    sorted_improvements = sorted(
        comparison.query_improvements.items(),
        key=lambda x: x[1],
        reverse=True
    )

    lines.extend([
        "TOP 5 IMPROVEMENTS",
        "-" * 40,
    ])
    for query_id, imp in sorted_improvements[:5]:
        lines.append(f"  {query_id:30} | Δ: {imp:+.2%}")

    lines.append("")
    lines.extend([
        "BOTTOM 5 (Degradations/Smallest Improvements)",
        "-" * 40,
    ])
    for query_id, imp in sorted_improvements[-5:]:
        lines.append(f"  {query_id:30} | Δ: {imp:+.2%}")

    lines.extend([
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


def analyze_single_run(results_path: Path, queries: List[BenchmarkQuery]) -> str:
    """Analyze a single benchmark run."""
    results = load_results(results_path)
    scores = get_query_scores(results)

    # Build query metadata lookup
    query_meta = {q.id: q for q in queries}

    lines = [
        "=" * 60,
        "NNsight Skills Benchmark Analysis",
        "=" * 60,
        "",
        f"Agent: {results.get('metadata', {}).get('agent', 'unknown')}",
        f"Skills: {'enabled' if results.get('metadata', {}).get('with_skills') else 'disabled'}",
        f"Queries: {len(scores)}",
        "",
    ]

    # Overall score
    overall = sum(scores.values()) / len(scores) if scores else 0
    lines.extend([
        "OVERALL SCORE",
        "-" * 40,
        f"Mean Score: {overall:.2%}",
        "",
    ])

    # By difficulty
    by_difficulty: Dict[str, List[float]] = defaultdict(list)
    for query_id, score in scores.items():
        if query_id in query_meta:
            by_difficulty[query_meta[query_id].difficulty.value].append(score)

    if by_difficulty:
        lines.extend([
            "BY DIFFICULTY",
            "-" * 40,
        ])
        for diff in ["easy", "medium", "hard"]:
            if diff in by_difficulty:
                avg = sum(by_difficulty[diff]) / len(by_difficulty[diff])
                lines.append(f"  {diff.capitalize():8} | {avg:.2%} ({len(by_difficulty[diff])} queries)")
        lines.append("")

    # By skill
    by_skill: Dict[str, List[float]] = defaultdict(list)
    for query_id, score in scores.items():
        if query_id in query_meta:
            by_skill[query_meta[query_id].skill.value].append(score)

    if by_skill:
        lines.extend([
            "BY SKILL",
            "-" * 40,
        ])
        for skill in sorted(by_skill.keys()):
            avg = sum(by_skill[skill]) / len(by_skill[skill])
            lines.append(f"  {skill:25} | {avg:.2%} ({len(by_skill[skill])} queries)")
        lines.append("")

    # Individual query scores
    lines.extend([
        "INDIVIDUAL QUERIES",
        "-" * 40,
    ])
    for query_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"  {query_id:35} | {score:.2%}")

    lines.extend(["", "=" * 60])

    return "\n".join(lines)


# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two benchmark runs")
    compare_parser.add_argument("with_skills", type=Path,
                                help="Path to results with skills")
    compare_parser.add_argument("without_skills", type=Path,
                                help="Path to results without skills")
    compare_parser.add_argument("--output", "-o", type=Path,
                                help="Save report to file")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report for single run")
    report_parser.add_argument("results", type=Path, help="Path to results file")
    report_parser.add_argument("--output", "-o", type=Path,
                               help="Save report to file")

    args = parser.parse_args()

    # Load queries for metadata
    benchmark_dir = Path(__file__).parent
    queries = load_all_queries(benchmark_dir)

    if args.command == "compare":
        comparison = compare_results(
            args.with_skills,
            args.without_skills,
            queries
        )
        report = generate_report(comparison)

        if args.output:
            args.output.write_text(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)

    elif args.command == "report":
        report = analyze_single_run(args.results, queries)

        if args.output:
            args.output.write_text(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)

    else:
        parser.print_help()
