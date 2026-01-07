"""
Claude API benchmark runner.

This runner calls the Anthropic API directly to generate code for benchmark queries.
"""

import re
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from schema import BenchmarkQuery
from runners.base import BenchmarkRunner, RunConfig


class ClaudeAPIRunner(BenchmarkRunner):
    """
    Benchmark runner that uses Claude API directly.

    This runner calls the Anthropic API directly, which can be faster
    and more controllable than the CLI.

    Requirements:
    - ANTHROPIC_API_KEY environment variable must be set
    - anthropic package must be installed

    Usage:
        config = RunConfig(agent="claude-api", with_skills=True)
        runner = ClaudeAPIRunner(config, model="claude-sonnet-4-20250514")
        results = runner.run_benchmark(benchmark_dir)
    """

    def __init__(self, config: RunConfig, model: str = "claude-sonnet-4-20250514"):
        super().__init__(config)
        self.model = model
        self._client = None

    @property
    def client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError:
                raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
        return self._client

    def _build_system_prompt(self) -> str:
        """Build system prompt, optionally including skill content."""
        base_prompt = """You are an expert Python developer specializing in neural network interpretability using the nnsight library.

Generate clean, efficient, and correct Python code following nnsight 0.5+ best practices."""

        if self.config.with_skills:
            # Load skill content
            skills_dir = Path(__file__).parent.parent.parent / "plugins" / "nnsight" / "skills"
            skill_content = []

            for skill_dir in skills_dir.iterdir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    content = skill_file.read_text()
                    skill_content.append(f"## {skill_dir.name}\n\n{content}")

            if skill_content:
                base_prompt += "\n\n# Reference Documentation\n\n"
                base_prompt += "\n\n---\n\n".join(skill_content)

        return base_prompt

    def _build_user_prompt(self, query: BenchmarkQuery) -> str:
        """Build user prompt for the query."""
        return f"""Write Python code to solve this task using nnsight:

{query.query}

Requirements:
- Use the nnsight library (version 0.5+)
- Include all necessary imports
- The code should be complete and runnable

Respond with ONLY the Python code inside a single code block, no explanations."""

    def _extract_code(self, response: str) -> str:
        """Extract Python code from response."""
        code_block_pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            return max(matches, key=len).strip()

        return response.strip()

    def generate_code(self, query: BenchmarkQuery) -> str:
        """
        Generate code using Claude API.

        Args:
            query: The benchmark query to solve

        Returns:
            Generated code string
        """
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self._build_system_prompt(),
                messages=[
                    {"role": "user", "content": self._build_user_prompt(query)}
                ]
            )

            response_text = message.content[0].text
            return self._extract_code(response_text)

        except Exception as e:
            return f"# Error calling API: {e}\npass"

    @staticmethod
    def generate_output_filename() -> str:
        """Generate timestamped output filename."""
        now = datetime.now()
        return f"claude_{now.strftime('%Y_%m_%d_%H%M')}.json"


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmark with Claude API")
    parser.add_argument("--with-skills", action="store_true", default=True)
    parser.add_argument("--no-skills", action="store_true")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="Model to use")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    parser.add_argument("--skill", help="Filter by skill name")
    parser.add_argument("--output", "-o", help="Output file path (default: auto-generated)")

    args = parser.parse_args()

    from schema import Difficulty, Skill

    config = RunConfig(
        agent="claude-api",
        with_skills=not args.no_skills,
        num_runs=args.num_runs,
        difficulty=Difficulty(args.difficulty) if args.difficulty else None,
        skill=Skill(args.skill) if args.skill else None,
    )

    runner = ClaudeAPIRunner(config, model=args.model)

    benchmark_dir = Path(__file__).parent.parent
    results = runner.run_benchmark(benchmark_dir)

    # Update metadata with model info
    results.metadata.model = args.model

    output_path = Path(args.output) if args.output else Path("results") / runner.generate_output_filename()
    runner.save_results(results, output_path)

    print(f"\nOverall score: {results.overall_score:.2%}")
