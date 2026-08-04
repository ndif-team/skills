"""
Claude Code benchmark runner.

This runner invokes Claude Code CLI to generate code for benchmark queries.
"""

import re
import subprocess
import os
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from schema import BenchmarkQuery
from runners.base import BenchmarkRunner, RunConfig


class ClaudeCodeRunner(BenchmarkRunner):
    """
    Benchmark runner that uses Claude Code CLI to generate code.

    Requirements:
    - Claude Code CLI must be installed and available in PATH
    - For with_skills=True, the nnsight skills plugin must be installed

    Usage:
        config = RunConfig(agent="claude-code", with_skills=True)
        runner = ClaudeCodeRunner(config)
        results = runner.run_benchmark(benchmark_dir)
    """

    def __init__(self, config: RunConfig, model: str | None = None):
        super().__init__(config)
        self.claude_path, self.claude_version = self._find_claude_cli()
        self.model = model

    def _find_claude_cli(self) -> tuple[str, str]:
        """Find the Claude Code CLI executable and return (path, version)."""
        # Check common locations
        locations = [
            "claude",  # In PATH
            "/usr/local/bin/claude",
            os.path.expanduser("~/.local/bin/claude"),
            os.path.expanduser("~/.npm-global/bin/claude"),
        ]

        for loc in locations:
            try:
                result = subprocess.run(
                    [loc, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    version = result.stdout.strip()
                    return loc, version
            except (subprocess.SubprocessError, FileNotFoundError):
                continue

        raise RuntimeError(
            "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
        )

    @staticmethod
    def generate_output_filename() -> str:
        """Generate timestamped output filename."""
        now = datetime.now()
        return f"claude_code_{now.strftime('%Y_%m_%d_%H%M')}.json"

    def _build_prompt(self, query: BenchmarkQuery) -> str:
        """Build the prompt to send to Claude Code."""
        prompt = f"""Write Python code to solve this task using nnsight:

{query.query}

Requirements:
- Use the nnsight library (version 0.5+)
- Include all necessary imports
- The code should be complete and runnable
- Use best practices for the nnsight API

Respond with ONLY the Python code, no explanations."""

        return prompt

    def _extract_code(self, response: str) -> str:
        """Extract Python code from Claude Code response."""
        # Try to find code blocks
        code_block_pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            # Return the longest code block (likely the main solution)
            return max(matches, key=len).strip()

        # If no code blocks, try to extract anything that looks like Python
        lines = response.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            # Heuristic: lines starting with import, from, def, class, with, for, etc.
            if re.match(r'^(import |from |def |class |with |for |if |#|@|\s+)', line):
                in_code = True
            if in_code:
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines).strip()

        # Last resort: return the whole response
        return response.strip()

    def generate_code(self, query: BenchmarkQuery) -> str:
        """
        Generate code using Claude Code CLI.

        Args:
            query: The benchmark query to solve

        Returns:
            Generated code string
        """
        prompt = self._build_prompt(query)

        # Build command
        cmd = [self.claude_path]
        if self.model is not None:
            cmd += ["--model", self.model]
        cmd += [
            "--print",  # Non-interactive mode
            "-p", prompt,
        ]

        # Add skill configuration
        if self.config.with_skills:
            # Skills are enabled by default if installed
            # We can use --allowedTools to restrict if needed
            pass
        else:
            # Disable skills by not loading the plugin
            cmd.extend(["--disallowedTools", "Skill"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                cwd=str(Path(__file__).parent.parent.parent),  # Run from repo root
                env={
                    **os.environ,
                    # Ensure non-interactive
                    "CLAUDE_CODE_NON_INTERACTIVE": "1",
                }
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                return f"# Error running Claude Code: {error_msg}\npass"

            return self._extract_code(result.stdout)

        except subprocess.TimeoutExpired:
            return f"# Timeout after {self.config.timeout_seconds}s\npass"
        except Exception as e:
            return f"# Error: {e}\npass"


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run benchmark with Claude Code CLI")
    parser.add_argument("--with-skills", action="store_true", default=True)
    parser.add_argument("--no-skills", action="store_true")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--model", help="Model to use (optional)")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    parser.add_argument("--skill", help="Filter by skill name")
    parser.add_argument("--output", "-o", help="Output file path (default: auto-generated)")

    args = parser.parse_args()

    from schema import Difficulty, Skill

    config = RunConfig(
        agent="claude-code",
        with_skills=not args.no_skills,
        num_runs=args.num_runs,
        difficulty=Difficulty(args.difficulty) if args.difficulty else None,
        skill=Skill(args.skill) if args.skill else None,
    )

    runner = ClaudeCodeRunner(config, model=args.model)

    benchmark_dir = Path(__file__).parent.parent
    results = runner.run_benchmark(benchmark_dir)

    # Update metadata with model and version info
    results.metadata.model = runner.model
    results.metadata.claude_version = runner.claude_version

    output_path = Path(args.output) if args.output else Path("results") / runner.generate_output_filename()
    runner.save_results(results, output_path)

    print(f"\nOverall score: {results.overall_score:.2%}")
