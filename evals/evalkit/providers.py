"""Agent backends.

Every backend returns an :class:`AgentResponse` carrying the text plus the cost
of producing it — tokens by class, dollars, wall-clock, turns, and (agentic
mode) exactly which resource files the agent opened. That last field is what
tells you whether a documentation set's *routing* works, as distinct from
whether its content is correct.

Backends:

- ``claude-code`` — shells out to `claude -p --output-format stream-json`. Uses
  the local Claude Code login rather than API billing, and is the only backend
  that can load skills the way a real user does (`--plugin-dir`).
- ``anthropic`` — the Messages API. Static conditions only: it has no file
  tools, so an agentic condition would have nothing to navigate.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .conditions import Condition

# Tools the agent may use in agentic mode. Deliberately read-only: the question
# is whether a resource lets an agent *write correct code from documentation*,
# not whether it can iterate against a live interpreter. Allowing Bash would
# measure a different (also interesting) thing and make conditions incomparable.
AGENTIC_TOOLS = ["Read", "Grep", "Glob", "Skill"]
DENIED_TOOLS = ["Bash", "Write", "Edit", "NotebookEdit", "WebFetch", "WebSearch", "Task"]


# Signals that the account, not the task, is what failed. On a subscription the
# realistic way a long sweep dies is hitting a usage window — and if those cells
# were recorded as task failures they would silently corrupt the results.
LIMIT_MARKERS = (
    "usage limit",
    "rate limit",
    "rate_limit",
    "429",
    "quota",
    "not logged in",
    "please run /login",
    "authentication",
    "credit balance",
    "overloaded",
)


def classify_error(text: str) -> str:
    """'limit' for account/quota problems, 'task' for anything else."""
    lowered = (text or "").lower()
    return "limit" if any(marker in lowered for marker in LIMIT_MARKERS) else "task"


@dataclass
class AgentResponse:
    text: str
    ok: bool = True
    error: str = ""
    #: "" when fine, "limit" when the account is the problem, "task" otherwise.
    error_kind: str = ""
    duration_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    num_turns: int = 0
    tool_calls: list[dict] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    skills_used: list[str] = field(default_factory=list)
    raw_meta: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
        )

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "error": self.error,
            "error_kind": self.error_kind,
            "duration_seconds": round(self.duration_seconds, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "num_turns": self.num_turns,
            "n_tool_calls": len(self.tool_calls),
            "files_read": self.files_read,
            "skills_used": self.skills_used,
        }


SYSTEM_STATIC = """You are an expert nnsight user. Answer using the reference material below.

Reply with exactly one fenced ```python block for code questions, or a single letter for multiple choice. No commentary.

===== REFERENCE MATERIAL =====
{material}"""

SYSTEM_STATIC_NONE = """You are an expert nnsight user answering from your own knowledge.

Reply with exactly one fenced ```python block for code questions, or a single letter for multiple choice. No commentary."""

SYSTEM_AGENTIC = """You are answering a question about nnsight.

Reference material is available on disk at:
{paths}

Use your Read/Grep/Glob tools to consult it before answering. You cannot run code — write the answer from the documentation.

Reply with exactly one fenced ```python block for code questions, or a single letter for multiple choice. No commentary."""

SYSTEM_AGENTIC_SKILLS = """You are answering a question about nnsight.

Skills for nnsight are installed and available to you — use them.{extra}

You cannot run code — write the answer from the documentation.

Reply with exactly one fenced ```python block for code questions, or a single letter for multiple choice. No commentary."""


def build_system_prompt(condition: Condition) -> tuple[str, bool]:
    """Return (prompt_text, is_replacement).

    Static conditions replace Claude Code's system prompt entirely (there are no
    tools to describe). Agentic conditions append, so the default tool-using
    prompt is preserved.
    """
    if condition.mode == "static":
        material = condition.static_text()
        if not material:
            return SYSTEM_STATIC_NONE, True
        return SYSTEM_STATIC.format(material=material), True

    has_skills = bool(condition.plugin_dirs())
    directories = condition.directories()
    if has_skills:
        extra = ""
        if directories:
            listed = "\n".join(f"- {path}" for path in directories)
            extra = f"\n\nAdditional reference material is on disk at:\n{listed}"
        return SYSTEM_AGENTIC_SKILLS.format(extra=extra), False
    listed = "\n".join(f"- {path}" for path in directories) or "- (nothing)"
    return SYSTEM_AGENTIC.format(paths=listed), False


class ClaudeCodeProvider:
    """Drive the Claude Code CLI in headless mode."""

    name = "claude-code"

    def __init__(self, model: str = "sonnet", timeout: int = 900, max_turns: int = 30):
        self.model = model
        self.timeout = timeout
        self.max_turns = max_turns
        try:
            probe = subprocess.run(
                ["claude", "--version"], capture_output=True, text=True, timeout=30
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "`claude` CLI not on PATH. Install Claude Code and run `claude /login`."
            ) from exc
        if probe.returncode != 0:
            raise RuntimeError(probe.stderr.strip() or "claude --version failed")

    def ask(self, prompt: str, condition: Condition) -> AgentResponse:
        system, replace = build_system_prompt(condition)

        with tempfile.TemporaryDirectory(prefix="nnsight-eval-agent-") as workdir:
            system_file = Path(workdir) / "system.md"
            system_file.write_text(system)

            command = [
                "claude",
                "-p",
                "--output-format", "stream-json",
                "--verbose",
                "--model", self.model,
                "--permission-mode", "bypassPermissions",
                "--max-turns", str(self.max_turns),
                "--no-session-persistence",
            ]
            command += (
                ["--system-prompt-file", str(system_file)]
                if replace
                else ["--append-system-prompt-file", str(system_file)]
            )

            if condition.mode == "agentic":
                command += ["--allowed-tools", ",".join(AGENTIC_TOOLS)]
                command += ["--disallowed-tools", ",".join(DENIED_TOOLS)]
                for directory in condition.directories():
                    command += ["--add-dir", str(directory)]
                for plugin in condition.plugin_dirs():
                    command += ["--plugin-dir", str(plugin)]
            else:
                command += ["--disallowed-tools", "*"]

            if not condition.plugin_dirs():
                # Guarantee no skill leakage from the operator's own config.
                command.append("--disable-slash-commands")

            started = time.time()
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=workdir,  # keep repo CLAUDE.md out of the context
                )
            except subprocess.TimeoutExpired:
                return AgentResponse(
                    text="",
                    ok=False,
                    error=f"claude CLI timed out after {self.timeout}s",
                    error_kind="task",
                    duration_seconds=time.time() - started,
                )
            elapsed = time.time() - started

        response = _parse_stream_json(completed.stdout)
        response.duration_seconds = elapsed
        if not response.ok and not response.error:
            response.error = (completed.stderr or completed.stdout or "")[-500:]
        if not response.ok:
            response.error_kind = classify_error(response.error)
        elif not response.text.strip():
            # A silent empty answer is usually the CLI refusing to start.
            response.ok = False
            response.error = "empty response from the CLI"
            response.error_kind = classify_error(completed.stderr or "")
        return response


def _parse_stream_json(stdout: str) -> AgentResponse:
    response = AgentResponse(text="", ok=False)
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") != "tool_use":
                    continue
                name = block.get("name", "")
                payload = block.get("input", {}) or {}
                response.tool_calls.append({"name": name, "input": payload})
                if name in ("Read", "Grep", "Glob"):
                    target = payload.get("file_path") or payload.get("path") or payload.get("pattern")
                    if target:
                        response.files_read.append(str(target))
                elif name == "Skill":
                    skill = payload.get("skill")
                    if skill:
                        response.skills_used.append(str(skill))

        elif event.get("type") == "result":
            usage = event.get("usage", {}) or {}
            response.text = event.get("result", "") or ""
            response.ok = not event.get("is_error", False)
            response.input_tokens = usage.get("input_tokens", 0)
            response.output_tokens = usage.get("output_tokens", 0)
            response.cache_creation_tokens = usage.get("cache_creation_input_tokens", 0)
            response.cache_read_tokens = usage.get("cache_read_input_tokens", 0)
            response.cost_usd = event.get("total_cost_usd", 0.0) or 0.0
            response.num_turns = event.get("num_turns", 0)
            response.raw_meta = {
                "stop_reason": event.get("stop_reason"),
                "terminal_reason": event.get("terminal_reason"),
                "permission_denials": event.get("permission_denials", []),
                "duration_api_ms": event.get("duration_api_ms"),
            }
            if not response.ok:
                response.error = str(event.get("result", ""))[:500]
    return response


class AnthropicProvider:
    """Anthropic Messages API. Static conditions only (no file tools)."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", timeout: int = 600, max_tokens: int = 4096):
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError("pip install anthropic to use --provider anthropic") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    def ask(self, prompt: str, condition: Condition) -> AgentResponse:
        import anthropic

        if condition.mode != "static":
            return AgentResponse(
                text="",
                ok=False,
                error="the anthropic backend supports static conditions only; "
                "use --provider claude-code for agentic ones",
            )

        system, _ = build_system_prompt(condition)
        client = anthropic.Anthropic(timeout=self.timeout)

        started = time.time()
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001 — surfaced in the record
            message_text = f"{type(exc).__name__}: {exc}"
            return AgentResponse(
                text="", ok=False, error=message_text,
                error_kind=classify_error(message_text),
                duration_seconds=time.time() - started,
            )

        text = "".join(block.text for block in message.content if block.type == "text")
        usage = message.usage
        return AgentResponse(
            text=text,
            ok=True,
            duration_seconds=time.time() - started,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            num_turns=1,
        )


def get_provider(name: str, model: str, **kwargs):
    if name == "claude-code":
        return ClaudeCodeProvider(model=model, **kwargs)
    if name == "anthropic":
        return AnthropicProvider(model=model, **kwargs)
    raise KeyError(f"unknown provider {name!r} (known: claude-code, anthropic)")
