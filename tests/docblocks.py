"""Extract runnable code blocks from skill markdown.

Every fenced ```python block in a SKILL.md or references/*.md is executed by the
test suite unless it carries a directive comment. Directives are HTML comments
placed on the line(s) immediately before the opening fence:

    <!-- test: skip -->                  don't execute (still syntax-checked)
    <!-- test: skip nocompile -->        don't execute, don't even compile
    <!-- test: remote -->                only run when NDIF_HOST is set
    <!-- test: gpu -->                   only run when CUDA is available
    <!-- test: slow -->                  only run with --run-slow
    <!-- test: expect-error ValueError --> must raise that exception
    <!-- test: setup -->                 marks a block other blocks build on
                                         (informational; all blocks share a
                                         namespace in document order anyway)

Multiple flags may appear in one comment, space separated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_ROOT = REPO_ROOT / "plugins" / "nnsight" / "skills"

DIRECTIVE_RE = re.compile(r"<!--\s*test:\s*(?P<body>.*?)\s*-->")
FENCE_RE = re.compile(r"^(?P<indent>\s*)```(?P<lang>[\w+-]*)\s*$")

KNOWN_FLAGS = {"skip", "nocompile", "remote", "gpu", "slow", "setup"}


@dataclass
class Block:
    """One fenced code block with its directives."""

    path: Path
    lang: str
    code: str
    line: int  # 1-indexed line of the opening fence
    flags: set[str] = field(default_factory=set)
    expect_error: str | None = None

    @property
    def location(self) -> str:
        return f"{self.path.relative_to(REPO_ROOT)}:{self.line}"

    @property
    def runnable(self) -> bool:
        return self.lang == "python" and "skip" not in self.flags


def _parse_directives(lines: list[str], fence_index: int) -> tuple[set[str], str | None]:
    """Collect `test:` directives immediately preceding a fence."""
    flags: set[str] = set()
    expect_error: str | None = None

    i = fence_index - 1
    while i >= 0:
        stripped = lines[i].strip()
        if not stripped:
            i -= 1
            continue
        match = DIRECTIVE_RE.search(stripped)
        if match is None:
            break
        tokens = match.group("body").split()
        j = 0
        while j < len(tokens):
            token = tokens[j]
            if token in ("expect-error", "expect-error:"):
                j += 1
                expect_error = tokens[j] if j < len(tokens) else None
            elif token.startswith("expect-error:"):
                expect_error = token.split(":", 1)[1]
            elif token in KNOWN_FLAGS:
                flags.add(token)
            else:
                raise ValueError(f"unknown test directive {token!r} in {lines[i]!r}")
            j += 1
        i -= 1

    return flags, expect_error


def extract_blocks(path: Path) -> list[Block]:
    """Return every fenced block in `path`, in document order."""
    lines = path.read_text().splitlines()
    blocks: list[Block] = []

    i = 0
    while i < len(lines):
        match = FENCE_RE.match(lines[i])
        if match is None:
            i += 1
            continue

        indent, lang = match.group("indent"), match.group("lang")
        closing = f"{indent}```"
        body: list[str] = []
        j = i + 1
        while j < len(lines) and lines[j].rstrip() != closing:
            body.append(lines[j][len(indent) :] if indent else lines[j])
            j += 1

        flags, expect_error = _parse_directives(lines, i)
        blocks.append(
            Block(
                path=path,
                lang=lang or "text",
                code="\n".join(body) + "\n",
                line=i + 1,
                flags=flags,
                expect_error=expect_error,
            )
        )
        i = j + 1

    return blocks


def skill_dirs() -> list[Path]:
    """Every skill directory (one SKILL.md each)."""
    return sorted(p.parent for p in SKILLS_ROOT.glob("*/SKILL.md"))


def markdown_files() -> list[Path]:
    """Every markdown file that may contain runnable examples."""
    files: list[Path] = []
    for skill in skill_dirs():
        files.append(skill / "SKILL.md")
        files.extend(sorted((skill / "references").glob("*.md")))
    return files
