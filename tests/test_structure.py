"""Packaging checks: frontmatter, symlinks, manifests, internal links."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from docblocks import REPO_ROOT, SKILLS_ROOT, extract_blocks, markdown_files, skill_dirs

CODEX_SKILLS = REPO_ROOT / ".codex" / "skills"
MARKETPLACE = REPO_ROOT / ".claude-plugin" / "marketplace.json"
PLUGIN = REPO_ROOT / "plugins" / "nnsight" / ".claude-plugin" / "plugin.json"
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
LINK_RE = re.compile(r"\[[^\]]*\]\((?P<target>[^)#\s]+)(?:#[^)]*)?\)")


def frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text()
    if not text.startswith("---\n"):
        raise AssertionError(f"{path}: missing YAML frontmatter")
    end = text.index("\n---", 4)
    fields: dict[str, str] = {}
    key = None
    for line in text[4:end].splitlines():
        if not line.strip():
            continue
        if line[0].isspace() and key:  # folded continuation
            fields[key] += " " + line.strip()
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        fields[key] = value.strip()
    return fields


@pytest.mark.parametrize("skill", skill_dirs(), ids=lambda p: p.name)
def test_frontmatter(skill: Path):
    fields = frontmatter(skill / "SKILL.md")
    assert fields.get("name") == skill.name, "frontmatter name must match the directory name"
    assert NAME_RE.match(skill.name), "skill names are lowercase kebab-case"
    description = fields.get("description", "")
    assert description, "a skill needs a description — it is what the agent selects on"
    assert len(description) <= 1024, f"description is {len(description)} chars (max 1024)"


@pytest.mark.parametrize("skill", skill_dirs(), ids=lambda p: p.name)
def test_codex_symlink(skill: Path):
    link = CODEX_SKILLS / skill.name
    assert link.is_symlink(), f"missing Codex symlink: .codex/skills/{skill.name}"
    assert link.resolve() == skill.resolve(), f"{link} points at {link.resolve()}"


def test_no_stale_codex_symlinks():
    names = {skill.name for skill in skill_dirs()}
    stale = [p.name for p in CODEX_SKILLS.iterdir() if p.name not in names]
    assert not stale, f"stale Codex symlinks: {stale}"


def test_manifests():
    marketplace = json.loads(MARKETPLACE.read_text())
    plugin = json.loads(PLUGIN.read_text())
    sources = [p["source"] for p in marketplace["plugins"]]
    assert "./plugins/nnsight" in sources
    for source in sources:
        assert (REPO_ROOT / source).is_dir(), f"marketplace source {source} does not exist"
    assert plugin["name"] == "nnsight"


def test_readme_lists_every_skill():
    readme = (REPO_ROOT / "README.md").read_text()
    missing = [skill.name for skill in skill_dirs() if f"`{skill.name}`" not in readme]
    assert not missing, f"skills missing from the README table: {missing}"


@pytest.mark.parametrize("path", markdown_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_relative_links_resolve(path: Path):
    broken = []
    for match in LINK_RE.finditer(path.read_text()):
        target = match.group("target")
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if not (path.parent / target).exists():
            broken.append(target)
    assert not broken, f"{path.relative_to(REPO_ROOT)}: broken relative links {broken}"


@pytest.mark.parametrize(
    "script",
    sorted(SKILLS_ROOT.glob("*/scripts/*.py")),
    ids=lambda p: str(p.relative_to(SKILLS_ROOT)),
)
def test_scripts_compile(script: Path):
    compile(script.read_text(), str(script), "exec")


@pytest.mark.parametrize("path", markdown_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_pre_08_api(path: Path):
    """These skills target nnsight 0.8. Catch idioms from older versions."""
    banned = {
        r"\.value\b": "0.8 saves return the value itself — no .value",
        r"\bnnsight\.(list|dict|int|float|bool|apply|cond|log|local)\(": "removed in 0.8",
        r"\btracer\.next\(": "removed in 0.8 — use tracer.iter",
        r"with\s+tracer\.(all\(\)|iter\[[^\]]*\])\s*:": "deprecated form — use `for _ in tracer.iter[...]`",
        r"\bmodel\.generator\.output\b": "deprecated — use tracer.result",
        r"\bLanguageModel\(": "deprecated — use TransformersModel",
        r"\bproxy\b": "0.8 has no proxies; values inside a trace are real",
    }
    # Only executable examples are checked. Prose (and ```python-legacy blocks)
    # may quote old idioms — the debugging skill has to name them to fix them.
    hits = []
    for block in extract_blocks(path):
        if block.lang != "python":
            continue
        # Comments may name old idioms ("not a proxy"); only real code counts.
        code = re.sub(r"#[^\n]*", "", block.code)
        for pattern, why in banned.items():
            for match in re.finditer(pattern, code):
                line = block.line + code[: match.start()].count("\n") + 1
                hits.append(f"line {line}: {match.group(0)!r} — {why}")
    assert not hits, f"{path.relative_to(REPO_ROOT)}: pre-0.8 API in a 0.8 skill:\n  " + "\n  ".join(hits)
