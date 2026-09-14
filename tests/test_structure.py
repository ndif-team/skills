"""Packaging checks: frontmatter, symlinks, manifests, internal links."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from docblocks import (
    MARKETPLACE,
    REPO_ROOT,
    SKILLS_ROOT,
    all_markdown_files,
    all_skill_dirs,
    extract_blocks,
    markdown_files,
    plugins,
)

# Codex reads `.agents/skills` (the cross-tool path its docs give) and, on some
# CLI builds, `.codex/skills`. Both trees are symlinks to the same skills.
CODEX_SKILLS = [REPO_ROOT / ".codex" / "skills", REPO_ROOT / ".agents" / "skills"]
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


@pytest.mark.parametrize("skill", all_skill_dirs(), ids=lambda p: p.name)
def test_frontmatter(skill: Path):
    fields = frontmatter(skill / "SKILL.md")
    assert fields.get("name") == skill.name, "frontmatter name must match the directory name"
    assert NAME_RE.match(skill.name), "skill names are lowercase kebab-case"
    description = fields.get("description", "")
    assert description, "a skill needs a description — it is what the agent selects on"
    assert len(description) <= 1024, f"description is {len(description)} chars (max 1024)"


@pytest.mark.parametrize("root", CODEX_SKILLS, ids=lambda p: p.name)
@pytest.mark.parametrize("skill", all_skill_dirs(), ids=lambda p: p.name)
def test_codex_symlink(skill: Path, root: Path):
    link = root / skill.name
    rel = root.relative_to(REPO_ROOT)
    assert link.is_symlink(), f"missing Codex symlink: {rel}/{skill.name}"
    assert link.resolve() == skill.resolve(), f"{link} points at {link.resolve()}"


@pytest.mark.parametrize("root", CODEX_SKILLS, ids=lambda p: p.name)
def test_no_stale_codex_symlinks(root: Path):
    names = {skill.name for skill in all_skill_dirs()}
    stale = [p.name for p in root.iterdir() if p.name not in names]
    assert not stale, f"stale symlinks in {root.relative_to(REPO_ROOT)}: {stale}"


def test_marketplace_sources_exist():
    marketplace = json.loads(MARKETPLACE.read_text())
    assert marketplace["plugins"], "the marketplace lists no plugins"
    for entry in marketplace["plugins"]:
        assert (REPO_ROOT / entry["source"]).is_dir(), (
            f"marketplace source {entry['source']} does not exist"
        )


@pytest.mark.parametrize("name,plugin_dir", plugins(), ids=lambda v: v if isinstance(v, str) else "")
def test_plugin_manifest(name: str, plugin_dir: Path):
    """Each marketplace plugin ships a manifest whose name matches its listing."""
    manifest = plugin_dir / ".claude-plugin" / "plugin.json"
    assert manifest.is_file(), f"{plugin_dir.name}: missing .claude-plugin/plugin.json"
    assert json.loads(manifest.read_text())["name"] == name
    assert (plugin_dir / "skills").is_dir(), f"{plugin_dir.name}: no skills/ directory"
    assert list((plugin_dir / "skills").glob("*/SKILL.md")), f"{plugin_dir.name}: no skills"


@pytest.mark.parametrize("name,plugin_dir", plugins(), ids=lambda v: v if isinstance(v, str) else "")
def test_readme_install_command_matches_the_manifests(name: str, plugin_dir: Path):
    """The install line a new user copies has to name the real marketplace and plugin.

    Nothing else catches this: the command is prose, and a wrong one fails only on
    the reader's machine.
    """
    readme = (REPO_ROOT / "README.md").read_text()
    marketplace = json.loads(MARKETPLACE.read_text())["name"]
    assert f"/plugin install {name}@{marketplace}" in readme


def test_readme_lists_every_skill():
    readme = (REPO_ROOT / "README.md").read_text()
    missing = [skill.name for skill in all_skill_dirs() if f"`{skill.name}`" not in readme]
    assert not missing, f"skills missing from the README table: {missing}"


@pytest.mark.parametrize("path", all_markdown_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
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
    """The nnsight skills target nnsight 0.8. Catch idioms from older versions.

    Scoped to `markdown_files()` — the nnsight plugin — because it is a rule about
    client-library examples, not about every plugin in the repo.
    """
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
