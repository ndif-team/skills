"""Resource conditions — what the agent is given to work with.

A condition is a set of resources plus a delivery mode:

- ``static``  — the resource text is concatenated into the system prompt and the
  agent has no tools. Measures "what if the agent had perfect recall of these
  documents?"
- ``agentic`` — the agent gets Read/Grep/Glob scoped to the resource directories
  (and, for skills, the plugin loaded natively) and must navigate to what it
  needs. Measures the resource **as designed**: routing, progressive disclosure,
  and whether the agent can find the right page at all.

Skills are only meaningful in agentic mode — a skill is a router plus references
loaded on demand plus scripts to run. ``skills`` is still offered in static mode
(all SKILL.md files concatenated, references excluded) as a controlled
comparison against docs-static.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Overridable so the grid can run against checkouts elsewhere.
NNSIGHT_PATH = Path(os.environ.get("NNSIGHT_PATH", "/home/localjadenfk/wd/nnsight"))
WEBSITE_PATH = Path(os.environ.get("NNSIGHT_WEBSITE_PATH", "/home/localjadenfk/wd/nnsight-website"))
SKILLS_PLUGIN = REPO_ROOT / "plugins" / "nnsight"

MAX_STATIC_BYTES = int(os.environ.get("EVAL_MAX_STATIC_BYTES", 600_000))


@dataclass(frozen=True)
class Resource:
    """One body of material the agent may be given."""

    name: str
    #: Directories the agent may Read/Grep in agentic mode.
    directories: tuple[Path, ...] = ()
    #: Files/dirs concatenated into the system prompt in static mode.
    static_selectors: tuple[Path, ...] = ()
    #: Extensions included when expanding a directory for static mode.
    static_suffixes: tuple[str, ...] = (".md",)
    #: Loaded as a Claude Code plugin (native skill resolution).
    plugin_dir: Path | None = None
    description: str = ""


RESOURCES: dict[str, Resource] = {
    "source": Resource(
        name="source",
        directories=(NNSIGHT_PATH / "src" / "nnsight",),
        static_selectors=(),  # far too large for a system prompt; agentic only
        static_suffixes=(".py",),
        description="the nnsight source tree",
    ),
    "docs": Resource(
        name="docs",
        directories=(NNSIGHT_PATH / "docs", NNSIGHT_PATH / "CLAUDE.md"),
        static_selectors=(NNSIGHT_PATH / "CLAUDE.md", NNSIGHT_PATH / "docs"),
        description="the nnsight repo's docs/ tree and its CLAUDE.md router",
    ),
    "tutorials": Resource(
        name="tutorials",
        directories=(WEBSITE_PATH / "docs" / "features", WEBSITE_PATH / "docs" / "tutorials"),
        static_selectors=(WEBSITE_PATH / "docs" / "features", WEBSITE_PATH / "docs" / "tutorials"),
        static_suffixes=(".md", ".ipynb"),
        description="nnsight.net feature walkthroughs and paper tutorials",
    ),
    "skills": Resource(
        name="skills",
        directories=(SKILLS_PLUGIN,),
        static_selectors=(SKILLS_PLUGIN / "skills",),
        plugin_dir=SKILLS_PLUGIN,
        description="the Claude Code skills in this repo",
    ),
}


@dataclass(frozen=True)
class Condition:
    """A named point in the resource grid."""

    name: str
    resources: tuple[str, ...]
    mode: str = "agentic"  # "static" | "agentic"

    @property
    def resource_objects(self) -> list[Resource]:
        return [RESOURCES[name] for name in self.resources]

    def directories(self) -> list[Path]:
        paths: list[Path] = []
        for resource in self.resource_objects:
            if resource.plugin_dir is not None:
                continue  # reached via --plugin-dir, not --add-dir
            paths.extend(path for path in resource.directories if path.exists())
        return paths

    def plugin_dirs(self) -> list[Path]:
        if self.mode != "agentic":
            return []
        return [
            resource.plugin_dir
            for resource in self.resource_objects
            if resource.plugin_dir is not None and resource.plugin_dir.exists()
        ]

    def static_text(self) -> str:
        if self.mode != "static":
            return ""
        chunks: list[str] = []
        budget = MAX_STATIC_BYTES
        for resource in self.resource_objects:
            for selector in resource.static_selectors:
                for path in _expand(selector, resource.static_suffixes):
                    try:
                        body = path.read_text(errors="replace")
                    except OSError:
                        continue
                    if len(body) > budget:
                        body = body[:budget]
                    budget -= len(body)
                    chunks.append(f"===== {_label(path)} =====\n{body}")
                    if budget <= 0:
                        return "\n\n".join(chunks)
        return "\n\n".join(chunks)


def _expand(selector: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if selector.is_file():
        return [selector]
    if not selector.exists():
        return []
    found: list[Path] = []
    for suffix in suffixes:
        found.extend(sorted(selector.rglob(f"*{suffix}")))
    return [p for p in found if "__pycache__" not in p.parts and ".ipynb_checkpoints" not in p.parts]


def _label(path: Path) -> str:
    for root in (NNSIGHT_PATH, WEBSITE_PATH, REPO_ROOT):
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    return str(path)


# The default grid. `none` is the parametric-knowledge baseline; the rest add one
# resource at a time so each delta is attributable, plus the two combinations
# worth knowing about (docs+tutorials = "everything published", everything).
CONDITIONS: dict[str, Condition] = {
    condition.name: condition
    for condition in [
        Condition("none", (), mode="static"),
        Condition("skills", ("skills",), mode="agentic"),
        Condition("docs", ("docs",), mode="agentic"),
        Condition("tutorials", ("tutorials",), mode="agentic"),
        Condition("source", ("source",), mode="agentic"),
        Condition("docs+tutorials", ("docs", "tutorials"), mode="agentic"),
        Condition("skills+docs", ("skills", "docs"), mode="agentic"),
        Condition("everything", ("skills", "docs", "tutorials", "source"), mode="agentic"),
        # Static comparisons: same material, no navigation required.
        Condition("docs-static", ("docs",), mode="static"),
        Condition("skills-static", ("skills",), mode="static"),
    ]
}

DEFAULT_GRID = ["none", "skills", "docs", "tutorials", "source", "docs+tutorials", "everything"]


def get_condition(name: str) -> Condition:
    if name not in CONDITIONS:
        raise KeyError(f"unknown condition {name!r}; known: {', '.join(CONDITIONS)}")
    return CONDITIONS[name]


def describe() -> str:
    lines = [f"{'condition':<18} {'mode':<8} resources"]
    lines.append("-" * 60)
    for condition in CONDITIONS.values():
        resources = ", ".join(condition.resources) or "(none)"
        lines.append(f"{condition.name:<18} {condition.mode:<8} {resources}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
    print()
    for name in ("docs-static", "skills-static"):
        text = get_condition(name).static_text()
        print(f"{name}: {len(text):,} bytes of static context")
