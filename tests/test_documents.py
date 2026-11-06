"""The clab plugin's examples are documents, not code: compile every one.

A ```json block under plugins/clab that holds `model` and `steps` is a complete clab
document, and `clab.Plan(**doc).compile()` checks it without a model, a network or a
cache: unknown fields, ops, components and names are refused by name. A json block
without `model` is a fragment and only has to parse. Python blocks are syntax-checked;
nothing under plugins/clab is executed. `<!-- test: skip -->` above a document keeps
it to parsing (one that loads a bundle or another run's output).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from docblocks import PLUGINS_ROOT, REPO_ROOT, Block, extract_blocks

CLAB_SKILLS = PLUGINS_ROOT / "clab" / "skills"


def _markdown() -> list[Path]:
    files: list[Path] = []
    for skill in sorted(p.parent for p in CLAB_SKILLS.glob("*/SKILL.md")):
        files.append(skill / "SKILL.md")
        files.extend(sorted((skill / "references").glob("*.md")))
    return files


def _is_document(block: Block) -> bool:
    try:
        doc = json.loads(block.code)
    except json.JSONDecodeError:
        return False
    return isinstance(doc, dict) and "model" in doc and "steps" in doc


@pytest.mark.parametrize("path", _markdown(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_json_blocks_parse(path: Path):
    bad = []
    for block in extract_blocks(path):
        if block.lang != "json":
            continue
        try:
            json.loads(block.code)
        except json.JSONDecodeError as exc:
            bad.append(f"{block.location}: {exc}")
    assert not bad, "json blocks that do not parse:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("path", _markdown(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_python_blocks_compile(path: Path):
    for block in extract_blocks(path):
        if block.lang == "python":
            compile(block.code, block.location, "exec")


@pytest.mark.parametrize("path", _markdown(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_documents_compile(path: Path):
    documents = [b for b in extract_blocks(path) if b.lang == "json" and "skip" not in b.flags and _is_document(b)]
    if not documents:
        pytest.skip("no complete documents")
    if importlib.util.find_spec("clab") is None:
        pytest.skip("needs the clab package installed")
    import clab

    failed = []
    for block in documents:
        try:
            clab.Plan(**json.loads(block.code)).compile()
        except Exception as exc:  # noqa: BLE001 - every refusal is reported together
            failed.append(f"{block.location}: {type(exc).__name__}: {exc}")
    assert not failed, "documents clab refuses:\n  " + "\n  ".join(failed)
