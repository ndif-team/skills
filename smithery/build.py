#!/usr/bin/env python3
"""Assemble the single-skill Smithery bundle from the nnsight plugin.

    python smithery/build.py            # writes smithery/nnsight/
    python smithery/build.py --check    # build, then fail if the output is stale

Smithery (smithery.ai) hosts one skill per bundle: a SKILL.md plus optional
scripts/, references/ and assets/ directories, with no other skill to cross-load.
The nnsight plugin is 18 skills that lean on each other by name, so this script
folds them into one:

    plugins/nnsight/skills/nnsight/SKILL.md            -> SKILL.md
    plugins/nnsight/skills/nnsight/references/X.md     -> references/X.md
    plugins/nnsight/skills/nnsight/scripts/*.py        -> scripts/*.py
    plugins/nnsight/skills/<S>/SKILL.md                -> references/<S>.md
    plugins/nnsight/skills/<S>/references/R.md         -> references/<S>-R.md

and rewrites what only makes sense across skills: relative links between the
trees, "the `<S>` skill" prose, the "Related skills" lists, and the test-harness
directives (`<!-- test: ... -->`) that mean nothing outside this repo.

The source skills stay the single source of truth — every code block in them is
executed by tests/test_skills.py. Re-run this after editing a skill.
"""

from __future__ import annotations

import argparse
import filecmp
import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_ROOT = REPO_ROOT / "plugins" / "nnsight" / "skills"
OUT = REPO_ROOT / "smithery" / "nnsight"
CORE = "nnsight"

DESCRIPTION = (
    "Read, modify, and analyze the internals of neural networks with nnsight 0.8 — "
    "tracing activations, intervening on modules, batching interventions, gradients, "
    "caching, generation, and remote execution on NDIF for models too large to run "
    "locally. Use for any task that touches model internals rather than just outputs: "
    "activation extraction, logit lens, activation and attribution patching, causal "
    "tracing, ablation, attention-head analysis, circuit discovery, probing, sparse "
    "autoencoders, steering vectors, model editing and LoRA, vLLM and tensor-parallel "
    "runs, quantized and multimodal models, and debugging or porting existing nnsight "
    "code. Load this before writing any nnsight code: idioms from older versions "
    "(.value, nnsight.list(), LanguageModel, proxies) are widespread and silently wrong "
    "on 0.8. Bundles one guide per technique in references/ and scripts that print a "
    "model's module paths without downloading weights."
)

FRONTMATTER = f"""---
name: nnsight
description: {DESCRIPTION}
license: MIT
metadata:
  author: ndif-team
  source: https://github.com/ndif-team/skills
  nnsight: "0.8"
  transformers: ">=5"
---
"""

# One row per bundled skill, in the order the guide table lists them.
GUIDES: list[tuple[str, str, str]] = [
    # (skill, group, one-line "use when")
    ("debugging", "Foundation", "an error, a hang, an empty result, or pre-0.8 code to port"),
    ("remote", "Foundation", "running on NDIF: sessions, request batching, download size, non-blocking jobs"),
    ("interp-experiment-design", "Techniques", "choosing a metric, controls, and sanity checks before running anything"),
    ("logit-lens", "Techniques", "decoding what each layer predicts; tracking where an answer emerges"),
    ("activation-patching", "Techniques", "locating the layers, positions, or heads that carry a behavior; DAS"),
    ("attribution-patching", "Techniques", "scaling patching to whole models with a gradient approximation"),
    ("causal-tracing", "Techniques", "corrupt-and-restore factual localization (ROME-style)"),
    ("ablation", "Techniques", "testing necessity: zero, mean, resample, and noise ablation"),
    ("attention-analysis", "Techniques", "attention patterns, per-head metrics, induction/copy head detection"),
    ("circuit-discovery", "Techniques", "finding and validating the subgraph behind a task (IOI-style)"),
    ("probing", "Techniques", "training classifiers on activations; what is linearly decodable, and whether it is used"),
    ("sae-and-dictionary-learning", "Techniques", "feature-level analysis; attaching, training, and evaluating SAEs"),
    ("model-steering", "Techniques", "steering vectors, function vectors, persistent behavioral edits"),
    ("model-editing-and-lora", "Techniques", "weight edits, ROME-style updates, adapters trained through a frozen model"),
    ("nnterp", "Runtimes and tooling", "one script that runs unchanged across GPT-2, Llama, Qwen, Gemma"),
    ("vllm", "Runtimes and tooling", "throughput, continuous batching, CUDA-graph taps, `model.edit()` sweeps, nnsight-serve"),
    ("tensor-parallel", "Runtimes and tooling", "a model too big for one GPU, sharded with transformers TP under `torchrun`"),
    ("quantization", "Runtimes and tooling", "a model too big for one GPU, held in 4 or 8 bits (`dtype=\"nf4\"`, `\"int8\"`)"),
    ("diffusion-and-multimodal", "Runtimes and tooling", "VLMs, diffusion pipelines, the diffusion lens, non-text tasks"),
]

DIRECTIVE = re.compile(r"^[ \t]*<!--\s*test:[^>]*-->[ \t]*\n", re.MULTILINE)
SKILL_PHRASE = re.compile(r"((?:[Tt]he\s+)?)`([a-z0-9-]+)` skill\b")
LINK = re.compile(r"\]\(([^)\s]+)\)")


def skill_dirs() -> list[Path]:
    return sorted(p for p in SKILLS_ROOT.iterdir() if (p / "SKILL.md").exists())


def split_frontmatter(text: str) -> tuple[str, str]:
    assert text.startswith("---\n"), "no frontmatter"
    end = text.index("\n---\n", 4)
    return text[4:end], text[end + 5 :]


def rewrite_links(text: str, skill: str, own_refs: set[str], in_reference: bool) -> str:
    """Point every relative link at where the file now lives in the bundle."""

    def sub(m: re.Match) -> str:
        target = m.group(1)
        if "://" in target or target.startswith("#"):
            return m.group(0)
        path, _, anchor = target.partition("#")
        anchor = f"#{anchor}" if anchor else ""
        # Links into the core skill's references from another skill.
        core = re.fullmatch(r"(?:\.\./)+nnsight/references/([^/]+\.md)", path)
        if core:
            return f"]({core.group(1)}{anchor})"
        if skill == CORE:
            return m.group(0)
        # A non-core skill's own references, from its SKILL.md or from a sibling.
        own = re.fullmatch(r"(?:references/)?([^/]+\.md)", path)
        if own and own.group(1) in own_refs:
            return f"]({skill}-{own.group(1)}{anchor})"
        raise ValueError(f"{skill}: unhandled link {target!r}")

    return LINK.sub(sub, text)


def rewrite_prose(text: str, in_reference: bool, skill: str, known: set[str]) -> str:
    """Cross-skill mentions become pointers into the bundle."""
    prefix = "" if in_reference else "references/"
    top = "the top-level `SKILL.md`"

    def link(name: str) -> str:
        if name == CORE:
            return "../SKILL.md" if in_reference else "SKILL.md"
        return f"{prefix}{name}.md"

    # "the `nnsight` skill → batching" names one of the core references.
    def core_ref(m: re.Match) -> str:
        ref = re.sub(r"\s+", "-", m.group(2).strip())
        assert (SKILLS_ROOT / CORE / "references" / f"{ref}.md").exists(), f"{skill}: no core reference {ref!r}"
        return f"{m.group(1)}[{ref}.md]({prefix}{ref}.md)"

    text = re.sub(r"((?:[Tt]he\s+)?)`nnsight` skill → ([a-z][a-z\s]+?)(?=[,.;)]|\s(?:has|for|is|section)\b)", core_ref, text)

    # "`scripts/x.py` in the `nnsight` skill": the script now sits in this bundle.
    def script_home(m: re.Match) -> str:
        if "scripts/" in text[max(0, m.start() - 160) : m.start()]:
            return "in this skill"
        return m.group(0)

    text = re.sub(r"(?:in|from) the\s+`nnsight` skill", script_home, text)

    # "`a` and `b` skills"
    def plural(m: re.Match) -> str:
        a, b = m.group(1), m.group(2)
        return f"`{a}` and `{b}` guides (`{link(a)}`, `{link(b)}`)"

    text = re.sub(r"`([a-z0-9-]+)` and `([a-z0-9-]+)` skills\b", plural, text)

    def sub(m: re.Match) -> str:
        article = "The " if m.group(1) and m.group(1).startswith("T") else "the "
        name = m.group(2)
        if name == CORE:
            return article + top[4:]
        return f"{article}`{name}` guide (`{link(name)}`)"

    text = SKILL_PHRASE.sub(sub, text)

    if in_reference:
        text = text.replace("this skill", "this guide").replace("the other skills", "the other guides")
        if skill != CORE:
            text = text.replace("the main SKILL", f"`{skill}.md`")

    # "## Related skills" bullet lists: every backticked skill name becomes a link.
    def related(section: re.Match) -> str:
        body = section.group(2)

        def name(m: re.Match) -> str:
            n = m.group(1)
            return f"[{n}]({link(n)})" if n in known else m.group(0)

        body = re.sub(r"(?m)^(- .*)$", lambda m: re.sub(r"`([a-z0-9-]+)`", name, m.group(1)), body)
        return f"## Related guides{section.group(1)}{body}"

    text = re.sub(r"## Related skills(\n)((?:(?!^## ).*\n?)*)", related, text, flags=re.MULTILINE)
    return text


def guide_table() -> str:
    lines = [
        "## Technique and runtime guides",
        "",
        "Everything the NDIF team ships as separate skills is bundled here as one guide",
        "each. Read the one that matches the task before writing code; each is",
        "self-contained and its examples run against real models in CI.",
        "",
    ]
    group = None
    for skill, g, when in GUIDES:
        if g != group:
            group = g
            lines += [f"**{group}**", "", "| Guide | Use when... |", "|---|---|"]
        lines.append(f"| [{skill}](references/{skill}.md) | {when} |")
        nxt = GUIDES.index((skill, g, when)) + 1
        if nxt == len(GUIDES) or GUIDES[nxt][1] != g:
            lines.append("")
    return "\n".join(lines)


def build_core(text: str, known: set[str]) -> str:
    _, body = split_frontmatter(text)
    body = DIRECTIVE.sub("", body)

    # The intro sentence that points at other skills.
    old = (
        "This skill is the general-purpose reference. Techniques built on it (logit lens,\n"
        "activation patching, steering, …) have their own skills; see the bottom of this\n"
        "file."
    )
    new = (
        "This file is the general-purpose reference. Techniques built on it (logit lens,\n"
        "activation patching, steering, …) each have a guide under `references/`; see\n"
        "[Technique and runtime guides](#technique-and-runtime-guides) at the bottom."
    )
    assert old in body, "core intro changed; update build_core"
    body = body.replace(old, new)

    # The core's own "Related skills" list becomes the full guide table.
    head, sep, _ = body.partition("## Related skills")
    assert sep, "core has no Related skills section; update build_core"
    body = head + guide_table()

    body = rewrite_links(body, CORE, set(), in_reference=False)
    body = rewrite_prose(body, False, CORE, known)
    return FRONTMATTER + body


def build_reference(text: str, skill: str, own_refs: set[str], known: set[str]) -> str:
    """A non-core SKILL.md becomes a reference; frontmatter goes, the H1 stays."""
    _, body = split_frontmatter(text)
    body = DIRECTIVE.sub("", body)
    body = rewrite_links(body, skill, own_refs, in_reference=True)
    body = rewrite_prose(body, True, skill, known)
    return body.lstrip("\n")


def build_sub_reference(text: str, skill: str, own_refs: set[str], known: set[str]) -> str:
    body = DIRECTIVE.sub("", text)
    body = rewrite_links(body, skill, own_refs, in_reference=True)
    body = rewrite_prose(body, True, skill, known)
    return body


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    (out / "references").mkdir(parents=True)
    (out / "scripts").mkdir()

    bundled = {s for s, _, _ in GUIDES}
    found = {d.name for d in skill_dirs()} - {CORE}
    known = found | {CORE}
    assert bundled == found, f"GUIDES out of sync with plugins/nnsight/skills: {bundled ^ found}"

    for d in skill_dirs():
        skill = d.name
        own_refs = {p.name for p in (d / "references").glob("*.md")}
        skill_md = (d / "SKILL.md").read_text()
        if skill == CORE:
            (out / "SKILL.md").write_text(build_core(skill_md, known))
            for ref in sorted((d / "references").glob("*.md")):
                (out / "references" / ref.name).write_text(build_sub_reference(ref.read_text(), skill, own_refs, known))
            for script in sorted((d / "scripts").glob("*.py")):
                shutil.copy(script, out / "scripts" / script.name)
        else:
            (out / "references" / f"{skill}.md").write_text(build_reference(skill_md, skill, own_refs, known))
            for ref in sorted((d / "references").glob("*.md")):
                (out / "references" / f"{skill}-{ref.name}").write_text(
                    build_sub_reference(ref.read_text(), skill, own_refs, known)
                )

    shutil.copy(REPO_ROOT / "LICENSE", out / "LICENSE")


def validate(out: Path) -> list[str]:
    problems: list[str] = []
    skill_md = (out / "SKILL.md").read_text()
    fm, body = split_frontmatter(skill_md)

    name = re.search(r"^name: (.+)$", fm, re.MULTILINE).group(1)
    if not re.fullmatch(r"[a-z0-9-]{1,64}", name):
        problems.append(f"name {name!r} violates ^[a-z0-9-]{{1,64}}$")
    desc = re.search(r"^description: (.+)$", fm, re.MULTILINE).group(1)
    if len(desc) > 1024:
        problems.append(f"description is {len(desc)} chars (max 1024)")
    if (n := body.count("\n")) > 500:
        problems.append(f"SKILL.md body is {n} lines (keep under 500)")

    for md in [out / "SKILL.md", *sorted((out / "references").glob("*.md"))]:
        text = md.read_text()
        rel = md.relative_to(out)
        if "<!-- test:" in text:
            problems.append(f"{rel}: test directive left in")
        for m in SKILL_PHRASE.finditer(text):
            problems.append(f"{rel}: unrewritten skill mention {m.group(0)!r}")
        if re.search(r"`[a-z0-9-]+` skills\b", text):
            problems.append(f"{rel}: unrewritten plural skill mention")
        if md.name != "SKILL.md" and "this skill" in text:
            problems.append(f"{rel}: 'this skill' inside a reference")
        if "## Related skills" in text:
            problems.append(f"{rel}: 'Related skills' heading left in")
        for m in LINK.finditer(text):
            target = m.group(1)
            if "://" in target or target.startswith("#") or target.startswith("mailto:"):
                continue
            path = target.partition("#")[0]
            if not (md.parent / path).exists():
                problems.append(f"{rel}: broken link {target!r}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="fail if the committed bundle differs from a fresh build")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    target = args.out
    if args.check:
        target = REPO_ROOT / "smithery" / ".build-check"
    build(target)
    problems = validate(target)
    for p in problems:
        print(f"error: {p}", file=sys.stderr)

    if args.check:
        cmp = filecmp.dircmp(target, args.out)
        stale = cmp.left_only + cmp.right_only + cmp.diff_files
        shutil.rmtree(target)
        if stale:
            print(f"error: smithery/nnsight is stale; re-run smithery/build.py ({stale})", file=sys.stderr)
            return 1
    if problems:
        return 1

    files = sorted(p.relative_to(target) for p in target.rglob("*") if p.is_file())
    size = sum((target / f).stat().st_size for f in files)
    print(f"built {target.relative_to(REPO_ROOT)}: {len(files)} files, {size / 1024:.0f} KiB")
    print(f"description: {len(DESCRIPTION)} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
