Hey 👋 @gsarti

I ran your skills through `tessl skill review` at work and found some targeted improvements. Here's the full before/after:

![Skill Review Score Card](score_card.png)

| Skill | Before | After | Change |
|-------|--------|-------|--------|
| attribution-patching | 76% | 97% | +21% |
| nnsight-basics | 76% | 92% | +16% |
| causal-tracing | 76% | 92% | +16% |
| model-steering | 70% | 83% | +13% |
| activation-patching | 82% | 92% | +10% |

I kept this PR focused on the 5 skills with the biggest improvements to keep the diff reviewable. logit-lens was already at 82% and didn't need changes. Happy to follow up with it in a separate PR if you'd like.

<details>
<summary>Changes made</summary>

**All 5 skills — description improvements:**
- Expanded frontmatter descriptions with specific concrete actions (e.g., "compute attribution scores", "patch activations between clean and corrupted runs") instead of abstract summaries
- Added domain-specific trigger keywords (`mechanistic interpretability`, `AtP`, `ActAdd`, `representation engineering`, `ablation studies`) for better agent skill selection
- Ensured all descriptions use quoted string format

**All 5 skills — content trimming for conciseness:**
- Removed introductory paragraphs and "Core Concept" sections that explain things an AI agent already understands
- Removed "Best Practices", "Interpretation", and "When to Use" bullet lists that restate common knowledge

**attribution-patching (+21%):**
- Replaced incomplete validation section (referenced undefined `actual_patching_results`) with a complete, executable validation workflow that runs activation patching on top-attributed layers and computes correlation

**model-steering (+13%):**
- Added a concrete "Verifying a Steering Vector" section with executable code to compare steered vs unsteered outputs across test prompts

**nnsight-basics (+16%):**
- Removed installation section (`pip install` commands) and "Common Pitfalls" list
- Tightened "Saving Values" section prose

**causal-tracing (+16%):**
- Removed "Core Concepts" section (three types of causal effects, interchange intervention explanation)
- Removed "Interpretation Guidelines" bullet list

**activation-patching (+10%):**
- Removed introductory paragraph and "Core Concept" numbered list
- Removed "Interpretation" bullet list

</details>

Honest disclosure — I work at @tesslio where we build tooling around skills like these. Not a pitch - just saw room for improvement and wanted to contribute.

Want to self-improve your skills? Just point your agent (Claude Code, Codex, etc.) at [this Tessl guide](https://docs.tessl.io/evaluate/optimize-a-skill-using-best-practices) and ask it to optimize your skill. Ping me - [@yogesh-tessl](https://github.com/yogesh-tessl) - if you hit any snags.

Thanks in advance 🙏
