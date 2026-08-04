# NNsight Skills Benchmark

This benchmark suite evaluates the effectiveness of NNsight skills in improving AI coding assistants' performance on neural network interpretability tasks.

## Overview

The benchmark measures how well AI agents (Claude Code, Codex, etc.) can generate correct NNsight code **with** vs **without** access to the skill documentation. This helps quantify the value these skills provide to users.

## Directory Structure

```
skills_benchmark/
├── README.md              # This file
├── schema.py              # Query and result dataclasses
├── queries/               # Benchmark queries organized by difficulty
│   ├── easy/              # Single-concept, direct pattern application
│   ├── medium/            # Multi-concept, requires adaptation
│   └── hard/              # Novel composition, edge cases
├── validators/            # Code validation modules
│   ├── __init__.py
│   ├── structural.py      # AST-based pattern checking
│   └── deprecated.py      # Pre-0.5 pattern detection
├── runners/               # Agent-specific test runners
│   ├── __init__.py
│   ├── base.py            # Base runner interface + MockRunner
│   └── claude_code.py     # Claude Code CLI and API runners
├── results/               # Benchmark results (gitignored)
└── analyze.py             # Results analysis and comparison
```

## Query Taxonomy

### By Skill

| Skill | Description | Key Patterns |
|-------|-------------|--------------|
| `nnsight-basics` | Core tracing, saving, interventions | `model.trace()`, `.save()`, `.output` |
| `logit-lens` | Layer-wise prediction decoding | `ln_f()`, `lm_head()`, softmax |
| `activation-patching` | Causal intervention via swapping | Cross-run patching, logit difference |
| `attribution-patching` | Gradient-based approximation | `.backward()` context, reverse order |
| `causal-tracing` | Mediation analysis | Direct/indirect effects, position-specific |
| `model-steering` | Steering vectors, persistent edits | `.edit()`, `tracer.all()`, contrastive |

### By Difficulty

| Level | Characteristics | Examples |
|-------|-----------------|----------|
| **Easy** | Single concept, direct application of documented pattern | Load model, extract hidden states, basic intervention |
| **Medium** | Combines 2-3 concepts, requires adaptation to specific use case | Layer-wise analysis with visualization, position-specific patching |
| **Hard** | Novel composition, edge cases, multi-step reasoning | Full causal tracing pipeline, attribution + validation |

## Query Format

Each query is defined in YAML format:

```yaml
id: "activation-patching-002"
skill: "activation-patching"
difficulty: "medium"
title: "Layer-wise Activation Patching"
query: |
  Write NNsight code to identify which layers are most important for
  GPT-2 predicting "John" vs "Mary" in the IOI task using activation patching.

# Concepts the solution should demonstrate
expected_concepts:
  - "separate trace contexts for clean/corrupted runs"
  - "layer output patching"
  - "logit difference metric"
  - "result normalization"

# Validation rules
validation:
  # Patterns that MUST appear in the solution
  must_include:
    - "model.trace"
    - ".save()"
    - "logit"

  # Patterns that must NOT appear (deprecated or incorrect)
  must_not_include:
    - pattern: "tracer.invoke.*tracer.invoke"
      reason: "Cross-prompt without barrier (pre-0.5 pattern)"
    - pattern: "\\.grad\\.save\\(\\).*\\.backward\\(\\)"
      reason: "Accessing grad before backward context"

  # Expected output characteristics (optional)
  expected_output:
    type: "tensor"
    shape: [12]  # 12 layers for GPT-2

# Reference solution (for validation development)
reference_solution: |
  # ... correct implementation ...

# Tags for filtering
tags:
  - "causal-intervention"
  - "ioi-task"
  - "gpt2"
```

## Evaluation Metrics

### 1. Execution Success

- **Executes**: Code runs without syntax/runtime errors
- **Produces Output**: Generates the expected output type

### 2. API Correctness

- **Uses Correct API**: Percentage of required NNsight patterns used correctly
- **Avoids Deprecated**: No pre-0.5 patterns detected

### 3. Functional Correctness

- **Output Correct**: Results match expected values (within tolerance)
- **Output Shape Match**: Tensor dimensions are correct

### 4. Efficiency

- **Forward Passes**: Number of model forward passes (fewer is better)

### Composite Score

```
score = (0.25 * executes) +
        (0.20 * uses_correct_api) +
        (0.15 * avoids_deprecated) +
        (0.40 * output_correct)
```

## Available Runners

### MockRunner

Uses reference solutions for testing the benchmark infrastructure:

```bash
python skills_benchmark/runners/base.py --num-runs 1 --output results/mock.json
```

### ClaudeCodeRunner (CLI)

Invokes Claude Code CLI to generate code. Requires Claude Code installed:

```bash
npm install -g @anthropic-ai/claude-code

# Run with skills
python skills_benchmark/runners/claude_code.py --with-skills

# Run without skills
python skills_benchmark/runners/claude_code.py --no-skills
```

### ClaudeAPIRunner (API)

Calls Claude API directly with skill content injected as system prompt. Faster and more controllable:

```bash
export ANTHROPIC_API_KEY=sk-...
pip install anthropic

# Run with skills (injected as context)
python skills_benchmark/runners/claude.py --with-skills

# Run without skills (no skill context)
python skills_benchmark/runners/claude.py --no-skills

# Use specific model
python skills_benchmark/runners/claude.py --model claude-sonnet-4-20250514
```

**Key difference**: The API runner includes skill content directly in the system prompt when `--with-skills` is set, simulating what happens when skills are available to Claude Code.

## Running Benchmarks

### Prerequisites

```bash
# Using pip
pip install pyyaml torch transformers nnsight anthropic

# Using uv (recommended - faster)
uv pip install pyyaml torch transformers nnsight anthropic

# Or with uv sync if using pyproject.toml
uv sync --extra benchmark
```

### Run Full Benchmark

```bash
# Run with skills available
python skills_benchmark/runners/claude_code.py --mode api --with-skills

# Run without skills (baseline)
python skills_benchmark/runners/claude_code.py --mode api --no-skills

# Run specific difficulty
python skills_benchmark/runners/claude_code.py --mode api --difficulty medium

# Run specific skill
python skills_benchmark/runners/claude_code.py --mode api --skill activation-patching
```

### Analyze Results

```bash
# Compare with vs without skills
python skills_benchmark/analyze.py compare results/with_skills.json results/without_skills.json

# Generate report for single run
python skills_benchmark/analyze.py report results/with_skills.json
```

## A/B Testing Protocol

### Experimental Design

1. **Control**: Agent without skill access (baseline capability)
2. **Treatment**: Agent with skills installed and available

### Procedure

1. Randomly sample N queries per difficulty level
2. Run each query M times (M=3 recommended) to account for variance
3. Collect all evaluation metrics
4. Compare treatment vs control using paired statistical tests

### Hypotheses

| ID  | Hypothesis                                | Metric              |
| --- | ----------------------------------------- | ------------------- |
| H1  | Skills improve overall accuracy           | `output_correct`    |
| H2  | Skills reduce deprecated pattern usage    | `avoids_deprecated` |
| H3  | Skills have larger impact on hard queries | `score` by difficulty |
| H4  | Skills reduce execution errors            | `executes`          |
| H5  | Skills improve computational efficiency   | `forward_passes`    |

## Adding New Queries

1. Create a new YAML file in the appropriate difficulty folder:

   ```bash
   touch skills_benchmark/queries/medium/layer-wise-activation-patching.yaml
   ```

2. Follow the query schema (see `schema.py` for dataclass definitions)

3. Include a reference solution for validation development

4. Test the validators:

   ```bash
   python skills_benchmark/validators/structural.py --query skills_benchmark/queries/medium/layer-wise-activation-patching.yaml
   ```

## Validation Development

### Structural Validator

Checks for required/forbidden patterns using AST analysis and regex:

```python
from skills_benchmark.validators import StructuralValidator

validator = StructuralValidator()
result = validator.validate(code, query.validation)
print(result.must_include_results)  # Which patterns were found
print(result.must_not_include_results)  # Which forbidden patterns detected
```

### Deprecated Pattern Checker

Specifically checks for pre-NNsight-0.5 patterns:

```python
from skills_benchmark.validators import DeprecatedPatternChecker

checker = DeprecatedPatternChecker()
findings = checker.check(code)
for f in findings:
    print(f"{f.pattern_name}: {f.reason} (line {f.line_number})")
```

## Results Format

Results are stored as JSON:

```json
{
  "metadata": {
    "agent": "claude-api",
    "with_skills": true,
    "timestamp": "2024-01-15T10:30:00Z",
    "nnsight_version": "0.5.0"
  },
  "results": [
    {
      "query_id": "activation-patching-002",
      "runs": [
        {
          "generated_code": "...",
          "execution_time_ms": 1234,
          "metrics": {
            "executes": true,
            "uses_correct_api": 0.95,
            "avoids_deprecated": true,
            "output_correct": 0.85,
            "forward_passes": 14
          },
          "score": 0.89
        }
      ],
      "aggregate": {
        "mean_score": 0.87,
        "std_score": 0.03
      }
    }
  ],
  "summary": {
    "overall_score": 0.82,
    "num_queries": 17
  }
}
```

## Contributing

When adding new queries or validators:

1. Ensure queries test realistic user workflows
2. Include clear expected concepts
3. Provide reference solutions
4. Test validators against reference solutions
5. Document any edge cases
