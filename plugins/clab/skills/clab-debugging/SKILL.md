---
name: clab-debugging
description: "Fix a clab document that was refused or gave wrong numbers. Use for a `clab: <where>: <what>` line or a PlanError, and to tell which stage refused it. Common ones: unknown fields, ops or components, a name no earlier step made, an operand of another layout or width, text not in a row, a multi-token answer, eager attention, a bundle or load that does not match. Use when nothing was refused but the numbers look wrong: a null that differs from the clean run, a scope on the wrong occurrence, rows missing from a mean. So too sweeps zipped into a diagonal and a patch equal to the clean run. Use when two runs differ (batch size, padding, threads, device, bfloat16), to cross-check engines, or after running out of memory. Writing a document is the `clab` skill; recipes are clab-patching, clab-ablation, clab-lenses-and-attention, clab-steering-and-generation and clab-trained-featurizers."
---

# Debugging clab documents

Verified on clab `830182f` with nnterp `e5d0f8e`, nnsight 0.8.2 and transformers 5.20.2; the documents run
on the CPU on `openai-community/gpt2` (float32, 12 layers of 12 heads, no BOS token), in seconds each.

A run either refuses, printing one `clab:` line and writing nothing, or finishes. This skill reads a refusal,
finds what is wrong when nothing was refused, and makes two runs agree; writing the document is the `clab` skill.

## What breaks first

- **A refused run leaves the previous output in place.** A refusal writes nothing, so `out/<name>` holds the
  last finished run: after a fix, check that `clab show`'s first line has a new `started` time.
- **A patch that equals the clean run to the bit is a mistake, not a result.** A write whose `ops` lost its edit
  only reads, and a freeze listed after its patch puts the clean value back. Read the entry's `ops` and their
  order at the site ([references/symptoms.md](references/symptoms.md)).
- **Reading an effect before its null.** Write back what was read, at the same site and `pos`: the null must
  equal the clean run to the bit, or the read and the write are not one selection.
- **Comparing two runs made at different settings.** Another batch size, padding, readout, thread count or
  device can move the last bits. Compare only runs whose run records agree ([below](#reproducing-a-number)).
- **Trusting a scoped read without looking.** `occurrence` defaults to the first match in the whole row.
  Read the entry's positions in `clab show` on a row of each kind.
- **Reading a mean over rows a read never reached.** Those rows are NaN, every reduction skips them, and the
  number does not say so. Count the rows before reading the mean.
- **Reading small effects from a bfloat16 run.** A bfloat16 null that equals the clean run certifies the
  wiring, not effects below bfloat16's rounding. Recompute the points you report in float32.

## How a refusal reads

```text
clab: <where>: <what>
```

`clab run` prints this line, exits with status 1 and writes no output. `<where>` names the stage, and the
first clause of `<what>` is the key to [references/refusals.md](references/refusals.md).

| `<where>` | refused | as GPT-2 printed it |
|---|---|---|
| a document path: `steps.…`, `sites.…`, `data.…`, `saves[i]`, `model` | before the model loads | `steps.ld.of: 'patched.logit' names no value produced before this step; known: counterfactual.v_cf, patched.logits, patched.patch` |
| none: a flag or an engine option | before the model loads | `batch_size 0: a batch holds one row or more (--batch-size N, N at least 1)` |
| a step's name without `steps.`: `<step>.<entry>` | after the model loads, before any forward | `counterfactual.v_cf: layer 12; the model has 12 layers` |
| `<step>: generate rejected …` | at run: earlier steps ran, nothing is written | `answer: generate rejected 'max_new_tokns': …` |

- **Check before you run.** `clab compile documents/x.json` needs no model: it prints what would run and exits
  0, or prints `clab: <where>: <what>`, the line `clab run` would print, and exits 1.
- **Compiling misses what only the load knows**: a `revision` the cache lacks compiles, and offline the load
  refuses it. A `model` key naming a run option (`device`, `torch_dtype`, `tp_size`) is refused at compile.
- **A refusal names its row and sweep point** (`base.r: row 1: …`, `counterfactual.v_cf at layers=12: …`),
  and lists what is valid where it can (`known: …`, `valid: …`): copy the spelling from it.

## Four refusals one line cannot explain

### An operand does not fit

An operand (`swap`'s and `add`'s `with`, `scale`'s `factor`, `kl`'s `against`) meets the value element for
element. Its axes are among the value's, in the value's order, and it broadcasts over those it lacks. `pos`
never broadcasts, a `select` list pairs by position, and it holds a slice for every layer the write acts at.
A value read at the head and swapped into the residual stream, refused before the model loads:

```text
clab: steps.patched.interventions.patch.ops[0].with: 'counterfactual.v_cf' is laid out ['rows', 'pos', 'vocab_size']; 'swap' acts on a value laid out ['rows', 'pos', 'hidden_size'], and an operand's axes are among the value's, in its order (those it leaves out it broadcasts over)
```

A scope pairs with another forward only where every row's scope holds as many tokens, counted after the model
loads. An unscoped `pos` on inputs of different lengths passes where the tokens after it (before it, for an
index from the start) are alike on both: the shared end and the one token next to it. Past that it is refused,
naming both tokens; rows of equal length are never checked ([refusals](references/refusals.md#operands)).

### A position is not in a row

In the prefill the prompt alone decides where a `pos` lands. So every row is resolved after the tokenizer
loads and before any forward, and a row that lacks the text, the occurrence or a static index is refused by
number. GPT-2 adds no BOS, so position 0 is the prompt's first word and nothing comes before it:

```text
clab: base.r: row 0: text 'When' begins the row, at position 0, which no position comes before
```

A refused row cannot be skipped: change the data (`take`, `skip`, or a dataset of the rows that fit). An index
with an open end (`{"index": "8:"}`) is not refused for a read; its short rows are missing instead (below).

### An answer does not line up

A metric tokenizes each row's answer as written, without special tokens, and scores answer token i at the
read's position i. Checked against every row after the model loads:

```text
clab: ld: pairs.cf_answer row 0: ' Colosseum' is 4 tokens; the read has 1 position on that row; token i of an answer scores the read's position i, so the read carries one per token (a scope with "before": true reads the positions that predict the answer's tokens)
```

A missing leading space is refused only where the bare word splits. On GPT-2 `Rome` is 2 tokens and is
refused; `Paris` is one token, another id, and scores 1.0e-06 ([symptoms](references/symptoms.md#answers-and-metrics)).
Sets and teacher forcing have their own messages ([refusals](references/refusals.md#answers)); writing answer
columns is the `clab` skill's [data reference](../clab/references/data.md).

### Rows a read did not reach

A read at a position a row lacks is not refused: an open-ended index past a short row, or a decode call a row
never made. That row is `alignment_missing`: NaN, skipped by every reduction, `—` in `clab show`, flagged in
`clab table`, so a mean over rows covers the rows reached and does not say so. A write whose operand is missing
on a row it lands on is refused at run (`operand 'g.r' holds no value on row 2 (alignment_missing): …`). Count
the rows reached as the `clab` skill's [positions reference](../clab/references/positions.md#rows-a-scope-never-reached) says.

## Wrong numbers, nothing refused

Open `clab show OUT` first. Each part answers one question:

- **The first line and `served:`**: is this the run you mean (document, engine, batch size, padding, readout,
  `started`), and what dtype did the model run in?
- **The entry table**: where each read and write landed on each row; `—` is a row it did not reach.
- **The result table**: each value's `axes` and `shape`; `2 points` where you meant a 2 by 2 grid is two sweeps
  zipped under one label.
- **The step table**: `tokens` against `real` is the padding; a forward whose own write is swept, listed once
  with `all N points` under `serves`, ran one point for all ([symptoms](references/symptoms.md#sweeps)).

### The nulls

Each null changes nothing, so its value is known exactly; the four equal-to-clean nulls run in one GPT-2
document ([symptoms](references/symptoms.md#nulls)) and give the clean run's logit difference to the bit.

| null | written as | must equal | a miss means |
|---|---|---|---|
| self-swap | `swap` with the base's own read, same site and `pos` | the clean run, to the bit | the read and the write are not one selection |
| dose 0 | `add` with `"scale": 0.0`, the first point of a dose sweep | the clean run, to the bit | something else writes at the site |
| scale 1 | `scale` with `"factor": 1.0` | the clean run, to the bit | something else writes at the site |
| identity permutation | a `permute` step with `"seed": null`, swapped in | the clean run, to the bit | the permuted read is not the one the write replaces |
| gradient null | the metric's ops with `{"op": "mul", "with": 0}` appended, and a `grad` of that entry | exactly 0 | the gradient is not of the zeroed entry alone |
| `epochs: 0` | the same fit, training nothing | the draw, which the trained featurizer must beat | the fit learned nothing (`clab-trained-featurizers`) |

`documents/null_patch.json`, the self-swap null beside a patch of layer 6's residual stream at the last token:

```json
{
  "description": "Patch layer 6's residual stream at the last token from a counterfactual city prompt into the base prompt, beside the null control: the base's own value written back, which must equal the clean run.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"input": "The Eiffel Tower is in the city of", "cf_input": "The Colosseum is in the city of", "base_answer": " Paris", "cf_answer": " Rome"},
    {"input": "The Colosseum is in the city of", "cf_input": "The Eiffel Tower is in the city of", "base_answer": " Rome", "cf_answer": " Paris"},
    {"input": "Big Ben is in the city of", "cf_input": "The Brandenburg Gate is in the city of", "base_answer": " London", "cf_answer": " Berlin"},
    {"input": "The Brandenburg Gate is in the city of", "cf_input": "Big Ben is in the city of", "base_answer": " Berlin", "cf_answer": " London"}
  ]}},
  "sites": {"target": {"component": "layer_output", "layers": 6}, "head": {"component": "lm_head_output"}},
  "interventions": {"scored": {"logits": {"site": "head", "pos": -1}}},
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": ["scored", {"v_self": {"site": "target", "pos": -1}}]},
    "counterfactual": {"kind": "forward", "input": "pairs.cf_input", "interventions": {"v_cf": {"site": "target", "pos": -1}}},
    "null": {"kind": "forward", "input": "pairs.input",
             "interventions": ["scored", {"patch": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "clean.v_self"}]}}]},
    "patched": {"kind": "forward", "input": "pairs.input",
                "interventions": ["scored", {"patch": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]}}]},
    "clean_ld": {"kind": "op", "op": "logit_diff", "of": "clean.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "null_ld": {"kind": "op", "op": "logit_diff", "of": "null.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "patched_ld": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"}
  },
  "saves": ["clean_ld", "null_ld", "patched_ld"]
}
```

```text
$ clab run documents/null_patch.json
$ clab table out/null_patch clean_ld null_ld patched_ld
row  pos  pos_token  input                                   cf_answer  base_answer  clean_ld             null_ld              patched_ld
0    9    Ġof        The Eiffel Tower is in the city of       Rome       Paris       -2.4247055053710938  -2.4247055053710938  -2.0760040283203125
2    6    Ġof        Big Ben is in the city of                Berlin     London      -1.9296493530273438  -1.9296493530273438  -0.9876708984375
```

`clab table` prints each float in full, so equal printed values are equal bits. The effect is `patched_ld`
minus `null_ld`. With the patch's `ops` deleted, `patched_ld` prints `-2.4247055053710938` on row 0 too.

The gradient null sits in the gradient's own forward, beside the metric. A fragment, that forward's
`interventions` over the rows and sites above ([complete document](references/symptoms.md#gradients)):

```json
{
  "g": {"site": "target", "pos": -1, "ops": [{"op": "grad", "of": "metric"}, {"op": "norm"}]},
  "g0": {"site": "target", "pos": -1, "ops": [{"op": "grad", "of": "zero"}, {"op": "norm"}]},
  "metric": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]},
  "zero": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}, {"op": "mul", "with": 0}]}
}
```

```text
$ clab table out/grad_check taken.metric taken.g taken.g0
row  pos  pos_token  input                                   cf_answer  base_answer  taken.metric         taken.g             taken.g0
0    9    Ġof        The Eiffel Tower is in the city of       Rome       Paris       -2.4247055053710938  0.4240272641181946  0.0
3    8    Ġof        The Brandenburg Gate is in the city of   London     Berlin      -5.286407470703125   0.4044497311115265  0.0
```

**Write the null with `mul`, never `scale`.** `{"op": "scale", "factor": 0}` alone at the head is a write,
and listed before `metric` it zeroes the logits: `taken.metric`, `taken.g` and `taken.g0` all print `0.0`.
**The null proves the wiring, not the metric.** Exactly 0 proves the gradient is of the entry you zeroed, wired
exactly; it cannot tell you that entry is the metric you meant. A central difference at one point does ([check
the gradient](../clab-patching/references/attribution.md#check-the-gradient)), and real patching of the top points
does for the ranking.

### Where a scope landed

`When Mary and John went to the store, John gave a drink to` says its `subject` column, ` John`, twice. Two
entries of one forward at layer 6 ([complete document](references/symptoms.md#positions-and-selections)):

```json
{
  "first": {"site": "resid", "pos": {"scope": {"column": "ioi.subject"}}, "ops": [{"op": "norm"}]},
  "repeat": {"site": "resid", "pos": {"scope": {"column": "ioi.subject", "occurrence": -1}}, "ops": [{"op": "norm"}]}
}
```

```text
$ clab show out/scope_occurrence
  entry        at  positions
  base.first   —   [3] on all 2 rows
  base.repeat  —   [9] on all 2 rows
```

Position 3 is the first mention and 9 the repeat, `ĠJohn` in `clab table`'s `pos_token`. On a dataset that mixes
kinds, `occurrence: -1` is the query's copy only on rows whose query repeats that name: check a row of each kind.

Every silent mistake, the check that catches it and its fix: [references/symptoms.md](references/symptoms.md).

## Reproducing a number

**One document run at one batch size, padding, readout, device and thread count writes byte-identical value
files.** Change any of them and a row can move in its last bits:

| setting | why it moves bits |
|---|---|
| `--batch-size` | another batch is a matmul of another shape, which sums in another order |
| `--padding` (`dataset`, `batch`, `sorted`) | a row padded to another width is another matmul; under `dataset` the input's longest row sets every row's width |
| `--readout at-reads` | the head runs at the read positions alone, a matmul of another shape |
| an op in an entry, not an op step | an entry's op runs per batch on that batch's value and a step's on the whole value, so a matmul (`readout`, `encode`) or a long reduction can round otherwise, on the CPU too |
| thread count (`OMP_NUM_THREADS`) | the CPU splits a matmul's sum by thread |
| device, machine | other kernels; a bfloat16 `model.dtype` rounds more ([below](#the-float32-rule)) |

The run record holds each setting; `clab show` prints all but the threads, `run.threads` in `manifest.json`,
which follow the `OMP_NUM_THREADS` the `clab` process inherits (a wrapper that sets its own wins).

**To see a difference, change a row's padded width.** `documents/null_patch.json` ([the nulls](#the-nulls)) has
rows of 10, 10, 7 and 9 tokens. Under `--padding batch`, batch size 1 pads no row and 4 pads rows 2 and 3 to 10:

```text
$ clab run documents/null_patch.json --padding batch --batch-size 1 --out out/bs1
$ clab run documents/null_patch.json --padding batch --batch-size 4 --out out/bs4
$ for f in out/bs1/*; do [ "${f##*/}" = manifest.json ] || cmp "$f" "out/bs4/${f##*/}"; done
out/bs1/clean_ld.safetensors out/bs4/clean_ld.safetensors differ: byte 85, line 1
out/bs1/null_ld.safetensors out/bs4/null_ld.safetensors differ: byte 85, line 1
out/bs1/patched_ld.safetensors out/bs4/patched_ld.safetensors differ: byte 90, line 1
```

Row 3's `clean_ld` is `-5.286415100097656` at batch size 1 and `-5.286407470703125` at 4, and its `null_ld` moves
with it; a second run at 4 is byte-identical to the first. On rows of one length, or under the default padding, no
row's width changes with the batch size. Whether the batch size alone then moves bits depends on the thread count:
compare two runs, never infer it.

**To the bit** holds for a rerun at one setting on one machine, for a null against its clean run, and for nnterp
against hooks or local ndif on the CPU in float32. Across settings, machines, `tp` or vLLM expect rounding, and a
fit's trained values can move further, since Adam scales a rounding-sized gradient up to its step size. The
settings themselves are the `clab` skill's [running reference](../clab/references/running-and-engines.md).

### The float32 rule

Metrics, reductions and arithmetic compute in float32; a read keeps the model's dtype; an edit computes in
float32 and rounds once to the value's dtype. No op undoes the model's own rounding. GPT-2's logits are near
-100, where bfloat16's step is 0.5, so the self-swap document in `"dtype": "bfloat16"` prints:

```text
row  pos  pos_token  input                                   cf_answer  base_answer  clean_ld  null_ld  patched_ld
1    9    Ġof        The Colosseum is in the city of          Paris      Rome        -1.5      -1.5     -1.5
2    6    Ġof        Big Ben is in the city of                Berlin     London      -1.5      -1.5     -1.0
```

The null still equals the clean run, and row 1's float32 effect (-1.487 to -1.421) is gone. Recompute every
effect you report near the dtype's step in float32.

## Cross-checking engines

The hooks engine reaches the same modules through plain forward hooks, and `--engine ndif -o remote=local` makes
the ndif engine's trip in this process with no server. Where both agree with nnterp, the result does not depend
on how the model was reached:

```text
$ clab run documents/null_patch.json --out out/x_nnterp
$ clab run documents/null_patch.json --engine hooks --out out/x_hooks
$ clab run documents/null_patch.json --engine ndif -o remote=local --out out/x_ndif
$ for e in hooks ndif; do for f in out/x_nnterp/*; do [ "${f##*/}" = manifest.json ] || cmp "$f" "out/x_$e/${f##*/}"; done; done
$
```

`cmp` prints nothing: every value file and `tokens.json` match. Hooks serves module boundaries only, and
refuses the attention interior after the model loads, eager or not (`the hooks engine does not serve
attention_probabilities: nnterp reads it inside a module's forward …`). Compare engines under one `attn_implementation`.

## Out of memory

A step that runs out of CUDA memory stops the run, and is the one refusal that writes output: the steps before
it are saved and `clab show` prints a `stopped:` line. Rerun with a smaller `--batch-size`, or name and save
fewer values, since every value a step binds stays on the device until the run ends.
