---
name: nnspec
description: "Run an interpretability experiment on a transformer language model as one JSON document, with no intervention code to write. Load this first, before writing any code, whenever a request asks what a model's internals do and names no library: logit and tuned lens, activation, path and attribution patching, causal tracing, zero, mean and resample ablation, attention patterns and finding induction or other attention heads, steering vectors and scoring what a model generates, linear probes, SAEs, DAS, LoRA. Covers which layer, position, head or neuron carries a behaviour, what each layer predicts and what a direction does. Also for any nnspec task: writing a document (model, data, sites, steps, ops, sweeps, saves), nnspec compile, run and table, the nnterp, hooks, ndif and vllm engines, a refused document, or wrong numbers. When the experiment needs Python logic a document cannot express, use the nnterp skill; for a model family nnterp does not support, or raw module access, use the nnsight skill."
---

# nnspec

nnspec runs an interpretability experiment on a transformer language model from one JSON document. The
document holds the checkpoint, the data, the forwards and generations, the reads and writes at the model's
components, the ops that score them, and what to save. `nnspec run` runs it in one nnsight session, naming
components in nnterp's vocabulary, and writes each saved value to a directory that `nnspec table` prints as rows.

This file is enough for a patch, an ablation, a lens and a steering experiment; every other task has a guide,
listed by task under [Guides](#guides).

## Which library

nnspec is the top of three libraries. When the user names a library, use it. Otherwise take the highest row that fits,
and take a row only when its skill is available in this session:

| Use | When |
|---|---|
| nnspec (this skill) | A standard experiment on a transformer language model (lens, patching, ablation, attention, steering, probes, SAEs, DAS) that fits a JSON document. No intervention code to write. |
| `nnterp` skill | The experiment needs Python that a document cannot express, on a family nnterp knows. One module vocabulary on every family. |
| `nnsight` skill | nnterp does not know the family, or the work needs access below nnterp's standard values: arbitrary modules, `.source`, your own or non-transformer models, diffusion, vLLM, tensor parallelism. Also existing nnsight code, and any error an nnsight trace raises. |

## Install and check

nnspec is not on PyPI; it installs from a checkout, with Python 3.10 or later:

```bash
git clone https://github.com/JadenFiotto-Kaufman/nnspec.git && cd nnspec
uv sync                      # nnsight, nnterp, torch and the rest, into .venv
source .venv/bin/activate
nnspec describe records      # prints the format's records: the install works
```

- **nnterp comes from a checkout** of `ndif-team/nnterp` (its `v2.0.0` branch), at the path `[tool.uv.sources]` in nnspec's `pyproject.toml` names: point that path at yours before `uv sync`.
- **Into an environment that already has the dependencies** `pyproject.toml` lists, that nnterp among them: `uv pip install --no-deps -e .` from the checkout.
- **Without the venv activated** the command is `<env>/bin/nnspec`, beside that environment's `python` (there is no `python -m nnspec`), or `uv run --no-sync nnspec …` from the checkout; a plain `uv run` re-syncs the environment first.
- **The first run downloads the checkpoint**; after it, `HF_HUB_OFFLINE=1` runs from the cache.

## Write, check, run, read

```text
nnspec describe [records|steps|ops|components|featurizers|pos|engines]   the format this install accepts
nnspec compile documents/x.json    no model: prints what would run, or the refusal `nnspec run` would print
nnspec run documents/x.json        runs it, on the CPU unless --device says otherwise, and writes out/x/
nnspec show out/x                  the run record, and where every read and write landed on every row
nnspec table out/x NAME ...        saved values as rows, labelled with row, position, token and data columns
```

- **`describe` is the reference.** Take an op's arguments, a component's layout or a step's fields from it, not from memory.
- **Compile after every edit.** A refusal is one line, `nnspec: <where>: <what>`, exit status 1, nothing written. It names the document path and lists what is valid there (`known: …`, `valid: …`): fix from the message.
- **Keep documents in `documents/` of a project folder and run from that folder.** Output goes to `out/<document name>`, and files a document names (`data/`, bundles, earlier runs) resolve against the folder.

## A complete experiment

Does the residual stream leaving block 8 at the last token carry the name GPT-2 will say? Each base prompt
has a counterfactual whose second subject is the other name, so its answer is the other name. Saved as
`documents/ioi_patch.json`:

```json
{
  "description": "Swap GPT-2's block 8 output at the last token in from a prompt naming the other subject, and score the other name against the base's. The null swaps the base's own value back.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"input": "When Mary and John went to the store, John gave a drink to",
     "counterfactual": "When Mary and John went to the store, Mary gave a drink to", "base_answer": " Mary", "cf_answer": " John"},
    {"input": "When Tom and Anna went to the park, Anna gave a ball to",
     "counterfactual": "When Tom and Anna went to the park, Tom gave a ball to", "base_answer": " Tom", "cf_answer": " Anna"},
    {"input": "When Sarah and James went to the office, James gave a book to",
     "counterfactual": "When Sarah and James went to the office, Sarah gave a book to", "base_answer": " Sarah", "cf_answer": " James"},
    {"input": "When Paul and Lisa went to the school, Lisa gave a pen to",
     "counterfactual": "When Paul and Lisa went to the school, Paul gave a pen to", "base_answer": " Paul", "cf_answer": " Lisa"}
  ]}},
  "sites": {
    "resid": {"component": "layer_output", "layers": 8},
    "head": {"component": "lm_head_output"}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": {
      "v_self": {"site": "resid", "pos": -1},
      "logits": {"site": "head", "pos": -1}}},
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {
      "v_cf": {"site": "resid", "pos": -1}}},
    "null": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "resid", "pos": -1, "ops": [{"op": "swap", "with": "clean.v_self"}]},
      "logits": {"site": "head", "pos": -1}}},
    "patched": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "resid", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
      "logits": {"site": "head", "pos": -1}}},
    "clean_ld": {"kind": "op", "op": "logit_diff", "of": "clean.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "null_ld": {"kind": "op", "op": "logit_diff", "of": "null.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "patched_ld": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "flipped": {"kind": "op", "op": "match", "of": "patched.logits", "expected": "pairs.cf_answer"}
  },
  "saves": ["clean_ld", "null_ld", "patched_ld", "flipped"]
}
```

- `model` is one checkpoint, pinned: `revision` is required, the commit hash. For a cached checkpoint it is the directory name under `~/.cache/huggingface/hub/models--<org>--<name>/snapshots/`. Two checkpoints are two documents. Where the model runs is `--device` at run time, never the document's.
- `data.pairs` is inline rows, read as `pairs.<column>`. The counterfactual is a column of the base rows' dataset, never a dataset of its own, and each answer carries the space the prompt implies.
- `sites` names a place once, so a read and a write land on the same one.
- An entry with no `ops` reads: `clean` and `counterfactual` read at `pos: -1`, each row's own last token.
- An entry whose `ops` hold an edit (`swap`) writes, and the forward runs on from it. `logits` sits at a later site, so it reads the written run.
- The `op` steps score: `logit_diff` is `a`'s logit less `b`'s, and `match` is 1 where `expected` is the top token.
- `saves` lists what the run writes, one file each.

```text
$ nnspec run documents/ioi_patch.json
ioi_patch on nnterp (batch size 8, device cpu, tp 1) -> out/ioi_patch
  clean_ld.safetensors    clean_ld    [4,1]
  ...
$ nnspec table out/ioi_patch clean_ld null_ld patched_ld flipped
row  pos  pos_token  input                                             cf_answer  base_answer  clean_ld             null_ld              patched_ld          flipped
0    13   Ġto        When Mary and John went to the store, John gave…   John       Mary        -2.0063934326171875  -2.0063934326171875  1.2577362060546875  1.0
1    13   Ġto        When Tom and Anna went to the park, Anna gave a…   Anna       Tom         -2.5811233520507812  -2.5811233520507812  4.474357604980469   1.0
2    13   Ġto        When Sarah and James went to the office, James …   James      Sarah       -2.4727325439453125  -2.4727325439453125  2.5319671630859375  1.0
3    13   Ġto        When Paul and Lisa went to the school, Lisa gav…   Lisa       Paul        -4.1465301513671875  -4.1465301513671875  3.94964599609375    1.0
```

The null equals the clean run on every row, to the bit, so the read and the write are one place; only then is
`patched_ld` an effect. The patch flips all four rows. `pos` 13 and `pos_token` `Ġto` say where the scored read
landed: GPT-2 adds no BOS, so position 0 is `When`.

## An ablation, a sweep, a lens, a steer

**Ablation** changes what `patched` writes: `{"op": "swap", "with": 0.0}` for zero, or `{"op": "swap", "with":
"mean"}` with the step `"mean": {"kind": "op", "op": "mean", "of": "clean.v_self"}` written after `clean`. A
`mean` with no `over` removes the rows and positions, pooling every token read once, and the vector left
broadcasts over both. The self-swap stays the null.
Averaged over the rows, the logit difference is -2.80 clean, +0.26 under zero and -2.05 under the mean: a zero
vector is a point no forward reaches, so report mean (or resample) ablation and keep zero as the extreme.

**A sweep** runs the document at each value on one model load: `"layers": {"sweep": [0, 1, 2, 3, 4, 5, 6, 7, 8,
9, 10, 11]}` on `resid` patches each layer in turn. The read and the write share the swept site, so they pair
layer by layer, and every step that uses it has twelve points, keyed in `at`. With `mean` steps over `null_ld`
and `patched_ld`:

```text
$ nnspec table out/ioi_layer_sweep mean_null mean_patched
at             mean_null           mean_patched
{"layers":0}   -2.801694869995117  -2.801473617553711
{"layers":6}   -2.801694869995117  -2.7089176177978516
{"layers":7}   -2.801694869995117  -0.9488182067871094
{"layers":8}   -2.801694869995117  3.053426742553711
{"layers":9}   -2.801694869995117  3.8118953704833984
```

Reads share one forward across points, and each point that writes runs its own: a document does not batch
different interventions into one forward, which is the nnsight skill's ground. A second sweep with its own
`label` crosses the first into a grid ([sweeps](references/sweeps.md)).

**The logit lens** reads what each layer predicts. `readout` projects a residual-stream value through the
model's own final norm, head and whatever follows the head (Gemma-2's logit softcap), and any metric scores the
result as it scores logits. `"layers": ":"` reads every block in one forward. `documents/ioi_lens.json` is `ioi_patch.json` with these `sites`, `steps` and `saves`:

```json
{
  "sites": {"resid": {"component": "layer_output", "layers": ":"}, "out": {"component": "logits"}},
  "steps": {
    "run": {"kind": "forward", "input": "pairs.input", "interventions": {
      "h": {"site": "resid", "pos": -1}, "final": {"site": "out", "pos": -1}}},
    "lens": {"kind": "op", "op": "readout", "of": "run.h"},
    "p": {"kind": "op", "op": "token_prob", "of": "lens", "token": "pairs.base_answer"},
    "p_answer": {"kind": "op", "op": "mean", "of": "p"},
    "kl": {"kind": "op", "op": "kl", "of": "lens", "against": "run.final"},
    "to_final": {"kind": "op", "op": "max", "of": "kl"},
    "top": {"kind": "op", "op": "top_k", "of": "lens", "k": 3}
  },
  "saves": ["p_answer", "to_final", "top.tokens"]
}
```

```text
$ nnspec table out/ioi_lens p_answer to_final
layer  p_answer                to_final
8      0.00025329587515443563  2.042168617248535
9      0.780062198638916       1.293728232383728
10     0.8180603981018066      0.7158426642417908
11     0.42094409465789795     0.0
```

The null is the wiring check: `to_final`, the largest KL from `logits`, the model's own output, must be 0 at
the last layer (0.0 here, float32 rounding on other families), or the lens is not the model's computation. A
reduction keeps the `layers` axis, so each value is one number per layer; the answer forms at layer 9.
`top.tokens` is each layer's three likeliest tokens, decoded.

**Steering** adds a direction at a dose. The direction is a mean difference of two prompt sets, and the dose a
fraction of the activations' own norm, so dose 0 is the null. These are the `steps` of `steer.json`, whose
dataset `tone` holds four sentence pairs in columns `pos` and `neg`, and `prompts` three prompts to steer
(the whole document is in the [steering guide](references/steering.md)); `resid` is block 6's `layer_output`:

```json
{
  "steps": {
    "pos": {"kind": "forward", "input": "tone.pos", "interventions": {"r": {"site": "resid", "pos": -1}}},
    "neg": {"kind": "forward", "input": "tone.neg", "interventions": {"r": {"site": "resid", "pos": -1}}},
    "pos_mean": {"kind": "op", "op": "mean", "of": "pos.r"},
    "neg_mean": {"kind": "op", "op": "mean", "of": "neg.r"},
    "difference": {"kind": "op", "op": "sub", "of": "pos_mean", "with": "neg_mean"},
    "direction": {"kind": "op", "op": "normalize", "of": "difference"},
    "base": {"kind": "forward", "input": "prompts.input", "interventions": {
      "r": {"site": "resid", "pos": -1}, "logits": {"site": "head", "pos": -1}}},
    "norms": {"kind": "op", "op": "norm", "of": "base.r"},
    "typical": {"kind": "op", "op": "median", "of": "norms"},
    "steer": {"kind": "op", "op": "scale", "of": "direction", "factor": "typical"},
    "steered": {"kind": "forward", "input": "prompts.input", "interventions": {
      "add": {"site": "resid", "pos": -1,
              "ops": [{"op": "add", "with": "steer", "scale": {"sweep": [0.0, 0.25, 0.5, 1.0], "label": "dose"}}]},
      "logits": {"site": "head", "pos": -1}}},
    "ld": {"kind": "op", "op": "logit_diff", "of": "steered.logits", "a": {"text": " great"}, "b": {"text": " terrible"}},
    "effect": {"kind": "op", "op": "mean", "of": "ld"},
    "kl": {"kind": "op", "op": "kl", "of": "steered.logits", "against": "base.logits"},
    "drift": {"kind": "op", "op": "mean", "of": "kl"}
  },
  "saves": ["typical", "effect", "drift"]
}
```

```text
$ nnspec table out/steer effect drift
at             effect              drift
{"dose":0.0}   2.5226821899414062  0.0
{"dose":0.25}  4.361429691314697   0.077471524477005
{"dose":0.5}   5.752100467681885   0.2544526755809784
{"dose":1.0}   7.399055480957031   0.7366573214530945
```

Dose 0 gives the unsteered effect and a drift of exactly 0.0. Dose 0.5 adds a vector half as long as the median
activation (`typical`, 89.2): the logit difference moves by 3.2 for 0.25 nats of drift. Report the effect
beside the drift, and beside a random direction of the same length, the control that shows the direction
matters; that control, and steering a `generate` step, are in the steering guide.

## What breaks

Each mistake below runs without a refusal and gives wrong numbers.

- **An entry with no edit only reads.** `mul`, `neg` and the other arithmetic ops never write: `{"op": "mul", "with": 0}` leaves the clean logits. Write with an edit: `swap`, `add`, `scale`, `project_out`.
- **An answer without its leading space is another token.** On GPT-2, `"Paris"` scores 1e-6 where `" Paris"` scores 0.07. Write the answer as the model continues the prompt ([data](references/data.md#answer-columns)).
- **A result without its null proves nothing.** Writing back what was read must equal the clean run to the bit; a gap means the read and the write are not one selection. Dose 0 and a lens's last layer are nulls of the same kind. Compare two runs only at one batch size, padding and device ([debugging](references/debugging.md#reproducing-a-number)).
- **A write at a slice or list of layers patches them all at once.** To patch one layer at a time, sweep `layers`.
- **Two sweeps of one field without labels zip into the diagonal.** Give each its own `label` for a grid, and check the `axes` column of `nnspec show` ([sweeps](references/sweeps.md#labels-zip-or-cross)).
- **A swept write `pos` with an unswept read `pos` puts one read value at every position.** Sweep both under one label.
- **An unscoped `pos` between two inputs pairs index with index**, whatever tokens sit there. Where the prompts differ in length or wording, scope each side to its own word ([positions](references/positions.md#pairing-with-another-forward)).
- **`occurrence` counts matches in the whole row, from the first.** On the IOI prompts `{"text": " to"}` is position 5, not 13: write `"occurrence": -1` for the last, and check each entry's positions in `nnspec show`.
- **A generate step without `"do_sample": false` may sample**, from the checkpoint's generation config, unseeded. A write at `pos: -1` there lands in the prefill only ([generation](references/generation.md#writes-during-generation)).
- **A refused run writes nothing, so `out/<name>` still holds the last run that finished.** Check `started` in `nnspec show`, and give each run its own `--out`.

Fits, gradients, generated text and bfloat16 have their own traps, in their guides.

## The document

The root keys are `description`, `model`, `data`, `sites`, `interventions`, `featurizers`, `steps` and `saves`;
`model`, `data` and `steps` are required. `model` holds `key`, `revision` (a commit) and `dtype`, and any other
key goes to the loader (`"attn_implementation": "eager"`); `key` and `revision` are required, and a document
has one model. `nnspec describe records` prints every field.

**Steps** are `forward`, `generate`, `op` (one op over an earlier value), `fit` (train a featurizer inside the
run) and `load` (a value an earlier run saved). They run in the order their keys are written, and a value is
named only after the step that makes it: an entry's is `<step>.<entry>` (`clean.logits`), an op step's its name
(`clean_ld`), a column `<dataset>.<column>`. A bare string where an answer or a text goes is a column
reference; a literal is `{"text": " great"}`.

**An entry** is `{"site": ..., "pos": ..., "ops": [...]}`. With no edit among its ops it reads; with one it
writes, and the ops after the last edit shape what the entry binds, not what the model sees. Entries on one
component and layer compose in the order written. An operand (`with`, `factor`, `scale`) is a number or a named
value, broadcast over the axes it leaves out. A root `interventions` bundle is a dict of entries that steps
pull in by name ([sites and interventions](references/sites-and-interventions.md)).

**A site** is a `component`, its `layers` and an optional `select`. `nnspec describe components` lists every
component with its layout; most documents use these:

- `layer_output`, `layer_input`: the residual stream leaving, or entering, block i, `[rows, pos, hidden_size]`.
- `attention_output`, `mlp_output`: what a sublayer adds to the stream.
- `attention_premix`: each head's result, `[rows, pos, num_heads, head_dim]`, the per-head site to patch or ablate.
- `attention_probabilities`: the pattern, `[rows, pos, num_heads, key]`; needs `"attn_implementation": "eager"` in `model`.
- `mlp_activation`: the MLP's neurons, `[rows, pos, intermediate_size]`.
- `lm_head_output`, `logits`: the head's projection, and the model's output after any softcap, `[rows, pos, vocab_size]`.

`layers` is an int (no `layers` axis), a slice or list (`":"`, `"2:6"`, `[1, 0]`: a `layers` axis first), or a
sweep; the model's own values take none. `select` narrows a feature axis and keeps it: `{"num_heads": [9, 6]}`
is two heads in that order, and an index along the axis then counts within the list.

**`pos`** picks positions: a row's input tokens from 0, a BOS included where the tokenizer adds one.

| you mean | `pos` | on the IOI prompts |
|---|---|---|
| each row's last token | `-1` | 13 `Ġto` |
| the last three tokens | `{"index": "-3:"}` | 11, 12, 13 |
| a row's own word, last token | `{"scope": {"column": "pairs.base_answer"}, "index": -1}` | 1 `ĠMary` |
| the last match of a text | `{"scope": {"text": " to", "occurrence": -1}}` | 13 |
| the position that predicts it | the same scope with `"before": true` | 12 |
| every decode call of a generation | `{"frame": "1:"}` | one per generated input |

Leaving `pos` out reads, or writes, every prompt token. `nnspec describe pos` has the grammar and
[positions](references/positions.md) the rest.

**Ops** run as an `op` step or inside an entry; one over rows (`mean`, `permute`, `pca`) belongs in a step,
since an entry sees one batch. `nnspec describe ops` gives each op's arguments, and [ops](references/ops.md)
which to use for what.

- **Metrics**, one number per row and position: `match`, `logit_diff`, `token_prob`, `token_logit`, `cross_entropy`, `kl`, `js`, `cosine`, `contains`.
- **Reductions**, over rows and positions unless `over` lists the axes (`"over": ["pos"]`): `mean`, `sum`, `min`, `max`, `std`, `median`, `count`, `norm`, `dot`, `entropy`, `pca`.
- **Transforms**: `readout`, `argmax`, `top_k`, `select`, `detokenize`, `permute`, `encode`, `grad`.
- **Arithmetic and comparisons**, never written back: `abs`, `exp`, `log`, `sign`, `sqrt`, `square`, `neg`, `mul`, `div`, `pow`, `eq`, `ne`, `gt`, `ge`, `lt`, `le`.
- **Edits**, written back inside an entry: `swap`, `add`, `sub`, `scale`, `noise`, `clamp`, `project_out`, `normalize`, `renormalize`, `decode`, `apply`.

A metric scores one answer token per position. For an answer of several tokens, append it to the prompt
(`"input": ["facts.prompt", "facts.answer"]`), read the head at `{"scope": {"column": "facts.answer",
"occurrence": -1, "before": true}}`, and sum `cross_entropy`, each token's negative log-probability, over
`["pos"]` ([teacher forcing](references/generation-teacher-forcing.md)).

**`saves`** names values or their members (`clean_ld`, `patched.logits`, `top.tokens`). Save what you will read:
`patched.logits` is `[4, 1, 50257]` floats where `patched_ld` is `[4, 1]`.

## Running and reading back

```text
nnspec run DOC [--engine nnterp|hooks|ndif|vllm] [--out DIR] [--batch-size N] [--padding dataset|batch|sorted]
             [--readout full|at-reads] [--device D] [-o KEY=VALUE ...]
```

`nnterp`, the default, runs the model here and serves everything. `hooks` is a second implementation on plain
transformers to cross-check a result, `ndif` runs on an NDIF server, and `vllm` in vLLM's worker on a GPU
([running and engines](references/running-and-engines.md)).

```python
import nnspec
from nnspec import Plan

plan = nnspec.run("documents/ioi_patch.json")        # compiles, runs, writes out/ioi_patch, returns the filled plan
plan = Plan.load("out/ioi_patch")                    # or read a finished run back
plan.results["patched_ld"][:, 0]                     # tensor([1.2577, 4.4744, 2.5320, 3.9496])
plan.table("patched_ld", "flipped")[0]               # {'row': 0, 'pos': 13, 'pos_token': 'Ġto', ..., 'patched_ld': 1.2577362060546875, 'flipped': 1.0}
plan.steps["patched"].interventions["patch"].positions   # {'pairs': [[13], [13], [13], [13]]}

swept = Plan.load("out/ioi_layer_sweep")
swept.sweeps                                         # {'layers': Axis(values=[0, 1, ..., 11])}
swept.results["mean_patched"][8]                     # tensor(3.0534): a swept save is a dict keyed by the swept value,
                                                     # by a tuple in the order of .sweeps where it has several axes
swept.table("mean_patched")[8]                       # {'at': {'layers': 8}, 'value': 3.053426742553711}
```

- **A value is laid out as `nnspec show` prints its `dims`**, rows in dataset order. Where rows hold different numbers of positions it is as wide as the widest row, NaN past each row's own.
- **No op compares across sweep points or finds the first layer where something holds.** Save the per-point or per-layer values and compute such scores (the first layer whose top token is the answer, the fraction of an effect recovered, a ranking) in Python from `results`.
- **Each save is also a file**, `out/<name>/<save>.safetensors`, or `.json` where it holds text, its swept entries keyed `layers=8` ([output](references/output.md)).

## Guides

- **Which layer, position, head, neuron or path carries a behaviour**: maps by layer and position, prompts of different lengths, windows, causal tracing, patching into a generation: [patching](references/patching.md). One head, head sets, neurons, path and attention-pattern patching: [heads, neurons and paths](references/patching-heads-neurons-and-paths.md). Ranking every component from one backward: [attribution patching](references/patching-attribution.md).
- **What is lost without a component**: resample ablation, heads and neurons with random controls, a direction, noise, an optimal constant, the hydra effect: [ablation](references/ablation.md). Attention knockout and mixture-of-experts routing: [attention and experts](references/ablation-attention-and-experts.md).
- **What each layer predicts**, beyond the lens above: the model's top tokens, a sublayer's lens, a tuned lens: [lens](references/lens.md).
- **Where heads look**: induction and previous-token heads, one head's pattern, entropy, queries and keys: [attention](references/attention.md).
- **Steering with a direction**: the dose, random-direction controls, layer grids, steering a generation: [steering](references/steering.md).
- **Generating text and scoring it**: did it say the answer, where and how confidently, the text after a marker, reasoning length, chats, teacher forcing an answer of several tokens: [generation](references/generation.md).
- **Probes, SAEs, DAS, masks over heads or units, LoRA** (`featurizers`, `fit`, saved bundles): [featurizers](references/featurizers.md), then [subspaces and masks](references/featurizers-subspaces-and-masks.md), [SAEs](references/featurizers-sae.md), [probes and LoRA](references/featurizers-probes-and-lora.md), and [bundles and cached harvests](references/featurizers-bundles-and-harvests.md) to reuse one run's values in another.
- **Datasets**: files, hub datasets, narrowing and deriving columns, answer columns, joined inputs, chats: [data](references/data.md).
- **Reads and writes**: choosing a component, `layers` and `select`, the order entries run in, operands, reusable entries, a gradient: [sites and interventions](references/sites-and-interventions.md).
- **Pointing at a word, a span, a chat turn or generated tokens**, and checking where it landed: [positions](references/positions.md).
- **Which op** for a score, a reduction, a ranking or a comparison: [ops](references/ops.md).
- **One document over layers, positions, heads, doses or seeds**, crossed or zipped, and reading the points back: [sweeps](references/sweeps.md).
- **Engines, devices, batch size and padding**, quantized and sharded models, NDIF, vLLM, nnspec from Python: [running and engines](references/running-and-engines.md).
- **Reading a run**: `nnspec show`, `nnspec table`, results in Python, a saved value elsewhere: [output](references/output.md).
- **A refusal, wrong numbers with nothing refused, two runs that differ, out of memory**: [debugging](references/debugging.md).

Extending nnspec with a new op, kind, component or engine is in its docs:
`https://github.com/JadenFiotto-Kaufman/nnspec/tree/master/docs`.
