---
name: clab
description: Write, run and read back a clab experiment, a JSON document that reads and writes a transformer's internals by nnterp component name and scores them. Load it first for any clab task. It covers the document's keys (model, data, sites, interventions, featurizers, steps, saves), the step kinds forward, generate, op, fit and load, entries and their ops, pos and scopes, sweeps and labels, and saves. It covers checking with clab compile, running with clab run on the nnterp, hooks, ndif or vllm engine, and reading back with clab show, clab table, Plan.load, the manifest and the safetensors files. It answers what clab can express (clab describe) and lists the silent mistakes that run without a refusal. For a technique's recipe go on to clab-patching, clab-ablation, clab-lenses-and-attention, clab-steering-and-generation or clab-trained-featurizers. For a refusal or wrong numbers, go to clab-debugging.
---

# clab

clab runs an interpretability experiment on a transformer language model from one JSON document. The
document holds the checkpoint, the data, the forwards and generations, the reads and writes at the model's
components, the ops that score them, and what to save. `clab run` runs it in one nnsight session, naming
components in nnterp's vocabulary, and writes each saved value to a directory that `clab table` prints as rows.

Verified on clab `dca9630` with nnterp `e5d0f8e`, nnsight 0.8.2 and transformers 5.20.2; the documents run on
the CPU on `openai-community/gpt2` in seconds, with their rows inline.

## Install and check

clab installs from a checkout of `https://github.com/JadenFiotto-Kaufman/clab`:

```bash
uv sync                      # nnsight, nnterp, torch and the rest, into .venv
source .venv/bin/activate
clab describe records        # prints the format's records: the install works
```

- **nnterp comes from a checkout** of `ndif-team/nnterp`, at the path `[tool.uv.sources]` in clab's `pyproject.toml` names: point it at yours.
- **Into another environment**, install nnterp first, then clab with `--no-deps`, since its `pyproject.toml` names nnterp by that local path.
- **Inside the venv a command is `clab …`**; outside it, `uv run --no-sync clab …`, since a plain `uv run` re-syncs and moves transformers.
- **The first run downloads the checkpoint**; after that, `HF_HUB_OFFLINE=1 clab run …` runs from the cache.
- **Check a document without a model**: `clab compile documents/x.json` prints what would run and exits 0, or prints the refusal `clab run` would print, naming the document path, and exits 1.
- **`clab describe` is the spec** on your machine; `clab describe ops` (or `components`, `pos`, `steps`, …) prints one section.

## A complete experiment

Does the residual stream leaving block 8 at the last token carry the name GPT-2 will say? Each base prompt
has a counterfactual whose second subject is the other name, so its answer is the other name.
`documents/ioi_patch.json` reads, writes, scores and saves:

```json
{
  "description": "Interchange on GPT-2's indirect-object task: swap in the residual stream leaving block 8 at the last token from a prompt naming the other subject, and score the counterfactual's answer against the base's. The null swaps the base's own value back.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"example_id": "mary", "input": "When Mary and John went to the store, John gave a drink to",
     "counterfactual": "When Mary and John went to the store, Mary gave a drink to", "base_answer": " Mary", "cf_answer": " John"},
    {"example_id": "tom", "input": "When Tom and Anna went to the park, Anna gave a ball to",
     "counterfactual": "When Tom and Anna went to the park, Tom gave a ball to", "base_answer": " Tom", "cf_answer": " Anna"},
    {"example_id": "sarah", "input": "When Sarah and James went to the office, James gave a book to",
     "counterfactual": "When Sarah and James went to the office, Sarah gave a book to", "base_answer": " Sarah", "cf_answer": " James"},
    {"example_id": "paul", "input": "When Paul and Lisa went to the school, Lisa gave a pen to",
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

- `model` pins the weights by commit; where they run is `--device` at run time, never the document's.
- `data.pairs` is four inline rows, read column by column as `pairs.<column>`; an answer carries the space the prompt implies.
- `sites` names a place once, so a read and a write land on the same one: `resid` is block 8's output, `head` the head's projection.
- `clean` and `counterfactual` are forwards whose entries have no ops, so they only read, at `pos: -1`, each row's own last token.
- `null` and `patched` write: an entry whose `ops` hold an edit (`swap`) replaces the value there, and the forward runs on from it.
- Their `logits` entries sit at a later site, so they read the written run.
- The `op` steps score: `logit_diff` is `cf_answer`'s logit less `base_answer`'s, and `match` is 1 where `cf_answer` is the top token.
- `saves` brings four values home, one file each; the forwards' reads stay in the run.

```text
$ clab run documents/ioi_patch.json
ioi_patch on nnterp (batch size 8, device cpu, tp 1) -> out/ioi_patch
  clean_ld.safetensors    clean_ld    [4,1]
  null_ld.safetensors     null_ld     [4,1]
  patched_ld.safetensors  patched_ld  [4,1]
  flipped.safetensors     flipped     [4,1]
  out/ioi_patch/manifest.json
$ clab table out/ioi_patch clean_ld null_ld patched_ld flipped
example_id  row  pos  pos_token  input                                             cf_answer  base_answer  clean_ld             null_ld              patched_ld          flipped
mary        0    13   Ġto        When Mary and John went to the store, John gave…   John       Mary        -2.0063934326171875  -2.0063934326171875  1.2577362060546875  1.0
tom         1    13   Ġto        When Tom and Anna went to the park, Anna gave a…   Anna       Tom         -2.5811233520507812  -2.5811233520507812  4.474357604980469   1.0
sarah       2    13   Ġto        When Sarah and James went to the office, James …   James      Sarah       -2.4727325439453125  -2.4727325439453125  2.5319671630859375  1.0
paul        3    13   Ġto        When Paul and Lisa went to the school, Lisa gav…   Lisa       Paul        -4.1465301513671875  -4.1465301513671875  3.94964599609375    1.0
```

The null equals the clean run on every row, so the read and the write are one place. The patch flips all four
rows to the other name. `pos` 13 and `pos_token` `Ġto` say where the scored read landed: GPT-2 adds no BOS, so
position 0 is `When`. `clab show out/ioi_patch` prints each step's cost, every entry's positions per row and each
saved value's layout ([output](references/output.md)).

## What breaks

Each mistake below runs without a refusal and gives wrong numbers.

- **An entry with no edit only reads.** `mul`, `neg` and the other arithmetic ops never write; `{"op": "mul", "with": 0}` leaves the clean logits. Write with an edit: `scale`, `swap`, `add` ([sites and interventions](references/sites-and-interventions.md#read-or-write)).
- **An answer without its leading space is another token.** On GPT-2, `"Paris"` scores 1e-6 where `" Paris"` scores 0.07, with no refusal. Write the answer as the model continues the prompt ([data](references/data.md#answer-columns)).
- **A result without a null proves nothing.** Writing back what was read must equal the clean run, as `null_ld` does above; compare runs only at one batch size, padding, readout, device and thread count ([running](references/running-and-engines.md#when-two-runs-give-the-same-numbers)).
- **After a fit, the bare featurizer name is the untrained draw.** Name the trained one `<fit>.<name>` (`fit.probe`) in every step after the fit ([featurizers and fit](references/featurizers-and-fit.md#members)).
- **A fit with no `epochs: 0` point has no floor.** Sweep `"epochs": {"sweep": [0, N]}`; the trained featurizer means something only where it beats its own draw ([featurizers and fit](references/featurizers-and-fit.md#epochs-0-is-the-null)).
- **A generate step without `do_sample: false` may sample**, from the checkpoint's generation config, and nothing seeds it. Write `"do_sample": false` on every generate step ([generate](references/generate.md#options)).
- **Two sweeps of one field without labels zip into the diagonal.** Give each its own `label` for a grid, then check the `axes` column of `clab show` ([sweeps](references/sweeps.md#labels-zip-or-cross)).
- **A write at a slice or list of layers patches them all at once.** To patch one layer at a time, sweep `layers` ([sites and interventions](references/sites-and-interventions.md#writes-across-layers)).
- **A swept write `pos` with an unswept read `pos` puts one read value at every position.** Sweep the read's `pos` with the write's, under one label ([sweeps](references/sweeps.md#sweep-one-value)).
- **An unscoped `pos` between two inputs of one length pairs index with index**, whatever tokens sit there. Scope both sides to the word they share ([positions](references/positions.md#pairing-with-another-forward)).
- **`occurrence` counts matches in the whole row**, from the first. On the IOI prompts `{"text": " to"}` is position 5, not 13; write `"occurrence": -1` for the last, and check `positions` in `clab show` ([positions](references/positions.md#scopes)).
- **An index along a `select`ed axis counts within the select list.** `argmax` over `{"num_heads": [9, 6]}` gives 1 for head 6; map it back through the list ([sites and interventions](references/sites-and-interventions.md#select)).
- **A search of generated text loses the first token's space on a SentencePiece tokenizer.** Give the column `"strip": true` wherever `contains` or a scope searches a generation ([generate](references/generate.md#the-first-generated-tokens-space)).
- **A mean over a generation skips rows it never reached.** Count the `alignment_missing` rows first ([positions](references/positions.md#rows-a-scope-never-reached)).
- **A freeze listed after the patch it protects undoes the patch.** Of two writes on one component and layer the later wins; list the freeze first ([sites and interventions](references/sites-and-interventions.md#the-order-entries-run-in)).
- **A `grad` null cannot catch the wrong metric.** A `grad` of the metric with `{"op": "mul", "with": 0}` appended must be exactly 0, which proves the gradient is of that entry and the wiring exact. That the entry is the metric you report takes a central difference at one point, and the ranking real patching of the top points ([check the gradient](../clab-patching/references/attribution.md#check-the-gradient)).
- **A published SAE with a wrong `centered`, `activation` or layer is not refused.** State all three inline from its release, and compare its mean L0 with the release's ([featurizers and fit](references/featurizers-and-fit.md#a-published-sparse-autoencoder)).
- **A null that passes in bfloat16 certifies the wiring, not small effects.** Recompute a few points in float32 before reading an effect near bfloat16's rounding ([the float32 rule](../clab-debugging/SKILL.md#the-float32-rule)).

## The document

| root key | holds |
|---|---|
| `description` | prose; copied to the manifest, never checked |
| `model` | `key`, `revision` (a commit), `dtype`; any other key goes to the loader, such as `"attn_implementation": "eager"`, but one that respells these or names a run option (`torch_dtype`, `device`) is refused at compile |
| `data` | each dataset by name: inline `rows`, a `path` under `data/`, or a hub `hf` at a `revision` |
| `sites` | named places: a `component`, its `layers`, an optional `select` |
| `interventions` | named bundles of entries that a step pulls in by name |
| `featurizers` | named spaces and maps an entry encodes into or applies |
| `steps` | what runs, in the order written |
| `saves` | the values written to the output directory, one file each |

`model`, `data` and `steps` are required, and any other root key is refused with the valid list.

### Steps and names

| kind | use it to |
|---|---|
| `forward` | run every row of `input` once, its entries reading and writing |
| `generate` | generate from every row; entries act at the prefill and the decode calls ([generate](references/generate.md)) |
| `op` | run one op over an earlier value: `{"kind": "op", "op": "mean", "of": "patched_ld"}` |
| `fit` | train featurizers inside the run, through the frozen model ([featurizers and fit](references/featurizers-and-fit.md)) |
| `load` | bring in a value an earlier run saved: `{"kind": "load", "from": "out/x", "value": "acts"}` |

- **Steps run in the order their keys are written**, and a value is named only after the step that makes it: `names no value produced before this step` means a step comes too early.
- **Steps, datasets and featurizers share one namespace**: a step named like a dataset is refused (`steps.lens: 'lens' is also a dataset or a featurizer`).
- **An entry's value is `<step>.<entry>`** (`clean.logits`), an op step's is its name (`clean_ld`), a column is `<dataset>.<column>` (`pairs.cf_answer`).
- **A fit's members are `<fit>.<featurizer>`, `<fit>.<eval>.<op>` and `<fit>.history.<op>`**; a `top_k` step's are `.ids`, `.probs` over a vocabulary or `.values` over another axis, and `.tokens` over logits.
- **A forward has no value of its own**: save its entries, never the step.

`clab describe steps` prints every kind's fields.

### Entries: read or write

An entry is `{"site": ..., "pos": ..., "ops": [...]}`. With no edit among its ops it reads; with one it writes,
and the forward runs on from the written value. The edits are the `edits` group of `clab describe ops`
(`swap`, `add`, `scale`, `project_out`, …). Ops after the last edit shape what the entry binds, not what the
model sees. Entries on one component and layer compose in the order written, whatever their site names. An operand
(`with`, `factor`, `scale`) is a number or a named value, and broadcasts over any axis it leaves out. A root
`interventions` bundle is a dict of entries that steps pull in by name. An entry whose first op is
`grad` reads a gradient. All of it is in [sites and interventions](references/sites-and-interventions.md).

### Sites

| component | layout | what it is |
|---|---|---|
| `layer_output`, `layer_input` | `rows, pos, hidden_size` | the residual stream leaving, or entering, block i |
| `attention_output`, `mlp_output` | `rows, pos, hidden_size` | what a sublayer adds to the stream |
| `attention_premix` | `rows, pos, num_heads, head_dim` | each head's result as the output projection takes it |
| `attention_probabilities` | `rows, pos, num_heads, key` | the pattern; needs `"attn_implementation": "eager"` |
| `mlp_activation`, `mlp_neurons` | `rows, pos, intermediate_size` | the MLP's neurons |
| `lm_head_output`, `logits` | `rows, pos, vocab_size` | the head's projection; the model's output after any softcap |

`layers` is an int (no layers axis), a slice or list (`":"`, `"2:6"`, `[1, 0]`: a `layers` axis first), or a
sweep (one point per layer); the model's own values take none. `select` narrows a feature axis and keeps it:
`{"num_heads": [9, 6]}` is two heads, in that order. `clab describe components` lists every component.

### Positions

| you mean | `pos` | on the IOI prompts |
|---|---|---|
| each row's last token | `-1` | 13 `Ġto` |
| the first token (GPT-2 adds no BOS) | `0` | 0 `When` |
| the last three tokens | `{"index": "-3:"}` | 11, 12, 13 |
| a row's own word, last token | `{"scope": {"column": "pairs.io"}, "index": -1}` | 1 `ĠMary` |
| the last match of a word | `{"scope": {"column": "pairs.s", "occurrence": -1}}` | 9 `ĠJohn` |
| the position that predicts it | `{"scope": {"column": "pairs.s", "occurrence": -1, "before": true}}` | 8 `,` |
| everything up to a text | `{"scope": {"text": ",", "until": true}}` | 0 to 8 |
| every decode call of a generation | `{"frame": "1:"}` | one position per generated input |

Leaving `pos` out reads every prompt token. A `frame` picks forward calls, a `scope` picks text, an `index`
counts within it. `clab describe pos` has the grammar; [positions](references/positions.md) has the rest.

### Ops

| class | ops | does |
|---|---|---|
| metrics | `match`, `logit_diff`, `token_prob`, `cross_entropy`, `kl`, `cosine`, `contains`, … | score each row and position, in float32 |
| reductions | `mean`, `sum`, `max`, `std`, `count`, `norm`, `entropy`, `pca`, … | remove the axes `over` names; by default rows and positions |
| transforms | `readout`, `argmax`, `top_k`, `detokenize`, `encode`, `grad`, `permute` | change the layout without writing back |
| arithmetic | `abs`, `exp`, `log`, `sign`, `sqrt`, `square`, `neg`, `mul`, `div`, `pow` | compute per entry, in float32 |
| comparisons | `eq`, `ne`, `gt`, `ge`, `lt`, `le` | a 0/1 mask |
| edits | `swap`, `add`, `scale`, `noise`, … | write back, inside an entry |

An op runs as an `op` step or inside an entry, as the same function. An entry's op runs on each batch and a
step's on the whole value. So an op whose bits depend on the value's shape (`readout`, `encode`, a long sum)
can differ in its last bits between the two, on the CPU too. Inside an entry an op sees one batch, so an op
over rows belongs in a step. `clab describe ops` lists every op, and [ops](references/ops.md) says which to
use for what.

### Sweeps

`{"sweep": [...]}` where a value goes runs the document at each value on one model load. Swept over layers,
`ioi_patch.json`'s site is `"resid": {"component": "layer_output", "layers": {"sweep": [0, 1, 2, 3, 4, 5, 6,
7, 8, 9, 10, 11]}}`, and mean steps over `null_ld` and `patched_ld` give, trimmed:

```text
at             mean_null           mean_patched
{"layers":0}   -2.801694869995117  -2.801473617553711
{"layers":6}   -2.801694869995117  -2.7089176177978516
{"layers":7}   -2.801694869995117  -0.9488182067871094
{"layers":8}   -2.801694869995117  3.053426742553711
{"layers":9}   -2.801694869995117  3.8118953704833984
```

The read and the write share the swept site, so they pair layer by layer. A step that uses a swept value
inherits its points; reads share one forward, and each write point runs its own. [Sweeps](references/sweeps.md)
has labels, costs and reading points back.

### Saves

Each name in `saves` is a step's value or a step's member (`clean_ld`, `patched.logits`, `fit.probe`), and
becomes one file. A name that names nothing is refused with every name that could be saved. Save what you
will read: `patched.logits` is `[4, 1, 50257]` floats where `patched_ld` is `[4, 1]`.

## Running

```text
clab run DOC [--engine nnterp|hooks|ndif|vllm] [--out DIR] [--batch-size N] [--padding dataset|batch|sorted]
             [--readout full|at-reads] [--device D] [-o KEY=VALUE ...]
```

<!-- test: skip -->
```python
import clab
plan = clab.run("documents/ioi_patch.json", out="out/ioi_patch", batch_size=8)   # the filled plan
plan.results["patched_ld"][:, 0]        # tensor([1.2577, 4.4744, 2.5320, 3.9496])
```

| engine | runs the model | use it for |
|---|---|---|
| `nnterp` (default) | here, as nnterp's standardized model in one nnsight session | everything |
| `hooks` | here, as plain transformers with forward hooks | a second implementation to cross-check a result |
| `ndif` | on an NDIF server (`-o remote=local` simulates the trip here) | models too large to load yourself |
| `vllm` | in vLLM's worker on the visible GPUs, one request per row, no gradients | cross-checking against vLLM's kernels |

`ioi_patch.json` writes byte-identical value files on nnterp, on hooks and on ndif under `-o remote=local`.
Without `--out` a run writes `out/<document name>`, and a second run into it leaves stale files.
[Running and engines](references/running-and-engines.md) has every flag, devices, quantized dtypes, `tp` and
memory.

## Reading back

- **`clab show OUT`** prints the run record, each step's cost, every entry's positions per row, and each saved value's layout and shape.
- **`clab table OUT NAME ...`** prints saved values as rows, labelled with row, position, token and the columns their steps used; `--csv` prints every cell whole.
- **`Plan.load(OUT)`** gives `results` (tensors by name), `table(...)` (rows as dicts), `steps` (each entry's `positions`) and `run` (versions, sizes, tokens, costs).
- **Each save is `<name>.safetensors`**, its tensor under the save's name (a swept one's keyed `layers=8`, a bundle's by its tensors), or `<name>.json` where it holds text.

[Output](references/output.md) has the columns, the joins and the manifest.

## Which skill

- **`clab-patching`**: which layer, position, head, neuron or path carries a behaviour; interchange, path and attribution patching.
- **`clab-ablation`**: what a component contributes when removed: zero, mean, resample, constant, neuron and expert ablation.
- **`clab-lenses-and-attention`**: what each layer predicts (logit and tuned lens) and where a head looks.
- **`clab-steering-and-generation`**: steering with a direction, and scoring what a generation says.
- **`clab-trained-featurizers`**: DAS, DBM, head masks, SAEs, probes, LoRA, cached harvests and bundles.
- **`clab-debugging`**: a refusal, numbers that look wrong, runs that differ, out of memory.

Depth, and extending clab with a new op, kind, component or engine, are in the docs:
`https://github.com/JadenFiotto-Kaufman/clab/tree/master/docs`.
