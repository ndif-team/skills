# Bundles and cached harvests

A bundle is a safetensors file holding one featurizer's tensors and a header naming what made them. A cached
harvest is a saved read that a `load` step brings into another document. Both resolve against the corpus
root, and both are checked against the loading document before the model loads. Output directories, `clab
show` and `Plan.load` are the `clab` skill's [output reference](../../clab/references/output.md); `--out` and
the corpus root are its [running reference](../../clab/references/running-and-engines.md).

## What breaks here

- **`file_path` and `from` resolve against the corpus root**
  ([which folder that is](../../clab/references/running-and-engines.md#where-output-goes)). `clab run` writes
  `out/<document>/` under the working directory, so run, show and table from the root; from elsewhere a bundle
  is refused (`out/das/fit.rot.safetensors does not exist under the corpus root; run the document that writes
  it first`).
- **Two runs agree to the bit at one batch size and thread count**, a fit's body at its own `batch_size`;
  what moves bits is the `clab-debugging` skill's
  [reproducibility](../../clab-debugging/SKILL.md#reproducing-a-number).
- **A cache is the run that made it.** Change a row, a `take` or a `shuffle`, and every load of the old cache is
  refused until the harvest runs again. Only names in the harvesting run's `saves` have files.

## What a run writes

A saved `fit.<name>`, and a saved `pca` op step, is a bundle: `out/<document>/<name>.safetensors`.

| kind | tensors (GPT-2, 768 wide) | header records, besides the model and site |
|---|---|---|
| `subspace` | `basis` `[768, k]` | `k`, `parametrization` |
| `pca` | `basis` `[768, k]`, `mean` `[768]` | `k` |
| `gate` | `theta` `[768]`; over heads, `[layers, num_heads]` | over heads, `over` and `kept.num_heads` |
| `sae` | `W_enc` `[768, latents]`, `b_enc`, `W_dec` `[latents, 768]`, `b_dec` | `latents`, `activation`, `centered` |
| `linear` | `W` `[768, classes]`, `b` `[classes]` | `classes`, as `json.dumps` writes the sorted list |
| `lora` | `A` `[768, rank]`, `B` `[rank, 768]` | `rank`, `alpha`; across sites, `with.component`, `with.layer` |
| `affine` | `W` `[768, 768]`, `b` `[768]`; over layers, `[layers, 768, 768]` and `[layers, 768]` | over layers, `over`, and `layer` lists them (`[4, 5]`) |
| `constant` | `vector` `[768]` | nothing more |

Every header also records `format` (`clab/1`), `kind`, `model.key`, `model.revision`, `model.dtype`,
`component`, `layer`, `width` and `document`, each as a string. To read one, from the corpus root:

<!-- test: skip -->
```python
from safetensors import safe_open

with safe_open("out/das/fit.rot.safetensors", "pt") as f:
    print(dict(sorted(f.metadata().items())), {k: list(f.get_tensor(k).shape) for k in f.keys()})
```

```text
{'component': 'layer_output', 'document': 'das', 'format': 'clab/1', 'k': '4', 'kind': 'subspace', 'layer': '5', 'model.dtype': 'float32', 'model.key': 'openai-community/gpt2', 'model.revision': '607a30d783dfa663caf39e06633721c8d4cfcd7e', 'parametrization': 'cayley', 'width': '768'} {'basis': [768, 4]}
```

A swept fit writes one file whose keys carry each point's coordinate (`mask.mask=1.0/theta`). No featurizer
loads it; rerun the document unswept at the point you keep. A hand-made bundle loads if it has the tensors
and the header a clab one has: a probe's direction as a `subspace` is in
[probes-and-lora.md](probes-and-lora.md#use-a-probe-causally). A file with no header is a published one, and
only an `sae` reads it ([sae.md](sae.md#splice-in-a-published-dictionary)).

## What a load is checked against

A field the header records may be left out of the loading document, which takes the header's value:
`das_apply.json` gives no `k`, and `lora_apply.json` neither `rank` nor `alpha`. A field the document does
give must match. Refused before the model loads, each against `das_apply.json`:

| the loading document | refusal |
|---|---|
| uses the rotation at layer 6 | `out/das/fit.rot.safetensors: header 'layer' is '5'; this document uses 'rot' at layer_output layer 6 (counterfactual.v_cf)` |
| writes `"k": 8` | `header 'k' is '4'; this document expects '8'` |
| loads the model in bfloat16 | `header 'model.dtype' is 'float32'; this document expects 'bfloat16'` |
| loads a swept fit's file | `out/dbm/fit.mask.safetensors holds one bundle per point of a sweep (mask.mask=1.0, ...); a featurizer loads one bundle, so run the document that writes it unswept` |
| trains the loaded featurizer in a fit | `steps.fit.train[0]: 'rot' is loaded from a bundle, which fixes it; a fit trains a featurizer it draws` |

A site's width, a layer counted from the end and a `select` open at its end are checked once the model
loads, before any forward.

## Cached harvests

A harvest is a read over every row of a dataset, saved. A later document that only fits or reduces it needs
the values, not the forwards. A `load` step brings a saved value in with its record (layout, positions, the
read's site), so every step after it treats it as the value it was.

```json
{
  "description": "Harvest GPT-2's layer 5 output at every token after the first of the seven weekday prompts, and save it for other documents to load.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"days": {"rows": [
    {"input": "If today is Monday, tomorrow is"}, {"input": "If today is Tuesday, tomorrow is"},
    {"input": "If today is Wednesday, tomorrow is"}, {"input": "If today is Thursday, tomorrow is"},
    {"input": "If today is Friday, tomorrow is"}, {"input": "If today is Saturday, tomorrow is"},
    {"input": "If today is Sunday, tomorrow is"}
  ]}},
  "steps": {
    "harvest": {"kind": "forward", "input": "days.input",
                "interventions": {"acts": {"site": {"component": "layer_output", "layers": 5}, "pos": {"index": "1:"}}}}
  },
  "saves": ["harvest.acts"]
}
```

```text
clab run documents/harvest.json
harvest on nnterp (batch size 8, device cpu, tp 1) -> out/harvest
  harvest.acts.safetensors  harvest  [7,6,768]
```

The loading document keeps the same `model` and the same dataset, under the same name, and loads the read in
place of the forward:

<!-- test: skip -->
```json
{
  "description": "Train a 64-latent dictionary on the harvest harvest.json saved: the load replaces the forward, so the run loads no weights and the fit makes no model call.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"days": {"rows": [
    {"input": "If today is Monday, tomorrow is"}, {"input": "If today is Tuesday, tomorrow is"},
    {"input": "If today is Wednesday, tomorrow is"}, {"input": "If today is Thursday, tomorrow is"},
    {"input": "If today is Friday, tomorrow is"}, {"input": "If today is Saturday, tomorrow is"},
    {"input": "If today is Sunday, tomorrow is"}
  ]}},
  "featurizers": {"dictionary": {"kind": "sae", "latents": 64}},
  "steps": {
    "harvest": {"kind": "load", "from": "out/harvest", "value": "harvest.acts"},
    "fit": {
      "kind": "fit", "train": ["dictionary"], "objective": {"dictionary.reconstruction": 1.0, "dictionary.l1": 5.0},
      "epochs": 30, "batch_size": 4, "seed": 0, "optimizer": {"name": "adam", "lr": 0.01},
      "steps": {"latents": {"kind": "op", "op": "encode", "featurizer": "fit.dictionary", "of": "harvest"}}
    }
  },
  "saves": ["fit.history.dictionary.reconstruction", "fit.dictionary"]
}
```

- **The step's value is named by the step**, `harvest`, as an op step's is.
- **No forward runs.** `clab show out/sae_cached` lists 0 forwards for both steps and `served: float32 on
  meta`: a document of loads and op steps loads no weights.
- **The fit equals one on the harvest made in-document.** The same document with `harvest` a forward of
  `days.input` at layer 5 from position 1 trains a dictionary equal to the bit, at one batch size.
- **An `encode` of a loaded read uses the featurizer at the read's site**, so this document declares no
  `sites`. An op's value over a read, loaded or not, is no read's own, and its `encode` is refused.

A loaded read feeds any step a read does, and no fit is needed. These steps, in a document with
`harvest.json`'s model and `days` and the sites `target` (layer 5's `layer_output`) and `head`
(`lm_head_output`), mean-ablate the last token with the harvest's mean (the `clab-ablation` skill's recipe):

```json
{
  "harvest": {"kind": "load", "from": "out/harvest", "value": "harvest.acts"},
  "average": {"kind": "op", "op": "mean", "of": "harvest"},
  "clean": {"kind": "forward", "input": "days.input", "interventions": {"logits": {"site": "head", "pos": -1}}},
  "ablated": {"kind": "forward", "input": "days.input", "interventions": {
    "abl": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "average"}]},
    "logits": {"site": "head", "pos": -1}}},
  "kl": {"kind": "op", "op": "kl", "of": "clean.logits", "against": "ablated.logits"}
}
```

`clab show` lists 0 forwards for `harvest` and `average`, and `kl` equals the same document's with the
harvest's forward in place of the load to the bit.

A load is refused before any model loads when it does not match its source:

| the loading document | refusal |
|---|---|
| holds 6 of the 7 rows | `steps.harvest.value: 'harvest.acts' was made over the rows of dataset 'days', and this document's 'days' is not the rows out/harvest read, in their order: 6 rows where it read 7` |
| names the dataset `prompts` | `'harvest.acts' was made over the rows of dataset 'days', which this document does not declare; declare it as out/harvest did` |
| loads the model in bfloat16 | `steps.harvest.from: out/harvest ran another model than this document's: dtype 'float32' where this document's is 'bfloat16'` |
| loads a bundle as a value | `steps.b.value: 'basis' is a bundle, which a featurizer of its kind loads, with 'file_path': 'out/pca_harvest/basis.safetensors'` |

A value in a featurizer's coordinates, a generation's value and a load inside a fit's body do not load either.
A harvest swept over `layers` loads at every point, keyed as the run keyed it, with no sweep in the loading
document.

## A harvest in shards

A harvest too large for one run is made in shards: one shuffled order, cut with `skip` and `take`, one run per
shard. `harvest.json` with these dataset blocks in place of its `days` (the same seven rows inline):

| shard | dataset block, beside `"rows"` | value |
|---|---|---|
| 0 | `"shuffle": true, "skip": 0, "take": 4` | `[4,6,768]` |
| 1 | `"shuffle": true, "skip": 4, "take": 3` | `[3,6,768]` |
| all | `"shuffle": true` | `[7,6,768]` |

The two shards joined in order equal the harvest of all seven shuffled rows to the bit. A load of shard 1 needs
a dataset block giving the same rows: `"shuffle": true, "skip": 4` loads it. Nothing in a document joins
shards into one value; join them in Python, or fit on each.
