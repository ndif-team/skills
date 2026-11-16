# Bundles and cached harvests

A bundle is a safetensors file holding one featurizer's tensors and a header naming what made them; a
featurizer loads it by `file_path`. A cached harvest is any other saved value, which a `load` step brings
into another document. Both are checked against the loading document before the model loads. Declaring and
loading a featurizer is in [featurizers](featurizers.md#bundles).

## What breaks

- **`file_path` and a load's `from` resolve against the corpus root**, the project folder that holds
  `documents/` ([the rule](running-and-engines.md#where-output-goes)). `nnspec run` writes `out/<document>/`
  under the working directory, so run, show and table from that folder; from elsewhere the file `does not
  exist under the corpus root`.
- **A cache is the run that made it.** Change a row, a `take` or a `shuffle`, and every load of the old cache
  is refused until the harvest runs again. Only names in the harvesting run's `saves` have files.

## What a run writes

A saved `<fit>.<featurizer>`, and a saved `pca` op step, is a bundle, `out/<document>/<save>.safetensors`:

| kind | tensors (GPT-2, 768 wide) | its header also records |
|---|---|---|
| `subspace` | `basis` `[768, k]` | `k`, `parametrization` |
| `pca` | `basis` `[768, k]`, `mean` `[768]` | `k` |
| `gate` | `theta` `[768]`; over heads, `[layers, num_heads]` | over axes: `over`, and the indices kept on each (`kept.num_heads`) |
| `sae` | `W_enc` `[768, latents]`, `b_enc`, `W_dec` `[latents, 768]`, `b_dec` | `latents`, `activation`, `centered` |
| `linear` | `W` `[768, classes]`, `b` `[classes]` | `classes`, the sorted list as `json.dumps` writes it |
| `lora` | `A` `[768, rank]`, `B` `[rank, 768]` | `rank`, `alpha`; across sites, `with.component`, `with.layer` |
| `affine` | `W` `[768, 768]`, `b` `[768]`; over layers, `[layers, 768, 768]` and `[layers, 768]` | over layers: `over`, and `layer` lists them (`[4, 5]`) |
| `constant` | `vector` `[768]` | nothing more |

Every header records `format` (`nnspec/1`), `kind`, `model.key`, `model.revision`, `model.dtype`,
`component`, `layer`, `width` and `document`, each as a string. To read one:

```python
from safetensors import safe_open

with safe_open("out/das/fit.rot.safetensors", "pt") as f:
    print(dict(sorted(f.metadata().items())), {k: list(f.get_tensor(k).shape) for k in f.keys()})
```

```text
{'component': 'layer_output', 'document': 'das', 'format': 'nnspec/1', 'k': '4', 'kind': 'subspace', 'layer': '5', 'model.dtype': 'float32', 'model.key': 'openai-community/gpt2', 'model.revision': '607a30d783dfa663caf39e06633721c8d4cfcd7e', 'parametrization': 'cayley', 'width': '768'} {'basis': [768, 4]}
```

A swept fit writes one file whose keys carry each point's coordinate (`epochs=8/W`). No featurizer loads it:
run the document unswept at the point you keep. A file with no header is a published one, and only an `sae`
reads it ([SAEs](featurizers-sae.md#splice-in-a-published-dictionary)).

## What a load is checked against

A field the header records may be left out of the loading document, which takes the header's value: a
subspace's `k`, a lora's `rank` and `alpha`. A field the document does give must match. Refused when the
document compiles, each naming the header key or the file:

- a use at another component or layer than the header's (`header 'layer' is '5'; this document uses 'rot' at
  layer_output layer 6`);
- a field given another value, or another model `key`, `revision` or `dtype`;
- a swept fit's file, and a file that is not there;
- the loaded featurizer in a fit's `train`: a bundle fixes it.

The site's width is checked once the model loads, before any forward.

## A hand-made bundle

A featurizer trained outside nnspec loads if its file has the tensors and the header an nnspec bundle has;
without a header it is refused. A probe's scores are `h @ W + b`, each class at its index in the sorted list
of the `classes` column's values. Every header value is a string:

```python
import json
from safetensors.torch import save_file

save_file({"W": W, "b": b}, "artifacts/hand_probe.safetensors", metadata={     # W [768, 4], b [4]
    "format": "nnspec/1", "kind": "linear", "classes": json.dumps(["France", "Germany", "Italy", "Japan"]),
    "model.key": "openai-community/gpt2", "model.revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    "model.dtype": "float32", "component": "layer_output", "layer": "8", "width": "768", "document": "hand_made"})
```

Loaded as `{"kind": "linear", "classes": "test.country", "file_path": "artifacts/hand_probe.safetensors"}`
with the weights `probe_train.json` trained, it scores the held-out cities of
[`probe.json`](featurizers.md#a-fit-start-to-finish) at that run's 1.0. `classes` is the list exactly as
`json.dumps` writes it, its `", "` separator included: written without the spaces it is refused against the
document's `'["France", "Germany", "Italy", "Japan"]'`. Any kind loads this way with its row of the table
above: a probe's span of several classes, for one, as a `subspace` whose `basis` holds orthonormal columns,
with `k` and `"parametrization": "cayley"` in the header.

## Cached harvests

A harvest is a read over every row of a dataset, saved. A later document that only fits or reduces it needs
the values, not the forwards: a `load` step brings a saved value in with its record (its layout, its
positions, the read's site), so every step after it treats it as the value it was. Saved as
`documents/harvest.json`:

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
$ nnspec run documents/harvest.json
harvest on nnterp (batch size 8, device cpu, tp 1) -> out/harvest
  harvest.acts.safetensors  harvest  [7,6,768]
```

The loading document keeps the same `model` and the same dataset under the same name, and loads the read in
place of the forward. `documents/sae_cached.json` has `harvest.json`'s `model` and `data`, and:

```json
{
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

- **`from` is the harvesting run's output directory, `value` a name in its `saves`.** The step's value is
  named by the step, `harvest`, as an op step's is.
- **No forward runs, and no weights load**: `nnspec show out/sae_cached` lists 0 forwards for both steps and
  `served: float32 on meta`.
- **The fit equals one on the harvest made in-document.** With `harvest` a forward of `days.input` at the
  same site and `pos`, the trained dictionary is equal to the bit, at one batch size.
- **An `encode` of a loaded read uses the featurizer at the read's site**, so this document declares no
  `sites`.
- **A loaded read feeds any step a read does.** A `mean` step over it, swapped into a forward as a mean
  ablation, gives the scores of the same document with the harvest's forward in place of the load, to the bit.
- **A harvest swept over `layers` loads at every point**, keyed as the run keyed it, with no sweep written in
  the loading document.

A load is refused when the document compiles if it does not match its source: other rows or another order
(`… this document's 'days' is not the rows out/harvest read, in their order: 6 rows where it read 7`), the
dataset under another name, another `model` block, or a name the run did not save. A bundle does not load as
a value, nor does a value in a featurizer's coordinates, a generation's value, or anything inside a fit's
body.

**A harvest in shards.** A harvest too large for one run is one shuffled order cut with `skip` and `take`,
one run per shard: `harvest.json` with `"shuffle": true, "skip": 0, "take": 4` in its `days` block saves
`[4,6,768]`, and with `"skip": 4, "take": 3` the other `[3,6,768]`. Joined in order, the two equal the harvest
of all seven shuffled rows to the bit. A load of a shard needs a dataset block giving the same rows
(`"shuffle": true, "skip": 4`), and nothing in a document joins shards: join them in Python, or fit on each.
