# Featurizers and fit

A featurizer is a space or a map declared once under the root key `featurizers`, by name, and reached from an
entry's ops with `encode`, `decode` and `apply`. It is drawn, trained by a `fit` step inside the run, or loaded
from a bundle file. The recipes (DAS, DBM, head masks, SAEs, probes, LoRA, cached harvests) are the
`clab-trained-featurizers` skill's, the tuned lens the `clab-lenses-and-attention` skill's, and the optimal
constant the `clab-ablation` skill's. This page is the fields and the mechanics.

## Kinds

| kind | pick it to | fields | ops | drawn as |
|---|---|---|---|---|
| `subspace` | swap or edit a rank-`k` share of a value | `k`, `parametrization`, `seed` | encode, decode, apply | a uniformly random rank-`k` subspace |
| `gate` | learn which units, or which heads of every layer, carry a behaviour | `over`, `temperature` | encode, decode, apply; with `over`, apply alone | every unit on |
| `pca` | edit along a harvest's top principal directions | `k` | encode, decode, apply | never: only the bundle a `pca` op wrote |
| `sae` | read or edit a sparse dictionary's latents | `latents`, `activation`, `k`, `centered`, `site`, `seed` | encode, decode, apply | a dictionary of `latents` from `seed` |
| `linear` | ask whether a label column is linearly decodable | `classes` | encode | `W` and `b` zero |
| `lora` | learn a low-rank update at one site | `rank`, `alpha`, `seed` | apply | the identity (`B` zero) |
| `affine` | learn a full map of a site onto itself, one per layer with `over: ["layers"]` | `over` | apply | the identity |
| `constant` | replace a value by one vector, set or learned | `init` | apply | `init`: a number, or a one-vector value such as a `mean` |

Every kind takes `file_path` to load a bundle. A field a kind does not take is refused with its list:
`featurizers.probe: 'linear' takes no 'seed'; its arguments: classes, file_path`. `clab describe featurizers`
prints each kind's fields and the math of its ops.

## In an entry

- **`[encode f]` reads coordinates.** The value's last axis becomes `features` (a probe's: `classes`); nothing is written.
- **`[encode f, <edits>, decode f]` edits coordinates and writes the change back** as a delta, so what the space does not hold (a subspace's complement, an SAE's error term) passes through untouched.
- **`[encode f, decode f]` with nothing between returns the value exactly.**
- **`[apply f]` writes the featurizer's own version of the value**: a projection, a masked value, a reconstruction without its error term, the adapted value, `h + h W + b`, or the constant's vector.
- **Each `decode` closes the most recent open `encode`**; a write with an `encode` still open is refused, and pairs nest.
- **An operand between a pair must be in the same coordinates**: read through the same `encode`, with the same `select`, or a number.
- **`select` on the `encode` keeps some coordinates**: `{"op": "encode", "featurizer": "sae", "select": {"features": {"sweep": [0, 1, 2]}}}`; every other coordinate changes by exactly zero.
- **A kind refuses another kind's ops by name**: `'fit.probe' is a 'linear' featurizer, which has no decode: it serves encode; it has no way back into the value`.
- **As a step, `encode` and `apply` run over a read** (`{"kind": "op", "op": "encode", "featurizer": "fit.probe", "of": "harvest.h"}`); `decode` and an op's value are refused there.
- **A featurizer's width is the last axis of every site it is used at**, and one featurizer at two widths is refused.
- **`encode` runs in the value's dtype**; `apply` and the edits between a pair compute in float32 and round once.

## A sweep over featurizers

An `encode`'s, `decode`'s or `apply`'s `featurizer` may be a sweep over featurizer names, declared or a fit's
trained ones: `{"op": "encode", "featurizer": {"sweep": ["sae_a", "sae_b"], "label": "sae"}}`. Each point is
that featurizer's use, as if the document named it alone. The `clab-trained-featurizers` skill compares a
trained rotation with a PCA and a draw this way, and two dictionaries.

- **A `decode` carries its `encode`'s sweep**, the same label and names. A `decode` of one name is refused at the point: `… at space="pc": decode of 'rot', and the open encode is of 'pc' (ops[0])`.
- **The coordinates are the names**: a swept save's `at` in `clab table` is `{"sae":"sae_a"}`.
- **A read that only encodes shares one forward**; a write runs one per point.
- **After a fit, `{"sweep": ["rot", "fit.rot"]}` scores the draw beside the trained featurizer.** In the body the two names are one featurizer, refused at compile: `label 'r' sweeps 'fit.rot' twice ('rot' and 'fit.rot' in the body of fit 'fit' are one featurizer, …)`.
- **A sweep in the body over the featurizers the fit trains is refused once the model loads**, since each point would train every one and use one: `fit at r="fit.rot": 'rot2' is trained but used at no site at this point, …`.
- **`train` takes no sweep**: write one fit per featurizer it trains.
- **Sweep a field where one declaration serves** (a subspace's `seed`); name several featurizers where they differ in kind, `k` or bundle.

## A published sparse autoencoder

An `sae` whose `file_path` names a published safetensors file (SAELens and Gemma Scope's layout) states the three
fields the file cannot carry, `activation`, `centered` and the `site` it was trained at:

```json
{"dictionary": {"kind": "sae", "file_path": "artifacts/sae_layer6.safetensors",
                "activation": "jumprelu", "centered": false, "site": {"component": "layer_output", "layers": 6}}}
```

A wrong `centered`, `activation` or layer runs unrefused; what each field means, what is refused and the mean-L0
check are the `clab-trained-featurizers` skill's ([published dictionaries](../../clab-trained-featurizers/references/sae.md#splice-in-a-published-dictionary)).

## The fit step

| field | holds |
|---|---|
| `train` | the declared featurizers it trains |
| `objective` | `{term: weight}`, minimized; a term is a body op, averaged over every axis, or a trained featurizer's scalar (`<featurizer>.mask`, `.reconstruction`, `.l1`) |
| `epochs`, `batch_size`, `seed` | passes over the training rows, minibatch rows, the draw and shuffle seed |
| `optimizer` | `{"name": "adamw" or "adam" or "sgd", "lr": ..., "weight_decay": ...}`; `weight_decay` defaults to 0 |
| `early_stop` | `{"metric": <body op>, "mode": "max" or "min", "patience": N}`; the best epoch's tensors are restored |
| `eval` | `{<name>: {<dataset>: <dataset read in its place>}}`; `{}` is the data it trained on |
| `steps` | the body: its own steps, run on each minibatch with gradients on |

- **Pick `adamw`**: it and `adam` move each parameter by about `lr` a step whatever the gradient's scale, and at `weight_decay` 0 they are one optimizer. `sgd` steps by `lr` times the gradient, so its `lr` depends on the objective's scale.
- **`weight_decay` pulls the trained tensors toward zero** each step; leave it at 0 unless held-out scores fall while training scores rise.
- **`early_stop` ends a fit once a body op stops improving**, which saves the epochs after a plateau. It judges the training rows, not held-out ones, so it is no guard against overfitting; to compare fits, sweep `epochs`.

`documents/fit_probe.json` harvests the last token at block 6 once, trains a probe on the harvest beside an
`epochs: 0` point, and scores held-out rows through the trained probe after the fit:

```json
{
  "description": "The fit step on a linear probe: harvest block 6's last-token output once, train a probe of the topic column on it beside an epochs-0 point, and score held-out rows through the trained probe.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {
    "train": {"rows": [
      {"input": "The Eiffel Tower is in the city of", "topic": "place"},
      {"input": "The Golden Gate Bridge is in the city of", "topic": "place"},
      {"input": "The Kremlin is in the city of", "topic": "place"},
      {"input": "When Mary and John went to the store, John gave a drink to", "topic": "person"},
      {"input": "When Tom and Anna went to the park, Anna gave a ball to", "topic": "person"},
      {"input": "When Sarah and James went to the office, James gave a book to", "topic": "person"}
    ]},
    "test": {"rows": [
      {"input": "The Brandenburg Gate is in the city of", "topic": "place"},
      {"input": "The Space Needle is in the city of", "topic": "place"},
      {"input": "When Paul and Lisa went to the school, Lisa gave a pen to", "topic": "person"},
      {"input": "When Kate and Mark went to the beach, Mark gave a hat to", "topic": "person"}
    ]}
  },
  "featurizers": {"probe": {"kind": "linear", "classes": "train.topic"}},
  "sites": {"resid": {"component": "layer_output", "layers": 6}},
  "steps": {
    "harvest": {"kind": "forward", "input": "train.input", "interventions": {"h": {"site": "resid", "pos": -1}}},
    "fit": {
      "kind": "fit",
      "train": ["probe"],
      "objective": {"ce": 1.0},
      "epochs": {"sweep": [0, 8]},
      "batch_size": 6,
      "seed": 0,
      "optimizer": {"name": "adamw", "lr": 0.0001},
      "eval": {"train": {}},
      "steps": {
        "scores": {"kind": "op", "op": "encode", "featurizer": "fit.probe", "of": "harvest.h"},
        "ce": {"kind": "op", "op": "cross_entropy", "of": "scores", "target": "train.topic"},
        "hit": {"kind": "op", "op": "match", "of": "scores", "expected": "train.topic"}
      }
    },
    "held": {"kind": "forward", "input": "test.input", "interventions": {
      "scores": {"site": "resid", "pos": -1, "ops": [{"op": "encode", "featurizer": "fit.probe"}]}}},
    "held_hit": {"kind": "op", "op": "match", "of": "held.scores", "expected": "test.topic"},
    "held_acc": {"kind": "op", "op": "mean", "of": "held_hit"}
  },
  "saves": ["fit.history.ce", "fit.train.hit", "held_acc", "fit.probe"]
}
```

```text
$ clab run documents/fit_probe.json
fit_probe on nnterp (batch size 8, device cpu, tp 1) -> out/fit_probe
  fit.history.ce.safetensors  fit            2 points, each [0] or [8]
  fit.train.hit.safetensors   fit.train.hit  2 points, each [6,1]
  held_acc.safetensors        held_acc       2 points, each []
  fit.probe.safetensors       fit            2 points, each {"W":[768,2],"b":[2]}
  out/fit_probe/manifest.json
$ clab table out/fit_probe held_acc
at            value
{"epochs":0}  0.5
{"epochs":8}  1.0
$ clab table out/fit_probe fit.history.ce
at            epochs  value
{"epochs":8}  0       0.6077650189399719
{"epochs":8}  1       0.5312594771385193
...
{"epochs":8}  7       0.22974468767642975
```

What the run does:

- **Steps before the fit run once**: `harvest` is made before training, and the body takes its rows by each minibatch's row ids.
- **The fit draws what it trains** from its `seed`, unless the featurizer has its own, and makes the optimizer.
- **Each epoch shuffles the training rows** by `seed` and the epoch, runs the body on each minibatch with gradients on, and steps once per minibatch.
- **After each epoch the body runs once more over every training row**, scored, without gradients: that pass is the epoch's history entry and what `early_stop` judges.
- **Then each `eval` runs the body as ordinary steps**, through the trained featurizer; `fit.train.hit` is the `train` eval's `hit`.
- **Steps after the fit reach the trained probe as `fit.probe`**: `held` scores the test rows through it.

### The body and its costs

- **A body's steps run in the order their keys are written**, as a document's do: generated JSON whose dict order shifts is refused with `names no value produced before this step`.
- **A value made before the fit with a row per example of a body dataset is a harvest.** Read what does not depend on the training before the fit, once; this body makes no model call at all.
- **A harvest of another dataset than the body's is refused**, since a minibatch's ids would pair its rows with others.
- **An eval cannot read a harvest's dataset as another**: `{"held_out": {"train": "test"}}` here is refused with `… evaluate on 'test' in steps after the fit, which reach what it trained as 'fit.probe'`, as `held` does.
- **An objective none of whose terms depends on the trained featurizer is refused**, naming it; a term that is no body op is refused listing the body ops.
- **A reduction over rows inside the body is the minibatch's statistic**; a statistic of the dataset is a step before the fit.
- **Count the optimizer steps**: `epochs × ⌈rows / batch_size⌉`, eight here. Adam moves each parameter by about `lr` a step, so `lr × steps` bounds how far a featurizer gets.
- **The body runs at the fit's own `batch_size`**, whatever `--batch-size` says; a harvest before the fit runs at `--batch-size`.
- **The model's weights never change**, and a forward after the fit gives the logits a forward before it gave.
- **A fit holds the minibatch's backward, the scored pass over every training row, and every eval's values** on the model's device; only the first shrinks with `batch_size`.

## Members

| member | is |
|---|---|
| `<fit>.<featurizer>` | the trained featurizer; saved, a bundle (`fit.probe`) |
| `<fit>.<eval>.<op>` | a body op's value in that eval (`fit.train.hit`) |
| `<fit>.<eval>.<step>.<entry>` | a body entry's value in that eval |
| `<fit>.history.<op>` | each body op's mean after each epoch, `[epochs run]` (`fit.history.ce`) |
| `<fit>.history.<featurizer>.<scalar>` | a gate's or an sae's scalar after each epoch |
| `<fit>.<featurizer>.<tensor>` | one of the trained featurizer's tensors, a value, as the fit left it (`fit.probe.W`) |
| `<featurizer>.<tensor>` | one of the declared featurizer's tensors, as drawn or loaded (`dictionary.W_dec`) |

- **After the fit, the bare name is the declared featurizer as drawn, never trained.** With `"featurizer": "probe"` in `held`, the same run gives `held_acc` 0.5, one value for both points: the untrained probe.
- **Inside the body both names are the one being trained.**
- **Every body op gets a history**, in the objective or not, so put an aggregate you want to watch in the body.
- **A history has one entry per epoch run**: fewer under `early_stop`, none at `epochs: 0`.

A featurizer's tensors are its bundle's, by name, each a value laid out on axis names: a probe's `W`
`[hidden_size, classes]` and `b` `[classes]`, an sae's `W_dec` `[features, hidden_size]`, a subspace's `basis`
`[hidden_size, features]`. The `clab-trained-featurizers` skill's [bundle table](../../clab-trained-featurizers/references/harvest-and-bundles.md#what-a-run-writes)
lists every kind's.

- **`hidden_size` is the last axis of the first site a step before the name uses the featurizer at.** A name before any such step is refused: `'probe.W' is a tensor of 'probe', laid out on the last axis of the site it is used at, and no step before this one encodes or applies 'probe' at a site; …`.
- **Inside the fit that trains it, a tensor is no value yet**: `'fit.probe.W' is a tensor of 'fit.probe', which this fit trains: it is a value once the fit ends, …`.
- **The bare name's tensors are the draw's**: `probe.W` after the fit is all zeros.
- **A name the kind does not hold is refused, listing those it does**: `'fit.probe.U' names no tensor of 'fit.probe', a linear, whose tensors are W, b`. So is an `affine`'s `W`, two axes under one name: save the affine whole, as its bundle.
- **`select` takes one class, latent or column** of a tensor ([ops](ops.md#transform)); an sae's `W_dec` row 376, summed over `features`, is a `[hidden_size]` vector of norm 1.

## `epochs: 0` is the null

`epochs: 0` trains nothing: the featurizers are scored as drawn, every eval runs, and every history is empty.
Sweep it beside the trained point, `"epochs": {"sweep": [0, N]}`, and read the trained one only where it beats
the draw on the same rows.

| kind | `epochs: 0` scores |
|---|---|
| `subspace` | a uniformly random subspace of the same rank |
| `gate` | every unit on: `apply` is the value, and `apply` with `with` the clean run |
| `linear` | every class scored 0, ties to the lower class: accuracy is the share of rows whose label sorts first (0.5 above, where `person` sorts first) |
| `lora`, `affine` | the identity, exactly |
| `constant` | its `init` |
| `sae` | the dictionary drawn from `latents` and the seed |

A swept fit's members are keyed by coordinate (`{"epochs":0}`), and its bundle file holds one bundle per point,
which no document can load.

## Bundles

A bundle is a safetensors file of one featurizer's tensors with a header saying what made them. A run writes one
for every saved `<fit>.<featurizer>` and every saved `pca` op step. `fit_probe.json` unswept at
`"epochs": 8`, with `"saves": ["held_acc", "fit.probe"]`, is `documents/probe_train.json`; run it into the
directory the loading document names:

```text
$ clab run documents/probe_train.json --out artifacts/probe
probe_train on nnterp (batch size 8, device cpu, tp 1) -> artifacts/probe
  held_acc.safetensors   held_acc  []
  fit.probe.safetensors  fit       {"W":[768,2],"b":[2]}
  artifacts/probe/manifest.json
```

The loading document declares the featurizer with `file_path` and uses it by its declared name. It needs
`artifacts/probe/fit.probe.safetensors` beside `documents/`:

<!-- test: skip -->
```json
{
  "description": "Load the probe probe_train.json trained and saved under artifacts/probe, and score the held-out rows through it; the classes column, the model and the site are checked against the bundle's header before anything runs.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"test": {"rows": [
    {"input": "The Brandenburg Gate is in the city of", "topic": "place"},
    {"input": "The Space Needle is in the city of", "topic": "place"},
    {"input": "When Paul and Lisa went to the school, Lisa gave a pen to", "topic": "person"},
    {"input": "When Kate and Mark went to the beach, Mark gave a hat to", "topic": "person"}
  ]}},
  "featurizers": {"probe": {"kind": "linear", "classes": "test.topic", "file_path": "artifacts/probe/fit.probe.safetensors"}},
  "sites": {"resid": {"component": "layer_output", "layers": 6}},
  "steps": {
    "held": {"kind": "forward", "input": "test.input", "interventions": {
      "scores": {"site": "resid", "pos": -1, "ops": [{"op": "encode", "featurizer": "probe"}]}}},
    "held_hit": {"kind": "op", "op": "match", "of": "held.scores", "expected": "test.topic"},
    "held_acc": {"kind": "op", "op": "mean", "of": "held_hit"}
  },
  "saves": ["held_acc"]
}
```

```text
project/
  documents/probe_train.json
  documents/probe_apply.json
  artifacts/probe/fit.probe.safetensors     W [768, 2], b [2], and the header
  artifacts/probe/manifest.json, held_acc.safetensors, tokens.json
```

Its `held_acc` is 1.0, the trained run's. The header, read with `safetensors.safe_open(path, "pt").metadata()`:

```text
{'classes': '["person", "place"]', 'component': 'layer_output', 'document': 'probe_train', 'format': 'clab/1', 'kind': 'linear', 'layer': '6', 'model.dtype': 'float32', 'model.key': 'openai-community/gpt2', 'model.revision': '607a30d783dfa663caf39e06633721c8d4cfcd7e', 'width': '768'}
```

- **`file_path` resolves against the corpus root** ([where output goes](running-and-engines.md#where-output-goes)): `project/` in the layout above, which needs no `data/`.
- **The header is checked before anything runs**: the kind, the model's `key`, `revision` and `dtype`, the component and layer at every use, and the width once the model loads. A use at block 5 is refused: `featurizers.probe.file_path: artifacts/probe/fit.probe.safetensors: header 'layer' is '6'; this document uses 'probe' at layer_output layer 5 (held.scores)`.
- **A field the header records may be left out** of the loading document (a subspace's `k`, a lora's `rank`), and one given must match; a probe's `classes` column is required and checked: another class list is refused, quoting both.
- **A missing file is refused at compile**: `artifacts/nope.safetensors does not exist under the corpus root; run the document that writes it first`.
- **A swept run's bundle is refused**: `out/fit_probe/fit.probe.safetensors holds one bundle per point of a sweep (epochs=0, epochs=8); a featurizer loads one bundle, so run the document that writes it unswept`.
- **A loaded featurizer is fixed**: a fit refuses to train one with a `file_path`.
- **Give each training run its own `--out`**: a second run into one directory replaces its manifest and leaves the first run unreadable.
- **`clab show` and `clab table` read a loaded bundle again** from its `file_path`, so run them from the corpus root with the bundle in place.
- **A `pca` op step's value is a bundle**, `basis` `[width, k]` and `mean`, which a `{"kind": "pca", "k": 4, "file_path": ...}` featurizer loads; a `pca` needs more vectors than its `k`.
- **A value that is not a bundle loads with a `load` step** (`{"kind": "load", "from": "out/x", "value": "harvest.h"}`), which the `clab-trained-featurizers` skill uses for cached harvests.
