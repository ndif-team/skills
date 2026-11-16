# Featurizers and fits

A featurizer is a space or a map declared once under the root key `featurizers` and reached from an entry's
ops with `encode`, `decode` and `apply`: a probe, a sparse dictionary, a subspace, a mask, an adapter. It is
drawn, trained by a `fit` step inside the run through the frozen model, or loaded from a bundle file an
earlier run saved. This page is what all of them share.

## What breaks

Each of these runs without a refusal.

- **After a fit, the bare name is the untrained draw.** Steps after the fit name the trained featurizer
  `<fit>.<name>`. The document below scores 1.0 through `fit.probe` and 0.25 through `probe`.
- **A fit without its `epochs: 0` point proves nothing.** Sweep `"epochs": {"sweep": [0, N]}` and read the
  trained featurizer only where it beats its own draw on the same rows.
- **`lr × steps` bounds how far a featurizer gets.** A fit takes `epochs × ⌈rows / batch_size⌉` optimizer
  steps, and Adam moves each parameter by about `lr` a step: too few steps leave the draw, and too large an
  `lr` scrambles it. Sweep `lr` before reading a flat curve
  ([an example](featurizers-subspaces-and-masks.md#what-breaks)).
- **`apply` replaces; the `encode`, edit, `decode` bracket edits.** A subspace's `apply` is its projection and
  an SAE's drops the error term. An interchange or a latent ablation is the bracket, which keeps everything
  the space does not hold.
- **A featurizer at a site that picks several layers is one featurizer, applied at each.** Sweep the site's
  `layers` for one per layer; only a `gate` or an `affine` with `layers` in its `over` holds its own per layer.
- **A swept fit saves one bundle per point in one file, which no featurizer loads.** Train unswept to save a
  bundle another document loads.

## Kinds

| kind | pick it to | drawn as | ops | recipe |
|---|---|---|---|---|
| `linear` | ask whether a label column is linearly decodable: a probe | `W` and `b` zero | `encode` | below; [per layer, many at once, its direction](featurizers-probes-and-lora.md) |
| `sae` | read or edit a sparse dictionary's latents | a dictionary of `latents` from `seed` | `encode`, `decode`, `apply` | [SAEs](featurizers-sae.md) |
| `subspace` | find which directions of a value carry a variable: DAS | a uniformly random rank-`k` subspace from `seed` | `encode`, `decode`, `apply` | [DAS](featurizers-subspaces-and-masks.md#das-how-many-dimensions-carry-the-day) |
| `pca` | edit along a harvest's top principal directions: a subspace's control | never: it is the bundle a `pca` op step saved | `encode`, `decode`, `apply` | [controls](featurizers-subspaces-and-masks.md#controls-of-matched-rank) |
| `gate` | learn which units, or with `over` which heads of every layer, carry a behaviour | every unit on | `encode`, `decode`, `apply`; with `over`, `apply` alone | [DBM](featurizers-subspaces-and-masks.md#dbm-which-units-carry-the-day), [head masking](featurizers-subspaces-and-masks.md#head-masking) |
| `lora` | learn a low-rank update at one site | the identity (`B` zero) | `apply` | [LoRA](featurizers-probes-and-lora.md#lora-train-an-adapter) |
| `affine` | learn a full map of a site onto itself, one per layer with `"over": ["layers"]` | the identity | `apply` | the tuned lens, in [lens](lens.md) |
| `constant` | put one vector in a value's place, set or learned | `init`: a number, or a one-vector value such as a `mean` | `apply` | the optimal constant, in [ablation](ablation.md) |

Every kind takes `file_path` to load a bundle. `nnspec describe featurizers` prints each kind's fields and the
math of its ops.

## In an entry

- **`[encode f]` reads coordinates.** The value's last axis becomes `features` (a probe's: `classes`), and
  nothing is written.
- **`[encode f, <edits>, decode f]` edits coordinates and writes the change back** as a delta, so what the
  space does not hold (a subspace's complement, an SAE's error term) passes through untouched. With nothing
  between the pair the value comes back exactly; pairs nest.
- **`[apply f]` writes the featurizer's own version of the value**: a projection, a masked value, a
  reconstruction without its error term, the adapted value, `h + h W + b`, or the constant's vector.
- **`select` on the `encode` keeps some coordinates**, `{"op": "encode", "featurizer": "sae", "select":
  {"features": [376, 497]}}`, and every other coordinate changes by exactly zero. It takes ids written in the
  document, never a value a step computed.
- **An operand between a pair is in the same coordinates**: a number, or a value read through the same
  `encode` with the same `select`.
- **As an op step, `encode` and `apply` run over a read**, at the read's site: `{"kind": "op", "op":
  "encode", "featurizer": "fit.probe", "of": "harvest.h"}`. A `decode` is an entry's alone.
- **A featurizer has one width**, the last axis of every site it is used at. `encode` runs in the model's
  dtype; `apply` and the edits between a pair compute in float32 and round once.

## A fit, start to finish

Does GPT-2's block 8 hold a city's country at the city's last token, for cities the probe never saw? Saved as
`documents/probe.json`: harvest once, train a probe of the `country` column on the harvest beside an
`epochs: 0` point, and score held-out cities through the trained probe.

```json
{
  "description": "A linear probe for the country of a city on GPT-2: harvest block 8's output at the last token (the city's last token) once, train the probe on the harvest beside an epochs-0 point, and score cities it never saw through the trained probe.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {
    "train": {"rows": [
      {"input": "I live in Paris", "country": "France"}, {"input": "I live in Rome", "country": "Italy"},
      {"input": "I live in Tokyo", "country": "Japan"}, {"input": "I live in Berlin", "country": "Germany"},
      {"input": "I live in Lyon", "country": "France"}, {"input": "I live in Milan", "country": "Italy"},
      {"input": "I live in Osaka", "country": "Japan"}, {"input": "I live in Munich", "country": "Germany"},
      {"input": "I live in Marseille", "country": "France"}, {"input": "I live in Naples", "country": "Italy"},
      {"input": "I live in Kyoto", "country": "Japan"}, {"input": "I live in Hamburg", "country": "Germany"},
      {"input": "I live in Toulouse", "country": "France"}, {"input": "I live in Turin", "country": "Italy"},
      {"input": "I live in Yokohama", "country": "Japan"}, {"input": "I live in Frankfurt", "country": "Germany"},
      {"input": "I live in Nice", "country": "France"}, {"input": "I live in Venice", "country": "Italy"},
      {"input": "I live in Nagoya", "country": "Japan"}, {"input": "I live in Cologne", "country": "Germany"},
      {"input": "I live in Bordeaux", "country": "France"}, {"input": "I live in Florence", "country": "Italy"},
      {"input": "I live in Kobe", "country": "Japan"}, {"input": "I live in Stuttgart", "country": "Germany"}
    ]},
    "test": {"rows": [
      {"input": "I live in Lille", "country": "France"}, {"input": "I live in Bologna", "country": "Italy"},
      {"input": "I live in Sapporo", "country": "Japan"}, {"input": "I live in Dresden", "country": "Germany"},
      {"input": "I live in Nantes", "country": "France"}, {"input": "I live in Genoa", "country": "Italy"},
      {"input": "I live in Hiroshima", "country": "Japan"}, {"input": "I live in Leipzig", "country": "Germany"},
      {"input": "I live in Strasbourg", "country": "France"}, {"input": "I live in Palermo", "country": "Italy"},
      {"input": "I live in Fukuoka", "country": "Japan"}, {"input": "I live in Hanover", "country": "Germany"}
    ]}
  },
  "featurizers": {"probe": {"kind": "linear", "classes": "train.country"}},
  "sites": {"resid": {"component": "layer_output", "layers": 8}},
  "steps": {
    "harvest": {"kind": "forward", "input": "train.input", "interventions": {"h": {"site": "resid", "pos": -1}}},
    "fit": {
      "kind": "fit", "train": ["probe"], "objective": {"ce": 1.0},
      "epochs": {"sweep": [0, 30]}, "batch_size": 8, "seed": 0, "optimizer": {"name": "adamw", "lr": 0.01},
      "eval": {"train": {}},
      "steps": {
        "scores": {"kind": "op", "op": "encode", "featurizer": "fit.probe", "of": "harvest.h"},
        "ce": {"kind": "op", "op": "cross_entropy", "of": "scores", "target": "train.country"},
        "hit": {"kind": "op", "op": "match", "of": "scores", "expected": "train.country"},
        "acc": {"kind": "op", "op": "mean", "of": "hit"}
      }
    },
    "held": {"kind": "forward", "input": "test.input", "interventions": {
      "scores": {"site": "resid", "pos": -1, "ops": [{"op": "encode", "featurizer": "fit.probe"}]}}},
    "held_hit": {"kind": "op", "op": "match", "of": "held.scores", "expected": "test.country"},
    "held_acc": {"kind": "op", "op": "mean", "of": "held_hit"},
    "held_ce": {"kind": "op", "op": "cross_entropy", "of": "held.scores", "target": "test.country"},
    "held_mce": {"kind": "op", "op": "mean", "of": "held_ce"}
  },
  "saves": ["fit.history.ce", "fit.train.acc", "held_acc", "held_mce", "fit.probe"]
}
```

```text
$ nnspec run documents/probe.json
probe on nnterp (batch size 8, device cpu, tp 1) -> out/probe
  fit.history.ce.safetensors  fit            2 points, each [0] or [30]
  fit.train.acc.safetensors   fit.train.acc  2 points, each []
  held_acc.safetensors        held_acc       2 points, each []
  held_mce.safetensors        held_mce       2 points, each []
  fit.probe.safetensors       fit            2 points, each {"W":[768,4],"b":[4]}
$ nnspec table out/probe held_acc held_mce fit.train.acc
at             held_acc  held_mce               fit.train.acc
{"epochs":0}   0.25      1.3862943649291992     0.25
{"epochs":30}  1.0       0.0029896905180066824  1.0
```

- **Steps before the fit run once.** `harvest` has a row per example of the body's dataset, so each run of
  the body takes its minibatch's rows of it by id, and this fit makes no model call. Read whatever does not
  depend on the training before the fit.
- **The body is the fit's own `steps`**, run on each minibatch with gradients on; the optimizer steps once
  per minibatch on the `objective`, 90 steps here. After each epoch the body runs once more over every
  training row, scored without gradients: that pass is the epoch's entry of `fit.history.<op>`.
- **Each `eval` then runs the body as ordinary steps** through the trained featurizer: `fit.train.acc`.
- **Steps after the fit reach the trained probe as `fit.probe`.** `held` scores cities the probe never saw;
  a metric scores a probe's `classes` against a label column as it scores logits against tokens.
- **The `epochs: 0` point is the floor**: an untrained probe scores every class 0 and a tie goes to the class
  that sorts first, `France`, 3 of the 12 held-out rows. The trained probe scores 1.0 on its own training
  rows, so only the held-out column says anything.

## The fit step

| field | holds |
|---|---|
| `train` | the declared featurizers it trains |
| `objective` | `{term: weight}`, minimized. A term is a body op, averaged over every axis, or a trained featurizer's scalar: a gate's `<name>.mask`, an sae's `<name>.reconstruction` and `<name>.l1` |
| `epochs`, `batch_size`, `seed` | passes over the training rows; rows per minibatch; the seed of the draw and the shuffle |
| `optimizer` | `{"name": "adamw", "lr": ...}`; `adam` and `sgd` are the other names, `weight_decay` defaults to 0 |
| `early_stop` | `{"metric": <body op>, "mode": "max" or "min", "patience": N}`; the best epoch's tensors are restored |
| `eval` | `{<name>: {<dataset>: <dataset read in its place>}}`, the body once more after the fit; `{}` is the data it trained on |
| `steps` | the body |

`epochs`, `batch_size`, `seed`, an objective weight, `lr`, `weight_decay` and `patience` may sweep, which
makes the fit, its members and their users a list of points. `nnspec describe steps` has the rest.

- **The body sees every value made before the fit, and nothing leaves it but the fit's members.** Inside it
  the trained featurizer is `<fit>.<name>`, and its tensors are no values until the fit ends.
- **An eval substitution reads another dataset wherever the body names one**: `{"held_out": {"train":
  "test"}}` runs the body's forwards on `test`, with its columns as targets. Where the body reads a harvest
  of `train`, as above, it is refused: score held-out rows in steps after the fit, through `fit.<name>`.
- **The body and the evals run at the fit's own `batch_size`**, whatever `--batch-size` says; a step before
  the fit runs at `--batch-size`.
- **A reduction over rows inside the body is the minibatch's statistic.** A statistic of the dataset is a
  step before the fit.
- **`adamw` and `adam` step each parameter by about `lr`**, whatever the objective's scale; `sgd` steps by
  `lr` times the gradient, so its `lr` depends on that scale.
- **`early_stop` judges the training rows**, so it saves epochs after a plateau and is no guard against
  overfitting. To compare fits, sweep `epochs`.
- **A fit holds three things in memory**: the minibatch's backward, the scored pass over every training row
  after each epoch, and every eval's values. Only the first shrinks with `batch_size`; for the others read
  fewer positions or rows, and score logits inside the head's entry.

## Members

| member | is |
|---|---|
| `<fit>.<featurizer>` | the trained featurizer; saved, a bundle (`fit.probe`) |
| `<fit>.<eval>.<op>`, `<fit>.<eval>.<step>.<entry>` | a body op's or entry's value in that eval (`fit.train.acc`) |
| `<fit>.history.<op>` | each body op's mean after each epoch, `[epochs run]`: fewer under `early_stop`, none at `epochs: 0` |
| `<fit>.history.<featurizer>.<scalar>` | a gate's or an sae's scalar after each epoch |
| `<fit>.<featurizer>.<tensor>` | one of the trained featurizer's tensors, a value (`fit.probe.W`) |
| `<featurizer>.<tensor>` | one of the declared featurizer's tensors, as drawn or loaded (`dictionary.W_dec`) |

Every body op gets a history, in the objective or not, so put an aggregate you want to watch in the body.

A featurizer's tensors are its bundle's, by name, each a value laid out on axis names: a probe's `W`
`[hidden_size, classes]` and `b` `[classes]`, an sae's `W_dec` `[features, hidden_size]`, a subspace's `basis`
`[hidden_size, features]` ([every kind's](featurizers-bundles-and-harvests.md#what-a-run-writes)). An op step
takes one like any value, after a step that uses the featurizer at a site; the bare name's are the draw's,
so `probe.W` after the fit above is all zeros. The `select` op takes one class, latent or column of a tensor
([a probe's direction](featurizers-probes-and-lora.md#use-a-probe-causally)).

## `epochs: 0` is the null

`epochs: 0` trains nothing: the featurizers are scored as drawn, every eval runs, and every history is empty.

| kind | `epochs: 0` scores |
|---|---|
| `subspace` | a uniformly random subspace of the same rank |
| `gate` | every unit on: `apply` is the value, and `apply` with `with` the clean run |
| `linear` | every class scored 0, ties to the class that sorts first: the share of rows with that label |
| `lora`, `affine` | the identity, exactly |
| `constant` | its `init` |
| `sae` | the dictionary drawn from `latents` and the seed |

A fit draws what it trains from the featurizer's own `seed`, else from the fit's; outside a fit a featurizer
draws from its own `seed`, 0 when unset. So give a drawn kind its own `seed`: then the bare name after the
fit, the `epochs: 0` point and the same declaration in a document with no fit are one control, the same draw
([DAS does](featurizers-subspaces-and-masks.md#controls-of-matched-rank)).

## A sweep over featurizers

An `encode`'s, `decode`'s or `apply`'s `featurizer` may be a sweep over featurizer names, declared or a
fit's trained ones: `{"op": "encode", "featurizer": {"sweep": ["rot", "pc", "draw"], "label": "space"}}`.
Each point is that featurizer's use, keyed by its name (`{"space":"pc"}`), so one document scores a trained
featurizer beside its controls
([an example](featurizers-subspaces-and-masks.md#the-trained-rotation-the-pca-and-the-draw-in-one-document)).
A `decode` carries its `encode`'s sweep, the same label and names. After a fit, `{"sweep": ["rot",
"fit.rot"]}` scores the draw beside the trained featurizer; inside the body the two names are one featurizer,
so that sweep is refused there. Sweep a field where one declaration serves (a subspace's `seed` or `k`), and
name several featurizers where they differ in kind or bundle.

## Bundles

A saved `<fit>.<featurizer>`, and a saved `pca` op step, is a bundle: a safetensors file of the featurizer's
tensors with a header saying what made them. `probe.json` unswept at `"epochs": 30`, saving `held_acc` and
`fit.probe`, is `documents/probe_train.json`:

```text
$ nnspec run documents/probe_train.json
probe_train on nnterp (batch size 8, device cpu, tp 1) -> out/probe_train
  held_acc.safetensors   held_acc  []
  fit.probe.safetensors  fit       {"W":[768,4],"b":[4]}
```

A second document declares the featurizer with `file_path` and uses it by its declared name.
`documents/probe_apply.json` keeps `probe.json`'s `model`, `sites`, `test` rows and the steps `held`,
`held_hit` and `held_acc`, with `probe` for `fit.probe` and this declaration. Its `held_acc` is the training
run's 1.0:

```json
{"featurizers": {"probe": {"kind": "linear", "classes": "test.country", "file_path": "out/probe_train/fit.probe.safetensors"}}}
```

- **`file_path` resolves against the corpus root**, the project folder that holds `documents/`
  ([the rule](running-and-engines.md#where-output-goes)). Run, show and table from it.
- **The header is checked before anything runs**: the kind, the model's `key`, `revision` and `dtype`, the
  component and layer at every use, and the width once the model loads. A mismatch is refused naming the key.
- **A field the header records may be left out** of the loading document (a subspace's `k`, a lora's `rank`),
  and one given must match. A probe's `classes` column is required, and its class list checked.
- **A loaded featurizer is fixed**: a fit refuses to train it.
- **Give each training run its own output directory.** A second run into one replaces the manifest, and the
  first run's values no longer table.

What each kind's file holds, hand-made bundles, and a `load` step for values that are not bundles (a cached
harvest) are in [bundles and cached harvests](featurizers-bundles-and-harvests.md).
