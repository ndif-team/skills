# Probes and LoRA

A `linear` featurizer is a probe, `h W + b`, one score per class: it only encodes. A `lora` is a low-rank
adapter at one site: it only applies. Both train in a fit through the frozen model; the fit step is in
[featurizers](featurizers.md).

## What breaks

- **A probe's classes are the label column's distinct values, sorted**, compared exactly: `yes` and `Yes` are
  two classes. `top_k` and `argmax` over a probe's scores need `"over": "classes"`, and their ids index the
  sorted list.
- **The floor is not 1/classes.** An untrained probe scores every class 0, and `match` breaks the tie to the
  lower class id, so its accuracy is the share of rows whose label sorts first.
- **Hold out what the probe must generalise over.** A probe trained and scored on the same cities reads token
  identity; one scored on unseen cities reads the property.
- **Decodable is not used.** Project the probe's direction out, beside a random direction
  ([below](#use-a-probe-causally)).
- **An adapter stands for a weight change, so it acts at every position**: `{"index": ":"}`. Narrowed to the
  last token it is an activation edit there.
- **In a generation, write the adapter at `"frame": ":"`.** A `pos` with no `frame` is the prefill only, and
  every decode call runs unadapted.
- **Hold out what the adapter must transfer to**: another phrasing, not other rows of the phrasing it trained
  on.

## A probe per layer

[`probe.json`](featurizers.md#a-fit-start-to-finish) trains one probe, of a city's country, on a harvest of
block 8. Sweep the site's `layers` and the harvest, the fit and the held-out steps all get a point per layer,
crossed with the fit's `epochs` sweep: `"resid": {"component": "layer_output", "layers": {"sweep": [0, 2, 4,
8, 11]}}` is ten fits.

```text
$ nnspec table out/probe_layers held_acc held_mce
at                         held_acc            held_mce
{"layers":0,"epochs":0}    0.25                1.3862943649291992
{"layers":0,"epochs":30}   0.5                 2.3142340183258057
{"layers":2,"epochs":30}   0.8333333134651184  0.7212057113647461
{"layers":4,"epochs":30}   0.9166666865348816  0.41435277462005615
{"layers":8,"epochs":30}   1.0                 0.0029896905180066824
{"layers":11,"epochs":30}  0.9166666865348816  0.5249781608581543
```

- **`epochs` 0 is the floor**, 0.25 at every layer.
- **The 12 unseen cities decode at 0.5 from layer 0**, where only the token is there, and at 1.0 from layer 8.
  Every trained point scores 1.0 on its training rows.
- **Rank layers by the mean cross entropy where accuracies tie**: layers 4 and 11 both get 11 of 12, and
  `held_mce` puts 4 ahead. A row's cross entropy that prints `-0.0` is a row scored at probability 1 in
  float32.

**Many probes in one fit.** One forward per minibatch serves any number of probes: one `linear` featurizer,
one entry and one objective term each. A probe's parameters appear in its own term alone, so it trains as it
would alone. Here the body holds the forward, so an `eval` can read `test` wherever the body says `train`:
its inputs, its targets and the classes scored. With `probe.json`'s `model` and `data`, and a `match` and a
`mean` step per probe in the body (`acc4`, `acc8`):

```json
{
  "sites": {"l4": {"component": "layer_output", "layers": 4}, "l8": {"component": "layer_output", "layers": 8}},
  "featurizers": {"p4": {"kind": "linear", "classes": "train.country"}, "p8": {"kind": "linear", "classes": "train.country"}},
  "steps": {"fit": {
    "kind": "fit", "train": ["p4", "p8"], "objective": {"ce4": 1.0, "ce8": 1.0},
    "epochs": 30, "batch_size": 8, "seed": 0, "optimizer": {"name": "adam", "lr": 0.01},
    "eval": {"held_out": {"train": "test"}},
    "steps": {
      "read": {"kind": "forward", "input": "train.input", "interventions": {
        "p4": {"site": "l4", "pos": -1, "ops": [{"op": "encode", "featurizer": "fit.p4"}]},
        "p8": {"site": "l8", "pos": -1, "ops": [{"op": "encode", "featurizer": "fit.p8"}]}}},
      "ce4": {"kind": "op", "op": "cross_entropy", "of": "read.p4", "target": "train.country"},
      "ce8": {"kind": "op", "op": "cross_entropy", "of": "read.p8", "target": "train.country"}
    }}}
}
```

`fit.held_out.acc4` is 0.9166666865348816 and `fit.held_out.acc8` 1.0, the single probes' numbers at those
layers. The class list stays the training column's, and a held-out label outside it is refused. `early_stop`
judges one body op and ends the fit for every probe at once, so leave it out.

## Use a probe causally

A trained probe's tensors are values: `fit.probe.W`, `[hidden_size, classes]`, and `fit.probe.b`. A two-class
probe's direction is `W[:, 1] - W[:, 0]`: two `select`s of `classes`, each summed over it, then a `sub`, and
`project_out` takes the result in the same document. Here `train` holds 12 film reviews and `test` 8, half of
each positive, each a row like

```json
{"input": "I loved this film. Overall, the movie was", "label": "positive", "answer": " good", "other": " bad"}
```

The probe reads GPT-2's layer 10 output at the last token; `stream` is that site and `head` is
`lm_head_output`:

```json
{
  "featurizers": {"probe": {"kind": "linear", "classes": "train.label"}},
  "steps": {
    "harvest": {"kind": "forward", "input": "train.input", "interventions": {"h": {"site": "stream", "pos": -1}}},
    "fit": {"kind": "fit", "train": ["probe"], "objective": {"ce": 1.0},
            "epochs": 50, "batch_size": 12, "seed": 0, "optimizer": {"name": "adam", "lr": 0.01},
            "steps": {
              "scores": {"kind": "op", "op": "encode", "featurizer": "fit.probe", "of": "harvest.h"},
              "ce": {"kind": "op", "op": "cross_entropy", "of": "scores", "target": "train.label"}}},
    "positive": {"kind": "op", "op": "select", "of": "fit.probe.W", "select": {"classes": 1}},
    "positive_w": {"kind": "op", "op": "sum", "of": "positive", "over": ["classes"]},
    "negative": {"kind": "op", "op": "select", "of": "fit.probe.W", "select": {"classes": 0}},
    "negative_w": {"kind": "op", "op": "sum", "of": "negative", "over": ["classes"]},
    "direction": {"kind": "op", "op": "sub", "of": "positive_w", "with": "negative_w"},
    "zero": {"kind": "op", "op": "mul", "of": "direction", "with": 0.0},
    "random": {"kind": "op", "op": "noise", "of": "zero", "scale": 1.0, "seed": 0},
    "clean": {"kind": "forward", "input": "test.input", "interventions": {
      "scores": {"site": "stream", "pos": -1, "ops": [{"op": "encode", "featurizer": "fit.probe"}]},
      "ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "test.answer", "b": "test.other"}]}}},
    "projected": {"kind": "forward", "input": "test.input", "interventions": {
      "out": {"site": "stream", "pos": {"index": ":"},
              "ops": [{"op": "project_out", "direction": {"sweep": ["direction", "random", "zero"], "label": "fill"}}]},
      "ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "test.answer", "b": "test.other"}]}}},
    "held_hit": {"kind": "op", "op": "match", "of": "clean.scores", "expected": "test.label"},
    "held_acc": {"kind": "op", "op": "mean", "of": "held_hit"},
    "clean_margin": {"kind": "op", "op": "mean", "of": "clean.ld"},
    "margin": {"kind": "op", "op": "mean", "of": "projected.ld"}
  },
  "saves": ["held_acc", "clean_margin", "margin"]
}
```

On those reviews the probe reads the held-out sentiment on 8 of 8 rows, and `margin`, the mean logit of each
row's answer less the other, is 0.475 with the direction projected out, 1.019 with a random direction and
1.018 with the zero direction, which is `clean_margin`.

- **The null is the zero direction**: `project_out` leaves every vector as it is along it, so its margin is
  the clean run's to the bit. The control is the random direction, a `noise` draw on zeros.
- **`select` keeps `classes` at size 1, labelled with the class it kept**, and the `sum` over it drops the
  axis exactly. A `sub` of the two selects without the sums is refused, since their labels differ.
- **The classes sort**, `negative` before `positive`, so `direction` points toward positive.
- **Name a tensor after the fit that trains it.** Inside the fit it is no value yet, and the bare `probe.W`
  after it is the draw, all zeros.
- **A probe of more than two classes has a span, not one direction.** Take the class differences the question
  needs, a `select` pair and a `sub` each.

## LoRA: train an adapter

A rank-4 adapter at layer 5's output, at every position, trained to answer the day after tomorrow on one
phrasing and scored on another. Saved as `documents/lora_train.json`:

```json
{
  "description": "Train a rank-4 LoRA at GPT-2's layer 5 output, at every position, to answer the day after tomorrow on one phrasing, and score it on another. B starts at zero, so the untrained adapter is the model: clean_mce is what an epochs: 0 fit's held-out mce equals.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {
    "train": {"rows": [
      {"input": "If today is Monday, tomorrow is", "target": " Wednesday"}, {"input": "If today is Tuesday, tomorrow is", "target": " Thursday"},
      {"input": "If today is Wednesday, tomorrow is", "target": " Friday"}, {"input": "If today is Thursday, tomorrow is", "target": " Saturday"},
      {"input": "If today is Friday, tomorrow is", "target": " Sunday"}, {"input": "If today is Saturday, tomorrow is", "target": " Monday"},
      {"input": "If today is Sunday, tomorrow is", "target": " Tuesday"}
    ]},
    "test": {"rows": [
      {"input": "Today is Monday, so tomorrow will be", "target": " Wednesday"}, {"input": "Today is Tuesday, so tomorrow will be", "target": " Thursday"},
      {"input": "Today is Wednesday, so tomorrow will be", "target": " Friday"}, {"input": "Today is Thursday, so tomorrow will be", "target": " Saturday"},
      {"input": "Today is Friday, so tomorrow will be", "target": " Sunday"}, {"input": "Today is Saturday, so tomorrow will be", "target": " Monday"},
      {"input": "Today is Sunday, so tomorrow will be", "target": " Tuesday"}
    ]}
  },
  "sites": {"target": {"component": "layer_output", "layers": 5}, "head": {"component": "lm_head_output"}},
  "featurizers": {"lora": {"kind": "lora", "rank": 4, "alpha": 8, "seed": 0}},
  "steps": {
    "clean": {"kind": "forward", "input": "test.input", "interventions": {"logits": {"site": "head", "pos": -1}}},
    "clean_ce": {"kind": "op", "op": "cross_entropy", "of": "clean.logits", "target": "test.target"},
    "clean_mce": {"kind": "op", "op": "mean", "of": "clean_ce"},
    "fit": {
      "kind": "fit", "train": ["lora"], "objective": {"ce": 1.0},
      "epochs": 20, "batch_size": 7, "seed": 0, "optimizer": {"name": "adam", "lr": 0.01},
      "eval": {"held_out": {"train": "test"}},
      "steps": {
        "adapted": {"kind": "forward", "input": "train.input", "interventions": {
          "adapt": {"site": "target", "pos": {"index": ":"}, "ops": [{"op": "apply", "featurizer": "fit.lora"}]},
          "logits": {"site": "head", "pos": -1}}},
        "ce": {"kind": "op", "op": "cross_entropy", "of": "adapted.logits", "target": "train.target"},
        "mce": {"kind": "op", "op": "mean", "of": "ce"},
        "hit": {"kind": "op", "op": "match", "of": "adapted.logits", "expected": "train.target"},
        "acc": {"kind": "op", "op": "mean", "of": "hit"}
      }
    }
  },
  "saves": ["clean_mce", "fit.held_out.mce", "fit.held_out.acc", "fit.held_out.ce", "fit.history.mce", "fit.lora"]
}
```

```text
$ nnspec table out/lora_train clean_mce fit.held_out.mce fit.held_out.acc
clean_mce          fit.held_out.mce    fit.held_out.acc
4.160462856292725  0.1650763601064682  1.0
```

- **The null is a clean forward.** `B` is drawn as zeros, so the untrained adapter is the model: at
  `"epochs": 0` the held-out `ce` equals `clean_ce` to the bit, row for row, and `fit.held_out.acc` is 0.0.
- **Trained, it transfers to the other phrasing**: 7 of 7 held out, at a scale `alpha / rank` of 2.

### Apply it in chosen steps

The loading document declares the adapter once, as a named bundle of entries under the root `interventions`.
A step that names the bundle runs the adapted model; one that does not runs the model.
`documents/lora_apply.json` keeps `lora_train.json`'s `model`, `sites` and `test` rows, with:

```json
{
  "featurizers": {"lora": {"kind": "lora", "file_path": "out/lora_train/fit.lora.safetensors"}},
  "interventions": {
    "adapted": {"adapt": {"site": "target", "pos": {"frame": ":", "index": ":"}, "ops": [{"op": "apply", "featurizer": "lora"}]}},
    "prefill_only": {"adapt": {"site": "target", "pos": {"index": ":"}, "ops": [{"op": "apply", "featurizer": "lora"}]}}
  },
  "steps": {
    "adapted": {"kind": "forward", "input": "test.input", "interventions": ["adapted", {"logits": {"site": "head", "pos": -1}}]},
    "adapted_ce": {"kind": "op", "op": "cross_entropy", "of": "adapted.logits", "target": "test.target"},
    "gen": {"kind": "generate", "input": "test.input", "interventions": "adapted", "max_new_tokens": 4, "do_sample": false},
    "gen_prefill": {"kind": "generate", "input": "test.input", "interventions": "prefill_only", "max_new_tokens": 4, "do_sample": false},
    "said": {"kind": "op", "op": "detokenize", "of": "gen"},
    "said_prefill": {"kind": "op", "op": "detokenize", "of": "gen_prefill"}
  },
  "saves": ["adapted_ce", "said", "said_prefill"]
}
```

- **`rank` and `alpha` come from the bundle's header.**
- **`adapted_ce` equals `lora_train.json`'s `fit.held_out.ce` to the bit.**
- **`"frame": ":"` keeps the adapter on at every decode call.** `said` reads ` Thursday.` on the Tuesday row;
  with the adapter at the prefill only, `said_prefill` reads ` Thursday, according to`: the first token is the
  same and 5 of 7 continuations differ.

### Across sites: `with`

An adapter on a projection reads the projection's input and adds to its output: `with` names the input, an
entry of the same forward at a site the forward reaches first. On layer 5's attention output projection it
reads `attention_premix` and adds at `attention_output`. `lora_train.json` with the sites `premix`
(`attention_premix`, layer 5) and `o_proj` (`attention_output`, layer 5) beside `head`, and this body forward:

```json
{"adapted": {"kind": "forward", "input": "train.input", "interventions": {
  "x": {"site": "premix", "pos": {"index": ":"}},
  "adapt": {"site": "o_proj", "pos": {"index": ":"}, "ops": [{"op": "apply", "featurizer": "fit.lora", "with": "adapted.x"}]},
  "logits": {"site": "head", "pos": -1}}}}
```

It reaches 7 of 7 held out, at a mean cross entropy of 0.18211933970451355. `adapted.x` is `[rows, pos,
num_heads, head_dim]`, and `apply` flattens the axes after the positions into the adapter's 768. The bundle's
header records the input's site, so a loading document gives both entries in its named bundle, where a bare
`"with": "x"` means that bundle's `x`:

```json
{"adapted": {"x": {"site": "premix", "pos": {"frame": ":", "index": ":"}},
             "adapt": {"site": "o_proj", "pos": {"frame": ":", "index": ":"}, "ops": [{"op": "apply", "featurizer": "lora", "with": "x"}]}}}
```

A forward that names it gives the fit's `fit.held_out.ce` to the bit. A use with no `with`, or a `with` at
another site, is refused. So is that bundle in a `generate` step, which serves no operand from its own
entries.
