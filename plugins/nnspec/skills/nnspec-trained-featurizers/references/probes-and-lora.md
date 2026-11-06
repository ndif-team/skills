# Probes and LoRA

A `linear` featurizer is a probe, `h W + b`, one score per class; it only encodes. A `lora` is a low-rank
adapter at one site; it only applies. Both train in a fit through the frozen model. Each kind's fields are the
`clab` skill's [featurizers-and-fit reference](../../clab/references/featurizers-and-fit.md).

## What breaks here

- **A probe's classes are the label column's distinct values, sorted**, compared exactly: `yes` and `Yes` are
  two classes. `top_k` and `argmax` over a probe's scores need `"over": "classes"`, and their ids index the
  sorted list.
- **The floor is not 1/classes.** An untrained probe scores every class 0, and `match` breaks the tie to the
  lower class id, so its accuracy is the share of rows whose label sorts first.
- **Hold out what the probe must generalise over.** A probe trained and scored on the same cities reads token
  identity; one scored on unseen cities reads the property.
- **Decodable is not used.** Project the probe's direction out beside a random direction
  ([below](#use-a-probe-causally)).
- **An adapter stands for a weight change, so it acts at every position**: `{"index": ":"}`, the first
  position included. Narrowed to the last token it is an activation edit there.
- **In a generation, write the adapter at `"frame": ":"`.** A `pos` with no `frame` is the prefill only, and
  every decode call runs unadapted.
- **Hold out what the adapter must transfer to.** An adapter trained on five days and scored on the other two
  learns those five answers, and its held-out cross entropy rises.

## A probe per layer

Which layers of GPT-2 encode a city's country, for cities the probe never saw? The probe reads the last token,
which is the city's last token, and trains on 24 cities of four countries:

```json
{
  "description": "A linear probe per layer for the country of a city, read at GPT-2's last token (the city's last token), trained on 24 cities and scored on 12 others. epochs 0 is the untrained floor; mce, the mean held-out cross entropy, ranks layers whose accuracies tie.",
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
  "sites": {"stream": {"component": "layer_output", "layers": {"sweep": [0, 2, 4, 8, 11]}}},
  "featurizers": {"probe": {"kind": "linear", "classes": "train.country"}},
  "steps": {
    "fit": {
      "kind": "fit", "train": ["probe"], "objective": {"ce": 1.0},
      "epochs": {"sweep": [0, 30]}, "batch_size": 8, "seed": 0, "optimizer": {"name": "adam", "lr": 0.01},
      "eval": {"held_out": {"train": "test"}, "train": {}},
      "steps": {
        "read": {"kind": "forward", "input": "train.input", "interventions": {
          "scores": {"site": "stream", "pos": -1, "ops": [{"op": "encode", "featurizer": "fit.probe"}]}}},
        "ce": {"kind": "op", "op": "cross_entropy", "of": "read.scores", "target": "train.country"},
        "hit": {"kind": "op", "op": "match", "of": "read.scores", "expected": "train.country"},
        "acc": {"kind": "op", "op": "mean", "of": "hit"},
        "mce": {"kind": "op", "op": "mean", "of": "ce"}
      }
    }
  },
  "saves": ["fit.held_out.acc", "fit.held_out.mce", "fit.held_out.ce", "fit.train.ce"]
}
```

```text
clab run documents/probe.json
clab table out/probe fit.held_out.acc fit.held_out.mce
at                         fit.held_out.acc    fit.held_out.mce
{"layers":0,"epochs":0}    0.25                1.3862943649291992
{"layers":0,"epochs":30}   0.5                 2.3142340183258057
{"layers":2,"epochs":30}   0.8333333134651184  0.7212057113647461
{"layers":4,"epochs":30}   0.9166666865348816  0.41435277462005615
{"layers":8,"epochs":30}   1.0                 0.0029896905180066824
{"layers":11,"epochs":30}  0.9166666865348816  0.5249781608581543
```

- **The two sweeps cross**, since they sweep different fields: ten fits, keyed by layer and epochs.
- **`epochs=0` is the floor**, 0.25 at every layer: France sorts first and is 3 of the 12 held-out rows.
- **The 12 held-out cities** decode at 0.5 from layer 0, where only the token is there, and at 1.0 from layer 8.
  Every trained point scores 1.0 on its training rows, so only the held-out column says anything.
- **Rank layers by `mce` where accuracies tie**: layers 4 and 11 both get 11 of 12, and `mce` puts 4 ahead
  (0.414 against 0.525). On an easy set a row's cross entropy underflows and prints `-0.0`, as `fit.train.ce`
  does on 13 to 23 of the 24 training rows at each trained layer.
- **The `held_out` eval** reads `test` wherever the body says `train`: the input, the target and the classes
  scored. The class list stays the training column's, and a held-out label outside it is refused before any
  forward.
- **Each fit takes seconds on a CPU**, with a forward per minibatch. A swept fit's `fit.probe` holds a probe
  per point, which no featurizer loads: fit the layer you keep unswept to save one.

## Many probes in one fit

One forward per minibatch can serve many probes: one `linear` featurizer, one entry and one objective term per
probe. A probe's parameters appear in its own term alone, so it trains as it would alone. `probe.json` with
everything but its `description`, `model` and `data` replaced by this:

```json
{
  "sites": {"l4": {"component": "layer_output", "layers": 4}, "l8": {"component": "layer_output", "layers": 8}},
  "featurizers": {"p4": {"kind": "linear", "classes": "train.country"}, "p8": {"kind": "linear", "classes": "train.country"}},
  "steps": {
    "fit": {
      "kind": "fit", "train": ["p4", "p8"], "objective": {"ce4": 1.0, "ce8": 1.0},
      "epochs": 30, "batch_size": 8, "seed": 0, "optimizer": {"name": "adam", "lr": 0.01},
      "eval": {"held_out": {"train": "test"}},
      "steps": {
        "read": {"kind": "forward", "input": "train.input", "interventions": {
          "p4": {"site": "l4", "pos": -1, "ops": [{"op": "encode", "featurizer": "fit.p4"}]},
          "p8": {"site": "l8", "pos": -1, "ops": [{"op": "encode", "featurizer": "fit.p8"}]}}},
        "ce4": {"kind": "op", "op": "cross_entropy", "of": "read.p4", "target": "train.country"},
        "ce8": {"kind": "op", "op": "cross_entropy", "of": "read.p8", "target": "train.country"},
        "hit4": {"kind": "op", "op": "match", "of": "read.p4", "expected": "train.country"},
        "hit8": {"kind": "op", "op": "match", "of": "read.p8", "expected": "train.country"},
        "acc4": {"kind": "op", "op": "mean", "of": "hit4"},
        "acc8": {"kind": "op", "op": "mean", "of": "hit8"}
      }
    }
  },
  "saves": ["fit.held_out.acc4", "fit.held_out.acc8", "fit.held_out.ce4", "fit.held_out.ce8"]
}
```

`fit.held_out.acc4` is 0.917 and `fit.held_out.acc8` 1.0, and each probe's held-out `ce` equals the swept
document's at that layer to the bit. Build a document of many probes in Python as a dict; `early_stop` judges
one body op and ends the fit for every probe at once, so leave it out.

## Use a probe causally

A trained probe's tensors are values: `fit.probe.W`, `[hidden_size, classes]`, and `fit.probe.b`, `[classes]`,
as the fit left them. A two-class probe's direction is `W[:, 1] - W[:, 0]`: two `select`s of `classes`, each
summed over it, then a `sub`. `project_out` takes it in the same document (the `clab-ablation` skill sends a
probe here). This one trains a sentiment probe on GPT-2's layer 10 output at the last token of 12 reviews and
projects its direction out of 8 held-out ones:

```json
{
  "description": "A two-class sentiment probe's direction, projected out of GPT-2's layer 10 output in one document. The fit trains a linear probe on 12 reviews' last-token stream; its weight fit.probe.W is a value, [hidden_size, classes], whose classes sort negative then positive. Two selects of classes, each summed over it, and a sub give W[:, 1] - W[:, 0]. On 8 held-out reviews, projected removes it at every position, swept as fill against a random direction (standard normal, seed 0) and the zero direction, the null, which leaves the clean run. margin is the mean logit of each row's answer less the other sentiment's.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {
    "train": {"rows": [
      {"input": "I loved this film. Overall, the movie was", "label": "positive", "answer": " good", "other": " bad"},
      {"input": "I hated this film. Overall, the movie was", "label": "negative", "answer": " bad", "other": " good"},
      {"input": "The acting was superb and the story moving. Overall, the movie was", "label": "positive", "answer": " good", "other": " bad"},
      {"input": "The acting was wooden and the story dull. Overall, the movie was", "label": "negative", "answer": " bad", "other": " good"},
      {"input": "A delightful, funny and warm picture. Overall, the movie was", "label": "positive", "answer": " good", "other": " bad"},
      {"input": "A boring, clumsy and joyless picture. Overall, the movie was", "label": "negative", "answer": " bad", "other": " good"},
      {"input": "Brilliant direction and a wonderful cast. Overall, the movie was", "label": "positive", "answer": " good", "other": " bad"},
      {"input": "Awful direction and a miscast lead. Overall, the movie was", "label": "negative", "answer": " bad", "other": " good"},
      {"input": "I enjoyed every minute of it. Overall, the movie was", "label": "positive", "answer": " good", "other": " bad"},
      {"input": "I was bored from start to finish. Overall, the movie was", "label": "negative", "answer": " bad", "other": " good"},
      {"input": "A beautiful and touching story. Overall, the movie was", "label": "positive", "answer": " good", "other": " bad"},
      {"input": "An ugly and pointless story. Overall, the movie was", "label": "negative", "answer": " bad", "other": " good"}
    ]},
    "test": {"rows": [
      {"input": "The best film I have seen this year. Overall, the movie was", "label": "positive", "answer": " good", "other": " bad"},
      {"input": "The worst film I have seen this year. Overall, the movie was", "label": "negative", "answer": " bad", "other": " good"},
      {"input": "Clever, charming and great fun. Overall, the movie was", "label": "positive", "answer": " good", "other": " bad"},
      {"input": "Tedious, shallow and far too long. Overall, the movie was", "label": "negative", "answer": " bad", "other": " good"},
      {"input": "A moving and joyful ending. Overall, the movie was", "label": "positive", "answer": " good", "other": " bad"},
      {"input": "A dull and lifeless ending. Overall, the movie was", "label": "negative", "answer": " bad", "other": " good"},
      {"input": "Smart writing and lovely music. Overall, the movie was", "label": "positive", "answer": " good", "other": " bad"},
      {"input": "Lazy writing and awful music. Overall, the movie was", "label": "negative", "answer": " bad", "other": " good"}
    ]}
  },
  "sites": {"stream": {"component": "layer_output", "layers": 10}, "head": {"component": "lm_head_output"}},
  "featurizers": {"probe": {"kind": "linear", "classes": "train.label"}},
  "steps": {
    "harvest": {"kind": "forward", "input": "train.input", "interventions": {"h": {"site": "stream", "pos": -1}}},
    "fit": {"kind": "fit", "train": ["probe"], "objective": {"ce": 1.0},
            "epochs": 50, "batch_size": 12, "seed": 0, "optimizer": {"name": "adam", "lr": 0.01},
            "steps": {
              "scores": {"kind": "op", "op": "encode", "featurizer": "fit.probe", "of": "harvest.h"},
              "ce": {"kind": "op", "op": "cross_entropy", "of": "scores", "target": "train.label"},
              "hit": {"kind": "op", "op": "match", "of": "scores", "expected": "train.label"}}},
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
  "saves": ["fit.history.hit", "held_acc", "clean_margin", "margin"]
}
```

```text
clab run documents/probe_direction.json
clab table out/probe_direction held_acc clean_margin
held_acc  clean_margin
1.0       1.0175151824951172
clab table out/probe_direction margin
at                    value
{"fill":"direction"}  0.4754676818847656
{"fill":"random"}     1.0189380645751953
{"fill":"zero"}       1.0175151824951172
```

- **The probe's direction is used.** It reads the held-out sentiment on 8 of 8 rows, and projected out it
  takes the margin from 1.018 to 0.475; a random direction moves it by 0.001.
- **The null is the zero direction**: `project_out` leaves every vector as it is along it, so its margin is
  the clean run's to the bit.
- **`select` keeps `classes` at size 1, labelled with the class it kept**, and the `sum` over it drops the
  axis exactly: `direction` equals `W[:, 1] - W[:, 0]` read from the saved bundle, to the bit.
- **A `sub` of the two selects without the sums is refused**: `'negative' holds classes [0], and the value it
  acts on holds [1]; …`.
- **The classes sort**, `negative` before `positive`, so `direction` points toward positive.
- **Name a tensor after the fit that trains it**: inside it the body is refused (`'fit.probe.W' is a tensor of
  'fit.probe', which this fit trains: …`), and the bare `probe.W` after it is the draw, all zeros.
- **A probe of more than two classes has a span, not one direction.** Take the class differences the question
  needs, a `select` pair and a `sub` each. To patch along the whole span, save its orthonormal basis as a
  `subspace` bundle with a clab header ([harvest-and-bundles.md](harvest-and-bundles.md#what-a-run-writes)).

## LoRA: train an adapter

A rank-4 adapter at layer 5's output, at every position, trained to answer the day after tomorrow on one
phrasing and scored on another:

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
clab run documents/lora_train.json
clab table out/lora_train clean_mce fit.held_out.mce fit.held_out.acc
clean_mce          fit.held_out.mce    fit.held_out.acc
4.160462856292725  0.1650763601064682  1.0
```

- **The null is a clean forward.** `B` is drawn as zeros, so the untrained adapter is the model. At
  `"epochs": 0` the held-out `ce` equals `clean_ce` to the bit, row for row, and `fit.held_out.acc` is 0.0.
- **Trained, it transfers to the other phrasing**: 7/7 held out, mean `ce` 4.16 to 0.165, at `alpha / rank` 2.

## Apply it in chosen steps

The loading document declares the adapter once, as a named bundle of entries under the root `interventions`.
A step that names the bundle runs the adapted model; one that does not runs the model:

<!-- test: skip -->
```json
{
  "description": "The apply half of lora_train.json: the adapter loaded from its bundle (rank and alpha from the header) and declared once as the named bundle of entries 'adapted'. A step that names it runs the adapted model; one that does not runs the model.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"test": {"rows": [
    {"input": "Today is Monday, so tomorrow will be", "target": " Wednesday"}, {"input": "Today is Tuesday, so tomorrow will be", "target": " Thursday"},
    {"input": "Today is Wednesday, so tomorrow will be", "target": " Friday"}, {"input": "Today is Thursday, so tomorrow will be", "target": " Saturday"},
    {"input": "Today is Friday, so tomorrow will be", "target": " Sunday"}, {"input": "Today is Saturday, so tomorrow will be", "target": " Monday"},
    {"input": "Today is Sunday, so tomorrow will be", "target": " Tuesday"}
  ]}},
  "sites": {"target": {"component": "layer_output", "layers": 5}, "head": {"component": "lm_head_output"}},
  "featurizers": {"lora": {"kind": "lora", "file_path": "out/lora_train/fit.lora.safetensors"}},
  "interventions": {
    "adapted": {"adapt": {"site": "target", "pos": {"frame": ":", "index": ":"}, "ops": [{"op": "apply", "featurizer": "lora"}]}},
    "prefill_only": {"adapt": {"site": "target", "pos": {"index": ":"}, "ops": [{"op": "apply", "featurizer": "lora"}]}}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "test.input", "interventions": {"logits": {"site": "head", "pos": -1}}},
    "adapted": {"kind": "forward", "input": "test.input", "interventions": ["adapted", {"logits": {"site": "head", "pos": -1}}]},
    "clean_ce": {"kind": "op", "op": "cross_entropy", "of": "clean.logits", "target": "test.target"},
    "adapted_ce": {"kind": "op", "op": "cross_entropy", "of": "adapted.logits", "target": "test.target"},
    "gen": {"kind": "generate", "input": "test.input", "interventions": "adapted", "max_new_tokens": 4, "do_sample": false},
    "gen_prefill": {"kind": "generate", "input": "test.input", "interventions": "prefill_only", "max_new_tokens": 4, "do_sample": false},
    "said": {"kind": "op", "op": "detokenize", "of": "gen"},
    "said_prefill": {"kind": "op", "op": "detokenize", "of": "gen_prefill"}
  },
  "saves": ["clean_ce", "adapted_ce", "said", "said_prefill"]
}
```

```text
clab run documents/lora_apply.json
clab table out/lora_apply said
row  input                                    value
1    Today is Tuesday, so tomorrow will be     Thursday.\n\n
2    Today is Wednesday, so tomorrow will be   Friday.\n\n
```

- **`adapted_ce` equals `lora_train.json`'s `fit.held_out.ce` to the bit**, mean 0.165 against 4.160 clean.
- **`frame: ":"` keeps the adapter on at every decode call.** With it at the prefill only (`prefill_only`),
  the first token is the same and 5 of 7 continuations differ (row 2: `Friday, Saturday or`).

## Across sites: `with`

An adapter on a projection reads the projection's input and adds to its output: `with` names the input, an
entry of the same forward at a site the forward reaches first. On layer 5's attention output projection it
reads `attention_premix` and adds at `attention_output`. `lora_train.json` with the root sites `premix`
(`attention_premix`, layer 5) and `o_proj` (`attention_output`, layer 5) beside `head`, and this body step:

```json
{"adapted": {"kind": "forward", "input": "train.input", "interventions": {
  "x": {"site": "premix", "pos": {"index": ":"}},
  "adapt": {"site": "o_proj", "pos": {"index": ":"}, "ops": [{"op": "apply", "featurizer": "fit.lora", "with": "adapted.x"}]},
  "logits": {"site": "head", "pos": -1}}}}
```

It reaches 7/7 held out, mean cross entropy 0.182. `adapted.x` is `[rows, pos, num_heads, head_dim]`, and
`apply` flattens the axes after the positions into the adapter's 768. The header records the input's site
(`with.component` `attention_premix`, `with.layer` `5`). In the loading document both entries go in the
named bundle, and a bare `"with": "x"` means that bundle's `x`:

```json
{"adapted": {"x": {"site": "premix", "pos": {"frame": ":", "index": ":"}},
             "adapt": {"site": "o_proj", "pos": {"frame": ":", "index": ":"}, "ops": [{"op": "apply", "featurizer": "lora", "with": "x"}]}}}
```

A forward that names it gives the fit's `fit.held_out.ce` to the bit. A use with no `with`, or a `with` at
another site, is refused before any forward (`was trained reading its input at attention_premix layer 5
(with)`). A `generate` step refuses the bundle, since it serves no operand from its own entries.

