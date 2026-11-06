# Probes and LoRA

A `linear` featurizer is a probe, `h W + b`, one score per class; it only encodes. A `lora` is a low-rank
adapter at one site; it only applies. Both train in a fit through the frozen model. Each kind's fields are the
`clab` skill's [featurizers-and-fit reference](../../clab/references/featurizers-and-fit.md); positions are
its [positions reference](../../clab/references/positions.md), and generation options its
[generate reference](../../clab/references/generate.md).

## What breaks here

- **A probe's classes are the label column's distinct values, sorted**, compared exactly: `yes` and `Yes` are
  two classes. `top_k` and `argmax` over a probe's scores need `"over": "classes"`, and their ids index the
  sorted list.
- **The floor is not 1/classes.** An untrained probe scores every class 0, and `match` breaks the tie to the
  lower class id, so its accuracy is the share of rows whose label sorts first.
- **Hold out what the probe must generalise over.** A probe trained and scored on the same cities reads token
  identity; one scored on unseen cities reads the property.
- **Decodable is not used.** Test use with an interchange along the probe's span, beside random subspaces of
  the same rank ([below](#use-a-probe-causally)).
- **An adapter stands for a weight change, so it acts at every position**: `{"index": ":"}`, the first
  position included. Narrowed to the last token it is an activation edit there.
- **In a generation, write the adapter at `"frame": ":"`.** A `pos` with no `frame` is the prefill only, and
  every decode call runs unadapted.
- **Hold out what the adapter must transfer to.** An adapter trained on five days and scored on the other two
  learns those five answers: its held-out cross entropy rises from 2.4-4.8 to 11-14 nats.

## A probe per layer

Which layers of GPT-2 encode a city's country, for cities the probe never saw? The probe reads the last token,
which is the city's last token, and trains on 24 cities of four countries:

```json
{
  "description": "A linear probe per layer for the country of a city, read at GPT-2's last token (the city's last token), trained on 24 cities and scored on 12 others. epochs 0 is the untrained floor.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {
    "train": {"rows": [
      {"input": "I live in Paris", "country": "France"},
      {"input": "I live in Rome", "country": "Italy"},
      {"input": "I live in Tokyo", "country": "Japan"},
      {"input": "I live in Berlin", "country": "Germany"},
      {"input": "I live in Lyon", "country": "France"},
      {"input": "I live in Milan", "country": "Italy"},
      {"input": "I live in Osaka", "country": "Japan"},
      {"input": "I live in Munich", "country": "Germany"},
      {"input": "I live in Marseille", "country": "France"},
      {"input": "I live in Naples", "country": "Italy"},
      {"input": "I live in Kyoto", "country": "Japan"},
      {"input": "I live in Hamburg", "country": "Germany"},
      {"input": "I live in Toulouse", "country": "France"},
      {"input": "I live in Turin", "country": "Italy"},
      {"input": "I live in Yokohama", "country": "Japan"},
      {"input": "I live in Frankfurt", "country": "Germany"},
      {"input": "I live in Nice", "country": "France"},
      {"input": "I live in Venice", "country": "Italy"},
      {"input": "I live in Nagoya", "country": "Japan"},
      {"input": "I live in Cologne", "country": "Germany"},
      {"input": "I live in Bordeaux", "country": "France"},
      {"input": "I live in Florence", "country": "Italy"},
      {"input": "I live in Kobe", "country": "Japan"},
      {"input": "I live in Stuttgart", "country": "Germany"}
    ]},
    "test": {"rows": [
      {"input": "I live in Lille", "country": "France"},
      {"input": "I live in Bologna", "country": "Italy"},
      {"input": "I live in Sapporo", "country": "Japan"},
      {"input": "I live in Dresden", "country": "Germany"},
      {"input": "I live in Nantes", "country": "France"},
      {"input": "I live in Genoa", "country": "Italy"},
      {"input": "I live in Hiroshima", "country": "Japan"},
      {"input": "I live in Leipzig", "country": "Germany"},
      {"input": "I live in Strasbourg", "country": "France"},
      {"input": "I live in Palermo", "country": "Italy"},
      {"input": "I live in Fukuoka", "country": "Japan"},
      {"input": "I live in Hanover", "country": "Germany"}
    ]}
  },
  "sites": {"stream": {"component": "layer_output", "layers": {"sweep": [0, 2, 4, 8, 11]}}},
  "featurizers": {"probe": {"kind": "linear", "classes": "train.country"}},
  "steps": {
    "fit": {
      "kind": "fit", "train": ["probe"], "objective": {"ce": 1.0},
      "epochs": {"sweep": [0, 30]}, "batch_size": 8, "seed": 0, "optimizer": {"name": "adam", "lr": 0.01},
      "eval": {"held_out": {"train": "test"}},
      "steps": {
        "read": {"kind": "forward", "input": "train.input", "interventions": {
          "scores": {"site": "stream", "pos": -1, "ops": [{"op": "encode", "featurizer": "fit.probe"}]}}},
        "ce": {"kind": "op", "op": "cross_entropy", "of": "read.scores", "target": "train.country"},
        "hit": {"kind": "op", "op": "match", "of": "read.scores", "expected": "train.country"},
        "acc": {"kind": "op", "op": "mean", "of": "hit"}
      }
    }
  },
  "saves": ["fit.held_out.acc", "fit.held_out.ce"]
}
```

```text
clab run documents/probe.json
clab table out/probe fit.held_out.acc
at                         value
{"layers":0,"epochs":0}    0.25
{"layers":0,"epochs":30}   0.5
{"layers":2,"epochs":0}    0.25
{"layers":2,"epochs":30}   0.8333333134651184
{"layers":4,"epochs":0}    0.25
{"layers":4,"epochs":30}   0.9166666865348816
{"layers":8,"epochs":0}    0.25
{"layers":8,"epochs":30}   1.0
{"layers":11,"epochs":0}   0.25
{"layers":11,"epochs":30}  0.9166666865348816
```

- **The two sweeps cross**, since they sweep different fields: ten fits, keyed by layer and epochs.
- **`epochs=0` is the floor**, 0.25 at every layer: France sorts first and is 3 of the 12 held-out rows.
- **The 12 held-out cities** decode at 0.5 from layer 0, where only the token is there, and at 1.0 from layer 8.
  Every trained point scores 1.0 on its training rows, so only the held-out column says anything.
- **The `held_out` eval** reads `test` wherever the body says `train`: the input, the target and the classes
  scored. The class list stays the training column's, and a held-out label outside it is refused before any
  forward.
- **Ten fits run in under a minute on a CPU**, each with a forward per minibatch. A swept fit's `fit.probe`
  holds a probe per point, which no featurizer loads: fit the layer you keep unswept to save one.

## Many probes in one fit

One forward per minibatch can serve many probes: one `linear` featurizer, one entry and one objective term per
probe. A probe's parameters appear in its own term alone, so it trains as it would alone. `probe.json` with
its two sweeps replaced by these, at `"epochs": 30`:

```json
{
  "sites": {"l4": {"component": "layer_output", "layers": 4}, "l8": {"component": "layer_output", "layers": 8}},
  "featurizers": {"p4": {"kind": "linear", "classes": "train.country"}, "p8": {"kind": "linear", "classes": "train.country"}},
  "train": ["p4", "p8"],
  "objective": {"ce4": 1.0, "ce8": 1.0},
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
```

The root `sites` and `featurizers`, and inside the fit `train`, `objective` and `steps`, replace their
counterparts. `fit.held_out.acc4` is 0.917 and `fit.held_out.acc8` 1.0, and each probe's `fit.held_out.ce`
equals the swept document's at that layer to the bit. Build a document of many probes in Python as a dict;
`early_stop` judges one body op and ends the fit for every probe at once, so leave it out.

## Use a probe causally

A probe is not an operand, and a `linear` featurizer has no `decode`, so no op writes along it. To test
whether the model uses what a probe reads, save its span as a `subspace` bundle and run an interchange
through it. This probe reads the day at `das.json`'s site, layer 5 at the day token:

```json
{
  "description": "A probe for the day at GPT-2's layer 5 output at the day token, the site das.json trains its rotation at; saved, unswept, so its weights can be read back.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"days": {"rows": [
    {"input": "If today is Monday, tomorrow is", "day": "Monday"},
    {"input": "If today is Tuesday, tomorrow is", "day": "Tuesday"},
    {"input": "If today is Wednesday, tomorrow is", "day": "Wednesday"},
    {"input": "If today is Thursday, tomorrow is", "day": "Thursday"},
    {"input": "If today is Friday, tomorrow is", "day": "Friday"},
    {"input": "If today is Saturday, tomorrow is", "day": "Saturday"},
    {"input": "If today is Sunday, tomorrow is", "day": "Sunday"}
  ]}},
  "sites": {"target": {"component": "layer_output", "layers": 5}},
  "featurizers": {"probe": {"kind": "linear", "classes": "days.day"}},
  "steps": {
    "fit": {
      "kind": "fit", "train": ["probe"], "objective": {"ce": 1.0},
      "epochs": 30, "batch_size": 7, "seed": 0, "optimizer": {"name": "adam", "lr": 0.01},
      "eval": {"train": {}},
      "steps": {
        "read": {"kind": "forward", "input": "days.input", "interventions": {
          "scores": {"site": "target", "pos": -4, "ops": [{"op": "encode", "featurizer": "fit.probe"}]}}},
        "ce": {"kind": "op", "op": "cross_entropy", "of": "read.scores", "target": "days.day"},
        "hit": {"kind": "op", "op": "match", "of": "read.scores", "expected": "days.day"},
        "acc": {"kind": "op", "op": "mean", "of": "hit"}
      }
    }
  },
  "saves": ["fit.train.acc", "fit.probe"]
}
```

`fit.train.acc` is 1.0. From the corpus root, after the run, this writes the span of its class differences
(rank 6) as a subspace bundle with the header clab writes:

<!-- test: skip -->
```python
import torch
from safetensors import safe_open
from safetensors.torch import save_file

with safe_open("out/day_probe/fit.probe.safetensors", "pt") as f:
    W, header = f.get_tensor("W"), f.metadata()                        # [768, 7]: one column per class
span = torch.linalg.qr(W - W.mean(1, keepdim=True)).Q[:, :W.shape[1] - 1].contiguous()   # the class differences
keep = ["model.key", "model.revision", "model.dtype", "component", "layer", "width"]
save_file({"basis": span}, "out/day_probe/span.safetensors", metadata={
    **{k: header[k] for k in keep}, "format": "clab/1", "kind": "subspace",
    "k": str(span.shape[1]), "parametrization": "cayley", "document": "probe_span"})
```

`das_apply.json` (in [SKILL.md](../SKILL.md)) with `rot` replaced by `{"kind": "subspace", "file_path":
"out/day_probe/span.safetensors"}` swaps the probe's span from the counterfactual into the base:

| subspace swapped at layer 5, day token | rank | held-out acc | mean ce |
|---|---|---|---|
| the probe's span | 6 | 1.0 | 1.707 |
| a drawn subspace, seeds 0, 1, 2 | 6 | 0.0 at each | 2.462, 2.470, 2.440 |
| DAS, trained (`das.json`) | 4 | 1.0 | 1.093 |

The probe's directions are used: swapping them moves the answer where random directions of the same rank do
not. The drawn rows are `das_apply.json` with `{"kind": "subspace", "k": 6, "seed": {"sweep": [0, 1, 2]}}`.

## LoRA: train an adapter

A rank-4 adapter at layer 5's output, at every position, trained to answer the day after tomorrow on one
phrasing and scored on another:

```json
{
  "description": "Train a rank-4 LoRA at GPT-2's layer 5 output, at every position, to answer the day after tomorrow on one phrasing, and score it on another. B starts at zero, so the untrained adapter is the model: clean_mce is what an epochs: 0 fit's held-out mce equals.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {
    "train": {"rows": [
      {"input": "If today is Monday, tomorrow is", "target": " Wednesday"},
      {"input": "If today is Tuesday, tomorrow is", "target": " Thursday"},
      {"input": "If today is Wednesday, tomorrow is", "target": " Friday"},
      {"input": "If today is Thursday, tomorrow is", "target": " Saturday"},
      {"input": "If today is Friday, tomorrow is", "target": " Sunday"},
      {"input": "If today is Saturday, tomorrow is", "target": " Monday"},
      {"input": "If today is Sunday, tomorrow is", "target": " Tuesday"}
    ]},
    "test": {"rows": [
      {"input": "Today is Monday, so tomorrow will be", "target": " Wednesday"},
      {"input": "Today is Tuesday, so tomorrow will be", "target": " Thursday"},
      {"input": "Today is Wednesday, so tomorrow will be", "target": " Friday"},
      {"input": "Today is Thursday, so tomorrow will be", "target": " Saturday"},
      {"input": "Today is Friday, so tomorrow will be", "target": " Sunday"},
      {"input": "Today is Saturday, so tomorrow will be", "target": " Monday"},
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
clab table out/lora_train clean_mce
value
4.160462856292725
clab table out/lora_train fit.held_out.mce
value
0.1650763601064682
clab table out/lora_train fit.held_out.acc
value
1.0
```

- **The null is a clean forward.** `B` is drawn as zeros, so the untrained adapter is the model. At
  `"epochs": 0` the held-out `ce` equals `clean_ce` to the bit, row for row, and `fit.held_out.acc` is 0.0.
- **Trained, it transfers to the other phrasing**: 7/7 held out, mean cross entropy 4.16 down to 0.165.
- **The scale is `alpha / rank`**, 2 here. The bundle, `out/lora_train/fit.lora.safetensors`, holds `A`
  `[768, 4]` and `B` `[4, 768]`, its header `rank` and `alpha`.
- **One adapter at a site that picks several layers is shared.** Sweep `layers` for one adapter per layer.

## Apply it in chosen steps

The loading document declares the adapter once, as a named bundle of entries under the root `interventions`.
A step that names the bundle runs the adapted model; one that does not runs the model:

<!-- test: skip -->
```json
{
  "description": "The apply half of lora_train.json: the adapter loaded from its bundle (rank and alpha from the header) and declared once as the named bundle of entries 'adapted'. A step that names it runs the adapted model; one that does not runs the model.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"test": {"rows": [
    {"input": "Today is Monday, so tomorrow will be", "target": " Wednesday"},
    {"input": "Today is Tuesday, so tomorrow will be", "target": " Thursday"},
    {"input": "Today is Wednesday, so tomorrow will be", "target": " Friday"},
    {"input": "Today is Thursday, so tomorrow will be", "target": " Saturday"},
    {"input": "Today is Friday, so tomorrow will be", "target": " Sunday"},
    {"input": "Today is Saturday, so tomorrow will be", "target": " Monday"},
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
clab table out/lora_apply said_prefill
row  input                                    value
1    Today is Tuesday, so tomorrow will be     Thursday, according to
2    Today is Wednesday, so tomorrow will be   Friday, Saturday or
```

- **`adapted_ce` equals `lora_train.json`'s `fit.held_out.ce` to the bit**: the same rows through the same
  adapter. Mean 0.165 adapted against 4.160 clean.
- **`frame: ":"` keeps the adapter on at every decode call.** With it at the prefill only (`prefill_only`),
  the first token is the same and 5 of 7 continuations differ.

## Across sites: `with`

An adapter on a projection reads the projection's input and adds to its output: `with` names the input, an
entry of the same forward at a site the forward reaches first. On layer 5's attention output projection it
reads `attention_premix` and adds at `attention_output`. `lora_train.json` with these sites and this body
forward, the rest unchanged:

```json
{
  "sites": {"premix": {"component": "attention_premix", "layers": 5},
            "o_proj": {"component": "attention_output", "layers": 5},
            "head": {"component": "lm_head_output"}},
  "adapted": {"kind": "forward", "input": "train.input", "interventions": {
    "x": {"site": "premix", "pos": {"index": ":"}},
    "adapt": {"site": "o_proj", "pos": {"index": ":"},
              "ops": [{"op": "apply", "featurizer": "fit.lora", "with": "adapted.x"}]},
    "logits": {"site": "head", "pos": -1}}}
}
```

It reaches 7/7 held out, mean cross entropy 0.182. `adapted.x` is `[rows, pos, num_heads, head_dim]`, and
`apply` flattens the axes after the positions into the adapter's 768. The header records the input's site
(`with.component` `attention_premix`, `with.layer` `5`). In the loading document both entries go in the
named bundle, and a bare `"with": "x"` means that bundle's `x`:

```json
{
  "adapted": {
    "x": {"site": "premix", "pos": {"frame": ":", "index": ":"}},
    "adapt": {"site": "o_proj", "pos": {"frame": ":", "index": ":"},
              "ops": [{"op": "apply", "featurizer": "lora", "with": "x"}]}
  }
}
```

A forward that names it gives the fit's `fit.held_out.ce` to the bit. Refused before any forward, the last
because a `generate` step serves no operand from its own entries:

```text
clab: adapted.adapt: out/lora_oproj/fit.lora.safetensors was trained reading its input at attention_premix layer 5 (with), and this use gives no 'with'; give 'with' an entry at attention_premix layer 5
clab: adapted.adapt: out/lora_oproj/fit.lora.safetensors was trained reading its input at attention_premix layer 5 (with), and this use's with 'adapted.x' is layer_input layer 5
clab: interventions.adapted.adapt.ops[0].with in steps.gen: 'gen.x' is an entry of this generation; an operand from an entry of its own step is served in a forward, not inside a generate
```

