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
- **Decodable is not used.** Patch or ablate along the probe's span, beside drawn subspaces of the same rank
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

`project_out` takes a value and a `linear` featurizer has no `decode`, so the way to patch, ablate or project
out along a probe's direction is a `subspace` bundle of its weights (the `clab-ablation` skill sends a probe
here). This probe reads the day at `das.json`'s site, layer 5 at the day token:

```json
{
  "description": "A probe for the day at GPT-2's layer 5 output at the day token, the site das.json trains its rotation at; saved, unswept, so its weights can be read back.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"days": {"rows": [
    {"input": "If today is Monday, tomorrow is", "day": "Monday"}, {"input": "If today is Tuesday, tomorrow is", "day": "Tuesday"},
    {"input": "If today is Wednesday, tomorrow is", "day": "Wednesday"}, {"input": "If today is Thursday, tomorrow is", "day": "Thursday"},
    {"input": "If today is Friday, tomorrow is", "day": "Friday"}, {"input": "If today is Saturday, tomorrow is", "day": "Saturday"},
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

`fit.train.acc` is 1.0. From the corpus root, after the run, this writes the probe's span as a subspace bundle
with the header clab writes, and three random bases of the same rank as its control. The span's rank is the
classes less one: 6 here, and for two classes the one direction `W[:, 1] - W[:, 0]`, normalised.

<!-- test: skip -->
```python
import torch
from safetensors import safe_open
from safetensors.torch import save_file

with safe_open("out/day_probe/fit.probe.safetensors", "pt") as f:
    W, header = f.get_tensor("W"), f.metadata()                        # [768, classes]: one column per class
if W.shape[1] == 2:                                                    # one direction, toward the second class
    span = ((W[:, 1] - W[:, 0]) / (W[:, 1] - W[:, 0]).norm())[:, None]
else:                                                                  # the class differences: rank classes - 1
    span = torch.linalg.qr(W - W.mean(1, keepdim=True)).Q[:, :W.shape[1] - 1]
bases = {"span": span}
for seed in range(3):                                                  # random bases of the same rank
    bases[f"random{seed}"] = torch.linalg.qr(torch.randn(span.shape, generator=torch.Generator().manual_seed(seed))).Q
keep = ["model.key", "model.revision", "model.dtype", "component", "layer", "width"]
for name, basis in bases.items():
    save_file({"basis": basis.contiguous()}, f"out/day_probe/{name}.safetensors", metadata={
        **{k: header[k] for k in keep}, "format": "clab/1", "kind": "subspace",
        "k": str(basis.shape[1]), "parametrization": "cayley", "document": "probe_span"})
```

To patch along it, `das_apply.json` (in [SKILL.md](../SKILL.md)) takes `{"kind": "subspace", "file_path":
"out/day_probe/span.safetensors"}` as `rot`. To ablate along it, a `swap` to 0 inside the bracket removes what
lies along the span and keeps the rest, as `project_out` would along each of its directions:

<!-- test: skip -->
```json
{
  "description": "Ablate the day probe's span at GPT-2's layer 5 output at the day token: encode through the span bundle, set its coordinates to 0, decode. A second entry reads what is left along the span; the null is the bracket with no edit, which must equal the clean run.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"days": {"rows": [
    {"input": "If today is Monday, tomorrow is", "answer": " Tuesday"}, {"input": "If today is Tuesday, tomorrow is", "answer": " Wednesday"},
    {"input": "If today is Wednesday, tomorrow is", "answer": " Thursday"}, {"input": "If today is Thursday, tomorrow is", "answer": " Friday"},
    {"input": "If today is Friday, tomorrow is", "answer": " Saturday"}, {"input": "If today is Saturday, tomorrow is", "answer": " Sunday"},
    {"input": "If today is Sunday, tomorrow is", "answer": " Monday"}
  ]}},
  "sites": {"target": {"component": "layer_output", "layers": 5}, "head": {"component": "lm_head_output"}},
  "featurizers": {"span": {"kind": "subspace", "file_path": "out/day_probe/span.safetensors"}},
  "steps": {
    "clean": {"kind": "forward", "input": "days.input", "interventions": {
      "logits": {"site": "head", "pos": -1, "ops": [{"op": "token_prob", "token": "days.answer"}]}}},
    "null": {"kind": "forward", "input": "days.input", "interventions": {
      "kept": {"site": "target", "pos": -4, "ops": [{"op": "encode", "featurizer": "span"}, {"op": "decode", "featurizer": "span"}]},
      "logits": {"site": "head", "pos": -1, "ops": [{"op": "token_prob", "token": "days.answer"}]}}},
    "ablated": {"kind": "forward", "input": "days.input", "interventions": {
      "abl": {"site": "target", "pos": -4, "ops": [{"op": "encode", "featurizer": "span"}, {"op": "swap", "with": 0.0},
                                                   {"op": "decode", "featurizer": "span"}]},
      "left": {"site": "target", "pos": -4, "ops": [{"op": "encode", "featurizer": "span"}]},
      "logits": {"site": "head", "pos": -1, "ops": [{"op": "token_prob", "token": "days.answer"}]}}},
    "p_clean": {"kind": "op", "op": "mean", "of": "clean.logits"},
    "p_null": {"kind": "op", "op": "mean", "of": "null.logits"},
    "p_ablated": {"kind": "op", "op": "mean", "of": "ablated.logits"},
    "left_abs": {"kind": "op", "op": "abs", "of": "ablated.left"},
    "worst_left": {"kind": "op", "op": "max", "of": "left_abs", "over": ["rows", "pos", "features"]}
  },
  "saves": ["p_clean", "p_null", "p_ablated", "worst_left"]
}
```

```text
clab run documents/probe_ablate.json
clab table out/probe_ablate p_clean p_null p_ablated worst_left
p_clean              p_null               p_ablated            worst_left
0.22559413313865662  0.22559413313865662  0.12499908357858658  7.718801498413086e-06
```

| subspace at layer 5, day token | rank | patched in: held-out acc, mean ce | projected out: mean P(answer) |
|---|---|---|---|
| the probe's span | 6 | 1.0, 1.707 | 0.125 |
| a drawn subspace, seeds 0, 1, 2 | 6 | 0.0 at each; 2.462, 2.470, 2.440 | 0.243, 0.218, 0.223 |
| `random0` to `random2` from the snippet | 6 | not run | 0.222, 0.216, 0.234 |
| DAS, trained (`das.json`'s `fit.rot`) | 4 | 1.0, 1.093 | 0.104 |

- **The probe's directions are used.** Projected out, they take the answer's probability from 0.226 to 0.125;
  drawn subspaces of the same rank move it by under 0.02.
- **The null is the bracket with no edit**, `p_null`, equal to `p_clean` to the bit; `left`, the span's
  coordinates after the write, is at most 7.7e-6.
- **The controls are random directions of the same rank.** The drawn rows put `{"kind": "subspace", "k": 6,
  "seed": {"sweep": [0, 1, 2]}}` in place of `span`, in one document. A random bundle loads like `span`, one
  document each, since a sweep cannot stand for `file_path` or an op's `featurizer`. For two classes these are
  random unit directions, and `project_out` normalises its direction, so a random direction of the probe
  direction's norm is the same control as a unit one.

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

