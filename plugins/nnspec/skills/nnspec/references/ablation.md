# Ablation

An ablation writes something in a component's place in one forward and scores it against a clean forward over
the same rows: what the score loses is what the component contributed. [SKILL.md](../SKILL.md#a-complete-experiment)
shows zero and mean ablation as one changed write. This guide has the document every other kind changes, and
each kind's write, null and control. Attention knockout and mixtures of experts are in
[attention and experts](ablation-attention-and-experts.md).

## What breaks

- **A small total effect can hide a large one**: later components make up for an ablated one. Measure
  [the hydra effect](#the-hydra-effect) before calling a component unimportant.
- **`permute` pairs any two rows, and one seed is not a result.** Sweep `seed`, and write `"derangement": true`,
  or a row can keep its own value, a null inside the average.
- **A drop without a control of the same kind is not evidence**: a head at the same depth that the task does
  not use, random neurons of the same layer, a random direction, a key the query does not read.
- **A `mean` over a read whose rows hold different numbers of positions weighs tokens, not rows.**
- **An optimal constant fits the rows it trains on.** Score it on held-out rows, and with `kl` beside the
  metric it was trained to keep.
- **On a gated MLP `mlp_activation` is the gate alone**; ablate `mlp_neurons`, what the down projection sums.

## The document every recipe changes

A clean forward that reads the site and scores, a `mean` step over that read, and one ablated forward per value
of `fill`. Saved as `documents/zero_vs_mean.json`:

```json
{
  "description": "Zero and mean ablation of GPT-2's residual stream at the last token after layers 3 and 9. fill sweeps what the write puts there: 0.0, the mean over the rows, and the clean read itself, the null.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"ioi": {"rows": [
    {"input": "When Mary and John went to the store, John gave a drink to", "answer": " Mary", "other": " John"},
    {"input": "When Alice and Bob went to the office, Bob gave a book to", "answer": " Alice", "other": " Bob"},
    {"input": "When Anna and Mark went to the beach, Mark gave a gift to", "answer": " Anna", "other": " Mark"},
    {"input": "When Tom and James went to the train station, Tom gave a ball to", "answer": " James", "other": " Tom"},
    {"input": "When Sarah and David went to the swimming pool, Sarah gave a ring to", "answer": " David", "other": " Sarah"},
    {"input": "When Lisa and Paul went to the movie theater, Lisa gave a letter to", "answer": " Paul", "other": " Lisa"}
  ]}},
  "sites": {
    "target": {"component": "layer_output", "layers": {"sweep": [3, 9], "label": "layer"}},
    "head": {"component": "lm_head_output"}
  },
  "interventions": {"score": {"ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "ioi.answer", "b": "ioi.other"}]}}},
  "steps": {
    "clean": {"kind": "forward", "input": "ioi.input", "interventions": ["score", {"acts": {"site": "target", "pos": -1}}]},
    "mean": {"kind": "op", "op": "mean", "of": "clean.acts"},
    "ablated": {"kind": "forward", "input": "ioi.input", "interventions": [
      {"abl": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": {"sweep": [0.0, "mean", "clean.acts"], "label": "fill"}}]}},
      "score"]},
    "clean_ld": {"kind": "op", "op": "mean", "of": "clean.ld"},
    "ablated_ld": {"kind": "op", "op": "mean", "of": "ablated.ld"}
  },
  "saves": ["clean_ld", "ablated_ld"]
}
```

```text
$ nnspec table out/zero_vs_mean clean_ld ablated_ld
at                               clean_ld            ablated_ld
{"layer":3,"fill":0.0}           2.7463290691375732  -1.5115877389907837
{"layer":3,"fill":"mean"}        2.7463290691375732  2.822462797164917
{"layer":3,"fill":"clean.acts"}  2.7463290691375732  2.7463290691375732
{"layer":9,"fill":0.0}           2.7463290691375732  0.2309519499540329
{"layer":9,"fill":"mean"}        2.7463290691375732  0.7644945979118347
{"layer":9,"fill":"clean.acts"}  2.7463290691375732  2.7463290691375732
```

- **The null is the third `fill`**: the clean read written back equals `clean_ld` to the bit. A sweep holds
  numbers and names alike.
- **Read zero and the mean together.** After layer 3 zero costs 4.26 and the mean nothing: nothing that
  varies across these rows is there yet. After layer 9 zero costs 2.52 and the mean 1.98.
- **Each recipe below replaces `target`, the `mean` step or the `fill` values, and keeps the rest.** Rows 0 to
  2 name the answer first and rows 3 to 5 second, which the direction recipes use.

## What to put in its place

| kind | the write | removes | its null or check |
|---|---|---|---|
| zero, mean | `swap` with `0.0`, or with a `mean` step over a clean read | everything; what varies across the rows | the clean read swapped back |
| [resample](#resample) | `swap` with a `permute` step | this row's value, for another row's | `"seed": null` |
| [optimal constant](#the-optimal-constant) | `{"op": "apply", "featurizer": "fit.c"}` | this row's value, for the one vector that best keeps a score | `epochs: 0`: mean ablation |
| [direction](#a-direction) | `{"op": "project_out", "direction": "d"}` | one direction, the rest kept | a `cosine` read after it: 0 |
| [noise](#noise-and-clamp) | `{"op": "noise", "scale": s, "seed": k}` | some signal, at a dose | `scale` 0 |
| [clamp](#noise-and-clamp) | `{"op": "clamp", "min": a, "max": b}` | every value past a bound | bounds wider than the values |
| [hold](#holding-a-coordinate) | `project_out` of a unit direction, then `add` of it at a length | a direction's coordinate, set to one value | a `dot` read after it: the value |
| [keys](ablation-attention-and-experts.md#knocking-out-keys) | -1e9 at `attention_scores` | where a query looks | every head's scores restored |
| [expert](ablation-attention-and-experts.md#expert-knockout) | `scale` of `expert_weights` by a mask | an expert's share of each token | the mask of an id no slot holds |

A null that is a write gives the clean run's score to the bit.

## Resample

Each row takes another row's value at the site, so the write stays on the distribution and removes what is this
input's own. `permute` reorders the rows of a read the document already has (row i holds row π(i)'s), so the
donor costs no forward. `documents/resample.json` is `zero_vs_mean.json` with these two steps in place of
`mean`, and `{"sweep": ["donor", "identity"], "label": "fill"}` as the fills:

```json
{
  "donor": {"kind": "op", "op": "permute", "of": "clean.acts", "seed": {"sweep": [0, 1, 2], "label": "seed"}, "derangement": true},
  "identity": {"kind": "op", "op": "permute", "of": "clean.acts", "seed": null}
}
```

```text
$ nnspec table out/resample clean_ld ablated_ld
at                                   clean_ld            ablated_ld
{"layer":3,"seed":0,"fill":"donor"}  2.7463290691375732  2.823350191116333
{"layer":3,"fill":"identity"}        2.7463290691375732  2.7463290691375732
{"layer":3,"seed":1,"fill":"donor"}  2.7463290691375732  2.823247194290161
{"layer":3,"seed":2,"fill":"donor"}  2.7463290691375732  2.9020755290985107
{"layer":9,"seed":0,"fill":"donor"}  2.7463290691375732  0.17917124927043915
{"layer":9,"fill":"identity"}        2.7463290691375732  2.7463290691375732
{"layer":9,"seed":1,"fill":"donor"}  2.7463290691375732  -0.4623769223690033
{"layer":9,"seed":2,"fill":"donor"}  2.7463290691375732  0.8134854435920715
```

- **`"seed": null` is the identity permutation**, so its swap is the null.
- **Report the spread over seeds.** After layer 9 three seeds give 0.18, -0.46 and 0.81: a row that takes a
  donor naming the answer in the other place follows the donor.
- **Each seed's π is in the step's record**, `point.permutation` for each point of
  `Plan.load("out/resample").steps["donor"]`: seed 0 drew `[5, 2, 4, 1, 0, 3]`. Without `derangement` it draws
  `[2, 1, 0, 5, 3, 4]`, and row 1 keeps its own value.
- **A rule for which rows may pair is data**: give each row its donor prompt in a column and read it with a
  forward of its own.

## Means

- **Per position.** `"over": ["rows"]` on the `mean` step keeps `pos`: with the read and the write both at
  `{"index": "-3:"}`, each of the last three tokens takes the mean over the rows at its own place (0.74 is
  left after layer 9). Such a mean is written at the static width it was read at.
- **From another dataset.** A `mean` has no rows axis, so it writes into any dataset's rows, where the read it
  was made from is refused ([operands](sites-and-interventions.md#operands)).
  [The optimal constant](#the-optimal-constant)'s `held_mean` swaps the training rows' mean into held-out rows.
- **From another run.** A read another document saved comes in with a `load` step, and its `mean` needs no
  forward ([bundles and cached harvests](featurizers-bundles-and-harvests.md)).

**Over a span whose length differs by row**, a plain `mean` averages the positions present, so a two-token span
counts twice; average over `pos` first to weigh rows equally. `documents/ragged_mean.json` is
`zero_vs_mean.json` with the `columns` below added to `data.ioi` (each row's place, cut from its prompt), the
`pos` below on both `acts` and `abl`, the three steps in place of `mean`, and `["per_token", "per_row"]` as
the fills:

```json
{
  "columns": {"place": {"from": "input", "after": {"text": " to the"}, "before": {"text": ","}}},
  "pos": {"scope": {"column": "ioi.place"}},
  "steps": {
    "per_token": {"kind": "op", "op": "mean", "of": "clean.acts"},
    "row_means": {"kind": "op", "op": "mean", "of": "clean.acts", "over": ["pos"]},
    "per_row": {"kind": "op", "op": "mean", "of": "row_means"}
  }
}
```

After layer 3 `per_token` gives 2.807 and `per_row` 2.772 against a clean 2.746: the place is a part of the
prompt the task does not read. `nnspec show out/ragged_mean` prints where the scope landed,
`[7], [7], [7], [7,8], [7,8], [7,8]`, the same lists for the read and the write. That is this recipe's check:
the clean read does not swap back here, since a read of another forward pairs only at a width every row shares.

## Heads and neurons

Both are `zero_vs_mean.json` with another `target`. Every other key stands, so each point has its zero, its
mean and its null. For every head of layers 9 to 11, one at a time:

```json
{"target": {"component": "attention_premix", "layers": {"sweep": [9, 10, 11], "label": "layer"},
            "select": {"num_heads": {"sweep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "label": "head"}}}}
```

```text
$ nnspec table out/head_knockout clean_ld ablated_ld
at                                          clean_ld            ablated_ld
{"layer":10,"head":5,"fill":0.0}            2.7463290691375732  2.7439472675323486
{"layer":10,"head":7,"fill":0.0}            2.7463290691375732  3.5669047832489014
{"layer":10,"head":7,"fill":"mean"}         2.7463290691375732  3.379924774169922
{"layer":10,"head":7,"fill":"clean.acts"}   2.7463290691375732  2.7463290691375732
{"layer":11,"head":0,"fill":0.0}            2.7463290691375732  2.5007336139678955
{"layer":11,"head":0,"fill":"mean"}         2.7463290691375732  2.729647397994995
```

- **The mean is each head's own**, since `clean.acts` is read at the swept site, and the null equals the
  clean run at all 36 heads. `attention_premix` needs no eager attention.
- **The control is a head at the same depth that the task does not use**: 10.5 moves the score by -0.002
  under either fill.
- **A head can work against the answer**: without 10.7 the score rises. Zero still overstates: 11.0's zero
  costs 0.25 and its mean 0.02.
- **A head set is a list**, `"select": {"num_heads": [6, 9]}`: both heads in one write, one point.
- **To ablate at every position, leave `pos` out of the read and of the write.** A `mean` with no `over` then
  pools every token of every row into one vector per head, and the null still equals the clean run where
  the rows are one length ([pairing](positions.md#pairing-with-another-forward)).

For a set of neurons beside random sets of its size from its layer, the sets are one sweep:

```json
{"target": {"component": "mlp_activation", "layers": 6, "select": {"intermediate_size": {"sweep": [
   [1645, 1811, 2314, 2864], [165, 1060, 1577, 1723], [258, 550, 1044, 2334], [231, 347, 375, 1478]], "label": "set"}}}}
```

```text
$ nnspec table out/neuron_ablation clean_ld ablated_ld
at                                                clean_ld            ablated_ld
{"set":[1645,1811,2314,2864],"fill":0.0}          2.7463290691375732  2.1083984375
{"set":[1645,1811,2314,2864],"fill":"mean"}       2.7463290691375732  2.7490463256835938
{"set":[1645,1811,2314,2864],"fill":"clean.acts…  2.7463290691375732  2.7463290691375732
{"set":[165,1060,1577,1723],"fill":0.0}           2.7463290691375732  2.74395751953125
{"set":[258,550,1044,2334],"fill":0.0}            2.7463290691375732  2.7366535663604736
```

- **The first set is the layer's top four by activation times gradient of the score**
  ([attribution patching](patching-attribution.md)); the others are drawn at random outside the document,
  since a `select` holds fixed ids.
- **Zero and the mean disagree, and the mean is the finding.** Zeroing the four costs 0.64 and their mean
  nothing: they are on for every row (0.5 to 2.6), so nothing that varies across the rows passes there.
  Activation times gradient estimates the zero ablation, so it ranks always-on neurons first.
- **The control is the random sets**: at most 0.01 under zero.
- **A set across layers is a site per layer**, since a `select` applies at every layer of its site
  ([select](sites-and-interventions.md#select)). Its control matches the set's count in each layer.

## A direction

Use it to ask whether the model needs one direction of the residual stream, with everything orthogonal to it
kept. Here the direction at each layer is the mean last-token stream of the prompts that name the answer
first, less that of those that name it second, projected out of every token at every layer in one entry.
`documents/direction.json` is `zero_vs_mean.json` with two more datasets, `first` and `second` (the six rows
with `"take": 3` and with `"skip": 3`), the site `every` in place of `target`, and these steps in place of
`clean`, `mean` and `ablated`. `worst_cos`, saved too, is the largest `abs` of `ablated.abl` (`max` over
`layers`, `rows` and `pos`):

```json
{
  "sites": {"every": {"component": "layer_output", "layers": ":"}},
  "steps": {
    "h_first": {"kind": "forward", "input": "first.input", "interventions": {"acts": {"site": "every", "pos": -1}}},
    "h_second": {"kind": "forward", "input": "second.input", "interventions": {"acts": {"site": "every", "pos": -1}}},
    "m_first": {"kind": "op", "op": "mean", "of": "h_first.acts"},
    "m_second": {"kind": "op", "op": "mean", "of": "h_second.acts"},
    "contrast": {"kind": "op", "op": "sub", "of": "m_first", "with": "m_second"},
    "clean": {"kind": "forward", "input": "ioi.input", "interventions": ["score", {"acts": {"site": "every", "pos": -1}}]},
    "average": {"kind": "op", "op": "mean", "of": "clean.acts"},
    "zeros": {"kind": "op", "op": "mul", "of": "average", "with": 0.0},
    "random": {"kind": "op", "op": "noise", "of": "zeros", "scale": 1.0, "seed": 0},
    "ablated": {"kind": "forward", "input": "ioi.input", "interventions": [
      {"abl": {"site": "every", "pos": {"index": ":"},
               "ops": [{"op": "project_out", "direction": {"sweep": ["contrast", "random", "average"], "label": "fill"}},
                       {"op": "cosine", "with": {"sweep": ["contrast", "random", "average"], "label": "fill"}}]}},
      "score"]}
  }
}
```

```text
$ nnspec table out/direction clean_ld ablated_ld worst_cos
at                   clean_ld            ablated_ld          worst_cos
{"fill":"contrast"}  2.7463290691375732  2.005376100540161   8.195638656616211e-08
{"fill":"random"}    2.7463290691375732  2.9454143047332764  1.862645149230957e-08
{"fill":"average"}   2.7463290691375732  0.5337104797363281  4.075467586517334e-06
```

- **One direction per layer.** A `mean` keeps the `layers` axis of a read at `":"`, so `contrast` is
  `[12, 768]`, and at layer L the write takes the slice labelled L. For one direction at every layer, reduce
  the layers too: `"over": ["layers", "rows", "pos"]`.
- **The check is the `cosine` after the edit**, which reads what is left along the direction: at most 8e-8.
  `project_out` normalises the direction, so its length does not matter.
- **The control is a random direction per layer**, drawn by `noise` on zeros in an op step: it moves the
  score by +0.20 where the contrast direction costs 0.74. The mean direction, which every row shares, costs
  2.21.
- **A probe's direction is a value** `project_out` takes ([probes and LoRA](featurizers-probes-and-lora.md)).

## Noise and clamp

Both are edits whose null is the end of their range: `scale` 0, and bounds wider than every value.
`documents/noise_and_clamp.json` is `zero_vs_mean.json` with three forwards in place of `ablated`. One writes
`{"op": "noise", "scale": {"sweep": [0.0, 1.5, 6.0], "label": "scale"}, "seed": {"sweep": [1, 2, 3], "label":
"seed"}}` at layer 9's last token. The others write `{"op": "clamp", "min": -100.0, "max": 100.0}`, and the same
at ±3000, at every token of layer 2 (`"pos": {"index": ":"}`). Its clean forward also reads the sizes the doses
are set against: `norm` of layer 9 at the last token, whose mean is 166.6, and `abs` of layer 2, whose `max`
is 2459 at position 0 and 38 past it.

```text
$ nnspec table out/noise_and_clamp clean_ld noisy_ld
at                      clean_ld            noisy_ld
{"scale":0.0,"seed":1}  2.7463290691375732  2.7463290691375732
{"scale":6.0,"seed":1}  2.7463290691375732  3.8966224193573
{"scale":6.0,"seed":2}  2.7463290691375732  2.3361129760742188
{"scale":6.0,"seed":3}  2.7463290691375732  2.4351749420166016
```

- **Set the dose from the stream's own size.** Standard-normal noise over 768 entries has a norm of about 28
  times `scale`, so `scale` 6 is near layer 9's last-token norm.
- **One draw is not a result**: at `scale` 6 the score moves either way. Report the spread over seeds.
- **The clamp at 100 cuts one thing**, the first token's outsized activation, and leaves 2.607; at ±3000 it
  cuts nothing and equals the clean run.

## Holding a coordinate

To hold a value's coordinate along a direction at one value, `project_out` the unit direction, then `add` it
back at that length. `documents/hold_coordinate.json` holds every row at one group's mean coordinate along the
name-order direction at layer 9's last token (site `l9`). `unit` is `normalize` of the direction recipe's
`contrast` read there, `c_first` the `dot` of `m_first` with `unit` (37.94; `c_second` is -17.80), and
`at_first` is `unit` under `{"op": "scale", "factor": "c_first"}`. The write and its check are two entries of
one forward:

```json
{
  "hold": {"site": "l9", "pos": -1, "ops": [{"op": "project_out", "direction": "unit"},
                                           {"op": "add", "with": {"sweep": ["at_first", "at_second"], "label": "at"}}]},
  "after": {"site": "l9", "pos": -1, "ops": [{"op": "dot", "with": "unit"}]}
}
```

- **The check is `after`**, a `dot` read listed after the write: the target on every row, to 1e-5.
- **Set the value against the clean coordinate**, a `dot` of the clean read with `unit`: 36 to 40 on rows 0
  to 2, -14 to -22 on rows 3 to 5.
- **A row held at the other group's value loses the answer**: at `at_first`, rows 3 to 5 fall from 3.10, 2.46
  and 2.48 to 1.58, -1.26 and -1.51. A row held near its own coordinate moves by at most 0.23.
- **`{"op": "add", "with": "unit", "scale": 37.94}`** is the same write with the length as a number. A
  featurizer's coordinate is held between its `encode` and `decode` ([featurizers](featurizers.md)).

## The optimal constant

Use it to ask how much of a component's value is this input's own. One vector trained to stand in for the
value on every row keeps whatever is not, and the score it still loses is what no constant can supply. A
`constant` featurizer holds one vector of the site's width: `init` draws it, a fit trains it, and `apply`
writes it in place of the value. The fit step's fields are in [featurizers](featurizers.md).
`documents/optimal_constant.json` is `zero_vs_mean.json` with `target` at layer 9 alone, the dataset and the
featurizer below, and the `fit` step in place of `ablated`:

```json
{
  "data": {"held": {"rows": [
    {"input": "When Emma and Jack went to the library, Jack gave a pen to", "answer": " Emma", "other": " Jack"},
    {"input": "When Kate and Mike went to the museum, Kate gave a cake to", "answer": " Mike", "other": " Kate"},
    {"input": "When Laura and Steve went to the restaurant, Steve gave a key to", "answer": " Laura", "other": " Steve"},
    {"input": "When Amy and Peter went to the hospital, Amy gave a card to", "answer": " Peter", "other": " Amy"}
  ]}},
  "featurizers": {"c": {"kind": "constant", "init": "mean"}},
  "steps": {
    "fit": {"kind": "fit", "train": ["c"], "objective": {"sq": 1.0}, "epochs": {"sweep": [0, 20]},
            "batch_size": 2, "seed": 0, "optimizer": {"name": "adam", "lr": 0.05}, "steps": {
      "abl": {"kind": "forward", "input": "ioi.input", "interventions": [
        {"abl": {"site": "target", "pos": -1, "ops": [{"op": "apply", "featurizer": "fit.c"}]}}, "score"]},
      "change": {"kind": "op", "op": "sub", "of": "abl.ld", "with": "clean.ld"},
      "sq": {"kind": "op", "op": "square", "of": "change"}}}
  }
}
```

After the fit, three forwards of `held.input` score the held-out rows with `logit_diff` and, against the clean
one's logits, `kl`: `held_clean` writes nothing, `held_c` writes `{"op": "apply", "featurizer": "fit.c"}` at
`target`, and `held_mean` writes `{"op": "swap", "with": "mean"}` there. A `mean` step averages each (`clean_ld`,
`mean_ld`, `c_ld`, `mean_kl`, `c_kl`).

```text
$ nnspec table out/optimal_constant clean_ld mean_ld c_ld mean_kl c_kl
at             clean_ld            mean_ld             c_ld                mean_kl             c_kl
{"epochs":0}   3.4645137786865234  0.2601776123046875  0.2601776123046875  0.6141062378883362  0.6141062378883362
{"epochs":20}  3.4645137786865234  0.2601776123046875  0.6446895599365234  0.6141062378883362  0.8974226117134094
```

- **The null is the `epochs: 0` point**: the untrained constant is mean ablation, to the bit, in the logit
  difference and in the KL. Outside a fit the bare `c` is that same draw.
- **The trained constant learned the six rows, not the task.** Their mean squared change (`fit.history.sq`)
  falls from 3.32 after the first epoch to 0.0027 after the twentieth. On the held-out rows it keeps 0.64 of a
  clean 3.46, against the mean's 0.26, and moves the rest of the distribution more (KL 0.90 against 0.61).
- **`init` names one vector laid out on the site's last axis alone**, or is a number (`0` is zero ablation).
- **A constant per head is a site per head**, `"select": {"num_heads": [7]}`, whose `init` reduces the head
  axis too, `"over": ["rows", "pos", "num_heads"]`. Unfitted, it gives head 10.7's mean ablation above to
  the bit.

## The hydra effect

When a component is ablated, later ones can change what they write and make up for it, so the drop in the
total understates what it did. Measure each later layer's direct effect, clean and ablated, in one run: the
score less the score with that layer's attention output subtracted from the final residual stream.
`documents/hydra.json` is `zero_vs_mean.json` with these `sites` and these steps in place of `clean`, `mean`
and `ablated`; `de_c` and `de_a`, saved too, are `mean` steps over `de_clean` and `de_ablated`:

```json
{
  "sites": {
    "attn9": {"component": "attention_output", "layers": 9},
    "probe": {"component": "attention_output", "layers": {"sweep": [9, 10, 11], "label": "layer"}},
    "final": {"component": "layer_output", "layers": -1},
    "head": {"component": "lm_head_output"}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "ioi.input", "interventions": ["score", {
      "a9": {"site": "attn9", "pos": -1}, "a": {"site": "probe", "pos": -1}}]},
    "mean9": {"kind": "op", "op": "mean", "of": "clean.a9"},
    "ablated": {"kind": "forward", "input": "ioi.input", "interventions": [{
      "abl": {"site": "attn9", "pos": -1, "ops": [{"op": "swap", "with": {"sweep": ["mean9", "clean.a9"], "label": "fill"}}]},
      "a": {"site": "probe", "pos": -1}}, "score"]},
    "clean_off": {"kind": "forward", "input": "ioi.input", "interventions": [{
      "off": {"site": "final", "pos": -1, "ops": [{"op": "sub", "with": "clean.a"}]}}, "score"]},
    "ablated_off": {"kind": "forward", "input": "ioi.input", "interventions": [{
      "abl": {"site": "attn9", "pos": -1, "ops": [{"op": "swap", "with": {"sweep": ["mean9", "clean.a9"], "label": "fill"}}]},
      "off": {"site": "final", "pos": -1, "ops": [{"op": "sub", "with": "ablated.a"}]}}, "score"]},
    "de_clean": {"kind": "op", "op": "sub", "of": "clean.ld", "with": "clean_off.ld"},
    "de_ablated": {"kind": "op", "op": "sub", "of": "ablated.ld", "with": "ablated_off.ld"}
  }
}
```

```text
$ nnspec table out/hydra clean_ld ablated_ld de_c de_a
at                              clean_ld            ablated_ld          de_c                  de_a
{"fill":"mean9","layer":9}      2.7463290691375732  2.640977144241333   2.6578826904296875    0.481964111328125
{"fill":"mean9","layer":10}     2.7463290691375732  2.640977144241333   -0.24392573535442352  1.8836263418197632
{"fill":"mean9","layer":11}     2.7463290691375732  2.640977144241333   -3.237017869949341    -3.112107515335083
{"fill":"clean.a9","layer":9}   2.7463290691375732  2.7463290691375732  2.6578826904296875    2.6578826904296875
{"fill":"clean.a9","layer":10}  2.7463290691375732  2.7463290691375732  -0.24392573535442352  -0.24392573535442352
{"fill":"clean.a9","layer":11}  2.7463290691375732  2.7463290691375732  -3.237017869949341    -3.237017869949341
```

- **Read it as a balance.** The total effect is 2.75 - 2.64 = 0.11. Layer 9's attention loses 2.18 of its
  direct effect (2.66 to 0.48), and layer 10's gains 2.13 (-0.24 to 1.88): it compensates.
- **The null** (`fill` `clean.a9`) gives every direct effect its clean value and `ablated_ld` the clean run's,
  to the bit.
- **`ablated.a` is read under the ablation**, and a later forward subtracts it: a read inside an intervened
  forward is an operand like any other. At layer 9 it is listed after the write, so it reads the written mean.
- **The direct effect goes through the final norm**, so the layers' direct effects do not sum to the total.
  It is per layer: no component holds one head's contribution to the stream.
