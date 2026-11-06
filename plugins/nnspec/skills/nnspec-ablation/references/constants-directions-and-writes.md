# Constants, means, directions, noise, clamp and holds

The fills beyond zero, mean and resample, each as a complete document on `openai-community/gpt2` and the IOI
prompts of the skill (clean logit difference 2.75), with its null and its numbers; each runs in seconds on a CPU.

## The optimal constant

Use it to ask how much of a component's value is this input's own. One vector trained to stand in for the
value at every row keeps whatever is not, and the score it still loses is what no constant can supply. A
`constant` featurizer holds one vector of the site's width; `init` draws it, a fit trains it, and `apply`
writes it in place of the value. The fit's fields are the `clab` skill's
[featurizers and fit](../../clab/references/featurizers-and-fit.md) reference.

```json
{
  "description": "Optimal constant ablation of GPT-2's residual stream at the last token after layer 9. The constant c is drawn as the mean over the six training rows and trained to keep each row's logit difference; epochs sweeps 0 (the untrained draw, mean ablation) and 20. Scored on four held-out rows by the logit difference and by the KL from the clean next-token distribution, beside a plain swap of the mean.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {
    "ioi": {"rows": [
      {"input": "When Mary and John went to the store, John gave a drink to", "answer": " Mary", "other": " John"},
      {"input": "When Tom and James went to the train station, Tom gave a ball to", "answer": " James", "other": " Tom"},
      {"input": "When Alice and Bob went to the office, Bob gave a book to", "answer": " Alice", "other": " Bob"},
      {"input": "When Sarah and David went to the swimming pool, Sarah gave a ring to", "answer": " David", "other": " Sarah"},
      {"input": "When Anna and Mark went to the beach, Mark gave a gift to", "answer": " Anna", "other": " Mark"},
      {"input": "When Lisa and Paul went to the movie theater, Lisa gave a letter to", "answer": " Paul", "other": " Lisa"}
    ]},
    "held": {"rows": [
      {"input": "When Emma and Jack went to the library, Jack gave a pen to", "answer": " Emma", "other": " Jack"},
      {"input": "When Kate and Mike went to the museum, Kate gave a cake to", "answer": " Mike", "other": " Kate"},
      {"input": "When Laura and Steve went to the restaurant, Steve gave a key to", "answer": " Laura", "other": " Steve"},
      {"input": "When Amy and Peter went to the hospital, Amy gave a card to", "answer": " Peter", "other": " Amy"}
    ]}
  },
  "sites": {"resid": {"component": "layer_output", "layers": 9}, "head": {"component": "lm_head_output"}},
  "featurizers": {"c": {"kind": "constant", "init": "mean"}},
  "steps": {
    "clean": {"kind": "forward", "input": "ioi.input", "interventions": {
      "acts": {"site": "resid", "pos": -1},
      "ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "ioi.answer", "b": "ioi.other"}]}}},
    "mean": {"kind": "op", "op": "mean", "of": "clean.acts"},
    "fit": {"kind": "fit", "train": ["c"], "objective": {"sq": 1.0}, "epochs": {"sweep": [0, 20], "label": "epochs"},
            "batch_size": 2, "seed": 0, "optimizer": {"name": "adam", "lr": 0.05}, "steps": {
      "abl": {"kind": "forward", "input": "ioi.input", "interventions": {
        "abl": {"site": "resid", "pos": -1, "ops": [{"op": "apply", "featurizer": "fit.c"}]},
        "ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "ioi.answer", "b": "ioi.other"}]}}},
      "change": {"kind": "op", "op": "sub", "of": "abl.ld", "with": "clean.ld"},
      "sq": {"kind": "op", "op": "square", "of": "change"}}},
    "held_clean": {"kind": "forward", "input": "held.input", "interventions": {
      "logits": {"site": "head", "pos": -1},
      "ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "held.answer", "b": "held.other"}]}}},
    "held_mean": {"kind": "forward", "input": "held.input", "interventions": {
      "abl": {"site": "resid", "pos": -1, "ops": [{"op": "swap", "with": "mean"}]},
      "ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "held.answer", "b": "held.other"}]},
      "kl": {"site": "head", "pos": -1, "ops": [{"op": "kl", "against": "held_clean.logits"}]}}},
    "held_c": {"kind": "forward", "input": "held.input", "interventions": {
      "abl": {"site": "resid", "pos": -1, "ops": [{"op": "apply", "featurizer": "fit.c"}]},
      "ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "held.answer", "b": "held.other"}]},
      "kl": {"site": "head", "pos": -1, "ops": [{"op": "kl", "against": "held_clean.logits"}]}}},
    "clean_ld": {"kind": "op", "op": "mean", "of": "held_clean.ld"},
    "mean_ld": {"kind": "op", "op": "mean", "of": "held_mean.ld"},
    "c_ld": {"kind": "op", "op": "mean", "of": "held_c.ld"},
    "mean_kl": {"kind": "op", "op": "mean", "of": "held_mean.kl"},
    "c_kl": {"kind": "op", "op": "mean", "of": "held_c.kl"}
  },
  "saves": ["fit.history.sq", "fit.c", "clean_ld", "mean_ld", "c_ld", "mean_kl", "c_kl"]
}
```

```text
$ clab run documents/optimal_constant.json
$ clab table out/optimal_constant clean_ld mean_ld c_ld mean_kl c_kl
at             clean_ld            mean_ld             c_ld                mean_kl             c_kl
{"epochs":0}   3.4645137786865234  0.2601795196533203  0.2601795196533203  0.6141049265861511  0.6141049265861511
{"epochs":20}  3.4645137786865234  0.2601795196533203  0.6511878967285156  0.6141049265861511  0.8856005668640137
$ clab table out/optimal_constant fit.history.sq
at             epochs  value
{"epochs":20}  0       3.4940803050994873
...
{"epochs":20}  19      0.002737880451604724
```

- **The null is the `epochs: 0` point**: the untrained constant is mean ablation, to the bit, in both the
  logit difference (0.260) and the KL (0.614). Outside a fit the bare `c` is the same draw.
- **The trained constant learned the six rows, not the task.** Its mean squared change on them fell from 3.49
  after the first epoch to 0.0027 after the twentieth. On the held-out rows it keeps 0.65 of a clean 3.46,
  against mean ablation's 0.26, and moves the rest of the distribution more (KL 0.89 against 0.61).
- **`init` names one vector laid out on the site's last axis**, here the step `mean`, made before the
  featurizer's first use: a `mean` of a read at the site over every other axis. `"init": 0` is zero ablation
  the same way.
- **`fit.c` is the trained constant, the bare `c` its draw.** Saved, `fit.c` is a bundle holding `vector`;
  loading and saving bundles is the `clab-trained-featurizers` skill.
- **Table `fit.history.sq` on its own.** Beside a per-row value its `epochs` axis crosses the rows, one
  line per row and epoch.
- **The vector is shared across every axis but the last**, so a constant per head is a site per head,
  `"select": {"num_heads": [h]}`. That read keeps `num_heads`, and its plain `mean` is refused as `init`
  (`'mean' is laid out ['num_heads', 'head_dim']`): reduce the head axis too, a fragment of `steps` that
  `"init": "c_init"` names:

```json
{"c_init": {"kind": "op", "op": "mean", "of": "clean.z", "over": ["rows", "pos", "num_heads"]}}
```

The kind's fields: `clab describe featurizers`.

## Means over a ragged scope or another dataset

A scope over a column's text reads each row's own tokens, here every token of the place (one for " store",
two for " train station"), so the read is ragged. `mean` over it averages the positions present into one
vector, which a write at the same ragged scope broadcasts over each row's own tokens.

```json
{
  "description": "Mean ablation over a ragged scope: every token of each row's place in GPT-2's residual stream after layer 3. per_token writes the mean over every place token of the six rows, per_row the mean of each row's own mean; the logit difference is scored.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"ioi": {"rows": [
    {"input": "When Mary and John went to the store, John gave a drink to", "answer": " Mary", "other": " John", "place": " store"},
    {"input": "When Tom and James went to the train station, Tom gave a ball to", "answer": " James", "other": " Tom", "place": " train station"},
    {"input": "When Alice and Bob went to the office, Bob gave a book to", "answer": " Alice", "other": " Bob", "place": " office"},
    {"input": "When Sarah and David went to the swimming pool, Sarah gave a ring to", "answer": " David", "other": " Sarah", "place": " swimming pool"},
    {"input": "When Anna and Mark went to the beach, Mark gave a gift to", "answer": " Anna", "other": " Mark", "place": " beach"},
    {"input": "When Lisa and Paul went to the movie theater, Lisa gave a letter to", "answer": " Paul", "other": " Lisa", "place": " movie theater"}
  ]}},
  "sites": {"resid": {"component": "layer_output", "layers": 3}, "head": {"component": "lm_head_output"}},
  "steps": {
    "clean": {"kind": "forward", "input": "ioi.input", "interventions": {
      "acts": {"site": "resid", "pos": {"scope": {"column": "ioi.place"}, "index": ":"}},
      "ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "ioi.answer", "b": "ioi.other"}]}}},
    "per_token": {"kind": "op", "op": "mean", "of": "clean.acts"},
    "row_means": {"kind": "op", "op": "mean", "of": "clean.acts", "over": ["pos"]},
    "per_row": {"kind": "op", "op": "mean", "of": "row_means"},
    "gap": {"kind": "op", "op": "sub", "of": "per_token", "with": "per_row"},
    "gap_norm": {"kind": "op", "op": "norm", "of": "gap"},
    "ablated": {"kind": "forward", "input": "ioi.input", "interventions": {
      "abl": {"site": "resid", "pos": {"scope": {"column": "ioi.place"}, "index": ":"},
              "ops": [{"op": "swap", "with": {"sweep": ["per_token", "per_row"], "label": "mean"}}]},
      "ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "ioi.answer", "b": "ioi.other"}]}}},
    "clean_ld": {"kind": "op", "op": "mean", "of": "clean.ld"},
    "ablated_ld": {"kind": "op", "op": "mean", "of": "ablated.ld"}
  },
  "saves": ["gap_norm", "clean_ld", "ablated_ld"]
}
```

```text
$ clab table out/ragged_mean clean_ld ablated_ld
at                    clean_ld            ablated_ld
{"mean":"per_token"}  2.7463290691375732  2.8072738647460938
{"mean":"per_row"}    2.7463290691375732  2.7723872661590576
$ clab table out/ragged_mean gap_norm
value
5.010769367218018
```

- **Check where the scope landed** with `clab show out/ragged_mean`: `clean.acts` sits at `[7]` on the
  one-token places and `[7,8]` on the two-token ones. The `clab` skill's
  [positions](../../clab/references/positions.md) reference has the rest of `pos`.
- **A plain `mean` weighs tokens**, so a two-token place counts twice. To weigh rows equally, average over
  `pos` first, then over the rows; here the two means differ by a norm of 5.0.
- **This is a part of the prompt the task does not read**: the logit difference moves by +0.06 and +0.03,
  nothing like the 2.75 at stake.

**A mean from another dataset.** A `mean` has no rows axis, so it writes into any dataset's rows. The
optimal-constant document above does it: `held_mean` swaps the mean harvested on `ioi` into `held`'s rows.
The harvest itself, with its rows, is refused as an operand on another dataset's rows.

## A direction at every layer

Use it to ask whether the model needs one direction of the residual stream, with everything orthogonal to
it kept: project it out of every token at every layer in one entry. Here the direction at each layer is
the mean last-token stream of the prompts that name the answer first, less that of the prompts that name
it second.

```json
{
  "description": "Directional ablation on GPT-2's IOI prompts. At each layer the direction is the mean residual stream at the last token of the prompts that name the answer first, less that of the prompts that name it second. It is projected out of every token at every layer in one entry, whose cosine with the same direction after the edit reads what is left along it. fill sweeps it against a random direction per layer (standard normal, seed 0) and the plain mean direction.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {
    "first": {"rows": [
      {"input": "When Mary and John went to the store, John gave a drink to", "answer": " Mary", "other": " John"},
      {"input": "When Alice and Bob went to the office, Bob gave a book to", "answer": " Alice", "other": " Bob"},
      {"input": "When Anna and Mark went to the beach, Mark gave a gift to", "answer": " Anna", "other": " Mark"}
    ]},
    "second": {"rows": [
      {"input": "When Tom and James went to the train station, Tom gave a ball to", "answer": " James", "other": " Tom"},
      {"input": "When Sarah and David went to the swimming pool, Sarah gave a ring to", "answer": " David", "other": " Sarah"},
      {"input": "When Lisa and Paul went to the movie theater, Lisa gave a letter to", "answer": " Paul", "other": " Lisa"}
    ]},
    "ioi": {"rows": [
      {"input": "When Mary and John went to the store, John gave a drink to", "answer": " Mary", "other": " John"},
      {"input": "When Alice and Bob went to the office, Bob gave a book to", "answer": " Alice", "other": " Bob"},
      {"input": "When Anna and Mark went to the beach, Mark gave a gift to", "answer": " Anna", "other": " Mark"},
      {"input": "When Tom and James went to the train station, Tom gave a ball to", "answer": " James", "other": " Tom"},
      {"input": "When Sarah and David went to the swimming pool, Sarah gave a ring to", "answer": " David", "other": " Sarah"},
      {"input": "When Lisa and Paul went to the movie theater, Lisa gave a letter to", "answer": " Paul", "other": " Lisa"}
    ]}
  },
  "sites": {"every": {"component": "layer_output", "layers": ":"}, "head": {"component": "lm_head_output"}},
  "steps": {
    "h_first": {"kind": "forward", "input": "first.input", "interventions": {"acts": {"site": "every", "pos": -1}}},
    "h_second": {"kind": "forward", "input": "second.input", "interventions": {"acts": {"site": "every", "pos": -1}}},
    "m_first": {"kind": "op", "op": "mean", "of": "h_first.acts"},
    "m_second": {"kind": "op", "op": "mean", "of": "h_second.acts"},
    "contrast": {"kind": "op", "op": "sub", "of": "m_first", "with": "m_second"},
    "clean": {"kind": "forward", "input": "ioi.input", "interventions": {
      "acts": {"site": "every", "pos": -1},
      "ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "ioi.answer", "b": "ioi.other"}]}}},
    "average": {"kind": "op", "op": "mean", "of": "clean.acts"},
    "zeros": {"kind": "op", "op": "mul", "of": "average", "with": 0.0},
    "random": {"kind": "op", "op": "noise", "of": "zeros", "scale": 1.0, "seed": 0},
    "ablated": {"kind": "forward", "input": "ioi.input", "interventions": {
      "abl": {"site": "every", "pos": {"index": ":"},
              "ops": [{"op": "project_out", "direction": {"sweep": ["contrast", "random", "average"], "label": "fill"}},
                      {"op": "cosine", "with": {"sweep": ["contrast", "random", "average"], "label": "fill"}}]},
      "ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "ioi.answer", "b": "ioi.other"}]}}},
    "clean_ld": {"kind": "op", "op": "mean", "of": "clean.ld"},
    "ablated_ld": {"kind": "op", "op": "mean", "of": "ablated.ld"},
    "abs_cos": {"kind": "op", "op": "abs", "of": "ablated.abl"},
    "worst_cos": {"kind": "op", "op": "max", "of": "abs_cos", "over": ["layers", "rows", "pos"]}
  },
  "saves": ["contrast", "clean_ld", "ablated_ld", "worst_cos"]
}
```

```text
$ clab table out/direction clean_ld ablated_ld worst_cos
at                   clean_ld            ablated_ld          worst_cos
{"fill":"contrast"}  2.7463290691375732  2.005376100540161   8.195638656616211e-08
{"fill":"random"}    2.7463290691375732  2.9454143047332764  1.862645149230957e-08
{"fill":"average"}   2.7463290691375732  0.5337104797363281  4.075467586517334e-06
```

- **One direction per layer.** `h_first.acts` is read at `layers: ":"`, so `mean` keeps its `layers` axis
  and `contrast` is `[12, 768]`; at layer L the write takes the slice labelled L. For one direction at
  every layer, reduce the layers too, `"over": ["layers", "rows", "pos"]`.
- **The `cosine` after the edit reads what is left along the direction**: at most 8e-8, so the write is
  where the direction was. One label in two ops of the entry is one axis, and each point runs its own
  forward. `project_out` normalises the direction, so its length does not matter.
- **The control is a random direction per layer**, drawn by `noise` on zeros: it moves the score +0.20,
  where the contrast direction costs 0.74. The plain mean direction, which every row shares, costs 2.21.
- A direction read at fewer layers than the write is made at is refused before any forward.
- **A probe's direction is a value** `project_out` takes: two `select`s of `fit.probe.W` and a `sub`, in the
  `clab-trained-featurizers` skill's [probes reference](../../clab-trained-featurizers/references/probes-and-lora.md#use-a-probe-causally).

## Noise and clamp

`noise` adds Gaussian noise of standard deviation `scale` at a dose; `clamp` cuts every element past a
bound. Both are edits in an entry, with the null at the end of their range: scale 0, and bounds wider than
every value. The `clean` forward also reads the magnitudes that set the doses.

```json
{
  "description": "Noise and clamping as writes on GPT-2's IOI prompts. noisy adds Gaussian noise to layer 9's residual stream at the last token, its standard deviation swept from 0 (the null) to 6 (a noise norm near the stream's own), each at three seeds. clamped clamps every element of layer 2's residual stream at every token to [-100, 100], which cuts only the first token's massive activation; clamp_null clamps to [-3000, 3000], which cuts nothing.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"ioi": {"rows": [
    {"input": "When Mary and John went to the store, John gave a drink to", "answer": " Mary", "other": " John"},
    {"input": "When Tom and James went to the train station, Tom gave a ball to", "answer": " James", "other": " Tom"},
    {"input": "When Alice and Bob went to the office, Bob gave a book to", "answer": " Alice", "other": " Bob"},
    {"input": "When Sarah and David went to the swimming pool, Sarah gave a ring to", "answer": " David", "other": " Sarah"},
    {"input": "When Anna and Mark went to the beach, Mark gave a gift to", "answer": " Anna", "other": " Mark"},
    {"input": "When Lisa and Paul went to the movie theater, Lisa gave a letter to", "answer": " Paul", "other": " Lisa"}
  ]}},
  "sites": {"l9": {"component": "layer_output", "layers": 9}, "l2": {"component": "layer_output", "layers": 2},
            "head": {"component": "lm_head_output"}},
  "interventions": {"score": {"ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "ioi.answer", "b": "ioi.other"}]}}},
  "steps": {
    "clean": {"kind": "forward", "input": "ioi.input", "interventions": ["score", {
      "first": {"site": "l2", "pos": 0, "ops": [{"op": "abs"}]},
      "rest": {"site": "l2", "pos": {"index": "1:"}, "ops": [{"op": "abs"}]},
      "size9": {"site": "l9", "pos": -1, "ops": [{"op": "norm"}]}}]},
    "first_max": {"kind": "op", "op": "max", "of": "clean.first", "over": ["rows", "pos", "hidden_size"]},
    "rest_max": {"kind": "op", "op": "max", "of": "clean.rest", "over": ["rows", "pos", "hidden_size"]},
    "norm9": {"kind": "op", "op": "mean", "of": "clean.size9"},
    "noisy": {"kind": "forward", "input": "ioi.input", "interventions": [{
      "noise": {"site": "l9", "pos": -1, "ops": [{"op": "noise", "scale": {"sweep": [0.0, 1.5, 6.0], "label": "scale"},
                                                  "seed": {"sweep": [1, 2, 3], "label": "seed"}}]}}, "score"]},
    "clamped": {"kind": "forward", "input": "ioi.input", "interventions": [{
      "clamp": {"site": "l2", "pos": {"index": ":"}, "ops": [{"op": "clamp", "min": -100.0, "max": 100.0}]}}, "score"]},
    "clamp_null": {"kind": "forward", "input": "ioi.input", "interventions": [{
      "clamp": {"site": "l2", "pos": {"index": ":"}, "ops": [{"op": "clamp", "min": -3000.0, "max": 3000.0}]}}, "score"]},
    "clean_ld": {"kind": "op", "op": "mean", "of": "clean.ld"},
    "noisy_ld": {"kind": "op", "op": "mean", "of": "noisy.ld"},
    "clamped_ld": {"kind": "op", "op": "mean", "of": "clamped.ld"},
    "clamp_null_ld": {"kind": "op", "op": "mean", "of": "clamp_null.ld"}
  },
  "saves": ["first_max", "rest_max", "norm9", "clean_ld", "noisy_ld", "clamped_ld", "clamp_null_ld"]
}
```

```text
$ clab table out/noise_and_clamp clean_ld noisy_ld
at                      clean_ld            noisy_ld
{"scale":0.0,"seed":1}  2.7463290691375732  2.7463290691375732
...
{"scale":1.5,"seed":1}  2.7463290691375732  3.2704620361328125
{"scale":1.5,"seed":2}  2.7463290691375732  2.5854721069335938
{"scale":1.5,"seed":3}  2.7463290691375732  2.5263073444366455
{"scale":6.0,"seed":1}  2.7463290691375732  3.8152503967285156
{"scale":6.0,"seed":2}  2.7463290691375732  2.1724777221679688
{"scale":6.0,"seed":3}  2.7463290691375732  2.4574038982391357
$ clab table out/noise_and_clamp first_max rest_max norm9 clamped_ld clamp_null_ld
first_max          rest_max           norm9               clamped_ld          clamp_null_ld
2458.553466796875  37.67317199707031  166.64654541015625  2.6074612140655518  2.7463290691375732
```

- **Scale 0 is the null**: the clean run to the bit at every seed.
- **Set the dose from the stream's own size.** Standard-normal noise over 768 entries has a norm of about
  28 times `scale`; at `scale` 6 that is near layer 9's last-token norm, 167 (`norm9`).
- **One draw is not a result.** At `scale` 6 the three seeds give 3.82, 2.17 and 2.46: the score moves either
  way. Report the spread over seeds. Mean ablation at the same site leaves 0.76: noise removes far less.
- **A row's noise is its own draw** from the seed, the row and the position alone, so batch size, sweep
  point and forward never change it.
- **The clamp at 100 cuts one value class only**: the first token's massive activation (2459 at layer 2,
  where every other token stays under 38). It costs 0.14 of the logit difference; bounds at 3000 cut
  nothing and equal the clean run.

### Holding a coordinate

To hold a value's coordinate along a direction at a value, `project_out` the unit direction, then `add` it
back at that length: `{"op": "add", "with": "at"}` with `at` a step's scaled direction, or `{"op": "add",
"with": "unit", "scale": v}`. Here every row is held at one template's name-order coordinate:

```json
{
  "description": "Holding a direction's coordinate at a value, on GPT-2's IOI prompts at layer 9's last token. The direction is the unit difference of the mean stream of the prompts that name the answer first and of those that name it second. held projects it out of every row and adds it back at the first group's mean coordinate, or the second's; after reads the coordinate left.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {
    "first": {"rows": [
      {"input": "When Mary and John went to the store, John gave a drink to", "answer": " Mary", "other": " John"},
      {"input": "When Alice and Bob went to the office, Bob gave a book to", "answer": " Alice", "other": " Bob"},
      {"input": "When Anna and Mark went to the beach, Mark gave a gift to", "answer": " Anna", "other": " Mark"}
    ]},
    "second": {"rows": [
      {"input": "When Tom and James went to the train station, Tom gave a ball to", "answer": " James", "other": " Tom"},
      {"input": "When Sarah and David went to the swimming pool, Sarah gave a ring to", "answer": " David", "other": " Sarah"},
      {"input": "When Lisa and Paul went to the movie theater, Lisa gave a letter to", "answer": " Paul", "other": " Lisa"}
    ]},
    "ioi": {"rows": [
      {"input": "When Mary and John went to the store, John gave a drink to", "answer": " Mary", "other": " John"},
      {"input": "When Alice and Bob went to the office, Bob gave a book to", "answer": " Alice", "other": " Bob"},
      {"input": "When Anna and Mark went to the beach, Mark gave a gift to", "answer": " Anna", "other": " Mark"},
      {"input": "When Tom and James went to the train station, Tom gave a ball to", "answer": " James", "other": " Tom"},
      {"input": "When Sarah and David went to the swimming pool, Sarah gave a ring to", "answer": " David", "other": " Sarah"},
      {"input": "When Lisa and Paul went to the movie theater, Lisa gave a letter to", "answer": " Paul", "other": " Lisa"}
    ]}
  },
  "sites": {"l9": {"component": "layer_output", "layers": 9}, "head": {"component": "lm_head_output"}},
  "steps": {
    "h_first": {"kind": "forward", "input": "first.input", "interventions": {"r": {"site": "l9", "pos": -1}}},
    "h_second": {"kind": "forward", "input": "second.input", "interventions": {"r": {"site": "l9", "pos": -1}}},
    "m_first": {"kind": "op", "op": "mean", "of": "h_first.r"},
    "m_second": {"kind": "op", "op": "mean", "of": "h_second.r"},
    "contrast": {"kind": "op", "op": "sub", "of": "m_first", "with": "m_second"},
    "unit": {"kind": "op", "op": "normalize", "of": "contrast"},
    "c_first": {"kind": "op", "op": "dot", "of": "m_first", "with": "unit"},
    "c_second": {"kind": "op", "op": "dot", "of": "m_second", "with": "unit"},
    "at_first": {"kind": "op", "op": "scale", "of": "unit", "factor": "c_first"},
    "at_second": {"kind": "op", "op": "scale", "of": "unit", "factor": "c_second"},
    "clean": {"kind": "forward", "input": "ioi.input", "interventions": {
      "r": {"site": "l9", "pos": -1},
      "ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "ioi.answer", "b": "ioi.other"}]}}},
    "coord": {"kind": "op", "op": "dot", "of": "clean.r", "with": "unit"},
    "held": {"kind": "forward", "input": "ioi.input", "interventions": {
      "hold": {"site": "l9", "pos": -1, "ops": [{"op": "project_out", "direction": "unit"},
                                               {"op": "add", "with": {"sweep": ["at_first", "at_second"], "label": "at"}}]},
      "after": {"site": "l9", "pos": -1, "ops": [{"op": "dot", "with": "unit"}]},
      "ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "ioi.answer", "b": "ioi.other"}]}}}
  },
  "saves": ["c_first", "c_second", "coord", "clean.ld", "held.ld", "held.after"]
}
```

```text
$ clab table out/hold_coordinate coord clean.ld held.ld
at                  row  pos  pos_token  input                                             coord                answer  other   clean.ld            held.ld
{"at":"at_first"}   0    13   Ġto        When Mary and John went to the store, John gave…  37.67071533203125     Mary    John   2.0063858032226562  2.0174789428710938
{"at":"at_first"}   1    13   Ġto        When Alice and Bob went to the office, Bob gave…  39.87980651855469     Alice   Bob    3.3816299438476562  3.2701492309570312
{"at":"at_first"}   2    13   Ġto        When Anna and Mark went to the beach, Mark gave…  36.27251052856445     Anna    Mark   3.048126220703125   3.1407241821289062
{"at":"at_first"}   3    14   Ġto        When Tom and James went to the train station, T…  -14.355535507202148   James   Tom    3.0953598022460938  1.5794219970703125
{"at":"at_first"}   4    14   Ġto        When Sarah and David went to the swimming pool,…  -17.320711135864258   David   Sarah  2.4641342163085938  -1.2577667236328125
{"at":"at_first"}   5    14   Ġto        When Lisa and Paul went to the movie theater, L…  -21.73158073425293    Paul    Lisa   2.4823379516601562  -1.5134429931640625
{"at":"at_second"}  0    13   Ġto        When Mary and John went to the store, John gave…  37.67071533203125     Mary    John   2.0063858032226562  -0.17346954345703125
{"at":"at_second"}  1    13   Ġto        When Alice and Bob went to the office, Bob gave…  39.87980651855469     Alice   Bob    3.3816299438476562  -0.13166046142578125
{"at":"at_second"}  2    13   Ġto        When Anna and Mark went to the beach, Mark gave…  36.27251052856445     Anna    Mark   3.048126220703125   0.2547454833984375
...
```

- **The check is `after`**: `clab table out/hold_coordinate held.after` gives every row the target,
  `c_first` 37.94 or `c_second` -17.80, to 1e-5.
- **A row held at the other template's value loses the answer.** At `at_first` the second template's rows
  fall to 1.58, -1.26 and -1.51; at `at_second` the first template's fall to -0.17, -0.13 and 0.25. A row
  held near its own coordinate moves by at most 0.23 (row 5 at `at_second`, 2.48 to 2.26).
- **Set the value against `coord`**, the clean coordinate: 36 to 40 on one template, -14 to -22 on the other.
- **Held at 0 it is the direction projected out**; held at the rows' mean coordinate it is a mean ablation
  along one direction. An SAE feature or another featurizer's coordinate is held between its `encode` and
  `decode`: the `clab-trained-featurizers` skill.

Edits and reductions: `clab describe ops`.
