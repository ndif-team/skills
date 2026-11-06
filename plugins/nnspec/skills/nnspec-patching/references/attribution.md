# Attribution patching

Attribution patching estimates what patching would measure, to first order: the metric's gradient at a site
on the base run, times the counterfactual's value there less the base's, summed over the site's features. One
backward gives the gradient at every site of a forward, so a map over every head costs two forwards where
patching costs one per head. In nnspec the gradient is a read: an entry whose first op is `grad`. Rank with
attribution, then confirm what you report with real patching ([the top points](#confirm-the-top-points)).

## What breaks

- **The `mul 0` null proves the wiring, not the metric.** It is a second entry at the head with the metric's
  ops and `{"op": "mul", "with": 0}` appended, and a `grad` of it. Its gradient is exactly 0 when the `grad` is
  of the entry you zeroed and the wiring is exact: 0.0 on every value of the `[12, 8, 1, 12, 64]` gradient
  below. It cannot tell you that entry is the metric you meant to report; the central difference at one point
  does ([check the gradient](#check-the-gradient)), and real patching of the top points does for the ranking.
- **`scale` 0 is not the null.** It is an edit, so after a metric it is refused:

  ```text
  nnspec: steps.base.interventions.zero.ops[1]: 'scale' is an edit, which writes back, and it follows 'logit_diff', which turned the value into ['rows', 'pos']: there is no live value to write; an edit comes before it, or put the transform in a later entry at this site
  ```

  `mul` writes nothing, so the null can stand anywhere in the forward.
- **A central difference along every element at once checks nothing on a LayerNorm model.** Adding one number
  to every element of GPT-2's residual stream leaves the logits unchanged, since each LayerNorm subtracts the
  mean: the gradient summed over `hidden_size` is 7.15e-07 below. Check along the counterfactual's difference
  instead, the direction attribution uses.
- **The estimate is linear, and patching is not.** At the token the counterfactual changes, early layers,
  patching moves the weekday logit difference by +3.386 at layer 0, and attribution estimates +2.070.
- **`grad` differentiates the sum of `of` over every axis it has.** Read the metric at one position (`pos: -1`)
  so each row contributes one number.
- **On the residual stream the last layers at the last token rank high**, and a patch there reproduces the
  counterfactual, so their sign agrees by construction. Layers 11 and 10 at the last token top the map below
  (3.368 and 3.273); test a ranking on points off the last token, or on heads.
- **`of` names an entry of the same forward, bare, at a site the forward reaches later.** `grad` is refused in
  a `generate`, in a fit's body and under `tp`.
- **A bfloat16 model's gradient is bfloat16.** Recompute the components you report in float32 (the
  `nnspec-debugging` skill has the rule).

## Every head from one backward

Read every head's result at `layers: ":"` on both prompts, take its gradient in the base forward, and combine
them in op steps after it. This runs on the IOI rows of `documents/head_patching.json` (in
[heads, neurons and paths](heads-neurons-and-paths.md)), so its map can be checked against real head patching.

```json
{
  "description": "Attribution patching of every head on the indirect-object task: each head's result at the last token, the gradient of the logit difference there on the base run, times the counterfactual's result less the base's, summed over head_dim. A counterfactual forward and a base forward with its backward give a [layers, num_heads] map. g0 is the gradient of the metric times 0 and must be exactly zero; self_gap is the base read again in a forward that takes no gradient, and must be exactly zero.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"example_id": "Mary-John", "input": "When Mary and John went to the store, John gave a drink to",
     "counterfactual": "When Mary and John went to the store, Mary gave a drink to", "base_answer": " Mary", "cf_answer": " John"},
    {"example_id": "John-Mary", "input": "When John and Mary went to the store, Mary gave a drink to",
     "counterfactual": "When John and Mary went to the store, John gave a drink to", "base_answer": " John", "cf_answer": " Mary"},
    {"example_id": "Alice-Bob", "input": "When Alice and Bob went to the store, Bob gave a drink to",
     "counterfactual": "When Alice and Bob went to the store, Alice gave a drink to", "base_answer": " Alice", "cf_answer": " Bob"},
    {"example_id": "Bob-Alice", "input": "When Bob and Alice went to the store, Alice gave a drink to",
     "counterfactual": "When Bob and Alice went to the store, Bob gave a drink to", "base_answer": " Bob", "cf_answer": " Alice"},
    {"example_id": "Sarah-David", "input": "When Sarah and David went to the store, David gave a drink to",
     "counterfactual": "When Sarah and David went to the store, Sarah gave a drink to", "base_answer": " Sarah", "cf_answer": " David"},
    {"example_id": "David-Sarah", "input": "When David and Sarah went to the store, Sarah gave a drink to",
     "counterfactual": "When David and Sarah went to the store, David gave a drink to", "base_answer": " David", "cf_answer": " Sarah"},
    {"example_id": "Anna-Mark", "input": "When Anna and Mark went to the store, Mark gave a drink to",
     "counterfactual": "When Anna and Mark went to the store, Anna gave a drink to", "base_answer": " Anna", "cf_answer": " Mark"},
    {"example_id": "Mark-Anna", "input": "When Mark and Anna went to the store, Anna gave a drink to",
     "counterfactual": "When Mark and Anna went to the store, Mark gave a drink to", "base_answer": " Mark", "cf_answer": " Anna"}
  ]}},
  "sites": {"heads": {"component": "attention_premix", "layers": ":"}, "head": {"component": "lm_head_output"}},
  "steps": {
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {"z": {"site": "heads", "pos": -1}}},
    "own": {"kind": "forward", "input": "pairs.input", "interventions": {"z": {"site": "heads", "pos": -1}}},
    "base": {"kind": "forward", "input": "pairs.input", "interventions": {
      "z": {"site": "heads", "pos": -1},
      "dz": {"site": "heads", "pos": -1, "ops": [{"op": "grad", "of": "metric"}]},
      "g0": {"site": "heads", "pos": -1, "ops": [{"op": "grad", "of": "zero"}]},
      "metric": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]},
      "zero": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
                                                  {"op": "mul", "with": 0}]}}},
    "difference": {"kind": "op", "op": "sub", "of": "counterfactual.z", "with": "base.z"},
    "term": {"kind": "op", "op": "scale", "of": "difference", "factor": "base.dz"},
    "per_head": {"kind": "op", "op": "sum", "of": "term", "over": ["pos", "head_dim"]},
    "map": {"kind": "op", "op": "mean", "of": "per_head", "over": ["rows"]},
    "g0_abs": {"kind": "op", "op": "abs", "of": "base.g0"},
    "g0_max": {"kind": "op", "op": "max", "of": "g0_abs", "over": ["layers", "rows", "pos", "num_heads", "head_dim"]},
    "self_gap": {"kind": "op", "op": "sub", "of": "own.z", "with": "base.z"},
    "self_abs": {"kind": "op", "op": "abs", "of": "self_gap"},
    "self_max": {"kind": "op", "op": "max", "of": "self_abs", "over": ["layers", "rows", "pos", "num_heads", "head_dim"]}
  },
  "saves": ["map", "g0_max", "self_max"]
}
```

- **`base` takes one backward per distinct `of`**, here two: one for `metric`, one for the null `zero`. Each
  fills its `grad` reads at all 12 layers, `[layers, rows, pos, num_heads, head_dim]`.
- **`metric` writes nothing**, so `base` is the clean run and `base.z` the base's own results.
- **`scale` in an op step multiplies two values elementwise**; `sum` over `pos` and `head_dim` leaves one
  number per layer, row and head, and `mean` over `rows` leaves the `[layers, num_heads]` map.
- **`own` is the self-counterfactual null**: the base read in a forward without gradients. Its difference from
  `base.z` must be exactly zero, which checks that the two reads are one selection.

```text
$ nnspec table out/attribution_heads g0_max self_max
g0_max  self_max
0.0     0.0

$ nnspec table out/attribution_heads map
layer  num_heads  value
7      3          0.6353127360343933
7      9          0.8699591755867004
8      6          0.7383905649185181
8      10         1.1651463508605957
9      9          1.0515251159667969
10     7          -1.2949762344360352
11     10         -1.2108073234558105
...
```

The run takes seconds on a CPU, against under a minute for the 144 patched forwards of head patching. Beside
the patching effects of `documents/head_patching.json` (its `ld_mean` less `clean_mean` at each head):

| head | patching | attribution |
|---|---|---|
| 8.10 | +1.155 | +1.165 |
| 9.9 | +1.018 | +1.052 |
| 7.9 | +0.906 | +0.870 |
| 8.6 | +0.817 | +0.738 |
| 7.3 | +0.674 | +0.635 |
| 11.10 | −1.195 | −1.211 |
| 10.7 | −1.561 | −1.295 |

The two rank the top eight heads in the same order, and their correlation over all 144 heads is 0.996. That
agreement is this task's, at the last token: check it on yours before trusting a ranking.

## Check the gradient

The null passes a `grad` of an entry other than the metric you report, and one of the metric at a wrong selection.
A central difference catches both: compare it once per new document with two forwards that move the site by plus
and minus a step along the counterfactual's difference. The step is swept under one label, so the nudges and the
division zip. This fragment is the `sites`, `interventions` and `steps` of `documents/gradient_check.json`, whose
`model` and `data` are those of the document above:

```json
{
  "sites": {"target": {"component": "layer_output", "layers": 6}, "head": {"component": "lm_head_output"}},
  "interventions": {"scored": {"metric": {"site": "head", "pos": -1,
                                          "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]}}},
  "steps": {
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {"v": {"site": "target", "pos": -1}}},
    "base": {"kind": "forward", "input": "pairs.input", "interventions": {"v": {"site": "target", "pos": -1}}},
    "difference": {"kind": "op", "op": "sub", "of": "counterfactual.v", "with": "base.v"},
    "back": {"kind": "op", "op": "neg", "of": "difference"},
    "taken": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"h": {"site": "target", "pos": -1, "ops": [{"op": "grad", "of": "metric"}, {"op": "scale", "factor": "difference"},
                                                  {"op": "sum", "over": ["hidden_size"]}]},
       "ones": {"site": "target", "pos": -1, "ops": [{"op": "grad", "of": "metric"}, {"op": "sum", "over": ["hidden_size"]}]}},
      "scored"]},
    "up": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"nudge": {"site": "target", "pos": -1,
                 "ops": [{"op": "add", "with": "difference", "scale": {"sweep": [1.0, 0.3, 0.1, 0.03, 0.01], "label": "step"}}]}},
      "scored"]},
    "down": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"nudge": {"site": "target", "pos": -1,
                 "ops": [{"op": "add", "with": "back", "scale": {"sweep": [1.0, 0.3, 0.1, 0.03, 0.01], "label": "step"}}]}},
      "scored"]},
    "rise": {"kind": "op", "op": "sub", "of": "up.metric", "with": "down.metric"},
    "half": {"kind": "op", "op": "scale", "of": "rise", "factor": 0.5},
    "central": {"kind": "op", "op": "div", "of": "half", "with": {"sweep": [1.0, 0.3, 0.1, 0.03, 0.01], "label": "step"}},
    "gap": {"kind": "op", "op": "sub", "of": "taken.h", "with": "central"},
    "h_abs": {"kind": "op", "op": "abs", "of": "taken.h"},
    "gap_abs": {"kind": "op", "op": "abs", "of": "gap"},
    "ones_abs": {"kind": "op", "op": "abs", "of": "taken.ones"},
    "h_max": {"kind": "op", "op": "max", "of": "h_abs"},
    "gap_max": {"kind": "op", "op": "max", "of": "gap_abs"},
    "ones_max": {"kind": "op", "op": "max", "of": "ones_abs"}
  },
  "saves": ["taken.h", "central", "h_max", "gap_max", "ones_max"]
}
```

```text
$ nnspec table out/gradient_check h_max ones_max
h_max                ones_max
0.24178534746170044  7.152557373046875e-07

$ nnspec table out/gradient_check gap_max
at             value
{"step":1.0}   0.004122670739889145
{"step":0.3}   0.0005037952214479446
{"step":0.1}   0.0002415776252746582
{"step":0.03}  0.000986989587545395
{"step":0.01}  0.002079218626022339
```

- **The gap falls as the step falls, then rises once float32 rounding takes over.** From step 1.0 to 0.3 it
  falls 8 times, near the 11 the step squared predicts; below 0.1 the forwards' rounding, divided by the step,
  grows.
- **Pass where the smallest gap is a small fraction of `h_max`**, the largest attribution at the point: 0.00024
  against 0.242 here, a thousandth. Set the tolerance before the run.
- **A gap far above it that does not shrink with the step** is a `grad` of an entry other than the metric you
  report, or of it at the wrong selection.
- **`ones_max`, the gradient summed over `hidden_size`, is 7.15e-07**: the all-ones direction is invisible to
  GPT-2's LayerNorms, so a check along it passes whatever the gradient is.

## Every layer and position from one backward

A site at `layers: ":"` read at a window of static width gives a `[layers, rows, pos, hidden_size]` gradient
from the same one backward. This fragment is the `sites` and `steps` of `documents/attribution_map.json`, whose
`model` and `data` are those of `documents/interchange.json` in [SKILL.md](../SKILL.md):

```json
{
  "sites": {"resid": {"component": "layer_output", "layers": ":"}, "head": {"component": "lm_head_output"}},
  "steps": {
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {
      "z": {"site": "resid", "pos": {"index": "0:7"}}}},
    "base": {"kind": "forward", "input": "pairs.input", "interventions": {
      "z": {"site": "resid", "pos": {"index": "0:7"}},
      "dz": {"site": "resid", "pos": {"index": "0:7"}, "ops": [{"op": "grad", "of": "metric"}]},
      "metric": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]}}},
    "difference": {"kind": "op", "op": "sub", "of": "counterfactual.z", "with": "base.z"},
    "term": {"kind": "op", "op": "scale", "of": "difference", "factor": "base.dz"},
    "per_row": {"kind": "op", "op": "sum", "of": "term", "over": ["hidden_size"]},
    "map": {"kind": "op", "op": "mean", "of": "per_row", "over": ["rows"]}
  },
  "saves": ["map"]
}
```

`map` is `[layers, pos]`; `nnspec table out/attribution_map map` prints a row per layer and position. Its `pos`
counts within the read's `0:7` window, here the position itself; in a `-3:` window, 0 is the third token from
the end. Beside the patching map's effect (its `ld_mean` less `clean_mean`) at the day's token and the last token:

| layer | day, patching | day, attribution | last token, patching | last token, attribution |
|---|---|---|---|---|
| 0 | +3.386 | +2.070 | −0.001 | −0.001 |
| 4 | +3.417 | +2.722 | −0.023 | −0.025 |
| 6 | +2.811 | +2.259 | +0.536 | +0.447 |
| 8 | +1.646 | +1.409 | +1.680 | +1.492 |
| 9 | +0.499 | +0.320 | +2.762 | +2.724 |
| 11 | 0.000 | 0.000 | +3.385 | +3.368 |

- **Attribution finds the same band**: the day's token through layer 6, the last token from layer 8.
- **It falls short where one patch restores the whole answer**: at the day's token in layer 0 it gives 61% of
  the patched effect. The ranking survives; the size does not.
- **The positions the two prompts share give exactly 0.0** at every layer, since their difference is zero.
  So does every position but the last at layer 11, whose gradient is zero.

## Confirm the top points

A list of scattered (layer, position) points cannot be one sweep axis: one label over different values is
refused. Patch each point as one entry, at a one-layer site and a counted `pos`, in a step of its own for its
own effect, or with the others in one forward for the set's. This fragment is the `sites`, `interventions` and
`steps` of `documents/top_points.json`, whose `model` and `data` are those of `documents/interchange.json`. It
patches two top points of the map above, the day at layer 2 and the last token at layer 10:

```json
{
  "sites": {"every": {"component": "layer_output", "layers": ":"}, "head": {"component": "lm_head_output"}},
  "interventions": {"scored": {"ld": {"site": "head", "pos": -1,
                                      "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]}}},
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": ["scored", {
      "at3": {"site": "every", "pos": 3}, "at6": {"site": "every", "pos": 6}}]},
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {
      "at3": {"site": "every", "pos": 3}, "at6": {"site": "every", "pos": 6}}},
    "null": {"kind": "forward", "input": "pairs.input", "interventions": [{
      "p2_3": {"site": {"component": "layer_output", "layers": 2}, "pos": 3, "ops": [{"op": "swap", "with": "clean.at3"}]},
      "p10_6": {"site": {"component": "layer_output", "layers": 10}, "pos": 6, "ops": [{"op": "swap", "with": "clean.at6"}]}},
      "scored"]},
    "l2_p3": {"kind": "forward", "input": "pairs.input", "interventions": [{
      "patch": {"site": {"component": "layer_output", "layers": 2}, "pos": 3,
                "ops": [{"op": "swap", "with": "counterfactual.at3"}]}}, "scored"]},
    "l10_p6": {"kind": "forward", "input": "pairs.input", "interventions": [{
      "patch": {"site": {"component": "layer_output", "layers": 10}, "pos": 6,
                "ops": [{"op": "swap", "with": "counterfactual.at6"}]}}, "scored"]},
    "clean_mean": {"kind": "op", "op": "mean", "of": "clean.ld"},
    "null_mean": {"kind": "op", "op": "mean", "of": "null.ld"},
    "l2_p3_mean": {"kind": "op", "op": "mean", "of": "l2_p3.ld"},
    "l10_p6_mean": {"kind": "op", "op": "mean", "of": "l10_p6.ld"}
  }
}
```

```text
$ nnspec table out/top_points clean_mean null_mean l2_p3_mean l10_p6_mean
clean_mean           null_mean            l2_p3_mean          l10_p6_mean
-1.6586096286773682  -1.6586096286773682  1.7238038778305054  1.6707872152328491
```

- **Each step gives its cell of `documents/layer_pos_map.json` to the bit**: an effect of +3.382 at layer 2,
  position 3, where attribution gave +2.783, and +3.329 at layer 10, position 6, against +3.273.
- **A read at `layers: ":"` is an operand for a write at any one layer**, which takes the slice labelled with
  its layer: one counterfactual read per position serves every point at that position.
- **The null writes both points in one forward**, the form that patches a set together, and equals the clean run.
- **Chosen heads at several layers are one site per head**, each in a step of its own:
  `{"component": "attention_premix", "layers": 9, "select": {"num_heads": [9]}}` for 9.9. Patched so, 10.7,
  8.10 and 9.9 give their cells of `documents/head_patching.json` to the bit.
- **A grid that covers the points also works**: sweep the layers and the positions they use, and read the
  cells you want; the extra cells cost a forward each.

## One prompt, no counterfactual

Zeroing a neuron moves its activation `a` by `−a`, so to first order the metric changes by `−a·g`, with `g` its
gradient there. Rank `a·g`, the predicted drop: the largest are the neurons whose removal lowers the metric
most. It is attribution with zero as the counterfactual, from one prompt and one backward; `top_k` over
`["layers", "intermediate_size"]` ranks every neuron of every layer together.

```json
{
  "description": "Rank every MLP neuron of GPT-2 on one prompt, with no counterfactual. Each neuron's activation at the last token times the gradient of P(' Paris') there is the first-order drop in P when it is zeroed; top_k ranks them over layers and neurons together. g0 is the gradient of the metric times 0 and must be exactly zero.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"prompts": {"rows": [
    {"example_id": "eiffel", "input": "The Eiffel Tower is in the city of", "answer": " Paris"}
  ]}},
  "sites": {"neurons": {"component": "mlp_activation", "layers": ":"}, "head": {"component": "lm_head_output"}},
  "steps": {
    "base": {"kind": "forward", "input": "prompts.input", "interventions": {
      "a": {"site": "neurons", "pos": -1},
      "da": {"site": "neurons", "pos": -1, "ops": [{"op": "grad", "of": "p"}]},
      "g0": {"site": "neurons", "pos": -1, "ops": [{"op": "grad", "of": "zero"}]},
      "p": {"site": "head", "pos": -1, "ops": [{"op": "token_prob", "token": "prompts.answer"}]},
      "zero": {"site": "head", "pos": -1, "ops": [{"op": "token_prob", "token": "prompts.answer"}, {"op": "mul", "with": 0}]}}},
    "drop": {"kind": "op", "op": "scale", "of": "base.a", "factor": "base.da"},
    "top": {"kind": "op", "op": "top_k", "of": "drop", "k": 5, "over": ["layers", "intermediate_size"]},
    "g0_abs": {"kind": "op", "op": "abs", "of": "base.g0"},
    "g0_max": {"kind": "op", "op": "max", "of": "g0_abs", "over": ["layers", "rows", "pos", "intermediate_size"]}
  },
  "saves": ["base.p", "top.ids", "top.values", "g0_max"]
}
```

```text
$ nnspec run documents/rank_neurons.json
$ nnspec table --csv out/rank_neurons top.ids top.values
example_id,row,pos,pos_token,input,top.ids,top.values
eiffel,0,9,Ġof,The Eiffel Tower is in the city of,"{""intermediate_size"":[483,1931,1434,1437,1793],""layers"":[6,7,8,5,10]}","[0.01716592349112034,0.016079207882285118,0.010278166271746159,0.00914700049906969,0.008967471309006214]"
```

- **`g0_max` is 0.0**, and `base.p`, the probability of ` Paris`, is 0.06997019797563553.
- **`top.ids` holds one list per axis**, in rank order: neuron 483 of layer 6 first. `--csv` prints the cell whole.
- **Rank `a·g`, never its negation**: `−a·g` ranks the neurons whose zeroing raises the metric. For the bottom
  k, `neg` then `top_k`.

Confirm by zeroing the top five together. A `select` applies at every layer its site picks, so neurons in
several layers are one site per layer. Group the (layer, neuron) pairs of `top.ids` by layer, read from the
`--csv` cell or from `Plan.load(out).results["top.ids"]` (one tensor per axis), and give each layer's neurons
as its site's `select` list. This fragment is the `sites` and `steps` of `documents/zero_top.json`, whose
`model` and `data` are those above; `scale` by 0 zeroes, and `keep` 1.0 writes each neuron back as it was:

```json
{
  "sites": {
    "l5": {"component": "mlp_activation", "layers": 5, "select": {"intermediate_size": [1437]}},
    "l6": {"component": "mlp_activation", "layers": 6, "select": {"intermediate_size": [483]}},
    "l7": {"component": "mlp_activation", "layers": 7, "select": {"intermediate_size": [1931]}},
    "l8": {"component": "mlp_activation", "layers": 8, "select": {"intermediate_size": [1434]}},
    "l10": {"component": "mlp_activation", "layers": 10, "select": {"intermediate_size": [1793]}},
    "head": {"component": "lm_head_output"}
  },
  "steps": {
    "zeroed": {"kind": "forward", "input": "prompts.input", "interventions": {
      "z5": {"site": "l5", "pos": -1, "ops": [{"op": "scale", "factor": {"sweep": [0.0, 1.0], "label": "keep"}}]},
      "z6": {"site": "l6", "pos": -1, "ops": [{"op": "scale", "factor": {"sweep": [0.0, 1.0], "label": "keep"}}]},
      "z7": {"site": "l7", "pos": -1, "ops": [{"op": "scale", "factor": {"sweep": [0.0, 1.0], "label": "keep"}}]},
      "z8": {"site": "l8", "pos": -1, "ops": [{"op": "scale", "factor": {"sweep": [0.0, 1.0], "label": "keep"}}]},
      "z10": {"site": "l10", "pos": -1, "ops": [{"op": "scale", "factor": {"sweep": [0.0, 1.0], "label": "keep"}}]},
      "p": {"site": "head", "pos": -1, "ops": [{"op": "token_prob", "token": "prompts.answer"}]}}}
  }
}
```

```text
$ nnspec table out/zero_top zeroed.p
at            ...  value
{"keep":0.0}  ...  0.027501443400979042
{"keep":1.0}  ...  0.06997019797563553
```

- **The null equals the clean probability to the bit**, and zeroing the five takes it from 0.0700 to 0.0275.
- **The first-order drops sum to 0.0617 against a real drop of 0.0425**: the estimate ranks, and overshoots.
- **Neurons drawn at random, the same count per layer, are the control** that says the top set is special; it
  is the `nnspec-ablation` skill's. When the sets span several layers, write one document per random draw: a
  shared sweep label needs identical values, so per-layer sites of different sets cannot be one sweep.
