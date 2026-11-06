# Attribution patching

Attribution patching estimates what patching would measure, to first order: the metric's gradient at a site
on the base run, times the counterfactual's value there less the base's, summed over the site's features. One
backward gives the gradient at every site of a forward, so a map over every head costs two forwards where
patching costs one per head. In clab the gradient is a read: an entry whose first op is `grad`. Rank with
attribution, then confirm what you report with real patching.

## What breaks

- **A `grad` of the wrong entry runs and gives a plausible map.** Carry the null: a second entry at the head
  with the metric's ops and `{"op": "mul", "with": 0}` appended, and a `grad` of it. Its gradient must be
  exactly zero; below it is 0.0 on every value of its `[12, 8, 1, 12, 64]` gradient.
- **`scale` 0 is not the null.** It is an edit, so after a metric it is refused:

  ```text
  clab: steps.base.interventions.zero.ops[1]: 'scale' is an edit, which writes back, and it follows 'logit_diff', which turned the value into ['rows', 'pos']: there is no live value to write; an edit comes before it, or put the transform in a later entry at this site
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
- **`of` names an entry of the same forward, bare, at a site the forward reaches later.** `grad` is refused in
  a `generate`, in a fit's body and under `tp`.
- **A bfloat16 model's gradient is bfloat16.** Recompute the components you report in float32 (the
  `clab-debugging` skill has the rule).

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
$ clab table out/attribution_heads g0_max self_max
g0_max  self_max
0.0     0.0

$ clab table out/attribution_heads map
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

A `grad` of the right entry with a wrong selection still gives numbers. Compare it once per new document with
a central difference: two forwards that move the site by plus and minus a step along the counterfactual's
difference. The step is swept under one label, so the nudges and the division zip. This fragment is the
`sites`, `interventions` and `steps` of `documents/gradient_check.json`, whose `model` and `data` are those of
the document above:

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
$ clab table out/gradient_check h_max ones_max
h_max                ones_max
0.24178534746170044  7.152557373046875e-07

$ clab table out/gradient_check gap_max
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
- **Pass where the smallest gap is a small fraction of the largest attribution**: 0.00024 against 0.242 here,
  a thousandth. Set the tolerance before the run.
- **A gap far above it that does not shrink with the step** is a `grad` of the wrong entry, or of the right
  entry at the wrong selection.
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

`map` is `[layers, pos]`; `clab table out/attribution_map map` prints a row per layer and position. Beside
the patching map's effect (its `ld_mean` less `clean_mean`) at the day's token and the last token:

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
