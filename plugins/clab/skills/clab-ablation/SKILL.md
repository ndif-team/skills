---
name: clab-ablation
description: Measure what a component contributes by removing it in a clab document, scored against a clean forward on the same rows: a test of necessity with no counterfactual prompt. Use it to ask whether the model still works without a layer, an attention head, an MLP neuron or a token, or to knock out every head one at a time. Also for neurons zeroed or mean-ablated beside random ones, attention knockout of keys, expert knockout, forced routing and routing usage on a mixture of experts, and the hydra effect. Covers zero, mean, resample (`permute`), optimal-constant, direction, noise and clamp ablation, each with its null control. Copying a value from another prompt is clab-patching, and reading attention patterns or the logit lens is clab-lenses-and-attention. Trained masks, probes and SAEs are clab-trained-featurizers; a refusal or a null that fails is clab-debugging; the document format is the clab skill.
---

# Ablation in clab

Verified on clab `830182f` with nnterp `e5d0f8e`, nnsight 0.8.2 and transformers 5.20.2; the documents run on
the CPU on `openai-community/gpt2` (12 layers of 12 heads, 768 wide), the expert ones on
`hf-internal-testing/tiny-random-MixtralForCausalLM`, in seconds each.

An ablation is a write in one forward, scored against a clean forward over the same rows: what the score
loses is what the component contributed. The kinds differ only in what the write puts in the component's
place, and that choice decides what the number means. Sites, `pos`, entries, ops, sweeps and fits are the
[`clab` skill](../clab/SKILL.md)'s; this skill is the recipes and their nulls. Beyond the four below,
[constants, directions and writes](references/constants-directions-and-writes.md) has the optimal constant,
ragged means, directions, noise, clamp and holding a coordinate; [keys, experts and hydra](references/keys-experts-and-hydra.md)
has attention knockout, the hydra effect and mixtures of experts.

## What breaks

- **Zero ablation overstates.** Zeroing GPT-2's last-token residual stream after layer 3 takes the logit
  difference from 2.75 to -1.51; the mean over the rows leaves 2.82. A zero vector is a point no forward
  reaches. Report mean or resample ablation; keep zero as the extreme.
- **An entry without an edit in `ops` only reads** (the `clab` skill's list).
- **A small total effect can hide a large one.** Mean-ablating layer 9's attention at the last token costs
  0.11 of the logit difference, yet its direct effect falls by 2.18: layer 10 makes up 2.13. Measure the
  [hydra effect](references/keys-experts-and-hydra.md#the-hydra-effect) before calling a component unimportant.
- **`permute` pairs any two rows.** On these prompts the resample at layer 9 gives 1.22 at seed 0 and -0.41
  at seed 2, where four rows take a donor that names the answer in the other place. Sweep `seed`; a
  constraint on which rows pair is data, a donor prompt in a column read by a forward of its own.
- **Write `derangement: true` on every resample `permute`.** Without it seed 0 left row 1 its own value, a
  null inside the average.
- **The optimal constant fits the score it trains on**: held-out rows kept 0.65 of a clean 3.46 and their KL
  rose. Score it on held-out rows and with `kl` ([the recipe](references/constants-directions-and-writes.md#the-optimal-constant)).
- **`attention_probabilities` and `attention_scores` need eager attention**; `attention_premix` does not ([eager](../clab/references/sites-and-interventions.md#attention-needs-eager)).
- **Keep a swept edit and a read op on the same sweep in separate entries**: in one entry the first point's
  forward stands for every point, with no refusal ([`clab-debugging`'s symptoms](../clab-debugging/references/symptoms.md#sweeps)).

## What to put in its place

| kind | the write | removes | null, and what it equals | recipe |
|---|---|---|---|---|
| zero | `{"op": "swap", "with": 0.0}` | everything, and adds a value no forward produces | swap the clean read back: the clean run | [below](#zero-and-mean-ablation) |
| mean | `{"op": "swap", "with": "mean"}`, `mean` a `mean` step over a clean read | what varies across the rows | the same | [below](#zero-and-mean-ablation) |
| resample | `{"op": "swap", "with": "donor"}`, `donor` a `permute` step | this row's value, put another row's in | `"seed": null`: the clean run | [below](#resample-ablation) |
| optimal constant | `{"op": "apply", "featurizer": "fit.c"}`, `c` of kind `constant` | this row's value, put the one vector that best keeps a score in | `epochs: 0`: mean ablation | [reference](references/constants-directions-and-writes.md#the-optimal-constant) |
| direction | `{"op": "project_out", "direction": "d"}` | one direction, the rest kept | a random direction per layer | [reference](references/constants-directions-and-writes.md#a-direction-at-every-layer) |
| noise | `{"op": "noise", "scale": s, "seed": k}` | some signal, at a dose | `scale` 0: the clean run | [reference](references/constants-directions-and-writes.md#noise-and-clamp) |
| clamp | `{"op": "clamp", "min": a, "max": b}` | every value past a bound | bounds wider than the values: the clean run | [reference](references/constants-directions-and-writes.md#noise-and-clamp) |
| hold | `project_out` of a unit direction, then `add` of it at a value | a direction's coordinate, set to one value | a `dot` read after it: the value at every row | [reference](references/constants-directions-and-writes.md#holding-a-coordinate) |
| keys | `-1e9` at `attention_scores`, or 0 at `attention_probabilities` then `renormalize` with `"norm": "sum"` | where a query looks | a key the task does not read | [reference](references/keys-experts-and-hydra.md#knocking-out-keys) |
| expert | `scale` of `expert_weights` by a mask; `-1e9` at `router_logits` | an expert's share; or its tokens rerouted | the mask of an id no slot holds: KL 0 | [reference](references/keys-experts-and-hydra.md#expert-knockout) |

## Zero and mean ablation

The shape every recipe shares, on six IOI prompts GPT-2 answers: a clean forward that reads the site and
scores, a `mean` step over that read, and one ablated forward per value `fill` sweeps.

```json
{
  "description": "Zero and mean ablation of GPT-2's residual stream at the last token, after layers 3 and 9, on six IOI prompts. `fill` sweeps what the write puts there: 0.0, the mean over the rows, and the clean read itself (the null).",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"ioi": {"rows": [
    {"input": "When Mary and John went to the store, John gave a drink to", "answer": " Mary", "other": " John"},
    {"input": "When Tom and James went to the train station, Tom gave a ball to", "answer": " James", "other": " Tom"},
    {"input": "When Alice and Bob went to the office, Bob gave a book to", "answer": " Alice", "other": " Bob"},
    {"input": "When Sarah and David went to the swimming pool, Sarah gave a ring to", "answer": " David", "other": " Sarah"},
    {"input": "When Anna and Mark went to the beach, Mark gave a gift to", "answer": " Anna", "other": " Mark"},
    {"input": "When Lisa and Paul went to the movie theater, Lisa gave a letter to", "answer": " Paul", "other": " Lisa"}
  ]}},
  "sites": {
    "resid": {"component": "layer_output", "layers": {"sweep": [3, 9], "label": "layer"}},
    "head": {"component": "lm_head_output"}
  },
  "interventions": {"score": {"ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "ioi.answer", "b": "ioi.other"}]}}},
  "steps": {
    "clean": {"kind": "forward", "input": "ioi.input", "interventions": ["score", {"acts": {"site": "resid", "pos": -1}}]},
    "mean": {"kind": "op", "op": "mean", "of": "clean.acts"},
    "ablated": {"kind": "forward", "input": "ioi.input", "interventions": [
      {"abl": {"site": "resid", "pos": -1, "ops": [{"op": "swap", "with": {"sweep": [0.0, "mean", "clean.acts"], "label": "fill"}}]}},
      "score"]},
    "clean_ld": {"kind": "op", "op": "mean", "of": "clean.ld"},
    "ablated_ld": {"kind": "op", "op": "mean", "of": "ablated.ld"}
  },
  "saves": ["clean.ld", "clean_ld", "ablated_ld"]
}
```

```text
$ clab run documents/zero_vs_mean.json
$ clab table out/zero_vs_mean clean_ld ablated_ld
at                               clean_ld            ablated_ld
{"layer":3,"fill":0.0}           2.7463290691375732  -1.5115877389907837
{"layer":3,"fill":"mean"}        2.7463290691375732  2.8224666118621826
{"layer":3,"fill":"clean.acts"}  2.7463290691375732  2.7463290691375732
{"layer":9,"fill":0.0}           2.7463290691375732  0.2309519499540329
{"layer":9,"fill":"mean"}        2.7463290691375732  0.7644971013069153
{"layer":9,"fill":"clean.acts"}  2.7463290691375732  2.7463290691375732
```

- **The null is the third `fill`**: writing back what was read equals `clean_ld` to the bit. If it does
  not, the write is not where the read was.
- **Read the two kinds together.** After layer 3 the zero says the last token carries the task and the mean
  says nothing that varies across the rows is there yet; after layer 9 both say it has arrived. Sweep
  `layers` over every layer to see where.
- **`mean` with no `over` removes the rows and the positions**: one 768-wide vector per `layer` point,
  broadcast over every row and position it is written at. A mean per position is `"over": ["rows"]`; it
  keeps `pos`, so it is written at the static width it was read at, as a resample donor is.
- A sweep may hold numbers and names alike, and `score` is a bundle of entries each forward names. Means over
  [a ragged scope or from another dataset](references/constants-directions-and-writes.md#means-over-a-ragged-scope-or-another-dataset)
  are in the reference; a harvest another document saved comes in by a [`load` step](../clab-trained-featurizers/references/harvest-and-bundles.md#cached-harvests), and a `mean` of it needs no forward.

Ops used: `swap`, `mean`, `logit_diff`; their lists are `clab describe ops`.

## Resample ablation

Each row takes another row's value at the site, so the write stays on the distribution and removes what is
this input's own. `permute` reorders the rows of a read the document already has: row i holds row π(i)'s,
and the donor costs no forward.

```json
{
  "description": "Resample ablation of GPT-2's residual stream at the last token, after layers 3 and 9: each row takes another row's value (a derangement, three seeds). `identity` is the same permute under seed null, whose swap is the null.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"ioi": {"rows": [
    {"input": "When Mary and John went to the store, John gave a drink to", "answer": " Mary", "other": " John"},
    {"input": "When Tom and James went to the train station, Tom gave a ball to", "answer": " James", "other": " Tom"},
    {"input": "When Alice and Bob went to the office, Bob gave a book to", "answer": " Alice", "other": " Bob"},
    {"input": "When Sarah and David went to the swimming pool, Sarah gave a ring to", "answer": " David", "other": " Sarah"},
    {"input": "When Anna and Mark went to the beach, Mark gave a gift to", "answer": " Anna", "other": " Mark"},
    {"input": "When Lisa and Paul went to the movie theater, Lisa gave a letter to", "answer": " Paul", "other": " Lisa"}
  ]}},
  "sites": {
    "resid": {"component": "layer_output", "layers": {"sweep": [3, 9], "label": "layer"}},
    "head": {"component": "lm_head_output"}
  },
  "interventions": {"score": {"ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "ioi.answer", "b": "ioi.other"}]}}},
  "steps": {
    "clean": {"kind": "forward", "input": "ioi.input", "interventions": ["score", {"acts": {"site": "resid", "pos": -1}}]},
    "donor": {"kind": "op", "op": "permute", "of": "clean.acts", "seed": {"sweep": [0, 1, 2], "label": "seed"}, "derangement": true},
    "identity": {"kind": "op", "op": "permute", "of": "clean.acts", "seed": null},
    "resampled": {"kind": "forward", "input": "ioi.input", "interventions": [
      {"abl": {"site": "resid", "pos": -1, "ops": [{"op": "swap", "with": "donor"}]}}, "score"]},
    "null": {"kind": "forward", "input": "ioi.input", "interventions": [
      {"abl": {"site": "resid", "pos": -1, "ops": [{"op": "swap", "with": "identity"}]}}, "score"]},
    "clean_ld": {"kind": "op", "op": "mean", "of": "clean.ld"},
    "resample_ld": {"kind": "op", "op": "mean", "of": "resampled.ld"},
    "null_ld": {"kind": "op", "op": "mean", "of": "null.ld"}
  },
  "saves": ["clean_ld", "resample_ld", "null_ld"]
}
```

```text
$ clab table out/resample clean_ld resample_ld null_ld
at                    clean_ld            resample_ld           null_ld
{"layer":3,"seed":0}  2.7463290691375732  2.84393572807312      2.7463290691375732
{"layer":3,"seed":1}  2.7463290691375732  2.7677524089813232    2.7463290691375732
{"layer":3,"seed":2}  2.7463290691375732  2.847675323486328     2.7463290691375732
{"layer":9,"seed":0}  2.7463290691375732  1.2174110412597656    2.7463290691375732
{"layer":9,"seed":1}  2.7463290691375732  1.274139404296875     2.7463290691375732
{"layer":9,"seed":2}  2.7463290691375732  -0.40846505761146545  2.7463290691375732
```

- **`"seed": null` is the identity, the null**: `null_ld` equals `clean_ld` to the bit at both layers.
- **Each seed's π is in the step's record**, `point.permutation` for each point of
  `Plan.load("out/resample").steps["donor"]`. Seed 0 drew `[5, 2, 4, 1, 0, 3]` and seed 2 `[1, 3, 0, 4, 5, 2]`:
  the even rows name the answer first and the odd rows second, and seed 2 crosses four rows between them.
  Below zero, the model follows the donor's template and picks the other name.
- **A donor needs a static width and two rows.** Every token of 14-token prompts is `{"index": "0:14"}`. An
  open `{"index": ":"}` or no `pos` is refused even on prompts of one length (`'donor' has no width compile
  can fix`), and so is a bare `"0:14"`: a slice goes in an object.

## Heads one at a time

A head's result is `attention_premix` with `select` on `num_heads`; sweeping `layers` and the head under
two labels makes every head a point and a forward. No eager attention is needed.

```json
{
  "description": "Every head of GPT-2's layers 9 to 11 knocked out one at a time at the last token, by zero and by its mean over the rows: one point per layer and head.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"ioi": {"rows": [
    {"input": "When Mary and John went to the store, John gave a drink to", "answer": " Mary", "other": " John"},
    {"input": "When Tom and James went to the train station, Tom gave a ball to", "answer": " James", "other": " Tom"},
    {"input": "When Alice and Bob went to the office, Bob gave a book to", "answer": " Alice", "other": " Bob"},
    {"input": "When Sarah and David went to the swimming pool, Sarah gave a ring to", "answer": " David", "other": " Sarah"},
    {"input": "When Anna and Mark went to the beach, Mark gave a gift to", "answer": " Anna", "other": " Mark"},
    {"input": "When Lisa and Paul went to the movie theater, Lisa gave a letter to", "answer": " Paul", "other": " Lisa"}
  ]}},
  "sites": {
    "one_head": {"component": "attention_premix", "layers": {"sweep": [9, 10, 11], "label": "layer"},
                 "select": {"num_heads": {"sweep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "label": "head"}}},
    "head": {"component": "lm_head_output"}
  },
  "interventions": {"score": {"ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "ioi.answer", "b": "ioi.other"}]}}},
  "steps": {
    "clean": {"kind": "forward", "input": "ioi.input", "interventions": ["score", {"z": {"site": "one_head", "pos": -1}}]},
    "mean": {"kind": "op", "op": "mean", "of": "clean.z"},
    "zeroed": {"kind": "forward", "input": "ioi.input", "interventions": [
      {"abl": {"site": "one_head", "pos": -1, "ops": [{"op": "swap", "with": 0.0}]}}, "score"]},
    "meaned": {"kind": "forward", "input": "ioi.input", "interventions": [
      {"abl": {"site": "one_head", "pos": -1, "ops": [{"op": "swap", "with": "mean"}]}}, "score"]},
    "clean_ld": {"kind": "op", "op": "mean", "of": "clean.ld"},
    "zero_ld": {"kind": "op", "op": "mean", "of": "zeroed.ld"},
    "mean_ld": {"kind": "op", "op": "mean", "of": "meaned.ld"}
  },
  "saves": ["clean_ld", "zero_ld", "mean_ld"]
}
```

```text
$ clab table out/head_knockout clean_ld zero_ld mean_ld
at                      clean_ld            zero_ld             mean_ld
{"layer":9,"head":9}    2.7463290691375732  2.678971529006958   2.6351585388183594
{"layer":10,"head":0}   2.7463290691375732  2.4221343994140625  2.4627342224121094
{"layer":10,"head":5}   2.7463290691375732  2.7439472675323486  2.744709014892578
{"layer":10,"head":7}   2.7463290691375732  3.5669047832489014  3.3799235820770264
{"layer":11,"head":0}   2.7463290691375732  2.5007336139678955  2.7296485900878906
{"layer":11,"head":10}  2.7463290691375732  3.357891082763672   3.0663044452667236
```

- **The control is a head at the same depth that the task does not use**: 10.5 moves the logit difference
  by -0.002 under either fill. An early head is a poor control for a mean over positions, which replaces
  what each position held.
- **A head can work against the answer**: knocking out 10.7 or 11.10 raises it. Zero still overstates:
  11.0's zero costs 0.25 and its mean 0.02.
- **The mean is per head**: `clean.z` is read at the swept site, so each point writes its own head's mean.
  A head set at once is a list, `"select": {"num_heads": [6, 9]}`, sorted, since the list is positional.
- **Resample or a constant on one head** take the same site; the constant's `init` must reduce the head axis
  ([the optimal constant](references/constants-directions-and-writes.md#the-optimal-constant)).

## Neurons

A neuron is one index of `mlp_activation`'s `intermediate_size`. Zero or mean-ablate a `select`ed set
beside random sets of its size from its layer: the sets are one sweep, crossed by `fill`.

```json
{
  "description": "Zero and mean ablation of four MLP neurons of GPT-2's layer 6 at the last token, beside three random sets of four from the same layer. The four rank first in the layer by activation times gradient of the logit difference on these six IOI prompts. `fill` sweeps 0.0, the mean over the rows, and the clean read (the null).",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"ioi": {"rows": [
    {"input": "When Mary and John went to the store, John gave a drink to", "answer": " Mary", "other": " John"},
    {"input": "When Tom and James went to the train station, Tom gave a ball to", "answer": " James", "other": " Tom"},
    {"input": "When Alice and Bob went to the office, Bob gave a book to", "answer": " Alice", "other": " Bob"},
    {"input": "When Sarah and David went to the swimming pool, Sarah gave a ring to", "answer": " David", "other": " Sarah"},
    {"input": "When Anna and Mark went to the beach, Mark gave a gift to", "answer": " Anna", "other": " Mark"},
    {"input": "When Lisa and Paul went to the movie theater, Lisa gave a letter to", "answer": " Paul", "other": " Lisa"}
  ]}},
  "sites": {
    "neurons": {"component": "mlp_activation", "layers": 6, "select": {"intermediate_size": {"sweep": [
      [1645, 1811, 2314, 2864], [165, 1060, 1577, 1723], [258, 550, 1044, 2334], [231, 347, 375, 1478]], "label": "set"}}},
    "head": {"component": "lm_head_output"}
  },
  "interventions": {"score": {"ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "ioi.answer", "b": "ioi.other"}]}}},
  "steps": {
    "clean": {"kind": "forward", "input": "ioi.input", "interventions": ["score", {"n": {"site": "neurons", "pos": -1}}]},
    "mean": {"kind": "op", "op": "mean", "of": "clean.n"},
    "ablated": {"kind": "forward", "input": "ioi.input", "interventions": [
      {"abl": {"site": "neurons", "pos": -1, "ops": [{"op": "swap", "with": {"sweep": [0.0, "mean", "clean.n"], "label": "fill"}}]}},
      "score"]},
    "clean_ld": {"kind": "op", "op": "mean", "of": "clean.ld"},
    "ablated_ld": {"kind": "op", "op": "mean", "of": "ablated.ld"}
  },
  "saves": ["clean.n", "clean_ld", "ablated_ld"]
}
```

```text
$ clab table out/neuron_ablation clean_ld ablated_ld
at                                              clean_ld            ablated_ld
{"set":[1645,1811,2314,2864],"fill":0.0}        2.7463290691375732  2.1083984375
{"set":[1645,1811,2314,2864],"fill":"mean"}     2.7463290691375732  2.749048948287964
{"set":[1645,1811,2314,2864],"fill":"clean.n"}  2.7463290691375732  2.7463290691375732
{"set":[165,1060,1577,1723],"fill":0.0}         2.7463290691375732  2.74395751953125
{"set":[258,550,1044,2334],"fill":0.0}          2.7463290691375732  2.7366535663604736
{"set":[231,347,375,1478],"fill":0.0}           2.7463290691375732  2.748163938522339
```

- **The null is the third `fill`**: writing `clean.n` back equals `clean_ld` to the bit at every set.
- **The four are on for every row**, between 0.5 and 2.6 (`clab table out/neuron_ablation clean.n`). Zeroing
  them costs 0.64; their mean costs nothing (+0.003): nothing that varies across these rows passes there.
- **Activation × gradient estimates the zero ablation**, so it ranks always-on neurons first; the gradient
  times the activation less its mean estimates the mean ablation (the `clab-patching` skill's shortcut).
- **The control is the random sets**: at most 0.01 under zero, 0.002 under the mean. A `select` holds fixed
  ids, so draw the sets outside the document and write them in sorted.
- **Zero is not a resting neuron's value**: GELU's floor is -0.17, and seven of the twelve random neurons
  read below -0.1 on every row.
- **Neurons in several layers need a site per layer**: a `select` list applies at every layer of its site.
- **On a gated MLP** `mlp_activation` is the gate alone; mean-ablate `mlp_neurons`, what the down projection sums.

## Which skill for the neighbouring questions

- The format, running and reading back: the [`clab` skill](../clab/SKILL.md), with its
  [sweeps](../clab/references/sweeps.md) and [sites and interventions](../clab/references/sites-and-interventions.md).
- Copying a value in from a counterfactual prompt, path patching, attribution: the `clab-patching` skill.
- Where a head looks, attention patterns and stripes, the logit lens: the `clab-lenses-and-attention` skill.
- A trained mask over heads, DAS, a probe's direction, SAE feature ablation: the `clab-trained-featurizers` skill.
- A null that does not equal the clean run, two runs that differ, a refusal: the `clab-debugging` skill.
