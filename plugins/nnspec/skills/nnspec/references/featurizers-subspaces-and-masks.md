# Subspaces and masks: DAS, its controls, DBM, head masking

A `subspace` is a rank-`k` share of a value and a `gate` a mask over its units, or over the heads of every
layer. A fit trains either one to carry out an interchange, and what it keeps says which directions, units or
heads carry a variable. The fit step and bundles are in [featurizers](featurizers.md).

## What breaks

- **A trained subspace is evidence only beside controls of the same rank**, on the same rows, scored by the
  same op: a drawn subspace and a PCA of `k` 4 against a trained `k` 4, never another rank.
- **What a PCA reads decides how strong a control it is.** A PCA of the day token alone swaps the day on 7 of
  7 pairs below; a PCA of every token after the first on 0 of 7. State which one a comparison used.
- **Accuracy ties hide differences.** Where two featurizers both reach 7 of 7, compare the held-out cross
  entropy of the counterfactual answer.
- **The learning rate decides whether a rotation trains at all.** With `"lr": {"sweep": [0.0001, 0.001, 0.01,
  0.1]}`, `das.json` below swaps the day on 0, 7, 2 and 0 of its 7 held-out pairs: sweep `lr` before reading a
  flat curve.
- **A gate's draw is every unit on**, so its null is the full-vector interchange, not the clean run.
- **A gate's L1 weight scales with the number of units.** `<gate>.mask` is the mean of the soft mask, so a
  unit kept costs the weight over the unit count. Move a sweep to another site by that ratio.

## DAS: how many dimensions carry the day

GPT-2 answers "If today is Monday, tomorrow is" with " Tuesday", for every day. DAS trains a rank-4 subspace
of layer 5's output at the day token, so that swapping those 4 of 768 coordinates in from another day's prompt
gives that day's successor. Saved as `documents/das.json`:

```json
{
  "description": "DAS on GPT-2: a rank-4 subspace of layer 5's output at the day token that carries the day. Trained on 21 pairs, scored on 7 held-out pairs through fit.rot, and again through the bare rot, the fit's starting draw.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {
    "train": {"rows": [
      {"input": "If today is Sunday, tomorrow is", "counterfactual": "If today is Monday, tomorrow is", "cf_answer": " Tuesday"},
      {"input": "If today is Tuesday, tomorrow is", "counterfactual": "If today is Sunday, tomorrow is", "cf_answer": " Monday"},
      {"input": "If today is Saturday, tomorrow is", "counterfactual": "If today is Thursday, tomorrow is", "cf_answer": " Friday"},
      {"input": "If today is Monday, tomorrow is", "counterfactual": "If today is Tuesday, tomorrow is", "cf_answer": " Wednesday"},
      {"input": "If today is Sunday, tomorrow is", "counterfactual": "If today is Tuesday, tomorrow is", "cf_answer": " Wednesday"},
      {"input": "If today is Thursday, tomorrow is", "counterfactual": "If today is Wednesday, tomorrow is", "cf_answer": " Thursday"},
      {"input": "If today is Wednesday, tomorrow is", "counterfactual": "If today is Thursday, tomorrow is", "cf_answer": " Friday"},
      {"input": "If today is Tuesday, tomorrow is", "counterfactual": "If today is Wednesday, tomorrow is", "cf_answer": " Thursday"},
      {"input": "If today is Thursday, tomorrow is", "counterfactual": "If today is Monday, tomorrow is", "cf_answer": " Tuesday"},
      {"input": "If today is Thursday, tomorrow is", "counterfactual": "If today is Friday, tomorrow is", "cf_answer": " Saturday"},
      {"input": "If today is Wednesday, tomorrow is", "counterfactual": "If today is Monday, tomorrow is", "cf_answer": " Tuesday"},
      {"input": "If today is Tuesday, tomorrow is", "counterfactual": "If today is Monday, tomorrow is", "cf_answer": " Tuesday"},
      {"input": "If today is Monday, tomorrow is", "counterfactual": "If today is Sunday, tomorrow is", "cf_answer": " Monday"},
      {"input": "If today is Monday, tomorrow is", "counterfactual": "If today is Wednesday, tomorrow is", "cf_answer": " Thursday"},
      {"input": "If today is Thursday, tomorrow is", "counterfactual": "If today is Sunday, tomorrow is", "cf_answer": " Monday"},
      {"input": "If today is Wednesday, tomorrow is", "counterfactual": "If today is Friday, tomorrow is", "cf_answer": " Saturday"},
      {"input": "If today is Tuesday, tomorrow is", "counterfactual": "If today is Saturday, tomorrow is", "cf_answer": " Sunday"},
      {"input": "If today is Sunday, tomorrow is", "counterfactual": "If today is Thursday, tomorrow is", "cf_answer": " Friday"},
      {"input": "If today is Saturday, tomorrow is", "counterfactual": "If today is Sunday, tomorrow is", "cf_answer": " Monday"},
      {"input": "If today is Friday, tomorrow is", "counterfactual": "If today is Saturday, tomorrow is", "cf_answer": " Sunday"},
      {"input": "If today is Friday, tomorrow is", "counterfactual": "If today is Thursday, tomorrow is", "cf_answer": " Friday"}
    ]},
    "test": {"rows": [
      {"input": "If today is Sunday, tomorrow is", "counterfactual": "If today is Friday, tomorrow is", "cf_answer": " Saturday"},
      {"input": "If today is Wednesday, tomorrow is", "counterfactual": "If today is Sunday, tomorrow is", "cf_answer": " Monday"},
      {"input": "If today is Tuesday, tomorrow is", "counterfactual": "If today is Thursday, tomorrow is", "cf_answer": " Friday"},
      {"input": "If today is Saturday, tomorrow is", "counterfactual": "If today is Friday, tomorrow is", "cf_answer": " Saturday"},
      {"input": "If today is Monday, tomorrow is", "counterfactual": "If today is Friday, tomorrow is", "cf_answer": " Saturday"},
      {"input": "If today is Sunday, tomorrow is", "counterfactual": "If today is Saturday, tomorrow is", "cf_answer": " Sunday"},
      {"input": "If today is Friday, tomorrow is", "counterfactual": "If today is Sunday, tomorrow is", "cf_answer": " Monday"}
    ]}
  },
  "sites": {"target": {"component": "layer_output", "layers": 5}, "head": {"component": "lm_head_output"}},
  "featurizers": {"rot": {"kind": "subspace", "k": 4, "seed": 0}},
  "steps": {
    "counterfactual": {"kind": "forward", "input": "train.counterfactual",
                       "interventions": {"h": {"site": "target", "pos": -4}}},
    "fit": {
      "kind": "fit", "train": ["rot"], "objective": {"ce": 1.0},
      "epochs": 10, "batch_size": 7, "seed": 0, "optimizer": {"name": "adam", "lr": 0.001},
      "eval": {"train": {}},
      "steps": {
        "v_cf": {"kind": "op", "op": "encode", "featurizer": "fit.rot", "of": "counterfactual.h"},
        "patched": {"kind": "forward", "input": "train.input", "interventions": {
          "patch": {"site": "target", "pos": -4, "ops": [{"op": "encode", "featurizer": "fit.rot"},
                                                         {"op": "swap", "with": "v_cf"},
                                                         {"op": "decode", "featurizer": "fit.rot"}]},
          "logits": {"site": "head", "pos": -1}}},
        "ce": {"kind": "op", "op": "cross_entropy", "of": "patched.logits", "target": "train.cf_answer"},
        "hit": {"kind": "op", "op": "match", "of": "patched.logits", "expected": "train.cf_answer"},
        "acc": {"kind": "op", "op": "mean", "of": "hit"}
      }
    },
    "held_cf": {"kind": "forward", "input": "test.counterfactual", "interventions": {
      "v_cf": {"site": "target", "pos": -4, "ops": [{"op": "encode", "featurizer": "fit.rot"}]}}},
    "held": {"kind": "forward", "input": "test.input", "interventions": {
      "patch": {"site": "target", "pos": -4, "ops": [{"op": "encode", "featurizer": "fit.rot"},
                                                     {"op": "swap", "with": "held_cf.v_cf"},
                                                     {"op": "decode", "featurizer": "fit.rot"}]},
      "logits": {"site": "head", "pos": -1}}},
    "held_ce": {"kind": "op", "op": "cross_entropy", "of": "held.logits", "target": "test.cf_answer"},
    "held_hit": {"kind": "op", "op": "match", "of": "held.logits", "expected": "test.cf_answer"},
    "held_acc": {"kind": "op", "op": "mean", "of": "held_hit"},
    "ctl_cf": {"kind": "forward", "input": "test.counterfactual", "interventions": {
      "v_cf": {"site": "target", "pos": -4, "ops": [{"op": "encode", "featurizer": "rot"}]}}},
    "ctl": {"kind": "forward", "input": "test.input", "interventions": {
      "patch": {"site": "target", "pos": -4, "ops": [{"op": "encode", "featurizer": "rot"},
                                                     {"op": "swap", "with": "ctl_cf.v_cf"},
                                                     {"op": "decode", "featurizer": "rot"}]},
      "logits": {"site": "head", "pos": -1}}},
    "ctl_hit": {"kind": "op", "op": "match", "of": "ctl.logits", "expected": "test.cf_answer"},
    "ctl_acc": {"kind": "op", "op": "mean", "of": "ctl_hit"}
  },
  "saves": ["held_acc", "held_ce", "ctl_acc", "fit.history.acc", "fit.rot"]
}
```

- **`v_cf` encodes the harvest in the subspace being trained**, and the bracket swaps those 4 coordinates into
  the base prompt's. `decode` writes back only the change, so the other 764 dimensions stay the base's.
- **`ce` is the objective; `acc`, the interchange accuracy, is reported.**
- **`pos: -4` is the day token on every row**, since every prompt is the same 7 tokens. `nnspec show out/das`
  lists `[3] on all 21 rows` for `counterfactual.h`.
- **`rot` carries its own `seed`**, so the bare `rot` after the fit is exactly the fit's starting draw, and
  the `ctl_*` steps are this document's null.

```text
$ nnspec table out/das held_acc
value
1.0
$ nnspec table out/das ctl_acc
value
0.0
$ nnspec table out/das fit.history.acc
epochs  value
0       0.0476190485060215
3       0.3333333432674408
6       1.0
9       0.9523809552192688
```

The trained subspace swaps the day on all 7 held-out pairs, and its draw on none.

### Load it in a second document

The run writes `out/das/fit.rot.safetensors`, one tensor `basis` `[768, 4]` and a header.
`documents/das_apply.json` keeps `das.json`'s `model`, `sites`, `test` rows and held-out steps (`held_cf`,
`held`, `held_ce`, `held_hit`, `held_acc`), with `rot` for `fit.rot` and this declaration; `k` comes from the
header:

```json
{"featurizers": {"rot": {"kind": "subspace", "file_path": "out/das/fit.rot.safetensors"}}}
```

Its `held_ce` equals `das.json`'s to the bit, row for row, at one batch size and thread count.

## The rank curve

`das.json` with `"k": {"sweep": [1, 2, 4, 8]}` on `rot` fits four rotations; the bare `rot`, swept with it, is
a draw at each rank:

```text
$ nnspec table out/das_rank held_acc
at       value
{"k":1}  0.1428571492433548
{"k":2}  0.4285714328289032
{"k":4}  1.0
{"k":8}  1.0
```

`ctl_acc` is 0.0 at every rank: four dimensions carry the day at this site.

## Controls of matched rank

**Seeds.** `das.json` with `rot`'s seed swept, `"seed": {"sweep": [0, 1, 2]}`, fits three rotations, each from
its own draw. With `mean` steps over `held_ce` and over the draw's cross entropy:

```text
$ nnspec table out/das_seeds held_acc held_mean ctl_acc ctl_mean
at          held_acc            held_mean           ctl_acc  ctl_mean
{"seed":0}  1.0                 1.093050479888916   0.0      2.4739482402801514
{"seed":1}  0.8571428656578064  1.3086893558502197  0.0      2.471331834793091
{"seed":2}  1.0                 1.0828101634979248  0.0      2.447234630584717
```

The spread across draws is the null every trained point clears by more than a nat.

**Random subspaces without a fit.** A declared subspace with no `file_path` is drawn from its seed when the
run starts. `das_apply.json` with `{"kind": "subspace", "k": 4, "seed": {"sweep": [0, 1, 2]}}` in place of
the loaded `rot` scores three draws with no training: 0 of 7 at every seed, mean cross entropy
2.4739482402801514, 2.471331834793091 and 2.447234630584717. Seed 0 is `das.json`'s draw: its bare `rot`, its
`"epochs": 0` point and this declaration give the same number.

**Principal directions.** The `pca` op reduces a harvest to its top-`k` directions and the mean they are
centred on, saved as a bundle that a `pca` featurizer loads. `documents/pca_harvest.json` has `das.json`'s
`model`, a dataset `days` of the seven prompts `If today is <day>, tomorrow is`, and:

```json
{
  "sites": {"target": {"component": "layer_output", "layers": 5}},
  "steps": {
    "harvest": {"kind": "forward", "input": "days.input", "interventions": {"acts": {"site": "target", "pos": -4}}},
    "basis": {"kind": "op", "op": "pca", "of": "harvest.acts", "k": 4}
  },
  "saves": ["basis"]
}
```

```text
$ nnspec run documents/pca_harvest.json
pca_harvest on nnterp (batch size 8, device cpu, tp 1) -> out/pca_harvest
  basis.safetensors  basis  {"basis":[768,4],"mean":[768]}
```

A PCA needs more vectors than its `k`. Read every token after the first instead (`{"index": "1:"}`, 42
vectors) and the top 4 directions are the sentence's variance, not the day's: in the interchange below it
swaps 0 of 7.

### The trained rotation, the PCA and the draw in one document

An `encode`'s and a `decode`'s `featurizer` may be a sweep over featurizer names, so one document scores every
control by the same interchange. `documents/das_controls.json` is `das_apply.json` with these featurizers,
`{"sweep": ["rot", "pc", "draw"], "label": "space"}` as the `featurizer` of every `encode` and `decode`, and a
`mean` step over `held_ce`. It needs `out/das/` and `out/pca_harvest/` from the two runs above:

```json
{"featurizers": {
  "rot": {"kind": "subspace", "file_path": "out/das/fit.rot.safetensors"},
  "pc": {"kind": "pca", "k": 4, "file_path": "out/pca_harvest/basis.safetensors"},
  "draw": {"kind": "subspace", "k": 4, "seed": 0}}}
```

```text
$ nnspec table out/das_controls held_acc held_mean
at                held_acc  held_mean
{"space":"rot"}   1.0       1.093050479888916
{"space":"pc"}    1.0       1.4620352983474731
{"space":"draw"}  0.0       2.4739482402801514
```

The PCA of the day token swaps the day on 7 of 7, at a cross entropy above the trained rotation's; the draw
swaps none. With the PCA of every token after the first as `pc`, its row reads 0.0 and 2.454253911972046.
`held_cf` only encodes, so one forward serves all three points; `held` writes, and runs one each.

## DBM: which units carry the day

A `gate` learns a mask over the units of a value: soft, σ(θ/T), while the fit trains it, and hard, θ > 0,
whenever it is scored. `documents/dbm.json` is `das.json` with these changes:

- the featurizer `"mask": {"kind": "gate", "temperature": {"start": 1.0, "end": 0.05}}` in place of `rot`, and
  `mask` and `fit.mask` wherever `das.json` writes `rot` and `fit.rot`;
- in the fit, `"objective": {"ce": 1.0, "mask.mask": {"sweep": [1.0, 2.0, 3.0, 5.0]}}`, `"epochs": 20` and
  `"lr": 0.1`;
- `"saves": ["held_acc", "ctl_acc", "fit.mask"]`.

`mask.mask` is the gate's own scalar, the mean of its soft mask: the pressure toward keeping few units. The
temperature falls linearly over the epochs. The units a fit kept are where its saved `theta` is positive. A
bundle does not table, so read it in Python:

```python
from nnspec import Plan

plan = Plan.load("out/dbm")
for weight, gate in plan.results["fit.mask"].items():
    print(weight, int((gate["theta"] > 0).sum()), "of", gate["theta"].numel())
```

| `mask.mask` weight | units kept of 768 | `held_acc` |
|---|---|---|
| 1.0 | 330 | 1.0 |
| 2.0 | 169 | 0.8571428656578064 |
| 3.0 | 98 | 0.4285714328289032 |
| 5.0 | 30 | 0.0 |

`ctl_acc`, the bare `mask` with every unit on, is 1.0: the full-vector interchange. Four directions carry the
day where no small set of units does, so DAS and DBM answer different questions about one site.

## Head masking

Head patching moves one head at a time. Head masking trains a gate with one unit per head of every layer, all
at once: a head that is off takes its value from a corrupt prompt, the objective keeps the clean logit
difference, and an L1 weight turns heads off. The clean prompts are GPT-2's indirect-object task, and each
corrupt prompt replaces the second subject with a third name. Saved as `documents/head_gates.json`:

```json
{
  "description": "Head masking on GPT-2's indirect-object task: one gate unit per head of every layer at attention_premix. A head that is off takes the corrupt prompt's value at every position; the objective keeps the clean logit difference (IO less S) while the swept L1 weight turns heads off.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"input": "When Mary and John went to the store, John gave a drink to", "corrupt": "When Mary and John went to the store, Tom gave a drink to", "io": " Mary", "s": " John"},
    {"input": "When Sarah and David went to the store, David gave a drink to", "corrupt": "When Sarah and David went to the store, Paul gave a drink to", "io": " Sarah", "s": " David"},
    {"input": "When Lisa and Mark went to the store, Mark gave a drink to", "corrupt": "When Lisa and Mark went to the store, Jack gave a drink to", "io": " Lisa", "s": " Mark"},
    {"input": "When Anna and Peter went to the store, Peter gave a drink to", "corrupt": "When Anna and Peter went to the store, Kate gave a drink to", "io": " Anna", "s": " Peter"},
    {"input": "When John and Mary went to the store, Mary gave a drink to", "corrupt": "When John and Mary went to the store, Emma gave a drink to", "io": " John", "s": " Mary"},
    {"input": "When David and Sarah went to the store, Sarah gave a drink to", "corrupt": "When David and Sarah went to the store, Mark gave a drink to", "io": " David", "s": " Sarah"},
    {"input": "When Mark and Lisa went to the store, Lisa gave a drink to", "corrupt": "When Mark and Lisa went to the store, Anna gave a drink to", "io": " Mark", "s": " Lisa"},
    {"input": "When Peter and Anna went to the store, Anna gave a drink to", "corrupt": "When Peter and Anna went to the store, Tom gave a drink to", "io": " Peter", "s": " Anna"}
  ]}},
  "sites": {"premix": {"component": "attention_premix", "layers": ":"}, "head": {"component": "lm_head_output"}},
  "featurizers": {"heads": {"kind": "gate", "over": ["layers", "num_heads"], "temperature": {"start": 1.0, "end": 0.05}}},
  "steps": {
    "corrupt": {"kind": "forward", "input": "pairs.corrupt", "interventions": {"z": {"site": "premix", "pos": {"index": "0:14"}}}},
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": {"logits": {"site": "head", "pos": -1}}},
    "clean_ld": {"kind": "op", "op": "logit_diff", "of": "clean.logits", "a": "pairs.io", "b": "pairs.s"},
    "fit": {
      "kind": "fit", "train": ["heads"], "objective": {"sq": 1.0, "heads.mask": {"sweep": [3.0, 10.0, 30.0, 100.0]}},
      "epochs": 20, "batch_size": 4, "seed": 0, "optimizer": {"name": "adam", "lr": 0.1},
      "eval": {"train": {}},
      "steps": {
        "run": {"kind": "forward", "input": "pairs.input", "interventions": {
          "gated": {"site": "premix", "pos": {"index": "0:14"},
                    "ops": [{"op": "apply", "featurizer": "fit.heads", "with": "corrupt.z"}]},
          "logits": {"site": "head", "pos": -1}}},
        "ld": {"kind": "op", "op": "logit_diff", "of": "run.logits", "a": "pairs.io", "b": "pairs.s"},
        "diff": {"kind": "op", "op": "sub", "of": "ld", "with": "clean_ld"},
        "sq": {"kind": "op", "op": "square", "of": "diff"},
        "mean_ld": {"kind": "op", "op": "mean", "of": "ld"}
      }
    },
    "clean_mean": {"kind": "op", "op": "mean", "of": "clean_ld"}
  },
  "saves": ["clean_mean", "fit.train.mean_ld", "fit.history.sq", "fit.heads"]
}
```

- **`"over": ["layers", "num_heads"]`** at `attention_premix` over every layer lays `theta` out `[12, 12]`, one
  unit per head, broadcast over `head_dim`. At layer L the write takes row L of the mask.
- **`apply` with `with`** writes `h·m + corrupt·(1−m)`: the clean value where a head is on and the corrupt one
  where it is off, each exactly.
- **The corrupt read and the write meet position for position**, at the static window `0:14`, on prompts that
  all tokenize to 14 tokens. On rows of different lengths read a span every row holds.
- **`sq`, the squared change from the clean logit difference, is the objective**; `mean_ld` is reported.

```text
$ nnspec table out/head_gates clean_mean
value
3.0798864364624023
$ nnspec table out/head_gates fit.train.mean_ld
at                    value
{"heads.mask":3.0}    3.1980886459350586
{"heads.mask":10.0}   2.7434635162353516
{"heads.mask":30.0}   2.5992813110351562
{"heads.mask":100.0}  1.6189403533935547
```

The heads each weight kept are where the saved `theta`, laid out `[layers, num_heads]`, is positive:

```python
from nnspec import Plan

plan = Plan.load("out/head_gates")
for weight, gate in plan.results["fit.heads"].items():
    kept = (gate["theta"] > 0).nonzero().tolist()
    print(weight, len(kept), [f"{layer}.{head}" for layer, head in kept])
```

```text
10.0 19 ['0.1', '0.9', '3.0', '3.6', '4.3', '5.5', '5.9', '6.0', '7.3', '7.9', '8.6', '8.10', '9.7', '9.9', '10.0', '10.6', '10.7', '10.10', '11.10']
30.0 14 ['0.1', '3.0', '4.3', '5.5', '5.9', '6.0', '7.3', '7.9', '8.6', '8.10', '9.7', '9.9', '10.0', '10.6']
100.0 8 ['0.1', '5.5', '7.3', '7.9', '8.10', '9.7', '9.9', '11.10']
```

Weight 3.0 keeps 42 heads. At 30.0, 14 of 144 heads hold the logit difference at 2.60 against the clean 3.08.

### Its nulls

- **The untrained gate is the clean run.** Every θ starts at +1, so with `"epochs": 0` the `train` eval's
  `fit.train.ld` equals `clean_ld` to the bit, row for row.
- **A head that is off writes the corrupt value exactly.** At a weight of 3000 every head is off: the saved
  `fit.train.run.gated` equals `corrupt.z` to the bit, and `fit.train.ld` the corrupt run's logit difference.
- **Every head off is the corrupt run only where the read position's token is shared.** These prompts differ
  at position 9 alone. Where prompts differ at the read position, its own embedding and MLPs still see the
  clean token.

### Choosing the weight

The objective is a mean over the training rows, so a mask can hold the mean while single rows move: save
`fit.train.ld` beside `clean_ld` and compare them row by row. Choose the weight on held-out prompts of the
same length, in steps after the fit: a harvest of their corrupt prompts at `premix`, a gated forward through
`fit.heads` with that harvest as `with`, and as a floor the same forward with a `swap` of the harvest at
every head. Fix the tolerance before the sweep, and take the largest weight that stays within it.
