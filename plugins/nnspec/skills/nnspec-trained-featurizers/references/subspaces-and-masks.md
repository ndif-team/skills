# Subspaces and masks: DAS, its controls, DBM, head masking

Every document here builds on `das.json` in [SKILL.md](../SKILL.md): GPT-2, layer 5's output at the day token
(`pos: -4`), 21 training pairs and 7 held-out pairs of "If today is X, tomorrow is". Numbers are held-out
accuracy out of 7 unless a line says otherwise. The fit fields are the `clab` skill's
[featurizers-and-fit reference](../../clab/references/featurizers-and-fit.md); sweeps and their labels are its
[sweeps reference](../../clab/references/sweeps.md).

## What breaks here

- **A trained subspace is evidence only beside controls of the same rank**, on the same rows, scored by the
  same op. Compare `k: 4` with drawn and PCA subspaces of `k: 4`, never with another rank, in one document
  ([all three](#the-trained-rotation-the-pca-and-the-draw-in-one-document)).
- **What a PCA reads decides how strong a control it is.** A PCA of the day token alone swaps the day on 7/7;
  a PCA of every token after the first swaps it on 0/7. State which one a comparison used.
- **Accuracy ties hide differences.** Where two featurizers both reach 7/7, compare a continuous score: the
  held-out cross entropy of the counterfactual answer (`held_ce`).
- **A gate's draw is every unit on.** Its null is the full-vector interchange, not the clean run: DBM's
  `ctl_acc` is 7/7 at layer 5.
- **A head gate's corrupt read and its write need prompts of one length.** On these 14-token rows
  `{"index": ":"}` gives `{"index": "0:14"}`'s numbers to the bit; on rows of different lengths it is refused
  before any forward, so read a span every row holds.
- **The L1 weight scales with the number of units.** `mask.mask` is the mean of the soft mask, so a unit kept
  costs the weight over the unit count. Move a sweep to another model or site by that ratio.

## The rank curve

`das.json` with `"k": {"sweep": [1, 2, 4, 8]}` on `rot` and `"saves": ["held_acc", "ctl_acc"]` fits four
rotations; the bare `rot`, swept with it, is a draw at each rank:

```text
clab table out/das_rank held_acc
at       value
{"k":1}  0.1428571492433548
{"k":2}  0.4285714328289032
{"k":4}  1.0
{"k":8}  1.0
clab table out/das_rank ctl_acc
at       value
{"k":1}  0.0
{"k":2}  0.0
{"k":4}  0.0
{"k":8}  0.0
```

Four dimensions carry the day at this site; one moves 1 pair of 7. The swept fit's `fit.rot` file holds one
bundle per rank, so a document that loads a rotation needs one fit at the rank it keeps.

## Controls of matched rank

### Seeds

`das.json` with `rot`'s seed swept, `{"kind": "subspace", "k": 4, "seed": {"sweep": [0, 1, 2]}}`, fits three
rotations, each from its own draw. These three steps beside the held-out ones give a continuous score:

```json
{
  "held_mean": {"kind": "op", "op": "mean", "of": "held_ce"},
  "ctl_ce": {"kind": "op", "op": "cross_entropy", "of": "ctl.logits", "target": "test.cf_answer"},
  "ctl_mean": {"kind": "op", "op": "mean", "of": "ctl_ce"}
}
```

| seed | trained `held_acc` | trained `held_mean` | draw `ctl_acc` | draw `ctl_mean` |
|---|---|---|---|---|
| 0 | 1.0 | 1.093 | 0.0 | 2.474 |
| 1 | 0.857 | 1.309 | 0.0 | 2.471 |
| 2 | 1.0 | 1.083 | 0.0 | 2.447 |

The spread across draws, 2.447 to 2.474 nats, is the null every trained point clears by more than a nat.

### Random subspaces without a fit

A declared subspace with no `file_path` is drawn from its seed where the run starts. `das_apply.json` with
`{"kind": "subspace", "k": 4, "seed": {"sweep": [0, 1, 2]}}` in place of the loaded `rot` scores three draws
with no training at all: 0/7 at every seed, mean ce 2.474, 2.471 and 2.447. Seed 0 is `das.json`'s draw, the
one control of [What breaks](../SKILL.md#what-breaks).

### Principal directions

The `pca` op reduces a harvest to its top-`k` directions and the mean they are centred on, saved as a bundle
that a `pca` featurizer loads. This harvest reads the day token of the seven prompts:

```json
{
  "description": "The top-4 principal directions of GPT-2's layer 5 output at the day token, over the seven weekday prompts, saved as a bundle a pca featurizer loads.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"days": {"rows": [
    {"input": "If today is Monday, tomorrow is"}, {"input": "If today is Tuesday, tomorrow is"},
    {"input": "If today is Wednesday, tomorrow is"}, {"input": "If today is Thursday, tomorrow is"},
    {"input": "If today is Friday, tomorrow is"}, {"input": "If today is Saturday, tomorrow is"},
    {"input": "If today is Sunday, tomorrow is"}
  ]}},
  "sites": {"target": {"component": "layer_output", "layers": 5}},
  "steps": {
    "harvest": {"kind": "forward", "input": "days.input", "interventions": {"acts": {"site": "target", "pos": -4}}},
    "basis": {"kind": "op", "op": "pca", "of": "harvest.acts", "k": 4}
  },
  "saves": ["basis"]
}
```

```text
clab run documents/pca_harvest.json
pca_harvest on nnterp (batch size 8, device cpu, tp 1) -> out/pca_harvest
  basis.safetensors  basis  {"basis":[768,4],"mean":[768]}
```

- **A PCA needs more vectors than its `k`.** One position of four rows is refused before any forward
  (`pca of 4 vectors has no rank-4 basis; it needs more than 4`); read more rows or a range of positions.
- **Read every token instead** (`{"index": "1:"}`, 42 vectors) and the top 4 directions are the sentence's
  variance, not the day's: as `pc` in the interchange below it swaps 0/7, mean `ce` 2.454.
- **A `pca` is never drawn or trained.** It exists only as the bundle a `pca` op wrote.

### The trained rotation, the PCA and the draw in one document

An `encode`'s and a `decode`'s `featurizer` may be a sweep over featurizer names, so one document scores
every control by the same interchange. This one needs `out/das/fit.rot.safetensors` from `das.json` and
`out/pca_harvest/basis.safetensors` from the harvest above, under the corpus root:

<!-- test: skip -->
```json
{
  "description": "das.json's trained rotation, the top-4 principal directions of the day token and the rotation's starting draw, each scored by the same interchange on the 7 held-out pairs. One sweep over featurizer names, space, runs in the encode, the decode and the counterfactual's encode, so each point is one featurizer's interchange.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"test": {"rows": [
    {"input": "If today is Sunday, tomorrow is", "counterfactual": "If today is Friday, tomorrow is", "cf_answer": " Saturday"},
    {"input": "If today is Wednesday, tomorrow is", "counterfactual": "If today is Sunday, tomorrow is", "cf_answer": " Monday"},
    {"input": "If today is Tuesday, tomorrow is", "counterfactual": "If today is Thursday, tomorrow is", "cf_answer": " Friday"},
    {"input": "If today is Saturday, tomorrow is", "counterfactual": "If today is Friday, tomorrow is", "cf_answer": " Saturday"},
    {"input": "If today is Monday, tomorrow is", "counterfactual": "If today is Friday, tomorrow is", "cf_answer": " Saturday"},
    {"input": "If today is Sunday, tomorrow is", "counterfactual": "If today is Saturday, tomorrow is", "cf_answer": " Sunday"},
    {"input": "If today is Friday, tomorrow is", "counterfactual": "If today is Sunday, tomorrow is", "cf_answer": " Monday"}
  ]}},
  "sites": {"target": {"component": "layer_output", "layers": 5}, "head": {"component": "lm_head_output"}},
  "featurizers": {
    "rot": {"kind": "subspace", "file_path": "out/das/fit.rot.safetensors"},
    "pc": {"kind": "pca", "k": 4, "file_path": "out/pca_harvest/basis.safetensors"},
    "draw": {"kind": "subspace", "k": 4, "seed": 0}
  },
  "steps": {
    "counterfactual": {"kind": "forward", "input": "test.counterfactual", "interventions": {
      "v_cf": {"site": "target", "pos": -4, "ops": [
        {"op": "encode", "featurizer": {"sweep": ["rot", "pc", "draw"], "label": "space"}}]}}},
    "patched": {"kind": "forward", "input": "test.input", "interventions": {
      "patch": {"site": "target", "pos": -4, "ops": [
        {"op": "encode", "featurizer": {"sweep": ["rot", "pc", "draw"], "label": "space"}},
        {"op": "swap", "with": "counterfactual.v_cf"},
        {"op": "decode", "featurizer": {"sweep": ["rot", "pc", "draw"], "label": "space"}}]},
      "logits": {"site": "head", "pos": -1}}},
    "ce": {"kind": "op", "op": "cross_entropy", "of": "patched.logits", "target": "test.cf_answer"},
    "hit": {"kind": "op", "op": "match", "of": "patched.logits", "expected": "test.cf_answer"},
    "acc": {"kind": "op", "op": "mean", "of": "hit"},
    "mean_ce": {"kind": "op", "op": "mean", "of": "ce"}
  },
  "saves": ["acc", "mean_ce"]
}
```

```text
clab run documents/das_controls.json
clab table out/das_controls acc mean_ce
at                acc  mean_ce
{"space":"rot"}   1.0  1.0930441617965698
{"space":"pc"}    1.0  1.4620361328125
{"space":"draw"}  0.0  2.4739460945129395
```

- **The PCA swaps the day on 7/7**, at a mean `ce` of 1.462 against the trained rotation's 1.093. The draw,
  0/7 at 2.474, is `das.json`'s own null.
- **The `decode` carries its `encode`'s sweep**, the same label and names. A `decode` of one name is refused
  at the point: `… at space="pc": decode of 'rot', and the open encode is of 'pc' (ops[0])`.
- **`counterfactual` only encodes**, so one forward serves all 3 points; `patched` writes, and runs one each.
- **Sweep a field where one declaration serves**, as the seeds above do; name several featurizers where they
  differ in kind, `k` or bundle.

## DBM: which units carry the day

A `gate` learns a mask over the units of a value: soft, σ(θ/T), while the fit trains it, and hard, θ > 0,
whenever it is scored. `dbm.json` is `das.json` with these changes, and nothing else:

- the featurizer `"mask": {"kind": "gate", "temperature": {"start": 1.0, "end": 0.05}}` in place of `rot`, and
  `mask` and `fit.mask` wherever `das.json` writes `rot` and `fit.rot`;
- in the fit, `"objective": {"ce": 1.0, "mask.mask": {"sweep": [1.0, 2.0, 3.0, 5.0]}}`, `"epochs": 20` and
  `"lr": 0.1`;
- `"saves": ["held_acc", "ctl_acc", "fit.mask"]`.

`mask.mask` is the gate's own scalar, the mean of its soft mask: the pressure toward keeping few units. The
temperature falls linearly over the epochs. The units a fit kept are where its saved `theta` is positive, read
in Python: `clab table` refuses a bundle (`'fit.mask' is a trained featurizer's bundle, which has no rows to
table; read it from plan.results`).

<!-- test: skip -->
```python
from clab import Plan

plan = Plan.load("out/dbm")
for weight, gate in plan.results["fit.mask"].items():
    print(weight, int((gate["theta"] > 0).sum()), "of", gate["theta"].numel())
```

| `mask.mask` weight | units kept of 768 | `held_acc` |
|---|---|---|
| 1.0 | 330 | 1.0 |
| 2.0 | 169 | 0.857 |
| 3.0 | 98 | 0.429 |
| 5.0 | 30 | 0.0 |

`ctl_acc`, the bare `mask` with every unit on, is 1.0: the full-vector interchange. Four directions carry the
day where no small set of units does, so DAS and DBM answer different questions about one site. Each fit
takes seconds on a CPU. The swept `fit.mask` file holds a gate per weight; rerun at one weight to save a
loadable gate.

## Head masking: which heads the indirect object needs

Head patching moves one head at a time. Head masking trains a gate with one unit per head of every layer,
all at once. A head that is off takes its value from a corrupt prompt, the objective keeps the clean logit
difference, and an L1 weight turns heads off. Clean prompts are GPT-2's indirect-object task, where it gives
the indirect object (IO) over the subject (S); a corrupt prompt replaces the second subject with a third name:

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

- **`over: ["layers", "num_heads"]`** at `attention_premix` over every layer lays `theta` out `[12, 12]`, one
  unit per head, broadcast over `head_dim`. At layer L the write takes row L of the mask.
- **`apply` with `with`** writes `h·m + corrupt·(1−m)`: the clean value where a head is on, the corrupt one
  where it is off, each exactly. Write it this way: the same interpolation as a `sub`, `apply`, `add` chain
  rounds, and its untrained null misses the clean logit difference in its last bits.
- **The corrupt read and the write meet position for position**, at the static window `0:14`, on prompts that
  all tokenize to 14 tokens with one-token names.
- **`sq`, the squared change from the clean logit difference, is the objective**; `mean_ld` is reported.

```text
clab run documents/head_gates.json
clab table out/head_gates clean_mean
value
3.0798864364624023
clab table out/head_gates fit.train.mean_ld
at                    value
{"heads.mask":3.0}    3.1980886459350586
{"heads.mask":10.0}   2.7434635162353516
{"heads.mask":30.0}   2.5992813110351562
{"heads.mask":100.0}  1.6189403533935547
```

The heads each weight kept, as `[layer, head]` pairs of `theta`, which is laid out `[layers, num_heads]` in
the selection's order:

<!-- test: skip -->
```python
from clab import Plan

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

At weight 3.0 it keeps 42 heads. At 30.0, 14 of 144 heads hold the logit difference at 2.60 against the clean
3.08. Among them are heads the published indirect-object circuit of GPT-2 small names: duplicate-token heads
0.1 and 3.0, induction head 5.5, S-inhibition heads 7.3, 7.9, 8.6 and 8.10, and name movers 9.9 and 10.0.
Each fit takes seconds on a CPU.

### Its nulls

- **The untrained gate is the clean run.** Every θ starts at +1, so with `"epochs": 0` the `train` eval's
  `fit.train.ld` equals `clean_ld` to the bit, row for row.
- **A head that is off writes the corrupt value exactly.** At a weight of 3000 every head is off, and the saved
  `fit.train.run.gated` equals `corrupt.z` to the bit.
- **Every head off is the corrupt run only where the read position's token is shared.** These prompts differ
  at position 9 alone, and every head off gives the corrupt run's logit difference to the bit. Where the
  prompts differ at the read position, its own embedding and MLPs still see the clean token.

### Choosing the weight

The objective is a mean over the training rows, so a mask can hold the mean while single rows move. Save
`fit.train.ld` beside `clean_ld` and compare them row by row. Choose the weight on held-out prompts of the
same length, in steps after the fit:

- a harvest of the held-out corrupt prompts at `premix`, at the body's window;
- a gated forward through `fit.heads` with that harvest as `with`;
- a floor: the same forward with a `swap` of the harvest at every head.

Take the largest weight whose held-out metric stays within a tolerance of clean, relative to that floor, fixed
before the sweep.
