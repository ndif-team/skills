---
name: clab-trained-featurizers
description: Train a featurizer inside a clab run, or load one from a bundle, and test what it learned. Distributed alignment search (DAS, a trained rank-k subspace swapped in from a counterfactual, interchange intervention accuracy) and boundless masking (DBM, a gate over units), with PCA and random subspaces of matched rank as controls; head masking (a gate per attention head of every layer, the L1 weight, the heads kept); sparse autoencoders (a published Gemma Scope or SAELens JumpReLU or TopK file spliced in, its L0, latent ablation, max-activating examples, training a dictionary); linear probes per layer and their floor; LoRA adapters trained through the frozen model and switched on in chosen steps; cached harvests (`load`, shards) and bundles (`file_path`). Covers fit, subspace, gate, pca, sae, linear, lora, encode, decode, apply, epochs 0, fit.<name>, eval and history. The tuned lens is clab-lenses-and-attention, the optimal constant clab-ablation, the fit step's fields the clab skill.
---

# Trained featurizers

Verified on clab `830182f` with nnterp `e5d0f8e`, nnsight 0.8.2 and transformers 5.20.2; the documents run on
the CPU on `openai-community/gpt2` (12 layers of 12 heads, 768 wide), each in seconds to a minute.

A featurizer is a space or a function declared under `featurizers` and used from an entry's `ops` with
`encode`, `decode` and `apply`. A `fit` step trains it through the frozen model; saved, it is a bundle that a
second document loads with `file_path`. Each kind's fields, the fit step's fields and members, and the bundle
format are the `clab` skill's [featurizers-and-fit reference](../clab/references/featurizers-and-fit.md). This
skill is the recipes, each with its null.

## What breaks

Each of these runs without a refusal and costs a conclusion.

- **The bare name after a fit is the untrained draw.** Steps after the fit name the trained one `fit.<name>`:
  `das.json` below scores 7/7 through `fit.rot` and 0/7 through `rot`. Use the bare name on purpose, as the null.
- **A fit without its `epochs: 0` point proves nothing.** A trained featurizer means something only where it
  beats its own draw on the same rows. Score the bare name after the fit, with the featurizer given the fit's
  `seed`, or sweep `"epochs": {"sweep": [0, N]}`.
- **`lr × steps` bounds how far a featurizer gets.** Count the steps, `epochs × ⌈rows / batch_size⌉`. On
  `das.json`, lr 1e-4, 1e-3, 1e-2 and 1e-1 give 0/7, 7/7, 2/7 and 0/7: sweep `lr` before reading a flat curve.
- **A published SAE stated wrongly runs.** Write its `activation`, `centered` and layer from the release, the
  layer inline in `site`. A dictionary on GPT-2 stated with the wrong `centered` gives mean L0 21.7 where its
  truth gives 24.6. Compare mean L0 over every token after the first with the release's ([SAEs](references/sae.md)).
- **`apply` replaces; the `encode`, edit, `decode` bracket edits.** A subspace's `apply` is its projection and an
  SAE's drops the error term. An interchange or a latent ablation is the bracket, which keeps everything else.
- **A featurizer at a site that picks several layers is one shared featurizer.** Sweep the site's `layers`
  for one per layer. Only a gate or an affine with `layers` in its `over` holds units per layer.
- **A probe that decodes a variable does not show the model uses it.** Test use with an interchange along
  the probe's span ([probes](references/probes-and-lora.md#use-a-probe-causally)).
- **GPT-2's first position is an outlier.** Its layer 5 output has norm about 2990 against about 85 elsewhere.
  Harvest from position 1 (`{"index": "1:"}`) for a PCA or a dictionary, or position 0 takes a direction.

These are refused, and the fix is not obvious from the message:

- **A loading document needs a corpus root.** `file_path` and a load's `from` resolve under the nearest
  directory above the document that holds `data/`. With inline rows only, create an empty `data/` there, or the
  bundle is refused (`does not exist under the corpus root`). Run, show and table from that directory.
- **Two training runs into one output directory keep only the last manifest.** The first run's values then
  table as `'held_acc' was not saved; add it to saves`. Leave each training run its own `out/<document>/` and
  point `file_path` there, as every document in this skill does.
- **A swept fit writes one bundle per point, which no featurizer loads.** Train unswept to save a bundle.
- **An eval cannot read a harvest's dataset as another.** Score held-out rows in steps after the fit
  through `fit.<name>`, as `das.json` does.

## The shape every fit shares

`das.json` below is the shape every recipe in this skill reuses:

- **A harvest before the fit** reads what does not depend on the training, once (`counterfactual.h`). Each run
  of the body takes its minibatch's rows of it by id, so the model runs only where the body has a forward.
- **The body** (`steps` inside the fit) trains. Every body op gets a history, `fit.history.<op>`; one or more are
  the `objective`.
- **`eval`** runs the body again after training: `{"train": {}}` on the training rows (`fit.train.acc`),
  `{"held_out": {"train": "test"}}` with `test` read for `train` where the body holds no harvest.
- **Steps after the fit** score held-out rows through `fit.rot` and the null through the bare `rot`.
- **A saved `fit.<name>` is a bundle**, `fit.rot.safetensors`, which a second document loads by `file_path`.

## DAS: how many dimensions carry the day

GPT-2 answers "If today is Monday, tomorrow is" with " Tuesday" for all seven days. Swapping layer 5's whole
output at the day token in from another day's prompt gives that day's successor on every pair. DAS trains a
rank-4 subspace to do the same swap with 4 of the 768 dimensions.

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

- **`pos: -4` is the day token on every row**, since every prompt is the same 7 tokens and each day one
  token. `clab show out/das` lists `[3] on all 21 rows` for `counterfactual.h`; check it on rows of each kind.
- **`v_cf` encodes the harvest** in the subspace being trained, and the bracket swaps those 4 coordinates into
  the base prompt's. `decode` writes back only the change, so the other 764 dimensions stay the base's.
- **`ce` is the objective; `acc`**, the interchange intervention accuracy, **is reported**.
- **`rot` carries `"seed": 0`**, the fit's seed, so the bare `rot` after the fit is the fit's starting draw.

```text
clab run documents/das.json
clab table out/das held_acc
value
1.0
clab table out/das ctl_acc
value
0.0
clab table out/das fit.history.acc
epochs  value
0       0.0476190485060215
3       0.3333333432674408
6       1.0
9       0.9523809552192688
```

The trained subspace swaps the day on all 7 held-out pairs; its draw on none. The `ctl_*` steps equal an
`epochs: 0` run's `held_*` values to the bit, so this document carries its own null. Twenty-one pairs train
it in seconds on a CPU. The rank curve, the spread across seeds, the PCA control and DBM are
[subspaces-and-masks.md](references/subspaces-and-masks.md).

### Load it in a second document

The run writes `out/das/fit.rot.safetensors`: one tensor, `basis` `[768, 4]`, and a header recording `kind`,
`k`, `parametrization`, the model's `key`, `revision` and `dtype`, `component`, `layer` and `width`. This
document loads it; `k` comes from the header:

<!-- test: skip -->
```json
{
  "description": "The apply half of das.json: the rotation it saved as fit.rot, loaded from its bundle and scored on the same 7 held-out pairs. k comes from the bundle's header.",
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
  "featurizers": {"rot": {"kind": "subspace", "file_path": "out/das/fit.rot.safetensors"}},
  "steps": {
    "counterfactual": {"kind": "forward", "input": "test.counterfactual", "interventions": {
      "v_cf": {"site": "target", "pos": -4, "ops": [{"op": "encode", "featurizer": "rot"}]}}},
    "patched": {"kind": "forward", "input": "test.input", "interventions": {
      "patch": {"site": "target", "pos": -4, "ops": [{"op": "encode", "featurizer": "rot"},
                                                     {"op": "swap", "with": "counterfactual.v_cf"},
                                                     {"op": "decode", "featurizer": "rot"}]},
      "logits": {"site": "head", "pos": -1}}},
    "ce": {"kind": "op", "op": "cross_entropy", "of": "patched.logits", "target": "test.cf_answer"},
    "hit": {"kind": "op", "op": "match", "of": "patched.logits", "expected": "test.cf_answer"},
    "acc": {"kind": "op", "op": "mean", "of": "hit"}
  },
  "saves": ["acc", "ce"]
}
```

```text
clab run documents/das_apply.json
clab table out/das_apply ce
row  pos  pos_token  input                               cf_answer  value
0    6    Ġis        If today is Sunday, tomorrow is      Saturday  1.1828811168670654
1    6    Ġis        If today is Wednesday, tomorrow is   Monday    0.8661616444587708
4    6    Ġis        If today is Monday, tomorrow is      Saturday  0.6069786548614502
```

`ce` equals `das.json`'s `held_ce` to the bit, row for row, and `acc` is 1.0: the same rows through the same
rotation. That holds at one batch size and one thread count; at another the last bits move (reproducibility is
the `clab-debugging` skill's). A header that disagrees with the loading document is refused before the model loads,
naming the key (`header 'layer' is '5'; this document uses 'rot' at layer_output layer 6`). The checks and
the file layouts of every kind are [harvest-and-bundles.md](references/harvest-and-bundles.md).

## Which kind for which question

| question | kind | recipe |
|---|---|---|
| which directions of a value carry a variable | `subspace` | above; rank, seeds, controls in [subspaces-and-masks.md](references/subspaces-and-masks.md) |
| which units of a value carry it | `gate` | [DBM](references/subspaces-and-masks.md#dbm-which-units-carry-the-day) |
| which heads a behaviour needs, all at once | `gate` with `over: ["layers", "num_heads"]` | [head masking](references/subspaces-and-masks.md#head-masking-which-heads-the-indirect-object-needs) |
| what a trained subspace has to beat | a drawn `subspace`; a loaded `pca` | [controls](references/subspaces-and-masks.md#controls-of-matched-rank) |
| which interpretable features are active, and what an edit of one does | `sae` | [sae.md](references/sae.md) |
| whether a property is linearly decodable, and where | `linear` | [probes](references/probes-and-lora.md#a-probe-per-layer) |
| what a low-rank weight change learns, on in some steps | `lora` | [LoRA](references/probes-and-lora.md#lora-train-an-adapter) |
| reuse a harvest, fit with no forward | a `load` step | [cached harvests](references/harvest-and-bundles.md#cached-harvests) |
| a full linear map per layer (tuned lens) | `affine` | the `clab-lenses-and-attention` skill |
| one vector in place of a value (optimal constant) | `constant` | the `clab-ablation` skill |

Every kind's fields and ops are in `clab describe featurizers`; the fit and load steps are in
`clab describe steps`.

## References

- [subspaces-and-masks.md](references/subspaces-and-masks.md): DAS by rank and seed, random subspaces
  without a fit, the PCA control, DBM's sparsity curve, head masking on the indirect-object task.
- [sae.md](references/sae.md): a published file's layout and the three fields it needs, L0 and splice KL,
  ranking and ablating latents, max-activating examples, training a dictionary and loading it.
- [probes-and-lora.md](references/probes-and-lora.md): a probe per layer and its floor, many probes in one
  fit, a probe's span used in an interchange, LoRA trained, applied in chosen steps, and across sites.
- [harvest-and-bundles.md](references/harvest-and-bundles.md): bundle files, headers and the checks;
  `load` a saved harvest, fit on it with no forward, and harvest in shards.
