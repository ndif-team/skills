# Ops

An op is one function that runs in two places: as an `op` step over a value an earlier step made
(`{"kind": "op", "op": "mean", "of": "patched_ld"}`), or inside an entry's `ops` on the entry's selection. Both
give the same numbers. A step holds one op; a chain is one step per op. `clab describe ops` lists every op's
arguments and what it does to the axes.

## A step or an entry

`documents/ops_tour.json` takes a metric and a KL both ways, ranks the top three tokens, turns a comparison into
a rate, and reduces a per-layer readout:

```json
{
  "description": "Ops as steps and inside entries on GPT-2's indirect-object prompts: a metric both ways, KL against a clean read both ways, top_k and its members, a rate from a comparison, a spread, and a per-layer readout reduced to one number per layer.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"example_id": "mary", "input": "When Mary and John went to the store, John gave a drink to", "io": " Mary", "s": " John"},
    {"example_id": "tom", "input": "When Tom and Anna went to the park, Anna gave a ball to", "io": " Tom", "s": " Anna"},
    {"example_id": "sarah", "input": "When Sarah and James went to the office, James gave a book to", "io": " Sarah", "s": " James"},
    {"example_id": "paul", "input": "When Paul and Lisa went to the school, Lisa gave a pen to", "io": " Paul", "s": " Lisa"}
  ]}},
  "sites": {
    "head": {"component": "lm_head_output"},
    "resid": {"component": "layer_output", "layers": 6}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": {
      "logits": {"site": "head", "pos": -1},
      "ld_entry": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.io", "b": "pairs.s"}]},
      "lens": {"site": {"component": "layer_output", "layers": ":"}, "pos": -1,
               "ops": [{"op": "readout"}, {"op": "token_prob", "token": "pairs.io"}]}}},
    "damped": {"kind": "forward", "input": "pairs.input", "interventions": {
      "damp": {"site": "resid", "pos": -1, "ops": [{"op": "scale", "factor": 0.5}]},
      "logits": {"site": "head", "pos": -1},
      "kl_entry": {"site": "head", "pos": -1, "ops": [{"op": "kl", "against": "clean.logits"}]}}},
    "ld": {"kind": "op", "op": "logit_diff", "of": "clean.logits", "a": "pairs.io", "b": "pairs.s"},
    "kl": {"kind": "op", "op": "kl", "of": "damped.logits", "against": "clean.logits"},
    "top": {"kind": "op", "op": "top_k", "of": "clean.logits", "k": 3},
    "best": {"kind": "op", "op": "argmax", "of": "clean.logits"},
    "first_pick": {"kind": "op", "op": "eq", "of": "top.ids", "with": "best"},
    "positive": {"kind": "op", "op": "gt", "of": "ld", "with": 0},
    "rate": {"kind": "op", "op": "mean", "of": "positive"},
    "spread": {"kind": "op", "op": "std", "of": "ld"},
    "lens_by_layer": {"kind": "op", "op": "mean", "of": "clean.lens"}
  },
  "saves": ["ld", "clean.ld_entry", "kl", "damped.kl_entry", "top", "first_pick", "rate", "spread", "lens_by_layer"]
}
```

```text
$ clab table out/ops_tour ld clean.ld_entry kl damped.kl_entry
example_id  row  pos  pos_token  input                                             io      s       ld                  clean.ld_entry      kl                    damped.kl_entry
mary        0    13   Ġto        When Mary and John went to the store, John gave…   Mary    John   2.0063934326171875  2.0063934326171875  0.03818587213754654   0.03818587213754654
tom         1    13   Ġto        When Tom and Anna went to the park, Anna gave a…   Tom     Anna   2.5811233520507812  2.5811233520507812  0.1748969703912735    0.1748969703912735
sarah       2    13   Ġto        When Sarah and James went to the office, James …   Sarah   James  2.4727325439453125  2.4727325439453125  0.11409776657819748   0.11409776657819748
paul        3    13   Ġto        When Paul and Lisa went to the school, Lisa gav…   Paul    Lisa   4.1465301513671875  4.1465301513671875  0.055061765015125275  0.055061765015125275
$ clab table out/ops_tour rate spread
rate  spread
1.0   0.9305830001831055
$ clab table out/ops_tour lens_by_layer
layer  value
0      1.1615198900472024e-06
...
8      0.00025329587515443563
9      0.780062198638916
10     0.8180603981018066
11     0.42094409465789795
```

- **A metric inside the head's entry equals the step over the saved logits**, and keeps memory flat: a row's logits never outlive its batch.
- **An op step holds its whole input on the model's device** until the run ends, and computes over it at once; `kl` over two saved `[rows, pos, vocab_size]` reads builds several float32 copies of that size.
- **An entry sees one batch**, so an op over rows (`mean`, `std` over rows, `permute`, `pca`) is refused there; make it a step.
- **An entry acts at each layer in turn**, so a reduction over `layers` belongs in a step; `lens_by_layer` keeps one number per layer by default.
- **A reduction over `pos` alone may stay in the entry**, and only its result leaves the model.
- **A mean ablation's mean is an op step before the forward that swaps it in**; moved after, the entry has no edit and only reads.
- **An op step over two values needs both at one width.** Two reads of one forward at an open `pos` (`{"index": ":"}`) are refused even there; read them at a static index.

## Score an answer

A metric returns one float32 number per row and position. How a column becomes answer tokens (the leading
space, sets, `strip`, `lower`, literals) is [data](data.md#answer-columns).

| op | arguments | gives |
|---|---|---|
| `match` | `expected` | 1 where the top token is the answer (any member of a set), else 0; a tie goes to the lower id |
| `logit_diff` | `a`, `b` | `a`'s logit less `b`'s: a contrast, free of the offset every logit shares |
| `token_prob` | `token` | the answer's probability (a set's members summed): `match` made soft |
| `token_logit` | `token` | the raw logit; compare it only with another logit of the same read |
| `cross_entropy` | `target` | minus the log-probability, in nats |
| `contains` | `text` | 1 where a row's `detokenize`d generation contains the text ([generate](generate.md)) |

- **A metric needs the whole vocabulary.** A head narrowed with `select: {"vocab_size": [...]}` is refused under one; score the full head and read a narrowed copy beside it.
- **An answer of several tokens scores position i against token i**, so the read needs as many positions; a `before` scope reads them.
- **After a probe's `encode` the last axis is `classes`**, and the same metrics score it against a label column ([featurizers and fit](featurizers-and-fit.md)).

## Compare two values

| op | arguments | gives |
|---|---|---|
| `kl` | `against`, `reverse` | KL(`of` ‖ `against`) between next-token distributions in nats; `"reverse": true` is KL(`against` ‖ `of`) |
| `js` | `against` | the symmetric Jensen–Shannon divergence, at most ln 2 |
| `cosine` | `with` | cosine similarity over `hidden_size` |

- **KL(clean ‖ patched)** is the patched head entry `{"op": "kl", "against": "clean.logits", "reverse": true}`; without `reverse` it is KL(patched ‖ clean), as `kl` above.
- **The side with a `layers` axis is `of`**: an operand may leave out an axis, never carry one `of` lacks. A lens against the final logits is the lens's entry with `against` the logits.
- **A `kl` of two near-equal distributions can be a little below zero**, from float32 rounding; read it as zero.

## Reduce over an axis

`mean`, `sum`, `min`, `max`, `std` and `median` remove the rows and positions by default and keep a `layers`
axis; `over` names the axes to remove instead.

| `of` laid out | op | result |
|---|---|---|
| `layers, rows, pos` | `{"op": "mean"}` | `layers`: one number per layer |
| `layers, rows, pos` | `{"op": "mean", "over": ["layers"]}` | `rows, pos` |
| `rows, pos, hidden_size` | `{"op": "mean", "over": ["pos"]}` | `rows, hidden_size`: one vector per row |
| `rows, new` (a generation's ids) | `{"op": "mean", "over": ["rows", "new"]}` | one number |

- **`std` is the sample standard deviation**, n − 1 below, NaN under two values: `spread` above is the spread a mean effect is read against.
- **`median` of an even count is the mean of the two middle values**, as numpy's is.
- **`norm` and `dot` reduce over `hidden_size` by default**; a value without one names its axis (`"over": ["head_dim"]`). `norm` takes `p`.
- **`count` counts the positions along `over` (by default `pos`) that hold a value**: 0 on a row never reached, so a `mean` of it takes that row in.
- **`entropy` is the entropy in nats of a float32 softmax** over `vocab_size`, else `num_experts`, or the axes `over` names.
- **A reduction weighs positions, not rows**, over a ragged read, and skips missing values.
- **A reduction that would remove no axis is refused.**
- **`pca` is a reduction whose value is a bundle** a `pca` featurizer loads ([featurizers and fit](featurizers-and-fit.md#bundles)).

## Transform

- **`readout` projects a stream value through the final norm and the head**: the logit lens. It takes `token_embeddings`, `layer_input`, `attention_output`, `layer_mid`, `mlp_output` and `layer_output`, whole along `hidden_size`. The lens recipes are the `clab-lenses-and-attention` skill's.
- **`argmax` gives the top index along `over`**, by default the token id; an index along a selected axis counts within the select list.
- **`top_k` gives the `k` best `ids` with their float32 `probs`** over a vocabulary, or their `values` over another axis (`key`, `features`).
- **A `top_k` step has members**: `top.ids`, `top.probs` (or `top.values`) and, over logits, `top.tokens`, decoded. `first_pick` above, `eq` of `top.ids` with `best`, is `[1.0, 0.0, 0.0]` on every row.
- **The `top_k` value itself is several tensors**, which no op takes; an op names a member. A `top_k` inside an entry has no members.
- **`top_k` over a list of axes ranks their entries together**: over `["rows", "pos"]` it finds a latent's max-activating examples, with `ids.rows` and `ids.pos`.
- **`detokenize` turns a generation's ids into text**, stopped before each row's first EOS; with `after` it keeps the text after a marker.
- **`encode` gives a featurizer's coordinates** ([featurizers and fit](featurizers-and-fit.md)); **`grad` is an entry's first op** ([sites and interventions](sites-and-interventions.md#a-gradient-at-a-site)).
- **`permute` reorders a value's rows**, row i holding row π(i)'s, π drawn from `seed` alone; with `"derangement": true` no row stays. It is an op step only.

## Edit

The edits write back inside an entry ([sites and interventions](sites-and-interventions.md#read-or-write)); as a step
they compute the same value without a model.

| to | ops |
|---|---|
| put another run's value in | `{"op": "swap", "with": "<step>.<entry>"}` |
| put a constant or a mean in | `{"op": "swap", "with": 0.0}`, `{"op": "swap", "with": "<a mean step>"}` |
| add a direction at a dose | `{"op": "add", "with": "<direction>", "scale": 4.0}` |
| subtract, multiply | `{"op": "sub", "with": ...}`, `{"op": "scale", "factor": 0.5}` or a named value |
| keep the entries where a comparison holds | `{"op": "scale", "factor": "<a comparison>"}` |
| add Gaussian noise, the same in every forward | `{"op": "noise", "scale": 0.5, "seed": 7}` |
| bound every element | `{"op": "clamp", "min": -1.0, "max": 1.0}` |
| remove a direction (its length does not matter) | `{"op": "project_out", "direction": "<direction>"}` |
| give each vector unit length | `{"op": "normalize"}` |
| restore the norm the site had before this step's writes (`original`'s) | `{"op": "renormalize"}`, after the write it corrects; `"norm": "sum"` on a pattern |
| through a featurizer | `decode`, `apply` ([featurizers and fit](featurizers-and-fit.md)) |

The recipes that use them are the `clab-ablation`, `clab-patching` and `clab-steering-and-generation` skills'.

## Arithmetic and comparisons

`abs`, `exp`, `log`, `sign`, `sqrt`, `square`, `neg`, `mul`, `div` and `pow` compute each entry in float32 and keep
the layout; `eq`, `ne`, `gt`, `ge`, `lt` and `le` give 1 where the comparison holds and 0 where not.

- **They never write**: `neg` and `mul` in a read change only what the entry binds, where `scale` would write back.
- **A rate is the `mean` of a comparison**: `gt` of `ld` with 0, then `mean`, is the fraction of rows above zero (`rate` 1.0 above).
- **A mask is `mul` by a comparison** in a read or a step, or `scale` by it in a write.
- **A relative effect is `sub` then `div`, as steps**; `div` is NaN wherever the divisor is 0.
- **`log` is `-inf` at 0, `sqrt` NaN below 0.**
- **A comparison is NaN where either side is missing**, so a `mean` of it is over the rows present.

## What is not an op

No op reduces across sweep points, since an op sees one point at a time. There is no grouped mean, no interval and
no two-branch `where`. A second branch is `mul` by the opposite comparison, then `add`. A confidence interval, a
ranking across points or a join across runs is computed in Python from `Plan.load(out).results`
([output](output.md#in-python)).

## Missing values

A row or position a read did not reach is NaN, and so is every metric, edit, transform and comparison of it.
`argmax` and `top_k` give it id -1, which every later op takes as missing again. Reductions skip it: a `mean` is
over the values present, NaN only where none is. Where missing rows come from is
[positions](positions.md#rows-a-scope-never-reached).

## Dtypes

Metrics, reductions but `pca`, and the arithmetic compute in float32 and return float32, whatever the model's
dtype. An edit computes in float32 and rounds once to the value's dtype; a read keeps the model's. An op adds no
rounding, and cannot remove the model's: run the clab skill's worked document at `"dtype": "nf4"` and
the logit differences come back in steps of 0.5 (`-2.5`, `-1.5`, `-2.0`), the null still equal to the clean run.
Recompute a few points in float32 before reading a small effect from a bfloat16 run; the full rule is the
`clab-debugging` skill's.
