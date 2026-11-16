# Ops

An op is one function that runs in two places: as an `op` step over a value an earlier step made
(`{"kind": "op", "op": "mean", "of": "patched_ld"}`), or inside an entry's `ops` on the entry's selection. A
step holds one op; a chain is one step per op. `nnspec describe ops` prints every op's arguments, what it does
to the axes and how it treats missing values: this page says which op answers which question.

## A step or an entry

`documents/ops_tour.json` takes a metric and a KL both ways, ranks the top three tokens, turns a comparison into
a rate, and reduces a per-layer readout:

```json
{
  "description": "Ops as steps and inside entries on GPT-2's indirect-object prompts: a metric and a KL both ways, top_k and its members, a rate from a comparison, a spread, and a per-layer readout reduced per layer.",
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
$ nnspec table out/ops_tour ld clean.ld_entry kl damped.kl_entry
example_id  row  pos  pos_token  input                                             io      s       ld                  clean.ld_entry      kl                    damped.kl_entry
mary        0    13   Ġto        When Mary and John went to the store, John gave…   Mary    John   2.0063934326171875  2.0063934326171875  0.03818587213754654   0.03818587213754654
tom         1    13   Ġto        When Tom and Anna went to the park, Anna gave a…   Tom     Anna   2.5811233520507812  2.5811233520507812  0.1748969703912735    0.1748969703912735
...
$ nnspec table out/ops_tour rate spread
rate  spread
1.0   0.9305830001831055
```

- **A metric inside the head's entry equals the step over the saved logits** here to the bit, and keeps memory flat: a row's logits never outlive its batch. An op step holds its whole input on the model's device until the run ends.
- **An entry's op runs on each batch and a step's on the whole value.** An op whose bits depend on the value's shape (a matmul, as in `readout` and `encode`, or a long sum) can differ in its last bits between the two forms, on the CPU too.
- **An entry sees one batch**, so an op over rows (`mean`, `std` over rows, `permute`, `pca`) is refused there; make it a step. An entry acts at each layer in turn, so a reduction over `layers` belongs in a step too. A reduction over `pos` alone may stay in the entry, and only its result leaves the model.
- **A mean ablation's mean is an op step before the forward that swaps it in.**
- **An op step over two reads needs both at one width**, two reads of one forward too ([pairing](positions.md#pairing-with-another-forward)).

## Score an answer

A metric returns one float32 number per row and position. How a column becomes answer tokens is
[data](data.md#answer-columns).

- **`match`** is 1 where the top token is the answer: accuracy, as its mean. It stays 0 until the top token flips, so read a graded metric beside it.
- **`logit_diff`** (`a`, `b`) is a contrast, free of the offset every logit shares. **`token_logit`** is the raw logit: compare it only with another logit of the same read.
- **`token_prob`** is the answer's probability, a set's members summed.
- **`cross_entropy`** (`target`) is minus the log-probability in nats: a loss, and summed over `pos` a forced answer's sequence score.
- **`contains`** is 1 where a row's `detokenize`d generation contains the text, a substring test ([generation](generation.md#did-it-say-the-answer)).
- **A metric needs the whole vocabulary.** A head narrowed with `select: {"vocab_size": [...]}` is refused under one; score the full head and read a narrowed copy beside it.
- **After a probe's `encode` the last axis is `classes`**, and the same metrics score it against a label column ([featurizers](featurizers.md)).

## Compare two values

- **`kl`** is KL(`of` ‖ `against`) between next-token distributions in nats; `"reverse": true` is KL(`against` ‖ `of`). The side with a `layers` axis is `of`, since an operand may leave out an axis but never carry one `of` lacks: a lens against the final logits is the lens's `kl` with `against` the logits. A `kl` of two near-equal distributions can be a little below zero, from float32 rounding.
- **`js`** is the symmetric Jensen–Shannon divergence, at most ln 2; **`cosine`** the similarity over `hidden_size`.
- **Two values are equal where `max` of `abs` of their `sub` is 0**, over every axis named: the form of every exact null.

## Reduce over an axis

`mean`, `sum`, `min`, `max`, `std` and `median` remove the rows and positions by default and keep a `layers`
axis; `over` names the axes to remove instead, and every axis it does not name stays.

| `of` laid out | op | result |
|---|---|---|
| `layers, rows, pos` | `{"op": "mean"}` | `layers`: one number per layer, as `lens_by_layer` above |
| `layers, rows, pos` | `{"op": "mean", "over": ["layers"]}` | `rows, pos` |
| `rows, pos, hidden_size` | `{"op": "mean", "over": ["pos"]}` | `rows, hidden_size`: one vector per row |
| `rows, new` (a generation's ids) | `{"op": "mean", "over": ["rows", "new"]}` | one number |

- **Name every axis a reduction removes.** `max` of a `[rows, pos, vocab_size]` read is `[vocab_size]`, each token's largest logit; the largest logit is `"over": ["rows", "pos", "vocab_size"]`.
- **`std` is the sample standard deviation**, the spread a mean effect is read against (`spread` above). **`median`** is robust to an outlier position where `mean` is not.
- **`norm` and `dot` reduce over `hidden_size` by default**; a value without one names its axis (`"over": ["head_dim"]`). `dot` with a unit direction is a value's coordinate along it: `normalize` the direction in a step first.
- **`count` counts the positions along `over` (by default `pos`) that hold a value**: 0 on a row never reached.
- **`entropy` is the entropy of a softmax** over `vocab_size`, else `num_experts`, or the axes `over` names: give it logits or scores, not probabilities.
- **A reduction weighs positions, not rows**, over a ragged read, and skips missing values.
- **`pca` is a reduction whose value is a bundle** a `pca` featurizer loads ([featurizers](featurizers-subspaces-and-masks.md#controls-of-matched-rank)).

## Rank and transform

- **`argmax` gives the top index along `over`**, by default the token id; `eq` of two runs' `argmax` is whether they pick one token.
- **`top_k` gives the `k` best `ids`** with their float32 `probs` over a vocabulary, or their `values` over another axis (`key`, `features`, `classes`). As a step it has members: `top.ids`, `top.probs` or `top.values` and, over logits, `top.tokens`, decoded. The value itself is several tensors, which no op takes; an op names a member. `first_pick` above, `eq` of `top.ids` with `best`, is `[1.0, 0.0, 0.0]` on every row. A `top_k` inside an entry has no members.
- **`top_k` over a list of axes ranks their entries together**: over `["layers", "num_heads"]` every head of the model, over `["rows", "pos"]` a latent's max-activating examples. Its `ids` hold one index per ranked axis, counted along the value's own axes. A ranking with no rows does not table; read it from `plan.results` ([output](output.md#in-python)).
- **The `k` smallest are `neg`, then `top_k`**, whose `values` come back negated.
- **An index along a narrowed axis counts within the selection**: under `"layers": "6:12"`, layer 10 is index 4.
- **`readout` projects a stream value through the final norm and the head**: the logit lens ([lens](lens.md)).
- **`detokenize` turns a generation's ids into text**, stopped before each row's first EOS ([generation](generation.md)).
- **`permute` reorders a value's rows**, row i holding row π(i)'s, π drawn from `seed` alone; with `"derangement": true` no row stays ([resample ablation](ablation.md#resample)).
- **`select` as a step narrows a value along an axis a read cannot narrow**: a class of a probe's `W`, a latent's row of an sae's `W_dec` (`{"kind": "op", "op": "select", "of": "fit.probe.W", "select": {"classes": 1}}`). An int keeps its axis at size 1; a `sum` over the axis drops it, exactly. `rows`, `pos` and `key` are not a select's to narrow.
- **`encode`** gives a featurizer's coordinates ([featurizers](featurizers.md)); **`grad`** is an entry's first op ([a gradient at a site](sites-and-interventions.md#a-gradient-at-a-site)).

## Edit

The edits write back inside an entry ([read or write](sites-and-interventions.md#read-or-write)); as a step
they compute the same value without a model.

| to | ops |
|---|---|
| put another run's value, a constant or a mean in | `{"op": "swap", "with": "<step>.<entry>"}`, `"with": 0.0`, `"with": "<a mean step>"` |
| add a direction at a dose | `{"op": "add", "with": "<direction>", "scale": 4.0}` |
| subtract, multiply | `{"op": "sub", "with": ...}`, `{"op": "scale", "factor": 0.5}` or a named value |
| keep the entries where a comparison holds | `{"op": "scale", "factor": "<a comparison>"}` |
| add Gaussian noise, the same in every forward | `{"op": "noise", "scale": 0.5, "seed": 7}` |
| bound every element | `{"op": "clamp", "min": -1.0, "max": 1.0}` |
| remove a direction (its length does not matter) | `{"op": "project_out", "direction": "<direction>"}` |
| give each vector unit length | `{"op": "normalize"}` |
| restore the norm the site had before this step's writes | `{"op": "renormalize"}`, after the write it corrects; `"norm": "sum"` on a pattern |
| through a featurizer | `decode`, `apply` ([featurizers](featurizers.md#in-an-entry)) |

## Arithmetic and comparisons

Each computes entry by entry, in float32, and keeps the layout. They never write: `neg` and `mul` in a read
change only what the entry binds, where `scale` would write back.

- **A rate is the `mean` of a comparison**: `gt` of `ld` with 0, then `mean`, is the fraction of rows above zero (`rate` 1.0 above). A comparison is NaN where either side is missing, so the mean is over the rows present.
- **A relative effect is `sub` then `div`**, NaN where the divisor is 0; that makes `div` by a 0/1 mask the way to turn rows into holes a reduction skips.
- **A mask is `mul` by a comparison**; a second branch is `mul` by the opposite comparison, then `add`.
- **An RMS is `square`, `mean`, `sqrt`**; `exp` of a `neg`ated `cross_entropy` is a probability; a `mean` of `sign` is the share of rows that move up less the share that move down.

## What is not an op

No op reduces or compares across sweep points, since an op sees one point at a time. There is no grouped mean,
no interval, no two-branch `where`, and no first index where a condition holds. A confidence interval, a
ranking across points, the first layer at which the answer is on top or a join across runs is computed in
Python from `Plan.load(out).results` ([output](output.md#in-python)).

## Missing values

A row or position a read did not reach is NaN, and so is every metric, edit, transform and comparison of it.
`argmax` and `top_k` give it id -1, which every later op takes as missing again. Reductions skip it: a `mean` is
over the values present, NaN only where none is. Where missing rows come from is
[positions](positions.md#rows-a-scope-never-reached). Metrics, reductions and arithmetic compute in float32
and cannot remove the model's own rounding ([the float32 rule](debugging.md#the-float32-rule)).
