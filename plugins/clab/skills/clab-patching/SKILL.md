---
name: clab-patching
description: Find which layer, position, attention head, MLP neuron, path or attention pattern carries a behaviour, with a clab document that swaps a counterfactual prompt's value into a base prompt. Covers interchange (activation) patching, denoising and noising, corrupt-and-restore (causal tracing), the layer-by-position map, head and head-set patching, neuron patching, path patching (to the logits or a later head's query), attention-pattern patching, and attribution patching (gradient times the counterfactual difference) to rank every head with one backward. Includes the self-swap null, the exactly-zero gradient null, scoping both prompts of a pair, windows, IIA (`match`), logit difference, `token_prob`, `kl`, `js`, `cosine`, and IOI-style name pairs. Ablation (zero, mean, resample) is `clab-ablation`; DAS is `clab-trained-featurizers`; the format is `clab`; a refusal or failing null is `clab-debugging`.
---

# Patching with clab

Verified on clab `830182f` with nnterp `e5d0f8e`, nnsight 0.8.2 and transformers 5.20.2; the documents run on
the CPU on `openai-community/gpt2` (float32, 12 layers of 12 heads), each in seconds to under a minute.

Patching asks whether a value carries what decides the answer. Read it on a counterfactual prompt, swap it into
the base prompt at the same place, and score the base run. In clab that is a forward that reads, a forward
whose entry writes with `swap`, and metric steps after them; every recipe here carries the null that proves
the wiring.

## What breaks

- **An entry without an edit in `ops` only reads.** A patch written without `"ops": [{"op": "swap", "with":
  "<step>.<entry>"}]` runs and returns the clean numbers.
- **A patch without its null proves nothing.** Read the base's own value at the patch's site and `pos`, swap it
  back in a forward of its own, and check its scores equal the clean run's to the bit. A gap means the read and
  the write are not the same selection: fix the document before reading any effect.
- **Both prompts of a pair belong in one dataset.** A counterfactual read on a dataset of its own is refused as
  an operand (`'counterfactual.v_cf' of 'cfs'; an op takes one dataset's rows`). Put it in a column of the base
  rows and run its forward on that column.
- **A counted `pos` between two prompts pairs index with index.** On rows of different lengths the run refuses
  a pair it cannot vouch for and names the two tokens. On rows of equal length nothing checks it. Scope each
  side on its own word: `{"scope": {"column": "pairs.day"}, "index": -1}` on the base, `pairs.cf_day` on the
  counterfactual.
- **A `layers` slice or list on a write patches every layer it picks at once.** Sweep `layers` to patch one
  layer per point.
- **A swept write `pos` against an unswept read writes the one read value at every point.** Sweep the read's
  `pos` over the same values under the same `label`.
- **Two sweeps of one field zip unless each has its own `label`.** Sweeps of different fields (`layers`,
  `num_heads`) cross. Check the `axes` column of `clab show` before trusting a grid.
- **A swept `pos`'s coordinate is not the table's `pos` column.** `at` holds the swept value; `pos` is where
  the tabled read landed. Label the patch's sweep (`"label": "patch_pos"`) to keep the two apart.
- **`match` alone hides a graded effect.** It stays 0 until the top token flips: at layer 7, the last position
  of the map below has `iia` 0.0 while the logit difference moved by +1.22. Read `logit_diff` beside it.
- **The last layer at the last position always gives full effect.** It reproduces the counterfactual's logits,
  so a layer sweep at `-1` measures how late the answer can still be swapped in, not where it is computed.
- **A freeze listed after the patch it protects overwrites it**, every point equals the clean run, and the
  nulls still pass ([heads, neurons and paths](references/heads-neurons-and-paths.md)).
- **A `grad` of the wrong entry gives a plausible map.** Carry the null whose gradient must be exactly zero
  ([attribution](references/attribution.md)).
- **An answer column is the token the model continues with, leading space included** (`" Tuesday"`); the
  `clab` skill's [data reference](../clab/references/data.md) has the rule.

## The interchange

Use it to test one site at one position: does the value there carry the answer? The pair is one dataset. Each
row holds the base prompt (`input`), the counterfactual prompt, each prompt's word and each answer, and the
counterfactual forward runs on its own column, so its values keep the base's rows.

```json
{
  "description": "The base interchange on GPT-2: the residual stream leaving block 4 at the day's token, read on the counterfactual prompt and swapped into the base prompt at its own day's token. The null swaps the base's own value back in and must equal the clean run.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"example_id": "mon-thu", "input": "If today is Monday, tomorrow is", "day": "Monday", "base_answer": " Tuesday",
     "counterfactual": "If today is Thursday, tomorrow is", "cf_day": "Thursday", "cf_answer": " Friday"},
    {"example_id": "tue-fri", "input": "If today is Tuesday, tomorrow is", "day": "Tuesday", "base_answer": " Wednesday",
     "counterfactual": "If today is Friday, tomorrow is", "cf_day": "Friday", "cf_answer": " Saturday"},
    {"example_id": "wed-sat", "input": "If today is Wednesday, tomorrow is", "day": "Wednesday", "base_answer": " Thursday",
     "counterfactual": "If today is Saturday, tomorrow is", "cf_day": "Saturday", "cf_answer": " Sunday"},
    {"example_id": "thu-sun", "input": "If today is Thursday, tomorrow is", "day": "Thursday", "base_answer": " Friday",
     "counterfactual": "If today is Sunday, tomorrow is", "cf_day": "Sunday", "cf_answer": " Monday"},
    {"example_id": "fri-mon", "input": "If today is Friday, tomorrow is", "day": "Friday", "base_answer": " Saturday",
     "counterfactual": "If today is Monday, tomorrow is", "cf_day": "Monday", "cf_answer": " Tuesday"},
    {"example_id": "sat-tue", "input": "If today is Saturday, tomorrow is", "day": "Saturday", "base_answer": " Sunday",
     "counterfactual": "If today is Tuesday, tomorrow is", "cf_day": "Tuesday", "cf_answer": " Wednesday"},
    {"example_id": "sun-wed", "input": "If today is Sunday, tomorrow is", "day": "Sunday", "base_answer": " Monday",
     "counterfactual": "If today is Wednesday, tomorrow is", "cf_day": "Wednesday", "cf_answer": " Thursday"}
  ]}},
  "sites": {"resid": {"component": "layer_output", "layers": 4}, "head": {"component": "lm_head_output"}},
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": {
      "v_self": {"site": "resid", "pos": {"scope": {"column": "pairs.day"}, "index": -1}},
      "logits": {"site": "head", "pos": -1}}},
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {
      "v_cf": {"site": "resid", "pos": {"scope": {"column": "pairs.cf_day"}, "index": -1}}}},
    "null": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "resid", "pos": {"scope": {"column": "pairs.day"}, "index": -1},
                "ops": [{"op": "swap", "with": "clean.v_self"}]},
      "logits": {"site": "head", "pos": -1}}},
    "patched": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "resid", "pos": {"scope": {"column": "pairs.day"}, "index": -1},
                "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
      "logits": {"site": "head", "pos": -1}}},
    "clean_top": {"kind": "op", "op": "match", "of": "clean.logits", "expected": "pairs.base_answer"},
    "clean_ld": {"kind": "op", "op": "logit_diff", "of": "clean.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "null_ld": {"kind": "op", "op": "logit_diff", "of": "null.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "ld": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "iia": {"kind": "op", "op": "match", "of": "patched.logits", "expected": "pairs.cf_answer"},
    "p_cf": {"kind": "op", "op": "token_prob", "of": "patched.logits", "token": "pairs.cf_answer"},
    "effect": {"kind": "op", "op": "sub", "of": "ld", "with": "null_ld"}
  },
  "saves": ["clean_top", "clean_ld", "null_ld", "ld", "iia", "p_cf", "effect"]
}
```

```text
clab run documents/interchange.json
clab table out/interchange clean_ld null_ld ld iia
```

```text
example_id  row  pos  pos_token  input                               cf_answer   base_answer  clean_ld             null_ld              ld                  iia
mon-thu     0    6    Ġis        If today is Monday, tomorrow is      Friday      Tuesday     -1.043609619140625   -1.043609619140625   1.0651702880859375  1.0
tue-fri     1    6    Ġis        If today is Tuesday, tomorrow is     Saturday    Wednesday   -1.4108200073242188  -1.4108200073242188  2.1431808471679688  1.0
...
sat-tue     5    6    Ġis        If today is Saturday, tomorrow is    Wednesday   Sunday      -2.8488235473632812  -2.8488235473632812  2.2489700317382812  1.0
sun-wed     6    6    Ġis        If today is Sunday, tomorrow is      Thursday    Monday      -1.568878173828125   -1.568878173828125   1.0600814819335938  1.0
```

- **The clean run gets the task.** `clean_top` is 1.0 on all 7 rows: GPT-2 puts the next day on top, with
  `clean_ld` (counterfactual answer less base answer) between −1.04 and −2.85.
- **The null equals the clean run to the bit** on every row. `effect`, the patched score less the null's, is
  2.11 to 5.10.
- **The patch flips every row**: `iia` is 1.0, and `p_cf`, the probability of the counterfactual's next day,
  is 0.18 to 0.31.
- **`layer_output` at layer 4** is the residual stream leaving block 4. At a position it holds everything
  computed there so far, so the patch moves the whole state; the references narrow it to a head or neurons.
- **`index: -1` inside the scope** takes the word's last token, one position on every row whatever the word's
  length. `clab show out/interchange` prints where each entry landed (`[3] on all 7 rows` here).

How a scope finds its word, and how to check it, is the `clab` skill's
[positions reference](../clab/references/positions.md); entries and operands are its
[sites and interventions reference](../clab/references/sites-and-interventions.md).

## Choosing the metric

| op | scores | use it for |
|---|---|---|
| `match` | 1 where the top token is `expected`, else 0 | interchange accuracy (IIA), as its mean over rows |
| `logit_diff` | logit of `a` less logit of `b` | the graded effect, which moves before any top token flips |
| `token_prob` | the probability of `token` | a soft accuracy between the two |
| `kl`, `js` | divergence of the patched next-token distribution from the clean one, in nats | how far the whole prediction moved |
| `cosine` | cosine similarity over `hidden_size` | how far the residual stream moved, per layer and position |

`kl`, `js` and `cosine` are recipes in [pairs, scopes and maps](references/pairs-scopes-and-maps.md).
`clab describe ops` lists every metric.

## A layer-by-position map

Use it to find where in depth and in the sequence the answer sits. One document holds two sweeps of `pos`, on
the read and on the write, under one label, `patch_pos`, so they zip: position p is read on the counterfactual
and written at p. The `layers` sweep on the shared site crosses them, giving 12 × 7 = 84 points.

```json
{
  "description": "A layer-by-position map on GPT-2: the counterfactual's residual stream at layer L, position p, swapped into the base at layer L, position p, for every layer and every position of 7-token weekday prompts. The read's and the write's pos sweeps share the label patch_pos, so they zip; the layers sweep on the site crosses them: 84 points. The null swaps the base's own value back at every point; cf_mean is the counterfactual run's own score, which the last layer's last position must reproduce.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"example_id": "mon-thu", "input": "If today is Monday, tomorrow is", "base_answer": " Tuesday",
     "counterfactual": "If today is Thursday, tomorrow is", "cf_answer": " Friday"},
    {"example_id": "tue-fri", "input": "If today is Tuesday, tomorrow is", "base_answer": " Wednesday",
     "counterfactual": "If today is Friday, tomorrow is", "cf_answer": " Saturday"},
    {"example_id": "wed-sat", "input": "If today is Wednesday, tomorrow is", "base_answer": " Thursday",
     "counterfactual": "If today is Saturday, tomorrow is", "cf_answer": " Sunday"},
    {"example_id": "thu-sun", "input": "If today is Thursday, tomorrow is", "base_answer": " Friday",
     "counterfactual": "If today is Sunday, tomorrow is", "cf_answer": " Monday"},
    {"example_id": "fri-mon", "input": "If today is Friday, tomorrow is", "base_answer": " Saturday",
     "counterfactual": "If today is Monday, tomorrow is", "cf_answer": " Tuesday"},
    {"example_id": "sat-tue", "input": "If today is Saturday, tomorrow is", "base_answer": " Sunday",
     "counterfactual": "If today is Tuesday, tomorrow is", "cf_answer": " Wednesday"},
    {"example_id": "sun-wed", "input": "If today is Sunday, tomorrow is", "base_answer": " Monday",
     "counterfactual": "If today is Wednesday, tomorrow is", "cf_answer": " Thursday"}
  ]}},
  "sites": {
    "resid": {"component": "layer_output", "layers": {"sweep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}},
    "head": {"component": "lm_head_output"}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": {
      "v_self": {"site": "resid", "pos": {"index": {"sweep": [0, 1, 2, 3, 4, 5, 6], "label": "patch_pos"}}},
      "logits": {"site": "head", "pos": -1}}},
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {
      "v_cf": {"site": "resid", "pos": {"index": {"sweep": [0, 1, 2, 3, 4, 5, 6], "label": "patch_pos"}}},
      "logits": {"site": "head", "pos": -1}}},
    "null": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "resid", "pos": {"index": {"sweep": [0, 1, 2, 3, 4, 5, 6], "label": "patch_pos"}},
                "ops": [{"op": "swap", "with": "clean.v_self"}]},
      "logits": {"site": "head", "pos": -1}}},
    "patched": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "resid", "pos": {"index": {"sweep": [0, 1, 2, 3, 4, 5, 6], "label": "patch_pos"}},
                "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
      "logits": {"site": "head", "pos": -1}}},
    "clean_ld": {"kind": "op", "op": "logit_diff", "of": "clean.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "cf_ld": {"kind": "op", "op": "logit_diff", "of": "counterfactual.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "null_ld": {"kind": "op", "op": "logit_diff", "of": "null.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "ld": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "iia": {"kind": "op", "op": "match", "of": "patched.logits", "expected": "pairs.cf_answer"},
    "clean_mean": {"kind": "op", "op": "mean", "of": "clean_ld"},
    "cf_mean": {"kind": "op", "op": "mean", "of": "cf_ld"},
    "null_mean": {"kind": "op", "op": "mean", "of": "null_ld"},
    "ld_mean": {"kind": "op", "op": "mean", "of": "ld"},
    "iia_mean": {"kind": "op", "op": "mean", "of": "iia"}
  },
  "saves": ["clean_mean", "cf_mean", "null_mean", "ld_mean", "iia_mean", "ld"]
}
```

`clean` and `counterfactual` only read, so one forward of each serves all 84 points (`clab show` prints `all
84 points` beside them); `null` and `patched` run one forward per point. Every mean over rows is one number per
point, and `clab table` keys it by the point's coordinate in `at`:

```text
clab run documents/layer_pos_map.json
clab table out/layer_pos_map clean_mean cf_mean
clab table out/layer_pos_map ld_mean null_mean iia_mean
```

```text
clean_mean           cf_mean
-1.6586096286773682  1.7260316610336304

at                           ld_mean              null_mean            iia_mean
{"layers":0,"patch_pos":2}   -1.6586096286773682  -1.6586096286773682  0.0
{"layers":0,"patch_pos":3}   1.7269853353500366   -1.6586096286773682  1.0
{"layers":5,"patch_pos":3}   1.667783498764038    -1.6586096286773682  1.0
{"layers":8,"patch_pos":3}   -0.0129231046885252  -1.6586096286773682  0.2857142984867096
{"layers":11,"patch_pos":3}  -1.6586096286773682  -1.6586096286773682  0.0
{"layers":11,"patch_pos":6}  1.7260316610336304   -1.6586096286773682  1.0
```

`clab table out/layer_pos_map ld` prints the per-row values with the same `at` beside each row, and `pos` 6,
the logits read. In Python a swept result is a dict keyed by the coordinate, a tuple in the order of
`plan.sweeps` (here `layers`, then `patch_pos`):

<!-- test: skip -->
```python
from clab import Plan

plan = Plan.load("out/layer_pos_map")
ld = plan.results["ld_mean"]                        # {(layer, patch_pos): tensor}
tokens = plan.run.tokens["pairs.input"][0]
print("     " + "".join(f"{t:>10}" for t in tokens))
for layer in range(12):
    print(f"L{layer:<4}" + "".join(f"{ld[(layer, p)].item():10.2f}" for p in range(7)))
```

```text
             If    Ġtoday       Ġis   ĠMonday         , Ġtomorrow       Ġis
L0        -1.66     -1.66     -1.66      1.73     -1.66     -1.67     -1.66
L1        -1.66     -1.66     -1.66      1.73     -1.65     -1.67     -1.66
L2        -1.66     -1.66     -1.66      1.72     -1.64     -1.67     -1.67
L3        -1.66     -1.66     -1.66      1.73     -1.64     -1.67     -1.68
L4        -1.66     -1.66     -1.66      1.76     -1.65     -1.67     -1.68
L5        -1.66     -1.66     -1.66      1.67     -1.64     -1.68     -1.61
L6        -1.66     -1.66     -1.66      1.15     -1.64     -1.67     -1.12
L7        -1.66     -1.66     -1.66      0.44     -1.63     -1.66     -0.44
L8        -1.66     -1.66     -1.66     -0.01     -1.63     -1.65      0.02
L9        -1.66     -1.66     -1.66     -1.16     -1.61     -1.62      1.10
L10       -1.66     -1.66     -1.66     -1.69     -1.60     -1.64      1.67
L11       -1.66     -1.66     -1.66     -1.66     -1.66     -1.66      1.73
```

Read the corners and the nulls before the middle:

- **`null_mean` equals `clean_mean`, −1.6586096286773682, at all 84 points.**
- **Positions 0 to 2 equal the clean run exactly** at every layer: the two prompts share those tokens, so
  their values are the same before any patch.
- **Layer 11 at the last position gives 1.7260316610336304**, the counterfactual run's own `cf_mean`, to the
  bit. Every other position at layer 11 leaves the clean run's score, since no later block moves it.
- **The day's token carries the answer through layer 5**, and between layers 6 and 9 the answer moves to the
  last token. That band is what the map localizes.

The map counts positions, which pairs one word with the same word only because both prompts come from one
template and tokenize to 7 tokens. Check `clab show`'s positions before trusting a counted map, and on pairs of
different lengths map by scope ([pairs, scopes and maps](references/pairs-scopes-and-maps.md)). How labels zip
and cross, a point's cost and reading by coordinate are the `clab` skill's
[sweeps reference](../clab/references/sweeps.md).

## The other recipes

| question | recipe | reference |
|---|---|---|
| the word has several tokens, or the prompts differ in length | scope both sides, `index: -1` | [pairs, scopes and maps](references/pairs-scopes-and-maps.md) |
| does a span of positions carry it jointly | a window, `{"index": "-4:-1"}` | [pairs, scopes and maps](references/pairs-scopes-and-maps.md) |
| restore clean states into a noised run (causal tracing) | corrupt and restore | [pairs, scopes and maps](references/pairs-scopes-and-maps.md) |
| how far the whole prediction or the stream moved | `kl`, `js`, `cosine` | [pairs, scopes and maps](references/pairs-scopes-and-maps.md) |
| patching and another experiment against one clean run | named bundles | [pairs, scopes and maps](references/pairs-scopes-and-maps.md) |
| which attention head, or which set of heads | `attention_premix` with `select` on `num_heads` | [heads, neurons and paths](references/heads-neurons-and-paths.md) |
| which MLP neurons | `mlp_activation` with `select` on `intermediate_size` | [heads, neurons and paths](references/heads-neurons-and-paths.md) |
| a head's direct effect, or its effect on a later head's query | path patching | [heads, neurons and paths](references/heads-neurons-and-paths.md) |
| where a head looks, apart from what it moves | attention-pattern patching | [heads, neurons and paths](references/heads-neurons-and-paths.md) |
| rank every head or position with one backward | attribution patching | [attribution](references/attribution.md) |

Neighbouring tasks live elsewhere. Removing a component (zero, mean or resample ablation, head knockout) is
`clab-ablation`. A trained rotation or mask of a subspace (DAS, DBM) is `clab-trained-featurizers`. Reading an
attention pattern or a logit lens is `clab-lenses-and-attention`. A refusal, or a null that does not equal the
clean run, is `clab-debugging`. The lists are on the machine: `clab describe components` for sites,
`clab describe ops` for metrics and edits, `clab describe pos` for positions.
