# Sweeps

Write `{"sweep": [v1, v2, ...]}` where a value goes, and the document runs once at each value, all on one model
load. A sweep's `label` names its axis; sweeps that share a label pair their points, and sweeps with different
labels make a grid. [SKILL.md](../SKILL.md#an-ablation-a-sweep-a-lens-a-steer) sweeps a site's `layers`; this
page is the rest.

## What may sweep

| may sweep | example |
|---|---|
| a site's `layers` and `select` | `"select": {"num_heads": {"sweep": [0, 1, 2]}}`; a list inside is one point of several heads, `[[6], [9], [6, 9]]` |
| an entry's `pos`: its `frame`, its `index`, a scope's `occurrence` | `{"frame": 0, "index": {"sweep": [-1, -2]}}` |
| an op's arguments, in an entry or a step | `add`'s `scale` (a dose), `scale`'s `factor`, `top_k`'s `k`, `permute`'s `seed` |
| an operand, over value names and numbers | `"with": {"sweep": [0.0, "mean", "clean.acts"], "label": "fill"}` |
| a featurizer's fields, and an `encode`'s, `decode`'s or `apply`'s `featurizer` | `k`, `seed`; `{"sweep": ["sae_a", "sae_b"], "label": "sae"}` |
| a fit's values | `epochs`, `seed`, an objective weight, `lr`, `batch_size` |
| a generate step's options | `max_new_tokens` |

- **Each value is a number, a string, a boolean or a list of them**, listed once: a point is keyed by its value. To vary a `pos` object, sweep a scalar inside it.
- **Not sweepable**: the names that choose what runs (`kind`, `op`, `site`, `component`, `input`, `of`, `train`, `interventions`, `saves`, `file_path`, a scope's `column`), a featurizer's `over`, the optimizer's `name`, and anything under `model` or `data`.
- **Two datasets or two scope columns**: write the steps or the entry twice. **Two checkpoints or revisions**: a run loads one model, so write one document per checkpoint and join their results in Python.

## What a sweep shares

- **A step that references a swept value inherits its points**; one that uses nothing swept runs once and serves every point.
- **One swept site used by several entries is one axis**, so the read at layer L pairs with the write at layer L.
- **A swept write against an unswept read writes the read's one value at every point.** A write `pos` swept alone moves where the counterfactual's value lands, not where it was read: sweep the read's `pos` with the write's, under one label ([a layer-by-position map](patching.md#a-layer-by-position-map)).
- **A read used as a swept write's operand must hold the write's picks**: sweep the read's `select` with the write's.

## Labels: zip or cross

With no `label`, a sweep's label is its field's key, so two `layers` sweeps share `layers` and zip.
`documents/src_dst_cross.json` reads the counterfactual at layer `src` and writes it at layer `dst`, labelled
apart so they cross:

```json
{
  "description": "Read the counterfactual's residual stream at layer src and swap it in at layer dst, for every (src, dst) of layers 6 and 8: two labels, so the two sweeps cross.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"input": "When Mary and John went to the store, John gave a drink to",
     "counterfactual": "When Mary and John went to the store, Mary gave a drink to", "base_answer": " Mary", "cf_answer": " John"},
    {"input": "When Tom and Anna went to the park, Anna gave a ball to",
     "counterfactual": "When Tom and Anna went to the park, Tom gave a ball to", "base_answer": " Tom", "cf_answer": " Anna"},
    {"input": "When Sarah and James went to the office, James gave a book to",
     "counterfactual": "When Sarah and James went to the office, Sarah gave a book to", "base_answer": " Sarah", "cf_answer": " James"},
    {"input": "When Paul and Lisa went to the school, Lisa gave a pen to",
     "counterfactual": "When Paul and Lisa went to the school, Paul gave a pen to", "base_answer": " Paul", "cf_answer": " Lisa"}
  ]}},
  "sites": {
    "src": {"component": "layer_output", "layers": {"sweep": [6, 8], "label": "src"}},
    "dst": {"component": "layer_output", "layers": {"sweep": [6, 8], "label": "dst"}},
    "head": {"component": "lm_head_output"}
  },
  "steps": {
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {"v_cf": {"site": "src", "pos": -1}}},
    "patched": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "dst", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
      "logits": {"site": "head", "pos": -1}}},
    "ld": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "mean_ld": {"kind": "op", "op": "mean", "of": "ld"}
  },
  "saves": ["mean_ld"]
}
```

```text
$ nnspec table out/src_dst_cross mean_ld
at                 value
{"src":6,"dst":6}  -2.7089176177978516
{"src":6,"dst":8}  -0.3985748291015625
{"src":8,"dst":6}  -0.7093219757080078
{"src":8,"dst":8}  3.053426742553711
$ nnspec show out/src_dst_cross
  ...
  step            at           serves        seconds  memory  forwards  tokens  real
  counterfactual  src=6        all 2 points  …        —       1         56      56
  patched         src=6,dst=6  —             …        —       1         56      56
  ...
  result   step     dims  axes      shape
  mean_ld  mean_ld        src, dst  4 points, each []
```

With both `label`s removed the two sweeps zip into the diagonal, two points keyed `{"layers":6}` and
`{"layers":8}`.

| sweeps | labels | points |
|---|---|---|
| two `layers` sweeps over `[6, 8]`, no `label` | `layers`, `layers` | 2, zipped: layer L with layer L |
| the same with `"label": "src"` and `"label": "dst"` | `src`, `dst` | 4, crossed |
| one swept site used by two steps, or one swept bundle pulled into two | one | each step has the same points |
| two options of one generate step both swept, no `label` | two field names | crossed |
| one label in two ops of one entry, or in two entries of a step | one | zipped; a forward per point where any of them writes |
| one label over different values | — | refused: one label is one axis |

- **Check the `axes` column of `nnspec show` before trusting a grid**: a grid that zipped has one axis where you expected two.
- **A label is letters, digits and underscores**, since a coordinate is written `<label>=<value>`.
- **Scattered points are not one axis.** Five chosen (layer, head) pairs under one label are refused as one label over different values. Cross the layers and heads they use and read the cells you want, or write one site per pair ([confirm the top points](patching-attribution.md#confirm-the-top-points)).
- **A control at one value takes no sweep, or its own label**: a plain number repeats beside every point of the other sweep, where a one-value sweep under the other's label is refused.
- **A write's `pos` sweep wants its own label** (`"label": "patch_pos"`), so its coordinate is not confused with the table's `pos` column.

## What a sweep costs

- **Points that differ only in what they read share one forward**: `counterfactual` above runs once for both `src` points (`serves all 2 points`).
- **Each point that changes a write, an operand or an option runs its own forward**: `patched` runs four. A twelve-layer patch with its null is 26 forwards, two reads and twelve of each write.
- **A document does not batch different interventions into one forward.** Where a task asks for many interventions as rows of a single forward, that is the nnsight skill's work.
- **A large grid grows the manifest and compile time** faster than the points; split it into documents by layer range.

## Widen the read instead

When a sweep only chooses which part of one forward to read, one wider read gives the same numbers as one value
with an axis:

| swept read | one wider read |
|---|---|
| `"layers": {"sweep": [0, 1]}` | `"layers": ":"`, a `layers` axis |
| `"pos": {"sweep": [-2, -1]}` | `{"index": "-2:"}` |
| `{"frame": {"sweep": [0, 1, 2]}, "index": -1}` in a generation | `{"frame": "0:3"}` |

An op sees one point at a time, so it reduces or compares along an axis but never across points: a `mean` over
`layers` works on a `layers: ":"` read and is refused on a swept one. Sweep when each point is a different
intervention.

## Reading swept results

A swept save holds one value per point, keyed by its coordinate, its value on each axis:

| where | key | example |
|---|---|---|
| `nnspec table`, `plan.table`: column `at` | an object | `{"src":6,"dst":8}` |
| `plan.results[name]` | a plain dict: the value for one axis, a tuple in axis order for several | `8`, `(6, 8)` |
| output files, `nnspec show`, `plan.run.costs`, `plan.run.tokens` | `label=value` joined by commas | `src=6,dst=8` |

```python
from nnspec import Plan

plan = Plan.load("out/src_dst_cross")
plan.sweeps                              # {'src': Axis(values=[6, 8]), 'dst': Axis(values=[6, 8])}
list(plan.results["mean_ld"])            # [(6, 6), (6, 8), (8, 6), (8, 8)]
plan.results["mean_ld"][(6, 8)]          # tensor(-0.3986)
```

- **A coordinate tuple follows `plan.sweeps`**, whose axes follow the format's root keys (`data`, `sites`, `interventions`, `featurizers`, `steps`), then document order inside each. Print `plan.sweeps` rather than assume the order: a `select` sweep on a site declared before a `layers` sweep puts the head first.
- **A result is keyed by the axes its own value depends on** (its `axes` in `nnspec show`); a read that depends on none is one value, whatever its step's sweeps.
- **A swept `pos`'s coordinate is not the table's `pos` column**: `at` holds the swept value, `pos` the position the tabled read landed on.
- **Values swept over different axes cross in a table**, a row per example at every pair of points; table them one at a time.
