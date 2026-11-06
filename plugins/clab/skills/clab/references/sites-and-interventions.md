# Sites and interventions

A site is where an entry reads or writes: a `component`, the `layers` it is taken at, and an optional
`select` that keeps part of one of its axes. Declare it under `sites` when several entries share it, or write it
inline (`"site": {"component": "lm_head_output"}`). Which tokens is the entry's `pos`, not the site's
([positions](positions.md)).

## Components

Names are nnterp's, the same on every model family. The ones an experiment reaches for, from
`clab describe components`:

| component | layout | what it holds |
|---|---|---|
| `token_embeddings` | `rows, pos, hidden_size` | the embedding module's output; takes no `layers` |
| `layer_input`, `layer_output` | `rows, pos, hidden_size` | the residual stream entering and leaving block i |
| `layer_mid` | `rows, pos, hidden_size` | the stream between the sublayers |
| `attention_output`, `mlp_output` | `rows, pos, hidden_size` | what each sublayer adds to the stream |
| `attention_norm_output`, `mlp_norm_output`, `mlp_input` | `rows, pos, hidden_size` | the normed stream a sublayer is called with |
| `attention_premix` | `rows, pos, num_heads, head_dim` | every head's result as the output projection takes it; no eager needed |
| `attention_queries`, `attention_keys`, `attention_values` | `rows, pos, num_(kv_)heads, (qk_)head_dim` | after projection and RoPE; keys and values per KV head |
| `attention_scores`, `attention_probabilities` | `rows, pos, num_heads, key` | the pattern before and after its softmax |
| `attention_head_outputs` | `rows, pos, num_heads, head_dim` | each head's output before the projection |
| `mlp_activation`, `mlp_neurons` | `rows, pos, intermediate_size` | the activation, and what the down projection sums; they differ on a gated MLP |
| `final_norm_output` | `rows, pos, hidden_size` | what the head projects; takes no `layers` |
| `lm_head_output`, `logits` | `rows, pos, vocab_size` | the head's projection; what the model returns, after any softcap |

On GPT-2 these hold to the bit: `layer_input` + `attention_output` is `layer_mid`, `layer_mid` + `mlp_output`
is `layer_output`, block 2's `layer_output` is block 3's `layer_input`, and `lm_head_output` is `logits`.
Check them on another architecture before relying on one. A mixture of experts adds `router_logits`,
`expert_weights`, `expert_indices`, `expert_outputs`, `routed_output` and `shared_expert_output`, whose recipes
are the `clab-ablation` skill's.

### Attention needs eager

`attention_queries`, `attention_keys`, `attention_values`, `attention_scores`, `attention_probabilities` and
`attention_head_outputs` are read inside the attention's forward. On a model loaded with fused attention, which
GPT-2 is by default, they are refused after the model loads:

```text
clab: f.p: the nnterp engine does not serve attention_probabilities: read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'
```

Add `"attn_implementation": "eager"` to the `model` block. `attention_premix` needs no eager. The hooks engine
refuses the interior under either implementation ([running and engines](running-and-engines.md#engines)).

## `layers`

| `layers` | value |
|---|---|
| `8`, `-1` | that layer, no `layers` axis; negative counts from the last |
| `":"`, `"2:6"`, `[1, 0]` | every layer it picks in one forward, a `layers` axis first, labelled with the model's layers in the order picked |
| `{"sweep": [0, 1, 2]}` | one point per layer ([sweeps](sweeps.md)) |

- **`0` and `[0]` read the same numbers**, but `[0]` keeps a `layers` axis of size 1, and the table a `layer` column.
- **The model's own values take no `layers`**: `sites.head: lm_head_output is the model's own value and takes no layers`.
- **A layer past the last compiles** and is refused once the model loads: `clean.v_self: layer 12; the model has 12 layers`.
- **Read across layers with a selector; write one layer at a time with a sweep** ([below](#writes-across-layers)).

## `select`

`select` narrows feature axes, `{<axis>: <selector>}`: `num_heads` or `num_kv_heads`, `head_dim`,
`intermediate_size`, `hidden_size`, `vocab_size`, `num_experts`, `top_k`, or `key` on a pattern.
`documents/heads_select.json` reads two heads of block 9, in the order written:

```json
{
  "description": "Two heads of block 9 by select, in the order written: each head's result at the last token as the output projection takes it, its norm over head_dim, and the argmax over the two, which counts within the select list.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"example_id": "mary", "input": "When Mary and John went to the store, John gave a drink to"},
    {"example_id": "tom", "input": "When Tom and Anna went to the park, Anna gave a ball to"}
  ]}},
  "sites": {"heads": {"component": "attention_premix", "layers": 9, "select": {"num_heads": [9, 6]}}},
  "steps": {
    "f": {"kind": "forward", "input": "pairs.input", "interventions": {
      "size": {"site": "heads", "pos": -1, "ops": [{"op": "norm", "over": ["head_dim"]}]}}},
    "biggest": {"kind": "op", "op": "argmax", "of": "f.size", "over": "num_heads"}
  },
  "saves": ["f.size", "biggest"]
}
```

```text
$ clab table out/heads_select f.size
example_id  row  pos  pos_token  input                                             value
mary        0    13   Ġto        When Mary and John went to the store, John gave…  [4.627506256103516,5.233008861541748]
tom         1    13   Ġto        When Tom and Anna went to the park, Anna gave a…  [9.225279808044434,5.838878154754639]
$ clab table out/heads_select biggest
example_id  row  pos  pos_token  input                                             value
mary        0    13   Ġto        When Mary and John went to the store, John gave…  1
tom         1    13   Ġto        When Tom and Anna went to the park, Anna gave a…  0
```

- **The axis stays, in the order written**: `f.size` is `[2, 1, 2]`, index 0 head 9 and index 1 head 6.
- **An index along a selected axis counts within the list**: `biggest` 1 on row 0 is head 6. Map it back through the list.
- **An operand must carry the same picks in the same order** as the write it meets; build every site of a head set from one sorted list.
- **A write selects on one axis**; a read may select heads and dimensions together.
- **A pick outside the axis, or an axis the component lacks, is refused**: `select num_heads [12] outside the model's 12`; `layer_output has no axis 'num_heads' to select; its axes: hidden_size`.
- **`{"key": {"offset": -1}}` is each query's previous token** on an attention pattern, a read only; the `clab-lenses-and-attention` skill has its recipes.
- **`readout` needs the whole stream, and a metric the whole vocabulary**: select after them, in a second read, never before.

## Read or write

An entry is `{"site": ..., "pos": ..., "ops": [...]}`, binding `<step>.<entry>` to its value after its ops.

- **An entry with an edit in its ops writes; one without only reads.** The edits are `swap`, `add`, `sub`, `scale`, `noise`, `clamp`, `project_out`, `normalize`, `renormalize`, `decode` and `apply`.
- **The arithmetic ops never write.** `mul` by 0 reads a zero and leaves the forward as it was; `scale` with `factor` 0 writes the zero.
- **A write binds what it wrote**: `patched.patch` is the value left at its selection.
- **A read leaves the model only when something names it** (`saves`, a later step or entry); an entry nothing names costs nothing.
- **`pos` defaults to every prompt token**, and `ops` to none.

## The order entries run in

`documents/entry_order.json` reads before and after a write at one site, makes a read whose op only looks like
a write, and writes three layers two ways, every step pulling in one bundle that scores:

```json
{
  "description": "How entries read and write: a read before and after a write at one site, an op that only reads, a write across three layers against three one-layer writes, and a root bundle every step pulls in to score.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"example_id": "mary", "input": "When Mary and John went to the store, John gave a drink to", "io": " Mary", "s": " John"},
    {"example_id": "tom", "input": "When Tom and Anna went to the park, Anna gave a ball to", "io": " Tom", "s": " Anna"}
  ]}},
  "sites": {
    "resid": {"component": "layer_output", "layers": 6},
    "head": {"component": "lm_head_output"}
  },
  "interventions": {
    "scored": {"ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.io", "b": "pairs.s"}]}}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": "scored"},
    "halved": {"kind": "forward", "input": "pairs.input", "interventions": ["scored", {
      "before": {"site": "resid", "pos": -1, "ops": [{"op": "norm"}]},
      "halve": {"site": "resid", "pos": -1, "ops": [{"op": "scale", "factor": 0.5}, {"op": "norm"}]},
      "after": {"site": "resid", "pos": -1, "ops": [{"op": "norm"}]}}]},
    "read_only": {"kind": "forward", "input": "pairs.input", "interventions": ["scored", {
      "times_zero": {"site": "resid", "pos": -1, "ops": [{"op": "mul", "with": 0}]}}]},
    "three": {"kind": "forward", "input": "pairs.input", "interventions": ["scored", {
      "damp": {"site": {"component": "layer_output", "layers": "6:9"}, "pos": -1, "ops": [{"op": "scale", "factor": 0.5}]}}]},
    "three_apart": {"kind": "forward", "input": "pairs.input", "interventions": ["scored", {
      "d6": {"site": {"component": "layer_output", "layers": 6}, "pos": -1, "ops": [{"op": "scale", "factor": 0.5}]},
      "d7": {"site": {"component": "layer_output", "layers": 7}, "pos": -1, "ops": [{"op": "scale", "factor": 0.5}]},
      "d8": {"site": {"component": "layer_output", "layers": 8}, "pos": -1, "ops": [{"op": "scale", "factor": 0.5}]}}]}
  },
  "saves": ["clean.ld", "halved.before", "halved.halve", "halved.after", "halved.ld", "read_only.ld", "three.ld", "three_apart.ld"]
}
```

```text
$ clab table out/entry_order halved.before halved.halve halved.after
example_id  row  pos  pos_token  input                                             halved.before      halved.halve        halved.after
mary        0    13   Ġto        When Mary and John went to the store, John gave…  80.34696960449219  40.173484802246094  40.173484802246094
tom         1    13   Ġto        When Tom and Anna went to the park, Anna gave a…  80.8749771118164   40.4374885559082    40.4374885559082
$ clab table out/entry_order clean.ld halved.ld read_only.ld three.ld three_apart.ld
example_id  row  pos  pos_token  input                                             io     s      clean.ld            halved.ld           read_only.ld        three.ld             three_apart.ld
mary        0    13   Ġto        When Mary and John went to the store, John gave…   Mary   John  2.0063934326171875  2.271759033203125   2.0063934326171875  1.5288887023925781   1.5288887023925781
tom         1    13   Ġto        When Tom and Anna went to the park, Anna gave a…   Tom    Anna  2.5811233520507812  2.2960662841796875  2.5811233520507812  0.16963958740234375  0.16963958740234375
```

- **Inside an entry, edits act on the live value and the last edit's result is written back.** The ops after it shape what the entry binds: `halve` writes half and binds its norm.
- **Entries at one site compose in the order written**: a read listed before a write sees the old value (`before`), one listed after it the written one (`after`).
- **Entries at different sites act as the forward reaches them**, whatever order they are listed in: `scored` comes first in every list, and its head entry reads the written run.
- **Of two writes at one site, the later wins where they overlap.** A freeze that protects a patch goes before it; listed after, it puts the clean value back and every point equals the clean run.
- **`mul` by 0 only reads**: `read_only.ld` is the clean run's to the bit.
- **An edit after an op that changed the layout has no live value to write**, and is refused at compile: `'add' is an edit, which writes back, and it follows 'readout', which turned the value into ['rows', 'pos', 'vocab_size']: there is no live value to write; …`.
- **An op that aggregates over rows is refused in an entry**, which sees one batch: `'mean' aggregates over rows, and inside an entry it sees one batch; make it a step after the forward`.

## Operands

An edit's operand (`with`, `factor`, `scale`, `direction`) is a number or a named value.

- **A number, or a value with no axes, broadcasts everywhere.**
- **A named value may leave out any axis of the write's selection** and broadcasts over it; every axis it keeps must be the selection's, in its order, with the same labels. One vector broadcasts over rows and positions; a `[rows, pos]` metric as `scale`'s `factor` scales each position's vector by its own number.
- **A value with a rows axis belongs to its dataset**, and another dataset's rows refuse it, even over the same file. Keep a counterfactual as a column of its base rows' dataset; a `mean` step removes the rows, and its value writes into any dataset.
- **Positions pair in order**, the operand's i-th with the write's i-th, at a width every row shares ([positions](positions.md#pairing-with-another-forward)).
- **An operand from a narrowed site holds the same heads or neurons in the same order** as the write's `select`.
- **An operand may name an earlier entry of its own forward**, at the same site or one the forward reaches first. It saves a forward, and it may be ragged, since each row pairs with itself.
- **Refused for a same-forward operand**: a later entry, an entry with ops after its edits, a site the forward reaches later, and any of these inside a `generate` step.
- **A later write whose operand is missing on a row it lands on is refused**, naming the row, since it would write NaN.

## Writes across layers

A site whose `layers` is a slice or list makes the entry at each layer it picks, in one forward, each on that
layer's live value. `three.ld` and `three_apart.ld` above are equal: one entry at `"6:9"` is three one-layer
entries.

- **A selector on a write patches every layer it picks at once.** To patch one layer at a time, sweep `layers`, and each point is its own forward.
- **An operand with a `layers` axis is sliced by label at each layer**; one without is the same at every layer; one that lacks a layer the write is made at is refused.
- **What such an entry binds has a `layers` axis first**, and no op inside it reduces over `layers`; reduce in an op step.

## Named bundles

The root `interventions` holds named dicts of entries. A step's `interventions` is a dict, a bundle's name, or a
list of either, run in order: `["scored", {...}]` above.

- **A bundle is a template**: its entries bind names in whichever step pulls it in (`clean.ld`, `halved.ld`).
- **An operand naming one of the bundle's own entries is written bare** (`"with": "src"`); `<step>.src` is refused.
- **An operand naming another step's value** (`"with": "counterfactual.v_cf"`) needs that step before every step that pulls the bundle in.
- **Two entries of one name in one step are refused**, a bundle's included.

## A gradient at a site

An entry whose first op is `grad` reads the gradient of a later entry of its own forward, named bare in `of`,
summed over every axis, with respect to the value at its site. It writes nothing, and the ops after it act on
the gradient. `documents/grad_null.json` takes the gradient of the logit difference at block 6's last token,
and its null:

```json
{
  "description": "A gradient at a site and its null: the gradient of the logit difference at the head with respect to the residual stream leaving block 6 at the last token, its norm, and the gradient of the same metric times 0, which must be exactly zero.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"example_id": "mary", "input": "When Mary and John went to the store, John gave a drink to", "io": " Mary", "s": " John"},
    {"example_id": "tom", "input": "When Tom and Anna went to the park, Anna gave a ball to", "io": " Tom", "s": " Anna"}
  ]}},
  "sites": {
    "resid": {"component": "layer_output", "layers": 6},
    "head": {"component": "lm_head_output"}
  },
  "steps": {
    "f": {"kind": "forward", "input": "pairs.input", "interventions": {
      "g": {"site": "resid", "pos": -1, "ops": [{"op": "grad", "of": "metric"}, {"op": "norm"}]},
      "g_null": {"site": "resid", "pos": -1, "ops": [{"op": "grad", "of": "metric_null"}, {"op": "norm"}]},
      "metric": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.io", "b": "pairs.s"}]},
      "metric_null": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.io", "b": "pairs.s"}, {"op": "mul", "with": 0}]}
    }}
  },
  "saves": ["f.g", "f.g_null", "f.metric"]
}
```

```text
$ clab table out/grad_null f.g f.g_null f.metric
example_id  row  pos  pos_token  input                                             f.g                 f.g_null  io     s      f.metric
mary        0    13   Ġto        When Mary and John went to the store, John gave…  0.3912978768348694  0.0        Mary   John  2.0063934326171875
tom         1    13   Ġto        When Tom and Anna went to the park, Anna gave a…  0.5705670714378357  0.0        Tom    Anna  2.5811233520507812
```

- **The null must be exactly 0**: a `grad` of the metric's entry with `{"op": "mul", "with": 0}` appended. A `grad` of the wrong entry still gives a plausible map.
- **Not `scale` 0 for the null**: after a metric it is refused as an edit, and alone at the head it zeroes the logits every later entry reads.
- **Rows do not mix in a forward**, so each row gets its own gradient.
- **At one site a `grad` reads as a read does**: listed after a write it differentiates the written value.
- **No gradient reaches the model's weights**, and one backward runs per distinct `of`.
- **`grad` is refused in a `generate`, in a fit's body, under `tp` and on the vllm engine.**

Attribution patching built on `grad` is the `clab-patching` skill's.

## The null control

Write back what was read, on the same input, at the same place: its scores must equal the clean run's, as
`null_ld` equals `clean_ld` in the clab skill's worked document. The other edits have nulls of the same kind:
`add` at `"scale": 0.0`, `scale` at `"factor": 1.0`, a permutation that is the identity. Using the nulls to
diagnose a run is the `clab-debugging` skill's.
