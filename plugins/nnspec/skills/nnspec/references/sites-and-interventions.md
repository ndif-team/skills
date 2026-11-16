# Sites and interventions

A site is where an entry reads or writes: a `component`, the `layers` it is taken at, and an optional `select`
that keeps part of one of its axes. Declare it under `sites` when several entries share it, or write it inline
(`"site": {"component": "lm_head_output"}`). Which tokens is the entry's `pos` ([positions](positions.md)).

## Components

Names are nnterp's, the same on every model family. `nnspec describe components` lists each with its layout,
the axes that narrow it, and a `stream` mark on those `readout` accepts. [SKILL.md](../SKILL.md#the-document)
has the six most documents use; reach for the others as follows:

| component | reach for it to |
|---|---|
| `token_embeddings` | read or corrupt the embedding module's output; takes no `layers` |
| `layer_mid` | split a block between its sublayers: the stream after attention, before the MLP |
| `attention_norm_output`, `mlp_norm_output`, `mlp_input` | read what a sublayer is called with, as a probe or SAE trained on normed activations does; a write there reaches that sublayer only |
| `attention_scores` | the pattern before its softmax: write a large negative score to stop a query attending a key |
| `attention_queries`, `attention_keys`, `attention_values` | a head's query and key geometry after RoPE, or what it moves from a position; keys and values are per KV head (`num_kv_heads`) |
| `attention_head_outputs` | each head's output read inside the attention; it needs eager where `attention_premix` does not |
| `mlp_neurons` | what the down projection sums; on a gated MLP it differs from `mlp_activation`, which is the gate alone |
| `final_norm_output` | the model's last representation, what the head projects; takes no `layers` |
| `logits` | what the model returns, after any softcap: score it where a family has one; the only component `--readout at-reads` acts on |
| `router_logits`, `expert_weights`, `expert_indices`, `expert_outputs`, `routed_output`, `shared_expert_output` | a mixture of experts' routing and expert values ([expert knockout](ablation-attention-and-experts.md#expert-knockout)) |

On GPT-2 these hold to the bit: `layer_input` plus `attention_output` is `layer_mid`, `layer_mid` plus
`mlp_output` is `layer_output`, block 3's `layer_output` is block 4's `layer_input`, `mlp_input` is
`mlp_norm_output`, and `lm_head_output` is `logits`. Check them on another architecture before relying on one.

### Attention needs eager

`attention_queries`, `attention_keys`, `attention_values`, `attention_scores`, `attention_probabilities` and
`attention_head_outputs` are read inside the attention's forward, so the `model` block needs
`"attn_implementation": "eager"`. Under fused attention, GPT-2's default, they are refused after the model
loads, and the refusal names the key. `attention_premix` needs no eager. An eager run differs from an `sdpa`
run in the last bits, so compare runs under one implementation. The hooks engine refuses the interior under
either ([engines](running-and-engines.md#engines)).

## `layers`

- **An int is that layer, with no `layers` axis**; negative counts from the last. `0` and `[0]` read the same numbers, but `[0]` keeps a `layers` axis of size 1, and the table a `layer` column.
- **A slice or list (`":"`, `"2:6"`, `[1, 0]`) is every layer it picks in one forward**, a `layers` axis first, labelled with the model's layers in the order picked.
- **A sweep is one point per layer** ([sweeps](sweeps.md)). Read across layers with a selector; write one layer at a time with a sweep ([below](#writes-across-layers)).
- **The model's own values take no `layers`**: `token_embeddings`, `final_norm_output`, `lm_head_output`, `logits`.

## `select`

`select` narrows feature axes, `{<axis>: <selector>}`: `num_heads` or `num_kv_heads`, `head_dim`,
`intermediate_size`, `hidden_size`, `vocab_size`, `num_experts`, `top_k`, or `key` on a pattern.

- **The axis stays, in the order written**: `{"num_heads": [9, 6]}` at the last token is `[rows, 1, 2, head_dim]`, index 0 head 9 and index 1 head 6.
- **An index along a selected axis counts within the list**: `argmax` over those two heads gives 1 for head 6, and `nnspec table` prints the index kept. Map it back through the list.
- **An operand must carry the same picks in the same order** as the write it meets. Declare one site and name it in both the read and the write, or build every site of a head set from one sorted list.
- **A write selects on one axis**; a read may select heads and dimensions together.
- **A `select` applies at every layer of its site.** Other picks at each layer are one site per layer, each with its own `select`.
- **`{"key": {"offset": -1}}` is each query's previous token** on an attention pattern, a read only; a plain `key` index counts from the row's start, and one past the input's shortest row is refused ([attention](attention.md)).
- **`readout` needs the whole stream, and a metric the whole vocabulary**: select after them, in a second read, never before.

## Read or write

An entry is `{"site": ..., "pos": ..., "ops": [...]}`, binding `<step>.<entry>` to its value after its ops.

- **An entry with an edit in its ops writes; one without only reads.** The edits are the `edits` group of `nnspec describe ops`, a featurizer's `decode` and `apply` among them.
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
$ nnspec table out/entry_order halved.before halved.halve halved.after
example_id  row  pos  pos_token  input                                             halved.before      halved.halve        halved.after
mary        0    13   Ġto        When Mary and John went to the store, John gave…  80.34696960449219  40.173484802246094  40.173484802246094
tom         1    13   Ġto        When Tom and Anna went to the park, Anna gave a…  80.8749771118164   40.4374885559082    40.4374885559082
$ nnspec table out/entry_order clean.ld halved.ld read_only.ld three.ld three_apart.ld
example_id  row  pos  pos_token  input                                             io     s      clean.ld            halved.ld           read_only.ld        three.ld             three_apart.ld
mary        0    13   Ġto        When Mary and John went to the store, John gave…   Mary   John  2.0063934326171875  2.271759033203125   2.0063934326171875  1.5288887023925781   1.5288887023925781
tom         1    13   Ġto        When Tom and Anna went to the park, Anna gave a…   Tom    Anna  2.5811233520507812  2.2960662841796875  2.5811233520507812  0.16963958740234375  0.16963958740234375
```

- **Inside an entry, edits act on the live value and the last edit's result is written back.** The ops after it shape what the entry binds: `halve` writes half and binds its norm.
- **Entries on one component and layer compose in the order written, whatever their site names.** A read listed before a write sees the old value (`before`), one listed after it the written one (`after`).
- **Entries at different sites act as the forward reaches them**, whatever order they are listed in: `scored` comes first in every list, and its head entry reads the written run.
- **Of two writes on one component and layer, the later wins where they overlap.** A freeze that protects a patch goes before it; listed after, it puts the clean value back and every point equals the clean run ([path patching](patching-heads-neurons-and-paths.md#path-patching)).
- **`mul` by 0 only reads**: `read_only.ld` is the clean run's to the bit.
- **An edit after an op that changed the layout has no live value to write**, and an op that aggregates over rows sees one batch inside an entry: both are refused at compile, naming the fix.

## Operands

An edit's operand (`with`, `factor`, `scale`, `direction`) is a number or a named value.

- **A number, or a value with no axes, broadcasts everywhere.**
- **A named value may leave out any axis of the write's selection** and broadcasts over it; every axis it keeps must be the selection's, in its order, with the same labels. One vector broadcasts over rows and positions; a `[rows, pos]` metric as `scale`'s `factor` scales each position's vector by its own number.
- **A value with a rows axis belongs to its dataset**, and another dataset's rows refuse it, even over the same file. A `mean` step removes the rows, and its value writes into any dataset.
- **Positions pair in order**, the operand's i-th with the write's i-th, at a width every row shares ([positions](positions.md#pairing-with-another-forward)).
- **An operand may name an earlier entry of its own forward**, at the same site or one the forward reaches first. It saves a forward, and it may be ragged, since each row pairs with itself. A later entry, an entry with ops after its edits, a site the forward reaches later, and any of these inside a `generate` step are refused.
- **A later write whose operand is missing on a row it lands on is refused**, naming the row, since it would write NaN.

## Writes across layers

A site whose `layers` is a slice or list makes the entry at each layer it picks, in one forward, each on that
layer's live value. `three.ld` and `three_apart.ld` above are equal: one entry at `"6:9"` is three one-layer
entries.

- **To patch one layer at a time, sweep `layers`**: each point is its own forward.
- **An operand with a `layers` axis is sliced by label at each layer**; one without is the same at every layer; one that lacks a layer the write is made at is refused.
- **What such an entry binds has a `layers` axis first**, and no op inside it reduces over `layers`; reduce in an op step.

## Named bundles

The root `interventions` holds named dicts of entries. A step's `interventions` is a dict, a bundle's name, or a
list of either, run in order: `["scored", {...}]` above.

- **A bundle is a template**: its entries bind names in whichever step pulls it in (`clean.ld`, `halved.ld`), so two experiments against one clean run are two bundles, each pulled into its own step.
- **An operand naming one of the bundle's own entries is written bare** (`"with": "src"`); `<step>.src` is refused.
- **An operand naming another step's value** needs that step before every step that pulls the bundle in.

## A gradient at a site

An entry whose first op is `grad` reads the gradient of a later entry of its own forward, named bare in `of`,
summed over every axis, with respect to the value at its site. It writes nothing, and the ops after it act on
the gradient. `documents/grad_null.json` is `entry_order.json` with these `steps` and
`"saves": ["f.g", "f.g_null", "f.metric"]`: the gradient of the logit difference at block 6's last token, and
its null.

```json
{
  "f": {"kind": "forward", "input": "pairs.input", "interventions": {
    "g": {"site": "resid", "pos": -1, "ops": [{"op": "grad", "of": "metric"}, {"op": "norm"}]},
    "g_null": {"site": "resid", "pos": -1, "ops": [{"op": "grad", "of": "metric_null"}, {"op": "norm"}]},
    "metric": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.io", "b": "pairs.s"}]},
    "metric_null": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.io", "b": "pairs.s"}, {"op": "mul", "with": 0}]}
  }}
}
```

```text
$ nnspec table out/grad_null f.g f.g_null f.metric
example_id  row  pos  pos_token  input                                             f.g                 f.g_null  io     s      f.metric
mary        0    13   Ġto        When Mary and John went to the store, John gave…  0.3912978768348694  0.0        Mary   John  2.0063934326171875
tom         1    13   Ġto        When Tom and Anna went to the park, Anna gave a…  0.5705670714378357  0.0        Tom    Anna  2.5811233520507812
```

- **The null must be exactly 0**: a `grad` of the metric's entry with `{"op": "mul", "with": 0}` appended. It proves the gradient is of the entry you zeroed and the wiring exact, not that the entry is the metric you report ([check the gradient](patching-attribution.md#check-the-gradient)).
- **Write the null with `mul`, never `scale`**: `scale` is an edit, refused after a metric, and alone at the head it would write.
- **`of` is an entry of the same forward at a site the forward reaches later**, named without its step. Rows do not mix in a forward, so each row gets its own gradient, and one backward runs per distinct `of`.
- **At one site a `grad` reads as a read does**: listed after a write it differentiates the written value. No gradient reaches the model's weights.
- **`grad` is refused in a `generate`, in a fit's body, under `tp` and on the vllm engine.**

## The null control

Write back what was read, on the same input, at the same place: its scores must equal the clean run's, as
`null_ld` equals `clean_ld` in [SKILL.md](../SKILL.md#a-complete-experiment). The other edits have nulls of the
same kind: `add` at `"scale": 0.0`, `scale` at `"factor": 1.0`, a `permute` under `"seed": null`
([the nulls](debugging.md#the-nulls)).
