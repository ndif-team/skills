# Heads, neurons and paths

Narrow a patch from a whole residual stream to one attention head, a set of heads, a block of MLP neurons, one
path, or where a head looks. Each recipe is [patching](patching.md)'s interchange at another site, on GPT-2's
indirect-object task: the base prompt answers the first name, and the counterfactual, with the other name
repeated, answers the second.

## What breaks

- **A freeze listed after the patch it protects overwrites it**: entries on one component and layer run in the
  order written, so every point equals the clean run, and the path nulls still pass
  ([path patching](#path-patching)).
- **The nulls also pass when a patch lands on the wrong head**, since they write clean values. Check the site
  each point wrote ([below](#check-where-a-head-write-landed)).
- **`attention_keys` and `attention_values` are per KV head.** Under grouped-query attention one KV head serves
  several query heads, so patching it moves all of them; `attention_premix` is one query head's result.
- **Eager attention moves the last bits.** `attention_probabilities` and `attention_queries` need
  `"attn_implementation": "eager"` in `model`, and their nulls equal an eager clean run, not an sdpa one.

## One head at a time

`attention_premix` is every head's result as the attention's output projection takes it, `[rows, pos,
num_heads, head_dim]`. A `select` on `num_heads` keeps one head, and the write replaces that head's share of
what attention adds to the stream. The sweeps of `layers` and `num_heads` have different labels, so they
cross: 144 points on GPT-2. Saved as `documents/head_patching.json`:

```json
{
  "description": "Head patching on GPT-2's indirect-object task: one head's result (attention_premix) at the last token, read on the counterfactual (the other name repeated) and swapped into the base, for every head of every layer, 144 points. The null reads the head on the base and writes it back in the same forward.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"example_id": "Mary-John", "input": "When Mary and John went to the store, John gave a drink to",
     "counterfactual": "When Mary and John went to the store, Mary gave a drink to", "base_answer": " Mary", "cf_answer": " John"},
    {"example_id": "John-Mary", "input": "When John and Mary went to the store, Mary gave a drink to",
     "counterfactual": "When John and Mary went to the store, John gave a drink to", "base_answer": " John", "cf_answer": " Mary"},
    {"example_id": "Alice-Bob", "input": "When Alice and Bob went to the store, Bob gave a drink to",
     "counterfactual": "When Alice and Bob went to the store, Alice gave a drink to", "base_answer": " Alice", "cf_answer": " Bob"},
    {"example_id": "Bob-Alice", "input": "When Bob and Alice went to the store, Alice gave a drink to",
     "counterfactual": "When Bob and Alice went to the store, Bob gave a drink to", "base_answer": " Bob", "cf_answer": " Alice"},
    {"example_id": "Sarah-David", "input": "When Sarah and David went to the store, David gave a drink to",
     "counterfactual": "When Sarah and David went to the store, Sarah gave a drink to", "base_answer": " Sarah", "cf_answer": " David"},
    {"example_id": "David-Sarah", "input": "When David and Sarah went to the store, Sarah gave a drink to",
     "counterfactual": "When David and Sarah went to the store, David gave a drink to", "base_answer": " David", "cf_answer": " Sarah"},
    {"example_id": "Anna-Mark", "input": "When Anna and Mark went to the store, Mark gave a drink to",
     "counterfactual": "When Anna and Mark went to the store, Anna gave a drink to", "base_answer": " Anna", "cf_answer": " Mark"},
    {"example_id": "Mark-Anna", "input": "When Mark and Anna went to the store, Anna gave a drink to",
     "counterfactual": "When Mark and Anna went to the store, Mark gave a drink to", "base_answer": " Mark", "cf_answer": " Anna"}
  ]}},
  "sites": {
    "one_head": {"component": "attention_premix", "layers": {"sweep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
                 "select": {"num_heads": {"sweep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}}},
    "head": {"component": "lm_head_output"}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": {"logits": {"site": "head", "pos": -1}}},
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {
      "v_cf": {"site": "one_head", "pos": -1}}},
    "null": {"kind": "forward", "input": "pairs.input", "interventions": {
      "v_self": {"site": "one_head", "pos": -1},
      "patch": {"site": "one_head", "pos": -1, "ops": [{"op": "swap", "with": "null.v_self"}]},
      "logits": {"site": "head", "pos": -1}}},
    "patched": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "one_head", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
      "logits": {"site": "head", "pos": -1}}},
    "clean_top": {"kind": "op", "op": "match", "of": "clean.logits", "expected": "pairs.base_answer"},
    "clean_ld": {"kind": "op", "op": "logit_diff", "of": "clean.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "null_ld": {"kind": "op", "op": "logit_diff", "of": "null.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "ld": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "clean_mean": {"kind": "op", "op": "mean", "of": "clean_ld"},
    "null_mean": {"kind": "op", "op": "mean", "of": "null_ld"},
    "ld_mean": {"kind": "op", "op": "mean", "of": "ld"}
  },
  "saves": ["clean_top", "clean_ld", "clean_mean", "null_mean", "ld_mean"]
}
```

```text
$ nnspec table out/head_patching ld_mean null_mean
at                            ld_mean              null_mean
{"layers":0,"num_heads":0}    -2.936856269836426   -2.9367809295654297
{"layers":8,"num_heads":10}   -1.7821941375732422  -2.9367809295654297
{"layers":9,"num_heads":9}    -1.9188365936279297  -2.9367809295654297
{"layers":10,"num_heads":7}   -4.498139381408691   -2.9367809295654297
{"layers":11,"num_heads":10}  -4.131328582763672   -2.9367809295654297
```

- **The clean run does the task**: `clean_top` is 1.0 on all 8 rows, and `clean_mean` is -2.9367809295654297.
- **The null reads and writes in one forward**: its operand, `null.v_self`, is an earlier entry of the same
  step, at the one site both name. `null_mean` equals `clean_mean` at all 144 points.
- **No op compares across sweep points**: rank the heads in Python, from the result keyed `(layer, head)`.

```python
from nnspec import Plan

plan = Plan.load("out/head_patching")
clean = plan.results["clean_mean"].item()
effect = {at: v.item() - clean for at, v in plan.results["ld_mean"].items()}   # keyed (layer, head)
ranked = sorted(effect.items(), key=lambda kv: kv[1], reverse=True)
for (layer, head), e in ranked[:5] + ranked[-2:]:
    print(f"{layer}.{head}  {e:+.3f}")
```

```text
8.10  +1.155
9.9  +1.018
7.9  +0.906
8.6  +0.817
7.3  +0.674
11.10  -1.195
10.7  -1.561
```

The signed order separates heads that push toward the counterfactual's name from heads that push against it;
sort by `abs(e)` for the heads that matter most either way, 10.7 first.

### Check where a head write landed

The compiled entry's `site` at each point is in the run record:

```python
from nnspec import Plan
for point in Plan.load("out/head_sets").steps["set9"]:
    print(point.at, point.interventions["patch"].site)
```

On the head-set document below its last line is `{'num_heads': 2} component='attention_premix' layers=9
select={'num_heads': [6, 9]}`. A point's `at` there is its index on each sweep axis, not the swept value.

## Head sets

A list inside a `select` sweep is one point that patches those heads together. Heads at different layers are
one write per layer, in one forward. These `sites` and forwards replace `head_patching.json`'s, in
`head_sets.json`; each forward's logits take a `logit_diff` and a `mean` step as there (`set9_mean`,
`across_mean`):

```json
{
  "sites": {
    "layer9": {"component": "attention_premix", "layers": 9, "select": {"num_heads": {"sweep": [[6], [9], [6, 9]]}}},
    "l7": {"component": "attention_premix", "layers": 7, "select": {"num_heads": [3, 9]}},
    "l8": {"component": "attention_premix", "layers": 8, "select": {"num_heads": [6, 10]}},
    "head": {"component": "lm_head_output"}
  },
  "steps": {
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {
      "v9": {"site": "layer9", "pos": -1}, "v7": {"site": "l7", "pos": -1}, "v8": {"site": "l8", "pos": -1}}},
    "set9": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "layer9", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v9"}]},
      "logits": {"site": "head", "pos": -1}}},
    "across": {"kind": "forward", "input": "pairs.input", "interventions": {
      "p7": {"site": "l7", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v7"}]},
      "p8": {"site": "l8", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v8"}]},
      "logits": {"site": "head", "pos": -1}}}
  }
}
```

```text
$ nnspec table out/head_sets set9_mean
at                   value
{"num_heads":[6]}    -3.3835153579711914
{"num_heads":[9]}    -1.9188365936279297
{"num_heads":[6,9]}  -2.1071090698242188
$ nnspec table out/head_sets across_mean
value
1.9949283599853516
```

Heads 7.3, 7.9, 8.6 and 8.10 patched together move the mean logit difference from -2.94 to +1.99, an effect of
+4.93 where their separate effects sum to +3.55: a set's effect is a finding of its own. Chosen heads patched
one at a time are one site per head ([attribution](patching-attribution.md#confirm-the-top-points)).

## Neurons

`mlp_activation` with a `select` on `intermediate_size` patches a block of neurons; a slice string is a block.
`neuron_patching.json` is `head_patching.json` with `one_head` redefined as layer 9's neurons, a quarter at a
time and then all of them, a site `mlp_out`, an entry `"o_cf": {"site": "mlp_out", "pos": -1}` in
`counterfactual`, and one more forward, scored as `mlp_mean` the way `patched` is scored as `ld_mean`:

```json
{
  "sites": {
    "one_head": {"component": "mlp_activation", "layers": 9,
                 "select": {"intermediate_size": {"sweep": ["0:768", "768:1536", "1536:2304", "2304:3072", ":"]}}},
    "mlp_out": {"component": "mlp_output", "layers": 9}
  },
  "steps": {
    "whole_mlp": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "mlp_out", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.o_cf"}]},
      "logits": {"site": "head", "pos": -1}}}
  }
}
```

```text
$ nnspec table out/neuron_patching clean_mean mlp_mean
clean_mean           mlp_mean
-2.9367809295654297  -2.5706310272216797
$ nnspec table out/neuron_patching ld_mean null_mean
at                                 ld_mean              null_mean
{"intermediate_size":"0:768"}      -2.6902027130126953  -2.9367809295654297
{"intermediate_size":"768:1536"}   -2.927596092224121   -2.9367809295654297
{"intermediate_size":"1536:2304"}  -2.7341108322143555  -2.9367809295654297
{"intermediate_size":"2304:3072"}  -3.0372772216796875  -2.9367809295654297
{"intermediate_size":":"}          -2.5706310272216797  -2.9367809295654297
```

- **Every neuron at once equals `mlp_output` to the bit**, the positive control: the down projection is linear,
  so the neurons are the MLP's whole contribution. The null equals the clean run at every block.
- **On a gated MLP `mlp_activation` and `mlp_neurons` are different values**: the gate's activation alone, and
  what the down projection sums. Patching every neuron of SmolLM2-135M-Instruct's layer 9 through one and
  through the other gives logits up to 0.09 apart, where the patch moves them by 0.10; on GPT-2's plain MLP
  the two agree to the bit. Say which one a result patched.
- **To rank single neurons**, use [attribution](patching-attribution.md#one-prompt-no-counterfactual). Neurons
  in several layers are one site per layer, since a `select` applies at every layer its site picks.

## Path patching

Patching head h measures everything h changes downstream, including what later heads make of it. Path patching
keeps one path: h's effect on the logits through no other head. Read every head's `attention_premix` at the
last token on the clean prompt (`layers: ":"`). In the patched forward list a freeze that swaps it back, then
the patch of head h: the patch overrides the freeze at h alone, and the MLPs and the stream recompute.
These keys replace `head_patching.json`'s `sites` and `steps`, in `path_patching.json`:

```json
{
  "sites": {
    "every_head": {"component": "attention_premix", "layers": ":"},
    "one_head": {"component": "attention_premix", "layers": {"sweep": [8, 9, 10, 11]},
                 "select": {"num_heads": {"sweep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}}},
    "every_layer": {"component": "attention_premix", "layers": ":",
                    "select": {"num_heads": {"sweep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}}},
    "head": {"component": "lm_head_output"}
  },
  "interventions": {
    "freeze": {"freeze": {"site": "every_head", "pos": -1, "ops": [{"op": "swap", "with": "clean.every_head"}]}},
    "score": {"ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]}}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"every_head": {"site": "every_head", "pos": -1}}, "score"]},
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {
      "v_cf": {"site": "every_layer", "pos": -1}}},
    "own": {"kind": "forward", "input": "pairs.input", "interventions": {"v": {"site": "every_layer", "pos": -1}}},
    "frozen": {"kind": "forward", "input": "pairs.input", "interventions": ["freeze", "score"]},
    "path": {"kind": "forward", "input": "pairs.input", "interventions": ["freeze",
      {"patch": {"site": "one_head", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]}}, "score"]},
    "path_null": {"kind": "forward", "input": "pairs.input", "interventions": ["freeze",
      {"patch": {"site": "one_head", "pos": -1, "ops": [{"op": "swap", "with": "own.v"}]}}, "score"]}
  }
}
```

Two more forwards are `path` without `"freeze"` (`plain`) and `path` with `"freeze"` listed after the patch
(`wrong_order`). A `mean` step over each forward's `ld` is saved (`clean_mean`, `frozen_mean`, `path_mean`,
`path_null_mean`, `plain_mean`, and `wrong_mean`). The counterfactual reads `every_layer`, and each point of
the patch takes the slice labelled with its layer.

```text
$ nnspec table out/path_patching clean_mean frozen_mean
clean_mean           frozen_mean
-2.9367809295654297  -2.9367809295654297
$ nnspec table out/path_patching path_mean plain_mean path_null_mean wrong_mean
at                            path_mean            plain_mean           path_null_mean       wrong_mean
{"layers":8,"num_heads":6}    -2.948061943054199   -2.1201019287109375  -2.9367809295654297  -2.9367809295654297
{"layers":9,"num_heads":9}    0.8731622695922852   -1.9188365936279297  -2.9367809295654297  -2.9367809295654297
{"layers":11,"num_heads":10}  -4.131328582763672   -4.131328582763672   -2.9367809295654297  -2.9367809295654297
```

- **`frozen_mean` equals `clean_mean`**: the freeze alone writes back what it read.
- **`path_null_mean` equals `clean_mean` at all 48 points**: the patch sourced from the base prompt.
- **At layer 11 `path_mean` equals `plain_mean` for every head**: nothing after the last layer is frozen.
- **`wrong_mean` equals `clean_mean` at all 48 points**: the freeze, listed after the patch, erased it.

A path reads differently from a plain patch. Head 9.9's direct path is +3.81 against +1.02 for the plain patch,
since freezing the later heads removes what they do to counter it; head 8.6's is -0.01 against +0.82, since its
plain effect passes through later heads.

### To a later head's query

Read the receiver's query in the path-patched forward, then swap it into a fresh forward of the base prompt, so
only that query moves. `path_to_query.json` is `path_patching.json` under `"attn_implementation": "eager"`,
with the sites `sender` (`one_head` at `"layers": {"sweep": [7, 8]}`) and `receiver` in place of `one_head`,
and these steps in place of `frozen`, `path`, `path_null`, `plain` and `wrong_order`; it saves `clean_mean`,
`received_mean` and `null_mean`:

```json
{
  "sites": {"receiver": {"component": "attention_queries", "layers": 9, "select": {"num_heads": [9]}}},
  "steps": {
    "path": {"kind": "forward", "input": "pairs.input", "interventions": ["freeze",
      {"patch": {"site": "sender", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
       "q": {"site": "receiver", "pos": -1}}]},
    "received": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"q": {"site": "receiver", "pos": -1, "ops": [{"op": "swap", "with": "path.q"}]}}, "score"]}
  }
}
```

`path_null` and `received_null` are the same two steps with `own.v` for the counterfactual's value, and
`received_mean` and `null_mean` the means of the two `ld`s.

```text
$ nnspec table out/path_to_query received_mean null_mean
at                           received_mean        null_mean
{"num_heads":3,"layers":7}   -2.865382194519043   -2.936779022216797
{"num_heads":6,"layers":8}   -2.6842498779296875  -2.936779022216797
{"num_heads":9,"layers":7}   -2.7898406982421875  -2.936779022216797
{"num_heads":10,"layers":8}  -2.711958885192871   -2.936779022216797
```

- **These four are the largest senders into 9.9's query**; the next is 8.5 at +0.004. `null_mean` equals the
  eager `clean_mean`, -2.936779022216797, at all 24 points.
- **The axes follow the order the document declares their sweeps**: `num_heads` comes first here, from
  `every_layer`, so a result's key is `(head, layer)`. Read the order from `plan.sweeps`.
- **Queries and keys are taken after any rotary embedding**, so a query carries its prompt's positions: patch
  one only between prompts that tokenize to one length.

## Attention-pattern patching

Patch `attention_probabilities` to make a head attend as it did on the counterfactual while its values stay the
base's: does where the head looks carry the answer, apart from what it moves? `pattern_patching.json` is
`head_patching.json` with `"attn_implementation": "eager"` in `model`, and `"component":
"attention_probabilities"` and `"layers": {"sweep": [9, 10]}` in `one_head`:

```text
$ nnspec table out/pattern_patching ld_mean null_mean
at                            ld_mean              null_mean
{"layers":9,"num_heads":6}    -3.3977251052856445  -2.936779022216797
{"layers":9,"num_heads":9}    -1.8761281967163086  -2.936779022216797
{"layers":10,"num_heads":0}   -2.4961042404174805  -2.936779022216797
{"layers":10,"num_heads":7}   -4.8377685546875     -2.936779022216797
```

- **Head 9.9's pattern alone moves the logit difference by +1.06**, about its whole result's +1.02: where 9.9
  looks carries its effect. `null_mean` equals the eager `clean_mean` at all 24 points.
- **A pattern swaps only between prompts of one length**, since key k must be the same token on both sides: on
  prompts of 9 and 10 tokens it is refused (`… so their key axes do not line up`).
