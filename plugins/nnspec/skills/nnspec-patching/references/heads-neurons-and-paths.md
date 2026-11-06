# Heads, neurons and paths

These recipes narrow the patched site to one head's result, a set of heads, a block of MLP neurons, one path,
or where a head looks. They run on GPT-2's indirect-object task (IOI): "When Mary and John went to the store,
John gave a drink to" answers ` Mary`, and the counterfactual ("…, Mary gave a drink to") answers ` John`.

## What breaks

- **A freeze listed after the patch it protects overwrites it.** Every point then equals the clean run, and the
  path nulls still pass: the wrong order below gives the clean −2.9367809295654297 at all 48 points.
- **The path nulls also pass when the patch lands on the wrong head**, since they write clean values. Check the
  site each point wrote, in the manifest (below).
- **An operand from a narrowed site must hold the write's heads in the write's order.** Declare one site and
  name it in both the read and the write, and the two agree by construction.
- **`attention_probabilities` and `attention_queries` need eager attention**
  ([sites and interventions](../../clab/references/sites-and-interventions.md#attention-needs-eager));
  compare runs under one implementation, since eager moves the last bits.
- **A pattern swapped between prompts of different lengths is refused**, since key k must be the same token on
  both sides:

  ```text
  clab: patched.look_there at layers=9,num_heads=0: operand 'counterfactual.their_pattern' was read on pairs.counterfactual, row 0 of which is 9 tokens; pairs.input has 10 there, so their key axes do not line up
  ```

- **On a gated MLP, `mlp_activation` (`act(gate x)`) and `mlp_neurons` (what the down projection sums) differ**,
  and patching one moves the logits differently from patching the other (on SmolLM2-135M-Instruct). On GPT-2's
  plain MLP the two patches are equal to the bit. Say which one a result patched.

## One head at a time

`attention_premix` is every head's result as the attention's output projection takes it, laid out `[rows, pos,
num_heads, head_dim]`. A `select` on `num_heads` keeps one head, and the write replaces that head's share of
what attention adds to the stream. Sweeping `layers` and `num_heads` crosses them, since their labels differ:
144 points on GPT-2.

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

The clean run puts the indirect object on top on all 8 rows (`clean_top` 1.0; `clean_ld` −2.01 to −4.08, mean
−2.9367809295654297). The null's operand is an earlier entry of its own forward.

```text
$ clab table out/head_patching ld_mean null_mean
at                            ld_mean              null_mean
{"layers":0,"num_heads":0}    -2.936856269836426   -2.9367809295654297
{"layers":8,"num_heads":10}   -1.7821941375732422  -2.9367809295654297
{"layers":9,"num_heads":9}    -1.9188365936279297  -2.9367809295654297
{"layers":10,"num_heads":7}   -4.498139381408691   -2.9367809295654297
{"layers":11,"num_heads":10}  -4.131328582763672   -2.9367809295654297
```

`null_mean` equals `clean_mean` at all 144 points. No op compares across points, so rank the heads in Python,
from the result keyed `(layer, head)`:

<!-- test: skip -->
```python
from clab import Plan

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

These are GPT-2's known IOI heads: 7.3, 7.9, 8.6 and 8.10 (S-inhibition) and 9.9 (a name mover) push toward the
counterfactual's name, and 10.7 and 11.10 (negative name movers) push against it. The signed order separates the
two; sort by `abs(e)` for the heads that matter most either way, 10.7 first.

**Check where a head write landed.** The compiled entry's `site` at each point is in the manifest:

<!-- test: skip -->
```python
from clab import Plan
for point in Plan.load("out/head_sets").steps["set9"]:
    print(point.at, point.interventions["patch"].site)
```

Its last line on the head-set document below is `{'num_heads': 2} component='attention_premix' layers=9
select={'num_heads': [6, 9]}`: a point's `at` there is its index on each sweep axis, not the swept value.

## Head sets

A list inside a `select` sweep is one point that patches those heads together. Heads at different layers are
one write per layer, in one forward. This fragment is the `sites` and forwards of `documents/head_sets.json`,
whose `model` and `data` are `documents/head_patching.json`'s. Its op steps take each forward's `logit_diff`
and its mean (`set9_mean`, `across_mean`), as there.

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
$ clab table out/head_sets set9_mean
at                   value
{"num_heads":[6]}    -3.3835153579711914
{"num_heads":[9]}    -1.9188365936279297
{"num_heads":[6,9]}  -2.1071090698242188

$ clab table out/head_sets across_mean
value
1.9949283599853516
```

The four S-inhibition heads together move the mean logit difference from −2.94 to +1.99, flipping the answer.
Their separate effects sum to +3.55 and together they give +4.93: a set's effect is a finding of its own.
Chosen heads patched one at a time are one site per head ([attribution](attribution.md#confirm-the-top-points)).

## Neurons

`mlp_activation` with a `select` on `intermediate_size` patches a block of neurons; a slice string is a block.
This fragment is the `sites` and forwards of `documents/neuron_patching.json`, whose `model` and `data` are
`documents/interchange.json`'s in [SKILL.md](../SKILL.md); its op steps are each forward's `logit_diff` and its
mean (`clean_mean`, `null_mean`, `ld_mean`, `mlp_mean`). It patches layer 9's 3072 neurons at the last token a
quarter at a time, then all of them, and `mlp_output` beside them:

```json
{
  "sites": {
    "neurons": {"component": "mlp_activation", "layers": 9,
                "select": {"intermediate_size": {"sweep": ["0:768", "768:1536", "1536:2304", "2304:3072", ":"]}}},
    "mlp_out": {"component": "mlp_output", "layers": 9},
    "head": {"component": "lm_head_output"}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": {"logits": {"site": "head", "pos": -1}}},
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {
      "v_cf": {"site": "neurons", "pos": -1}, "o_cf": {"site": "mlp_out", "pos": -1}}},
    "null": {"kind": "forward", "input": "pairs.input", "interventions": {
      "v_self": {"site": "neurons", "pos": -1},
      "patch": {"site": "neurons", "pos": -1, "ops": [{"op": "swap", "with": "null.v_self"}]},
      "logits": {"site": "head", "pos": -1}}},
    "patched": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "neurons", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
      "logits": {"site": "head", "pos": -1}}},
    "whole_mlp": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "mlp_out", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.o_cf"}]},
      "logits": {"site": "head", "pos": -1}}}
  }
}
```

```text
$ clab table out/neuron_patching clean_mean mlp_mean
clean_mean           mlp_mean
-1.6586096286773682  -0.7121614813804626

$ clab table out/neuron_patching ld_mean null_mean
at                                 ld_mean              null_mean
{"intermediate_size":"0:768"}      -1.4736175537109375  -1.6586096286773682
{"intermediate_size":"768:1536"}   -1.248659372329712   -1.6586096286773682
{"intermediate_size":"1536:2304"}  -1.5828094482421875  -1.6586096286773682
{"intermediate_size":"2304:3072"}  -1.3924102783203125  -1.6586096286773682
{"intermediate_size":":"}          -0.7121614813804626  -1.6586096286773682
```

- **Every neuron at once equals `mlp_output` to the bit** (−0.7121614813804626): the down projection is
  linear, so the neurons are the MLP's whole contribution.
- **The null equals the clean run** at every block.
- **Layer 9's MLP at the last token carries part of the answer**: +0.95 of the logit difference, spread over
  every quarter (+0.08 to +0.41 each).
- **To rank single neurons**, use [attribution](attribution.md#one-prompt-no-counterfactual), on a pair or one
  prompt. Neurons in several layers are one site per layer, since a `select` applies at every layer it picks.

## Path patching

Patching head h measures everything h changes downstream, including what later heads make of it. Path patching
keeps one path, h's effect on the logits through no other head. Read every head's `attention_premix` at the
last token on the clean prompt (`layers: ":"`). In the patched forward, list a freeze that swaps it back, then
the patch of head h. Entries on the same component and layers compose in the order written, whatever their site
names, so the patch (`one_head`) overrides the freeze (`every_head`) at h alone, and a read listed after both
sees the patched value. The MLPs and the residual stream recompute.

This fragment is the `sites`, `interventions` and `steps` of `documents/path_patching.json`, whose `model` and
`data` are those of `documents/head_patching.json`:

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
      {"patch": {"site": "one_head", "pos": -1, "ops": [{"op": "swap", "with": "own.v"}]}}, "score"]},
    "plain": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"patch": {"site": "one_head", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]}}, "score"]},
    "wrong_order": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"patch": {"site": "one_head", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]}}, "freeze", "score"]}
  }
}
```

Each forward's `ld` is averaged in a `mean` step named for it (`clean_mean`, `frozen_mean`, `path_mean`,
`path_null_mean`, `plain_mean`, `wrong_mean`), and those are the saves. The counterfactual reads
`every_layer`, and each point of the patch takes the slice labelled with its layer.

```text
$ clab table out/path_patching clean_mean frozen_mean
clean_mean           frozen_mean
-2.9367809295654297  -2.9367809295654297

$ clab table out/path_patching path_mean plain_mean path_null_mean wrong_mean
at                            path_mean            plain_mean           path_null_mean       wrong_mean
{"layers":8,"num_heads":6}    -2.948061943054199   -2.1201019287109375  -2.9367809295654297  -2.9367809295654297
{"layers":8,"num_heads":10}   -2.169589042663574   -1.7821941375732422  -2.9367809295654297  -2.9367809295654297
{"layers":9,"num_heads":9}    0.8731622695922852   -1.9188365936279297  -2.9367809295654297  -2.9367809295654297
{"layers":10,"num_heads":7}   -4.948905944824219   -4.498139381408691   -2.9367809295654297  -2.9367809295654297
{"layers":11,"num_heads":10}  -4.131328582763672   -4.131328582763672   -2.9367809295654297  -2.9367809295654297
```

Four nulls, each exact:

- **`frozen_mean` equals `clean_mean`**: the freeze alone writes back what it read.
- **`path_null_mean` equals `clean_mean` at all 48 points**: the patch sourced from the base prompt.
- **At layer 11, `path_mean` equals `plain_mean` for every head**: nothing after the last layer is frozen.
- **`wrong_mean` equals `clean_mean` at all 48 points**: the freeze listed after the patch erased it.

The paths read differently from the plain patches. Head 9.9's direct path is +3.81, against +1.02 for the
plain patch: freezing the later heads removes what they did to counter it. Head 8.6's direct path is about zero
(−2.948 against −2.937), since its plain effect of +0.82 passes through later heads.

### To a later head's query

Read the receiver's query in the path-patched forward, then swap it into a fresh forward of the base prompt, so
only that query moves. `documents/path_to_query.json` is `documents/path_patching.json` with `"attn_implementation":
"eager"` in the model block, `sender` and `receiver` sites in place of `one_head`, and these steps in place of
its steps after `own` but `clean_mean`; its saves are `clean_mean`, `received_mean` and `null_mean`:

```json
{
  "sites": {
    "sender": {"component": "attention_premix", "layers": {"sweep": [7, 8]},
               "select": {"num_heads": {"sweep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}}},
    "receiver": {"component": "attention_queries", "layers": 9, "select": {"num_heads": [9]}}
  },
  "steps": {
    "path": {"kind": "forward", "input": "pairs.input", "interventions": ["freeze",
      {"patch": {"site": "sender", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
       "q": {"site": "receiver", "pos": -1}}]},
    "received": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"q": {"site": "receiver", "pos": -1, "ops": [{"op": "swap", "with": "path.q"}]}}, "score"]},
    "path_null": {"kind": "forward", "input": "pairs.input", "interventions": ["freeze",
      {"patch": {"site": "sender", "pos": -1, "ops": [{"op": "swap", "with": "own.v"}]},
       "q": {"site": "receiver", "pos": -1}}]},
    "received_null": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"q": {"site": "receiver", "pos": -1, "ops": [{"op": "swap", "with": "path_null.q"}]}}, "score"]},
    "received_mean": {"kind": "op", "op": "mean", "of": "received.ld"},
    "null_mean": {"kind": "op", "op": "mean", "of": "received_null.ld"}
  }
}
```

```text
$ clab table out/path_to_query received_mean null_mean
at                           received_mean        null_mean
{"layers":7,"num_heads":3}   -2.865382194519043   -2.936779022216797
{"layers":7,"num_heads":9}   -2.7898406982421875  -2.936779022216797
{"layers":8,"num_heads":6}   -2.6842498779296875  -2.936779022216797
{"layers":8,"num_heads":10}  -2.711958885192871   -2.936779022216797
```

The four largest senders into 9.9's query are 8.6, 8.10, 7.9 and 7.3, the S-inhibition heads; the next is 8.5
at +0.004. `null_mean` equals the eager clean run, −2.936779022216797, at all 24 points.

## Attention-pattern patching

Patch `attention_probabilities` to make a head attend the way it did on the counterfactual while its values
stay the base's. It asks whether where the head looks carries the answer, apart from what it moves. This
fragment is the `sites` and forwards of `documents/pattern_patching.json`. Its model block is
`documents/head_patching.json`'s with `"attn_implementation": "eager"` added; its data and its `logit_diff` and
`mean` steps are that document's.

```json
{
  "sites": {
    "pattern": {"component": "attention_probabilities", "layers": {"sweep": [9, 10]},
                "select": {"num_heads": {"sweep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}}},
    "head": {"component": "lm_head_output"}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": {"logits": {"site": "head", "pos": -1}}},
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {
      "their_pattern": {"site": "pattern", "pos": -1}}},
    "null": {"kind": "forward", "input": "pairs.input", "interventions": {
      "own": {"site": "pattern", "pos": -1},
      "look_here": {"site": "pattern", "pos": -1, "ops": [{"op": "swap", "with": "null.own"}]},
      "logits": {"site": "head", "pos": -1}}},
    "patched": {"kind": "forward", "input": "pairs.input", "interventions": {
      "look_there": {"site": "pattern", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.their_pattern"}]},
      "logits": {"site": "head", "pos": -1}}}
  }
}
```

```text
$ clab table out/pattern_patching ld_mean null_mean
at                            ld_mean              null_mean
{"layers":9,"num_heads":6}    -3.3977251052856445  -2.936779022216797
{"layers":9,"num_heads":9}    -1.8761281967163086  -2.936779022216797
{"layers":10,"num_heads":0}   -2.4961042404174805  -2.936779022216797
{"layers":10,"num_heads":7}   -4.8377685546875     -2.936779022216797
```

- **Head 9.9's pattern alone moves the logit difference by +1.06**, about its whole result's +1.02: where 9.9
  looks carries its effect.
- **The null equals the eager clean run**, −2.936779022216797, at all 24 points.
