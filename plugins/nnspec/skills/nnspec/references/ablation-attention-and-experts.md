# Attention knockout and expert ablation

Two ablations at sites inside a sublayer: where a query looks, and which expert of a mixture a token is sent
to. The attention recipes change `zero_vs_mean.json` of the [ablation guide](ablation.md) and keep its rows and
its `score` bundle.

## What breaks

- **`renormalize` defaults to the L2 norm.** After zeroing keys in a pattern, write `{"op": "renormalize",
  "norm": "sum"}`. Under the default the knocked-out rows below sum to between 0.65 and 1.47, and the score
  comes out 0.17 where it is 0.25.
- **A `key` pick counts from the row's start and is the same on every row.** It names one token only where
  every row has that token at that place, and a pick past the shortest row's last token is refused.
- **Eager and sdpa differ in the last bits**: the clean score here is 2.7463226 under eager and 2.7463291
  under sdpa. Compare runs under one implementation.
- **An expert knockout changes how later blocks route**, and its effect includes that re-routing. Count it.

## Knocking out keys

Use it to stop a query attending to some positions and let the rest of the pattern take up the weight: does
the last token need to read the names? There are two routes, a large negative score before the softmax, or
the pattern zeroed and each row rescaled to sum to 1. Both read inside the attention, so `model` gains
`"attn_implementation": "eager"`. `documents/key_knockout.json` is `zero_vs_mean.json` with that key and these
`sites` and forwards:

```json
{
  "sites": {
    "scores": {"component": "attention_scores", "layers": "9:", "select": {"key": {"sweep": [[1, 3], [6]], "label": "keys"}}},
    "picked": {"component": "attention_probabilities", "layers": "9:", "select": {"key": {"sweep": [[1, 3], [6]], "label": "keys"}}},
    "pattern": {"component": "attention_probabilities", "layers": "9:"},
    "head": {"component": "lm_head_output"}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "ioi.input", "interventions": "score"},
    "by_scores": {"kind": "forward", "input": "ioi.input", "interventions": [{
      "knock": {"site": "scores", "pos": -1, "ops": [{"op": "swap", "with": -1e9}]},
      "p": {"site": "pattern", "pos": -1}}, "score"]},
    "by_pattern": {"kind": "forward", "input": "ioi.input", "interventions": [{
      "knock": {"site": "picked", "pos": -1, "ops": [{"op": "swap", "with": 0.0}]},
      "restore": {"site": "pattern", "pos": -1, "ops": [{"op": "renormalize", "norm": "sum"}]},
      "p": {"site": "pattern", "pos": -1}}, "score"]}
  }
}
```

`clean_ld`, `scores_ld` and `pattern_ld` are `mean` steps over each forward's `ld`. `worst_gap` is the largest
absolute difference of the two `p` reads (`sub`, `abs`, then `max` over every axis), and `total_min` and
`total_max` the least and greatest sum of `by_pattern.p` over `key`.

```text
$ nnspec table out/key_knockout clean_ld scores_ld pattern_ld worst_gap
at              clean_ld            scores_ld           pattern_ld          worst_gap
{"keys":[1,3]}  2.7463226318359375  0.2475382536649704  0.2475382536649704  4.172325134277344e-07
{"keys":[6]}    2.7463226318359375  2.7452316284179688  2.745234251022339   5.364418029785156e-07
$ nnspec table out/key_knockout total_min total_max
at              total_min           total_max
{"keys":[1,3]}  0.9999997019767761  1.000000238418579
{"keys":[6]}    0.9999997615814209  1.000000238418579
```

- **Keys 1 and 3 are the two names on every row** (GPT-2 adds no BOS, so key 0 is the first word). Blocking
  them at every head of layers 9 to 11 costs nearly everything: 2.75 falls to 0.25.
- **The control is key 6**, ` the` on every row, which the task does not read: it moves the score by -0.001.
- **The two routes give one pattern**, to 5e-7, and `restore` leaves every row a distribution again. It is a
  later entry on the whole row, since `renormalize` rescales to what the site held before the step's writes.

**One head.** A write selects on one axis, so `"select": {"key": [1, 3]}` knocks the keys out for every head,
and a write that selects `key` and `num_heads` together is refused. Knock the keys out, then restore the other
heads from a clean read of their scores, in a second write at the same component and layer. A fragment of
`documents/one_head_keys.json`, for head 10.7:

```json
{
  "sites": {
    "names": {"component": "attention_scores", "layers": 10, "select": {"key": [1, 3]}},
    "others": {"component": "attention_scores", "layers": 10, "select": {"num_heads": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]}}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "ioi.input", "interventions": ["score", {"others": {"site": "others", "pos": -1}}]},
    "one_head": {"kind": "forward", "input": "ioi.input", "interventions": [{
      "knock": {"site": "names", "pos": -1, "ops": [{"op": "swap", "with": -1e9}]},
      "keep_others": {"site": "others", "pos": -1, "ops": [{"op": "swap", "with": "clean.others"}]}}, "score"]}
  }
}
```

The score is 4.157 with head 10.7 alone blind to the names and 3.597 with every head of layer 10 blind to
them. The null restores all twelve heads the same way and gives the clean 2.7463226318359375 to the bit.

**A head's whole pattern.** `{"op": "swap", "with": 0.0}` at `attention_probabilities` with `"select":
{"num_heads": [7]}` removes what the head moves to that query. In the same document it gives
3.5669009685516357, and so does the same write at `attention_premix`, row for row to the bit; the premix
needs no eager attention.

Reading patterns is the [attention guide](attention.md); swapping in another prompt's pattern is
[heads, neurons and paths](patching-heads-neurons-and-paths.md).

## Expert knockout

Use it to ask what a mixture of experts loses without one expert. The clean forward reads a mask, 0 where a
routing slot chose expert e and 1 elsewhere, and the knockout scales every slot's weight by it: e's slots add
nothing, and nothing renormalizes the others. The checkpoint is a tiny Mixtral with random weights, so the
numbers show the mechanics and nothing about a trained model. Saved as `documents/expert_knockout.json`:

```json
{
  "description": "Expert knockout and the forced route at block 0 of the tiny random Mixtral (2 blocks, 4 experts, 2 slots per token), with each expert's usage, the routing entropy and block 1's re-routing. Each write scores the KL of the last token's distribution from the clean run's. Random weights: the numbers show the mechanics only.",
  "model": {"key": "hf-internal-testing/tiny-random-MixtralForCausalLM", "revision": "ccb12fe2fc142cb752085506c3db22572290e90c", "dtype": "float32"},
  "data": {"prompts": {"rows": [
    {"input": "When Mary and John went to the store, John gave a drink to"},
    {"input": "When Tom and James went to the park, Tom gave a ball to"},
    {"input": "When Alice and Bob went to the office, Bob gave a book to"},
    {"input": "When Anna and Mark went to the beach, Mark gave a gift to"}
  ]}},
  "sites": {
    "slots": {"component": "expert_indices", "layers": 0},
    "weights": {"component": "expert_weights", "layers": 0},
    "router": {"component": "router_logits", "layers": 0},
    "one_expert": {"component": "router_logits", "layers": 0, "select": {"num_experts": {"sweep": [0, 1, 2, 3], "label": "expert"}}},
    "later": {"component": "expert_indices", "layers": 1},
    "head": {"component": "lm_head_output"}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "prompts.input", "interventions": {
      "mask": {"site": "slots", "pos": {"index": ":15"}, "ops": [{"op": "ne", "with": {"sweep": [0, 1, 2, 3], "label": "expert"}}]},
      "none": {"site": "slots", "pos": {"index": ":15"}, "ops": [{"op": "ne", "with": 4}]},
      "chosen": {"site": "slots", "pos": {"index": ":"},
                 "ops": [{"op": "eq", "with": {"sweep": [0, 1, 2, 3], "label": "expert"}}, {"op": "sum", "over": ["pos", "top_k"]}]},
      "routing": {"site": "router", "pos": {"index": ":"}, "ops": [{"op": "entropy"}]},
      "later": {"site": "later", "pos": {"index": ":15"}},
      "logits": {"site": "head", "pos": -1}}},
    "knockout": {"kind": "forward", "input": "prompts.input", "interventions": {
      "kill": {"site": "weights", "pos": {"index": ":15"}, "ops": [{"op": "scale", "factor": "clean.mask"}]},
      "later": {"site": "later", "pos": {"index": ":15"}, "ops": [{"op": "ne", "with": "clean.later"}]},
      "kl": {"site": "head", "pos": -1, "ops": [{"op": "kl", "against": "clean.logits"}]}}},
    "null": {"kind": "forward", "input": "prompts.input", "interventions": {
      "kill": {"site": "weights", "pos": {"index": ":15"}, "ops": [{"op": "scale", "factor": "clean.none"}]},
      "kl": {"site": "head", "pos": -1, "ops": [{"op": "kl", "against": "clean.logits"}]}}},
    "forced": {"kind": "forward", "input": "prompts.input", "interventions": {
      "away": {"site": "one_expert", "pos": {"index": ":"}, "ops": [{"op": "swap", "with": -1e9}]},
      "kl": {"site": "head", "pos": -1, "ops": [{"op": "kl", "against": "clean.logits"}]}}},
    "usage": {"kind": "op", "op": "sum", "of": "clean.chosen", "over": ["rows"]},
    "rerouted": {"kind": "op", "op": "sum", "of": "knockout.later", "over": ["rows", "pos", "top_k"]},
    "knockout_kl": {"kind": "op", "op": "mean", "of": "knockout.kl"},
    "forced_kl": {"kind": "op", "op": "mean", "of": "forced.kl"},
    "null_kl": {"kind": "op", "op": "max", "of": "null.kl"},
    "entropy": {"kind": "op", "op": "mean", "of": "clean.routing"}
  },
  "saves": ["usage", "rerouted", "knockout_kl", "forced_kl", "null_kl", "entropy"]
}
```

```text
$ nnspec table out/expert_knockout usage rerouted knockout_kl forced_kl
at            usage  rerouted  knockout_kl             forced_kl
{"expert":0}  28.0   1.0       3.5498294892022386e-05  6.340032268781215e-05
{"expert":1}  22.0   0.0       2.191085059166653e-06   2.7261874038231326e-06
{"expert":2}  43.0   3.0       2.9413055017357692e-05  5.630953091895208e-05
{"expert":3}  27.0   3.0       1.7466069266447448e-06  3.5173716241843067e-06
$ nnspec table out/expert_knockout null_kl entropy
null_kl  entropy
0.0      1.3772038221359253
```

- **The null is exact**: the mask for id 4, which no slot holds, is all ones, and `null_kl` is 0.0. A null
  that is not exact means the mask and the write are not at the same slots.
- **Write the weights, never the ids.** `expert_indices` takes only a `swap` of an id below `num_experts`.
- **The mask comes from the clean forward.** Nothing before block 0 changed, so the routing the knockout
  meets is the clean run's. A mask computed in the knockout's own forward is refused: an operand from the
  same forward carries no op after its edits.
- **Read the mask at a width every prompt holds**, `{"index": ":15"}` on these 15-token prompts (the
  tokenizer adds a BOS). Tokens past it keep the expert, so give the prompts one length.
- **The forced route** writes -1e9 into e's router logit, so every token that chose e goes to its next
  choice. It needs no mask, and measures a substitute where the knockout measures a gap: report both.
- **Every expert is a point**: one `expert` label on the mask, the forced route's `select` and the usage
  count pairs them.
- **`rerouted` counts block 1's slots whose expert changed** under the knockout: 0 to 3 of 120 here.
- **`usage` sums each expert's slots over the rows**, 120 together. `entropy` of `router_logits` is each
  token's routing entropy in nats, ln 4 = 1.386 where the router is uniform, and the routing distribution's
  only on a softmax router.

### Which value to read or write

`nnspec describe components` prints each layout. On a block whose MLP is a mixture:

- **`router_logits`**: one logit per expert before the router scores them. Write it to force a route, and take
  its `entropy`.
- **`expert_indices`**: the expert each routing slot sends the token to. Read it for masks and usage.
- **`expert_weights`**: what each slot's output is scaled by. Write it to knock an expert out, and read it
  instead of recomputing weights from the logits: whether a family renormalizes its top k is the family's.
- **`expert_outputs`**: each slot's expert output times its weight, for what each chosen expert adds to a
  token.
- **`routed_output`**: their sum over the slots, without the shared expert, for what the routed experts add
  together. On this Mixtral it equals that sum, and `mlp_output`, to the bit.
- **`shared_expert_output`**: the shared expert's own output, to split the MLP's write between the two. On a
  mixture with none, as here, it is refused by name.
- **`mlp_output`** is still what the block adds. `mlp_activation` and `mlp_neurons` are refused at a mixture
  block, naming what is served there.

Run expert ablations on the nnterp engine. On this Mixtral the hooks engine serves `mlp_output` and
`routed_output`, whose reads and writes equal nnterp's to the bit, and refuses the routing and slot values
([engines](running-and-engines.md#engines)).
