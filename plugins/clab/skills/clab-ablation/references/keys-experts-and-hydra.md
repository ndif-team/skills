# Keys, patterns, the hydra effect and experts

Attention knockout and the hydra effect run on `openai-community/gpt2` and the six IOI prompts of the skill.
Expert ablation runs on `hf-internal-testing/tiny-random-MixtralForCausalLM`, whose weights are random: its
documents show the mechanics, and their numbers mean nothing about any trained model. Each runs in seconds
on a CPU.

## Knocking out keys

Use it to stop a query attending to some positions and let the rest of the pattern take up the weight
(attention knockout): does the last token need to read the names at layers 9 to 11? Write through the
scores, a large negative score before the softmax, or through the pattern, zeroed and then rescaled. Both
need `"attn_implementation": "eager"`.

```json
{
  "description": "Attention knockout on GPT-2's IOI prompts under eager attention. At every head of layers 9 to 11 the last token stops attending to keys 1 and 3, the two names on every row. by_scores writes -1e9 before the softmax; by_pattern zeroes the pattern there and rescales each row to sum to 1. Key 6 (' the' on every row) is the control the task does not read. p reads the pattern after the writes. head_pattern zeroes head 10.7's pattern at the last token and head_premix its premix there.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32",
            "attn_implementation": "eager"},
  "data": {"ioi": {"rows": [
    {"input": "When Mary and John went to the store, John gave a drink to", "answer": " Mary", "other": " John"},
    {"input": "When Tom and James went to the train station, Tom gave a ball to", "answer": " James", "other": " Tom"},
    {"input": "When Alice and Bob went to the office, Bob gave a book to", "answer": " Alice", "other": " Bob"},
    {"input": "When Sarah and David went to the swimming pool, Sarah gave a ring to", "answer": " David", "other": " Sarah"},
    {"input": "When Anna and Mark went to the beach, Mark gave a gift to", "answer": " Anna", "other": " Mark"},
    {"input": "When Lisa and Paul went to the movie theater, Lisa gave a letter to", "answer": " Paul", "other": " Lisa"}
  ]}},
  "sites": {
    "scores": {"component": "attention_scores", "layers": "9:", "select": {"key": {"sweep": [[1, 3], [6]], "label": "keys"}}},
    "picked": {"component": "attention_probabilities", "layers": "9:", "select": {"key": {"sweep": [[1, 3], [6]], "label": "keys"}}},
    "pattern": {"component": "attention_probabilities", "layers": "9:"},
    "pattern_10_7": {"component": "attention_probabilities", "layers": 10, "select": {"num_heads": [7]}},
    "premix_10_7": {"component": "attention_premix", "layers": 10, "select": {"num_heads": [7]}},
    "head": {"component": "lm_head_output"}
  },
  "interventions": {"score": {"ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "ioi.answer", "b": "ioi.other"}]}}},
  "steps": {
    "clean": {"kind": "forward", "input": "ioi.input", "interventions": ["score"]},
    "by_scores": {"kind": "forward", "input": "ioi.input", "interventions": [{
      "knock": {"site": "scores", "pos": -1, "ops": [{"op": "swap", "with": -1e9}]},
      "p": {"site": "pattern", "pos": -1}}, "score"]},
    "by_pattern": {"kind": "forward", "input": "ioi.input", "interventions": [{
      "knock": {"site": "picked", "pos": -1, "ops": [{"op": "swap", "with": 0.0}]},
      "restore": {"site": "pattern", "pos": -1, "ops": [{"op": "renormalize", "norm": "sum"}]},
      "p": {"site": "pattern", "pos": -1}}, "score"]},
    "head_pattern": {"kind": "forward", "input": "ioi.input", "interventions": [{
      "zero": {"site": "pattern_10_7", "pos": -1, "ops": [{"op": "swap", "with": 0.0}]}}, "score"]},
    "head_premix": {"kind": "forward", "input": "ioi.input", "interventions": [{
      "zero": {"site": "premix_10_7", "pos": -1, "ops": [{"op": "swap", "with": 0.0}]}}, "score"]},
    "gap": {"kind": "op", "op": "sub", "of": "by_scores.p", "with": "by_pattern.p"},
    "abs_gap": {"kind": "op", "op": "abs", "of": "gap"},
    "worst_gap": {"kind": "op", "op": "max", "of": "abs_gap", "over": ["layers", "rows", "pos", "num_heads", "key"]},
    "clean_ld": {"kind": "op", "op": "mean", "of": "clean.ld"},
    "scores_ld": {"kind": "op", "op": "mean", "of": "by_scores.ld"},
    "pattern_ld": {"kind": "op", "op": "mean", "of": "by_pattern.ld"}
  },
  "saves": ["clean_ld", "scores_ld", "pattern_ld", "worst_gap", "head_pattern.ld", "head_premix.ld"]
}
```

```text
$ clab run documents/key_knockout.json
$ clab table out/key_knockout clean_ld scores_ld pattern_ld worst_gap
at              clean_ld            scores_ld           pattern_ld          worst_gap
{"keys":[1,3]}  2.7463226318359375  0.2475382536649704  0.2475382536649704  4.172325134277344e-07
{"keys":[6]}    2.7463226318359375  2.7452316284179688  2.745234251022339   5.364418029785156e-07
$ clab table out/key_knockout head_pattern.ld head_premix.ld
row  pos  pos_token  input                                             answer  other   head_pattern.ld     head_premix.ld
0    13   Ġto        When Mary and John went to the store, John gave…   Mary    John   2.6998443603515625  2.6998443603515625
1    14   Ġto        When Tom and James went to the train station, T…   James   Tom    3.6678543090820312  3.6678543090820312
...
```

- **Blocking the names costs nearly everything**: 2.75 falls to 0.25. The null, key 6 (" the"), moves it
  by -0.001.
- **The two routes give one pattern**, to 5e-7, and logit differences within 3e-6 of each other.
- **`renormalize` must say `"norm": "sum"`**, and run in a later entry on the whole row. Its default `l2`
  rescales to the L2 norm; with it, this document's knocked-out rows summed to as much as 1.47.
- **A `key` selector counts from the start**, the same keys on every row. It names one token only where the
  rows share it at that place, as the names at 1 and 3 and " the" at 6 are here; a write past a row's own
  keys changes nothing. GPT-2 adds no BOS, so key 0 is each prompt's first word.
- **A write selects on one axis**, so `select: {"key": ...}` knocks the keys out for every head. For one head,
  restore the others with a second write at the same site from a clean read of their scores.
- **Zeroing a head's whole pattern removes what it moves there**: `head_pattern` and `head_premix` agree to
  the bit on every row. Under eager the numbers differ from an sdpa run in the last bits (clean 2.7463226
  here, 2.7463291 under sdpa); compare runs under one implementation.
- **Under sdpa these components are refused** before any forward ([eager](../../clab/references/sites-and-interventions.md#attention-needs-eager)).

Reading a pattern, stripes and induction heads are the `clab-lenses-and-attention` skill; swapping in another
prompt's pattern is the `clab-patching` skill. Components and their axes: `clab describe components`.

## The hydra effect

When a component is ablated, later ones can change what they write to make up for it, so the drop in the
total understates what it did. Measure each later layer's direct effect, clean and ablated, in the same run:
the logit difference less the logit difference with that layer's attention output subtracted from the final
residual stream.

```json
{
  "description": "The hydra effect on GPT-2's IOI prompts: mean-ablate layer 9's attention at the last token, and measure the direct effect of the attention of layers 9, 10 and 11, clean and ablated. A layer's direct effect is the logit difference less the logit difference with that layer's attention output subtracted from the final residual stream, in the same run. fill's second point writes back the clean read: the null.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"ioi": {"rows": [
    {"input": "When Mary and John went to the store, John gave a drink to", "answer": " Mary", "other": " John"},
    {"input": "When Tom and James went to the train station, Tom gave a ball to", "answer": " James", "other": " Tom"},
    {"input": "When Alice and Bob went to the office, Bob gave a book to", "answer": " Alice", "other": " Bob"},
    {"input": "When Sarah and David went to the swimming pool, Sarah gave a ring to", "answer": " David", "other": " Sarah"},
    {"input": "When Anna and Mark went to the beach, Mark gave a gift to", "answer": " Anna", "other": " Mark"},
    {"input": "When Lisa and Paul went to the movie theater, Lisa gave a letter to", "answer": " Paul", "other": " Lisa"}
  ]}},
  "sites": {
    "attn9": {"component": "attention_output", "layers": 9},
    "probe": {"component": "attention_output", "layers": {"sweep": [9, 10, 11], "label": "layer"}},
    "final": {"component": "layer_output", "layers": -1},
    "head": {"component": "lm_head_output"}
  },
  "interventions": {"score": {"ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "ioi.answer", "b": "ioi.other"}]}}},
  "steps": {
    "clean": {"kind": "forward", "input": "ioi.input", "interventions": ["score", {
      "a9": {"site": "attn9", "pos": -1}, "a": {"site": "probe", "pos": -1}}]},
    "mean9": {"kind": "op", "op": "mean", "of": "clean.a9"},
    "ablated": {"kind": "forward", "input": "ioi.input", "interventions": [{
      "abl": {"site": "attn9", "pos": -1, "ops": [{"op": "swap", "with": {"sweep": ["mean9", "clean.a9"], "label": "fill"}}]},
      "a": {"site": "probe", "pos": -1}}, "score"]},
    "clean_off": {"kind": "forward", "input": "ioi.input", "interventions": [{
      "off": {"site": "final", "pos": -1, "ops": [{"op": "sub", "with": "clean.a"}]}}, "score"]},
    "ablated_off": {"kind": "forward", "input": "ioi.input", "interventions": [{
      "abl": {"site": "attn9", "pos": -1, "ops": [{"op": "swap", "with": {"sweep": ["mean9", "clean.a9"], "label": "fill"}}]},
      "off": {"site": "final", "pos": -1, "ops": [{"op": "sub", "with": "ablated.a"}]}}, "score"]},
    "de_clean": {"kind": "op", "op": "sub", "of": "clean.ld", "with": "clean_off.ld"},
    "de_ablated": {"kind": "op", "op": "sub", "of": "ablated.ld", "with": "ablated_off.ld"},
    "clean_ld": {"kind": "op", "op": "mean", "of": "clean.ld"},
    "ablated_ld": {"kind": "op", "op": "mean", "of": "ablated.ld"},
    "de_c": {"kind": "op", "op": "mean", "of": "de_clean"},
    "de_a": {"kind": "op", "op": "mean", "of": "de_ablated"}
  },
  "saves": ["clean_ld", "ablated_ld", "de_c", "de_a"]
}
```

```text
$ clab run documents/hydra.json
$ clab table out/hydra clean_ld ablated_ld de_c de_a
at                              clean_ld            ablated_ld          de_c                  de_a
{"fill":"mean9","layer":9}      2.7463290691375732  2.6409759521484375  2.6578826904296875    0.48196282982826233
{"fill":"mean9","layer":10}     2.7463290691375732  2.6409759521484375  -0.24392573535442352  1.8836288452148438
{"fill":"mean9","layer":11}     2.7463290691375732  2.6409759521484375  -3.237017869949341    -3.1121113300323486
{"fill":"clean.a9","layer":9}   2.7463290691375732  2.7463290691375732  2.6578826904296875    2.6578826904296875
{"fill":"clean.a9","layer":10}  2.7463290691375732  2.7463290691375732  -0.24392573535442352  -0.24392573535442352
{"fill":"clean.a9","layer":11}  2.7463290691375732  2.7463290691375732  -3.237017869949341    -3.237017869949341
```

- **Read it as a balance.** The total effect is 2.75 - 2.64 = 0.11. Layer 9's attention lost 2.18 of its
  direct effect (2.66 to 0.48), and layer 10's attention gained 2.13 (-0.24 to 1.88): it compensates.
- **The null** (`fill` = `clean.a9`) gives every direct effect its clean value and `ablated_ld` the clean
  run's, to the bit.
- **`ablated.a` is read under the ablation**, and a later forward subtracts it: a read inside an intervened
  forward is an operand like any other. At layer 9 it reads the written mean, listed after the write there.
- **The direct effect goes through the final norm**, so the layers' direct effects do not sum to the total.
- **Sweeps that share a label pair their points**, so `fill` in `ablated` and `ablated_off` is one axis, and
  each point's `ablated.a` meets that point's `ablated_off`. `probe`'s `layer` crosses it.
- **A head's own direct effect** needs its contribution to the stream, which no component holds; this
  document works per layer.
- **With a resample instead of a mean**, give each row its donor prompt in a column and read it by a
  forward of its own, since `permute` pairs any two rows.

## Expert knockout

Use it to ask what a mixture loses without one expert. The clean forward reads a mask, 0 where a routing
slot chose expert e and 1 elsewhere. The knockout forward scales every slot's weight by it, so e's slots
add nothing, and nothing renormalizes the others.

```json
{
  "description": "Expert knockout, the forced route, routing usage and entropy on the tiny random Mixtral (2 blocks, 4 experts, 2 per token). At block 0, clean reads at the 15 tokens every prompt has a mask of the slots that did not choose expert e (ne, swept over the experts). It also reads the same mask for id 4, which no slot holds (the null), how many slots chose e per row, each token's routing entropy, and block 1's routing. knockout scales every slot's weight by its mask; null scales by the all-ones mask; forced writes -1e9 into expert e's router logit. Each scores the KL of the last token's next-token distribution from the clean run's. Random weights: the numbers show the mechanics only.",
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
    "one_expert": {"component": "router_logits", "layers": 0,
                   "select": {"num_experts": {"sweep": [0, 1, 2, 3], "label": "expert"}}},
    "later": {"component": "expert_indices", "layers": 1},
    "head": {"component": "lm_head_output"}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "prompts.input", "interventions": {
      "mask": {"site": "slots", "pos": {"index": ":15"},
               "ops": [{"op": "ne", "with": {"sweep": [0, 1, 2, 3], "label": "expert"}}]},
      "none": {"site": "slots", "pos": {"index": ":15"}, "ops": [{"op": "ne", "with": 4}]},
      "chosen": {"site": "slots", "pos": {"index": ":"},
                 "ops": [{"op": "eq", "with": {"sweep": [0, 1, 2, 3], "label": "expert"}},
                         {"op": "sum", "over": ["pos", "top_k"]}]},
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
  "saves": ["usage", "rerouted", "knockout_kl", "forced_kl", "null_kl", "entropy", "clean.routing"]
}
```

```text
$ clab run documents/expert_knockout.json
$ clab table out/expert_knockout usage rerouted knockout_kl forced_kl
at            usage  rerouted  knockout_kl             forced_kl
{"expert":0}  28.0   1.0       3.5498294892022386e-05  6.340032268781215e-05
{"expert":1}  22.0   0.0       2.191085059166653e-06   2.7261874038231326e-06
{"expert":2}  43.0   3.0       2.9413055017357692e-05  5.630953091895208e-05
{"expert":3}  27.0   3.0       1.7466069266447448e-06  3.5173716241843067e-06
$ clab table out/expert_knockout null_kl entropy
null_kl  entropy
0.0      1.3772038221359253
```

- **The null is exact**: the mask for id 4, which no slot holds, is all ones, and `null_kl` is 0.0. A null
  that is not exact means the mask and the write are not at the same slots.
- **Write the weights, never the ids.** `expert_indices` takes only a `swap` of an id below `num_experts`.
  A `swap` with 4 there is refused before any forward: `expert_indices holds ids, and an id is a whole
  number from 0 to 3`. An `add` is refused before the model loads: `'add' would compute on expert_indices,
  the ids the model indexes its num_experts with, as numbers`.
- **The mask comes from the clean forward.** Nothing before block 0 changed, so the routing the knockout
  meets is the clean run's. A mask read in the knockout's own forward is refused (`'knockout.idx' runs 'ne'
  once it is off the model, after this forward`).
- **Read the mask at a static width**, the tokens every prompt has: `{"index": ":15"}` on these 15-token
  prompts. An open `{"index": ":"}` is refused as another forward's operand (`'clean.mask' has no width
  compile can fix`). Tokens past the width keep the expert, so give the prompts one length.
- **The forced route** writes -1e9 into e's router logit before the router scores it, so every token that
  chose e goes to its next choice. It needs no mask, so no operand to pair and no width to keep. It
  measures a substitute where the knockout measures a gap; report both.
- **Every expert is a point.** One `expert` label on the mask, the forced route's `select` and the usage
  count pairs them: the clean forward reads every mask once, and each knockout point runs its own forward.
  A grid of blocks by experts crosses a `layers` sweep under another label.
- **Later blocks re-route.** `rerouted` counts block 1's slots whose expert changed under the knockout: 0 to
  3 of 120 here. The effect includes that re-routing.
- **Usage and entropy.** `usage` sums each expert's slots over the rows: 120 together, every slot of every
  token. `entropy` of `router_logits` is each token's routing entropy in nats, ln 4 = 1.386 where the router
  is uniform (1.377 on average here). It is the routing distribution's only on a softmax router.
- **Read `expert_weights` for what multiplies each slot**, rather than recomputing it from the logits:
  whether a family renormalizes its top k is the family's.
- **Read what the experts write** where the question is what an expert adds, not whether it was chosen.
  `expert_outputs` is each slot's expert output times its weight, `[rows, pos, top_k, hidden_size]`, for
  what each chosen expert adds to a token. `routed_output` is their sum over `top_k`, without the shared
  expert, for what the routed experts add together. `shared_expert_output` is the shared expert's own, to
  split the MLP's write between the two; on a mixture with none, as here, it is refused by name.

Run expert ablations on the nnterp engine, the default; which values each engine serves is the `clab` skill's
[running and engines](../../clab/references/running-and-engines.md) reference. Components: `clab describe
components`.
