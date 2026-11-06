---
name: nnspec-lenses-and-attention
description: "Find the layer where a model's prediction forms, and where its attention heads look, with clab documents. The logit lens: readout at every layer in one forward, its wiring check, top-k tokens and KL per layer. The tuned lens: one `affine` translator per layer, trained in the run. What the model says at a position, and a different correct next token at each position. Attention patterns under eager attention: previous-token and induction heads, sinks, head entropy, the weight on a subject, queries after RoPE. Knocking out a head or a key is clab-ablation. Patching a pattern, or testing which layer causally carries an answer, is clab-patching. Probes, SAEs, DAS and reusing a trained lens elsewhere are clab-trained-featurizers. Generating, scoring what a generation said, and steering are clab-steering-and-generation. A refusal or wrong numbers is clab-debugging; the format is the clab skill."
---

# Lenses and attention patterns

Verified on clab `dca9630` with nnterp `e5d0f8e`, nnsight 0.8.2 and transformers 5.20.2; the documents run on the
CPU in seconds on `openai-community/gpt2` (12 layers of 12 heads, no BOS), one on
`HuggingFaceTB/SmolLM2-135M-Instruct`.

Everything here reads; nothing writes into the forward. The format (sites, `pos`, ops, sweeps, running,
`clab table`) is the `clab` skill's ([SKILL.md](../clab/SKILL.md)); this skill gives the recipes.

## What breaks

- **A lens read without its wiring check.** Every layer looks plausible whether or not `readout` is the model's
  own computation. Write the check as op steps: `kl` of a `readout` step against `logits` read at the same `pos` is
  0.0 at the last layer, or the lens is wrong. In an entry `readout` runs per batch and can differ from the step in
  the last bits ([the `clab-debugging` skill](../clab-debugging/SKILL.md#reproducing-a-number)).
- **Reading only the output's top token.** On "The Colosseum is in the city of", ` Rome` is 0.84 at layer 9 and
  0.014 at the output, whose top token is ` P`. Score the answer at every layer.
- **An early layer's lens is not what the layer knows.** The final norm and head were trained on the last layer's
  stream. Compare layers with each other and with `to_final`, or train a [tuned lens](references/tuned-lens.md).
- **An answer without its leading space is another token** (the `clab` skill's list); `Ġ` in a decoded token is
  GPT-2's leading space.
- **An attention component needs eager attention**: add `"attn_implementation": "eager"` to the `model` block, or it
  is refused ([which components](../clab/references/sites-and-interventions.md#attention-needs-eager)).
- **A key counted from the start for a score relative to the query.** `select: {"key": 1}` is the same token at
  every query. A previous-token or induction score is `{"key": {"offset": -1}}`, the key at a fixed distance
  from each query. A key counted from the end (`-1`) is refused, and so is an offset in a write.
- **A `top_k` or `argmax` id along a narrowed axis counts within it** (the `clab` skill's list): under
  `"layers": "4:"`, add 4.

## The logit lens

Use it to see at which layer a prediction forms. `readout` projects a residual-stream value through the model's
own final norm and head, as if that layer were the last; `"layers": ":"` reads every layer in one forward.

```json
{
  "description": "The logit lens on GPT-2 small: every layer's residual stream at the last prompt token, read in one forward and projected through the model's own final norm and head by a `readout` step. `final` is the model's own logits at the same token, so `to_final`, KL(lens || final) at each layer, must be 0.0 at layer 11: the wiring check. `p_answer` is the answer's probability at each layer; `top3` each layer's three most likely tokens, decoded.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {
    "facts": {"rows": [
      {"example_id": "colosseum", "input": "The Colosseum is in the city of", "answer": " Rome"},
      {"example_id": "eiffel", "input": "The Eiffel Tower is in the city of", "answer": " Paris"}
    ]}
  },
  "sites": {
    "resid": {"component": "layer_output", "layers": ":"},
    "head": {"component": "logits"}
  },
  "steps": {
    "run": {
      "kind": "forward",
      "input": "facts.input",
      "interventions": {
        "h": {"site": "resid", "pos": -1},
        "final": {"site": "head", "pos": -1}
      }
    },
    "lens": {"kind": "op", "op": "readout", "of": "run.h"},
    "p_answer": {"kind": "op", "op": "token_prob", "of": "lens", "token": "facts.answer"},
    "to_final": {"kind": "op", "op": "kl", "of": "lens", "against": "run.final"},
    "top3": {"kind": "op", "op": "top_k", "of": "lens", "k": 3}
  },
  "saves": ["p_answer", "to_final", "top3.tokens", "top3.probs"]
}
```

- `run.h` is laid out `layers, rows, pos, hidden_size`; `lens` replaces `hidden_size` with `vocab_size`.
- `readout` takes a component `clab describe components` marks `stream`, such as `layer_output` or `mlp_output`.
  Another component, or a site that selects part of `hidden_size`, is refused before the model loads.
- `to_final` is KL(lens ‖ final) in nats. `run.final` has no `layers` axis, so every layer compares with the one
  final distribution.
- `top3` is a `top_k` step, which decodes its tokens. Its members `top3.ids`, `top3.probs` and `top3.tokens` are
  named, saved and tabled on their own. A `top_k` inside an entry holds no tokens, and no step can name its members.

```bash
clab run documents/logit_lens.json
clab table out/logit_lens p_answer to_final top3.tokens
```

```text
example_id  layer  row  pos  pos_token  input                               answer  p_answer                to_final            top3.tokens
colosseum   0      0    9    Ġof        The Colosseum is in the city of      Rome   1.2839038845413597e-06  4.745212554931641   ["Ġthe","Ġcourse","Ġa"]
eiffel      0      1    9    Ġof        The Eiffel Tower is in the city of   Paris  2.1919738628639607e-06  4.684253692626953   ["Ġthe","Ġcourse","Ġa"]
colosseum   7      0    9    Ġof        The Colosseum is in the city of      Rome   0.09875402599573135     2.0175745487213135  ["ĠRome","ĠIng","ĠLa"]
eiffel      7      1    9    Ġof        The Eiffel Tower is in the city of   Paris  0.0028008739463984966   2.274801015853882   ["ĠIng","ĠRome","ĠLa"]
colosseum   8      0    9    Ġof        The Colosseum is in the city of      Rome   0.6592757701873779      2.6900246143341064  ["ĠRome","ĠLa","ĠPal"]
eiffel      8      1    9    Ġof        The Eiffel Tower is in the city of   Paris  0.025490013882517815    1.6806339025497437  ["ĠRome","ĠLondon","ĠChicago"]
colosseum   9      0    9    Ġof        The Colosseum is in the city of      Rome   0.8437438607215881      3.512146472930908   ["ĠRome","ĠNaples","ĠAthens"]
eiffel      9      1    9    Ġof        The Eiffel Tower is in the city of   Paris  0.24841895699501038     0.9615956544876099  ["ĠLondon","ĠParis","ĠAmsterdam"]
colosseum   10     0    9    Ġof        The Colosseum is in the city of      Rome   0.38819628953933716     1.644049048423767   ["ĠRome","ĠAthens","ĠMarse"]
eiffel      10     1    9    Ġof        The Eiffel Tower is in the city of   Paris  0.18341495096683502     0.5615532994270325  ["ĠParis","ĠLondon","ĠAmsterdam"]
colosseum   11     0    9    Ġof        The Colosseum is in the city of      Rome   0.014371918514370918    0.0                 ["ĠP","ĠT","ĠC"]
eiffel      11     1    9    Ġof        The Eiffel Tower is in the city of   Paris  0.06997019797563553     0.0                 ["ĠParis","ĠLondon","ĠNew"]
```

- **The null is the wiring check:** `to_final` is 0.0 at layer 11 on both rows. A value above float32 rounding
  there means the lens is not the model's own last computation; read nothing else until it is.
- **Where the answer forms:** ` Rome` leads from layer 7 and peaks at 0.84 at layer 9. The last layer flattens
  it: the output's top tokens are single letters at 0.019 or less. ` Paris` overtakes ` London` at layer 10.

Variations, each a change to the document above:

- **Keep only the score.** An entry `{"site": "resid", "pos": -1, "ops": [{"op": "readout"}, {"op": "token_prob",
  "token": "facts.answer"}]}` holds one number per layer, row and position, never the vocabulary: for large runs,
  not the wiring check.
- **The last few tokens.** `"pos": {"index": "-3:"}` on both `h` and `final` lenses each row's last three tokens;
  `kl` pairs position with position, and `to_final` is 0.0 at layer 11 at each.
- **A sublayer's lens.** `mlp_output` or `attention_output` in place of `layer_output` lenses what one sublayer
  writes. On the Colosseum prompt MLP 11's lens ranks ` T`, ` P`, ` B` first; the output's top two are ` P`, ` T`.
- **Two lenses compared.** `kl` of a lens read at `"layers": [9, 3]` against one at `":"` pairs layer 9 with 9 and
  3 with 3 (0.0 within one forward). An `against` lens that lacks one of `of`'s layers is refused by name.

## What the model says at a position

The readout of the model itself: `top_k` of `logits` as a step, and `token_prob` of the expected answer.

```json
{
  "description": "What GPT-2 small says at the last prompt token, read out rather than scored: the five most likely next tokens, decoded, with their probabilities, and beside them the probability of the expected answer.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {
    "prompts": {"rows": [
      {"example_id": "monday", "input": "Monday, Tuesday, Wednesday,", "answer": " Thursday"},
      {"example_id": "counting", "input": "one, two, three, four,", "answer": " five"},
      {"example_id": "eiffel", "input": "The Eiffel Tower is in the city of", "answer": " Paris"}
    ]}
  },
  "steps": {
    "run": {
      "kind": "forward",
      "input": "prompts.input",
      "interventions": {"logits": {"site": {"component": "logits"}, "pos": -1}}
    },
    "top5": {"kind": "op", "op": "top_k", "of": "run.logits", "k": 5},
    "p_answer": {"kind": "op", "op": "token_prob", "of": "run.logits", "token": "prompts.answer"}
  },
  "saves": ["top5", "p_answer"]
}
```

After `clab run documents/readout.json`, `clab table --csv out/readout top5 p_answer` prints each `top_k` cell
whole (an aligned table cuts cells at 48 characters):

```text
example_id,row,pos,pos_token,input,top5,answer,p_answer
monday,0,5,",","Monday, Tuesday, Wednesday,","{""ids"":[3635,3217,290,3583,25381],""probs"":[0.7922688722610474,0.07389263063669205,0.043673012405633926,0.027983425185084343,0.013241195119917393],""tokens"":[""ĠThursday"",""ĠFriday"",""Ġand"",""ĠWednesday"",""Thursday""]}", Thursday,0.7922688722610474
counting,1,7,",","one, two, three, four,","{""ids"":[1936,2237,3478,1440,290],""probs"":[0.760408878326416,0.04302766174077988,0.030048692598938942,0.020308708772063255,0.01831410452723503],""tokens"":[""Ġfive"",""Ġsix"",""Ġten"",""Ġfour"",""Ġand""]}", five,0.760408878326416
eiffel,2,9,Ġof,The Eiffel Tower is in the city of,"{""ids"":[6342,3576,968,24377,11307],""probs"":[0.06997019797563553,0.0583544485270977,0.031019017100334167,0.02920154109597206,0.02351032756268978],""tokens"":[""ĠParis"",""ĠLondon"",""ĠNew"",""ĠAmsterdam"",""ĠBerlin""]}", Paris,0.06997019797563553
```

- `probs` is a float32 softmax over the whole vocabulary, so it equals `token_prob` of the same id.
- Each token is the tokenizer's piece, spelled as `pos_token` is: `ĠThursday` and `Thursday` are ids 3635 and 25381.
- At the last layer the logit lens lists the same tokens as this readout (` Paris`, ` London`, ` New` above).
- `detokenize` turns a `generate` step's ids into text (the `clab-steering-and-generation` skill), not logits.
- **A different answer at each position**, such as each next token of a repeat, is a shifted column read at a `pos`
  slice: [the recipe](references/attention.md#the-correct-next-token-at-each-position).

## Previous-token and induction heads

Use a stripe to find heads that look at a fixed distance from each query, for every head of every layer in one
forward. An induction head, at the second copy of a repeated sequence, attends to the token after the current
token's earlier copy. On 20 random words repeated once, that is key p − 19 at query p.

```json
{
  "description": "Previous-token and induction stripes on GPT-2 small, every head of every layer in one forward. Each `seqs` row is 20 random single-token words repeated once (40 tokens; GPT-2 adds no BOS). At query p >= 20 the token after the current token's earlier copy is key p - 19. `prev_score` is each head's mean weight on key p - 1 over queries 1 to 39. `induction_score` is its mean weight on key p - 19 over queries 20 to 39. `control_score` is the same read on 40 random words that never repeat, the null. `sink_score` is the weight on key 0, the first token. Eager attention, because sdpa never materializes the pattern.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32", "attn_implementation": "eager"},
  "data": {
    "seqs": {"rows": [
      {"example_id": "r0", "input": " palm printf bailout heads smarter seeded plaster labour salv mystery tropes floral sixty dart indemn dated chats weighed guiding stereo palm printf bailout heads smarter seeded plaster labour salv mystery tropes floral sixty dart indemn dated chats weighed guiding stereo"},
      {"example_id": "r1", "input": " varies deliber leaps chores buds mamm rulers tended cited motif races lightly voters fascism tackles sanct recal spends heavier bonding varies deliber leaps chores buds mamm rulers tended cited motif races lightly voters fascism tackles sanct recal spends heavier bonding"},
      {"example_id": "r2", "input": " swat alleg chap physi wishing aborted wait shall marked grad lousy books estates thumb footage glac decency sudo regener cents swat alleg chap physi wishing aborted wait shall marked grad lousy books estates thumb footage glac decency sudo regener cents"}
    ]},
    "fresh": {"rows": [
      {"example_id": "f0", "input": " cups till boolean happier connect sheet rejo ruling outlets field charges sellers arom pastry quint violate abused redist hither frozen undert class linem artwork anth applic subsequ aerobic overhe flaw process arrays stain footage ranks shelter tactile game mineral funk"},
      {"example_id": "f1", "input": " flowing retract intens annoyed precise tatt cinema enlight wrecked joke cook marches side process slaves erosion labs infect twice shower skept peers ugly attack scarf goof planet pouch dwarf notably sway open cartels rarely slender coastal doesn speeds peer obliged"},
      {"example_id": "f2", "input": " separat arrow roots barn layered wartime unit trail penal drought dreamed silicon freezer airing extrem stuffed swamp fuzzy planned snug midterm trading mosquit retains hopped suspic visuals acute padding smug credits sovere testify fatty provide blah schema nine timing tally"}
    ]}
  },
  "sites": {
    "prev_key": {"component": "attention_probabilities", "layers": ":", "select": {"key": {"offset": -1}}},
    "induction_key": {"component": "attention_probabilities", "layers": ":", "select": {"key": {"offset": -19}}},
    "first_key": {"component": "attention_probabilities", "layers": ":", "select": {"key": 0}}
  },
  "steps": {
    "run": {
      "kind": "forward",
      "input": "seqs.input",
      "interventions": {
        "prev": {"site": "prev_key", "pos": {"index": "1:40"}, "ops": [{"op": "mean", "over": ["pos"]}]},
        "induction": {"site": "induction_key", "pos": {"index": "20:40"}, "ops": [{"op": "mean", "over": ["pos"]}]},
        "sink": {"site": "first_key", "pos": {"index": "1:40"}, "ops": [{"op": "mean", "over": ["pos"]}]}
      }
    },
    "control": {
      "kind": "forward",
      "input": "fresh.input",
      "interventions": {
        "induction": {"site": "induction_key", "pos": {"index": "20:40"}, "ops": [{"op": "mean", "over": ["pos"]}]}
      }
    },
    "prev_score": {"kind": "op", "op": "mean", "of": "run.prev", "over": ["rows", "key"]},
    "induction_score": {"kind": "op", "op": "mean", "of": "run.induction", "over": ["rows", "key"]},
    "control_score": {"kind": "op", "op": "mean", "of": "control.induction", "over": ["rows", "key"]},
    "sink_score": {"kind": "op", "op": "mean", "of": "run.sink", "over": ["rows", "key"]},
    "top_induction": {"kind": "op", "op": "top_k", "of": "induction_score", "k": 5, "over": ["layers", "num_heads"]},
    "top_prev": {"kind": "op", "op": "top_k", "of": "prev_score", "k": 5, "over": ["layers", "num_heads"]},
    "is_sink": {"kind": "op", "op": "gt", "of": "sink_score", "with": 0.5},
    "sink_share": {"kind": "op", "op": "mean", "of": "is_sink", "over": ["layers", "num_heads"]},
    "control_max": {"kind": "op", "op": "max", "of": "control_score", "over": ["layers", "num_heads"]}
  },
  "saves": ["induction_score", "control_score", "prev_score", "sink_score", "top_induction", "top_prev", "control_max", "sink_share"]
}
```

- `{"offset": k}` is, at each query p the read holds, the row's key p + k; the `key` axis is one wide.
- The entry's `pos` picks the queries, and `mean` over `pos` inside the entry keeps only the mean.
- An offset read is NaN where p + k is no key of the row, before key 0 or past a shorter row's last; the
  reductions skip it. A key ahead of the query holds the causal mask's 0.
- The rows are random words that are one GPT-2 token each, space included, so each row is 40 tokens.
  `clab show` prints where each read landed (`[20,21,22,…,39] on all 3 rows`); check it on a new sequence.

Run it with `clab run documents/attention_stripes.json`. A `top_k` over `["layers", "num_heads"]` ranks every
head together; it has no rows, so read it from Python, which names each head `layer.head`:

<!-- test: skip -->
```python
from clab import Plan

plan = Plan.load("out/attention_stripes")
for name in ["top_induction", "top_prev"]:
    top = plan.results[name]
    heads = zip(top["ids"]["layers"].tolist(), top["ids"]["num_heads"].tolist(), top["values"].tolist())
    print(name, [f"{layer}.{head} {value:.3f}" for layer, head, value in heads])
```

```text
top_induction ['6.9 0.922', '5.5 0.916', '5.1 0.897', '7.10 0.892', '7.2 0.741']
top_prev ['4.11 0.994', '3.7 0.541', '2.2 0.521', '5.6 0.472', '6.8 0.452']
```

`clab table out/attention_stripes induction_score control_score prev_score sink_score` prints one line per
head; those heads:

```text
layer  num_heads  induction_score         control_score           prev_score             sink_score
4      11         9.613465161745614e-19   1.9059607153849205e-16  0.9935622215270996     0.0256479624658823
5      1          0.8967769742012024      0.001192394527606666    0.025976860895752907   0.5222184062004089
5      5          0.915633499622345       0.0021834017243236303   0.028818251565098763   0.4707637131214142
6      9          0.9224183559417725      0.0019522946095094085   0.02629624865949154    0.47449222207069397
7      2          0.7406908869743347      0.002379839075729251    0.027728542685508728   0.4875742495059967
7      10         0.8922820091247559      0.0012081883614882827   0.028035825118422508   0.4657127857208252
```

- **Five heads put over 0.5 on the induction key:** 5.1, 5.5, 6.9, 7.2 and 7.10. Head 4.11 puts 0.994 on the
  previous token.
- **The null is the control:** on words that never repeat, `control_max`, the largest score over every head, is
  0.033 (head 0.6). A head is an induction head where its score on the repeat stands far above its control.
- **Sinks:** `sink_share` is 0.507: 73 of 144 heads put over half their weight on key 0. GPT-2 adds no BOS, so
  key 0 is the first word. Split at query 20, the induction heads put 0.92 to 0.98 there over the first copy
  and under 0.09 over the second.
- **The bottom k**, for a control head: a `top_k` step over a `neg` step of `induction_score`; its `values` are the
  scores negated. Added to this document, it ranks 4.11, 2.2, 3.3, 3.7 and 3.2 lowest, each under 2e-6.
- **A stripe says where a head looks, not what it does.** The model's copying is the
  [next token at each position](references/attention.md#the-correct-next-token-at-each-position); test a candidate
  head with the `clab-ablation` or `clab-patching` skill.

## The tuned lens

A tuned lens trains one affine translator per layer, so that each layer's lens matches the model's own
next-token distribution. One `affine` featurizer over `layers` holds all of them and one fit trains them; the
`epochs: 0` point beside it must give exactly the logit lens, and the wiring check reads in the same forward. The
document, its null and its traps are [references/tuned-lens.md](references/tuned-lens.md).

## One head's pattern, sinks, a subject, queries

[references/attention.md](references/attention.md) has the `[rows, pos, num_heads, key]` layout, `top_k` over
`key` and the checks that a read is a pattern (sums to 1, zero ahead of the query). It also has the correct next
token at each position of a repeat, head entropy, the weight on a word that moves by row, and `attention_queries`
after RoPE, on SmolLM2.

## Elsewhere

| task | where |
|---|---|
| zero a head's pattern, knock out keys, mean-ablate a head | the `clab-ablation` skill |
| patch a pattern or a query from another prompt, patch a head's output | the `clab-patching` skill |
| the fit step's fields, `epochs: 0`, bundles | the `clab` skill's [featurizers-and-fit.md](../clab/references/featurizers-and-fit.md) |
| reuse a trained lens in another document; probes, SAEs, DAS | the `clab-trained-featurizers` skill |
| generate, score what a generation said, steer | the `clab-steering-and-generation` skill |
| `select`, components and their layouts | the `clab` skill's [sites-and-interventions.md](../clab/references/sites-and-interventions.md) |
| `top_k` members, `kl` direction, reductions over `over` | the `clab` skill's [ops.md](../clab/references/ops.md) |
| a refusal, or numbers wrong with nothing refused | the `clab-debugging` skill |

`clab describe ops` lists every op; `clab describe components` every component with its layout.
