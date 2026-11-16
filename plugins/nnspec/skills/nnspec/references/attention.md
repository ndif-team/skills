# Attention patterns

Where heads look: read `attention_probabilities`, or the scores, queries and keys behind it, and reduce the
read to a score per head. Nothing here writes into the forward. A document that reads the attention interior
loads the model with `"attn_implementation": "eager"` in its `model` block; without it those components are
refused ([which ones](sites-and-interventions.md#attention-needs-eager)). `nnspec describe components` gives
each one's layout: `[rows, pos, num_heads, key]` for the pattern and the scores, with `pos` the query.

## What breaks

- **An eager run compared with an sdpa run.** The two implementations differ in the last bits: P(` Mary`) on
  the first IOI prompt of [SKILL.md](../SKILL.md) is 0.4460211992263794 under sdpa and 0.4460209608078003 under
  eager. Compare runs under one implementation, the clean run included.
- **A `key` index names a position, not a role.** `select: {"key": 4}` is position 4 at every query of every
  row. For the key at a fixed distance from each query, a previous-token or induction score, write `{"key":
  {"offset": -1}}`, a read only; a key counted from the end is refused. For a word that moves by row, read the
  whole key axis and add a marker entry ([below](#a-word-that-moves-by-row)).
- **The `key` axis is ragged.** It is as wide as the input's longest row and NaN past a shorter row's last
  token. Reductions skip those keys and `nnspec table` leaves them out.
- **A `top_k` id read as a model index.** Ids count along the value's own axes: over `key` under a `select`,
  within the selection; over `layers` read at `"layers": "4:"`, head 6.9 comes back as `layers` 2. Over `key`
  a `top_k` gives `values`, the pattern's own weights, where over a vocabulary it gives `probs`.
- **Query heads and key heads counted alike.** `attention_keys` and `attention_values` are per KV head
  (`num_kv_heads`), before they are repeated across query heads: SmolLM2-135M has 9 query heads and 3 KV heads.
  The pattern, the scores and the queries stay per query head (`num_heads`) on such a model.
- **A stripe read as a function.** A high score says where a head looks, not what it does: test a candidate by
  [ablating](ablation.md) or [patching](patching-heads-neurons-and-paths.md) it.

## Previous-token and induction heads

Use a stripe to find heads that look at a fixed distance from each query, for every head of every layer in one
forward. At the second copy of a repeated sequence an induction head attends to the token after the current
token's earlier copy: on 20 random words repeated once, key p − 19 at query p. Saved as
`documents/attention_stripes.json`:

```json
{
  "description": "Previous-token, induction and sink scores for every head of every layer of GPT-2, in one forward. Each seqs row is 20 random one-token words repeated once, so at query p >= 20 the token after the current token's earlier copy is key p - 19. control_score is the induction read on 40 words that never repeat, the null.",
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
    "run": {"kind": "forward", "input": "seqs.input", "interventions": {
      "prev": {"site": "prev_key", "pos": {"index": "1:40"}, "ops": [{"op": "mean", "over": ["pos"]}]},
      "induction": {"site": "induction_key", "pos": {"index": "20:40"}, "ops": [{"op": "mean", "over": ["pos"]}]},
      "sink": {"site": "first_key", "pos": {"index": "1:40"}, "ops": [{"op": "mean", "over": ["pos"]}]}}},
    "control": {"kind": "forward", "input": "fresh.input", "interventions": {
      "induction": {"site": "induction_key", "pos": {"index": "20:40"}, "ops": [{"op": "mean", "over": ["pos"]}]}}},
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

- **`{"offset": k}` is, at each query p the read holds, the row's key p + k**, so the `key` axis is one wide; a
  list of offsets gives one key each. The read is NaN where p + k is no key of the row.
- **The entry's `pos` picks the queries**, and `mean` over `pos` inside the entry keeps only the mean.
- **Each word is one GPT-2 token, space included**, so a row is 40 tokens; GPT-2 adds no BOS. On other text,
  check the read's positions in `nnspec show` (`[20,21,22,…,39] on all 3 rows` here) before trusting an offset.

A `top_k` over `["layers", "num_heads"]` ranks every head together. It has no rows, so `nnspec table` refuses
it (`… read it from plan.results`); read it in Python:

```python
from nnspec import Plan

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

`nnspec table out/attention_stripes induction_score control_score prev_score sink_score` prints a line per
head; for the heads above:

```text
layer  num_heads  induction_score         control_score           prev_score             sink_score
4      11         9.613465161745614e-19   1.9059607153849205e-16  0.9935622215270996     0.0256479624658823
5      1          0.8967769742012024      0.001192394527606666    0.025976860895752907   0.5222184062004089
5      5          0.915633499622345       0.0021834017243236303   0.028818251565098763   0.4707637131214142
6      9          0.9224183559417725      0.0019522946095094085   0.02629624865949154    0.47449222207069397
7      2          0.7406908869743347      0.002379839075729251    0.027728542685508728   0.4875742495059967
7      10         0.8922820091247559      0.0012081883614882827   0.028035825118422508   0.4657127857208252
```

- **The null is the control.** On words that never repeat the largest induction score of any head,
  `control_max`, is 0.033. A head is an induction head where its score on the repeat stands far above its
  control: 5.1, 5.5, 6.9, 7.2 and 7.10 here. Head 4.11 puts 0.994 on the previous token.
- **Sinks.** `sink_share` is 0.507: 73 of 144 heads put over half their weight on key 0, the first word.
- **The bottom k**, for a control head, is a `top_k` over a `neg` step of the score, its `values` the scores
  negated: 4.11, 2.2, 3.3, 3.7 and 3.2 here.

## One head's pattern, and the checks

Read one head at one query to see every key it weighs, and check once per model that the read is a pattern.
These keys replace the `sites`, `steps` and `saves` of `attention_stripes.json`:

```json
{
  "sites": {
    "l6": {"component": "attention_probabilities", "layers": 6},
    "head_6_9": {"component": "attention_probabilities", "layers": 6, "select": {"num_heads": 9}},
    "next_key": {"component": "attention_probabilities", "layers": 6, "select": {"key": {"offset": 1}}}
  },
  "steps": {
    "run": {"kind": "forward", "input": "seqs.input", "interventions": {
      "look": {"site": "head_6_9", "pos": -1},
      "focus": {"site": {"component": "attention_scores", "layers": 6}, "pos": -1, "ops": [{"op": "entropy", "over": ["key"]}]},
      "total": {"site": "l6", "pos": {"index": ":"}, "ops": [{"op": "sum", "over": ["key"]}]},
      "ahead": {"site": "next_key", "pos": {"index": "0:39"}}}},
    "top3": {"kind": "op", "op": "top_k", "of": "run.look", "k": 3, "over": "key"},
    "total_min": {"kind": "op", "op": "min", "of": "run.total", "over": ["rows", "pos", "num_heads"]},
    "total_max": {"kind": "op", "op": "max", "of": "run.total", "over": ["rows", "pos", "num_heads"]},
    "ahead_max": {"kind": "op", "op": "max", "of": "run.ahead", "over": ["rows", "pos", "num_heads", "key"]}
  },
  "saves": ["run.look", "top3.ids", "top3.values", "run.focus", "total_min", "total_max", "ahead_max"]
}
```

```text
$ nnspec table out/attention_pattern total_min total_max ahead_max
total_min           total_max          ahead_max
0.9999997615814209  1.000000238418579  0.0
```

- **The checks**: every head's keys sum to 1 within float32 rounding at every query, and the weight one key
  ahead of the query is exactly 0.0. A read that fails either is not the pattern.
- **`run.look` is `[3, 1, 1, 40]`**: rows, one query, one head (`select` keeps the axis), 40 keys. Tabled, it
  is a line per key, with `key` and `key_token` columns. `top3.ids` is `[[20,0,12]]` on row 0: key 20, the
  word after the last query's earlier copy, takes 0.91 to 0.98 on the three rows, and key 0 comes second.
- **A head's entropy is `entropy` over `key` of `attention_scores`**: the op takes the softmax itself, so give
  it the scores, not the pattern. `run.focus` is 0.13 to 0.35 nats for head 9 and 0.82 to 3.09 for the other
  layer-6 heads: a focused head has low entropy.

## The correct next token at each position

Use it to measure the behaviour the stripes suggest: on the repeat, each query of the second copy has its own
correct next token. One read scores them all against a shifted column, a text column holding on each row the
tokens the read's positions predict; `token_prob` scores the read's position i against the column's token i
([answer columns](data.md#answer-columns)). `next_token.json` reads no attention, so its `model` block has no
`attn_implementation`. Its data is `attention_stripes.json`'s `seqs` rows, each with a column `next` of its
words 2 to 20 (row 0's is `" printf bailout heads … guiding stereo"`), and the rest is:

```json
{
  "sites": {"head": {"component": "logits"}},
  "steps": {
    "run": {"kind": "forward", "input": "seqs.input", "interventions": {
      "first": {"site": "head", "pos": {"index": "0:19"}},
      "second": {"site": "head", "pos": {"index": "20:39"}}}},
    "p_first": {"kind": "op", "op": "token_prob", "of": "run.first", "token": "seqs.next"},
    "p_second": {"kind": "op", "op": "token_prob", "of": "run.second", "token": "seqs.next"},
    "mean_first": {"kind": "op", "op": "mean", "of": "p_first"},
    "mean_second": {"kind": "op", "op": "mean", "of": "p_second"}
  },
  "saves": ["p_first", "p_second", "mean_first", "mean_second"]
}
```

```text
$ nnspec table out/next_token mean_first mean_second
mean_first              mean_second
1.3536184269469231e-05  0.6893020272254944
```

- **The shift**: position 20, the second copy's ` palm`, scores ` printf`. Queries 0 to 18 of the first copy
  predict the same words, so one column serves both reads, and the first copy is the control.
- **An unshifted column of the right length is not refused.** The second copy's own 20 words, read at queries
  20 to 39, score each position against the token it reads: 0.00025 where the shifted column gives 0.69.
- **Table the two reads apart**: their positions differ, so `p_first` and `p_second` join to no row.

## A word that moves by row

A `key` counted from the start and an offset from the query both miss a word at a different position on each
row. Read the whole key axis and add a marker: an entry scoped on the word, with no ops, that no step names. Its
record holds where the word landed on each row, and that position is the word's key. `attention_entity.json`
keeps the `model` block above:

```json
{
  "data": {"facts": {"rows": [
    {"example_id": "colosseum", "input": "The Colosseum is in the city of", "entity": " Colosseum"},
    {"example_id": "eiffel", "input": "The famous Eiffel Tower is in the city of", "entity": " Eiffel Tower"},
    {"example_id": "needle", "input": "Everyone knows that the Space Needle is in the city of", "entity": " Space Needle"}
  ]}},
  "sites": {"pattern": {"component": "attention_probabilities", "layers": ":"}},
  "steps": {
    "run": {"kind": "forward", "input": "facts.input", "interventions": {
      "pattern": {"site": "pattern", "pos": -1},
      "entity": {"site": "pattern", "pos": {"scope": {"column": "facts.entity"}, "index": -1}}}}
  },
  "saves": ["run.pattern"]
}
```

```python
from nnspec import Plan

plan = Plan.load("out/attention_entity")
pattern = plan.results["run.pattern"]                      # [layers, rows, pos, num_heads, key]
tokens = plan.run.tokens["facts.input"]
for row, (key,) in enumerate(plan.steps["run"].interventions["entity"].positions["facts"]):
    weight = pattern[:, row, 0, :, key]                     # [layers, num_heads]: each head's weight on the subject
    layer, head = divmod(int(weight.argmax()), weight.shape[1])
    print(row, key, tokens[row][key], f"{layer}.{head}", round(float(weight.max()), 3))
```

```text
0 4 um 9.8 0.496
1 5 ĠTower 9.8 0.218
2 6 le 9.8 0.233
```

`nnspec show out/attention_entity` prints the marker's positions too (`run.entity   —   [4], [5], [6]`). The
prompts are 10, 11 and 12 tokens, so `run.pattern` is 12 keys wide and NaN at the two keys row 0 lacks.

## Queries and keys after RoPE

`attention_queries` and `attention_keys` are taken after the projection, the head split and RoPE, so each
carries the position it was computed at. At layer 0 of a RoPE model two copies of one token have the same query
before the rotation; after it each head's query keeps its norm and changes direction. On SmolLM2
(`HuggingFaceTB/SmolLM2-135M-Instruct`, eager) and the row "The cat is here and the dog is there", this entry
reads the query at both ` is`:

```json
{"q": {"site": {"component": "attention_queries", "layers": 0}, "pos": {"scope": {"text": " is", "occurrence": ":"}}}}
```

- **Norms equal, vectors not.** `norm` over `qk_head_dim`, then `std` over `pos`, is at most 9.5e-07 over the
  heads; `std` over `pos` of the vectors themselves reaches 3.04. A query or key moved between prompts carries
  its old position, which matters when [patching one](patching-heads-neurons-and-paths.md).
- **Keys are rotated too.** `attention_keys` gives 4.8e-07 and 2.98, over 3 KV heads (`num_kv_heads` in place
  of `num_heads`) where the queries have 9.
- **GPT-2 has no RoPE.** It adds a learned position embedding to the stream, so on GPT-2 the same reductions
  give 1.15 for the norms: they differ too.
