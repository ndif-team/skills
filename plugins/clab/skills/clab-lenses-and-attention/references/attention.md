# Reading attention

Where one head looks, the checks that a read is a pattern, the weight on a word that moves by row, and queries
after RoPE. Every document here loads the model with `"attn_implementation": "eager"`; the stripes over every head
are in [SKILL.md](../SKILL.md). Each runs on the CPU in seconds.

## What breaks

- **Comparing an eager run with an sdpa run.** The two implementations differ in the last bits: GPT-2's
  ` Thursday` readout is 0.7922689 under sdpa and 0.7922713 under eager. Compare runs under one implementation.
- **A key axis read as a fixed width.** `key` is each row's own tokens, as wide as the input's longest row. Past a
  shorter row's last token the pattern is NaN, and `clab table` leaves those keys out.
- **A `key` selector for a word.** `select: {"key": 4}` is position 4 on every row, a different token wherever the
  rows differ. Read the whole key axis and find the word's key with a marker entry ([below](#a-word-that-moves-by-row)).
- **`top_k` over `key` read as probabilities.** Over an axis other than the vocabulary it gives `values`, the
  pattern's own weights, and `ids`, indices along `key`; under a `select` on `key`, indices into the selection.
- **Query heads and key heads counted alike.** `attention_keys` and `attention_values` are per KV head
  (`num_kv_heads`), before they are repeated across query heads. SmolLM2-135M has 9 query heads and 3 KV heads.

The layouts, by `clab describe components`:

| component | layout | what it is |
|---|---|---|
| `attention_probabilities` | `rows, pos, num_heads, key` | the pattern the values are mixed with; `pos` is the query |
| `attention_scores` | `rows, pos, num_heads, key` | the scaled, masked scores entering the softmax |
| `attention_queries` | `rows, pos, num_heads, qk_head_dim` | after the projection, the head split and RoPE |
| `attention_keys` | `rows, pos, num_kv_heads, qk_head_dim` | after RoPE, before `repeat_kv` |
| `attention_values` | `rows, pos, num_kv_heads, head_dim` | before `repeat_kv` |
| `attention_head_outputs` | `rows, pos, num_heads, head_dim` | each head's output, before the output projection |

`attention_premix`, every head's result as the output projection takes it, needs no eager. Patching or ablating
a head's output there is the `clab-patching` or `clab-ablation` skill's.

## One head's pattern, and the checks

On the repeated random words of the stripes document, this reads head 9 of layer 6 (the strongest induction head
there) at each row's last query, with its three heaviest keys. It adds every layer-6 head's entropy and two checks
that the read is a pattern:

```json
{
  "description": "Where one head looks, and the checks that a read of attention_probabilities is a pattern, on the repeated random words of attention_stripes.json. `look` is head 9 of layer 6 at each row's last query (39), `top3` its three heaviest keys. `focus` is every layer-6 head's entropy over keys at that query, in nats. `total` sums every layer-6 head's pattern over keys at every query (1 wherever it is a distribution); `ahead` is the weight on key p + 1, which the causal mask makes 0.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32", "attn_implementation": "eager"},
  "data": {
    "seqs": {"rows": [
      {"example_id": "r0", "input": " palm printf bailout heads smarter seeded plaster labour salv mystery tropes floral sixty dart indemn dated chats weighed guiding stereo palm printf bailout heads smarter seeded plaster labour salv mystery tropes floral sixty dart indemn dated chats weighed guiding stereo"},
      {"example_id": "r1", "input": " varies deliber leaps chores buds mamm rulers tended cited motif races lightly voters fascism tackles sanct recal spends heavier bonding varies deliber leaps chores buds mamm rulers tended cited motif races lightly voters fascism tackles sanct recal spends heavier bonding"},
      {"example_id": "r2", "input": " swat alleg chap physi wishing aborted wait shall marked grad lousy books estates thumb footage glac decency sudo regener cents swat alleg chap physi wishing aborted wait shall marked grad lousy books estates thumb footage glac decency sudo regener cents"}
    ]}
  },
  "sites": {
    "l6": {"component": "attention_probabilities", "layers": 6},
    "head_6_9": {"component": "attention_probabilities", "layers": 6, "select": {"num_heads": 9}},
    "next_key": {"component": "attention_probabilities", "layers": 6, "select": {"key": {"offset": 1}}}
  },
  "steps": {
    "run": {
      "kind": "forward",
      "input": "seqs.input",
      "interventions": {
        "look": {"site": "head_6_9", "pos": -1},
        "focus": {"site": {"component": "attention_scores", "layers": 6}, "pos": -1, "ops": [{"op": "entropy", "over": ["key"]}]},
        "total": {"site": "l6", "pos": {"index": ":"}, "ops": [{"op": "sum", "over": ["key"]}]},
        "ahead": {"site": "next_key", "pos": {"index": "0:39"}}
      }
    },
    "top3": {"kind": "op", "op": "top_k", "of": "run.look", "k": 3, "over": "key"},
    "total_min": {"kind": "op", "op": "min", "of": "run.total", "over": ["rows", "pos", "num_heads"]},
    "total_max": {"kind": "op", "op": "max", "of": "run.total", "over": ["rows", "pos", "num_heads"]},
    "ahead_max": {"kind": "op", "op": "max", "of": "run.ahead", "over": ["rows", "pos", "num_heads", "key"]}
  },
  "saves": ["run.look", "top3.ids", "top3.values", "run.focus", "total_min", "total_max", "ahead_max"]
}
```

- `run.look` is `[3, 1, 1, 40]`: rows, one query, one head (`select` keeps the axis), 40 keys.
- An int `layers` gives no `layers` axis; `":"` would add one first.
- `entropy` over `key` of `attention_scores` is the entropy of the head's softmax, in nats.

```bash
clab run documents/attention_pattern.json
clab table out/attention_pattern top3.ids top3.values
clab table out/attention_pattern total_min total_max ahead_max
```

```text
example_id  row  pos  pos_token  input                                             top3.ids     top3.values
r0          0    39   Ġstereo     palm printf bailout heads smarter seeded plast…  [[20,0,12]]  [[0.9770815968513489,0.01763666234910488,0.0014…
r1          1    39   Ġbonding    varies deliber leaps chores buds mamm rulers t…  [[20,0,19]]  [[0.927766740322113,0.05007994547486305,0.00577…
r2          2    39   Ġcents      swat alleg chap physi wishing aborted wait sha…  [[20,0,12]]  [[0.9132462739944458,0.07979655265808105,0.0016…
```

```text
total_min           total_max          ahead_max
0.9999997615814209  1.000000238418579  0.0
```

An aligned table cuts cells at 48 characters; `clab table --csv` prints them whole.

- **Key 20 takes 0.91 to 0.98** on every row: the first word of the second copy, the token after the last
  query's earlier copy at 19. Key 0, the first word, is second.
- **The checks:** every head's keys sum to 1 within float32 rounding at every query, and the weight one key
  ahead of the query is exactly 0.0. A read failing either is not the pattern.
- **`clab table out/attention_pattern run.look`** prints one line per key, with `key` and `key_token`:

```text
example_id  row  pos  pos_token  key  key_token  input                                             value
r0          0    39   Ġstereo    0    Ġpalm       palm printf bailout heads smarter seeded plast…  [0.01763666234910488]
r0          0    39   Ġstereo    20   Ġpalm       palm printf bailout heads smarter seeded plast…  [0.9770815968513489]
```

`clab table --csv out/attention_pattern run.focus` gives each layer-6 head's entropy at the last query. Head 9
is at 0.13 to 0.35 nats on the three rows, the other heads at 0.82 to 3.09: a focused head has low entropy.

## A word that moves by row

A `key` counted from the start, and an offset from the query, both miss a word at a different position on each
row. Read the whole key axis, and add a marker: an entry scoped on the word, with no ops, which no step names.
Its record holds where the word landed on each row, and that position is the word's key.

```json
{
  "description": "How much the last token attends to the subject, a word at a different position on each row. `pattern` is every head of every layer at the last query, over the whole key axis; `entity` is a marker: a read scoped on the subject's last token that no step names, whose record holds where that token is on each row, the key to index the pattern at.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32", "attn_implementation": "eager"},
  "data": {
    "facts": {"rows": [
      {"example_id": "colosseum", "input": "The Colosseum is in the city of", "entity": " Colosseum"},
      {"example_id": "eiffel", "input": "The famous Eiffel Tower is in the city of", "entity": " Eiffel Tower"},
      {"example_id": "needle", "input": "Everyone knows that the Space Needle is in the city of", "entity": " Space Needle"}
    ]}
  },
  "sites": {"pattern": {"component": "attention_probabilities", "layers": ":"}},
  "steps": {
    "run": {
      "kind": "forward",
      "input": "facts.input",
      "interventions": {
        "pattern": {"site": "pattern", "pos": -1},
        "entity": {"site": "pattern", "pos": {"scope": {"column": "facts.entity"}, "index": -1}}
      }
    }
  },
  "saves": ["run.pattern"]
}
```

`clab show out/attention_entity` prints the marker's positions (`run.entity   —   [4], [5], [6]`). In Python:

<!-- test: skip -->
```python
from clab import Plan

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

Head 9.8 puts the most weight on the subject's last token on all three rows. The marker reads the pattern at
the subject's own query, which nothing uses. How a scope finds a word (`column`, `text`, `occurrence`) is the
`clab` skill's [positions.md](../../clab/references/positions.md).

## Queries after RoPE

`attention_queries` and `attention_keys` are taken after RoPE, so a query carries the position it was computed
at. At layer 0 of a RoPE model the stream is the token's embedding alone, so two copies of one token have the same
query before the rotation. After it, each head's query keeps its norm and changes direction:

```json
{
  "description": "attention_queries is taken after RoPE. At layer 0 of SmolLM2-135M the stream is the token's embedding alone, so the two ` is` of this row have the same query before the rotation. After it, each head's query has the same norm at both positions (`norm_spread`, the spread of the norms across them, is rounding) while the vectors differ (`query_spread`).",
  "model": {"key": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac", "dtype": "float32", "attn_implementation": "eager"},
  "data": {"text": {"rows": [{"input": "The cat is here and the dog is there"}]}},
  "steps": {
    "run": {
      "kind": "forward",
      "input": "text.input",
      "interventions": {
        "q": {"site": {"component": "attention_queries", "layers": 0}, "pos": {"scope": {"text": " is", "occurrence": ":"}}}
      }
    },
    "q_norm": {"kind": "op", "op": "norm", "of": "run.q", "over": ["qk_head_dim"]},
    "norm_std": {"kind": "op", "op": "std", "of": "q_norm", "over": ["pos"]},
    "norm_spread": {"kind": "op", "op": "max", "of": "norm_std", "over": ["rows", "num_heads"]},
    "query_std": {"kind": "op", "op": "std", "of": "run.q", "over": ["pos"]},
    "query_spread": {"kind": "op", "op": "max", "of": "query_std", "over": ["rows", "num_heads", "qk_head_dim"]}
  },
  "saves": ["q_norm", "norm_spread", "query_spread"]
}
```

```text
$ clab table out/query_rope norm_spread query_spread
norm_spread          query_spread
9.5367431640625e-07  3.0358388423919678
```

- The scope finds both ` is` (`occurrence: ":"`), at positions 2 and 7; SmolLM2 adds no BOS here.
- `norm` and `std` take the axis to remove in `over`, here `qk_head_dim` and `pos`.
- **Norms equal, vectors not:** the norms agree to 1e-6 across the two positions, while one coordinate's
  standard deviation across them reaches 3.0.
- **Keys are rotated too:** on `attention_keys`, with `num_kv_heads` for `num_heads` in `over`, the same steps
  give 4.8e-7 and 2.98.
- **GPT-2 has no RoPE.** It adds a learned position embedding to the stream, so the same document on GPT-2 gives a
  `norm_spread` of 1.15: the norms differ too.
- **A query swapped between prompts of different lengths moves the position its head looks from.** Nothing
  refuses it; patching queries is the `clab-patching` skill's.

## Elsewhere

- Zeroing a head's pattern, knocking out keys through the scores or the pattern, `renormalize` with
  `"norm": "sum"`: the `clab-ablation` skill.
- Making a head attend as it did on another prompt, a query swap: the `clab-patching` skill.
- `select` over `num_heads`, `key` and offsets, in full: the `clab` skill's
  [sites-and-interventions.md](../../clab/references/sites-and-interventions.md).
