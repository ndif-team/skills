# Patching

Patching asks whether a value carries what decides the answer: read it on a counterfactual prompt, swap it into
the base prompt at the same place, and score the base run. [SKILL.md](../SKILL.md#a-complete-experiment) has
the interchange at the last token, its null and a sweep over layers. This guide puts the patch on a word, maps
it over layers and positions, pairs prompts of different lengths, and patches windows, noised runs and
generations.

In every document here `input` is the base, the prompt that is run, patched and scored, and `counterfactual`
the prompt the patched value is read on. Denoising asks what restores the answer: `input` holds the corrupted
prompt and `counterfactual` the clean one, and the metric's `a` is the clean prompt's answer. Noising asks what
breaks it: `input` holds the clean prompt and `counterfactual` the corrupted one. It is one document, with the
two columns and the two answers exchanged.

One head, a head set, neurons, one path or a head's pattern:
[heads, neurons and paths](patching-heads-neurons-and-paths.md). Every component ranked from one backward:
[attribution patching](patching-attribution.md).

## What breaks

- **A swept `pos`'s coordinate is not the table's `pos` column.** `at` holds the swept value, and `pos` is
  where the tabled read landed. Label the sweep (`"label": "patch_pos"`) to keep the two apart.
- **`match` alone hides a graded effect.** It stays 0 until the top token flips: at layer 7, the last position
  of the map below has `iia_mean` 0.0 while the logit difference moved by +1.22. Read `logit_diff` beside it.
- **The last layer at the last position always gives the full effect**, since it reproduces the
  counterfactual's logits. A layer sweep at `-1` measures how late the answer can still be swapped in, not
  where it is computed.
- **A counted `pos` on prompts of different lengths passes wherever their shared ending allows**, whatever
  words it pairs ([below](#prompts-of-different-lengths)).

## The interchange at a word

Does the value at one word carry the answer? Each row holds the base prompt, the counterfactual prompt, each
prompt's word and each answer, and each side's `pos` is a scope on its own column. Saved as
`documents/interchange.json`:

```json
{
  "description": "The base interchange on GPT-2: the residual stream leaving block 4 at the day's token, read on the counterfactual prompt and swapped into the base prompt at its own day's token. The null swaps the base's own value back in and must equal the clean run.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"example_id": "mon-thu", "input": "If today is Monday, tomorrow is", "day": "Monday", "base_answer": " Tuesday",
     "counterfactual": "If today is Thursday, tomorrow is", "cf_day": "Thursday", "cf_answer": " Friday"},
    {"example_id": "tue-fri", "input": "If today is Tuesday, tomorrow is", "day": "Tuesday", "base_answer": " Wednesday",
     "counterfactual": "If today is Friday, tomorrow is", "cf_day": "Friday", "cf_answer": " Saturday"},
    {"example_id": "wed-sat", "input": "If today is Wednesday, tomorrow is", "day": "Wednesday", "base_answer": " Thursday",
     "counterfactual": "If today is Saturday, tomorrow is", "cf_day": "Saturday", "cf_answer": " Sunday"},
    {"example_id": "thu-sun", "input": "If today is Thursday, tomorrow is", "day": "Thursday", "base_answer": " Friday",
     "counterfactual": "If today is Sunday, tomorrow is", "cf_day": "Sunday", "cf_answer": " Monday"},
    {"example_id": "fri-mon", "input": "If today is Friday, tomorrow is", "day": "Friday", "base_answer": " Saturday",
     "counterfactual": "If today is Monday, tomorrow is", "cf_day": "Monday", "cf_answer": " Tuesday"},
    {"example_id": "sat-tue", "input": "If today is Saturday, tomorrow is", "day": "Saturday", "base_answer": " Sunday",
     "counterfactual": "If today is Tuesday, tomorrow is", "cf_day": "Tuesday", "cf_answer": " Wednesday"},
    {"example_id": "sun-wed", "input": "If today is Sunday, tomorrow is", "day": "Sunday", "base_answer": " Monday",
     "counterfactual": "If today is Wednesday, tomorrow is", "cf_day": "Wednesday", "cf_answer": " Thursday"}
  ]}},
  "sites": {"resid": {"component": "layer_output", "layers": 4}, "head": {"component": "lm_head_output"}},
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": {
      "v_self": {"site": "resid", "pos": {"scope": {"column": "pairs.day"}, "index": -1}},
      "logits": {"site": "head", "pos": -1}}},
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {
      "v_cf": {"site": "resid", "pos": {"scope": {"column": "pairs.cf_day"}, "index": -1}}}},
    "null": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "resid", "pos": {"scope": {"column": "pairs.day"}, "index": -1},
                "ops": [{"op": "swap", "with": "clean.v_self"}]},
      "logits": {"site": "head", "pos": -1}}},
    "patched": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "resid", "pos": {"scope": {"column": "pairs.day"}, "index": -1},
                "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
      "logits": {"site": "head", "pos": -1}}},
    "clean_top": {"kind": "op", "op": "match", "of": "clean.logits", "expected": "pairs.base_answer"},
    "clean_ld": {"kind": "op", "op": "logit_diff", "of": "clean.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "null_ld": {"kind": "op", "op": "logit_diff", "of": "null.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "ld": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "iia": {"kind": "op", "op": "match", "of": "patched.logits", "expected": "pairs.cf_answer"},
    "p_cf": {"kind": "op", "op": "token_prob", "of": "patched.logits", "token": "pairs.cf_answer"},
    "effect": {"kind": "op", "op": "sub", "of": "ld", "with": "null_ld"}
  },
  "saves": ["clean_top", "clean_ld", "null_ld", "ld", "iia", "p_cf", "effect"]
}
```

```text
$ nnspec table out/interchange clean_ld null_ld ld iia
example_id  row  pos  pos_token  input                               cf_answer   base_answer  clean_ld             null_ld              ld                  iia
mon-thu     0    6    Ġis        If today is Monday, tomorrow is      Friday      Tuesday     -1.043609619140625   -1.043609619140625   1.0651702880859375  1.0
tue-fri     1    6    Ġis        If today is Tuesday, tomorrow is     Saturday    Wednesday   -1.4108200073242188  -1.4108200073242188  2.1431808471679688  1.0
...
```

- **Check the model does the task first**: `clean_top` is 1.0 on all 7 rows.
- **`index: -1` inside the scope takes the word's last token**, one position on every row whatever the word's
  length. `nnspec show out/interchange` prints `[3] on all 7 rows` for the reads and the writes.
- **The null equals the clean run to the bit** on every row; `effect`, the patched score less the null's, is
  2.11 to 5.10, and the patch flips every row (`iia` 1.0).

## Choosing the metric

`logit_diff` of the counterfactual's answer against the base's is the graded effect. `match`, as its mean over
rows, is interchange accuracy (IIA), and `token_prob` a soft accuracy between the two. For how far the whole
prediction moved, add `{"kind": "op", "op": "kl", "of": "patched.logits", "against": "clean.logits"}`, or `js`
with the same arguments; for how far the residual stream moved, read it at a later site in both runs and take
`{"kind": "op", "op": "cosine", "of": "patched.resid", "with": "clean.resid"}`. On `interchange.json`, with
`resid` an entry of `clean`, `null` and `patched` that reads `layer_output` at layers `[2, 6, 10]` and
`{"index": "2:5"}`:

- **`kl` is KL(patched ‖ clean)**, 0.25 to 0.92 nats over the rows; `js` is symmetric and at most ln 2.
- **A null's `kl` is exactly 0.0; its `cosine` is 1 only to rounding** (1.0000001192092896 at layer 2).
  Compare cosines with a tolerance, logits and divergences to the bit.
- **The cosine drops only where the write reaches**: at the day's token it is 0.92 at layer 6 and 0.98 at layer
  10, and 1 at layer 2, before the write at layer 4.

## A layer-by-position map

Use it to find where in depth and in the sequence the answer sits. The read and the write sweep `pos` under one
label, so they zip: position p is read on the counterfactual and written at p. The `layers` sweep on the shared
site crosses them: 12 × 7 = 84 points. `layer_pos_map.json` is `interchange.json` with

- `"layers": {"sweep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}` on `resid`;
- `{"index": {"sweep": [0, 1, 2, 3, 4, 5, 6], "label": "patch_pos"}}` as the `pos` of `clean.v_self`,
  `counterfactual.v_cf` and both `patch` entries;
- a `"logits": {"site": "head", "pos": -1}` entry in `counterfactual`;
- for its op steps, `clean_ld`, `null_ld`, `ld` and `iia` as they are, `cf_ld`, the same `logit_diff` of
  `counterfactual.logits`, and a `mean` step over each of the five (`clean_mean`, `cf_mean`, `null_mean`,
  `ld_mean`, `iia_mean`); it saves the means and `ld`.

`clean` and `counterfactual` only read, so one forward of each serves all 84 points; `null` and `patched` run
one forward per point. `nnspec table` keys each point in `at`:

```text
$ nnspec table out/layer_pos_map clean_mean cf_mean
clean_mean           cf_mean
-1.6586096286773682  1.7260316610336304
$ nnspec table out/layer_pos_map ld_mean null_mean iia_mean
at                           ld_mean              null_mean            iia_mean
{"layers":0,"patch_pos":2}   -1.6586096286773682  -1.6586096286773682  0.0
{"layers":0,"patch_pos":3}   1.7269853353500366   -1.6586096286773682  1.0
{"layers":8,"patch_pos":3}   -0.0129231046885252  -1.6586096286773682  0.2857142984867096
{"layers":11,"patch_pos":6}  1.7260316610336304   -1.6586096286773682  1.0
...
```

In Python a swept result is a dict keyed by coordinate, a tuple in the order of `plan.sweeps` (`layers`, then
`patch_pos`):

```python
from nnspec import Plan

plan = Plan.load("out/layer_pos_map")
ld = plan.results["ld_mean"]                        # {(layer, patch_pos): tensor}
tokens = plan.run.tokens["pairs.input"][0]
print("     " + "".join(f"{t:>10}" for t in tokens))
for layer in range(12):
    print(f"L{layer:<4}" + "".join(f"{ld[(layer, p)].item():10.2f}" for p in range(7)))
```

```text
             If    Ġtoday       Ġis   ĠMonday         , Ġtomorrow       Ġis
L0        -1.66     -1.66     -1.66      1.73     -1.66     -1.67     -1.66
...
L5        -1.66     -1.66     -1.66      1.67     -1.64     -1.68     -1.61
L6        -1.66     -1.66     -1.66      1.15     -1.64     -1.67     -1.12
L7        -1.66     -1.66     -1.66      0.44     -1.63     -1.66     -0.44
L8        -1.66     -1.66     -1.66     -0.01     -1.63     -1.65      0.02
L9        -1.66     -1.66     -1.66     -1.16     -1.61     -1.62      1.10
L10       -1.66     -1.66     -1.66     -1.69     -1.60     -1.64      1.67
L11       -1.66     -1.66     -1.66     -1.66     -1.66     -1.66      1.73
```

Read the nulls and the corners before the middle:

- **`null_mean` equals `clean_mean` at all 84 points.**
- **Positions 0 to 2 equal the clean run exactly** at every layer: the two prompts share those tokens.
- **Layer 11 at the last position gives `cf_mean` to the bit**, and every other position at layer 11 the clean
  run's score, since no later block moves it.
- **The band between is the finding**: the day's token carries the answer through layer 5, and between layers
  6 and 9 it moves to the last token.

The map counts positions, which pairs a word with the same word only because both prompts come from one
template and tokenize to 7 tokens. Check that each point's write landed where its read did: `nnspec show`
groups a swept entry's points by where they landed, and [output](output.md#in-python) has the loop that prints
each point's positions beside its `at`.

## Prompts of different lengths

Scope each side when the prompts differ in length or the word is several tokens. `entity_scope.json` is
`interchange.json` with this `data`, `pairs.entity` and `pairs.cf_entity` in the scopes, and `"layers":
{"sweep": [0, 4, 8]}` on `resid`:

```json
{"pairs": {"rows": [
  {"example_id": "eiffel-brandenburg", "input": "The Eiffel Tower is in the city of", "entity": "Eiffel Tower", "base_answer": " Paris",
   "counterfactual": "The Brandenburg Gate is in the city of", "cf_entity": "Brandenburg Gate", "cf_answer": " Berlin"},
  {"example_id": "brandenburg-vatican", "input": "The Brandenburg Gate is in the city of", "entity": "Brandenburg Gate", "base_answer": " Berlin",
   "counterfactual": "The Vatican is in the city of", "cf_entity": "Vatican", "cf_answer": " Rome"},
  {"example_id": "vatican-buckingham", "input": "The Vatican is in the city of", "entity": "Vatican", "base_answer": " Rome",
   "counterfactual": "Buckingham Palace is in the city of", "cf_entity": "Buckingham Palace", "cf_answer": " London"},
  {"example_id": "buckingham-louvre", "input": "Buckingham Palace is in the city of", "entity": "Buckingham Palace", "base_answer": " London",
   "counterfactual": "The Louvre is in the city of", "cf_entity": "Louvre", "cf_answer": " Paris"}
]}}
```

```text
$ nnspec show out/entity_scope
  ...
  clean.v_self         all 3 points  [4], [3], [1], [3]
  counterfactual.v_cf  all 3 points  [3], [1], [3], [2]
  patched.patch        all 3 points  [4], [3], [1], [3]
$ nnspec table out/entity_scope null_ld ld iia
at            example_id           ...  null_ld              ld                   iia
{"layers":0}  eiffel-brandenburg   ...  -1.0906295776367188  -1.1459503173828125  0.0
{"layers":0}  buckingham-louvre    ...  -3.7453079223632812  3.057952880859375    1.0
{"layers":8}  buckingham-louvre    ...  -3.7453079223632812  2.2940673828125      1.0
```

The read and the write land on a different position in each row, each on its own landmark's last token, and
`null_ld` equals `clean_ld` on every row at every layer. That token flips two rows at layer 0 and one at layers
4 and 8: it does not carry the whole landmark on every row.

- **Between prompts of different lengths a counted `pos` pairs only where every token between it and the end
  it counts from is the same on both.** These end ` is in the city of`, five tokens: `-6` runs, pairing
  ` Tower` with ` Gate` on row 0, and `-7` is refused, naming the two tokens it would pair (`'enburg'` with
  `'el'`).
- **A scope read whole pairs only where every row's word is as many tokens**; here it is refused, naming each
  row's count. Take one token with `index`, as above, or reduce over `pos`
  ([pairing](positions.md#pairing-with-another-forward)).

## Windows and the whole prompt

A window is its positions patched together: a static slice as the `pos` of the read and of the write.
`window.json` is `interchange.json` with `{"index": {"sweep": ["-4:-1", "0:7"], "label": "window"}}` as the
`pos` of `clean.v_self`, `counterfactual.v_cf` and both `patch` entries, a `logits` read at the head in
`counterfactual` scored as `cf_ld`, and a saved `mean` step over each of `clean_ld`, `cf_ld`, `null_ld` and
`ld`. `-4:-1` is the day, the comma and ` tomorrow`; `0:7` is the whole prompt.

```text
$ nnspec table out/window clean_mean cf_mean
clean_mean           cf_mean
-1.6586096286773682  1.7260316610336304
$ nnspec table out/window ld_mean null_mean
at                  ld_mean             null_mean
{"window":"-4:-1"}  1.7342965602874756  -1.6586096286773682
{"window":"0:7"}    1.7260316610336304  -1.6586096286773682
```

- **The whole prompt patched gives the counterfactual run's own score to the bit**: it is the positive control
  of a window, and it holds only where both prompts tokenize to one length.
- **One window entry equals one entry per position**: the first window written as three entries at `-4`, `-3`
  and `-2` gives the same logits to the bit.

## Corrupt and restore

The causal-tracing form patches clean states into a corrupted run of the same prompt, so it needs no
counterfactual. Corrupt the landmark's token embeddings with noise from a root bundle, then restore the clean
residual stream one layer at a time, at the landmark's last token and at the prompt's last token. These keys
replace `entity_scope.json`'s `sites`, `steps` and `saves`, in `corrupt_restore.json`:

```json
{
  "sites": {
    "embed": {"component": "token_embeddings"},
    "every_layer": {"component": "layer_output", "layers": ":"},
    "one_layer": {"component": "layer_output", "layers": {"sweep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}},
    "head": {"component": "lm_head_output"}
  },
  "interventions": {
    "corrupt": {"noise": {"site": "embed", "pos": {"scope": {"column": "pairs.entity"}, "index": ":"},
                          "ops": [{"op": "noise", "scale": 0.5, "seed": 1}]}},
    "scored": {"p": {"site": "head", "pos": -1, "ops": [{"op": "token_prob", "token": "pairs.base_answer"}]}}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": [
      {"at_entity": {"site": "every_layer", "pos": {"scope": {"column": "pairs.entity"}, "index": -1}},
       "at_last": {"site": "every_layer", "pos": -1}}, "scored"]},
    "corrupted": {"kind": "forward", "input": "pairs.input", "interventions": ["corrupt", "scored"]},
    "restored_entity": {"kind": "forward", "input": "pairs.input", "interventions": ["corrupt",
      {"restore": {"site": "one_layer", "pos": {"scope": {"column": "pairs.entity"}, "index": -1},
                   "ops": [{"op": "swap", "with": "clean.at_entity"}]}}, "scored"]},
    "restored_last": {"kind": "forward", "input": "pairs.input", "interventions": ["corrupt",
      {"restore": {"site": "one_layer", "pos": -1, "ops": [{"op": "swap", "with": "clean.at_last"}]}}, "scored"]},
    "p_clean": {"kind": "op", "op": "mean", "of": "clean.p"},
    "p_corrupted": {"kind": "op", "op": "mean", "of": "corrupted.p"},
    "p_entity": {"kind": "op", "op": "mean", "of": "restored_entity.p"},
    "p_last": {"kind": "op", "op": "mean", "of": "restored_last.p"}
  },
  "saves": ["p_clean", "p_corrupted", "p_entity", "p_last", "clean.p", "corrupted.p"]
}
```

```text
$ nnspec table out/corrupt_restore p_clean p_corrupted
p_clean              p_corrupted
0.12413130700588226  0.00924452394247055
$ nnspec table out/corrupt_restore p_entity p_last
at             p_entity              p_last
{"layers":0}   0.037955328822135925  0.010240381583571434
{"layers":4}   0.03968358039855957   0.008889460936188698
{"layers":7}   0.04144216701388359   0.00997314602136612
{"layers":9}   0.016302242875099182  0.06157633289694786
{"layers":11}  0.00924452394247055   0.12413130700588226
```

- **Pick the noise scale by the first pair of numbers**, before reading any restore: at 0.5 the right city's
  probability falls from 0.124 to 0.009.
- **A restore is read against the corruption it undoes**, so every step corrupts through the one bundle:
  `noise` draws from `seed`, the row and the position alone, and so draws alike in each.
- **Restoring layer 11 at the last token gives `p_clean` to the bit**, the positive control. At the landmark it
  gives `p_corrupted` to the bit, since no later block carries it on.
- **Where a restore lands between corrupted and clean is its effect**: the landmark's last token restores 0.04
  of the clean 0.12 through layer 7, and the last token takes over from layer 9.

## Patching into a generation

Use it when the answer is several tokens, or to see what the patched model goes on to say. Read the
counterfactual's value in a forward and write it in the base's `generate` step at `"frame": 0`, the prefill:
every later token follows from the patched state through the cache ([generation](generation.md)).
`generation_patch.json` is `interchange.json` on `HuggingFaceTB/SmolLM2-135M-Instruct` at revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, with `resid` at layer 8 and these keys for the steps after
`counterfactual` and the `saves`:

```json
{
  "steps": {
    "plain": {"kind": "generate", "input": "pairs.input", "max_new_tokens": 6, "do_sample": false},
    "null": {"kind": "generate", "input": "pairs.input", "max_new_tokens": 6, "do_sample": false, "interventions": {
      "patch": {"site": "resid", "pos": {"frame": 0, "scope": {"column": "pairs.day"}, "index": -1},
                "ops": [{"op": "swap", "with": "clean.v_self"}]}}},
    "patched": {"kind": "generate", "input": "pairs.input", "max_new_tokens": 6, "do_sample": false, "interventions": {
      "patch": {"site": "resid", "pos": {"frame": 0, "scope": {"column": "pairs.day"}, "index": -1},
                "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
      "conf": {"site": "head", "pos": {"frame": "1:", "scope": {"column": "pairs.cf_answer", "before": true}},
               "ops": [{"op": "token_prob", "token": "pairs.cf_answer"}]}}},
    "differs": {"kind": "op", "op": "ne", "of": "null", "with": "plain"},
    "null_gap": {"kind": "op", "op": "sum", "of": "differs", "over": ["rows", "new"]},
    "plain_text": {"kind": "op", "op": "detokenize", "of": "plain"},
    "patched_text": {"kind": "op", "op": "detokenize", "of": "patched"},
    "says_cf": {"kind": "op", "op": "contains", "of": "patched_text", "text": {"column": "pairs.cf_answer", "strip": true}}
  },
  "saves": ["null_gap", "plain_text", "patched_text", "says_cf", "patched.conf"]
}
```

```text
$ nnspec table out/generation_patch plain_text patched_text says_cf
example_id  row  input                               plain_text                         patched_text                      cf_answer   says_cf
mon-thu     0    If today is Monday, tomorrow is      Tuesday.\n\nThe key                Friday.\n\nThe key                Friday     1.0
tue-fri     1    If today is Tuesday, tomorrow is     Wednesday, and today is Thursday   Monday, and today is Tuesday      Saturday   0.0
wed-sat     2    If today is Wednesday, tomorrow is   Friday, and today is today         Sunday, and today is Monday       Sunday     1.0
...
```

- **The null is `null_gap` 0.0**: the self-swap generates the plain run's tokens on every row, by `ne` of the
  two generations' ids summed over `rows` and `new`.
- **Five rows of seven say the counterfactual's day**, and the text after it follows; a forward scores the
  next token alone. Read `plain_text` first: this small model names the right day on four rows of seven.
- **`conf` reads the call that chose each token of the counterfactual's answer** (`before`): 0.15 to 0.34 on
  the five rows that say it, at call 0, and `alignment_missing` on the two that never do
  ([rows a scope never reached](positions.md#rows-a-scope-never-reached)).
