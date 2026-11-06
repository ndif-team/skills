# Pairs, scopes and maps

The recipes that change where a patch lands or how it is scored: pairs of different lengths, windows, corrupt
and restore, whole-distribution metrics, two experiments against one clean run, and patching into a generation.
Each runs on the CPU in seconds, on `openai-community/gpt2` except the generation, on
`HuggingFaceTB/SmolLM2-135M-Instruct`. The weekday rows are those of `documents/interchange.json` in
[SKILL.md](../SKILL.md).

## What breaks

- **A counted `pos` on prompts of different lengths passes only where every token between it and the end it
  counts from matches.** `-1` and `0` always pass; `-k` needs the last k−1 tokens alike, `k` the first k. The
  landmark pairs below end ` is in the city of`, five tokens: `-6` pairs ` Tower` with ` Gate` and runs, and
  `-7` is refused, naming the two tokens:

  ```text
  clab: patched.patch: operand 'counterfactual.v_cf' was read on pairs.counterfactual, row 0 of which is 9 tokens; pairs.input has 10, so pairs.counterfactual's position 2 ('enburg') pairs with pairs.input's position 3 ('el'); use a scope to name the token in each row
  ```

  Prompts of one length are never checked, whatever tokens share an index. Past the boundary, scope each side.
- **A whole-word scope pairs only where every row's word is the same number of tokens**, and is refused
  naming each row's count ([positions](../../clab/references/positions.md#pairing-with-another-forward)).
- **An open `pos` (`{"index": ":"}`) from another forward pairs with nothing**, even on prompts of one length.
  Write the width: `{"index": "0:7"}` on 7-token prompts.
- **A restore is read against the corruption it undoes, so both steps must corrupt alike.** Declare the noise
  once in a root bundle and pull it into every step that corrupts; its draw depends only on `seed`, the row and
  the position.
- **A null's `kl` is exactly 0; its `cosine` is 1 only to rounding** (1.0000001 here). Compare cosines with a
  tolerance, logits and divergences to the bit.

## Pairs of different lengths

Use a scope on each side when the prompts differ in length or the word is several tokens. The landmarks below
are one to four tokens (`Eiffel Tower` is four, `Vatican` one), and GPT-2's clean run names the right city on
all four rows.

```json
{
  "description": "Pairs of different lengths: each landmark's last token, located by its own text on each side ('Eiffel Tower' is four tokens, 'Vatican' one), at the residual stream of layers 0, 4 and 8. The null swaps the base's own value back at the same scope.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"example_id": "eiffel-brandenburg", "input": "The Eiffel Tower is in the city of", "entity": "Eiffel Tower", "base_answer": " Paris",
     "counterfactual": "The Brandenburg Gate is in the city of", "cf_entity": "Brandenburg Gate", "cf_answer": " Berlin"},
    {"example_id": "brandenburg-vatican", "input": "The Brandenburg Gate is in the city of", "entity": "Brandenburg Gate", "base_answer": " Berlin",
     "counterfactual": "The Vatican is in the city of", "cf_entity": "Vatican", "cf_answer": " Rome"},
    {"example_id": "vatican-buckingham", "input": "The Vatican is in the city of", "entity": "Vatican", "base_answer": " Rome",
     "counterfactual": "Buckingham Palace is in the city of", "cf_entity": "Buckingham Palace", "cf_answer": " London"},
    {"example_id": "buckingham-louvre", "input": "Buckingham Palace is in the city of", "entity": "Buckingham Palace", "base_answer": " London",
     "counterfactual": "The Louvre is in the city of", "cf_entity": "Louvre", "cf_answer": " Paris"}
  ]}},
  "sites": {"resid": {"component": "layer_output", "layers": {"sweep": [0, 4, 8]}}, "head": {"component": "lm_head_output"}},
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": {
      "v_self": {"site": "resid", "pos": {"scope": {"column": "pairs.entity"}, "index": -1}},
      "logits": {"site": "head", "pos": -1}}},
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {
      "v_cf": {"site": "resid", "pos": {"scope": {"column": "pairs.cf_entity"}, "index": -1}}}},
    "null": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "resid", "pos": {"scope": {"column": "pairs.entity"}, "index": -1},
                "ops": [{"op": "swap", "with": "clean.v_self"}]},
      "logits": {"site": "head", "pos": -1}}},
    "patched": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "resid", "pos": {"scope": {"column": "pairs.entity"}, "index": -1},
                "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
      "logits": {"site": "head", "pos": -1}}},
    "clean_top": {"kind": "op", "op": "match", "of": "clean.logits", "expected": "pairs.base_answer"},
    "clean_ld": {"kind": "op", "op": "logit_diff", "of": "clean.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "null_ld": {"kind": "op", "op": "logit_diff", "of": "null.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "ld": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "iia": {"kind": "op", "op": "match", "of": "patched.logits", "expected": "pairs.cf_answer"}
  },
  "saves": ["clean_top", "clean_ld", "null_ld", "ld", "iia"]
}
```

`clab show out/entity_scope` prints where the reads and the write landed, a different position on each row:

```text
  clean.v_self         all 3 points  [4], [3], [1], [3]
  counterfactual.v_cf  all 3 points  [3], [1], [3], [2]
  patched.patch        all 3 points  [4], [3], [1], [3]
```

`clab table out/entity_scope null_ld ld iia`, trimmed to its value columns:

```text
at            example_id           null_ld              ld                   iia
{"layers":0}  eiffel-brandenburg   -1.0906295776367188  -1.1459503173828125  0.0
{"layers":0}  brandenburg-vatican  -4.872459411621094   1.1072235107421875   0.0
{"layers":0}  vatican-buckingham   -4.72564697265625    2.2488937377929688   1.0
{"layers":0}  buckingham-louvre    -3.7453079223632812  3.057952880859375    1.0
...
{"layers":8}  vatican-buckingham   -4.72564697265625    -1.6878890991210938  0.0
{"layers":8}  buckingham-louvre    -3.7453079223632812  2.2940673828125      1.0
```

`null_ld` equals `clean_ld` on every row at every layer. The landmark's last token flips two rows at layer 0,
one at layers 4 and 8, and never the Eiffel Tower row: it does not carry the whole landmark on every row.

## Windows and whole prompts

A window is its positions patched together. This fragment is the `sites` and `steps` of
`documents/window.json`, whose `model` and `data` are those of `documents/interchange.json`. It patches the
day, the comma and ` tomorrow` (`-4` to `-2`) as one window and as three entries, and the whole 7-token prompt
as a window of static width:

```json
{
  "sites": {"resid": {"component": "layer_output", "layers": 4}, "head": {"component": "lm_head_output"}},
  "steps": {
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {
      "window": {"site": "resid", "pos": {"index": "-4:-1"}},
      "m4": {"site": "resid", "pos": -4}, "m3": {"site": "resid", "pos": -3}, "m2": {"site": "resid", "pos": -2},
      "prompt": {"site": "resid", "pos": {"index": "0:7"}},
      "logits": {"site": "head", "pos": -1}}},
    "windowed": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "resid", "pos": {"index": "-4:-1"}, "ops": [{"op": "swap", "with": "counterfactual.window"}]},
      "logits": {"site": "head", "pos": -1}}},
    "three": {"kind": "forward", "input": "pairs.input", "interventions": {
      "at_m4": {"site": "resid", "pos": -4, "ops": [{"op": "swap", "with": "counterfactual.m4"}]},
      "at_m3": {"site": "resid", "pos": -3, "ops": [{"op": "swap", "with": "counterfactual.m3"}]},
      "at_m2": {"site": "resid", "pos": -2, "ops": [{"op": "swap", "with": "counterfactual.m2"}]},
      "logits": {"site": "head", "pos": -1}}},
    "whole": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "resid", "pos": {"index": "0:7"}, "ops": [{"op": "swap", "with": "counterfactual.prompt"}]},
      "logits": {"site": "head", "pos": -1}}},
    "gap": {"kind": "op", "op": "sub", "of": "windowed.logits", "with": "three.logits"},
    "gap_abs": {"kind": "op", "op": "abs", "of": "gap"},
    "gap_max": {"kind": "op", "op": "max", "of": "gap_abs", "over": ["rows", "pos", "vocab_size"]},
    "whole_gap": {"kind": "op", "op": "sub", "of": "whole.logits", "with": "counterfactual.logits"},
    "whole_abs": {"kind": "op", "op": "abs", "of": "whole_gap"},
    "whole_max": {"kind": "op", "op": "max", "of": "whole_abs", "over": ["rows", "pos", "vocab_size"]},
    "window_ld": {"kind": "op", "op": "logit_diff", "of": "windowed.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "window_mean": {"kind": "op", "op": "mean", "of": "window_ld"}
  },
  "saves": ["gap_max", "whole_max", "window_ld", "window_mean"]
}
```

```text
$ clab table out/window gap_max whole_max window_mean
gap_max  whole_max  window_mean
0.0      0.0        1.7342965602874756
```

- **`gap_max` is 0.0**: the window and the three entries give the same logits to the bit.
- **`whole_max` is 0.0**: the whole prompt patched at layer 4 gives the counterfactual run's logits to the bit.
  It is the positive control of a window, and it holds only where both prompts tokenize to one length.

## Corrupt and restore

The causal-tracing form patches clean states into a corrupted run of the same prompt, so it needs no
counterfactual. Corrupt the landmark's token embeddings with noise from a root bundle, then restore the clean
residual stream at one layer at a time, at the landmark's last token and at the prompt's last token.

```json
{
  "description": "Corrupt and restore (causal tracing) on GPT-2: noise on the landmark's token embeddings, drawn alike in every step that pulls the bundle in. The clean residual stream is then restored one layer at a time, at the landmark's last token and at the prompt's last token. p is the probability of the right city.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"prompts": {"rows": [
    {"example_id": "eiffel", "input": "The Eiffel Tower is in the city of", "entity": "Eiffel Tower", "answer": " Paris"},
    {"example_id": "brandenburg", "input": "The Brandenburg Gate is in the city of", "entity": "Brandenburg Gate", "answer": " Berlin"},
    {"example_id": "vatican", "input": "The Vatican is in the city of", "entity": "Vatican", "answer": " Rome"},
    {"example_id": "buckingham", "input": "Buckingham Palace is in the city of", "entity": "Buckingham Palace", "answer": " London"},
    {"example_id": "louvre", "input": "The Louvre is in the city of", "entity": "Louvre", "answer": " Paris"}
  ]}},
  "sites": {
    "embed": {"component": "token_embeddings"},
    "every_layer": {"component": "layer_output", "layers": ":"},
    "one_layer": {"component": "layer_output", "layers": {"sweep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}},
    "head": {"component": "lm_head_output"}
  },
  "interventions": {
    "corrupt": {"noise": {"site": "embed", "pos": {"scope": {"column": "prompts.entity"}, "index": ":"},
                          "ops": [{"op": "noise", "scale": 0.5, "seed": 1}]}},
    "scored": {"p": {"site": "head", "pos": -1, "ops": [{"op": "token_prob", "token": "prompts.answer"}]}}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "prompts.input", "interventions": [
      {"at_entity": {"site": "every_layer", "pos": {"scope": {"column": "prompts.entity"}, "index": -1}},
       "at_last": {"site": "every_layer", "pos": -1}}, "scored"]},
    "corrupted": {"kind": "forward", "input": "prompts.input", "interventions": ["corrupt", "scored"]},
    "restored_entity": {"kind": "forward", "input": "prompts.input", "interventions": ["corrupt",
      {"restore": {"site": "one_layer", "pos": {"scope": {"column": "prompts.entity"}, "index": -1},
                   "ops": [{"op": "swap", "with": "clean.at_entity"}]}}, "scored"]},
    "restored_last": {"kind": "forward", "input": "prompts.input", "interventions": ["corrupt",
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
$ clab table out/corrupt_restore p_clean p_corrupted
p_clean              p_corrupted
0.16293950378894806  0.008386699482798576

$ clab table out/corrupt_restore p_entity p_last
at             p_entity              p_last
{"layers":0}   0.06097676604986191   0.009248701855540276
{"layers":4}   0.08421291410923004   0.008264116011559963
{"layers":6}   0.09804319590330124   0.009586778469383717
{"layers":9}   0.021505799144506454  0.07951979339122772
{"layers":11}  0.008386699482798576  0.16293950378894806
```

- **The corruption works**: noise at scale 0.5 takes the right city's probability from 0.163 to 0.0084. Pick
  the scale by this pair of numbers before reading any restore.
- **Restoring layer 11 at the last token gives `p_clean` to the bit**, the positive control. At the landmark
  it gives `p_corrupted` to the bit, since no later block carries it on: the noise was drawn alike in both.
- **The landmark's last token restores 0.061 to 0.098 of the clean 0.163 through layer 7**, and the last
  token takes over from layer 9. The effect at a layer is where its score lands between corrupted and clean.

## Whole-distribution effects

`logit_diff` scores two tokens. `kl` and `js` score how far the whole next-token distribution moved, and
`cosine` how far the residual stream moved. This fragment is the `sites` and `steps` of
`documents/distribution.json`, whose `model` and `data` are those of `documents/interchange.json`:

```json
{
  "sites": {"resid": {"component": "layer_output", "layers": 4},
            "stream": {"component": "layer_output", "layers": [2, 6, 10]},
            "head": {"component": "lm_head_output"}},
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": {
      "v_self": {"site": "resid", "pos": {"scope": {"column": "pairs.day"}, "index": -1}},
      "resid": {"site": "stream", "pos": {"index": "2:5"}},
      "logits": {"site": "head", "pos": -1}}},
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {
      "v_cf": {"site": "resid", "pos": {"scope": {"column": "pairs.cf_day"}, "index": -1}}}},
    "null": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "resid", "pos": {"scope": {"column": "pairs.day"}, "index": -1},
                "ops": [{"op": "swap", "with": "clean.v_self"}]},
      "resid": {"site": "stream", "pos": {"index": "2:5"}},
      "logits": {"site": "head", "pos": -1}}},
    "patched": {"kind": "forward", "input": "pairs.input", "interventions": {
      "patch": {"site": "resid", "pos": {"scope": {"column": "pairs.day"}, "index": -1},
                "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
      "resid": {"site": "stream", "pos": {"index": "2:5"}},
      "logits": {"site": "head", "pos": -1}}},
    "kl": {"kind": "op", "op": "kl", "of": "patched.logits", "against": "clean.logits"},
    "js": {"kind": "op", "op": "js", "of": "patched.logits", "against": "clean.logits"},
    "kl_null": {"kind": "op", "op": "kl", "of": "null.logits", "against": "clean.logits"},
    "cos": {"kind": "op", "op": "cosine", "of": "patched.resid", "with": "clean.resid"},
    "cos_null": {"kind": "op", "op": "cosine", "of": "null.resid", "with": "clean.resid"},
    "cos_mean": {"kind": "op", "op": "mean", "of": "cos", "over": ["rows"]},
    "cos_null_mean": {"kind": "op", "op": "mean", "of": "cos_null", "over": ["rows"]}
  },
  "saves": ["kl", "js", "kl_null", "cos_mean", "cos_null_mean"]
}
```

```text
$ clab table out/distribution kl js kl_null
example_id  row  pos  pos_token  input                               kl                   js                   kl_null
mon-thu     0    6    Ġis        If today is Monday, tomorrow is     0.24973368644714355  0.06207166612148285  0.0
...
sat-tue     5    6    Ġis        If today is Saturday, tomorrow is   0.9206199049949646   0.19037958979606628  0.0

$ clab table out/distribution cos_mean cos_null_mean
layer  pos  cos_mean            cos_null_mean
2      1    1.0000001192092896  1.0000001192092896
6      0    0.9999998807907104  0.9999998807907104
6      1    0.923564076423645   1.0
10     1    0.9769387245178223  1.0
```

- **`kl` is KL(patched ‖ clean) in nats**, 0.25 to 0.92 here; `js` is symmetric and at most log 2. The null's
  `kl` is exactly 0.0 on every row.
- **A mean over rows has no rows, so its `pos` is the index along the window**: `pos` 1 is the day (position
  3), `pos` 0 the token before it.
- **The cosine is 1 wherever the write cannot reach**: layer 2 is before the write at layer 4, and `pos` 0 is
  before the day in the sequence. It drops at the day after layer 4 (0.92 at layer 6).

## Two experiments against one baseline

Declare each experiment as a bundle at the root and run each in its own step, its metric named apart, against
one clean run. This fragment is the `interventions` and `steps` of `documents/two_experiments.json`, whose
`model`, `data` and `sites` are those of `documents/interchange.json`; the second experiment is a mean ablation
(the `clab-ablation` skill owns it).

```json
{
  "interventions": {
    "patching": {"patch": {"site": "resid", "pos": {"scope": {"column": "pairs.day"}, "index": -1},
                           "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]},
                 "logits": {"site": "head", "pos": -1}},
    "ablation": {"ablate": {"site": "resid", "pos": {"scope": {"column": "pairs.day"}, "index": -1},
                            "ops": [{"op": "swap", "with": "mean"}]},
                 "logits": {"site": "head", "pos": -1}}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": {
      "acts": {"site": "resid", "pos": {"scope": {"column": "pairs.day"}, "index": -1}},
      "logits": {"site": "head", "pos": -1}}},
    "mean": {"kind": "op", "op": "mean", "of": "clean.acts"},
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {
      "v_cf": {"site": "resid", "pos": {"scope": {"column": "pairs.cf_day"}, "index": -1}}}},
    "patched": {"kind": "forward", "input": "pairs.input", "interventions": "patching"},
    "ablated": {"kind": "forward", "input": "pairs.input", "interventions": "ablation"},
    "clean_ld": {"kind": "op", "op": "logit_diff", "of": "clean.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "patching_ld": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "ablation_ld": {"kind": "op", "op": "logit_diff", "of": "ablated.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "ablation_top": {"kind": "op", "op": "match", "of": "ablated.logits", "expected": "pairs.base_answer"}
  },
  "saves": ["clean_ld", "patching_ld", "ablation_ld", "ablation_top"]
}
```

```text
$ clab table out/two_experiments clean_ld patching_ld ablation_ld ablation_top
example_id  ...  clean_ld             patching_ld         ablation_ld           ablation_top
mon-thu     ...  -1.043609619140625   1.0651702880859375  0.0742034912109375    0.0
tue-fri     ...  -1.4108200073242188  2.1431808471679688  0.6134719848632812    0.0
...
sun-wed     ...  -1.568878173828125   1.0600814819335938  -0.46514892578125     1.0
```

- **One table joins both experiments row by row**, since their metrics are steps of one dataset named apart;
  `patching_ld` equals `ld` of `documents/interchange.json` to the bit, since a bundle is its entries.
- **The two answer different questions**: the patch moves every row toward the counterfactual's day; removing
  the day's value leaves the base answer on top on one row of seven.

## Patching into a generation

Use it when the answer is several tokens, or to see what the patched model goes on to say. Read the
counterfactual's value in a forward and write it in the base's `generate` step at `"frame": 0`, the prefill;
every later token follows from the patched state through the cache. Frames are the `clab` skill's
[generation reference](../../clab/references/generate.md#frames); scoring what a generation says is the
`clab-steering-and-generation` skill's.

```json
{
  "description": "Patching into a generation on SmolLM2-135M-Instruct: the counterfactual's residual stream leaving block 8 at its landmark's last token, swapped into the base prompt's prefill (frame 0) at its own landmark's last token, then six greedy tokens. The null swaps the base's own value and must generate the plain run's tokens; conf reads the decode calls that say the counterfactual's city.",
  "model": {"key": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"example_id": "eiffel-colosseum", "input": "The Eiffel Tower is in the city of", "entity": "Eiffel Tower", "base_answer": " Paris",
     "counterfactual": "The Colosseum is in the city of", "cf_entity": "Colosseum", "cf_answer": " Rome"},
    {"example_id": "colosseum-liberty", "input": "The Colosseum is in the city of", "entity": "Colosseum", "base_answer": " Rome",
     "counterfactual": "The Statue of Liberty is in the city of", "cf_entity": "Statue of Liberty", "cf_answer": " New York"},
    {"example_id": "kremlin-acropolis", "input": "The Kremlin is in the city of", "entity": "Kremlin", "base_answer": " Moscow",
     "counterfactual": "The Acropolis is in the city of", "cf_entity": "Acropolis", "cf_answer": " Athens"},
    {"example_id": "liberty-kremlin", "input": "The Statue of Liberty is in the city of", "entity": "Statue of Liberty", "base_answer": " New York",
     "counterfactual": "The Kremlin is in the city of", "cf_entity": "Kremlin", "cf_answer": " Moscow"},
    {"example_id": "familia-golden", "input": "The Sagrada Familia is in the city of", "entity": "Sagrada Familia", "base_answer": " Barcelona",
     "counterfactual": "The Golden Gate Bridge is in the city of", "cf_entity": "Golden Gate Bridge", "cf_answer": " San Francisco"},
    {"example_id": "acropolis-sophia", "input": "The Acropolis is in the city of", "entity": "Acropolis", "base_answer": " Athens",
     "counterfactual": "The Hagia Sophia is in the city of", "cf_entity": "Hagia Sophia", "cf_answer": " Istanbul"},
    {"example_id": "golden-buckingham", "input": "The Golden Gate Bridge is in the city of", "entity": "Golden Gate Bridge", "base_answer": " San Francisco",
     "counterfactual": "Buckingham Palace is in the city of", "cf_entity": "Buckingham Palace", "cf_answer": " London"},
    {"example_id": "sophia-eiffel", "input": "The Hagia Sophia is in the city of", "entity": "Hagia Sophia", "base_answer": " Istanbul",
     "counterfactual": "The Eiffel Tower is in the city of", "cf_entity": "Eiffel Tower", "cf_answer": " Paris"}
  ]}},
  "sites": {"resid": {"component": "layer_output", "layers": 8}, "head": {"component": "lm_head_output"}},
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": {
      "v_self": {"site": "resid", "pos": {"scope": {"column": "pairs.entity"}, "index": -1}}}},
    "counterfactual": {"kind": "forward", "input": "pairs.counterfactual", "interventions": {
      "v_cf": {"site": "resid", "pos": {"scope": {"column": "pairs.cf_entity"}, "index": -1}}}},
    "plain": {"kind": "generate", "input": "pairs.input", "max_new_tokens": 6, "do_sample": false},
    "null": {"kind": "generate", "input": "pairs.input", "max_new_tokens": 6, "do_sample": false, "interventions": {
      "patch": {"site": "resid", "pos": {"frame": 0, "scope": {"column": "pairs.entity"}, "index": -1},
                "ops": [{"op": "swap", "with": "clean.v_self"}]}}},
    "patched": {"kind": "generate", "input": "pairs.input", "max_new_tokens": 6, "do_sample": false, "interventions": {
      "patch": {"site": "resid", "pos": {"frame": 0, "scope": {"column": "pairs.entity"}, "index": -1},
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
$ clab run documents/generation_patch.json
$ clab table out/generation_patch plain_text patched_text says_cf
example_id         row  ...  plain_text                   patched_text                 cf_answer       says_cf
eiffel-colosseum   0    ...  Paris, France.\n\n           Paris, France.\n\n           Rome           0.0
colosseum-liberty  1    ...  Rome, Italy. It is           New York.\n\nThe             New York       1.0
liberty-kremlin    3    ...  New York.\n\nThe             Moscow, Russia.\n\n          Moscow         1.0
sophia-eiffel      7    ...  Istanbul, Turkey.\n\n        Athens, Greece.\n\n          Paris          0.0
```

- **`null_gap` is 0.0** (`clab table out/generation_patch null_gap`): the null generates the plain run's
  tokens on every row, by `ne` of the two generations' ids summed over `rows` and `new`.
- **Four rows of eight say the counterfactual's city, and its country comes with it.** The forward recipes
  score the next token alone.
- **`conf` reads the decode calls that say the city**: 0.41 to 0.73 at the call that chose its first token,
  0.65 for ` York` after ` New`, and `alignment_missing` on a row that never says it.
