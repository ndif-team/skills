# Steering

Steering adds a direction to an activation and measures what moves. A steering curve is evidence when four
things hold: the dose is a length set against the activations, the unsteered point equals the clean run, a
random direction of the same length does less, and the cost is measured beside the effect.

## What breaks

- **A dose as a bare number means nothing** without the norm it is added to. Scale a unit direction by the
  `median` norm at the positions the write lands on, and sweep a fraction of that. A `mean` over prompt
  tokens is set by the first token's outlier ([the numbers](#the-norm-a-dose-is-set-against)).
- **A random direction of another length is no control.** Give it the steering vector's length, and draw it
  at several seeds.
- **A count of changed tokens is no evidence of a direction**: a random direction of the same length changes
  as many. Score the behaviour you steer.
- **A top dose collapses the text, and a word score falls with it.** Read two replies per dose before
  trusting a count, and report the doses below the collapse.
- **`renormalize` listed before the `add` is a no-op**: it rescales the clean activation to its own norm.
  List it after.
- **A direction that is rounding steers nowhere in particular.** Save the `norm` of the mean difference beside
  `typical`; `normalize` makes a unit vector of any difference, however small, and leaves an exactly zero one
  zero.

## A direction at a dose

The direction is the mean activation over one prompt set less the mean over another, made unit length, and the
dose a fraction of the steered prompts' own median norm at the write. Saved as `documents/steer.json`:

```json
{
  "description": "Steer GPT-2 with a sentiment direction at block 6's output, last token: the mean over positive sentences less the mean over negative ones, at a dose that is a fraction of the prompts' own median norm there. effect is the logit of ' great' less ' terrible', drift the KL from the unsteered distribution; dose 0 is the null.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {
    "tone": {"rows": [
      {"pos": "I love this movie.", "neg": "I hate this movie."},
      {"pos": "What a great day.", "neg": "What a horrible day."},
      {"pos": "The food was delicious.", "neg": "The food was disgusting."},
      {"pos": "This is the best book.", "neg": "This is the worst book."}]},
    "prompts": {"rows": [{"input": "The food at the restaurant was"}, {"input": "My trip to the beach was"},
                         {"input": "The new phone I bought is"}]}},
  "sites": {"resid": {"component": "layer_output", "layers": 6}, "head": {"component": "lm_head_output"}},
  "steps": {
    "pos": {"kind": "forward", "input": "tone.pos", "interventions": {"r": {"site": "resid", "pos": -1}}},
    "neg": {"kind": "forward", "input": "tone.neg", "interventions": {"r": {"site": "resid", "pos": -1}}},
    "pos_mean": {"kind": "op", "op": "mean", "of": "pos.r"},
    "neg_mean": {"kind": "op", "op": "mean", "of": "neg.r"},
    "difference": {"kind": "op", "op": "sub", "of": "pos_mean", "with": "neg_mean"},
    "direction": {"kind": "op", "op": "normalize", "of": "difference"},
    "base": {"kind": "forward", "input": "prompts.input", "interventions": {
      "r": {"site": "resid", "pos": -1}, "logits": {"site": "head", "pos": -1}}},
    "norms": {"kind": "op", "op": "norm", "of": "base.r"},
    "typical": {"kind": "op", "op": "median", "of": "norms"},
    "steer": {"kind": "op", "op": "scale", "of": "direction", "factor": "typical"},
    "steered": {"kind": "forward", "input": "prompts.input", "interventions": {
      "add": {"site": "resid", "pos": -1,
              "ops": [{"op": "add", "with": "steer", "scale": {"sweep": [0.0, 0.25, 0.5, 1.0], "label": "dose"}}]},
      "logits": {"site": "head", "pos": -1}}},
    "ld": {"kind": "op", "op": "logit_diff", "of": "steered.logits", "a": {"text": " great"}, "b": {"text": " terrible"}},
    "effect": {"kind": "op", "op": "mean", "of": "ld"},
    "kl": {"kind": "op", "op": "kl", "of": "steered.logits", "against": "base.logits"},
    "drift": {"kind": "op", "op": "mean", "of": "kl"}
  },
  "saves": ["typical", "effect", "drift"]
}
```

```text
$ nnspec table out/steer effect drift
at             effect              drift
{"dose":0.0}   2.5226821899414062  0.0
{"dose":0.25}  4.361429691314697   0.077471524477005
{"dose":0.5}   5.752100467681885   0.2544526755809784
{"dose":1.0}   7.399055480957031   0.7366573214530945
```

- **`steer` is the direction at the median activation's length** (`typical`, 89.2), so the `add`'s `scale` is
  the dose: 1.0 adds a vector as long as the median activation.
- **Dose 0 is the null**: the unsteered effect, and a drift of exactly 0.0.
- **`effect` is the target and `drift` the cost**, the KL from the unsteered next-token distribution. From
  dose 0.5 to 1.0 the effect rises by 1.6 and the drift triples.

## A random direction of the same length

The control is a Gaussian draw given the steering vector's length, added at the same doses. Four steps after
`steer` make it, `noise` on a value with no rows being one draw from its `seed`:

```json
{
  "zeros": {"kind": "op", "op": "mul", "of": "difference", "with": 0.0},
  "draw": {"kind": "op", "op": "noise", "of": "zeros", "scale": 1.0, "seed": {"sweep": [7, 8, 9], "label": "seed"}},
  "random_unit": {"kind": "op", "op": "normalize", "of": "draw"},
  "random": {"kind": "op", "op": "scale", "of": "random_unit", "factor": "typical"}
}
```

and the `add`'s operand sweeps over the two vectors by name, under a label that crosses the dose:
`{"op": "add", "with": {"sweep": ["steer", "random"], "label": "vector"}, "scale": {"sweep": [0.0, 0.25, 0.5,
1.0], "label": "dose"}}`. At dose 0.5:

```text
$ nnspec table out/steer_seeds effect drift
at                                        effect              drift
{"vector":"steer","dose":0.5}             5.752100467681885   0.2544526755809784
{"seed":7,"vector":"random","dose":0.5}   2.6558353900909424  0.11131032556295395
{"seed":8,"vector":"random","dose":0.5}   2.4644572734832764  0.12984603643417358
{"seed":9,"vector":"random","dose":0.5}   1.7903417348861694  0.12214399129152298
```

- **The seeds give the spread a direction has to clear.** The random directions land between 1.79 and 2.66,
  within 0.8 of the unsteered 2.52; the direction's 5.75 is far outside.
- **The steered points do not depend on the seed**, so each runs once and is keyed without it; every dose-0
  point, of either vector at any seed, equals the unsteered run.
- **A control at a single dose** writes a plain `"scale": 1.0`, or sweeps under a label of its own. Under the
  steer's label, `{"sweep": [1.0], "label": "dose"}` is refused: one label is one axis over one list of values
  ([sweeps](sweeps.md#labels-zip-or-cross)).

## The norm a dose is set against

Read the site where the write lands, at the positions it lands on, and take the `median` of the norms: `norm`
reduces `hidden_size`, and `median` then reduces rows and positions. In `steer_generation.json`
[below](#steering-a-generation), these entries in the `clean` step read the prompt's last token and every
prompt token beside the generated ones:

```json
{"last": {"site": "resid", "pos": -1}, "prompt": {"site": "resid", "pos": {"index": ":"}}}
```

The median norm is 118.5 at the prompt's last token, 132.4 over the generated inputs (`typical`) and 147.1
over every prompt token, whose mean is 849.6 and whose max 26,120.3.

- **The prompt's first token is the outlier.** Position 0, `<|im_start|>`, has norm 26,120 on every row, where
  no other prompt token exceeds 191; a dose set from the mean over prompt tokens is six times too long.
- **The last prompt token and the generated tokens agree within 12%** here, so the two entries share one
  `typical`. Where they differ, scale each entry's vector by its own.
- **Check the difference against it.** `{"kind": "op", "op": "norm", "of": "difference"}` in `steer.json` is
  16.2 against a `typical` of 89.2, a fifth of the activation's length.

## Direction or length

A long vector changes the activation's norm as well as its direction. `renormalize`, listed after the `add`,
rescales each written vector to the norm the activation had before the step's writes, which leaves the
direction's share:

```json
{"add": {"site": "resid", "pos": -1,
         "ops": [{"op": "add", "with": "steer", "scale": {"sweep": [0.0, 0.25, 0.5, 1.0], "label": "dose"}},
                 {"op": "renormalize"}]}}
```

| dose | `effect`, `add` alone | `effect`, with `renormalize` | norm written on row 0, alone and renormalized |
|---|---|---|---|
| 0.25 | 4.361 | 4.346 | 94.0, 90.3 |
| 0.5 | 5.752 | 5.534 | 102.5, 90.3 |
| 1.0 | 7.399 | 5.893 | 129.8, 90.3 |

The gap between the two effects is the part that was length: 0.02 at dose 0.25 and 1.51 at 1.0. A write binds
what it wrote, so `norm` of `steered.add` is the norm after the write; the clean norm on row 0 is 90.3. With
the two ops in the other order, every effect and drift equals the plain steer's to the bit.

## Layers by doses

Sweep the site's `layers` under its own label and it crosses the dose: `"resid": {"component": "layer_output",
"layers": {"sweep": [3, 6, 9], "label": "layer"}}`. Every read of the site follows the layer, so each layer
steers along its own difference at its own `typical` (61.9, 89.2 and 171.6):

```text
$ nnspec table out/steer_grid effect drift
at                       effect              drift
{"layer":3,"dose":0.5}   4.236968994140625   0.13140931725502014
{"layer":3,"dose":1.0}   5.775815486907959   1.1085377931594849
{"layer":6,"dose":0.5}   5.752100467681885   0.2544526755809784
{"layer":9,"dose":0.25}  6.8181633949279785  0.14769648015499115
{"layer":9,"dose":0.5}   10.23906421661377   0.41239508986473083
```

Compare layers at matched drift, not matched dose: near 0.13 to 0.15 nats, layer 9 at dose 0.25 moves the
effect to 6.82 and layer 3 at dose 0.5 to 4.24. Every dose-0 point equals the unsteered 2.52 with a drift of
0.0.

## Steering a generation

In a `generate` step a write at `pos: -1` lands in the prefill only, so a steer kept on is two entries, the
prompt's last token and every decode call, swept under the same labels. Score the behaviour in the text, and
keep the random direction beside it. Saved as `documents/steer_generation.json`:

```json
{
  "description": "Steer a chat model's replies with a mood direction at block 12's output: the mean at the prompt's last token over cheerful user messages less the mean over gloomy ones, at a dose set against the median norm of the clean generation's activations. It is added at the prompt's last token and at every generated token; a random direction of the same length is the control. cheer asks whether a reply uses a cheerful word, correct whether it still names the capital, changed is the share of ids that differ from the clean generation's.",
  "model": {"key": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac", "dtype": "float32"},
  "data": {
    "qa": {"template": {"add_generation_prompt": true}, "rows": [
      {"example_id": "france", "chat": [{"role": "user", "content": "What is the capital of France?"}], "answer": " Paris"},
      {"example_id": "japan", "chat": [{"role": "user", "content": "What is the capital of Japan?"}], "answer": " Tokyo"},
      {"example_id": "italy", "chat": [{"role": "user", "content": "What is the capital of Italy?"}], "answer": " Rome"},
      {"example_id": "germany", "chat": [{"role": "user", "content": "What is the capital of Germany?"}], "answer": " Berlin"},
      {"example_id": "spain", "chat": [{"role": "user", "content": "What is the capital of Spain?"}], "answer": " Madrid"},
      {"example_id": "egypt", "chat": [{"role": "user", "content": "What is the capital of Egypt?"}], "answer": " Cairo"}]},
    "mood": {"template": {"add_generation_prompt": true}, "rows": [
      {"up": [{"role": "user", "content": "I love sunny days. Everything feels wonderful!"}],
       "down": [{"role": "user", "content": "I hate rainy days. Everything feels awful."}]},
      {"up": [{"role": "user", "content": "What a beautiful morning, I am so happy!"}],
       "down": [{"role": "user", "content": "What a miserable morning, I am so sad."}]},
      {"up": [{"role": "user", "content": "This is the best day of my life!"}],
       "down": [{"role": "user", "content": "This is the worst day of my life."}]},
      {"up": [{"role": "user", "content": "I am thrilled, the party was amazing!"}],
       "down": [{"role": "user", "content": "I am furious, the party was terrible."}]},
      {"up": [{"role": "user", "content": "My friends are lovely and I feel great!"}],
       "down": [{"role": "user", "content": "My friends are cruel and I feel horrible."}]},
      {"up": [{"role": "user", "content": "We won the game! I am overjoyed!"}],
       "down": [{"role": "user", "content": "We lost the game. I am devastated."}]}]}
  },
  "sites": {"resid": {"component": "layer_output", "layers": 12}},
  "steps": {
    "up": {"kind": "forward", "input": "mood.up", "interventions": {"r": {"site": "resid", "pos": -1}}},
    "down": {"kind": "forward", "input": "mood.down", "interventions": {"r": {"site": "resid", "pos": -1}}},
    "up_mean": {"kind": "op", "op": "mean", "of": "up.r"},
    "down_mean": {"kind": "op", "op": "mean", "of": "down.r"},
    "difference": {"kind": "op", "op": "sub", "of": "up_mean", "with": "down_mean"},
    "difference_norm": {"kind": "op", "op": "norm", "of": "difference"},
    "direction": {"kind": "op", "op": "normalize", "of": "difference"},
    "clean": {"kind": "generate", "input": "qa.chat", "max_new_tokens": 16, "min_new_tokens": 16, "do_sample": false, "interventions": {
      "r": {"site": "resid", "pos": {"frame": "1:"}}}},
    "norms": {"kind": "op", "op": "norm", "of": "clean.r"},
    "typical": {"kind": "op", "op": "median", "of": "norms"},
    "steer": {"kind": "op", "op": "scale", "of": "direction", "factor": "typical"},
    "zeros": {"kind": "op", "op": "mul", "of": "difference", "with": 0.0},
    "draw": {"kind": "op", "op": "noise", "of": "zeros", "scale": 1.0, "seed": 7},
    "random_unit": {"kind": "op", "op": "normalize", "of": "draw"},
    "random": {"kind": "op", "op": "scale", "of": "random_unit", "factor": "typical"},
    "steered": {"kind": "generate", "input": "qa.chat", "max_new_tokens": 16, "min_new_tokens": 16, "do_sample": false, "interventions": {
      "prompt": {"site": "resid", "pos": -1, "ops": [{"op": "add", "with": {"sweep": ["steer", "random"], "label": "vector"},
                                                      "scale": {"sweep": [0.0, 0.5, 1.0, 2.0], "label": "dose"}}]},
      "decode": {"site": "resid", "pos": {"frame": "1:"}, "ops": [{"op": "add", "with": {"sweep": ["steer", "random"], "label": "vector"},
                                                                    "scale": {"sweep": [0.0, 0.5, 1.0, 2.0], "label": "dose"}}]}}},
    "text": {"kind": "op", "op": "detokenize", "of": "steered"},
    "cheer_row": {"kind": "op", "op": "contains", "of": "text",
                  "text": {"text": ["glad", "happy", "delight", "beautiful", "wonderful", "love"], "lower": true}},
    "correct_row": {"kind": "op", "op": "contains", "of": "text", "text": {"column": "qa.answer", "strip": true}},
    "cheer": {"kind": "op", "op": "mean", "of": "cheer_row"},
    "correct": {"kind": "op", "op": "mean", "of": "correct_row"},
    "differs": {"kind": "op", "op": "ne", "of": "steered", "with": "clean"},
    "changed": {"kind": "op", "op": "mean", "of": "differs", "over": ["rows", "new"]}
  },
  "saves": ["text", "cheer", "correct", "changed", "typical", "difference_norm"]
}
```

```text
$ nnspec table out/steer_generation cheer correct changed
at                              cheer               correct             changed
{"vector":"steer","dose":0.0}   0.0                 1.0                 0.0
{"vector":"steer","dose":0.5}   0.8333333134651184  1.0                 0.4166666567325592
{"vector":"steer","dose":1.0}   0.8333333134651184  0.1666666716337204  0.9583333134651184
{"vector":"steer","dose":2.0}   0.0                 0.1666666716337204  0.9479166865348816
{"vector":"random","dose":0.0}  0.0                 1.0                 0.0
{"vector":"random","dose":0.5}  0.0                 0.8333333134651184  0.6041666865348816
{"vector":"random","dose":1.0}  0.1666666716337204  0.0                 0.9791666865348816
{"vector":"random","dose":2.0}  0.0                 0.0                 1.0
```

`nnspec table --csv out/steer_generation text` prints whole texts. The Japan row under the direction reads
"The capital of Japan is Tokyo. It's a city that is known for its" at dose 0.5, "I'm glad you're happy to see
you! I'm glad you're not" at 1.0 and "Japan is Japan, Japan, Japan, Japan, Japan, Japan, Japan," at 2.0; under
the random direction at 1.0, "Japan's a beautiful and beautiful, but the "Hokaidai" is".

- **The null is dose 0**: `changed`, the share of ids that differ from the clean generation's (`ne` of the two
  id values, whose reduction names `new` in `over`), is 0.0 for both vectors.
- **The band is dose 0.5.** Five of six replies call the city beautiful and all six still name it. At 1.0
  replies turn into "I'm glad you're here!" and "I am delighted to inform you", and one of six names the
  capital.
- **The top dose collapses the text**: at 2.0 every reply repeats its country, and `cheer` is 0 again.
- **The random direction moves the text and not the mood.** It changes as many ids as the direction, 98%
  against 96% at dose 1.0, and one row's "beautiful" is its 0.17.
- **The difference is short**: `difference_norm` is 8.46 against a `typical` of 132.37, one fifteenth.
- **`nnspec show` prints where each write landed**: `steered.prompt` at `[36]` and `steered.decode` at
  `[37,38,39,…,51]` on all 6 rows.

### The prompt alone, or every position

A generation steered at the prompt's last token alone is the `steered` step without its `decode` entry,
writing `"with": "steer"` (`prefill_only`). Its replies change and their mood does not:

| dose | `cheer`, both entries | `cheer`, prompt alone | `changed`, both entries | `changed`, prompt alone |
|---|---|---|---|---|
| 0.5 | 0.83 | 0.0 | 0.42 | 0.05 |
| 1.0 | 0.83 | 0.0 | 0.96 | 0.89 |
| 2.0 | 0.0 | 0.17 | 0.95 | 0.89 |

To steer every position, the prompt included, write once at `{"frame": ":"}`; that includes position 0, whose
norm is far above the dose. Frames and the calls a write lands at are in
[generation](generation.md#writes-during-generation).

### Counts and whole words

`mean` of a row score is the fraction of replies and `sum` the count. `contains` matches substrings, so
`delight` scores "I am delighted"; a scope matches whole words. This entry in the `steered` step reads the
call that chose a cheerful word's last token, and `count` of it over `pos` is 1 on a row that says one:

```json
{"cheer_word": {"site": "resid", "pos": {"frame": "1:", "index": -1,
  "scope": {"text": ["glad", "happy", "delight", "beautiful", "wonderful", "love"], "lower": true, "before": true}}}}
```

With `word_row` its `count` over `["pos"]`, and `cheer_count` and `word_count` the `sum`s of `cheer_row` and
`word_row`, the direction gives:

| dose | `cheer_count` | `word_count` |
|---|---|---|
| 0.5 | 5.0 | 5.0 |
| 1.0 | 5.0 | 3.0 |

- **List every form a whole-word read should count**: at dose 1.0 two replies say "delighted", which
  `contains` counts and the scope does not.
- **`count` is 0 on a row the read missed**, printed blank and flagged `alignment_missing`, so `sum` counts
  the replies. `before` with `index: -1` also reaches a word on the row's last generated token.
