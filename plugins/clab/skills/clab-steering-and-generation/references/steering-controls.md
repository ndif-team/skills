# Steering controls

A steering curve is evidence when four things hold. The dose is a length set against the activations. The
unsteered point equals the clean run. A random direction of the same length does less. The cost is measured
beside the effect. Each section is a complete document, or a fragment of one it names, with the numbers its
run gave on the CPU.

## The norm a dose is set against

Read the site where the write lands, at the positions it lands on, and take the `median` of the norms:

```json
{
  "description": "The norms a dose is set against at SmolLM2's layer 12: the prompt's last token, every generated input, and every prompt token, whose first, <|im_start|>, carries a norm two orders of magnitude above the rest. `median` is robust to it and `mean` is not.",
  "model": {"key": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac", "dtype": "float32"},
  "data": {"qa": {"template": {"add_generation_prompt": true}, "rows": [
    {"example_id": "france", "chat": [{"role": "user", "content": "What is the capital of France?"}]},
    {"example_id": "japan", "chat": [{"role": "user", "content": "What is the capital of Japan?"}]},
    {"example_id": "italy", "chat": [{"role": "user", "content": "What is the capital of Italy?"}]},
    {"example_id": "germany", "chat": [{"role": "user", "content": "What is the capital of Germany?"}]},
    {"example_id": "spain", "chat": [{"role": "user", "content": "What is the capital of Spain?"}]},
    {"example_id": "egypt", "chat": [{"role": "user", "content": "What is the capital of Egypt?"}]}]}},
  "sites": {"resid": {"component": "layer_output", "layers": 12}},
  "steps": {
    "clean": {"kind": "generate", "input": "qa.chat", "max_new_tokens": 16, "min_new_tokens": 16, "do_sample": false,
      "interventions": {"last": {"site": "resid", "pos": -1},
                        "generated": {"site": "resid", "pos": {"frame": "1:"}},
                        "prompt": {"site": "resid", "pos": {"index": ":"}}}},
    "last_norm": {"kind": "op", "op": "norm", "of": "clean.last"},
    "generated_norm": {"kind": "op", "op": "norm", "of": "clean.generated"},
    "prompt_norm": {"kind": "op", "op": "norm", "of": "clean.prompt"},
    "last_median": {"kind": "op", "op": "median", "of": "last_norm"},
    "generated_median": {"kind": "op", "op": "median", "of": "generated_norm"},
    "prompt_median": {"kind": "op", "op": "median", "of": "prompt_norm"},
    "prompt_mean": {"kind": "op", "op": "mean", "of": "prompt_norm"},
    "prompt_max": {"kind": "op", "op": "max", "of": "prompt_norm"}
  },
  "saves": ["last_median", "generated_median", "prompt_median", "prompt_mean", "prompt_max"]
}
```

`clab table out/site_norms last_median generated_median prompt_median prompt_mean prompt_max`:

```text
last_median         generated_median    prompt_median      prompt_mean        prompt_max
118.54289245605469  132.37283325195312  147.0801239013672  849.5513916015625  26120.314453125
```

- **`norm` reduces `hidden_size`** by default, one length per row and position; `median` then reduces rows
  and positions.
- **The prompt's first token is an outlier.** Position 0, `<|im_start|>`, has norm 26,120 on every row, and it
  pulls the mean over prompt tokens to 850 against a median of 147. A dose set from that mean is six times
  too long.
- **The last prompt token and the generated tokens agree within 12%** here (118.5 and 132.4). A write at
  both entries can share one `typical`; where they differ, scale each entry's vector by its own.
- **Check the difference against it.** Save `norm` of the mean difference before `normalize`. In bfloat16 an
  element rounds by about 2^-8 of its size, so a difference under about 1% of `typical` can be rounding, which
  `normalize` turns into a unit vector that points nowhere in particular. An exactly zero difference stays
  zero and steers nothing.

## Steering a forward: the effect against the drift

A forward steers one next-token distribution, so the cost can be measured as a divergence. Here the direction
is a sentiment difference on GPT-2, the target is a logit difference, and the cost is the `kl` from the
unsteered distribution:

```json
{
  "description": "Steering a forward, with its controls. The direction is the mean residual stream at GPT-2's layer 6, last token, over positive sentences less the mean over negative ones, made unit length. The dose is a fraction of the median norm of the steered prompts' own activations there. `effect` is the mean logit of ' great' less ' terrible' at the last token, `drift` the mean KL of the steered next-token distribution from the unsteered one. `renormed` adds the same vector and then puts back each activation's norm. `random` adds a random direction of the same length at the same doses, drawn at three seeds.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {
    "positive": {"rows": [{"input": "I love this movie. It is wonderful."}, {"input": "What a great day, I am so happy."},
                          {"input": "The food was delicious and the staff were lovely."}, {"input": "This is the best book I have ever read."},
                          {"input": "We had an amazing time at the party."}, {"input": "The concert was fantastic and I enjoyed it."}]},
    "negative": {"rows": [{"input": "I hate this movie. It is awful."}, {"input": "What a horrible day, I am so sad."},
                          {"input": "The food was disgusting and the staff were rude."}, {"input": "This is the worst book I have ever read."},
                          {"input": "We had a miserable time at the party."}, {"input": "The concert was terrible and I hated it."}]},
    "prompts": {"rows": [{"input": "The food at the restaurant was"}, {"input": "My trip to the beach was"},
                         {"input": "The new phone I bought is"}, {"input": "The concert last night was"}]}
  },
  "sites": {"resid": {"component": "layer_output", "layers": 6}, "head": {"component": "logits"}},
  "steps": {
    "pos_h": {"kind": "forward", "input": "positive.input", "interventions": {"r": {"site": "resid", "pos": -1}}},
    "neg_h": {"kind": "forward", "input": "negative.input", "interventions": {"r": {"site": "resid", "pos": -1}}},
    "pos_mean": {"kind": "op", "op": "mean", "of": "pos_h.r"},
    "neg_mean": {"kind": "op", "op": "mean", "of": "neg_h.r"},
    "difference": {"kind": "op", "op": "sub", "of": "pos_mean", "with": "neg_mean"},
    "direction": {"kind": "op", "op": "normalize", "of": "difference"},
    "base": {"kind": "forward", "input": "prompts.input",
             "interventions": {"r": {"site": "resid", "pos": -1}, "logits": {"site": "head", "pos": -1}}},
    "lengths": {"kind": "op", "op": "norm", "of": "base.r"},
    "typical": {"kind": "op", "op": "median", "of": "lengths"},
    "difference_norm": {"kind": "op", "op": "norm", "of": "difference"},
    "unit_dose": {"kind": "op", "op": "scale", "of": "direction", "factor": "typical"},
    "zero": {"kind": "op", "op": "scale", "of": "difference", "factor": 0.0},
    "draw": {"kind": "op", "op": "noise", "of": "zero", "scale": 1.0, "seed": {"sweep": [7, 8, 9], "label": "seed"}},
    "random_unit": {"kind": "op", "op": "normalize", "of": "draw"},
    "random_dose": {"kind": "op", "op": "scale", "of": "random_unit", "factor": "typical"},
    "steered": {"kind": "forward", "input": "prompts.input", "interventions": {
      "steer": {"site": "resid", "pos": -1,
                "ops": [{"op": "add", "with": "unit_dose", "scale": {"sweep": [0.0, 0.25, 0.5, 1.0], "label": "dose"}}]},
      "logits": {"site": "head", "pos": -1}}},
    "renormed": {"kind": "forward", "input": "prompts.input", "interventions": {
      "steer": {"site": "resid", "pos": -1,
                "ops": [{"op": "add", "with": "unit_dose", "scale": {"sweep": [0.0, 0.25, 0.5, 1.0], "label": "dose"}},
                        {"op": "renormalize"}]},
      "logits": {"site": "head", "pos": -1}}},
    "random": {"kind": "forward", "input": "prompts.input", "interventions": {
      "steer": {"site": "resid", "pos": -1,
                "ops": [{"op": "add", "with": "random_dose", "scale": {"sweep": [0.0, 0.25, 0.5, 1.0], "label": "dose"}}]},
      "logits": {"site": "head", "pos": -1}}},
    "ld": {"kind": "op", "op": "logit_diff", "of": "steered.logits", "a": {"text": " great"}, "b": {"text": " terrible"}},
    "effect": {"kind": "op", "op": "mean", "of": "ld"},
    "kl": {"kind": "op", "op": "kl", "of": "steered.logits", "against": "base.logits"},
    "drift": {"kind": "op", "op": "mean", "of": "kl"},
    "renorm_ld": {"kind": "op", "op": "logit_diff", "of": "renormed.logits", "a": {"text": " great"}, "b": {"text": " terrible"}},
    "renorm_effect": {"kind": "op", "op": "mean", "of": "renorm_ld"},
    "renorm_kl": {"kind": "op", "op": "kl", "of": "renormed.logits", "against": "base.logits"},
    "renorm_drift": {"kind": "op", "op": "mean", "of": "renorm_kl"},
    "random_ld": {"kind": "op", "op": "logit_diff", "of": "random.logits", "a": {"text": " great"}, "b": {"text": " terrible"}},
    "random_effect": {"kind": "op", "op": "mean", "of": "random_ld"},
    "random_kl": {"kind": "op", "op": "kl", "of": "random.logits", "against": "base.logits"},
    "random_drift": {"kind": "op", "op": "mean", "of": "random_kl"},
    "steered_norm": {"kind": "op", "op": "norm", "of": "steered.steer"},
    "renormed_norm": {"kind": "op", "op": "norm", "of": "renormed.steer"}
  },
  "saves": ["typical", "difference_norm", "effect", "drift", "renorm_effect", "renorm_drift", "random_effect", "random_drift",
            "steered_norm", "renormed_norm", "lengths"]
}
```

`clab table out/steer_forward typical difference_norm`, then `effect drift renorm_effect renorm_drift`, then
`random_effect random_drift`:

```text
typical            difference_norm
87.97744750976562  17.7718448638916

at             effect             drift                renorm_effect      renorm_drift
{"dose":0.0}   2.464017868041992  0.0                  2.464017868041992  0.0
{"dose":0.25}  4.283960342407227  0.0722854882478714   4.276096343994141  0.07235488295555115
{"dose":0.5}   5.699560165405273  0.23923064768314362  5.481157302856445  0.23641058802604675
{"dose":1.0}   7.118005752563477  0.7412769794464111   5.661102294921875  0.8064576983451843

at                      random_effect       random_drift
{"seed":7,"dose":0.5}   2.5987319946289062  0.09526371210813522
{"seed":8,"dose":0.5}   2.45916748046875    0.12449802458286285
{"seed":9,"dose":0.5}   1.7612781524658203  0.11385045200586319
{"seed":7,"dose":1.0}   3.0163230895996094  0.514231264591217
{"seed":8,"dose":1.0}   2.810312271118164   0.6051769256591797
{"seed":9,"dose":1.0}   0.8475971221923828  0.5078771114349365
```

- **Dose 0 is the null.** `drift` is exactly 0.0 and `effect` is the unsteered 2.464, for the direction, the
  renormalized write and every seed.
- **`effect` is the target and `drift` the cost.** The direction moves the effect by 3.2 at dose 0.5 for a
  drift of 0.24 nats. Past the band the drift grows faster than the effect: 0.74 nats at dose 1.0.
- **The seeds give the spread a direction has to clear.** At dose 0.5 the random directions land between 1.76
  and 2.60, within 0.7 of the clean 2.464, at a drift of 0.10 to 0.12. The direction's 5.70 is far outside.
- **The seed sweep crosses the dose sweep** because the two carry different labels, so `random` runs 12 points.
- **`renormalize` separates the direction from the longer vector.** Listed after the `add` in the entry's
  `ops`, it rescales each vector to the norm the activation had before the step's writes. At dose 1.0
  `steered_norm` on row 0 is 132.00 and `renormed_norm` 90.33, the clean `lengths` to float rounding.
- **The gap between `effect` and `renorm_effect` is the part that was length**: 0.01 at dose 0.25, 0.22 at
  0.5 and 1.46 at 1.0.
- **Save `typical` and `difference_norm` beside the curve.** Here the difference is a fifth of the
  activation's length (17.8 against 88.0).
- **Holding a direction's coordinate at a value** is the clamp recipe in the `clab-ablation` skill.

## A dose by layer grid

Sweep the site's `layers` under its own label, and it crosses the `dose`. Every read of the site follows the
layer, so each layer steers along its own difference at its own `typical`:

```json
{
  "description": "A layer by dose grid for steering a forward on GPT-2. At each layer the direction is that layer's mean residual stream at the last token over positive sentences less the mean over negative ones, made unit length. It is scaled to the median norm of the prompts' own activations there. The site's `layers` and the write's dose sweep under two labels, so they cross. `effect` is the mean logit of ' great' less ' terrible', `drift` the mean KL from the unsteered distribution, and `clean_effect` the unsteered logit difference.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {
    "positive": {"rows": [{"input": "I love this movie. It is wonderful."}, {"input": "What a great day, I am so happy."},
                          {"input": "The food was delicious and the staff were lovely."}, {"input": "This is the best book I have ever read."},
                          {"input": "We had an amazing time at the party."}, {"input": "The concert was fantastic and I enjoyed it."}]},
    "negative": {"rows": [{"input": "I hate this movie. It is awful."}, {"input": "What a horrible day, I am so sad."},
                          {"input": "The food was disgusting and the staff were rude."}, {"input": "This is the worst book I have ever read."},
                          {"input": "We had a miserable time at the party."}, {"input": "The concert was terrible and I hated it."}]},
    "prompts": {"rows": [{"input": "The food at the restaurant was"}, {"input": "My trip to the beach was"},
                         {"input": "The new phone I bought is"}, {"input": "The concert last night was"}]}
  },
  "sites": {"resid": {"component": "layer_output", "layers": {"sweep": [3, 6, 9], "label": "layer"}}, "head": {"component": "logits"}},
  "steps": {
    "pos_h": {"kind": "forward", "input": "positive.input", "interventions": {"r": {"site": "resid", "pos": -1}}},
    "neg_h": {"kind": "forward", "input": "negative.input", "interventions": {"r": {"site": "resid", "pos": -1}}},
    "pos_mean": {"kind": "op", "op": "mean", "of": "pos_h.r"},
    "neg_mean": {"kind": "op", "op": "mean", "of": "neg_h.r"},
    "difference": {"kind": "op", "op": "sub", "of": "pos_mean", "with": "neg_mean"},
    "difference_norm": {"kind": "op", "op": "norm", "of": "difference"},
    "direction": {"kind": "op", "op": "normalize", "of": "difference"},
    "base": {"kind": "forward", "input": "prompts.input",
             "interventions": {"r": {"site": "resid", "pos": -1}, "logits": {"site": "head", "pos": -1}}},
    "lengths": {"kind": "op", "op": "norm", "of": "base.r"},
    "typical": {"kind": "op", "op": "median", "of": "lengths"},
    "unit_dose": {"kind": "op", "op": "scale", "of": "direction", "factor": "typical"},
    "steered": {"kind": "forward", "input": "prompts.input", "interventions": {
      "steer": {"site": "resid", "pos": -1,
                "ops": [{"op": "add", "with": "unit_dose", "scale": {"sweep": [0.0, 0.25, 0.5, 1.0], "label": "dose"}}]},
      "logits": {"site": "head", "pos": -1}}},
    "clean_ld": {"kind": "op", "op": "logit_diff", "of": "base.logits", "a": {"text": " great"}, "b": {"text": " terrible"}},
    "clean_effect": {"kind": "op", "op": "mean", "of": "clean_ld"},
    "ld": {"kind": "op", "op": "logit_diff", "of": "steered.logits", "a": {"text": " great"}, "b": {"text": " terrible"}},
    "effect": {"kind": "op", "op": "mean", "of": "ld"},
    "kl": {"kind": "op", "op": "kl", "of": "steered.logits", "against": "base.logits"},
    "drift": {"kind": "op", "op": "mean", "of": "kl"}
  },
  "saves": ["clean_effect", "effect", "drift", "typical", "difference_norm"]
}
```

`clab table out/steer_grid effect drift`, then `typical difference_norm`:

```text
at                       effect              drift
{"layer":3,"dose":0.0}   2.464017868041992   0.0
{"layer":3,"dose":0.25}  3.466856002807617   0.050004735589027405
{"layer":3,"dose":0.5}   4.588262557983398   0.18938933312892914
{"layer":3,"dose":1.0}   6.064092636108398   0.9432028532028198
{"layer":6,"dose":0.0}   2.464017868041992   0.0
{"layer":6,"dose":0.25}  4.283960342407227   0.0722854882478714
{"layer":6,"dose":0.5}   5.699560165405273   0.23923064768314362
{"layer":6,"dose":1.0}   7.118005752563477   0.7412769794464111
{"layer":9,"dose":0.0}   2.464017868041992   0.0
{"layer":9,"dose":0.25}  6.67633056640625    0.1484648734331131
{"layer":9,"dose":0.5}   9.95606803894043    0.3905811309814453
{"layer":9,"dose":1.0}   14.611265182495117  0.8766814470291138

at           typical             difference_norm
{"layer":3}  63.220176696777344  7.141604900360107
{"layer":6}  87.97744750976562   17.7718448638916
{"layer":9}  171.39151000976562  34.83696746826172
```

- **Every dose-0 point is the null**: `effect` equals `clean_effect`, 2.464, and `drift` is exactly 0.0 at each
  layer.
- **Compare layers at matched drift, not matched dose.** Near 0.15 to 0.19 nats, layer 9 at dose 0.25 moves the
  effect to 6.68 and layer 3 at dose 0.5 to 4.59.
- **`clab show out/steer_grid` lists the axes `layer, dose`** for `effect` and `drift`, 12 points. Sweeps
  that share a label are one axis and pair their points instead
  ([sweeps](../../clab/references/sweeps.md#labels-zip-or-cross)).

## Counts, whole words and a control at one dose

`mean` of a row score is the fraction of replies, and `sum` the count. Two fragments of
`steer_generation.json`, the skill's main recipe: an entry added to the `steered` step's `interventions`, then
steps that replace `random` with a control at dose 1.0 alone and count:

```json
{"cheer_word": {"site": "resid", "pos": {"frame": "1:", "index": -1,
  "scope": {"text": ["glad", "happy", "delight", "beautiful", "wonderful", "love"], "lower": true, "before": true}}}}
```

```json
{
  "random": {"kind": "generate", "input": "qa.chat", "max_new_tokens": 16, "min_new_tokens": 16, "do_sample": false,
    "interventions": {
      "prompt": {"site": "resid", "pos": -1, "ops": [{"op": "add", "with": "random_dose", "scale": 1.0}]},
      "decode": {"site": "resid", "pos": {"frame": "1:"}, "ops": [{"op": "add", "with": "random_dose", "scale": 1.0}]}}},
  "cheer_count": {"kind": "op", "op": "sum", "of": "cheer_row"},
  "random_cheer_count": {"kind": "op", "op": "sum", "of": "random_cheer_row"},
  "word_row": {"kind": "op", "op": "count", "of": "steered.cheer_word", "over": ["pos"]},
  "word_count": {"kind": "op", "op": "sum", "of": "word_row"}
}
```

Saved as `steer_counts.json` with the new steps and `cheer_row` added to `saves`, `clab table out/steer_counts
cheer_count word_count random_cheer_count`, then `cheer_row word_row` at dose 1.0:

```text
at            cheer_count  word_count  random_cheer_count
{"dose":0.0}  0.0          0.0         1.0
{"dose":0.5}  5.0          5.0         1.0
{"dose":1.0}  5.0          3.0         1.0
{"dose":2.0}  0.0          0.0         1.0

at            example_id  row  chat                                              cheer_row  alignment_missing  word_row
{"dose":1.0}  japan       1    [{"role":"user","content":"What is the capital …  1.0                           1.0
{"dose":1.0}  italy       2    [{"role":"user","content":"What is the capital …  1.0        true
{"dose":1.0}  germany     3    [{"role":"user","content":"What is the capital …  1.0        true
```

- **A control at one dose takes its own label, or none.** Under the steer's label, `{"sweep": [1.0], "label":
  "dose"}` is refused: `label 'dose' is also the sweep at steps.steered.interventions.prompt.ops.0.scale, over
  [0.0, 0.5, 1.0, 2.0]; one label is one axis, whose sweeps pair their points`.
- **A plain `"scale": 1.0` repeats the control on every `dose` row**, so the counts per dose and the control
  share one table. Its own label (`rdose`) crosses the axes instead, and every row's key holds both:
  `{"dose":0.5,"rdose":1.0}`.
- **`contains` matches substrings**: `delight` scores Italy's and Germany's "I'm delighted" and "I am
  delighted". A short word scores inside a longer one, as ` dark` does in "darkness" in the skill's scoring recipe.
- **A scope matches whole words**, so `word_count` is 3 where `cheer_count` is 5. List every form a whole-word
  read should count (`delighted` beside `delight`).
- **`count` is 0 on a row the read missed**, printed blank and flagged `alignment_missing`, so `sum` counts the
  replies. `before` with `index: -1` reads the call that chose the word's last token, which also reaches a word
  on the row's last generated token.

## Steering the prompt alone

A write at `pos: -1` in a generate step changes the prefill, and later tokens see it only through the cache.
A fragment of `steer_generation.json`'s `steps`, the skill's main recipe, adds a generation steered at the
prompt alone and compares it with the steer at every token:

```json
{
  "prefill_only": {"kind": "generate", "input": "qa.chat", "max_new_tokens": 16, "min_new_tokens": 16, "do_sample": false,
    "interventions": {"prompt": {"site": "resid", "pos": -1,
      "ops": [{"op": "add", "with": "unit_dose", "scale": {"sweep": [0.0, 0.5, 1.0, 2.0], "label": "dose"}}]}}},
  "po_text": {"kind": "op", "op": "detokenize", "of": "prefill_only"},
  "po_cheer_row": {"kind": "op", "op": "contains", "of": "po_text",
                   "text": {"text": ["glad", "happy", "delight", "beautiful", "wonderful", "love"], "lower": true}},
  "po_cheer": {"kind": "op", "op": "mean", "of": "po_cheer_row"},
  "po_diff": {"kind": "op", "op": "ne", "of": "prefill_only", "with": "clean"},
  "po_changed_frac": {"kind": "op", "op": "mean", "of": "po_diff", "over": ["rows", "new"]},
  "changed_frac": {"kind": "op", "op": "mean", "of": "diff", "over": ["rows", "new"]}
}
```

Saved as `steer_prefill_only.json` with these steps added, `clab table out/steer_prefill_only cheer po_cheer
changed_frac po_changed_frac` gives:

```text
at            cheer               po_cheer            changed_frac        po_changed_frac
{"dose":0.0}  0.0                 0.0                 0.0                 0.0
{"dose":0.5}  0.8333333134651184  0.0                 0.4166666567325592  0.0520833320915699
{"dose":1.0}  0.8333333134651184  0.0                 0.9583333134651184  0.8854166865348816
{"dose":2.0}  0.0                 0.1666666716337204  0.9479166865348816  0.8854166865348816
```

- **The prefill write changes the reply and not its mood.** At dose 1.0 it changes 89% of the ids. Four of
  six replies open "I'm sorry for any confusion, but as an AI designed to assist with language".
- **Name `new` in `over`** for a mean over ids: `{"over": ["rows", "new"]}` is the fraction of all generated
  ids that changed. The default removes `rows` and `pos` and leaves `[new]`.
- **To steer every position, the prompt included**, write once at `{"frame": ":"}`. That includes position 0,
  whose norm is far above the dose ([above](#the-norm-a-dose-is-set-against)).
- **A write's frames and the calls they land at** are the `clab` skill's
  [generate reference](../../clab/references/generate.md).
