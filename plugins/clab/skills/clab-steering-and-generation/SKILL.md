---
name: clab-steering-and-generation
description: Push a model with a direction and score what it generates, as clab JSON documents. Use for steering vectors and contrastive activation addition (a mean difference of two prompt sets), a dose set from the activations' norms, a matched random-direction control, dose and layer sweeps, renormalize, clamping, and steering every generated token. Also for generation experiments, which score what the model said: did it say the answer (contains, strip, a whole-word scope), its confidence as it said it, the text after a marker, a reasoning trace's length (until, count), teacher forcing an answer with a list input, chat prompts (template, add_generation_prompt, a turn), and counting alignment_missing rows before a mean. The format and the generate step's frames are the `clab` skill; patching through a generation is `clab-patching`; projecting a direction out is `clab-ablation`; steering with an SAE feature or a probe direction is `clab-trained-featurizers`.
---

# Steering and generation experiments in clab

Verified on clab `830182f` with nnterp `e5d0f8e`, nnsight 0.8.2 and transformers 5.20.2; the documents run on
the CPU on `HuggingFaceTB/SmolLM2-135M-Instruct` (every generation and chat) and `openai-community/gpt2`
(steering a forward), in seconds.

A steering experiment adds a direction to the residual stream and measures what moves. A generation
experiment scores the text a model writes. Both are documents in the format the `clab` skill teaches. The
`generate` step's options, frames and rows that stop early are in that skill's
[generate reference](../clab/references/generate.md), and `pos` in its
[positions reference](../clab/references/positions.md). This skill holds the recipes and their controls.

## What breaks

Each of these runs without a refusal and costs a conclusion.

- **A generate step without `do_sample: false`** takes its decoding from the checkpoint's generation config,
  which may sample, and nothing seeds it. Write `do_sample: false` on every generate step.
- **A steering write at `pos: -1` in a generate step** lands in the prefill only; later tokens see it through
  the cache. Write two entries, `pos: -1` and `{"frame": "1:"}`. Steered at the prompt alone, the mood
  direction below changes 89% of the generated ids at dose 1.0 and makes no reply cheerful.
- **A dose written as a bare number** means nothing without the norm it is added to. Set it as a fraction of
  the median norm at the positions the write lands on (`typical` below).
- **A `mean` of norms over every prompt token** is set by the first token's outlier. At SmolLM2's layer 12
  that token's norm is 26,120, the median 147 and the mean 850. Use `median`.
- **A random direction of another length** is no control. Give it the steering vector's length; the recipe
  draws one and scales it as `random_dose`.
- **A count of changed tokens as evidence** of a direction: a random direction of the same length changes as
  many. At dose 1.0 both change 12 to 16 of 16 ids per row. Score the behaviour you steer.
- **`contains` on a chat reply without `strip`** misses an answer that has no space before it. That happens
  when the answer is the reply's first word, which follows the header's newline, and after a quote mark
  (`"eight`). On a SentencePiece tokenizer, decoding also drops the first generated token's space. Give
  the answer column `strip: true` wherever `contains` or a scope searches generated text.
- **`contains` as a whole-word test** scores a substring: `{"text": "Pari"}` scores 1 on a reply that says
  "Paris". `strip` loosens it further. Count whole words with a scope over the decode calls.
- **`contains` on the whole reply** also finds words the reply repeats from the question. "Is Paris or Lyon
  the capital of France?" gets "Paris and Lyon are …", and the wrong answer ` Lyon` scores 1. Score the text
  after a marker, or teacher-force both candidates ([generation scoring](references/generation-scoring.md)).
- **A mean over a scoped read without counting its rows**: reductions skip `alignment_missing` rows, and the
  mean does not say so. Count the reached rows first (`reached` below) and quote the count beside the mean.
- **An answer on the row's last generated token** is `alignment_missing` to a read scoped to the word without
  `before`, since no call is fed that token. Leave room after the answer in `max_new_tokens`.
- **A teacher-forced answer scoped with the default `occurrence: 0`** lands on the prompt's copy when the
  prompt names it, with no refusal; so does an answer appended without its leading space. Write
  `"occurrence": -1`, keep the space, and check the read's `positions` in `clab show`.
- **`min_new_tokens` left out when two generations are compared**: rows stop on EOS at different calls, and
  a read at a fixed call is missing on some. Set it equal to `max_new_tokens`.
- **`renormalize` listed before the `add`** rescales the clean activation to its own norm, a no-op. The run
  then equals the steer without `renormalize`, to the bit. List it after the `add`.

## Steer a generation

Use this to push a behaviour with a direction and read the text it produces. The direction is a mean
difference of two prompt sets, and the dose is a fraction of the activations' own length. The control is a
random direction of the same length at the same doses:

```json
{
  "description": "Steer a chat model's answers with a mood direction: layer 12's residual stream at the prompt's last token, the mean over cheerful user messages less the mean over gloomy ones, made unit length and scaled to the median norm of the clean generation's activations there. It is added at the prompt's last token and at every generated token; a random direction of the same length is added at the same doses. `cheer` asks whether a reply uses a cheerful word, `correct` whether it still names the capital, `changed` counts the ids that differ from the clean generation's.",
  "model": {"key": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac", "dtype": "float32"},
  "data": {
    "qa": {"template": {"add_generation_prompt": true}, "rows": [
      {"example_id": "france", "chat": [{"role": "user", "content": "What is the capital of France?"}], "answer": " Paris"},
      {"example_id": "japan", "chat": [{"role": "user", "content": "What is the capital of Japan?"}], "answer": " Tokyo"},
      {"example_id": "italy", "chat": [{"role": "user", "content": "What is the capital of Italy?"}], "answer": " Rome"},
      {"example_id": "germany", "chat": [{"role": "user", "content": "What is the capital of Germany?"}], "answer": " Berlin"},
      {"example_id": "spain", "chat": [{"role": "user", "content": "What is the capital of Spain?"}], "answer": " Madrid"},
      {"example_id": "egypt", "chat": [{"role": "user", "content": "What is the capital of Egypt?"}], "answer": " Cairo"}]},
    "cheerful": {"template": {"add_generation_prompt": true}, "rows": [
      {"chat": [{"role": "user", "content": "I love sunny days. Everything feels wonderful!"}]},
      {"chat": [{"role": "user", "content": "What a beautiful morning, I am so happy!"}]},
      {"chat": [{"role": "user", "content": "This is the best day of my life!"}]},
      {"chat": [{"role": "user", "content": "I am thrilled, the party was amazing!"}]},
      {"chat": [{"role": "user", "content": "My friends are lovely and I feel great!"}]},
      {"chat": [{"role": "user", "content": "We won the game! I am overjoyed!"}]}]},
    "gloomy": {"template": {"add_generation_prompt": true}, "rows": [
      {"chat": [{"role": "user", "content": "I hate rainy days. Everything feels awful."}]},
      {"chat": [{"role": "user", "content": "What a miserable morning, I am so sad."}]},
      {"chat": [{"role": "user", "content": "This is the worst day of my life."}]},
      {"chat": [{"role": "user", "content": "I am furious, the party was terrible."}]},
      {"chat": [{"role": "user", "content": "My friends are cruel and I feel horrible."}]},
      {"chat": [{"role": "user", "content": "We lost the game. I am devastated."}]}]}
  },
  "sites": {"resid": {"component": "layer_output", "layers": 12}},
  "steps": {
    "up": {"kind": "forward", "input": "cheerful.chat", "interventions": {"r": {"site": "resid", "pos": -1}}},
    "down": {"kind": "forward", "input": "gloomy.chat", "interventions": {"r": {"site": "resid", "pos": -1}}},
    "up_mean": {"kind": "op", "op": "mean", "of": "up.r"},
    "down_mean": {"kind": "op", "op": "mean", "of": "down.r"},
    "difference": {"kind": "op", "op": "sub", "of": "up_mean", "with": "down_mean"},
    "difference_norm": {"kind": "op", "op": "norm", "of": "difference"},
    "direction": {"kind": "op", "op": "normalize", "of": "difference"},
    "clean": {"kind": "generate", "input": "qa.chat", "max_new_tokens": 16, "min_new_tokens": 16, "do_sample": false,
              "interventions": {"r": {"site": "resid", "pos": {"frame": "1:"}}}},
    "lengths": {"kind": "op", "op": "norm", "of": "clean.r"},
    "typical": {"kind": "op", "op": "median", "of": "lengths"},
    "unit_dose": {"kind": "op", "op": "scale", "of": "direction", "factor": "typical"},
    "zero": {"kind": "op", "op": "scale", "of": "difference", "factor": 0.0},
    "draw": {"kind": "op", "op": "noise", "of": "zero", "scale": 1.0, "seed": 7},
    "random_unit": {"kind": "op", "op": "normalize", "of": "draw"},
    "random_dose": {"kind": "op", "op": "scale", "of": "random_unit", "factor": "typical"},
    "steered": {"kind": "generate", "input": "qa.chat", "max_new_tokens": 16, "min_new_tokens": 16, "do_sample": false,
      "interventions": {
        "prompt": {"site": "resid", "pos": -1,
                   "ops": [{"op": "add", "with": "unit_dose", "scale": {"sweep": [0.0, 0.5, 1.0, 2.0], "label": "dose"}}]},
        "decode": {"site": "resid", "pos": {"frame": "1:"},
                   "ops": [{"op": "add", "with": "unit_dose", "scale": {"sweep": [0.0, 0.5, 1.0, 2.0], "label": "dose"}}]}}},
    "random": {"kind": "generate", "input": "qa.chat", "max_new_tokens": 16, "min_new_tokens": 16, "do_sample": false,
      "interventions": {
        "prompt": {"site": "resid", "pos": -1,
                   "ops": [{"op": "add", "with": "random_dose", "scale": {"sweep": [0.0, 0.5, 1.0, 2.0], "label": "dose"}}]},
        "decode": {"site": "resid", "pos": {"frame": "1:"},
                   "ops": [{"op": "add", "with": "random_dose", "scale": {"sweep": [0.0, 0.5, 1.0, 2.0], "label": "dose"}}]}}},
    "text": {"kind": "op", "op": "detokenize", "of": "steered"},
    "random_text": {"kind": "op", "op": "detokenize", "of": "random"},
    "cheer_row": {"kind": "op", "op": "contains", "of": "text",
                  "text": {"text": ["glad", "happy", "delight", "beautiful", "wonderful", "love"], "lower": true}},
    "random_cheer_row": {"kind": "op", "op": "contains", "of": "random_text",
                         "text": {"text": ["glad", "happy", "delight", "beautiful", "wonderful", "love"], "lower": true}},
    "correct_row": {"kind": "op", "op": "contains", "of": "text", "text": {"column": "qa.answer", "strip": true}},
    "random_correct_row": {"kind": "op", "op": "contains", "of": "random_text", "text": {"column": "qa.answer", "strip": true}},
    "cheer": {"kind": "op", "op": "mean", "of": "cheer_row"},
    "random_cheer": {"kind": "op", "op": "mean", "of": "random_cheer_row"},
    "correct": {"kind": "op", "op": "mean", "of": "correct_row"},
    "random_correct": {"kind": "op", "op": "mean", "of": "random_correct_row"},
    "diff": {"kind": "op", "op": "ne", "of": "steered", "with": "clean"},
    "changed": {"kind": "op", "op": "sum", "of": "diff", "over": ["new"]},
    "random_diff": {"kind": "op", "op": "ne", "of": "random", "with": "clean"},
    "random_changed": {"kind": "op", "op": "sum", "of": "random_diff", "over": ["new"]}
  },
  "saves": ["text", "random_text", "cheer", "random_cheer", "correct", "random_correct", "changed", "random_changed",
            "typical", "difference_norm"]
}
```

What each part does:

- **The harvests** read the last prompt token of each chat, where the reply is decided. Both sets render
  through the same template as the questions, so the three prompts end alike.
- **`direction`** is the mean difference made unit length by `normalize`.
- **`typical`** is the median norm of the clean generation's activations at the generated tokens, where most
  of the writes land; `unit_dose` is the direction at that length. Dose 1.0 adds a vector as long as the
  median activation.
- **`random_dose`** is one Gaussian draw from `seed` (`noise` on a value with no rows), made unit length and
  given the same length. The sweep over seeds is in [steering controls](references/steering-controls.md).
- **Both writes sweep under one `label`**, so they move together and `steered` and `random` share the `dose`
  axis ([sweeps](../clab/references/sweeps.md)).
- **`changed`** compares ids, which are `[rows, new]`; a reduction of them names `new` in `over`.

```bash
clab run documents/steer_generation.json
clab table out/steer_generation cheer random_cheer correct random_correct
```

```text
at            cheer               random_cheer        correct             random_correct
{"dose":0.0}  0.0                 0.0                 1.0                 1.0
{"dose":0.5}  0.8333333134651184  0.0                 1.0                 0.8333333134651184
{"dose":1.0}  0.8333333134651184  0.1666666716337204  0.1666666716337204  0.0
{"dose":2.0}  0.0                 0.0                 0.1666666716337204  0.0
```

`clab table --csv out/steer_generation text` prints whole texts; the Japan row at each dose:

```text
"{""dose"":0.0}",japan,1,"[{""role"":""user"",""content"":""What is the capital of Japan?""}]",The capital of Japan is Tokyo. It is a city that has been a major
"{""dose"":0.5}",japan,1,"[{""role"":""user"",""content"":""What is the capital of Japan?""}]",The capital of Japan is Tokyo. It's a city that is known for its
"{""dose"":1.0}",japan,1,"[{""role"":""user"",""content"":""What is the capital of Japan?""}]",I'm glad you're happy to see you! I'm glad you're not
"{""dose"":2.0}",japan,1,"[{""role"":""user"",""content"":""What is the capital of Japan?""}]","Japan is Japan, Japan, Japan, Japan, Japan, Japan, Japan,"
```

The same row of `random_text` at dose 1.0 reads `Japan's a beautiful and beautiful, but the "Hokaidai" is`.

- **The null is dose 0.** `clab table out/steer_generation changed random_changed` gives 0.0 on every row of
  both at `{"dose":0.0}`: `h + 0 * d` is `h` in float32, so the ids equal the clean generation's.
- **The band is dose 0.5.** Five of six replies call the city beautiful and all six still name it. At 1.0
  five replies turn into "I'm glad" and "I am delighted", and one names the capital. At 2.0 the text collapses.
- **The random direction moves the text and not the mood.** At 0.5 it repeats the answer, and at 1.0 it writes
  gibberish; one row's "beautiful" is its 0.17. It changes as many ids as the direction (12 to 16 of 16 per
  row at 1.0), so `changed` cannot tell the two apart.
- **Check the difference's length**: `difference_norm` is 8.46 against a `typical` of 132.37. A difference
  orders of magnitude below `typical` is rounding, not a direction.
- **`clab show out/steer_generation`** prints where each write landed: `steered.prompt` at `[36]` and
  `steered.decode` at `[37,38,39,…,51]` on all 6 rows.

A forward on GPT-2, with its drift (`kl`), seeds, `renormalize`, a dose by layer grid and clamping, is in
[steering controls](references/steering-controls.md).

## Score a generation

Use this to ask whether the model said the answer, where, and how sure it was. Each row's reply is scored
three ways: `contains` over the text, a read at the answer as a whole word, and the probability the model gave
the answer's tokens as it chose them:

```json
{
  "description": "Did a chat model say the answer, where, and how sure was it. Ten greedy tokens per row as the assistant. `said` is a substring test on the decoded text with the answer's leading space optional (`strip`); `said_spaced` is the same test without `strip`; `said_decoy` tests a plausible wrong answer, the null. `at` reads the head at the answer's last token where the reply says it as a whole word; `chose` reads the calls that chose its tokens (`before`). A row that never says it is alignment_missing in both; `reached` counts the rows `at` found, and `mean_p` averages over them only.",
  "model": {"key": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac", "dtype": "float32"},
  "data": {"qa": {"template": {"add_generation_prompt": true}, "rows": [
    {"example_id": "france", "chat": [{"role": "user", "content": "What is the capital of France?"}], "answer": " Paris", "decoy": " Lyon"},
    {"example_id": "japan", "chat": [{"role": "user", "content": "What is the capital of Japan?"}], "answer": " Tokyo", "decoy": " Osaka"},
    {"example_id": "lyon", "chat": [{"role": "user", "content": "Is Paris or Lyon the capital of France?"}], "answer": " Paris", "decoy": " Lyon"},
    {"example_id": "spider", "chat": [{"role": "user", "content": "How many legs does a spider have?"}], "answer": " eight", "decoy": " six"},
    {"example_id": "week", "chat": [{"role": "user", "content": "How many days are in a week?"}], "answer": " seven", "decoy": " five"},
    {"example_id": "bicycle", "chat": [{"role": "user", "content": "How many wheels does a bicycle have?"}], "answer": " two", "decoy": " three"}]}},
  "sites": {"head": {"component": "logits"}},
  "steps": {
    "answer": {"kind": "generate", "input": "qa.chat", "max_new_tokens": 10, "min_new_tokens": 10, "do_sample": false,
      "interventions": {
        "at": {"site": "head", "pos": {"frame": "1:", "scope": {"column": {"column": "qa.answer", "strip": true}}, "index": -1}},
        "chose": {"site": "head", "pos": {"frame": "1:", "scope": {"column": {"column": "qa.answer", "strip": true}, "before": true}}}}},
    "text": {"kind": "op", "op": "detokenize", "of": "answer"},
    "said_spaced": {"kind": "op", "op": "contains", "of": "text", "text": "qa.answer"},
    "said": {"kind": "op", "op": "contains", "of": "text", "text": {"column": "qa.answer", "strip": true}},
    "said_decoy": {"kind": "op", "op": "contains", "of": "text", "text": {"column": "qa.decoy", "strip": true}},
    "accuracy": {"kind": "op", "op": "mean", "of": "said"},
    "reached": {"kind": "op", "op": "count", "of": "answer.at", "over": ["rows", "pos"]},
    "p": {"kind": "op", "op": "token_prob", "of": "answer.chose", "token": {"column": "qa.answer", "strip": true}},
    "mean_p": {"kind": "op", "op": "mean", "of": "p"}
  },
  "saves": ["text", "said_spaced", "said", "said_decoy", "accuracy", "reached", "p", "mean_p"]
}
```

`clab table out/generated_capitals text said_spaced said said_decoy`, then `p`, then `accuracy reached mean_p`:

```text
example_id  row  chat                                              text                                              answer  said_spaced  said  decoy   said_decoy
france      0    [{"role":"user","content":"What is the capital …  The capital of France is Paris. It is a            Paris  1.0          1.0    Lyon   0.0
lyon        2    [{"role":"user","content":"Is Paris or Lyon the…  Paris and Lyon are indeed the capital cities of…   Paris  0.0          1.0    Lyon   1.0
spider      3    [{"role":"user","content":"How many legs does a…  The answer to this question is "eight legs."       eight  0.0          1.0    six    0.0
week        4    [{"role":"user","content":"How many days are in…  A week is approximately 31 days. However           seven  0.0          0.0    five   0.0

example_id  row  pos  pos_token  call  chat                                              answer  value               alignment_missing
france      0    41   Ġis        5     [{"role":"user","content":"What is the capital …   Paris  0.9883716106414795
lyon        2    38   Ċ          0     [{"role":"user","content":"Is Paris or Lyon the…   Paris  0.2772400975227356
spider      3    44   Ġ"         7     [{"role":"user","content":"How many legs does a…   eight  0.1590833216905594
week        4                          [{"role":"user","content":"How many days are in…   seven                      true

accuracy            reached  mean_p
0.6666666865348816  4.0      0.5577138066291809
```

- **`strip` doubles the accuracy here**: without it 2 of 6 rows score (0.33), with it 4 of 6 (0.67). The
  Lyon row's "Paris" opens the reply, and the spider row's "eight" follows a quote mark.
- **The null is `said_decoy`**: 0 on every row but Lyon, whose reply repeats the question's "Lyon".
- **`chose` reads the call that chose each of the answer's tokens.** For the Lyon row's opening "Paris" that
  is call 0, the prefill's last position (`Ċ`, the assistant header's newline).
- **Count before the mean**: `reached` is 4, so `mean_p` (0.558) is the mean over 4 of 6 rows. The week and
  bicycle rows answer "31 days" and "8 wheels" and are `alignment_missing`. Quote "0.558 over 4 of 6 rows".
- **At `max_new_tokens: 8`** the spider row's "eight" is its last generated token. `said` and `p` still find
  it (`before` counts the last token), and `reached` drops to 3.

The text after a marker, a reasoning trace's length, teacher forcing, chat turns and counting missing rows
per sweep point are in [generation scoring](references/generation-scoring.md).

## Where else to look

- [references/steering-controls.md](references/steering-controls.md): the norms a dose is set against;
  steering a forward with drift; matched random directions over seeds; a dose by layer grid; `renormalize`;
  holding a direction's coordinate at a value; elementwise `clamp`; steering the prefill alone.
- [references/generation-scoring.md](references/generation-scoring.md): the text after a marker (`after`);
  a reasoning trace's length (`until`, `count`); teacher forcing with a list input; chat prompts, `template`,
  `add_generation_prompt` and `turn` positions; counting `alignment_missing` rows inside a document and per
  sweep point.
- The `clab` skill: the format, `pos`, the generate step's mechanics ([generate](../clab/references/generate.md)),
  answer columns and chats ([data](../clab/references/data.md)), ops ([ops](../clab/references/ops.md)).
- Patching a counterfactual's activation into a generation: the `clab-patching` skill.
- Projecting a direction out at every layer (holding it at 0): the `clab-ablation` skill.
- Steering with an SAE feature, or using a trained probe's direction: the `clab-trained-featurizers` skill.
- What the model predicts at a position, decoded (`top_k`, the logit lens): the `clab-lenses-and-attention` skill.
- A refusal, or a null that does not equal the clean run: the `clab-debugging` skill.
