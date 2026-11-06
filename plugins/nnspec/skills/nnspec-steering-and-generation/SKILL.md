---
name: nnspec-steering-and-generation
description: Steer a model with a direction and score what it generates, as clab JSON documents. Use to push a behaviour with a steering vector (contrastive activation addition, a mean difference of two prompt sets). Covers the dose set from the activations' norms, a matched random-direction control, dose and layer sweeps, renormalize, and steering every generated token. Also use to score what a model generated: did it say the answer (contains, strip, a whole word), and how confident was it as it said it. Covers the text after a marker, a reasoning trace's length, teacher forcing a fixed answer or a chat reply, and chat prompts. The format and the generate step are the `clab` skill, and patching is `clab-patching`. Projecting a direction out and clamping are `clab-ablation`; SAE features and probe directions, `clab-trained-featurizers`; the logit lens and decoded top-k, `clab-lenses-and-attention`; refusals and wrong numbers, `clab-debugging`.
---

# Steering and generation experiments in clab

Verified on clab `dca9630` with nnterp `e5d0f8e`, nnsight 0.8.2 and transformers 5.20.2; the documents run on
the CPU on `HuggingFaceTB/SmolLM2-135M-Instruct` (every generation and chat) and `openai-community/gpt2`
(steering a forward), in seconds.

A steering experiment adds a direction to the residual stream and measures what moves. A generation
experiment scores the text a model writes. Both are documents in the format the `clab` skill teaches. The
`generate` step's options, frames and rows that stop early are in that skill's
[generate reference](../clab/references/generate.md), and `pos` in its
[positions reference](../clab/references/positions.md). This skill holds the recipes and their controls.

## What breaks

Each of these runs without a refusal and costs a conclusion.

- **A generate step without `do_sample: false`** may sample (the `clab` skill's list). Write it on every one.
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
- **`contains` on a chat reply without `strip`** misses an answer that has no space before it. A reply's first
  word follows the header's newline, a word may follow a quote mark (`"eight`), and a SentencePiece tokenizer
  drops the first generated token's space (the `clab` skill's list). Give the answer column `strip: true`
  wherever `contains` or a scope searches generated text.
- **`contains` as a whole-word test** scores a substring: ` dark` scores 1 on a reply that says "darkness".
  Score a whole word with `count` of a scoped read over `pos` (`word` below).
- **`contains` on the whole reply** also finds words the reply repeats from the question. "Is Paris or Lyon
  the capital of France?" gets "Paris and Lyon are …", and the wrong answer ` Lyon` scores 1. Score the text
  after a marker, or teacher-force both candidates ([generation scoring](references/generation-scoring.md)).
- **A mean over a scoped read skips the rows it never reached** (the `clab` skill's list). Count them first
  (`reached` below) and quote the count beside the mean.
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
  "description": "Steer a chat model's answers with a mood direction at layer 12's residual stream. The direction is the mean at the prompt's last token over cheerful user messages less the mean over gloomy ones, made unit length. It is scaled to the median norm of the clean generation's activations there and added at the prompt's last token and at every generated token. A random direction of the same length is added at the same doses. `cheer` asks whether a reply uses a cheerful word, `correct` whether it still names the capital, `changed` counts the ids that differ from the clean generation's.",
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
  five replies turn into "I'm glad" and "I am delighted", and one names the capital.
- **The top dose collapses the text**, and a word score falls with it: at 2.0 every reply repeats its country
  and `cheer` is 0 again. Read two replies per dose before trusting a count; report the doses below a collapse.
- **The random direction moves the text and not the mood.** At 0.5 it repeats the answer, and at 1.0 it writes
  gibberish; one row's "beautiful" is its 0.17. It changes as many ids as the direction (12 to 16 of 16 per
  row at 1.0), so `changed` cannot tell the two apart.
- **Check the difference's length**: `difference_norm` is 8.46 against a `typical` of 132.37. In bfloat16 an
  element rounds by about 2^-8 of its size, so a difference under about 1% of `typical` can be rounding.
- **`clab show out/steer_generation`** prints where each write landed: `steered.prompt` at `[36]` and
  `steered.decode` at `[37,38,39,…,51]` on all 6 rows.
- **A control at one dose takes its own label, or none**; under `dose` it is refused. A count of replies is
  `sum` where `mean` is the fraction. Both, and whole-word lists, are in
  [steering controls](references/steering-controls.md#counts-whole-words-and-a-control-at-one-dose).

A forward on GPT-2, with its drift (`kl`), seeds, `renormalize` and a complete dose by layer grid, is in
[steering controls](references/steering-controls.md).

## Score a generation

Use this to ask whether the model said the answer, where, and how sure it was. Each row's reply is scored
three ways: `contains` over the text, a read at the answer as a whole word, and the probability the model gave
the answer's tokens as it chose them:

```json
{
  "description": "Did a chat model say the answer, where, and how sure was it. Ten greedy tokens per row as the assistant. `said` is a substring test on the decoded text with the answer's leading space optional (`strip`); `said_spaced` is the same test without `strip`; `said_decoy` tests a plausible wrong answer, the null. `at` reads the head at the answer's last token where the reply says it as a whole word, and `word`, its count over `pos`, is the whole-word score. `chose` reads the calls that chose the answer's tokens (`before`). A row that never says the word is alignment_missing in both reads; `reached` counts the rows `at` found, and `mean_p` averages over them only.",
  "model": {"key": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac", "dtype": "float32"},
  "data": {"qa": {"template": {"add_generation_prompt": true}, "rows": [
    {"example_id": "france", "chat": [{"role": "user", "content": "What is the capital of France?"}], "answer": " Paris", "decoy": " Lyon"},
    {"example_id": "japan", "chat": [{"role": "user", "content": "What is the capital of Japan?"}], "answer": " Tokyo", "decoy": " Osaka"},
    {"example_id": "lyon", "chat": [{"role": "user", "content": "Is Paris or Lyon the capital of France?"}], "answer": " Paris", "decoy": " Lyon"},
    {"example_id": "spider", "chat": [{"role": "user", "content": "How many legs does a spider have?"}], "answer": " eight", "decoy": " six"},
    {"example_id": "week", "chat": [{"role": "user", "content": "How many days are in a week?"}], "answer": " seven", "decoy": " five"},
    {"example_id": "bicycle", "chat": [{"role": "user", "content": "How many wheels does a bicycle have?"}], "answer": " two", "decoy": " three"},
    {"example_id": "light", "chat": [{"role": "user", "content": "What is the opposite of light?"}], "answer": " dark", "decoy": " heavy"}]}},
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
    "word": {"kind": "op", "op": "count", "of": "answer.at", "over": ["pos"]},
    "reached": {"kind": "op", "op": "sum", "of": "word"},
    "p": {"kind": "op", "op": "token_prob", "of": "answer.chose", "token": {"column": "qa.answer", "strip": true}},
    "mean_p": {"kind": "op", "op": "mean", "of": "p"}
  },
  "saves": ["text", "said_spaced", "said", "said_decoy", "word", "accuracy", "reached", "p", "mean_p"]
}
```

`clab table out/generated_capitals text said_spaced said said_decoy word`, then `p`, then
`accuracy reached mean_p`:

```text
example_id  row  chat                                              text                                              answer  said_spaced  said  decoy   said_decoy  word  alignment_missing
france      0    [{"role":"user","content":"What is the capital …  The capital of France is Paris. It is a            Paris  1.0          1.0    Lyon   0.0         1.0
lyon        2    [{"role":"user","content":"Is Paris or Lyon the…  Paris and Lyon are indeed the capital cities of…   Paris  0.0          1.0    Lyon   1.0         1.0
spider      3    [{"role":"user","content":"How many legs does a…  The answer to this question is "eight legs."       eight  0.0          1.0    six    0.0         1.0
week        4    [{"role":"user","content":"How many days are in…  A week is approximately 31 days. However           seven  0.0          0.0    five   0.0               true
light       6    [{"role":"user","content":"What is the opposite…  The opposite of light is darkness. Darkness is     dark   1.0          1.0    heavy  0.0               true

example_id  row  pos  pos_token  call  chat                                              answer  value                alignment_missing
france      0    41   Ġis        5     [{"role":"user","content":"What is the capital …   Paris  0.9883716106414795
lyon        2    38   Ċ          0     [{"role":"user","content":"Is Paris or Lyon the…   Paris  0.2772400975227356
spider      3    44   Ġ"         7     [{"role":"user","content":"How many legs does a…   eight  0.15908415615558624
week        4                          [{"role":"user","content":"How many days are in…   seven                       true

accuracy            reached  mean_p
0.7142857313156128  4.0      0.55771404504776
```

- **`strip` adds two rows here**: without it 3 of 7 rows score, with it 5 of 7 (`accuracy` 0.714). The Lyon
  row's "Paris" opens the reply, and the spider row's "eight" follows a quote mark.
- **`word` is the whole-word score**, `count` of the `at` read over `pos`: 1 where the reply says the answer as
  a word. The light row's reply says "darkness", so `said` is 1 and `word` is missing.
- **The null is `said_decoy`**: 0 on every row but Lyon, whose reply repeats the question's "Lyon".
- **`chose` reads the call that chose each of the answer's tokens.** For the Lyon row's opening "Paris" that
  is call 0, the prefill's last position (`Ċ`, the assistant header's newline).
- **Count before the mean**: `count` is 0 on a missed row, so `reached`, the `sum` of `word`, is 4. `mean_p`
  (0.558) is over those 4 of 7 rows; the week and bicycle replies say "31 days" and "8 wheels". Quote "0.558
  over 4 of 7 rows".
- **At `max_new_tokens: 8`** the spider row's "eight" is its last generated token. `said` and `p` still find
  it (`before` counts the last token), and `reached` drops to 3.

The text after a marker, a reasoning trace's length, teacher forcing and scoring chats are in
[generation scoring](references/generation-scoring.md). Teacher forcing a chat reply means writing its prompt
as the template's rendered text, since a list input refuses a chat column; the recipe checks the text with
`agree` ([forcing an assistant reply](references/generation-scoring.md#forcing-an-assistant-reply)).

## Where else to look

- [references/steering-controls.md](references/steering-controls.md): the norms a dose is set against; a
  forward's drift and random directions over seeds; a dose by layer grid; `renormalize`; counts, whole words
  and a one-dose control; steering the prefill alone.
- [references/generation-scoring.md](references/generation-scoring.md): the text after a marker (`after`);
  a reasoning trace's length (`until`, `count`); teacher forcing an answer or a chat reply; a reply to a
  chat; missing rows per sweep point.
- The `clab` skill: the format, `pos` and rows a scope never reached
  ([positions](../clab/references/positions.md)), the generate step's mechanics
  ([generate](../clab/references/generate.md)), answer columns and chats ([data](../clab/references/data.md)),
  ops ([ops](../clab/references/ops.md)).
- Patching a counterfactual's activation into a generation: the `clab-patching` skill.
- Projecting a direction out, holding its coordinate at a value, or clamping: the `clab-ablation` skill.
- Steering with an SAE feature, or using a trained probe's direction: the `clab-trained-featurizers` skill.
- What the model predicts at a position, decoded (`top_k`, the logit lens): the `clab-lenses-and-attention` skill.
- A refusal, or a null that does not equal the clean run: the `clab-debugging` skill.
