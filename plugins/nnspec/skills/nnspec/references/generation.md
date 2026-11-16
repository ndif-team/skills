# Generation

A `generate` step runs the model's `generate` on every row of its `input`. Its value is the new token ids,
`[rows, new]`, which `detokenize` turns into text, and its entries read and write at the forward calls their
`pos`'s `frame` names. This guide is the step's mechanics, then the recipes that score what a model wrote.
Steering a generation is in the [steering guide](steering.md).

## What breaks

- **A search of generated text needs `strip` on the answer.** A reply's first word follows a header's newline
  or a quote mark, and a Llama-style SentencePiece tokenizer drops the first generated token's leading space
  when it decodes. Write `{"column": "qa.answer", "strip": true}` wherever `contains` or a scope searches a
  generation.
- **`contains` is a substring test over the whole reply.** ` dark` scores 1 on "darkness", and so does a wrong
  answer the reply repeats from the question. Score a whole word with `count` of a scoped read, the text after
  a marker, or both candidates [teacher-forced](generation-teacher-forcing.md).
- **A mean over a scoped read skips the rows it never reached**, and the number does not say so. Count the
  rows first ([below](#did-it-say-the-answer)).
- **A word on the row's last generated token is missing to a scope without `before`**, since no call is fed
  that token. Leave room after the answer in `max_new_tokens`.
- **Rows stop on EOS at different calls**, so a read at a fixed call is missing on some. To compare two
  generations call by call, set `min_new_tokens` equal to `max_new_tokens`.

## Options

Every key of the step but `kind`, `input` and `interventions` goes to the engine's `generate`, and may sweep.

- **`do_sample: false` on every step** ([SKILL.md](../SKILL.md#what-breaks)). An option the step leaves out
  comes from the checkpoint's generation config (`meta-llama/Llama-3.2-1B`'s sets `do_sample` true at
  temperature 0.6), and nothing seeds sampling.
- **`max_new_tokens`** is the most new tokens a row gets, **`min_new_tokens`** holds EOS off until a row has
  that many, and **`eos_token_id`** is the id, or ids, a row stops on (by default the generation config's).
- **Options that would change the rows, the value or the calls are refused at compile**, by name: `num_beams`,
  `num_return_sequences`, `return_dict_in_generate`, `use_cache`, `prefill_chunk_size`, and `generate`'s own
  parameters such as `generation_config`.
- **Compile does not know the model's options.** A misspelled one, or one `generate` does not take, fails
  when its step runs, after the steps before it: `answer: generate rejected 'seed': The following
  `model_kwargs` are not used by the model: ['seed'] …`.
- **A generation reproduces to the bit only at one batch size**
  ([running](running-and-engines.md#when-two-runs-give-the-same-numbers)).

## Frames

A row's calls are the prefill, then one decode call per new token but the last. On row 0 of the document
below, "The Eiffel Tower is in the city of" is 10 tokens, positions 0 to 9, and generated token k sits at
10 + k:

| `pos` | calls | holds on row 0 |
|---|---|---|
| `-1`, the same as `{"frame": 0, "index": -1}` | the prefill | position 9, `Ġof`; its logits are the distribution over token 0 |
| `{"frame": k}`, k ≥ 1 | decode call k | one position, 9 + k: token k−1 as input; its logits are the distribution over token k |
| `{"frame": "1:"}` | every decode call | one position per call, ragged across rows that stop at different lengths |
| `{"frame": ":"}` | the prefill and every decode call | at the head, one distribution per generated token; at a stream site, every prompt token, then each generated input |
| `{"frame": -1, "index": -1}` | the row's last call | the call that chose the row's last token |

- **The row's last generated token is never an input**, so no call holds it.
- **The prefill computes the head at its last position only**; a stream site there holds every prompt token.
- **Keep `index: -1` with `frame: -1`.** On a row whose first token is its EOS the last call is the prefill,
  and `{"frame": -1}` alone is every prompt token.
- **`nnspec table` adds a `call` column** beside `pos` and `pos_token`.

## Reads at every call

Saved as `documents/generate_reads.json`:

```json
{
  "description": "Greedy generation on GPT-2 read at its calls: the first word's probability at each call, and each expected token's at the call that chose it. Also the calls each row made, the text and whether it holds the word, the rows the scope reached, and a generation stopped at its first comma.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"prompts": {"rows": [
    {"example_id": "eiffel", "input": "The Eiffel Tower is in the city of", "said": " Paris", "first": " Paris"},
    {"example_id": "golden", "input": "The Golden Gate Bridge is in the city of", "said": " San Francisco", "first": " San"},
    {"example_id": "kremlin", "input": "The Kremlin is in the city of", "said": " Moscow", "first": " Moscow"}
  ]}},
  "sites": {
    "head": {"component": "lm_head_output"},
    "resid": {"component": "layer_output", "layers": 6}
  },
  "steps": {
    "g": {"kind": "generate", "input": "prompts.input", "max_new_tokens": 6, "do_sample": false, "interventions": {
      "p_first": {"site": "head", "pos": {"frame": ":"}, "ops": [{"op": "token_prob", "token": "prompts.first"}]},
      "conf": {"site": "head", "pos": {"frame": "1:", "scope": {"column": "prompts.said", "before": true}},
               "ops": [{"op": "token_prob", "token": "prompts.said"}]},
      "calls": {"site": "resid", "pos": {"frame": "1:"}, "ops": [{"op": "count"}]}}},
    "text": {"kind": "op", "op": "detokenize", "of": "g"},
    "said": {"kind": "op", "op": "contains", "of": "text", "text": {"column": "prompts.said", "strip": true}},
    "conf_positions": {"kind": "op", "op": "count", "of": "g.conf"},
    "reached_row": {"kind": "op", "op": "gt", "of": "conf_positions", "with": 0},
    "rows_reached": {"kind": "op", "op": "sum", "of": "reached_row"},
    "to_comma": {"kind": "generate", "input": "prompts.input", "max_new_tokens": 6, "do_sample": false, "eos_token_id": 11,
                 "interventions": {"calls": {"site": "resid", "pos": {"frame": "1:"}, "ops": [{"op": "count"}]}}},
    "comma_text": {"kind": "op", "op": "detokenize", "of": "to_comma"}
  },
  "saves": ["g.p_first", "g.conf", "g.calls", "text", "said", "conf_positions", "rows_reached", "to_comma.calls", "comma_text"]
}
```

```text
$ nnspec table out/generate_reads g.p_first
example_id  row  pos  pos_token   call  input                                     first    value
eiffel      0    9    Ġof         0     The Eiffel Tower is in the city of         Paris   0.06997036188840866
eiffel      0    10   ĠParis      1     The Eiffel Tower is in the city of         Paris   0.00012037073611281812
...
$ nnspec table out/generate_reads text said
example_id  row  input                                     text                            said
eiffel      0    The Eiffel Tower is in the city of         Paris, and the Eiff            1.0
golden      1    The Golden Gate Bridge is in the city of   San Francisco, and the Golden  1.0
kremlin     2    The Kremlin is in the city of              Sochi, where it has been       0.0
$ nnspec table out/generate_reads g.calls conf_positions
example_id  row  input                                     g.calls  said            conf_positions  alignment_missing
eiffel      0    The Eiffel Tower is in the city of        5.0       Paris          1.0
golden      1    The Golden Gate Bridge is in the city of  5.0       San Francisco  2.0
kremlin     2    The Kremlin is in the city of             5.0       Moscow                         true
```

- **`p_first` at `{"frame": ":"}` is one probability per call**, `[3, 6]`: the prefill's last position, then
  each generated input.
- **`conf` reads the calls that chose each token of the expected word** (`before`): for ` San Francisco`, call
  0 chose `ĠSan` at 0.54 and call 1 `ĠFrancisco` at 0.90.
- **`count` of a read at `{"frame": "1:"}` is the decode calls each row made**: five for six new tokens.
- **Row 2 never says Moscow**, so `conf` is `alignment_missing` there. `rows_reached`, the `sum` of a `gt` of
  the `count`, is 2.0: the rows a mean over `conf` covers
  ([positions](positions.md#rows-a-scope-never-reached)).

The ops after an entry's last edit run at each call, on that call's `[rows, 1, ...]` value, up to the first op
that waits; the rest run once the batch's generation ends.

- **Positionwise ops run at each call**: a metric of the head (`p_first`) holds one number a call, and each
  call's logits are freed.
- **An op waits where it removes `pos`, takes an operand with a `pos` axis, or scores an answer of several
  tokens**: `count` in `calls`, `token_prob` of ` San Francisco` in `conf`.
- **A read that waits, or has no ops, is held whole until the generation ends**: at the head, `calls ×
  vocab_size` values a row. Score inside the entry, or lower `--batch-size`.
- **The manifest says which**: `per_call` in an entry's record counts the ops it runs at each call (1 for
  `p_first`, absent for `conf`).

## Rows that stop early

`to_comma` stops each row at its first comma, id 11:

```text
$ nnspec table out/generate_reads to_comma.calls comma_text
example_id  row  input                                     to_comma.calls  comma_text
eiffel      0    The Eiffel Tower is in the city of        1.0              Paris
golden      1    The Golden Gate Bridge is in the city of  2.0              San Francisco
kremlin     2    The Kremlin is in the city of             1.0              Sochi
```

- **The ids are as wide as the longest row**, `[3, 3]` here, filled past a shorter row's EOS.
- **`detokenize` stops before the EOS**, so `contains` never finds the EOS's own text. With `"tokens": true`
  it gives each row's token strings instead.
- **`plan.run.tokens["to_comma"]` holds each row's tokens up to and including its EOS**: `['ĠParis', ',']` on
  row 0.
- **A read at a call a row never made is `alignment_missing` there**, and a later write that takes it as an
  operand on that row is refused.

## Writes during generation

A write lands as its call runs. One at the prompt's last position changes token 0 and, through the cache,
everything after it; one at decode call k changes tokens k onward. These steps, in place of
`generate_reads.json`'s, keep a write on through the whole generation, and write where a row says a word:

```json
{
  "steps": {
    "plain": {"kind": "generate", "input": "prompts.input", "max_new_tokens": 6, "do_sample": false},
    "damped": {"kind": "generate", "input": "prompts.input", "max_new_tokens": 6, "do_sample": false, "interventions": {
      "prompt": {"site": "resid", "pos": {"frame": 0, "index": -1}, "ops": [{"op": "scale", "factor": 0.2}]},
      "decode": {"site": "resid", "pos": {"frame": "1:"}, "ops": [{"op": "scale", "factor": 0.2}]}}},
    "online": {"kind": "generate", "input": "prompts.input", "max_new_tokens": 6, "do_sample": false, "interventions": {
      "zero": {"site": "resid", "pos": {"frame": "1:", "scope": {"column": "prompts.first"}, "index": -1},
               "ops": [{"op": "scale", "factor": 0.0}]}}},
    "plain_text": {"kind": "op", "op": "detokenize", "of": "plain"},
    "damped_text": {"kind": "op", "op": "detokenize", "of": "damped"},
    "online_text": {"kind": "op", "op": "detokenize", "of": "online"}
  },
  "saves": ["plain_text", "damped_text", "online_text"]
}
```

```text
$ nnspec table --csv out/generate_writes plain_text damped_text online_text
example_id,row,input,plain_text,damped_text,online_text
eiffel,0,The Eiffel Tower is in the city of," Paris, and the Eiff", Central being-it's-," Paris in the Netherlands.
"
golden,1,The Golden Gate Bridge is in the city of," San Francisco, and the Golden"," and was for- the,", San. The bridge is located
kremlin,2,The Kremlin is in the city of," Sochi, where it has been"," (in the in""it"," Sochi, where it has been"
$ nnspec show out/generate_writes
  ...
  entry          at  positions
  damped.prompt  —   [9], [8], [6]
  damped.decode  —   [10,11,12,13,14], [9,10,11,12,13], [7,8,9,10,11]
  online.zero    —   [10], [9], —
```

- **A write kept on for the whole generation is two entries**: `{"frame": 0, "index": -1}`, which `pos: -1`
  also means, and `{"frame": "1:"}`. A single `{"frame": ":"}` also writes at every prompt token.
- **A write's positions must be known as its call runs.** Beyond the prefill it takes no frame counted from
  the row's last call, no index counted from the end over several calls, no `before` and no `until`; each is
  refused at compile with the form to write instead. A read takes all of them.
- **A write with a `column` or `text` scope over decode calls is found online**: it lands at the call fed the
  occurrence's last token, so the text changes from the next token on. It takes `index: -1` and an
  `occurrence` counted from the first.
- **A row that never says the text is `alignment_missing` in the write's positions**, and generates as if
  unwritten: row 2 above. A mean over rows then dilutes the effect, so count those rows as for a read.
- **An entry of a generate step takes no operand from another entry of its own step**; read the operand in
  an earlier step.

## Did it say the answer

Each row's reply is scored three ways: a substring test, a read at the answer as a whole word, and the
probability the model gave the answer's tokens as it chose them. Saved as `documents/generated_capitals.json`:

```json
{
  "description": "Did a chat model say the answer, where, and how sure was it: ten greedy tokens per row as the assistant, scored by a substring test with and without strip, the same test on a decoy (the null), a whole-word read, and the calls that chose the answer's tokens.",
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

`nnspec table out/generated_capitals text said_spaced said said_decoy word`, then `p`, then
`accuracy reached mean_p`, some rows left out:

```text
example_id  row  chat                                              text                                              answer  said_spaced  said  decoy   said_decoy  word  alignment_missing
france      0    [{"role":"user","content":"What is the capital …  The capital of France is Paris. It is a            Paris  1.0          1.0    Lyon   0.0         1.0
lyon        2    [{"role":"user","content":"Is Paris or Lyon the…  Paris and Lyon are indeed the capital cities of…   Paris  0.0          1.0    Lyon   1.0         1.0
spider      3    [{"role":"user","content":"How many legs does a…  The answer to this question is "eight legs."       eight  0.0          1.0    six    0.0         1.0
week        4    [{"role":"user","content":"How many days are in…  A week is approximately 31 days. However           seven  0.0          0.0    five   0.0               true
light       6    [{"role":"user","content":"What is the opposite…  The opposite of light is darkness. Darkness is     dark   1.0          1.0    heavy  0.0               true

example_id  row  pos  pos_token  call  chat                                              answer  value                alignment_missing
france      0    41   Ġis        5     [{"role":"user","content":"What is the capital …   Paris  0.9883716106414795
lyon        2    38   Ċ          0     [{"role":"user","content":"Is Paris or Lyon the…   Paris  0.2772394120693207
week        4                          [{"role":"user","content":"How many days are in…   seven                       true

accuracy            reached  mean_p
0.7142857313156128  4.0      0.5577137470245361
```

- **`at` reads the answer's last token where the reply says it as a whole word**, and `word`, its `count`
  over `pos`, is the whole-word score. The light row's reply says "darkness", so `said` is 1 and `word` is
  missing.
- **`strip` adds two rows**: without it 3 of 7 rows score (`said_spaced`), with it 5 of 7. The Lyon row's
  "Paris" opens the reply, and the spider row's "eight" follows a quote mark.
- **The null is `said_decoy`**: 0 on every row but Lyon, whose reply repeats the question's "Lyon".
- **`chose` reads the call that chose each of the answer's tokens** (`before`). For the Lyon row's opening
  "Paris" that is call 0, the prefill's last position (`Ċ`, the assistant header's newline).
- **Count before the mean.** `count` is 0 on a missed row, so `reached`, the `sum` of `word`, is 4, and
  `mean_p` is over those 4 of 7 rows: report "0.558 over 4 of 7 rows".
- **At `max_new_tokens` 8 the spider row's "eight" is its last generated token**: `said` and `p` still find
  it, since `before` also sees the last token, and `reached` drops to 3.

## The text after a marker, and the reasoning before it

`detokenize` with `after` keeps each row's text after the first occurrence of a marker, so `contains` scores
what follows it alone. A read with an `until` scope over the decode calls covers every generated input up to
and including the marker, and its `count` is the reasoning's length in tokens. `after_marker.json` asks
SmolLM2 six capital questions as chats, five of them ending "Think step by step, then give the final answer
after 'Answer:'.", with `resid` at block 0's `layer_output`:

```json
{
  "steps": {
    "answer": {"kind": "generate", "input": "qa.chat", "max_new_tokens": 64, "min_new_tokens": 64, "do_sample": false,
      "interventions": {"trace": {"site": "resid", "pos": {"frame": "1:", "scope": {"text": "Step 2:", "until": true}}}}},
    "text": {"kind": "op", "op": "detokenize", "of": "answer"},
    "final": {"kind": "op", "op": "detokenize", "of": "answer", "after": {"text": "Step 2:"}},
    "marked": {"kind": "op", "op": "contains", "of": "text", "text": {"text": "Step 2:"}},
    "said": {"kind": "op", "op": "contains", "of": "text", "text": {"column": "qa.answer", "strip": true}},
    "said_final": {"kind": "op", "op": "contains", "of": "final", "text": {"column": "qa.answer", "strip": true}},
    "steps_len": {"kind": "op", "op": "count", "of": "answer.trace"},
    "marked_len": {"kind": "op", "op": "div", "of": "steps_len", "with": "marked"},
    "mean_len": {"kind": "op", "op": "mean", "of": "marked_len"},
    "n_marked": {"kind": "op", "op": "count", "of": "marked_len", "over": ["rows"]}
  }
}
```

```text
$ nnspec table out/after_marker marked said said_final steps_len
example_id  row  chat                                              marked  answer     said  said_final  steps_len
france      0    [{"role":"user","content":"What is the capital …  1.0      Paris     1.0   1.0         32.0
turkey      3    [{"role":"user","content":"Is the capital of Tu…  1.0      Ankara    1.0   0.0         33.0
australia   4    [{"role":"user","content":"Is the capital of Au…  0.0      Canberra  0.0   0.0         63.0
spain       5    [{"role":"user","content":"What is the capital …  0.0      Madrid    1.0   1.0         63.0
```

- **`after` changes the Turkey row**: its Step 1 lists "Istanbul and Ankara" and its Step 2 settles on
  Istanbul, so `said` is 1 and `said_final` 0.
- **A row without the marker keeps its whole text**, so `said_final` equals `said` there. Score `marked`
  beside it to tell such rows from rows that answered after the marker.
- **An `until` scope never misses a row.** A row that never writes the marker counts every generated input, 63
  here, so a plain mean of `steps_len` mixes lengths with budgets.
- **`div` makes the unmarked rows holes**: it gives NaN where its operand is 0, and reductions skip NaN.
  `mean_len` is 32.0 over `n_marked` 4 rows.
- **`after` takes a column or a `{"text": ...}` literal**, compares as `contains` does, and is refused beside
  `"tokens": true`.

## A fixed answer, a whole reply, a chat

- **To score an answer whatever the model would have said**, of one token or several, or to compare two
  candidates, append it to the prompt and read the positions that predict it in one forward:
  [teacher forcing](generation-teacher-forcing.md), which also forces a whole assistant reply.
- **A reply to a chat needs the assistant's header.** With `"template": {"add_generation_prompt": true}` on
  the dataset the prompt ends with the header and the model answers ([data](data.md#chats)). Without it the
  model writes the header itself: on three chats of a question, its answer and a follow-up ("And of Japan?"),
  four of ten generated tokens are `<|im_start|>`, `ass`, `istant`, `Ċ`, `detokenize` keeps them, and a read at
  a fixed decode call lands in the header.
- **A position inside a turn is a `turn` scope** ([positions](positions.md#chat-turns)). In that run
  `{"scope": {"turn": "user", "occurrence": 1}, "index": -1}` lands at 52, the `?` of the follow-up, and
  `pos: -1` at 58, the header's newline; check both in `nnspec show`.

## Missing rows at each sweep point

In a sweep, count the rows a scoped read reached at each point: a steered generation reaches fewer as the
dose rises. With this entry added to the `steered` step of the steering guide's `steer_generation.json`:

```json
{"said": {"site": "resid", "pos": {"frame": "1:", "scope": {"column": {"column": "qa.answer", "strip": true}}, "index": -1}}}
```

each point's record lists the rows the read missed:

```python
from nnspec import Plan

plan = Plan.load("out/steer_generation")
for point in plan.steps["steered"]:
    at = {label: plan.sweeps[label].values[i] for label, i in point.at.items()}
    rows = point.interventions["said"].positions["qa"]
    print(at, [row for row, where in enumerate(rows) if where == "alignment_missing"])
```

```text
{'vector': 'steer', 'dose': 0.0} []
{'vector': 'steer', 'dose': 0.5} []
{'vector': 'steer', 'dose': 1.0} [1, 2, 3, 4, 5]
{'vector': 'steer', 'dose': 2.0} [1, 2, 3, 4, 5]
{'vector': 'random', 'dose': 0.0} []
{'vector': 'random', 'dose': 0.5} [4]
{'vector': 'random', 'dose': 1.0} [0, 1, 2, 3, 4, 5]
{'vector': 'random', 'dose': 2.0} [0, 1, 2, 3, 4, 5]
```

A point's `at` holds its index on each sweep axis, and `positions` is keyed by dataset. A mean over the read
under the direction at dose 1.0 is one row's number: report it as "over 1 of 6 rows", or not at all. Check a
scoped write's `positions` the same way, since a write that never lands leaves its row unsteered.
