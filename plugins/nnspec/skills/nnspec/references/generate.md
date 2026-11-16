# Generation

A `generate` step runs the model's `generate` on every row of its `input`. Its value is the new token ids,
`[rows, new]`; `detokenize` makes them text. Its entries read and write at the forward calls their `pos`'s
`frame` names: call 0 is the prefill, call k ≥ 1 decode call k. Scoring what a generation says, steering it and
generating from chats are the `nnspec-steering-and-generation` skill's; this page is the mechanics.

## Options

Every key but `kind`, `input` and `interventions` goes to the engine's `generate` unchanged, and may sweep.

- **`do_sample: false` on every generate step.** An option the step leaves out comes from the checkpoint's generation config, and nothing seeds sampling. `meta-llama/Llama-3.2-1B`'s config sets `"do_sample": true` and `"temperature": 0.6`; GPT-2's sets neither.
- **`max_new_tokens`** is the most new tokens a row gets; a write over a frame that runs to the row's end (`"2:"`) needs it.
- **`min_new_tokens`** holds EOS off until a row has that many; equal to `max_new_tokens`, every row makes the same calls.
- **`eos_token_id`** is the id, or ids, a row stops on; without it, the generation config's.
- **Options that would change the rows, the value or the calls are refused at compile**, by name: `num_beams` and `num_return_sequences` among them (`steps.plain: num_beams 2; it stays 1: beams change the rows each forward call sees, …`).
- **A parameter of `generate` itself is refused**: `'generation_config' is a parameter of generate itself, …`; so is a value of the wrong type, `"max_new_tokens": "6"`.
- **There is no `seed` option**; one the engine rejects fails when the step runs, naming the step.
- **A generation repeats only at one batch size**, as a forward does ([running and engines](running-and-engines.md#when-two-runs-give-the-same-numbers)).

## Frames

A row's calls are the prefill, then one decode call per new token but the last. On row 0 of the document below,
"The Eiffel Tower is in the city of" is 10 tokens, positions 0 to 9, and generated token k sits at 10 + k:

| `pos` | calls | holds on row 0 |
|---|---|---|
| `-1`, the same as `{"frame": 0, "index": -1}` | the prefill | position 9, `Ġof`; its logits are the distribution over token 0 |
| `{"frame": k}`, k ≥ 1 | decode call k | one position, 9 + k: token k−1 as input; its logits are the distribution over token k |
| `{"frame": "1:"}` | every decode call | one position per call, ragged across rows that stop at different lengths |
| `{"frame": ":"}` | the prefill and every decode call | at the head, one distribution per generated token; on a stream site, every prompt token then each generated input |
| `{"frame": -1, "index": -1}` | the row's last call | the call that chose the row's last token |

- **The row's last generated token is never an input**, so no call holds it.
- **The prefill computes the head at its last position only**: `{"frame": 0, "index": -2}` at the head is refused, and the same `pos` at a stream site reads.
- **Keep `index: -1` with `frame: -1`**: on a row whose first token was EOS the last call is the prefill, and `{"frame": -1}` alone is every prompt token.
- **`nnspec table` adds a `call` column** beside `pos` and `pos_token` in a generation.

## Reads at every call

`documents/generate_reads.json` reads three greedy generations at their calls:

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
eiffel      0    9    Ġof         0     The Eiffel Tower is in the city of         Paris   0.0699695572257042
eiffel      0    10   ĠParis      1     The Eiffel Tower is in the city of         Paris   0.00012037013948429376
eiffel      0    11   ,           2     The Eiffel Tower is in the city of         Paris   0.009360282681882381
...
kremlin     2    6    Ġof         0     The Kremlin is in the city of              Moscow  0.021569373086094856
kremlin     2    7    ĠSochi      1     The Kremlin is in the city of              Moscow  3.3157612051581964e-05
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
$ nnspec table out/generate_reads rows_reached
value
2.0
```

- **`p_first` at `{"frame": ":"}` is one probability per call**, `[3, 6]`: the prefill's last position, then each generated input.
- **`conf` reads the calls that chose each token of the expected word**: for ` San Francisco`, call 0 chose `ĠSan` at 0.54 and call 1 `ĠFrancisco` at 0.90. Its table is in [positions](positions.md#rows-a-scope-never-reached).
- **`count` of a read at `{"frame": "1:"}` is the decode calls each row made**: five for six new tokens.
- **Row 2 never says Moscow**, so `conf` is `alignment_missing` there, and `rows_reached` counts the two rows it reached, as [positions](positions.md#rows-a-scope-never-reached) explains.

### Where the ops run

The ops after an entry's last edit run at each call on that call's `[rows, 1, ...]` value up to the first that
waits, then once the batch's generation ends.

- **Positionwise ops run at each call**: a metric of the head (`token_prob` in `p_first`) holds one number a call, and each call's logits are freed.
- **An op waits from the first one that removes `pos`, takes an operand with a `pos` axis, or scores an answer of several tokens**: `count` in `calls`, and `token_prob` of ` San Francisco` in `conf`.
- **The record says which**: `per_call` in the manifest's `steps` counts the ops an entry runs at each call (1 for `p_first`, absent for `conf`).
- **A read that waits is held whole until the generation ends**: a head read with no ops holds `calls × vocab_size` values a row. Score it inside the entry, or lower `--batch-size`.

## Rows that stop early

`to_comma` stops each row at its first comma, id 11:

```text
$ nnspec table out/generate_reads to_comma.calls comma_text
example_id  row  input                                     to_comma.calls  comma_text
eiffel      0    The Eiffel Tower is in the city of        1.0              Paris
golden      1    The Golden Gate Bridge is in the city of  2.0              San Francisco
kremlin     2    The Kremlin is in the city of             1.0              Sochi
```

- **The ids keep the shape `[rows, max_new_tokens]`**, filler past each row's EOS; `detokenize` stops before the EOS, so `contains` never finds its text.
- **`plan.run.tokens["to_comma"]` holds each row's tokens up to and including its EOS**: `['ĠParis', ',']` on row 0.
- **A read at a call a row never made is `alignment_missing` there**, and has no value as a later write's operand; hold EOS off with `min_new_tokens` first.
- **A row held past its EOS is a different generation** from there, not the first one extended.

## Writes during generation

A write lands as its call runs. One at the prompt's last position changes token 0 and, through the cache,
everything after it; one at decode call k changes tokens k onward. `documents/generate_writes.json` keeps a write
on through the whole generation, and writes where a row says a word:

```json
{
  "description": "Writes during a greedy generation on GPT-2: block 6's output scaled down at the prompt's last token and every decode call, and zeroed at the call fed each row's first word, found as the row generates it.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"prompts": {"rows": [
    {"example_id": "eiffel", "input": "The Eiffel Tower is in the city of", "first": " Paris"},
    {"example_id": "golden", "input": "The Golden Gate Bridge is in the city of", "first": " San"},
    {"example_id": "kremlin", "input": "The Kremlin is in the city of", "first": " Moscow"}
  ]}},
  "sites": {"resid": {"component": "layer_output", "layers": 6}},
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

- **A write kept on for the whole generation is two entries**: `{"frame": 0, "index": -1}` and `{"frame": "1:"}`. `damped.decode` lands on every generated input, the last token excepted, which is never fed.
- **A write's positions must be known as its call runs.** `{"frame": ":", "index": -1}` is refused: `frame ':' with index -1 counts back from the row's last position, known only when the row stops, and a write lands as each call runs; for the prompt's last positions and every decode call, write two entries: …`.
- **No frame counted from the row's last call, no `before`, no `until` beyond the prefill**, for the same reason; a read takes all of them.
- **A write `{"frame": ":"}` covers every call**: every prompt token and every generated input.
- **A write with a `column` or `text` scope over decode calls is found online**: it lands at the call fed the occurrence's last token, so the text changes from the next token on. `online.zero` lands where rows 0 and 1 say their word.
- **A row that never says the text is `alignment_missing` in the write's positions**, and generates as if unwritten: row 2 matches `plain_text`.
- **An online write takes `index: -1` and an `occurrence` counted from the first**; any other index, `occurrence: -1` and `before` are refused by name.
- **A generation entry cannot take an operand from an entry of the same step**; read it in an earlier step.

## The first generated token's space

A SentencePiece tokenizer drops the leading space of a row's first generated token when decoding, and
`detokenize` and a scope over generated text decode that way. GPT-2's byte-level tokenizer keeps it: `text` above
starts ` Paris`.

- **An answer written with its space is then not found when it is token 0**: `contains` scores 0, and a scope over the decode calls finds nothing.
- **Give the column `strip: true` wherever it searches generated text**: `{"column": "prompts.said", "strip": true}`, in `contains` and in a scope alike. It costs nothing on GPT-2.
- **`contains` is a substring test**: it finds a word inside a longer one, where a scope matches on word boundaries only.
- **A word ending on the row's last generated token** is `alignment_missing` to a scope without `before`; leave room after the answer.

Counting the rows a generation never reached is in [positions](positions.md#rows-a-scope-never-reached).
