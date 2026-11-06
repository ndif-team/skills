# Scoring generations

Recipes that score what a model wrote, beyond the skill's "did it say the answer". Every document runs on
`HuggingFaceTB/SmolLM2-135M-Instruct` on the CPU. Its tokenizer is byte-level (`Ġ` marks a space) and adds
no BOS, and its chat template writes a default system message when a chat has none.

## The text after a marker, and the reasoning before it

`contains` searches the whole text, so an answer named while reasoning counts as given. `detokenize` with
`after` keeps each row's text after the first occurrence of a marker. A read with an `until` scope over the
decode calls covers every generated input up to and including the marker, and its `count` is the length of
the reasoning in tokens:

```json
{
  "description": "Score what a generation says after a marker, and measure the reasoning before it. Asked to think step by step, the model writes 'Step 1: ...' and 'Step 2: ...'. `final` is each row's text after its first 'Step 2:' (the whole text on a row that never writes it), and `marked` says which rows wrote it. `said` searches the whole text and `said_final` only what follows the marker. `steps_len` is the count of generated inputs up to and including the marker's last token (a read with an `until` scope over the decode calls), every input on a row without it.",
  "model": {"key": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac", "dtype": "float32"},
  "data": {"qa": {"template": {"add_generation_prompt": true}, "rows": [
    {"example_id": "france", "chat": [{"role": "user", "content": "What is the capital of France? Think step by step, then give the final answer after 'Answer:'."}], "answer": " Paris"},
    {"example_id": "japan", "chat": [{"role": "user", "content": "What is the capital of Japan? Think step by step, then give the final answer after 'Answer:'."}], "answer": " Tokyo"},
    {"example_id": "italy", "chat": [{"role": "user", "content": "What is the capital of Italy? Think step by step, then give the final answer after 'Answer:'."}], "answer": " Rome"},
    {"example_id": "turkey", "chat": [{"role": "user", "content": "Is the capital of Turkey Istanbul or Ankara? Think step by step, then give the final answer after 'Answer:'."}], "answer": " Ankara"},
    {"example_id": "australia", "chat": [{"role": "user", "content": "Is the capital of Australia Sydney or Canberra? Think step by step, then give the final answer after 'Answer:'."}], "answer": " Canberra"},
    {"example_id": "spain", "chat": [{"role": "user", "content": "What is the capital of Spain?"}], "answer": " Madrid"}]}},
  "sites": {"resid": {"component": "layer_output", "layers": 0}},
  "steps": {
    "answer": {"kind": "generate", "input": "qa.chat", "max_new_tokens": 64, "min_new_tokens": 64, "do_sample": false,
      "interventions": {"trace": {"site": "resid", "pos": {"frame": "1:", "scope": {"text": "Step 2:", "until": true}}}}},
    "text": {"kind": "op", "op": "detokenize", "of": "answer"},
    "final": {"kind": "op", "op": "detokenize", "of": "answer", "after": {"text": "Step 2:"}},
    "marked": {"kind": "op", "op": "contains", "of": "text", "text": {"text": "Step 2:"}},
    "said": {"kind": "op", "op": "contains", "of": "text", "text": {"column": "qa.answer", "strip": true}},
    "said_final": {"kind": "op", "op": "contains", "of": "final", "text": {"column": "qa.answer", "strip": true}},
    "steps_len": {"kind": "op", "op": "count", "of": "answer.trace"}
  },
  "saves": ["text", "final", "marked", "said", "said_final", "steps_len"]
}
```

`clab table out/after_marker marked said said_final steps_len`:

```text
example_id  row  chat                                              marked  answer     said  said_final  steps_len
france      0    [{"role":"user","content":"What is the capital …  1.0      Paris     1.0   1.0         32.0
japan       1    [{"role":"user","content":"What is the capital …  1.0      Tokyo     1.0   1.0         31.0
italy       2    [{"role":"user","content":"What is the capital …  1.0      Rome      1.0   1.0         32.0
turkey      3    [{"role":"user","content":"Is the capital of Tu…  1.0      Ankara    1.0   0.0         33.0
australia   4    [{"role":"user","content":"Is the capital of Au…  0.0      Canberra  0.0   0.0         63.0
spain       5    [{"role":"user","content":"What is the capital …  0.0      Madrid    1.0   1.0         63.0
```

- **`after` changes the Turkey row.** Step 1 lists "Istanbul and Ankara" and Step 2 settles on Istanbul, so
  `said` is 1 and `said_final` 0.
- **The null is a row without the marker**: `after` keeps its whole text, so `said_final` equals `said`
  there (Australia, Spain). Score `marked` beside it to tell such rows from rows that answered after it.
- **An `until` scope never misses a row.** A row that never writes the marker counts every generated input:
  63, which is `max_new_tokens` less the last token, never fed. A mean of `steps_len` then mixes lengths with
  budgets.
- **Make the unmarked rows holes before a mean.** `div` gives NaN wherever its operand is 0, and reductions
  skip NaN. A fragment of the same document's `steps`:

```json
{
  "marked_len": {"kind": "op", "op": "div", "of": "steps_len", "with": "marked"},
  "mean_len": {"kind": "op", "op": "mean", "of": "marked_len"},
  "n_marked": {"kind": "op", "op": "count", "of": "marked_len", "over": ["rows"]}
}
```

`mean_len` is 32.0 over `n_marked` 4 rows; `marked_len` is NaN on Australia and Spain.

- **`after` takes a column or a literal** (`{"text": ...}`), compares as `contains` does, and cuts at the
  first occurrence. It is refused beside `tokens: true`.
- **The scope searches decoded generated text** and matches on token and word boundaries: `Step 2:` is the
  tokens `Step`, `Ġ`, `2`, `:` here.

## Teacher forcing a fixed answer

To score an answer whatever the model would have said, append it to the prompt and read the positions that
predict its tokens, in one forward. Here every prompt names both candidates, so each city occurs twice in the
forced input:

```json
{
  "description": "Teacher forcing: each prompt names two cities, and each city is appended to it in turn (a list input, joined per row with nothing between, so the answer's own leading space separates the words). The head is read at the positions that predict the appended city's tokens: a scope on the city with `occurrence: -1`, since the prompt already says it once, and `before`. `seq_*` sums each token's cross-entropy into the city's negative log-likelihood; `prefers_gold` is 1 where the right capital is the likelier; `exact` and `exact_other` are 1 where every token of that city was the top prediction. `said` is what the model generates from the same prompt. `first_gold` reads with the default `occurrence: 0`, which lands on the prompt's copy of the city.",
  "model": {"key": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac", "dtype": "float32"},
  "data": {"c": {"rows": [
    {"example_id": "australia", "input": "Is the capital of Australia Sydney or Canberra? The capital of Australia is", "gold": " Canberra", "other": " Sydney"},
    {"example_id": "canada", "input": "Is the capital of Canada Toronto or Ottawa? The capital of Canada is", "gold": " Ottawa", "other": " Toronto"},
    {"example_id": "turkey", "input": "Is the capital of Turkey Istanbul or Ankara? The capital of Turkey is", "gold": " Ankara", "other": " Istanbul"},
    {"example_id": "brazil", "input": "Is the capital of Brazil Rio or Brasilia? The capital of Brazil is", "gold": " Brasilia", "other": " Rio"},
    {"example_id": "argentina", "input": "Is the capital of Argentina Cordoba or Buenos Aires? The capital of Argentina is", "gold": " Buenos Aires", "other": " Cordoba"},
    {"example_id": "zealand", "input": "Is the capital of New Zealand Auckland or Wellington? The capital of New Zealand is", "gold": " Wellington", "other": " Auckland"}]}},
  "sites": {"head": {"component": "logits"}},
  "steps": {
    "gold": {"kind": "forward", "input": ["c.input", "c.gold"], "interventions": {
      "logits": {"site": "head", "pos": {"scope": {"column": "c.gold", "occurrence": -1, "before": true}}},
      "first": {"site": "head", "pos": {"scope": {"column": "c.gold", "before": true}}}}},
    "other": {"kind": "forward", "input": ["c.input", "c.other"], "interventions": {
      "logits": {"site": "head", "pos": {"scope": {"column": "c.other", "occurrence": -1, "before": true}}}}},
    "nll_gold": {"kind": "op", "op": "cross_entropy", "of": "gold.logits", "target": "c.gold"},
    "seq_gold": {"kind": "op", "op": "sum", "of": "nll_gold", "over": ["pos"]},
    "nll_other": {"kind": "op", "op": "cross_entropy", "of": "other.logits", "target": "c.other"},
    "seq_other": {"kind": "op", "op": "sum", "of": "nll_other", "over": ["pos"]},
    "prefers_gold": {"kind": "op", "op": "lt", "of": "seq_gold", "with": "seq_other"},
    "hit": {"kind": "op", "op": "match", "of": "gold.logits", "expected": "c.gold"},
    "exact": {"kind": "op", "op": "min", "of": "hit", "over": ["pos"]},
    "hit_other": {"kind": "op", "op": "match", "of": "other.logits", "expected": "c.other"},
    "exact_other": {"kind": "op", "op": "min", "of": "hit_other", "over": ["pos"]},
    "nll_first": {"kind": "op", "op": "cross_entropy", "of": "gold.first", "target": "c.gold"},
    "seq_first": {"kind": "op", "op": "sum", "of": "nll_first", "over": ["pos"]},
    "generated": {"kind": "generate", "input": "c.input", "max_new_tokens": 3, "min_new_tokens": 3, "do_sample": false},
    "said": {"kind": "op", "op": "detokenize", "of": "generated"}
  },
  "saves": ["seq_gold", "seq_other", "prefers_gold", "exact", "exact_other", "seq_first", "said", "nll_gold"]
}
```

`clab table out/forced_capitals seq_gold seq_other prefers_gold exact exact_other seq_first said`:

```text
example_id  row  input                                             gold           seq_gold             other      seq_other           prefers_gold  exact  exact_other  seq_first            said
australia   0    Is the capital of Australia Sydney or Canberra?…   Canberra      0.34803861379623413   Sydney    2.21793270111084    1.0           1.0    0.0           1.7569866180419922    Canberra.\n
canada      1    Is the capital of Canada Toronto or Ottawa? The…   Ottawa        1.2245898246765137    Toronto   0.5822196006774902  0.0           0.0    1.0           1.3168150186538696    Toronto.\n
turkey      2    Is the capital of Turkey Istanbul or Ankara? Th…   Ankara        1.4458869695663452    Istanbul  0.6227026581764221  0.0           0.0    1.0           0.6693405508995056    Istanbul, and
brazil      3    Is the capital of Brazil Rio or Brasilia? The c…   Brasilia      2.792855739593506     Rio       0.4169465899467468  0.0           0.0    1.0           3.598386526107788     Rio de Janeiro
argentina   4    Is the capital of Argentina Cordoba or Buenos A…   Buenos Aires  0.6178411245346069    Cordoba   4.082997798919678   1.0           1.0    0.0           0.46415406465530396   Buenos Aires.
zealand     5    Is the capital of New Zealand Auckland or Welli…   Wellington    1.7330877780914307    Auckland  0.6762439608573914  0.0           0.0    1.0           0.5200220942497253    Auckland.\n
```

`clab show out/forced_capitals` prints where each read landed:

```text
  entry         at  positions
  gold.logits   —   [13], [13], [14,15], [14,15], [15,16], [15]
  gold.first    —   [6], [6], [6,7], [6,7], [7,8], [7]
  other.logits  —   [13], [13], [14], [14], [15,16], [15]
```

- **The null is the city the model says.** Greedy decoding takes the top token at every step. So the city it
  generated, forced, is the top prediction at every token: `exact` or `exact_other` (`min` of `match` over
  `pos`, 1 only where every token matched) is 1 on every row, the one `said` names.
- **The model prefers the right capital on 2 of 6 rows**, and on each row it says the city it prefers.
- **Keep `occurrence: -1`.** The default `occurrence: 0` picks the prompt's copy: `gold.first` lands at 6 and 7,
  inside the question, where `gold.logits` lands at 13 to 16. It runs without a warning, and the Australia
  row's negative log-likelihood is then 1.757 instead of 0.348.
- **The sequence score of a forced answer is `cross_entropy` summed over `pos`**, its negative
  log-likelihood (`seq_gold`). `token_prob` and `cross_entropy` give one value per token: ` Ankara`,
  ` Brasilia` and ` Buenos Aires` are two tokens, and `clab table out/forced_capitals nll_gold` prints two
  lines on those rows.
- **The appended answer needs its leading space** (` Canberra`), since a list input joins with nothing
  between. Without it the answer joins the prompt's last word: `isCanberra`, tokens `Can`, `berra`. That is
  no whole-word occurrence, so `occurrence: -1` finds the prompt's copy instead. The read lands at 6 and 7,
  as `gold.first` does, and nothing is refused. Where the prompt does not name the answer, the scope refuses it.
- **A list input takes text columns and `{"text": ...}` texts only.** A chat column there is refused:

```text
clab: steps.f.input[0]: 'qa.chat' is not text on row 0 of 'qa'; a list input joins text
```

### Forcing an assistant reply

Teacher forcing a chat means writing the prompt out as the template renders it, as text, since a list input
refuses a chat column. Append the reply to that text, and check that the text and the chat give the same logits:

```json
{
  "description": "Teacher-force an assistant reply. A list input joins text, and a chat column is not text, so the prompt is written as the template renders it (`text`, the default system message included). `c` and `t` read the last prompt token from the chat and from the text; `agree` is 1 where every logit is equal. `forced` appends each row's reply and reads the positions that predict it: row 0's is the reply the model generates greedily, row 1's names another city. `hit` is 1 where a token was the top prediction, and `seq`, the cross-entropy summed over `pos`, is the reply's negative log-likelihood.",
  "model": {"key": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac", "dtype": "float32"},
  "data": {"qa": {"template": {"add_generation_prompt": true}, "rows": [
    {"example_id": "paris", "chat": [{"role": "user", "content": "What is the capital of France?"}],
     "text": "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
     "reply": "The capital of France is Paris."},
    {"example_id": "lyon", "chat": [{"role": "user", "content": "What is the capital of France?"}],
     "text": "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
     "reply": "The capital of France is Lyon."}]}},
  "sites": {"head": {"component": "logits"}},
  "steps": {
    "c": {"kind": "forward", "input": "qa.chat", "interventions": {"l": {"site": "head", "pos": -1}}},
    "t": {"kind": "forward", "input": "qa.text", "interventions": {"l": {"site": "head", "pos": -1}}},
    "forced": {"kind": "forward", "input": ["qa.text", "qa.reply"], "interventions": {
      "l": {"site": "head", "pos": {"scope": {"column": "qa.reply", "occurrence": -1, "before": true}}}}},
    "same": {"kind": "op", "op": "eq", "of": "c.l", "with": "t.l"},
    "agree": {"kind": "op", "op": "min", "of": "same", "over": ["rows", "pos", "vocab_size"]},
    "hit": {"kind": "op", "op": "match", "of": "forced.l", "expected": "qa.reply"},
    "nll": {"kind": "op", "op": "cross_entropy", "of": "forced.l", "target": "qa.reply"},
    "seq": {"kind": "op", "op": "sum", "of": "nll", "over": ["pos"]}
  },
  "saves": ["agree", "hit", "seq"]
}
```

`clab table out/forced_reply agree` prints `1.0`; `clab table out/forced_reply seq`:

```text
example_id  row  text                                              reply                            value
paris       0    <|im_start|>system\nYou are a helpful AI assist…  The capital of France is Paris.  0.5047430992126465
lyon        1    <|im_start|>system\nYou are a helpful AI assist…  The capital of France is Lyon.   11.343289375305176
```

- **`agree` is 1.0**: the text renders to the chat's tokens, and the logits are equal to the bit. Compare
  `plan.run.tokens["qa.chat"]` with `plan.run.tokens["qa.text"]` after a run to check another template.
- **The null is the greedy reply**: `hit` is 1.0 at all 7 positions of row 0, from the header's `Ċ` at 36 to
  `ĠParis` at 42. On row 1 it is 0.0 at 41, where the model predicts Paris and the reply says Lyon.
- **`seq` scores the whole reply**: 0.50 for the reply the model gives, 11.34 for the Lyon reply.
- **The reply has no leading space**: its first word follows the header's newline.
- **The text holds the template's default system message**, which the chat never states. Render it outside
  clab with the tokenizer's `apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`,
  which gives this row's `text` exactly, never from memory of the template.

## Scoring a reply to a chat

A chat column, `template`, `add_generation_prompt` and the system message a template writes are the `clab`
skill's [data reference](../../clab/references/data.md#chats), and the `turn` scope its
[positions reference](../../clab/references/positions.md#chat-turns). This document scores the reply to a
follow-up question, with and without the assistant's header:

```json
{
  "description": "A follow-up question in a two-turn chat. `asked` sets `add_generation_prompt` and `closed` does not, so on `closed` the model writes the assistant's header itself. `follow_up` reads layer 12 at the last token of the second user message, `reply` at the prompt's last token. `said` and `closed_said` ask whether each reply names the second country's capital.",
  "model": {"key": "HuggingFaceTB/SmolLM2-135M-Instruct", "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac", "dtype": "float32"},
  "data": {
    "asked": {"template": {"add_generation_prompt": true}, "rows": [
      {"example_id": "japan", "answer": " Tokyo", "chat": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}, {"role": "user", "content": "And of Japan?"}]},
      {"example_id": "spain", "answer": " Madrid", "chat": [{"role": "user", "content": "What is the capital of Italy?"}, {"role": "assistant", "content": "The capital of Italy is Rome."}, {"role": "user", "content": "And of Spain?"}]},
      {"example_id": "germany", "answer": " Berlin", "chat": [{"role": "user", "content": "What is the capital of Egypt?"}, {"role": "assistant", "content": "The capital of Egypt is Cairo."}, {"role": "user", "content": "And of Germany?"}]}]},
    "closed": {"rows": [
      {"example_id": "japan", "answer": " Tokyo", "chat": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}, {"role": "user", "content": "And of Japan?"}]},
      {"example_id": "spain", "answer": " Madrid", "chat": [{"role": "user", "content": "What is the capital of Italy?"}, {"role": "assistant", "content": "The capital of Italy is Rome."}, {"role": "user", "content": "And of Spain?"}]},
      {"example_id": "germany", "answer": " Berlin", "chat": [{"role": "user", "content": "What is the capital of Egypt?"}, {"role": "assistant", "content": "The capital of Egypt is Cairo."}, {"role": "user", "content": "And of Germany?"}]}]}
  },
  "sites": {"resid": {"component": "layer_output", "layers": 12}},
  "steps": {
    "answer": {"kind": "generate", "input": "asked.chat", "max_new_tokens": 10, "min_new_tokens": 10, "do_sample": false,
      "interventions": {"follow_up": {"site": "resid", "pos": {"scope": {"turn": "user", "occurrence": 1}, "index": -1}},
                        "reply": {"site": "resid", "pos": -1}}},
    "closed_answer": {"kind": "generate", "input": "closed.chat", "max_new_tokens": 10, "min_new_tokens": 10, "do_sample": false},
    "text": {"kind": "op", "op": "detokenize", "of": "answer"},
    "closed_text": {"kind": "op", "op": "detokenize", "of": "closed_answer"},
    "said": {"kind": "op", "op": "contains", "of": "text", "text": {"column": "asked.answer", "strip": true}},
    "closed_said": {"kind": "op", "op": "contains", "of": "closed_text", "text": {"column": "closed.answer", "strip": true}}
  },
  "saves": ["text", "closed_text", "said", "closed_said"]
}
```

`clab table out/chat_turns text said`, then `closed_text closed_said`, then `clab show out/chat_turns`:

```text
example_id  row  chat                                              text                                       answer   said
japan       0    [{"role":"user","content":"What is the capital …  The capital of Japan is Tokyo. \n\nPlease   Tokyo   1.0
example_id  row  chat                                              closed_text                                       answer   closed_said
japan       0    [{"role":"user","content":"What is the capital …  <|im_start|>assistant\nThe capital of Japan is …   Tokyo   1.0

  entry             at  positions
  answer.follow_up  —   [52] on all 3 rows
  answer.reply      —   [58] on all 3 rows
```

- **Without the header the model writes it**: 4 of `closed`'s 10 tokens are `<|im_start|>`, `ass`, `istant`
  and `Ċ`, and `detokenize` keeps them, so a read at a fixed decode call lands in the header.
- **Check a turn's positions in `clab show`**: `follow_up` lands at 52, the `?` of "And of Japan?", and `reply`
  at 58, the header's newline.
- **A table joins one dataset's values.** Tabling `said` beside `closed_said` is refused: `'said' has a row
  per example of 'asked' and 'closed_said' of 'closed'; a table joins the values of one dataset`. Table them
  apart.

## Counting missing rows per sweep point

A read at generated text is `alignment_missing` on a row that never says the text. How that passes through
ops and tables, and the count inside a document, are the `clab` skill's
[positions reference](../../clab/references/positions.md#rows-a-scope-never-reached). In a sweep, count each
point: a steered generation reaches fewer rows as the dose rises. A read added to the `steered` step of the
skill's `steer_generation.json` (a fragment of its `interventions`):

```json
{"said": {"site": "resid", "pos": {"frame": "1:", "scope": {"column": {"column": "qa.answer", "strip": true}}, "index": -1}}}
```

Saved as `steer_said.json`, each point's record lists the rows that read missed:

<!-- test: skip -->
```python
from clab import Plan

plan = Plan.load("out/steer_said")
for point in plan.steps["steered"]:
    rows = point.interventions["said"].positions["qa"]
    dose = plan.sweeps["dose"].values[point.at["dose"]]
    print(dose, [row for row, where in enumerate(rows) if where == "alignment_missing"])
```

```text
0.0 []
0.5 []
1.0 [1, 2, 3, 4, 5]
2.0 [1, 2, 3, 4, 5]
```

- **A mean over that read at dose 1.0 is one row's number.** Report it as "over 1 of 6 rows", or not at all.
- **`positions` is keyed by dataset** (`"qa"` here).
- **A write over decode calls that never lands** leaves that row unsteered, and a mean over rows dilutes the
  effect. Check a scoped write's `positions` the same way.
