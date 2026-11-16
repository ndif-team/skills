# Teacher forcing

To score an answer whatever the model would have said, append it to the prompt and read the positions that
predict its tokens, in one forward. Use it for an answer of several tokens, to compare two candidates, or to
score a whole reply. It is depth for the [generation guide](generation.md), and it needs no `generate` step.

## What breaks

- **The default `occurrence` lands on the prompt's own copy of the answer** when the prompt names it, with no
  refusal. Scope the appended copy with `"occurrence": -1`.
- **An answer appended without its leading space joins the prompt's last word** (`isCanberra`). The scope then
  finds the prompt's copy and the read lands there, unrefused. A metric of it is refused where the bare word
  tokenizes to another count (`'Canberra' is 2 tokens; the read has 1 position on that row; …`).
- **Check the read's positions in `nnspec show`** on a row of each kind: they must follow the prompt.

## A fixed answer

A list input joins its parts per row with nothing between ([data](data.md#inputs)). `forced_capitals.json`
runs SmolLM2 on text rows such as `{"input": "Is the capital of Australia Sydney or Canberra? The capital of
Australia is", "gold": " Canberra", "other": " Sydney"}`, so each city occurs twice in the forced input; its
`head` site is `logits`:

```json
{
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
  }
}
```

```text
$ nnspec table out/forced_capitals seq_gold seq_other prefers_gold exact exact_other said
example_id  row  input                                             gold           seq_gold             other      seq_other           prefers_gold  exact  exact_other  said
australia   0    Is the capital of Australia Sydney or Canberra?…   Canberra      0.34803861379623413   Sydney    2.21793270111084    1.0           1.0    0.0           Canberra.\n
canada      1    Is the capital of Canada Toronto or Ottawa? The…   Ottawa        1.2245898246765137    Toronto   0.5822196006774902  0.0           0.0    1.0           Toronto.\n
turkey      2    Is the capital of Turkey Istanbul or Ankara? Th…   Ankara        1.4458869695663452    Istanbul  0.6227026581764221  0.0           0.0    1.0           Istanbul, and
...
$ nnspec show out/forced_capitals
  entry         at  positions
  gold.logits   —   [13], [13], [14,15], [14,15], [15,16], [15]
  gold.first    —   [6], [6], [6,7], [6,7], [7,8], [7]
```

- **`before` with `occurrence: -1` reads the positions that predict the appended answer's tokens**, one per
  token: ` Ankara` is two.
- **The sequence score is `cross_entropy` summed over `pos`**, the answer's negative log-likelihood;
  `prefers_gold` compares two of them with `lt`.
- **`min` of `match` over `pos` is exactness**: 1 only where every token was the top prediction.
- **The null is the city the model says.** Greedy decoding takes the top token at every step, so the city it
  generated, forced, is exact: `exact` or `exact_other` is 1 on every row, the one `said` names.
- **`first` shows the wrong occurrence**: it lands inside the question, and the Australia row's score is then
  1.757 (`seq_first`) instead of 0.348.

## An assistant reply

A list input refuses a chat column (`'qa.chat' is not text on row 0 of 'qa'; a list input joins text`), so
write the prompt out as the template renders it, in a text column, append the reply to that, and check that
the text and the chat give the same logits. `forced_reply.json` has rows of a `chat`, its rendering `text`
and a `reply`, under `"template": {"add_generation_prompt": true}`, with `head` at `logits`:

```json
{
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
  }
}
```

For the chat "What is the capital of France?", `text` is `"<|im_start|>system\nYou are a helpful AI assistant
named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nWhat is the capital of
France?<|im_end|>\n<|im_start|>assistant\n"`, and three replies give:

```text
$ nnspec table out/forced_reply seq
example_id  row  text                                              reply                            value
paris       0    <|im_start|>system\nYou are a helpful AI assist…  The capital of France is Paris.  0.5047414898872375
lyon        1    <|im_start|>system\nYou are a helpful AI assist…  The capital of France is Lyon.   11.34328556060791
bare        2    <|im_start|>system\nYou are a helpful AI assist…  Paris                            3.183110237121582
```

- **`agree` is 1.0**: the text renders to the chat's tokens, and the logits are equal to the bit.
- **Render the text with the tokenizer**, `apply_chat_template(messages, tokenize=False,
  add_generation_prompt=True)`, never from memory: this template writes a system message the chat does not
  state.
- **The null is the greedy reply**: `hit` is 1.0 at all 7 positions of row 0, the reply the model generates.
  On row 1 it is 0.0 from the position that predicts `Lyon`.
- **A bare one-word reply scores a reply the model does not give.** It answers in sentences, so `hit` on row
  2's `Paris` is 0.0: expected, not a wiring fault.
- **The reply has no leading space**, since its first word follows the header's newline. A generation's
  answer column keeps its space (` Tokyo`), so a document that does both holds two columns.
