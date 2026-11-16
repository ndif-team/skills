# Data and inputs

The `data` block names a document's datasets. Each is a list of rows, each row a JSON object. A step's `input`
names the column it runs over, one row per example; other fields name columns as answers to score or as text to
find. `nnspec describe records` prints every field of `data.<name>`.

## Where rows come from

A dataset has exactly one source:

| source | holds | note |
|---|---|---|
| `"rows": [...]` | the rows, inline | they travel in the plan and the manifest, so a run's table reads them back from anywhere |
| `"path": "ioi/train"` | `data/ioi/train.json` under the corpus root, a JSON list of row objects | the corpus root is in [where output goes](running-and-engines.md#where-output-goes) |
| `"hf": "<owner>/<name>"` | a hub dataset at a required `revision` (a commit), with `config` and `split` | offline, it loads from the `datasets` cache one online load filled |

```json
{"ioi": {"path": "ioi/train", "split": "train", "take": 100},
 "glue": {"hf": "nyu-mll/glue", "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "config": "wnli", "split": "validation", "take": 3}}
```

- **A file is read again when you table the run**, from `data/` under the working directory, and rows that changed since the run are refused. Run `nnspec table` from the corpus root.
- **A hub dataset's rows never enter the plan**; the run records their digest and refuses rows whose digest differs.

## Narrowing and renaming

Five fields apply in this order: `split`, `shuffle`, `skip`, `take`, `columns`.

| field | does |
|---|---|
| `split` | a file: keeps rows whose `split` column equals it; the hub: which split. Refused on inline rows |
| `shuffle: true` | one fixed order, the same on every machine, depending on the row count alone |
| `skip: N`, `take: N` | drops the first N rows of that order, then keeps the first N left; after `shuffle`, a random sample |
| `columns` | renames (`{"new": "old"}`) or derives (`{"new": {"from": ..., "after"?, "before"?, "strip"?}}`) |

`documents/columns.json` holds the four IOI prompts of [SKILL.md](../SKILL.md#a-complete-experiment) as inline
rows with the columns `example_id` and `text`. Beside `"rows"`, its dataset `pairs` shuffles them, skips one,
keeps two, renames `text` to `prompt` and cuts the second subject out of the prompt as a column:

```json
{"shuffle": true, "skip": 1, "take": 2,
 "columns": {
   "prompt": "text",
   "giver": {"from": "text", "after": {"text": ", "}, "before": {"text": " gave"}}}}
```

A read at `{"scope": {"column": "pairs.giver", "occurrence": -1}}` of `pairs.prompt` then tables:

```text
example_id  row  pos  pos_token  prompt                                            value
paul        0    9    ĠLisa      When Paul and Lisa went to the school, Lisa gav…  63.120601654052734
sarah       1    9    ĠJames     When Sarah and James went to the office, James …  58.88503646850586
```

- **`row` counts the rows after narrowing**, not the source's order. `example_id`, where the rows have one, leads every table row: give rows one before you shuffle or take.
- **`take` without `shuffle` keeps the first N rows**, which on a sorted file is not a sample.
- **A renamed column answers to its new name only**, and a rename into a name the steps already read changes what they read, unrefused: check `run.tokens` after renaming.
- **A derived column cuts text**: `after` keeps what follows the first occurrence of its text, `before` what precedes it, `strip` drops surrounding whitespace. Each cut is `{"text": ...}`, and a row that lacks the text is refused by row. `giver` is `Lisa` with no leading space; a scope still finds `ĠLisa`, since a leading space is optional one way ([positions](positions.md#scopes)).

## Inputs

| `input` | each row's prompt |
|---|---|
| `"pairs.input"` | the column's text, or its chat |
| `"pairs.counterfactual_inputs[0]"` | entry 0 of a list column; every row needs one |
| `["facts.input", "facts.answer"]` | the parts joined with nothing between: teacher forcing |
| `["facts.input", {"text": " and"}, "facts.answer"]` | a fixed text among the parts |

- **Text is tokenized as the tokenizer does by default**, with BOS where it adds one; the run records every input's tokens under its name in `run.tokens`.
- **Nothing goes between the parts of a list.** A column without its leading space joins the previous word, which on GPT-2 is other tokens: put the space in the rows, or a `{"text": " "}` part between.
- **All columns of a list come from one dataset**, and a chat column in a list is refused, since a list joins text.
- **A counterfactual belongs in its base rows' dataset**, as a second column: a value with a rows axis belongs to its dataset, and another dataset's rows refuse it ([operands](sites-and-interventions.md#operands)).

## Answer columns

Where a field takes an answer or a text to find (a metric's answer, `contains`'s text, a scope's `column`),
write `"<dataset>.<column>"`, a reference with flags, or a literal:

| written | means |
|---|---|
| `"facts.first"` | each row's value of the column, tokenized exactly as written, without special tokens |
| `{"column": "facts.cities", "strip": true, "lower": true}` | the same, compared with whitespace optional and lowercased |
| a row value that is a list, `[" Paris", " London"]` | a set: any member is the answer |
| `{"text": " the"}` | the same literal on every row, under the same flags |

`documents/answers.json` scores one prompt's next token five ways, then teacher-forces the whole answer:

```json
{
  "description": "How an answer column becomes tokens on GPT-2's byte-level tokenizer: the first word with and without its space, a set with and without strip and lower, and a literal. Then the whole answer appended to the prompt, scored token by token.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"facts": {"rows": [
    {"example_id": "eiffel", "input": "The Eiffel Tower is in the city of", "answer": " Paris",
     "first": " Paris", "first_bare": "Paris", "cities": [" Paris", " London"]},
    {"example_id": "golden", "input": "The Golden Gate Bridge is in the city of", "answer": " San Francisco",
     "first": " San", "first_bare": "San", "cities": [" San", " Los"]}
  ]}},
  "sites": {"head": {"component": "lm_head_output"}},
  "steps": {
    "ask": {"kind": "forward", "input": "facts.input", "interventions": {"logits": {"site": "head", "pos": -1}}},
    "p_first": {"kind": "op", "op": "token_prob", "of": "ask.logits", "token": "facts.first"},
    "p_bare": {"kind": "op", "op": "token_prob", "of": "ask.logits", "token": "facts.first_bare"},
    "p_set": {"kind": "op", "op": "token_prob", "of": "ask.logits", "token": "facts.cities"},
    "p_loose": {"kind": "op", "op": "token_prob", "of": "ask.logits", "token": {"column": "facts.cities", "strip": true, "lower": true}},
    "p_the": {"kind": "op", "op": "token_prob", "of": "ask.logits", "token": {"text": " the"}},
    "forced": {"kind": "forward", "input": ["facts.input", "facts.answer"], "interventions": {
      "logits": {"site": "head", "pos": {"scope": {"column": "facts.answer", "occurrence": -1, "before": true}}}}},
    "nll": {"kind": "op", "op": "cross_entropy", "of": "forced.logits", "target": "facts.answer"}
  },
  "saves": ["p_first", "p_bare", "p_set", "p_loose", "p_the", "nll"]
}
```

```text
$ nnspec table out/answers p_first p_bare p_set p_loose p_the
example_id  row  pos  pos_token  input                                     first   p_first              first_bare  p_bare                  cities                p_set                p_loose             p_the
eiffel      0    9    Ġof        The Eiffel Tower is in the city of         Paris  0.06997019797563553  Paris       1.0362419970988412e-06  [" Paris"," London"]  0.12832464277744293  0.1283280849456787  0.00387186580337584
golden      1    8    Ġof        The Golden Gate Bridge is in the city of   San    0.5367922186851501   San         4.941660336044151e-06   [" San"," Los"]       0.6203228831291199   0.6203514337539673  0.001582340570166707
$ nnspec table out/answers nll
example_id  row  pos  pos_token  input                                     answer          value
eiffel      0    9    Ġof        The Eiffel Tower is in the city of         Paris          2.6596858501434326
golden      1    8    Ġof        The Golden Gate Bridge is in the city of   San Francisco  0.6221428513526917
golden      1    9    ĠSan       The Golden Gate Bridge is in the city of   San Francisco  0.1028292179107666
```

- **Write the answer as the model continues the prompt, leading space included.** On GPT-2's byte-level tokenizer `Paris` and `ĠParis` are both one token, so `p_bare` scores the wrong one: 1e-6 against 0.07, with no refusal. Where the bare word is more tokens than the spaced one, the metric is refused instead, without naming the space.
- **A set sums its members' probabilities** under `token_prob` and hits any under `match`; `logit_diff` and `token_logit` need one id per row, and a set there is refused by row.
- **`strip` and `lower` make the answer stand for every vocabulary token that compares the same**, so `p_loose` adds `Paris`, ` paris` and the rest to `p_set`. Drop the flag where one spelling is meant.
- **An answer of several tokens scores the read's position i against token i**, so the read carries that many positions: `before` with `occurrence: -1` reads the positions that predict the appended answer. One position for a two-token answer is refused. The recipe, and a forced chat reply, are in [teacher forcing](generation-teacher-forcing.md).
- **A hub column without the space is fixed outside the document**: write the rows inline from a script that adds it, and record the source and its commit in `description`.

## Chats

A column whose value on each row is a list of `{"role", "content"}` messages is a chat. nnspec renders it
through the checkpoint's own chat template and tokenizes the result without adding special tokens. GPT-2 has
no chat template; this `data` block is for a checkpoint that has one, such as
`HuggingFaceTB/SmolLM2-135M-Instruct`:

```json
{"chats": {"rows": [
  {"example_id": "paris", "chat": [
    {"role": "user", "content": "Name a city in France."},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "Which country is Paris in?"}]}
], "template": {"add_generation_prompt": true}}}
```

- **`template` is passed to the checkpoint's `apply_chat_template` as arguments** wherever a chat of that dataset renders.
- **Without `add_generation_prompt`** the prompt ends with the template's close of the last message, and `pos: -1` is that close's last token. **With it** the prompt ends with the assistant's header, so a `generate` continues as the assistant.
- **A template may write messages the row does not have.** SmolLM2's adds a system message when the chat has none. Its tokens are positions but no turn, so `{"turn": "system"}` there is refused. Check the rendering in `run.tokens` (`tokens.json`).
- **A template that writes today's date is refused until `template` pins it**; the refusal gives the pin.
- **Nothing in a document turns a text column into a chat.** Write the messages inline from a script.

Positions inside a turn are [chat turns](positions.md#chat-turns); generating from a chat and scoring the reply
is the [generation guide](generation.md).
