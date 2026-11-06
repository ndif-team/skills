# Data and inputs

The `data` block names a document's datasets. Each is a list of rows, each row a JSON object. A step's `input`
names the column it runs over, one row per example. Other fields name columns as answers to score or as text to
find. `clab describe records` prints every field of `data.<name>`.

## Where rows come from

A dataset has exactly one source; two are refused.

| source | holds | note |
|---|---|---|
| `"rows": [...]` | the rows, inline | they travel in the plan and the manifest, so a run's table reads them back from anywhere |
| `"path": "ioi/train"` | `data/ioi/train.json` under the corpus root, a JSON list of row objects | the corpus root is in [where output goes](running-and-engines.md#where-output-goes) |
| `"hf": "<owner>/<name>"` | a hub dataset at a required `revision` (a commit), with `config` and `split` | offline, it loads from the `datasets` cache one online load filled |

A fragment of a `data` block with the two other sources:

```json
{"ioi": {"path": "ioi/train", "split": "train", "take": 100},
 "glue": {"hf": "nyu-mll/glue", "revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c", "config": "wnli", "split": "validation", "take": 3}}
```

- **A file is read again when you table the run**, from `data/` under the working directory, and rows that changed since the run are refused. Run `clab table` from the corpus root.
- **A document whose `path` has no `data/` above it is refused**: `data.pairs.path: no file data/… under the corpus root`.
- **A hub dataset's rows never enter the plan**; the run records their digest and refuses rows whose digest differs.
- **A hub dataset with several splits and no `split` is refused**, listing the splits.

## Narrowing and renaming

Five fields apply in this order: `split`, `shuffle`, `skip`, `take`, `columns`.

| field | does | refused |
|---|---|---|
| `split` | a file: keeps rows whose `split` column equals it; the hub: which split | on inline rows; a split no row has |
| `shuffle: true` | one fixed order, the same on every machine, depending on the row count alone | — |
| `skip: N` | drops the first N rows of that order | an N that leaves no row |
| `take: N` | keeps the first N rows left; after `shuffle`, a random sample | an N past the rows left |
| `columns` | renames (`{"new": "old"}`) or derives (`{"new": {"from": ..., "after"?, "before"?, "strip"?}}`) | a source column the rows lack; a derived column's text absent on a row |

`documents/columns.json` shuffles four inline rows, skips one, keeps two, renames `text` to `prompt` and cuts
the second subject out of the prompt as a column, then scopes a read to it:

```json
{
  "description": "Narrowing and cutting inline rows: a fixed shuffle, skip and take, a renamed column, and a column cut from the prompt's text, used as a scope.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {
    "rows": [
      {"example_id": "mary", "text": "When Mary and John went to the store, John gave a drink to"},
      {"example_id": "tom", "text": "When Tom and Anna went to the park, Anna gave a ball to"},
      {"example_id": "sarah", "text": "When Sarah and James went to the office, James gave a book to"},
      {"example_id": "paul", "text": "When Paul and Lisa went to the school, Lisa gave a pen to"}
    ],
    "shuffle": true, "skip": 1, "take": 2,
    "columns": {
      "prompt": "text",
      "giver": {"from": "text", "after": {"text": ", "}, "before": {"text": " gave"}}
    }
  }},
  "sites": {"resid": {"component": "layer_output", "layers": 0}},
  "steps": {
    "f": {"kind": "forward", "input": "pairs.prompt", "interventions": {
      "at_giver": {"site": "resid", "pos": {"scope": {"column": "pairs.giver", "occurrence": -1}}, "ops": [{"op": "norm"}]}
    }}
  },
  "saves": ["f.at_giver"]
}
```

```text
$ clab table out/columns f.at_giver
example_id  row  pos  pos_token  prompt                                            value
paul        0    9    ĠLisa      When Paul and Lisa went to the school, Lisa gav…  63.120601654052734
sarah       1    9    ĠJames     When Sarah and James went to the office, James …  58.88503646850586
```

- **`row` counts the rows after narrowing**, not the source's order; `example_id`, where the rows have one, leads every table row, so give rows one before you shuffle or take.
- **A renamed column answers to its new name only**: `"input": "pairs.text"` is refused with `dataset 'pairs' has no column 'text' on every row; columns: example_id, prompt, giver`.
- **A derived column cuts text**: `after` keeps what follows the first occurrence of its text, `before` what precedes it, `strip` drops surrounding whitespace. `giver` is `Lisa` and `James`, with no leading space; the scope still finds `ĠLisa`, since a leading space is optional one way.
- **Each cut is `{"text": ...}`**: a bare string is refused (`data.pairs.columns.giver.after: ', ' is a string; text written as it is is {'text': ', '}`), and so is a row that lacks the text, by row.
- **`split` belongs to a file or the hub**: on inline rows it is refused (`data.pairs: 'split' is which split of a file or a hub dataset; inline 'rows' are the rows`).
- **`take` without `shuffle` keeps the first N rows**, which on a sorted file is not a sample.
- **A rename into a name the steps read changes what they read, unrefused.** Check `run.tokens` after renaming.

## Inputs

| `input` | each row's prompt |
|---|---|
| `"pairs.input"` | the column's text, or its chat |
| `"pairs.counterfactual_inputs[0]"` | entry 0 of a list column; every row needs one |
| `["facts.input", "facts.answer"]` | the parts joined with nothing between: teacher forcing |
| `["facts.input", {"text": " and"}, "facts.answer"]` | a fixed text among the parts |

- **Text is tokenized as the tokenizer does by default**, with BOS where it adds one; the run records every input's tokens under its name in `run.tokens`.
- **Nothing goes between the parts of a list**, so the answer's own leading space separates the words.
- **A column without its space joins the previous word** when appended: on GPT-2 that is other tokens. Put the space in the rows, or a `{"text": " "}` part between.
- **All columns of a list come from one dataset**, and a chat column in a list is refused, since a list joins text.
- **A counterfactual belongs in its base rows' dataset**, as a second column: a value with a rows axis belongs to its dataset, and another dataset's rows refuse it ([sites and interventions](sites-and-interventions.md#operands)).

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
$ clab table out/answers p_first p_bare p_set p_loose p_the
example_id  row  pos  pos_token  input                                     first   p_first              first_bare  p_bare                  cities                p_set                p_loose             p_the
eiffel      0    9    Ġof        The Eiffel Tower is in the city of         Paris  0.06997019797563553  Paris       1.0362419970988412e-06  [" Paris"," London"]  0.12832464277744293  0.1283280849456787  0.00387186580337584
golden      1    8    Ġof        The Golden Gate Bridge is in the city of   San    0.5367922186851501   San         4.941660336044151e-06   [" San"," Los"]       0.6203228831291199   0.6203514337539673  0.001582340570166707
$ clab table out/answers nll
example_id  row  pos  pos_token  input                                     answer          value
eiffel      0    9    Ġof        The Eiffel Tower is in the city of         Paris          2.6596858501434326
golden      1    8    Ġof        The Golden Gate Bridge is in the city of   San Francisco  0.6221428513526917
golden      1    9    ĠSan       The Golden Gate Bridge is in the city of   San Francisco  0.1028292179107666
```

- **Write the answer as the model continues the prompt, leading space included.** On GPT-2's byte-level tokenizer `Paris` and `ĠParis` are both one token, so `p_bare` scores the wrong one: 1e-6 against 0.07, with no refusal.
- **Where the bare word is more tokens than the spaced one**, the metric is refused instead, since the read has fewer positions than the answer has tokens.
- **A set sums its members' probabilities** under `token_prob` and hits any under `match`; `logit_diff` and `token_logit` need one id per row, and a set there is refused by row.
- **`strip` and `lower` make the answer stand for every vocabulary token that compares the same**, so `p_loose` adds `Paris`, ` paris` and the rest to `p_set`.
- **A bare string is always a column reference**: `"expected": " John"` is refused with `' John' is not <dataset>.<column> over a dataset; datasets: pairs; text written as it is, the same on every row, is {'text': ' John'}`.
- **An answer of several tokens scores the read's position i against token i.** The read must carry that many positions: `before` with `occurrence: -1` reads the positions that predict the appended answer, and a row of a one-token answer has one.
- **One position for a two-token answer is refused**: `pairs.cf_answer row 0: ' Johnathan' is 2 tokens; the read has 1 position on that row; …`.
- **Scope a teacher-forced answer with `"occurrence": -1`**, so a prompt that also says the word is not picked first.
- **A hub column without the space is fixed outside the document**: write the rows inline from a script that adds it, and record the source and its commit in `description`.

## Chats

A column whose value on each row is a list of `{"role", "content"}` messages is a chat. clab renders it
through the checkpoint's own chat template and tokenizes the result without adding special tokens. GPT-2 has
no chat template; the fragment below is a `data` block for a checkpoint that has one, such as
`HuggingFaceTB/SmolLM2-135M-Instruct`:

```json
{"chats": {"rows": [
  {"example_id": "paris", "chat": [
    {"role": "user", "content": "Name a city in France."},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "Which country is Paris in?"}]}
], "template": {"add_generation_prompt": true}}}
```

- **`template` is passed to the checkpoint's `apply_chat_template` as arguments** wherever a chat of that dataset renders; a `template` on a dataset no step reads as a chat is refused.
- **Without `add_generation_prompt`** the prompt ends with the template's close of the last message, and `pos: -1` is that close's last token.
- **With it** the prompt ends with the assistant's header, so a `generate` continues as the assistant: on SmolLM2 the rendering grows from 51 to 55 tokens, ending `<|im_start|> ass istant Ċ`.
- **A template may write messages the row does not have.** SmolLM2's adds a system message when the chat has none. Its tokens are positions but no turn, so `{"turn": "system"}` there is refused (`the row has no 'system' message; the template wrote a 'system' header of its own, …`). Check the rendering in `run.tokens` (`tokens.json`).
- **A template that writes today's date is refused until `template` pins it**, with the pin in the refusal (`{"date_string": "26 Jul 2024"}` on Llama-3.2's).
- **Nothing in a document turns a text column into a chat.** Write the messages inline from a script, and record the source in `description`.

Positions inside a turn are [positions](positions.md#chat-turns); generating from chats is the
`clab-steering-and-generation` skill's.
