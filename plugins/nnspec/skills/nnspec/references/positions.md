# Positions

An entry's `pos` says which positions of each row it reads or writes at its site. A row's positions are its
input tokens counted from 0: BOS where the tokenizer adds one, a chat template's markers, never padding.
Positions are per row, so `-1` is each row's own last token however the batch is padded. GPT-2 adds no BOS:
position 0 is the prompt's first word.

## The grammar

```text
pos      := int                                   # shorthand for {"frame": 0, "index": int}
          | {"frame"?: selector, "scope"?: scope, "index"?: selector}
selector := int | "<start>:<stop>:<step>" | [int, ...]   # one, a half-open slice, picks in order
```

- **Three stages narrow in turn**: `frame` picks forward calls, `scope` picks text among their tokens, `index` counts positions within what is left.
- **`frame` defaults to 0**, the prefill, a forward's only call; generations add decode calls ([frames](generation.md#frames)).
- **`index` defaults to `":"`**, every position the scope left; leaving `pos` out reads every prompt token.
- **A slice goes inside the object**, `{"index": "0:3"}`; a bare `"pos": "0:3"` is refused.
- **`index`, `frame` and a scope's `occurrence` may sweep**; the `pos` object itself may not ([sweeps](sweeps.md#what-may-sweep)).

`nnspec describe pos` prints the whole grammar and its rules.

## Where each form lands

`documents/positions_tour.json` reads block 0's output (`resid`) at eleven `pos` forms and saves nothing. Its
rows are the first two IOI prompts of [SKILL.md](../SKILL.md#a-complete-experiment) with the columns `io`
(` Mary`, ` Tom`) and `s` (` John`, ` Anna`), and these are the entries of its one forward:

```json
{
  "last": {"site": "resid", "pos": -1},
  "first": {"site": "resid", "pos": 0},
  "last_three": {"site": "resid", "pos": {"index": "-3:"}},
  "io": {"site": "resid", "pos": {"scope": {"column": "pairs.io"}}},
  "s1": {"site": "resid", "pos": {"scope": {"column": "pairs.s"}}},
  "s2": {"site": "resid", "pos": {"scope": {"column": "pairs.s", "occurrence": -1}}},
  "s_both": {"site": "resid", "pos": {"scope": {"column": "pairs.s", "occurrence": ":"}}},
  "before_s2": {"site": "resid", "pos": {"scope": {"column": "pairs.s", "occurrence": -1, "before": true}}},
  "gave": {"site": "resid", "pos": {"scope": {"text": "gave"}}},
  "to_first": {"site": "resid", "pos": {"scope": {"text": " to"}}},
  "upto_comma": {"site": "resid", "pos": {"scope": {"text": ",", "until": true}}}
}
```

`nnspec show` prints where each landed on each row, saved or not:

```text
$ nnspec show out/positions_tour
  ...
  entry         at  positions
  f.last        —   [13] on all 2 rows
  f.first       —   [0] on all 2 rows
  f.last_three  —   [11,12,13] on all 2 rows
  f.io          —   [1] on all 2 rows
  f.s1          —   [3] on all 2 rows
  f.s2          —   [9] on all 2 rows
  f.s_both      —   [3,9] on all 2 rows
  f.before_s2   —   [8] on all 2 rows
  f.gave        —   [10] on all 2 rows
  f.to_first    —   [5] on all 2 rows
  f.upto_comma  —   [0,1,2,3,4,5,6,7,8] on all 2 rows
```

Row 0 is `When ĠMary Ġand ĠJohn Ġwent Ġto Ġthe Ġstore , ĠJohn Ġgave Ġa Ġdrink Ġto`, positions 0 to 13.
`to_first` lands on the ` to` of "went to", not the prompt's last token: `occurrence` picks the first match.

## Scopes

A scope names text, and `index` then counts that text's tokens:

| scope | finds |
|---|---|
| `{"column": "<dataset>.<column>"}` | each row's own value of that column |
| `{"text": " to"}` | the same literal on every row |
| `{"turn": "user"}` | one chat message of that role ([chat turns](#chat-turns)) |
| a list of columns or of texts | every occurrence of each |

- **`occurrence` defaults to 0, the first match in the whole row.** `-1` is the last, `":"` every one. On a dataset that mixes kinds, the last occurrence is one role on some rows and another on others: check a row of each kind.
- **Occurrences are ordered by where they end.** Index `-1` of a scope is the last token of the occurrence it picked, not the row's last token.
- **Text matches on token and word boundaries only**: ` Tow` does not occur in ` Tower`, nor ` Sara` in ` Sarah`.
- **A leading space is optional one way.** `{"text": "gave"}` finds `Ġgave`; `{"text": " When"}` does not find the prompt's first word. `strip` and `lower` loosen the comparison: `{"text": " When", "strip": true}` lands on 0. On a column, write `{"column": {"column": "pairs.s", "strip": true}}`.
- **A text or column value must occur in every row.** In the prefill a row without it is refused before any forward, by row, and a refused row cannot be skipped: change the data (`take`, `skip`, other rows), or use a column whose value differs by row.
- **`before: true` is the position just before each token of the scope**, as many positions as tokens: where the head predicts that word. A scope that starts at position 0 has nothing before it, and is refused; on GPT-2 that is the prompt's first word.
- **`until: true` is every token of the frame up to and including the occurrence it picks.** A row without the text gets the whole frame, so an `until` scope is never refused for an absent text.
- **The BOS a tokenizer adds covers no text**, so no scope finds it; `0` names it.

## Static width and ragged rows

A read's width is how many positions each row holds. Where the `pos` alone fixes it (an int, picks, a slice
whose ends count from one side, one occurrence of a scope with an `index` in the prefill), every row holds that
many. Elsewhere rows can disagree, and the value is ragged: as wide as its widest row, NaN past each row's own.
`documents/ragged_scope.json` reads every token of a landmark one, three and four tokens long:

```json
{
  "description": "A scope whose rows hold different numbers of tokens: every token of each row's landmark, read at the residual stream leaving block 0, its norm per token, and the mean over each row's tokens.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"facts": {"rows": [
    {"example_id": "eiffel", "input": "The Eiffel Tower is in the city of", "landmark": " Eiffel Tower"},
    {"example_id": "gate", "input": "The Brandenburg Gate is in the city of", "landmark": " Brandenburg Gate"},
    {"example_id": "kremlin", "input": "The Kremlin is in the city of", "landmark": " Kremlin"}
  ]}},
  "sites": {"resid": {"component": "layer_output", "layers": 0}},
  "steps": {
    "f": {"kind": "forward", "input": "facts.input", "interventions": {
      "landmark": {"site": "resid", "pos": {"scope": {"column": "facts.landmark"}}, "ops": [{"op": "norm"}]},
      "landmark_last": {"site": "resid", "pos": {"scope": {"column": "facts.landmark"}, "index": -1}, "ops": [{"op": "norm"}]}
    }},
    "count": {"kind": "op", "op": "count", "of": "f.landmark"},
    "per_row": {"kind": "op", "op": "mean", "of": "f.landmark", "over": ["pos"]}
  },
  "saves": ["f.landmark", "f.landmark_last", "count", "per_row"]
}
```

```text
$ nnspec table out/ragged_scope f.landmark
example_id  row  pos  pos_token  input                                   value
eiffel      0    1    ĠE         The Eiffel Tower is in the city of      51.706668853759766
eiffel      0    2    iff        The Eiffel Tower is in the city of      54.16061782836914
eiffel      0    3    el         The Eiffel Tower is in the city of      53.75648498535156
eiffel      0    4    ĠTower     The Eiffel Tower is in the city of      60.143211364746094
gate        1    1    ĠBrand     The Brandenburg Gate is in the city of  59.77140426635742
gate        1    2    enburg     The Brandenburg Gate is in the city of  65.51815032958984
gate        1    3    ĠGate      The Brandenburg Gate is in the city of  59.88993835449219
kremlin     2    1    ĠKremlin   The Kremlin is in the city of           67.14871215820312
$ nnspec table out/ragged_scope count per_row
example_id  row  input                                   count  per_row
eiffel      0    The Eiffel Tower is in the city of      4.0    54.941749572753906
gate        1    The Brandenburg Gate is in the city of  3.0    61.726497650146484
kremlin     2    The Kremlin is in the city of           1.0    67.14871215820312
```

- **The saved tensor is `[3, 4]`**, NaN past each row's own positions; `nnspec table` leaves those out.
- **Index j along `pos` is the read's j-th position on that row**, not position j. A value without rows keeps that index as its `pos`: a `{"index": "-3:"}` read averaged `"over": ["rows"]` tables `pos` 0, 1, 2, where the rows' positions are 11, 12, 13.
- **`index: -1` makes it static**: `f.landmark_last` is one position per row, `[4], [3], [1]`.
- **`count` over `pos` is each row's own number of positions**, and a `mean` over a ragged read weighs positions, not rows: reduce `"over": ["pos"]` first, then over `rows`, to weigh rows equally.

## Pairing with another forward

Two values of different reads pair position by position, the i-th with the i-th, whether one forward or two
made them, so the width must be one number on every row.

| read `pos` | pairs with a write in another forward |
|---|---|
| a static index: `-1`, `[-1, 0]`, `{"index": "-3:"}` | yes |
| one occurrence of a scope in the prefill, with a static `index` | yes |
| one occurrence of a scope, read whole, not `until` | where every row's scope holds as many tokens |
| an open index in the prefill: `{"index": ":"}`, `"5:"`, no `pos` | where every row holds as many positions |
| several occurrences, `until`, a scope or an open index beyond the prefill | no |

Where the rows decide, a pairing they disagree on is refused before any forward, naming each count's rows and
a static span or index to read instead.

- **An unscoped `pos` between two prompts pairs index with index.** Two prompts of one length pass however their words line up.
- **On prompts of different lengths a counted index pairs only where every token between it and the end it counts from is the same on both**, and is refused past that, naming the two tokens.
- **Scope both sides of a patch to the word they share**: `{"scope": {"column": "pairs.entity"}, "index": -1}` on the base and the counterfactual's own column on the other side ([prompts of different lengths](patching.md#prompts-of-different-lengths)).
- **To pair a ragged read, take a static index or reduce it over `pos` first.** An operand from an earlier entry of the same forward may be ragged, since each row pairs with itself.

## Check where it landed

Every read and write records its positions on every row, saved or not. Look before trusting a number:

- **`nnspec show OUT`** prints one line per entry: `[13] on all 4 rows` where rows agree, else each row's list, and `—` for a row the entry never reached.
- **`nnspec table OUT NAME`** prints the `pos` and `pos_token` columns for a saved value with rows, and `call` in a generation.
- **In Python**, the positions and the tokens they index:

```python
from nnspec import Plan

plan = Plan.load("out/positions_tour")
tokens = plan.run.tokens["pairs.input"]
for row, where in enumerate(plan.steps["f"].interventions["s2"].positions["pairs"]):
    print(row, where, [tokens[row][p] for p in where])     # 0 [9] ['ĠJohn'], then 1 [9] ['ĠAnna']
```

A position past the prompt indexes the generated tokens: `tokens[<input>][row] + plan.run.tokens[<step>][row]`.
Prompts of different lengths put `-1` at different absolute positions per row, so compare tokens, not numbers.
On a swept document each point has its own positions ([output](output.md#in-python)).

## Chat turns

`{"scope": {"turn": "<role>", "occurrence": k}}` is the k-th message of that role in the row's chat, its content
as the template renders it. The template's markers belong to no turn, so `index: -1` of a turn is the message's
own last token, where a plain `pos: -1` on a chat reads whatever the template writes last. A turn is a frame-0
scope, refused on a text input. Rendering a chat is [data](data.md#chats).

## Generated tokens

In a `generate` step `frame` picks calls: `0` the prefill, `k` decode call k, `"1:"` every decode call, `":"`
all of them, `-1` the row's last call. A decode call holds one position, the token generated before it; the
row's last generated token is never an input ([frames](generation.md#frames)).

- **Every generated input**: `{"frame": "1:"}`.
- **The last call, one position**: `{"frame": -1, "index": -1}`; keep the `index`, since on a row whose first token was EOS the last call is the prefill.
- **Where the model said a word**: `{"frame": "1:", "scope": {"column": "prompts.said"}, "index": -1}`.
- **The calls that chose a word**: the same scope with `"before": true`, which also sees the row's last token.

## Rows a scope never reached

Beyond the prefill, a row that never made a call, or never said a scope's text, is `alignment_missing`: not
refused, recorded in the entry's `positions`, and NaN in every value made from it. So is a read at an open
index past a short row, in the prefill too. `generate_reads.json` ([reads at every call](generation.md#reads-at-every-call))
reads the calls that chose each row's expected word; GPT-2 says Sochi, not Moscow, on row 2:

```text
$ nnspec table out/generate_reads g.conf
example_id  row  pos  pos_token  call  input                                     said            value                alignment_missing
eiffel      0    9    Ġof        0     The Eiffel Tower is in the city of         Paris          0.06997036188840866
golden      1    8    Ġof        0     The Golden Gate Bridge is in the city of   San Francisco  0.5367891788482666
golden      1    9    ĠSan       1     The Golden Gate Bridge is in the city of   San Francisco  0.9022789001464844
kremlin     2                          The Kremlin is in the city of              Moscow                              true
```

- **Reductions skip missing rows**, so a `mean` here is over the two rows that said their word, and the number does not say so.
- **Count the rows reached in the document**: `count` of the read (over `pos`), then `gt` 0, then `sum`. Here that gives 2.0.
- **`count` over `["rows", "pos"]` counts positions, not rows**: the two-token " San Francisco" counts twice.
- **Or check the record after the run**: a row the entry never reached holds `"alignment_missing"` in place of its list in `plan.steps[<step>].interventions[<entry>].positions`.
- **A write whose operand is missing on a row it lands on is refused at run**, naming the row and the call: read at `{"frame": -1, "index": -1}`, or hold EOS off with `min_new_tokens`.
- **Teacher forcing moves the question into the prefill**, where every miss is a refusal ([teacher forcing](generation-teacher-forcing.md)).
