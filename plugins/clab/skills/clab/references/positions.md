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
- **`frame` defaults to 0**, the prefill, a forward's only call; generations add decode calls ([generate](generate.md#frames)).
- **`index` defaults to `":"`**, every position the scope left; leaving `pos` out reads every prompt token.
- **A negative int counts from the row's end**; a slice string is Python's (`"-4:-1"`, `"2:"`, `"::2"`); a list keeps its order.
- **A slice goes inside the object**, `{"index": "0:3"}`; a bare `"pos": "0:3"` is refused (`'0:3' is a string, which is not read as a number`).
- **A selector that picks nothing on any row** (`"0:0"`, `"5:2"`) is refused at compile.
- **`index`, `frame` and a scope's `occurrence` may sweep**; the `pos` object itself may not ([sweeps](sweeps.md#what-may-sweep)).

`clab describe pos` prints the whole grammar and its rules.

## Where each form lands

`documents/positions_tour.json` reads block 0's output at eleven `pos` forms and saves nothing; `clab show`
prints each entry's positions per row.

```json
{
  "description": "Where each pos form lands on GPT-2, which adds no BOS: reads of the residual stream leaving block 0 on the indirect-object prompts. Nothing is saved; clab show prints each entry's positions per row.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"example_id": "mary", "input": "When Mary and John went to the store, John gave a drink to", "io": " Mary", "s": " John"},
    {"example_id": "tom", "input": "When Tom and Anna went to the park, Anna gave a ball to", "io": " Tom", "s": " Anna"}
  ]}},
  "sites": {"resid": {"component": "layer_output", "layers": 0}},
  "steps": {
    "f": {"kind": "forward", "input": "pairs.input", "interventions": {
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
    }}
  },
  "saves": []
}
```

```text
$ clab show out/positions_tour
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

- **`occurrence` defaults to 0, the first match in the whole row.** `-1` is the last, `":"` every one; a word that also appears earlier in the prompt is found there first.
- **Occurrences are ordered by where they end.** Index `-1` of a scope is the last token of the occurrence it picked, not the row's last token.
- **Text matches on token and word boundaries only.** `{"text": " Tow"}` on "The Eiffel Tower" is refused before any forward: `f.t: row 0: text ' Tow' is not in its input; it occurs only inside ' Tower', cutting a word, and an occurrence lies on word boundaries`.
- **A leading space is optional one way.** `{"text": "gave"}` finds `Ġgave`; `{"text": " When"}` does not find the prompt's first word, and is refused (`f.x: row 0: text ' When' is not in its input`).
- **`strip` and `lower` loosen the comparison**: `{"text": " When", "strip": true}` lands on 0. On a column, write `{"column": {"column": "pairs.s", "strip": true}}`.
- **A text literal must be in every row.** `{"text": "mary", "lower": true}` finds `ĠMary` on row 0 and is refused on row 1, which has no Mary.
- **`before: true` is the position just before each token of the scope**, as many positions as tokens: where the head predicts that word. `before_s2` lands on 8, the comma.
- **A scope that starts at position 0 has nothing before it.** On GPT-2 a `before` scope on the first word is refused: `text 'When' begins the row, at position 0, which no position comes before`.
- **`until: true` is every token of the frame up to and including the occurrence it picks.** A row without the text gets the whole frame; an `until` scope is never refused for an absent text.
- **The BOS a tokenizer adds covers no text**, so no scope finds it; `0` names it.

## Static width and ragged rows

A read's width is how many positions each row holds. Where the `pos` alone fixes it (an int, picks, a
slice whose ends count from one side, one occurrence of a scope with an `index` in the prefill), every row holds
that many. Elsewhere rows can disagree, and the value is ragged: as wide as its widest row, NaN past each
row's own. `documents/ragged_scope.json` reads every token of a landmark one, three and four tokens long:

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
$ clab table out/ragged_scope f.landmark
example_id  row  pos  pos_token  input                                   value
eiffel      0    1    ĠE         The Eiffel Tower is in the city of      51.706668853759766
eiffel      0    2    iff        The Eiffel Tower is in the city of      54.16061782836914
eiffel      0    3    el         The Eiffel Tower is in the city of      53.75648498535156
eiffel      0    4    ĠTower     The Eiffel Tower is in the city of      60.143211364746094
gate        1    1    ĠBrand     The Brandenburg Gate is in the city of  59.77140426635742
gate        1    2    enburg     The Brandenburg Gate is in the city of  65.51815032958984
gate        1    3    ĠGate      The Brandenburg Gate is in the city of  59.88993835449219
kremlin     2    1    ĠKremlin   The Kremlin is in the city of           67.14871215820312
$ clab table out/ragged_scope count per_row
example_id  row  input                                   count  per_row
eiffel      0    The Eiffel Tower is in the city of      4.0    54.941749572753906
gate        1    The Brandenburg Gate is in the city of  3.0    61.726497650146484
kremlin     2    The Kremlin is in the city of           1.0    67.14871215820312
```

- **The saved tensor is `[3, 4]`**, NaN past each row's own positions; `clab table` leaves those out rather than printing NaN rows.
- **Index j along `pos` is the read's j-th position on that row**, not position j: `clab show` prints `[1,2,3,4], [1,2,3], [1]` for `f.landmark`.
- **A value without rows keeps that index as its `pos`**: a `{"index": "-3:"}` read averaged `"over": ["rows"]` tables `pos` 0, 1, 2, where the rows' positions are 11, 12, 13. The rows' own are in the entry's `positions`.
- **`index: -1` makes it static**: `f.landmark_last` is one position per row, `[4], [3], [1]`.
- **`count` over `pos` is each row's own number of positions**, and a `mean` over a ragged read weighs positions, not rows.

## Pairing with another forward

An operand read in one forward and written in another pairs position by position, the i-th with the i-th, so
its width must be one number on every row.

| read `pos` | pairs with a write in another forward |
|---|---|
| a static index: `-1`, `[-1, 0]`, `{"index": "-3:"}` | yes |
| one occurrence of a scope in the prefill, with a static `index` | yes |
| one occurrence of a scope, read whole, not `until` | where every row's scope holds as many tokens |
| `{"index": ":"}`, no `pos`, several occurrences, `until`, a scope beyond the prefill | no |

A whole scope whose rows disagree is refused before any forward, naming the counts:
`g.w: operand 'f.lm' holds 4 positions on row 0, 3 positions on row 1 and 1 position on row 2; a scoped read
pairs with another forward's value when every row's scope holds the same number of tokens: pick one with
'index' (-1, or '-2:'), or reduce it`. An open index on both sides is refused at compile:
`steps.g.interventions.w.ops[0].with: 'f.lm' has no width compile can fix …`.

- **An unscoped `pos` between two prompts pairs index with index.** Two prompts of one length pass however their words line up.
- **On prompts of different lengths a counted index pairs when every token between it and the end it counts from is the same on both**, whatever the two paired tokens are. On a base ending `, John gave a drink to` and a counterfactual one token longer ending `, Mary gave a drink to`, `-1` to `-5` pair, the names at `-5` included. `-6` pairs `,` with `,` and is refused, since `-5` differs; the refusal names the two tokens.
- **Scope both sides of a patch to the word they share**: `{"scope": {"column": "pairs.entity"}, "index": -1}` on the base and the counterfactual's own column on the other side.
- **To pair a ragged read, take a static index or reduce it over `pos` first.** An operand from an earlier entry of the same forward may be ragged, since each row pairs with itself ([sites and interventions](sites-and-interventions.md#operands)).

## Check where it landed

Every read and write records its positions on every row, saved or not. Look before trusting a number:

- **`clab show OUT`** prints one line per entry: `[13] on all 4 rows` where rows agree, else each row's list, and `—` for a row the entry never reached.
- **`clab table OUT NAME`** prints the `pos` and `pos_token` columns for a saved value with rows, and `call` in a generation.
- **In Python**, the positions and the tokens they index:

<!-- test: skip -->
```python
from clab import Plan

plan = Plan.load("out/positions_tour")
tokens = plan.run.tokens["pairs.input"]
for row, where in enumerate(plan.steps["f"].interventions["s2"].positions["pairs"]):
    print(row, where, [tokens[row][p] for p in where])     # 0 [9] ['ĠJohn'], then 1 [9] ['ĠAnna']
```

A position past the prompt indexes the generated tokens: `tokens[<input>][row] + plan.run.tokens[<step>][row]`.
Prompts of different lengths put `-1` at different absolute positions per row, so compare tokens, not numbers.

On a swept document `clab show` lists the points unlabelled; [output](output.md#in-python) has the loop that
checks a read's and a write's positions point by point.

## Chat turns

`{"scope": {"turn": "<role>", "occurrence": k}}` is the k-th message of that role in the row's chat, its content
as the template renders it. The template's markers belong to no turn, so `index: -1` of a turn is the message's
own last token. A turn is a frame-0 scope, refused on a text input. Rendering a chat, `add_generation_prompt`
and the messages a template writes itself are [data](data.md#chats).

## Generated tokens

In a `generate` step `frame` picks calls: `0` the prefill, `k` decode call k, `"1:"` every decode call, `":"`
all of them, `-1` the row's last call. A decode call holds one position, the token generated before it; the
row's last generated token is never an input. Frames and writes during generation are in
[generate](generate.md).

- **Every generated input**: `{"frame": "1:"}`.
- **The last call, one position**: `{"frame": -1, "index": -1}`; keep the `index`, since on a row whose first token was EOS the last call is the prefill.
- **Where the model said a word**: `{"frame": "1:", "scope": {"column": "prompts.said"}, "index": -1}`.
- **The calls that chose a word**: the same scope with `"before": true`, which also sees the row's last token.

## Rows a scope never reached

Beyond the prefill, a row that never made a call, or never said a scope's text, is `alignment_missing`: not
refused, recorded in the entry's `positions`, and NaN in every value made from it. In the prefill the same misses
are refused before any forward, naming the row, except an open index that picks none of a row's tokens.
`documents/generate_reads.json` ([generate](generate.md#reads-at-every-call)) reads the calls that chose each
row's expected word; GPT-2 says Sochi, not Moscow, on row 2:

```text
$ clab table out/generate_reads g.conf
example_id  row  pos  pos_token  call  input                                     said            value               alignment_missing
eiffel      0    9    Ġof        0     The Eiffel Tower is in the city of         Paris          0.0699695572257042
golden      1    8    Ġof        0     The Golden Gate Bridge is in the city of   San Francisco  0.5367915630340576
golden      1    9    ĠSan       1     The Golden Gate Bridge is in the city of   San Francisco  0.9022799134254456
kremlin     2                          The Kremlin is in the city of              Moscow                             true
```

- **Reductions skip missing rows**, so a `mean` here is over the two rows that said their word, and the number does not say so.
- **Count the rows reached in the document**: `count` of the read (over `pos`), then `gt` 0, then `sum`. Here that gives 2.0.
- **`count` over `["rows", "pos"]` counts positions, not rows**: 3.0 here, since " San Francisco" is two tokens. It counts rows only where each row's read is one position (`index: -1`).
- **Or check the record after the run**: a row the entry never reached holds `"alignment_missing"` in place of its list in `plan.steps[<step>].interventions[<entry>].positions`.
- **Teacher forcing moves the question into the prefill**, where every miss is a refusal ([data](data.md#inputs)).

What to do when a row is missing, and why, is the `clab-debugging` skill's.
