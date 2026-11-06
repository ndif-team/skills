# Wrong numbers nothing refused

The run finished and the numbers look wrong. Find what you saw below; each row gives what it is, the check
that tells, and the fix. Most checks read what the run recorded:

- `nnspec show OUT`: its first line (which run this is), `served:` (the dtype), each entry's positions per row
  (`—` for a row it did not reach) and each result's `axes` and `shape`;
- `nnspec table OUT NAME`: `pos` and `pos_token`, the token each row's value was read at, and
  `alignment_missing`;
- the nulls in [SKILL.md](../SKILL.md#the-nulls), each of which has a value known before the run.

Numbers below come from runs on `openai-community/gpt2` in float32 on the CPU. A refusal is
[refusals.md](refusals.md).

## Reads and writes

| what you see | what it is | the check | the fix |
|---|---|---|---|
| a patch's scores equal the clean run's to the bit | an entry with no edit in its `ops` only reads: the self-swap document with the patch's `ops` deleted prints `patched_ld` `-2.4247055053710938` on row 0, the clean value | the entry's `ops` hold its edit | restore the edit (`swap`, `add`, …) |
| every point of a patch equals the clean run, and the nulls pass | a freeze listed after the patch at one site puts the clean value back: entries at a site run in the order written | the order of the entries at the site | list the freeze first |
| a read changed the output, or a later read of the same forward | an entry whose `ops` include an edit (`add`, `scale`, `normalize`, …) writes back | which entries hold an edit | compute on a read in an op step, or with `mul` |
| a patch meant for one layer at a time changed every layer at once | a `layers` slice or list on a write writes at every layer it picks, in one forward | `axes` in `nnspec show` is `—` where you meant a layer sweep | `"layers": {"sweep": [...]}` |
| a vector read at one layer changed every layer of a write across `layers` | an operand without a `layers` axis is the same at each layer | the operand's site | read it at the same `layers`, paired by label |
| a steered run equals the unsteered one at every dose | `normalize` of a zero vector stays zero, and adding zero changes nothing | save the direction and read its `norm` | a direction with a norm well above rounding |

## Nulls

The four nulls that must equal the clean run, in one document. Each forward's `ld` is the metric at the head:

```json
{
  "description": "Four nulls beside the clean run, each of which must equal it to the bit: the base's own value swapped back, an add at dose 0, a scale by 1, and a swap of the identity permutation of the rows.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"input": "The Eiffel Tower is in the city of", "cf_input": "The Colosseum is in the city of", "base_answer": " Paris", "cf_answer": " Rome"},
    {"input": "The Colosseum is in the city of", "cf_input": "The Eiffel Tower is in the city of", "base_answer": " Rome", "cf_answer": " Paris"},
    {"input": "Big Ben is in the city of", "cf_input": "The Brandenburg Gate is in the city of", "base_answer": " London", "cf_answer": " Berlin"},
    {"input": "The Brandenburg Gate is in the city of", "cf_input": "Big Ben is in the city of", "base_answer": " Berlin", "cf_answer": " London"}
  ]}},
  "sites": {"target": {"component": "layer_output", "layers": 6}, "head": {"component": "lm_head_output"}},
  "interventions": {
    "scored": {"ld": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]}}
  },
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": ["scored", {"v_self": {"site": "target", "pos": -1}}]},
    "counterfactual": {"kind": "forward", "input": "pairs.cf_input", "interventions": {"v_cf": {"site": "target", "pos": -1}}},
    "identity": {"kind": "op", "op": "permute", "of": "clean.v_self", "seed": null},
    "self_swap": {"kind": "forward", "input": "pairs.input",
                  "interventions": ["scored", {"w": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "clean.v_self"}]}}]},
    "dose_0": {"kind": "forward", "input": "pairs.input",
               "interventions": ["scored", {"w": {"site": "target", "pos": -1, "ops": [{"op": "add", "with": "counterfactual.v_cf", "scale": 0.0}]}}]},
    "scale_1": {"kind": "forward", "input": "pairs.input",
                "interventions": ["scored", {"w": {"site": "target", "pos": -1, "ops": [{"op": "scale", "factor": 1.0}]}}]},
    "perm_null": {"kind": "forward", "input": "pairs.input",
                  "interventions": ["scored", {"w": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "identity"}]}}]}
  },
  "saves": ["clean.ld", "self_swap.ld", "dose_0.ld", "scale_1.ld", "perm_null.ld"]
}
```

```text
$ nnspec table out/nulls clean.ld self_swap.ld dose_0.ld scale_1.ld perm_null.ld
row  pos  pos_token  input                                   cf_answer  base_answer  clean.ld             self_swap.ld         dose_0.ld            scale_1.ld           perm_null.ld
0    9    Ġof        The Eiffel Tower is in the city of       Rome       Paris       -2.4247055053710938  -2.4247055053710938  -2.4247055053710938  -2.4247055053710938  -2.4247055053710938
3    8    Ġof        The Brandenburg Gate is in the city of   London     Berlin      -5.286407470703125   -5.286407470703125   -5.286407470703125   -5.286407470703125   -5.286407470703125
```

Every row prints one value five times. `cmp` of two of these files reports a difference at byte 1 even so: a
saved file's header holds the value's name. Compare values of different names in `nnspec table` or in torch,
and files byte for byte only between two runs of one document.

## Positions and selections

The default `occurrence` is the first match in the whole row. Two reads of the subject, said twice:

```json
{
  "description": "Two reads of the subject's name, which the prompt says twice: the default occurrence (0, the first mention) and occurrence -1 (the repeat before 'gave'). nnspec show prints where each landed.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"ioi": {"rows": [
    {"input": "When Mary and John went to the store, John gave a drink to", "subject": " John"},
    {"input": "When John and Mary went to the store, Mary gave a drink to", "subject": " Mary"}
  ]}},
  "sites": {"resid": {"component": "layer_output", "layers": 6}},
  "steps": {
    "base": {"kind": "forward", "input": "ioi.input", "interventions": {
      "first": {"site": "resid", "pos": {"scope": {"column": "ioi.subject"}}, "ops": [{"op": "norm"}]},
      "repeat": {"site": "resid", "pos": {"scope": {"column": "ioi.subject", "occurrence": -1}}, "ops": [{"op": "norm"}]}
    }}
  },
  "saves": ["base.first", "base.repeat"]
}
```

```text
$ nnspec show out/scope_occurrence
  entry        at  positions
  base.first   —   [3] on all 2 rows
  base.repeat  —   [9] on all 2 rows
$ nnspec table out/scope_occurrence base.repeat
row  pos  pos_token  input                                             value
0    9    ĠJohn      When Mary and John went to the store, John gave…  102.47533416748047
1    9    ĠMary      When John and Mary went to the store, Mary gave…  103.17230987548828
```

| what you see | what it is | the check | the fix |
|---|---|---|---|
| a read or write on the first mention of a word, not the one you meant | `occurrence` defaults to 0, the first; occurrences are whole words (`Sara` is not inside `Sarah`) | `pos_token` and the positions in `nnspec show` | `"occurrence": -1` for the last, `1` for the second, `":"` for every one |
| a teacher-forced score looks plausible but was read at the prompt's own mention of the answer | the default occurrence finds the prompt's copy first | the scored read's positions in `nnspec show` on a row whose prompt also names the answer: they must follow the prompt | `"occurrence": -1`, the appended copy |
| a patch scoped with `"occurrence": -1` gives a plausible curve on a mixed dataset | the last occurrence is the query's copy on some rows and the context's on others | the positions on a row of each kind | a scope that names the same role on every row |
| a window swap between two inputs moved other tokens than you meant | one unscoped `pos` pairs index with index; rows of equal length are not checked, whatever their tokens | both entries' positions and tokens | a scope on each side naming its own text |
| a read over every token includes a chat template's markers or a BOS | every input token is a position; GPT-2 has no BOS, so position 0 is the first word | `pos_token` at `pos` 0 | `{"index": "1:"}` on a model with a BOS; a `turn` scope in a chat |
| `pos: -1` on a chat input reads a template marker | the rendered chat ends with what the template writes last | `pos_token` at the read | a `turn` scope for the message's own tokens |
| `argmax` or `top_k` over heads gives 0 and 1 where you selected heads 3 and 1 | along a selected axis, indices count within the `select` list | the `select` list | map each index back through the list |
| a position sweep shows effects where the counterfactual was never read | a sweep of the write's `pos` alone; the read stays put | the read's positions in `nnspec show` at each point | sweep the read's `pos` over the same values, one label |
| a query patched between prompts moved where its head looks | queries and keys are taken after RoPE, so they carry their prompt's positions | the two prompts' token counts | patch between prompts that tokenize alike |

Scoped reads and positions are the `nnspec` skill's [positions reference](../../nnspec/references/positions.md).

## Gradients

`documents/grad_check.json`, the gradient of the metric beside the gradient of the same metric times 0:

```json
{
  "description": "The gradient null: the gradient of the metric at layer 6's last token, beside the gradient of the same metric times 0, which must be exactly zero on every row.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"input": "The Eiffel Tower is in the city of", "base_answer": " Paris", "cf_answer": " Rome"},
    {"input": "The Colosseum is in the city of", "base_answer": " Rome", "cf_answer": " Paris"},
    {"input": "Big Ben is in the city of", "base_answer": " London", "cf_answer": " Berlin"},
    {"input": "The Brandenburg Gate is in the city of", "base_answer": " Berlin", "cf_answer": " London"}
  ]}},
  "sites": {"target": {"component": "layer_output", "layers": 6}, "head": {"component": "lm_head_output"}},
  "steps": {
    "taken": {"kind": "forward", "input": "pairs.input", "interventions": {
      "g": {"site": "target", "pos": -1, "ops": [{"op": "grad", "of": "metric"}, {"op": "norm"}]},
      "g0": {"site": "target", "pos": -1, "ops": [{"op": "grad", "of": "zero"}, {"op": "norm"}]},
      "metric": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}]},
      "zero": {"site": "head", "pos": -1, "ops": [{"op": "logit_diff", "a": "pairs.cf_answer", "b": "pairs.base_answer"}, {"op": "mul", "with": 0}]}
    }}
  },
  "saves": ["taken.g", "taken.g0", "taken.metric"]
}
```

`taken.g0` is `0.0` on all four rows, beside gradient norms of 0.3246 to 0.4240.

| what you see | what it is | the check | the fix |
|---|---|---|---|
| after adding a gradient null, the metric is 0 on every row and every gradient is exactly 0 | `{"op": "scale", "factor": 0}` alone at the head is a write; listed before the metric it zeroes the logits (all three values printed `0.0` on GPT-2) | is the null's entry an edit? | the metric's ops with `{"op": "mul", "with": 0}` appended |
| a `grad` entry is exactly zero on every row, null or not | its `of` does not depend on the value at its site (a layer's pattern does not depend on its `attention_values`) | which site `of` is computed from | an `of` the site feeds |
| a gradient map is nonzero, plausible, and does not predict patching | `of` names another differentiable entry, which runs and maps that entry; the `mul 0` null proves only that the gradient is of the entry zeroed, wired exactly | a central difference at one point ([check the gradient](../../nnspec-patching/references/attribution.md#check-the-gradient)); for the ranking, real patching of the top points | the `of` you meant |
| an attribution map misses a component patching shows restores the answer | the gradient is first order, and one patch restoring the whole answer is far from linear | real patching of the components you report | confirm with patching (`nnspec-patching`) |

## Sweeps

| what you see | what it is | the check | the fix |
|---|---|---|---|
| a grid of two sweeps came back as its diagonal | two sweeps with no `label` on fields of one name share the field's name as label, and one label is one axis | `nnspec show`'s result line: two `layers` sweeps of 2 values each read `layers  2 points, each [4,1]` | its own `label` on each: the same document reads `read_layer, write_layer  4 points, each [4,1]` |
| a `mean` or `sum` came back with one number per layer | a reduction keeps a `layers` axis unless `over` names it | the result's `dims` in `nnspec show` | `"over": ["layers", "rows", "pos"]` on a `[layers, rows, pos]` value |
| a swept step's saved read is one tensor, not one per point | a result is keyed by the sweep axes its own value depends on | the result's `axes` in `nnspec show` | read it as one value; sweeps are the `nnspec` skill's [sweeps reference](../../nnspec/references/sweeps.md) |

## Rows a read did not reach

A read that misses a row (an open index past a short row, a decode call the row never made) leaves it
`alignment_missing` and out of every reduction. The mechanism and the count of rows reached are the `nnspec`
skill's [positions reference](../../nnspec/references/positions.md#rows-a-scope-never-reached).

| what you see | what it is | the check | the fix |
|---|---|---|---|
| an accuracy or mean covers fewer rows than the dataset, or some rows are NaN | the read missed those rows, which every reduction skips | `—` in `nnspec show`, then the count in the positions reference | an index every row holds; in a generation `min_new_tokens`, or `{"frame": -1, "index": -1}` |
| a steered generation's mean effect looks diluted | a write over decode calls that never landed on a row leaves that row unsteered | the write's positions, as for a read | count the write's missing rows too |
| `contains` and a scope disagree on how many rows said a word | `contains` is a substring test; a scope matches whole words | the rows each counts | count whole words with a scope over the decode calls |
| `{"frame": -1}` read every prompt token on some rows | on a row whose first generated token was EOS, the last call is the prefill | the positions on that row | `{"frame": -1, "index": -1}` |

## Answers and metrics

GPT-2's byte-level tokenizer writes ` Paris` and `Paris` as two different one-token answers, so the bare
spelling runs:

```json
{
  "description": "One prompt scored against two spellings of its answer: ' Paris', as GPT-2 continues the prompt, and 'Paris' without the leading space. Both are one token on GPT-2's byte-level BPE, so neither is refused, and they are different tokens.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"facts": {"rows": [
    {"input": "The Eiffel Tower is in the city of", "spaced": " Paris", "bare": "Paris"},
    {"input": "Big Ben is in the city of", "spaced": " London", "bare": "London"}
  ]}},
  "steps": {
    "clean": {"kind": "forward", "input": "facts.input", "interventions": {
      "logits": {"site": {"component": "lm_head_output"}, "pos": -1}
    }},
    "p_spaced": {"kind": "op", "op": "token_prob", "of": "clean.logits", "token": "facts.spaced"},
    "p_bare": {"kind": "op", "op": "token_prob", "of": "clean.logits", "token": "facts.bare"}
  },
  "saves": ["p_spaced", "p_bare"]
}
```

```text
$ nnspec table out/answer_spelling p_spaced p_bare
row  pos  pos_token  input                               spaced   p_spaced             bare    p_bare
0    9    Ġof        The Eiffel Tower is in the city of   Paris   0.06997019797563553  Paris   1.0362419970988412e-06
1    6    Ġof        Big Ben is in the city of            London  0.01322225108742714  London  9.907823823596118e-07
```

`Rome` without its space is two tokens on GPT-2, and that row is refused ([refusals.md](refusals.md#answers)).

| what you see | what it is | the check | the fix |
|---|---|---|---|
| a metric near zero on every row, or scoring another token than the model says | the answer column lacks the leading space the model continues with | a `top_k` at the read, decoded, against the column | write the answer as the model continues the prompt, ` Paris` |
| `token_prob` higher than the probability of the token you meant | `strip` or `lower` makes each answer token a class (` a` and `a`; ` a` and ` A`), and `token_prob` sums the class | the column's flags | drop the flag where one spelling is meant |
| a probe has more classes than you meant | its classes are the distinct values of its `classes` column, compared exactly (`yes` and `Yes` are two) | the column's distinct values | clean the column |
| a mean over each row's entity favours long entities | a reduction over a ragged read averages every position, so a three-token entity weighs three times | the scope's widths in `nnspec show` | reduce `"over": ["pos"]` first, then over `rows` |

Answer columns are the `nnspec` skill's [data reference](../../nnspec/references/data.md).

## Precision and reproducibility

| what you see | what it is | the check | the fix |
|---|---|---|---|
| small effects at every head of a bfloat16 sweep, early layers included | bfloat16 rounding of the model's own forward; GPT-2's bfloat16 logit differences move in steps of 0.5 | `served:` in `nnspec show`; the same points in float32 | recompute the points you report in float32 |
| the last bits of a value differ between two runs | another batch size, padding, readout, thread count or device, or an op moved between an entry and an op step | the run records: `nnspec show`'s first line, `run.threads` in `manifest.json`; the two documents | compare runs at one setting and one form ([SKILL.md](../SKILL.md#reproducing-a-number)) |
| one row's last bits moved when the dataset changed (`take`, `skip`, a split) | under the default padding the input's longest row sets every batch's width | the step's `tokens` against `real` | compare on the same rows |
| an apply document differs from its fit's eval in the last bits | the fit's body runs at the fit's own `batch_size` | the two batch sizes | run the apply document at the fit's `batch_size` |
| two runs of one `generate` step said different things | options a step leaves out come from the checkpoint's generation config, which may sample, unseeded | the step's options | `"do_sample": false` |

## Featurizers

| what you see | what it is | the check | the fix |
|---|---|---|---|
| a step after a fit scores like an untrained featurizer | the bare name is the declared draw; the trained one is `<fit>.<name>` | compare with the fit's `epochs: 0` point: equal means the draw | name `<fit>.<name>` |
| a published SAE's latents, reconstruction or L0 look wrong | `centered`, a TopK file's `activation` and the layer in `site` are the document's word, unchecked against the file | its mean L0 over every token after BOS against the release's | state them from the release, the layer inline in `site` ([sae](../../nnspec-trained-featurizers/references/sae.md#splice-in-a-published-dictionary)) |
| splicing an SAE in moved the output with no edit | `apply` replaces the value with the reconstruction, dropping the error term | the spliced run against the clean run | an `encode` … `decode` bracket, which keeps the error |
| an adapter or an always-on write changed the prompt and not the generated tokens | in a `generate` step a `pos` without a `frame` is the prefill only | the entry's positions | `"frame": ":"` |

Featurizers are the `nnspec-trained-featurizers` skill.

## Output

| what you see | what it is | the check | the fix |
|---|---|---|---|
| a fixed document still shows the old numbers | a refused run writes nothing, so the directory holds the last finished run | `nnspec show`'s first line: the document's name and `started` time | read the refusal and rerun |
| `nnspec table` refuses a value the run saved, `'<x>' was not saved` | a later run into the same `--out` replaced the manifest; the old files stay beside it | the manifest's `saves` | give each run its own `--out` |
