# Output

A run writes one directory: a file per saved value, `manifest.json` and `tokens.json`. `nnspec show` prints how
the run went, `nnspec table` prints saved values as rows labelled with their tokens and data, and `Plan.load`
reads it all back in Python. The examples read `out/ioi_patch`, the run of
[SKILL.md](../SKILL.md#a-complete-experiment)'s document.

## The directory

- **One file per name in `saves`**, named after it: `<name>.safetensors` for a tensor or a dict of tensors, `<name>.json` for a value that holds text (`detokenize`, a `top_k` step with its `tokens`).
- **A swept value is one file** with an entry per point of the sweep axes it depends on, keyed `layers=8` or `src=6,dst=8`.
- **A saved trained featurizer is a bundle** that another document loads ([bundles](featurizers.md#bundles)).
- **`manifest.json` is the compiled plan as the run completed it**, with the run record and a record of each save; it holds no values. **`tokens.json`** is every input's tokens per row and every generation's new tokens.
- **A run never clears its directory**: files of an earlier run stay beside the new ones, and the new manifest names only its own.

## `nnspec show`

```text
$ nnspec show out/ioi_patch
ioi_patch: run on nnterp, device cpu, tp 1, batch size 8, padding dataset, readout full, started …, … s
  served: float32 on cpu
  versions: nnspec …, nnsight …, nnterp …, torch …, transformers …, python …
  step            at  serves  seconds  memory  forwards  tokens  real
  clean           —   —       …        —       1         56      56
  ...
  clean_ld        —   —       …        —       0         0       0
  ...
  entry                at  positions
  clean.v_self         —   [13] on all 4 rows
  ...
  patched.patch        —   [13] on all 4 rows
  result      step        dims       axes  shape
  clean_ld    clean_ld    rows, pos  —     [4,1]
  ...
```

- **The first lines are the run record**: the engine, device, options, batch size, padding, readout, start and duration; `served:` is the dtype and device the weights were in; `versions:` the packages that ran it. A new `started` time says the directory holds the run you just made.
- **The step table is one row per run of a step**, at the point that ran it, with the points that run served (`all 12 points`), its seconds, peak CUDA memory (`—` off CUDA), model calls (`forwards`), tokens fed with padding (`tokens`) and the rows' own (`real`). The batch size in the first line is the forwards'; a fit's body runs at the fit's own.
- **The entry table is where every read and write landed**, per row in row order: `[13] on all 4 rows` where rows agree, each row's list otherwise, `—` for a row never reached. Check it before trusting a number.
- **The result table is each save's step, layout (`dims`), sweep `axes` and shape**, counted over points when swept (`12 points, each []`). Its `axes` say what is swept: a grid that zipped has one axis where you meant two.
- **A `stopped:` line follows the record** when a step ran out of CUDA memory.

## `nnspec table`

`nnspec table OUT NAME [NAME ...]` prints saved values as rows, one per example and position; `--csv` prints CSV
with every cell whole, where the aligned form cuts cells at 48 characters.

| column | present when | holds |
|---|---|---|
| `at` | the value is swept | the point's coordinates, `{"layers":8}` |
| `epochs` | a fit's `history` | the epoch, from 0 |
| `example_id` | the dataset has that column | the row's `example_id` |
| `row` | the value has rows | the row's index after `split`, `shuffle`, `skip` and `take` |
| `layer` | the value has a `layers` axis | the model's layer, not the index along the axis |
| `pos`, `pos_token` | the value has positions | on a value with rows, the position the read landed on and its token; without rows, the index within the read |
| `call` | the read was in a generation | the forward call: 0 the prefill, k decode call k |
| `key`, `key_token` | an attention pattern | each key position and its token, the row's own keys only |
| dataset columns | the value has rows | the columns its chain of steps used: the input, the answers |
| `value` | one name | the element, or the remaining axes as nested lists; with several names, one column each |
| `alignment_missing` | some row was not reached | `true` on a row a read beyond the prefill never reached |
| `k` | a `top_k` over `rows` | the pick's rank, from 0 |

- **Several names join on the keys they share** (sweep coordinates, `row`, `layer`, `pos`, `key`) and broadcast over the rest: a fit's `held_acc` repeats beside every epoch of its history.
- **Values that do not join are refused, and the fix is to table them one at a time**: metrics read on two inputs (a base forward's and a counterfactual's), rows of two datasets, two reads of one input that share no position, values at layers neither holds. Values swept over different axes cross, a row at every pair of points.
- **Remaining axes print as nested lists, in the value's order**: under `"select": {"num_heads": [9, 6]}`, index 0 is head 9.
- **Positions past a ragged row's own are left out**, not printed as NaN, and a row never reached prints flagged, with no value, even where the saved tensor holds one: a `count` of a missing read is 0 in the file and blank in the table.
- **Only saved names table**, and a bundle or a ranking with no rows does not: read those from `plan.results`.
- **A file dataset's rows are read again** from `data/` under the working directory and checked against the digest the run recorded, so run `nnspec table` from the corpus root and leave the data as it was. Inline rows are in the manifest, and `nnspec show` and `plan.results` need no rows.

## In Python

```python
from nnspec import Plan

plan = Plan.load("out/ioi_patch")                     # Plan(ioi_patch, run on nnterp, 4 results)
plan.results["patched_ld"][:, 0]                      # tensor([1.2577, 4.4744, 2.5320, 3.9496])
plan.table("patched_ld")[0]                           # {'row': 0, 'pos': 13, 'pos_token': 'Ġto', 'input': ..., 'cf_answer': ' John', 'base_answer': ' Mary', 'value': 1.2577...}
plan.table("patched_ld", columns=[])[0]               # the same row without the dataset columns
plan.steps["patched"].interventions["patch"].positions   # {'pairs': [[13], [13], [13], [13]]}
plan.run.costs["patched"]                             # {'seconds': ..., 'memory': None, 'forwards': 1, 'tokens': 56, 'real': 56}
plan.run.sizes["num_layers"]                          # 12
plan.run.tokens["pairs.input"][0][-3:]                # ['Ġa', 'Ġdrink', 'Ġto']
```

- **`results`** holds each save by name: a tensor, a dict of tensors (a `top_k`'s `ids` and `values`, a bundle's tensors), or a list or dict where it holds text.
- **A swept value is a dict keyed by coordinate**: the swept value itself for one axis (`results["mean_patched"][8]`), a tuple in the order of `plan.sweeps` for several (`results["mean_ld"][(6, 8)]`). A text save swept over `dose` reads back the same way, and its `.json` file is keyed `dose=0.5` ([reading swept results](sweeps.md#reading-swept-results)).
- **`table(*names, columns=None)`** returns the rows `nnspec table` prints, as dicts, a swept row carrying its `at`; `columns=` chooses the dataset columns, which the command line cannot.
- **`steps[<step>].interventions[<entry>].positions`** is each row's positions under the dataset, or `"alignment_missing"`. A swept step is a list of points, each with its `at`, every sweep's index into its values.
- **`run`** holds `engine`, `batch_size`, `padding`, `readout`, `device`, `options`, `versions`, `threads`, `dtype`, `devices`, `sizes`, `started`, `seconds`, `tokens`, `costs` and `stopped`.
- **A ragged value is as wide as its widest row**, NaN past each row's own positions; the entry's `positions` say how many each row holds.
- **`Plan.load` needs no data files**, but a run that loaded a bundle loads only where its `file_path` resolves.

A `top_k` over several axes has no rows to table. Its `ids` hold one tensor per ranked axis, each counting
along the value's own axes, so map an index back through a selection (`"6:12"`: add 6):

```python
top = Plan.load("out/attention_stripes").results["top_induction"]   # {'ids': {'layers': ..., 'num_heads': ...}, 'values': ...}
for layer, head, value in zip(top["ids"]["layers"].tolist(), top["ids"]["num_heads"].tolist(), top["values"].tolist()):
    print(layer, head, value)                                       # the stripes document of the attention guide
```

To check that a swept read and its write moved together, zip their points:

```python
plan = Plan.load("out/layer_pos_map")                 # the map of the patching guide
for read, write in zip(plan.steps["counterfactual"], plan.steps["patched"]):
    print(read.at, read.interventions["v_cf"].positions["pairs"][0], write.interventions["patch"].positions["pairs"][0])
```

## The manifest and the files

`manifest.json` holds the compiled plan (`description`, `model`, `data`, `steps`, `saves`, `document`,
`compiled`), the run record under `run`, and under `results` each save's `file`, the `step` that made it,
`dims`, `dataset`, the `read` whose positions it holds, `width`, `axes`, `shape` and `dtype`. A field at its
default is left out and positions are stored once in a shared list, so read it through `Plan.load` or
`nnspec.output.manifest(out)`, which returns the manifest with each record's positions back in place, and the
`results` records beside it.

```python
from safetensors.torch import load_file

load_file("out/ioi_patch/patched_ld.safetensors")     # {'patched_ld': tensor([[1.2577], [4.4744], [2.5320], [3.9496]])}
```

- **A file holds the value under its save's name**; a swept file one tensor per point, keyed by coordinate; a dict value one per member. The layout is the manifest's `dims`, rows in the dataset's order after narrowing.
- **Another document brings a saved value in with a `load` step**, `{"kind": "load", "from": "out/x", "value": "<save>"}`, with its layout, dataset and sweep axes ([cached harvests](featurizers-bundles-and-harvests.md#cached-harvests)).
