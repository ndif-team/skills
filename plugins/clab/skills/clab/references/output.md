# Output

A run writes one directory: a file per saved value, `manifest.json` and `tokens.json`. `clab show` prints how
the run went, `clab table` prints saved values as rows labelled with their tokens and data, and `Plan.load`
reads it all back in Python. The examples are the clab skill's worked document, `documents/ioi_patch.json`,
and the documents of the other references.

## The directory

```text
out/ioi_patch/
  clean_ld.safetensors
  flipped.safetensors
  manifest.json
  null_ld.safetensors
  patched_ld.safetensors
  tokens.json
```

- **One file per name in `saves`**, named after it: `<name>.safetensors` for a tensor or a dict of tensors, `<name>.json` for a value that holds text (`detokenize`, a `top_k` step with its `tokens`).
- **A swept value is one file** with an entry per point of the sweep axes it depends on, keyed `layers=8` or `src=6,dst=8`.
- **A saved trained featurizer is a bundle** that another document loads ([featurizers and fit](featurizers-and-fit.md#bundles)).
- **`manifest.json` is the compiled plan as the run completed it**, with the run record and a record of each save; it holds no values.
- **`tokens.json` is every input's tokens per row and every generation's new tokens**; a directory without it is refused.
- **A run never clears its directory**: files of an earlier run stay beside the new ones, and the new manifest names only its own.

## `clab show`

```text
$ clab show out/ioi_patch
ioi_patch: run on nnterp, device cpu, tp 1, batch size 8, padding dataset, readout full, started …, … s
  served: float32 on cpu
  versions: clab …, nnsight …, nnterp …, torch …, transformers …, python …
  step            at  serves  seconds  memory  forwards  tokens  real
  clean           —   —       …        —       1         56      56
  counterfactual  —   —       …        —       1         56      56
  null            —   —       …        —       1         56      56
  patched         —   —       …        —       1         56      56
  clean_ld        —   —       …        —       0         0       0
  ...
  entry                at  positions
  clean.v_self         —   [13] on all 4 rows
  clean.logits         —   [13] on all 4 rows
  counterfactual.v_cf  —   [13] on all 4 rows
  null.patch           —   [13] on all 4 rows
  ...
  result      step        dims       axes  shape
  clean_ld    clean_ld    rows, pos  —     [4,1]
  null_ld     null_ld     rows, pos  —     [4,1]
  patched_ld  patched_ld  rows, pos  —     [4,1]
  flipped     flipped     rows, pos  —     [4,1]
```

- **The first lines are the run record**: the engine, device, options, batch size, padding, readout, start and duration; `served:` is the dtype and device the weights were in; `versions:` the packages that ran it.
- **The step table is one row per run of a step**, at the point that ran it; `serves` lists the points that run served (`all 12 points`).
- **A step's `at` is the first point of the forward it runs in.** A step that takes an operand from a forward with a swept entry is listed at that point with `serves all N points`, though its own value is unswept. The result table's `axes` say what is swept.
- **Then come its seconds, peak CUDA memory** (`—` off CUDA), model calls (`forwards`), tokens fed with padding (`tokens`) and the rows' own tokens (`real`).
- **The entry table is where every read and write landed**, per row in row order: `[13] on all 4 rows` where rows agree, each row's list otherwise, `—` for a row never reached. Check it before trusting a number.
- **The result table is each save's step, layout (`dims`), sweep `axes` and shape**, counted over points when swept (`12 points, each []`).
- **A `stopped:` line follows the record** when a step ran out of CUDA memory.
- **Seconds vary from run to run**; compare tokens and forwards, not seconds.

## `clab table`

`clab table OUT NAME [NAME ...]` prints saved values as rows, one per example and position; `--csv` prints CSV
with every cell whole, where the aligned form cuts cells at 48 characters, as it does a long `top_k.ids` list.

| column | present when | holds |
|---|---|---|
| `at` | the value is swept | the point's coordinates, `{"layers":8}` ([sweeps](sweeps.md#reading-swept-results)) |
| `epochs` | a fit's `history` | the epoch, from 0 |
| `example_id` | the dataset has that column | the row's `example_id` |
| `row` | the value has rows | the row's index after `split`, `shuffle`, `skip` and `take` |
| `layer` | the value has a `layers` axis | the model's layer, not the index along the axis |
| `pos`, `pos_token` | the value has positions | on a value with rows, the position the read landed on and its token; without rows, the index within the read ([positions](positions.md#static-width-and-ragged-rows)) |
| `call` | the read was in a generation | the forward call: 0 the prefill, k decode call k |
| `key`, `key_token` | an attention pattern | each key position and its token, the row's own keys only |
| dataset columns | the value has rows | the columns its chain of steps used: the input, the answers |
| `value` | one name | the element, or the remaining axes as nested lists; with several names, one column each |
| `alignment_missing` | some row was not reached | `true` on a row a read beyond the prefill never reached |
| `k` | a `top_k` over `rows` | the pick's rank, from 0 |

- **Remaining axes print as nested lists, in the value's order**: `heads_select.json`'s `f.size` prints `[4.627506256103516,5.233008861541748]` per row, index 0 head 9 and index 1 head 6, the select list's order.
- **A value without rows prints one row per element**: `rate spread` is one row, `1.0  0.9305830001831055`; `lens_by_layer` prints a `layer` column, 0 to 11.
- **Several names join on the keys they share** (sweep coordinates, `row`, `layer`, `pos`, `key`) and broadcast over the rest: a fit's `held_acc` repeats beside every epoch of `fit.history.ce`.
- **Values swept over different axes cross**, a row at every pair of points; table them one at a time.
- **Two reads of one input share a row only where their positions coincide**: `f.landmark` and `f.landmark_last` of `ragged_scope.json` table together as the three last-token rows. Two reads that share no position print nothing and exit 0.
- **Positions past a ragged row's own are left out**, not printed as NaN.
- **A row never reached prints flagged, with no value**, even where the saved tensor holds a number there: a `count` of a missing read is 0 in the file and blank in the table.
- **Only saved names table**: `'patched.logits' was not saved; add it to saves`.
- **A bundle does not table**: `'fit.probe' is a trained featurizer's bundle, which has no rows to table; read it from plan.results`.
- **`clab table` joins values of one input's rows.** A metric of the base forward and one of the counterfactual's are refused together (`… the column does not mean one thing for both, so they do not join; table them one at a time`), and so are rows of two datasets. Table each input separately, or join their means over rows.

## In Python

<!-- test: skip -->
```python
from clab import Plan

plan = Plan.load("out/ioi_patch")                     # Plan(ioi_patch, run on nnterp, 4 results)
plan.results["patched_ld"][:, 0]                      # tensor([1.2577, 4.4744, 2.5320, 3.9496])
plan.table("patched_ld")[0]                           # {'example_id': 'mary', 'row': 0, 'pos': 13, 'pos_token': 'Ġto', 'input': ..., 'cf_answer': ' John', 'base_answer': ' Mary', 'value': 1.2577...}
plan.table("patched_ld", columns=[])[0]               # the same row without the dataset columns
plan.steps["patched"].interventions["patch"].positions   # {'pairs': [[13], [13], [13], [13]]}
plan.run.costs["patched"]                             # {'seconds': ..., 'memory': None, 'forwards': 1, 'tokens': 56, 'real': 56}
plan.run.sizes["num_layers"]                          # 12; run.threads is torch's thread count at the run
plan.run.tokens["pairs.input"][0][-3:]                # ['Ġa', 'Ġdrink', 'Ġto']
```

- **`results`** holds each save by name: a tensor, a dict of tensors, or a list or dict where it holds text. A swept value is a dict keyed by coordinate: the value for one axis, a tuple for several.
- **`table(*names, columns=None)`** returns the rows `clab table` prints, as dicts; `columns=` chooses the dataset columns, which the command line cannot.
- **`steps[<step>].interventions[<entry>].positions`** is each row's positions under the dataset, or `"alignment_missing"`; a swept step is a list of points, each with its `at`.
- **`run`** holds `engine`, `batch_size`, `padding`, `readout`, `device`, `options`, `versions`, `threads`, `dtype`, `devices`, `sizes`, `started`, `seconds`, `tokens`, `costs` and `stopped`.
- **`Plan.load` needs no data files**, but a run that loaded a bundle loads only where its `file_path` resolves.
- **What no op does** (an interval, a ranking across points, a join across runs) is done here, on `results`.

A `top_k` over several axes has no rows to table. Read it with the python clab is installed in; for
[ops](ops.md#transform)' `documents/head_rank.json`:

<!-- test: skip -->
```python
from clab import Plan

top = Plan.load("out/head_rank").results["top"]       # {'ids': {'layers': ..., 'num_heads': ...}, 'values': ...}
for layer, head, size in zip(top["ids"]["layers"].tolist(), top["ids"]["num_heads"].tolist(), top["values"].tolist()):
    print(layer, head, size)                         # 10 7 7.827009201049805, then 11 0 …, 11 8 …
```

Each `ids` entry counts along the value's own axes, so map it back through a selection (`"6:12"`: add 6).

On a swept document `plan.steps[<step>]` is a list of points, each with its `at`, every sweep's index into its
values. With `ioi_patch.json`'s `v_cf` and `patch` both at `{"index": {"sweep": [-1, -2], "label": "p"}}`, run into
`out/ioi_pos_sweep`, zip the read's points with the write's to check that they moved together:

<!-- test: skip -->
```python
plan = Plan.load("out/ioi_pos_sweep")
for read, write in zip(plan.steps["counterfactual"], plan.steps["patched"]):
    print(read.at, read.interventions["v_cf"].positions["pairs"][0], write.interventions["patch"].positions["pairs"][0])
# {'p': 0} [13] [13], then {'p': 1} [12] [12]
```

## The manifest

| key | holds |
|---|---|
| `description`, `model`, `data`, `steps`, `saves`, `document`, `compiled` | the compiled plan: sites inlined, sweeps expanded, each record's derived fields filled in; `data` holds each dataset's `sha256`, `n_rows` and inline rows |
| `run` | the run record, without its `tokens` |
| `results` | each save's `file`, the `step` that made it, `dims`, `dataset`, the `read` whose positions it holds, `width`, `axes`, `shape`, `dtype` |
| `positions` | each distinct list of resolved positions once, which a record names by index |

`manifest.json`'s `results.patched_ld` reads `{'file': 'patched_ld.safetensors', 'step': 'patched_ld', 'dims':
['rows', 'pos'], 'dataset': 'pairs', 'read': 'patched.logits', 'width': 1, 'axes': [], 'shape': [4, 1], 'dtype':
'float32'}`. Each entry's record holds `positions: {"pairs": 0}`, an index into the top-level list, and a forward's
record holds its `schedule`. A field at its default is left out, so read the manifest through
`Plan.load` or `clab.output.manifest(out)`, which put each record's positions back in place.

## A saved value elsewhere

<!-- test: skip -->
```python
from safetensors.torch import load_file

load_file("out/ioi_patch/patched_ld.safetensors")     # {'patched_ld': tensor([[1.2577], [4.4744], [2.5320], [3.9496]])}
```

- **A file holds the value under its save's name**; a swept file one tensor per point, keyed by coordinate; a dict value (`top_k`'s) one per member.
- **The layout is the manifest's `dims`**: `[rows, pos]` here, rows in the dataset's order after narrowing.
- **A `.json` save** (text) reads with `json.load`.
- **Another document brings a saved value in with a `load` step**, `{"kind": "load", "from": "out/x", "value": "<save>"}`, with its layout, dataset and sweep axes; the cached-harvest recipe is the `clab-trained-featurizers` skill's.

## Tabling needs the rows

`clab table` and `plan.table` label rows with dataset columns, so they need each dataset's rows. Inline rows are
in the manifest, so a run of inline rows tables from any directory. A file's rows are read again from `data/` under
the working directory and checked against the digest the run recorded; a hub dataset's from the hub or its cache.
Run `clab table` from the corpus root and leave the data as it was. `clab show` and `plan.results` need no rows.
