# Running and engines

`nnspec run DOC` compiles a document, loads its checkpoint on an engine, runs every step, and writes the saved
values to an output directory. `nnspec.run` does the same from Python and returns the filled plan. A refusal
prints `nnspec: <where>: <what>`, exits 1 and writes nothing, except a step that runs out of CUDA memory
([debugging](debugging.md#out-of-memory)).

## `nnspec run`

```text
nnspec run [--engine ENGINE] [--out OUT] [--batch-size N] [--padding dataset|batch|sorted] [--readout full|at-reads] [--device D] [-o KEY=VALUE ...] DOCUMENT
```

| flag | default | meaning |
|---|---|---|
| `--engine` | `nnterp` | `nnterp`, `hooks`, `ndif` or `vllm` ([below](#engines)) |
| `--out` | `out/<document name>` under the working directory | the output directory, created when missing and never cleared |
| `--batch-size` | `8` | rows per forward or generation call; a fit's body runs at the fit's own `batch_size` |
| `--padding` | `dataset` | what each batch is padded to, on the left ([below](#batch-size-padding-and-readout)) |
| `--readout` | `full` | where a forward computes the logits it reads |
| `--device` | `cpu`; `cuda:0` on vllm | one torch device (`cuda:0`) or a device map transformers computes (`auto`, `balanced`, …) |
| `-o KEY=VALUE` | none | one of the engine's own options, repeatable; the value is read as JSON, else as text |

- **`nnspec describe engines` lists each engine's options.** A flag, device or option that cannot be is refused before the checkpoint loads, naming what is valid.
- **`HF_HUB_OFFLINE=1` runs from the cache** once a first online run downloaded the checkpoint; offline, a `revision` the cache lacks is refused as the engine loads it.
- **Lines transformers and torch print while loading are theirs**, such as a note that `pad_token_id` may be ignored: nnspec pads each batch itself.
- **`nnspec compile DOC` is the first stage alone**: no engine, no model, no output directory. It reads the data files, a hub dataset's rows and the bundle files, prints each run of a step in run order with its kind, its input, the sweep point it runs at and the points it serves, and exits 0, or prints the refusal `nnspec run` would print. It cannot check what only the model knows: layers, `select` indices, positions and answer tokens are checked after the model loads, before any forward.

## From Python

```python
import nnspec
from nnspec import Plan
from nnspec.engines import load

plan = nnspec.run("documents/ioi_patch.json", out="out/ioi_patch", batch_size=8)    # flags as keywords
plan.results["patched_ld"]                                                          # tensor [4, 1]

engine = load("nnterp", Plan("documents/ioi_patch.json").model, "cpu")   # load once
for name in ["ioi_patch", "ioi_layer_sweep"]:                             # documents pinning one model block
    nnspec.run(f"documents/{name}.json", engine=engine, out=f"out/{name}")

doc = {"model": {...}, "data": {...}, "steps": {...}, "saves": [...]}     # a document built in Python
Plan(**doc).write("documents/built.json")                                 # as a file, or nnspec.run(Plan(**doc), out=...)
```

- **`nnspec.run` takes the flags as arguments** (`engine=`, `device=`, `batch_size=`, `padding=`, `readout=`) and the engine's options as keywords (`remote="local"`, `tp=2`).
- **A refusal raises `ValueError`** (`nnspec.plan.PlanError`, a `ValueError`, at compile), and a missing document `FileNotFoundError`. `Plan(path).compile()` is `nnspec compile`.
- **A loaded engine runs only documents with its exact `model` block**, and keeps the device and options it loaded with. One load serving many documents saves a model load per document.
- **`Plan(**fields)` builds a document from Python values**, the way to write many similar entries, rows from a script or one site per chosen head; `Plan(dict)` is refused. Such a plan is named `plan`, writes `out/plan` by default, and resolves files against the working directory.
- **`nnspec.describe("ops")` is `nnspec describe ops`** as a string, and `nnspec.cli.main(["table", "out/x", "ld"])` runs one command and returns its exit status.
- **A script that runs vllm needs `if __name__ == "__main__":`**, since vLLM spawns a process that imports it.

## Batch size, padding and readout

| `--padding` | each batch padded to | use it for |
|---|---|---|
| `dataset` (default) | the longest row of the step's whole input | numbers you compare, featurizers you train |
| `batch` | its own longest row | fewer tokens fed, rows in order |
| `sorted` | its own longest row, rows longest first, values put back in row order | the fewest tokens fed, on a large harvest |

- **Padding is never a position**, so results, positions and tables are laid out alike under all three. A large gap between a step's `tokens` and `real` in `nnspec show` is padding.
- **`--readout at-reads`** computes the head only at the positions a forward reads of `logits`. Use it on long rows read at a few positions, where the head over every position is most of a forward's memory. It changes nothing where a forward reads `lm_head_output`, every position, or a `grad` of the logits, and moves the last bits where it acts.

## When two runs give the same numbers

One document at one batch size, padding, readout, device and CPU thread count writes byte-identical value
files; change one and a row's last bits can move. The run records each setting, and `nnspec show` prints all
but the threads. What moves bits, and how to compare two runs or two engines, is
[reproducing a number](debugging.md#reproducing-a-number).

## Where the model runs

- **`--device` is a run option, never a document key.** `device`, `device_map`, `torch_dtype`, `tp_size` and `load_in_4bit` in the `model` block are refused at compile, each naming its spelling: `--device`, `dtype`, `-o tp=N`, a quantized `dtype`.
- **A device map may spread the model over GPUs**; each batch goes to the first parameter's device.
- **`nnspec show`'s `served:` line is where and in what dtype the weights actually were**: `served: float32 on cpu`.
- **Run a `bfloat16` model on a GPU**: a CPU without bfloat16 matrix instructions runs it much slower than float32, and its effects round ([the float32 rule](debugging.md#the-float32-rule)).

**Quantized dtypes.** `model.dtype` may name a quantization: `nf4` or `fp4` (bitsandbytes' 4 bits) and `int8`,
on the CPU or CUDA with `bitsandbytes` installed, and `fp8` on a GPU of compute capability 8.9 or later. A
quantized model computes in `bfloat16` (`int8` in `float16`): a GPT-2 document at `"dtype": "nf4"` runs on the
CPU, `served: bfloat16, uint8 on cpu`, with its logit differences in steps of 0.5 and its null still equal to
its clean run.

**`-o tp=N`** shards the model over N ranks by transformers' tensor parallelism, on the nnterp engine, under
torchrun. Use it for a checkpoint that does not fit on one GPU when the run needs nothing refused below; a
device map (`--device auto`) is the other way to spread one.

```bash
torchrun --nproc-per-node=3 --no-python nnspec run documents/x.json --device cuda -o tp=3
```

- **Every rank runs the whole plan**, and rank 0 alone writes and prints. Outside torchrun it is refused, printing the command.
- **The numbers are the unsharded ones to rounding**, not to the bit.
- **Refused under `tp`**: a checkpoint whose config gives no tensor-parallel plan, a value the ranks hold in pieces (the attention interior but `attention_premix`, `mlp_activation`), a bitsandbytes `dtype`, `grad`, and `tp` on hooks or ndif.

## Engines

| engine | runs the model | serves | options |
|---|---|---|---|
| `nnterp` (default) | here, as nnterp's standardized model in one nnsight session | every component nnterp serves on the checkpoint, the attention interior under eager | `tp` |
| `hooks` | here, as a plain transformers model with forward hooks, no nnsight at run time | components at a module boundary only | none |
| `ndif` | on an NDIF server; the client holds a model with no weights | what nnterp serves on the server's model | `remote`: `true` or `"local"` |
| `vllm` | in vLLM's worker on the visible GPUs, one request per row, no padding | the stream, the sublayers, the attention interior (its pattern read-only), the head at the last position; no gradient, no fit | `tp`, `memory`, `max_len`, `invariant` |

**hooks is the cross-check.** Positions, edits and ops are nnspec's own on both engines, so a result that
agrees on hooks does not depend on how the model was reached: on the CPU in float32 a GPT-2 patch writes
byte-identical value files on nnterp, on hooks and on ndif under `-o remote=local`
([cross-checking engines](debugging.md#cross-checking-engines)).

- **Only `readout` differs**: on hooks it is the final norm, the head and the config's logit softcap; a family that scales its logits after the head another way differs there.
- **Values inside a module's forward are refused** after the model loads: the attention interior, and a mixture's `router_logits`, `expert_weights`, `expert_indices` and `expert_outputs`.

**ndif is a server.**

- **The whole run goes as one request**; the server runs every step, and the client writes the output directory.
- **The server is nnsight's**: `NDIF_HOST` and `NDIF_API_KEY`, or `nnsight.CONFIG`.
- **`-o remote=local` makes the same trip in this process** without a server; it proves the request is self-contained, and the weights must fit here.
- **The server needs nnspec at your version**, `datasets` for a hub dataset, and the checkpoint deployed at the document's `key` and `revision`. File datasets, bundles and loaded values travel in the request, at most 512 MiB of loaded tensors.
- **The server places the model and picks its dtype**: any `--device` but `cpu` is refused, and so are quantized dtypes and `tp`. Nothing compares the deployment's float dtype with the document's, so check `nnspec show`'s `served:` and `versions:` lines after a remote run: they are the server's.
- **A refusal raised on the server arrives as nnsight's `RemoteError`**, and `nnspec run` prints its last line.

**vllm runs vLLM's kernels.** It is not run in this skill; these lines are `nnspec describe engines`' and the
library's docs.

- **It needs a GPU and vLLM installed** at the version nnsight instruments; `--device` is `cuda:0` and nothing else.
- **`max_new_tokens` is required on a generate step**, and other options map onto vLLM's sampling.
- **The head is the last position's** in the prefill, `attention_probabilities` and `attention_scores` are read-only, and a fit or a `grad` is refused.
- **Under the default `invariant: true` a run repeats its bits** at any batch size; `invariant: false` runs vLLM's fastest kernels, whose last bits move between runs. Compare vllm with vllm, and with nnterp on outcomes.
- **`memory` (0.9) is the GPU fraction vLLM reserves**; the run's own values live outside it on `cuda:0`, so lower it when they run out of memory there. **`max_len`** caps prompt plus new tokens, by default the checkpoint's.

## Memory

- **Every value a step binds stays on the model's device until the run ends**; save and name fewer values, or score inside an entry ([a step or an entry](ops.md#a-step-or-an-entry)).
- **A fit holds the minibatch's backward, the scored pass over every training row, and every eval's values**; only the first shrinks with its `batch_size`.

## Where output goes

- **`--out` names the directory**; without it, `out/<document name>` under the working directory.
- **A second run into one directory replaces the files it writes and leaves the others**, and the manifest names only the last run's: give each run its own `--out`.
- **Data paths, bundle files and `load` directories resolve against the corpus root**: the nearest directory holding `data/`, from the document's own directory up; with none, the directory above the document's own.
- **So `project/documents/x.json` resolves against `project/`** with no `data/`, and a document at the top of a folder needs an empty `data/` beside it. A document of inline rows that loads nothing needs neither.
- **Run from the corpus root**, so the default `out/` is where a later `load` or `file_path` looks, and `nnspec table` and `nnspec show` find the `data/` files and bundles they read again.
