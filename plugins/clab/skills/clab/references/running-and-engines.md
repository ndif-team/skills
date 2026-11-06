# Running and engines

`clab run DOC` compiles a document, loads its checkpoint on an engine, runs every step, and writes the saved
values to an output directory. `clab.run` does the same from Python and returns the filled plan. A refusal prints
`clab: <where>: <what>`, exits 1 and writes nothing, except a step that runs out of CUDA memory.

## `clab run`

```text
clab run [--engine ENGINE] [--out OUT] [--batch-size N] [--padding dataset|batch|sorted] [--readout full|at-reads] [--device D] [-o KEY=VALUE ...] DOCUMENT
```

| flag | default | meaning |
|---|---|---|
| `--engine` | `nnterp` | `nnterp`, `hooks`, `ndif` or `vllm` ([below](#engines)) |
| `--out` | `out/<document name>` under the working directory | the output directory, created when missing and never cleared |
| `--batch-size` | `8` | rows per forward or generation call |
| `--padding` | `dataset` | what each batch is padded to, on the left ([below](#batch-size-padding-and-readout)) |
| `--readout` | `full` | where a forward computes the logits it reads |
| `--device` | `cpu`; `cuda:0` on vllm | one torch device (`cuda:0`) or a device map transformers computes (`auto`, `balanced`, …) |
| `-o KEY=VALUE` | none | one of the engine's own options, repeatable; the value is read as JSON, else as text |

- **`HF_HUB_OFFLINE=1` runs from the cache** once a first online run downloaded the checkpoint; offline, a checkpoint the cache lacks is refused as the engine loads it.
- **Lines transformers and torch print while loading are theirs**, such as a note that `pad_token_id` may be ignored: clab pads each batch itself.
- **`clab describe engines` lists each engine's options.**

## Which stage refuses

| stage | checks | for example |
|---|---|---|
| `clab compile`, no model | every field, reference, column, save and sweep, the data, bundle headers | `steps.clean_ld.of: 'clean.logits' names no value produced before this step; known: none` |
| `clab run`, before the checkpoint loads | the flags, the device, the engine's options | `device 'gpu': not a torch device ('cpu', 'cuda:0') or a device map transformers computes (…)` |
| after the model loads, before any forward | layers, `select` indices, positions and answer tokens against the model; components the engine serves | `clean.v_self: layer 12; the model has 12 layers` |
| when the step runs | an option `generate` rejects; running out of CUDA memory | `answer: generate rejected 'seed': …` |

`clab compile DOC` is the first stage alone: no engine, no model, no output directory. It prints each run of a
step in run order, with its kind, its input, the sweep point it runs at and the points it serves, and exits 0.
`ioi_broken.json`, the same document with `patched.logit` for `patched.logits`, is refused:

```text
$ clab compile documents/ioi_patch.json
ioi_patch: compiled, not run
  step            kind     input                 at  serves
  clean           forward  pairs.input           —   —
  counterfactual  forward  pairs.counterfactual  —   —
  ...
  flipped         op       —                     —   —
$ clab compile documents/ioi_broken.json
clab: steps.patched_ld.of: 'patched.logit' names no value produced before this step; known: clean.logits, …
```

- **A refusal is the line `clab run` would print**, on stderr, with exit status 1.
- **It reads the data files, a hub dataset's rows and the bundle files**; under `HF_HUB_OFFLINE=1` a hub dataset's revision must be in the cache.
- **It imports transformers only to check a `generate` step's options**, and loads no checkpoint.
- **It takes the document alone**: `clab run`'s flags and `-o` options are checked by `clab run`.
- **From Python** it is `Plan(path).compile()`, which raises `clab.plan.PlanError` with the same message.

Reading a refusal and fixing it is the `clab-debugging` skill's.

## From Python

<!-- test: skip -->
```python
import clab
from clab import Plan
from clab.engines import load

plan = clab.run("documents/ioi_patch.json", out="out/ioi_patch", batch_size=8)      # flags as keywords
plan.results["patched_ld"]                                                          # tensor [4, 1]

engine = load("nnterp", Plan("documents/ioi_patch.json").model, "cpu")   # load once
for name in ["ioi_patch", "ioi_layer_sweep"]:                             # documents pinning one model block
    clab.run(f"documents/{name}.json", engine=engine, out=f"out/{name}")
```

- **`clab.run` takes the flags as arguments** (`engine=`, `device=`, `batch_size=`, `padding=`, `readout=`) and the engine's options as keywords (`remote="local"`, `tp=2`).
- **A refusal raises `ValueError`** (`PlanError` at compile), and a missing document `FileNotFoundError`.
- **A loaded engine runs only documents with its exact `model` block**, and keeps the device and options it loaded with.
- **The four calls behind it** are `Plan(path).compile()`, `load(name, compiled.model, device)`, `engine.execute(compiled, batch_size=8)` and `filled.dump(out)`, for stopping between stages.
- **A script that runs vllm needs `if __name__ == "__main__":`**, since vLLM spawns a process that imports it.

## Batch size, padding and readout

| `--padding` | each batch padded to | use it for |
|---|---|---|
| `dataset` (default) | the longest row of the step's whole input | numbers you compare, featurizers you train |
| `batch` | its own longest row | fewer tokens fed, rows in order |
| `sorted` | its own longest row, rows longest first, values put back in row order | the fewest tokens fed, on a large harvest |

- **Padding is never a position**, so results, positions and tables are laid out alike under all three.
- **A large gap between a step's `tokens` and `real` in `clab show` is padding.**
- **A fit's body and evals run at the fit's own `batch_size`**, padded as `--padding` says.
- **`--readout at-reads`** computes the head only at the positions a forward reads of `logits`. Use it on long rows read at a few positions, where the head over every position is most of a forward's memory. It changes nothing where a forward reads `lm_head_output`, every position, or a `grad` of the logits, and it moves the last bits where it acts.

## When two runs give the same numbers

One document at one batch size, padding, readout, device and CPU thread count writes byte-identical value files;
change one and a row's last bits can move. The run records each setting (`run.batch_size`, `run.padding`,
`run.readout`, `run.devices`, `run.threads`), and `clab show` prints all but the threads. `ioi_patch.json` writes
the same value files, `cmp` byte for byte, on nnterp, on hooks, on ndif under `-o remote=local` and under
`--readout at-reads`; the manifest always differs, since it records the engine and the time. What moves bits, how
far, and how to compare two runs are the `clab-debugging` skill's ([reproducing a number](../../clab-debugging/SKILL.md#reproducing-a-number)).

## Where the model runs

- **`--device` is a run option, never a document key.** `"device"` in the `model` block is refused: `model.device: the nnterp engine cannot honor it: where the model runs is a run option: --device`; so are `device_map`, `dispatch` and `torch_dtype` (spelled `dtype`).
- **A device that cannot be is refused before the model loads**: `device 'cuda:0': no CUDA device is visible (torch.cuda.is_available() is False); …`.
- **A device map may spread the model over GPUs**; each batch goes to the first parameter's device.
- **`clab show`'s `served:` line is where and in what dtype the weights actually were**: `served: float32 on cpu`.
- **Run a `bfloat16` model on a GPU**: a CPU without bfloat16 matrix instructions runs it much slower than float32.

### Quantized dtypes

`model.dtype` may name a quantization: `nf4` or `fp4` (bitsandbytes' 4 bits) and `int8`, on the CPU or CUDA with
`bitsandbytes` installed, and `fp8` on a GPU of compute capability 8.9 or later. A quantized model computes in
`bfloat16` (`int8` in `float16`). `ioi_patch.json` at `"dtype": "nf4"` runs on the CPU in seconds, `served:
bfloat16, uint8 on cpu`, its logit differences in steps of 0.5 and its null still equal to the clean run.

### `-o tp=N`

`tp` shards the model over N ranks by transformers' tensor parallelism, on the nnterp engine, under torchrun. Use it
for a checkpoint that does not fit on one GPU when the run needs nothing refused below; a device map (`--device
auto`) is the other way to spread one:

```bash
torchrun --nproc-per-node=3 --no-python clab run documents/x.json --device cpu -o tp=3
```

- **Every rank runs the whole plan**, and rank 0 alone writes and prints; `clab show` reads `tp 3`.
- **Outside torchrun it is refused**: `tp 2 shards the model over 2 processes, one per rank, that torchrun starts: …; this process is not one of them (WORLD_SIZE is unset)`.
- **A checkpoint with no tensor-parallel plan is refused**: on GPT-2, `tp 2: openai-community/gpt2 cannot be sharded 2 ways: its config gives no tensor-parallel plan nnsight can gather, or its sizes divide no degree above 1; …`.
- **The numbers are the unsharded ones to rounding**, not to the bit.
- **Refused under `tp`**: a value the ranks hold in pieces (the attention interior but `attention_premix`, `mlp_activation`), a bitsandbytes `dtype`, `grad`, and `tp` on hooks or ndif.

## Engines

| engine | runs the model | serves | options |
|---|---|---|---|
| `nnterp` (default) | here, as nnterp's standardized model in one nnsight session | every component nnterp serves on the checkpoint, the attention interior under eager | `tp` |
| `hooks` | here, as a plain transformers model with forward hooks, no nnsight at run time | components at a module boundary only | none |
| `ndif` | on an NDIF server; the client holds a model with no weights | what nnterp serves on the server's model | `remote`: `true` or `"local"` |
| `vllm` | in vLLM's worker on the visible GPUs, one request per row, no padding | the stream, the sublayers, the attention interior (its pattern read-only), the head at the last position; no gradient | `tp`, `memory`, `max_len`, `invariant` |

A wrong option is refused before anything loads: `the hooks engine takes no 'tp'; its options: none (where
the model runs is `device`, given as --device)`.

### hooks: the cross-check

- **Only `readout` differs**: positions, edits and ops are clab's own on both engines. On hooks the readout is the final norm, the head and the config's logit softcap; a family that scales its logits after the head differs there.
- **Values inside a module's forward are refused** after the model loads, naming where nnterp reads them: the attention interior, and a mixture's `expert_outputs`, `expert_weights` and `expert_indices`.
- **Compare under one attention implementation**: switching between `sdpa` and `eager` moves the last bits downstream.

### ndif: a server

- **The whole run goes as one request**; the server runs preflight and every step, and the client writes the output directory.
- **The server is nnsight's**: `NDIF_HOST` and `NDIF_API_KEY`, or `nnsight.CONFIG`.
- **`-o remote=local` makes the same trip in this process** without a server; it proves the request is self-contained, and the weights must fit here.
- **The server needs clab at your version**, `datasets` for a hub dataset, and the checkpoint deployed at the document's `key` and `revision`; a request carries at most 512 MiB of loaded values.
- **The server places the model and picks its dtype**: any `--device` but `cpu` is refused (`device: the ndif engine cannot honor 'cuda:0': the server places the model`), and so are quantized dtypes and `tp`.
- **Check `clab show`'s `served:` and `versions:` lines** after a true remote run: they are the server's.
- **A refusal raised on the server arrives as nnsight's `RemoteError`**, and `clab run` prints its last line.

### vllm: vLLM's kernels

vLLM is not run in this skill; these lines are the docs' and `clab describe engines`'s.

- **It needs a GPU and vLLM installed** at the version nnsight instruments; `--device` is `cuda:0` and nothing else.
- **`max_new_tokens` is required on a generate step**, and other options map onto vLLM's sampling; one with no vLLM name fails before any step runs.
- **The head is the last position's** in the prefill, `attention_probabilities` and `attention_scores` are recomputed and read-only, and a fit or a `grad` is refused.
- **Under the default `invariant: true` a run repeats its bits** at any batch size; compare vllm with vllm, and with nnterp on outcomes.
- **`memory` (0.9) is the GPU fraction vLLM reserves**; the run's own values live outside it on `cuda:0`, so lower it when they run out of memory there.
- **`max_len` caps prompt plus new tokens** (vLLM's `max_model_len`, by default the checkpoint's); lower it when vLLM cannot fit the full context's cache in its `memory`.
- **`invariant: false` runs vLLM's fastest kernels**, whose last bits move between runs: only for throughput where outcomes, not bits, are compared.

## Memory

- **Every value a step binds stays on the model's device until the run ends**; save and name fewer values, or score inside an entry ([ops](ops.md#a-step-or-an-entry)).
- **A step that runs out of CUDA memory stops the run**, naming the step, the batch size and every value the run holds.
- **The finished steps are written** as if the run had ended there, with the refusal as `run.stopped`; `clab show` prints a `stopped:` line, and `clab table` refuses a save the run never reached.
- **A fit's refusal names the fit's own `batch_size`.**

## Where output goes

- **`--out` names the directory**; without it, `out/<document name>` under the working directory.
- **A second run into one directory replaces the files it writes and leaves the others**; give each run its own `--out`.
- **Data paths, bundle files and `load` directories resolve against the corpus root**: the nearest directory above the document that holds `data/`; with none, the directory above the document's own.
- **So `project/documents/x.json` resolves against `project/`** with no `data/`, and a document at the top of a folder needs an empty `data/` beside it. A document of inline rows that loads nothing needs neither.
- **Run from the corpus root**, so the default `out/` is where a later `load` or `file_path` looks.
- **`clab table` reads a file dataset from `data/` under the working directory**, so run it from the corpus root ([output](output.md#tabling-needs-the-rows)).
