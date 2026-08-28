# Error Catalogue

Every exception nnsight 0.8 raises, with the message text as it actually appears
(reproduced against nnsight 0.8 + gpt2), what caused it, and the fix.

## Execution order

### `OutOfOrderError`

```
'model.transformer.h.2.output.i0' was requested but the model already ran past it
```

One class covers two situations:

- **Asked out of order** — you read a later module, then an earlier one.
- **Never delivered** — a worker is still waiting when the run ends: the module
  never executed (`tracer.stop()` before it, a `.skip()`ped module's children,
  generation ended early), or you asked for `.grad` on a slice rather than the
  tensor you captured.

Fix: reorder into forward order; use separate `tracer.invoke(...)` blocks for
independent orders; use `tracer.cache()` when order is unknown; for gradients, read
in reverse-forward order and only on the exact captured tensor.

The `.iN` suffix is the occurrence index — `.i3` means the 4th time that location
was reached (i.e. generation step 3).

### `UserWarning` — loop outran the model (not an exception)

```
'model.output.i2' was never reached: the model ran fewer iterations than the loop
requested. Values from reached iterations are kept.
```

`tracer.iter[:N]` asked for more steps than the model produced (EOS, a stop string,
or an unbounded `tracer.all()` over-running by one). Values from the steps that did
happen are kept. Bound the loop to the real step count if you also need trailing
code to run.

## Context and setup

| Exception | Message | Cause → fix |
|---|---|---|
| `ValueError` | ``Cannot access `model.transformer.h.0.output` outside of interleaving`` | `.output`/`.input` read with no trace running → open a trace, or `model.scan(...)` for shapes |
| `ValueError` | ``save() was called outside a trace. `.save()` / nnsight.save(x) marks a value to return from the enclosing `with model.trace(...):` block, so it only works inside one — move the save into the trace block.`` | a save before or after the `with` → move it inside |
| `ValueError` | ``trace() needs an input, or at least one `with tracer.invoke(...)` block`` | `with model.trace():` with an empty body → pass an input or add an invoke |
| `ValueError` | ``Cannot invoke while the model is already running.`` | nested `tracer.invoke`, or an invoke opened inside an `iter` loop → open all invokes at the top level of the trace |
| `ValueError` | ``The body of a traced `with` must start on its own line; nnsight runs the body itself, and can only intercept it at the start of a line.`` | `with model.trace(x): out = ...` → put the body on the next line |
| `WithBlockNotFoundError` | *(no message)* | the block's source text isn't retrievable — `exec(compile(str, ...))`, some bare REPLs, generated code → run from a file, IPython, or a notebook |
| `NotImplementedError` | ``NNsight does not support batching multiple invokes`` | base `NNsight` with 2+ *input* invokes → use one invoke, or implement `_batch_size`/`_batch`; empty invokes always work |

## Coordination

| Exception | Message | Cause → fix |
|---|---|---|
| `ValueError` | ``A barrier was never reached by every block it waits for; check the count it was created with`` | `tracer.barrier(n)` with `n` larger than the number of blocks that call it → pass the true count |
| `ValueError` | ``A batched `.skip()` has to cover every row: skip the module in every invoke, or none — a shared forward can't run for only the rows an invoke left unskipped.`` | `.skip()` in some invokes but not others → skip in all of them or none |
| `NameError` | ``name 'src' is not defined`` | a value from another invoke read before its producer ran → `tracer.barrier(n)`, or park the reader past the producing module first |

## Values and shapes

| Exception | Message | Cause → fix |
|---|---|---|
| `UnboundLocalError` | ``cannot access local variable 'h' where it is not associated with a value`` | forgot `.save()` → save it |
| `TypeError` | ``'tuple' object does not support item assignment`` | `attn.output[0] = x` → `attn.output[0][:] = x`, or rebuild the tuple and assign to `.output` |
| `AttributeError` | ``'TransformersModel' object (nor its module) has attribute 'model'`` | wrong module path for this architecture → `scripts/inspect_model.py <repo_id>` |
| `RuntimeError` | ``The expanded size of the tensor (10) must match the existing size (3) at non-singleton dimension 1.`` | a replacement tensor whose shape doesn't match the activation → build it from the activation (`torch.zeros_like(x)`, `x.shape`, `x.device`, `x.dtype`) |
| `GuardOnDataDependentSymNode` | ``Could not guard on data-dependent expression ...`` | branching on **values** inside `model.scan(...)` — fake tensors have no data → branch on shapes only, or use a real trace |

## vLLM engine refusals

Raised at the client as a `RuntimeError` carrying the message, once the request ends
(or at construction for the last two):

| Message | Cause → fix |
|---|---|
| ``'<location>' is not a tap on this engine, so a replayed CUDA graph never reaches it`` | reading a module location on a `taps=` engine that was not declared → add it to `taps`, or build the engine without `taps` (eager, every location served) |
| ``... prompt was split across steps by chunked prefill`` | `enable_chunked_prefill=True` was passed and this prompt exceeded the step's token budget → drop the flag (off by default), or raise `max_num_batched_tokens` |
| ``enforce_eager=... contradicts taps=...`` | `ValueError` at construction: the engine mode follows from `taps` → drop `enforce_eager` |

## Remote

`RemoteError` on a failed submission or a server-side `ERROR` status. A worker-side
error from a driver such as vLLM is re-raised at the client as a `RuntimeError`
carrying the original type, message, and traceback.

Common remote-only causes: a helper function from your own file that wasn't
registered (`nnsight.register(...)`), an import that isn't on NDIF's whitelist, a
model that is COLD rather than RUNNING, a missing API key, or a version skew
between your machine and the server. `python scripts/check_env.py --remote` (in the
`nnsight` skill) checks all of those at once. See the `nnsight-remote` skill.

## Not errors, but they end the run

| Exception | Meaning |
|---|---|
| `EarlyStopException` | `tracer.stop()` — a clean early exit, swallowed by the interleaver. Don't catch it; do `.save()` before calling `stop()`, since nothing after it in that block runs. |
| `SourceNotAvailable` | `.source` can't instrument that callable: it calls a submodule (use that submodule's own `.source`), it's a builtin/C function, it's an assignment op (no callee), or you used recursive `.source` outside a trace. Decorated forwards are fine. |

## Reading a traceback

By default nnsight strips its own frames so the traceback points at your code; the
exception type is unchanged, so `except ValueError:` still works. To see nnsight's
internals — when you suspect the bug is in the library rather than your
intervention:

```python
import nnsight
nnsight.CONFIG.APP.DEBUG = True
```

or `NNSIGHT_DEBUG=1 python script.py`, or pass `-v` on the command line. Debug mode
also makes remote runs log payload/result sizes and every status update.
