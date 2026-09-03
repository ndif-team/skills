# Error Catalogue

Every exception nnsight 0.8 raises, with the message text as it actually appears
(reproduced against nnsight 0.8 + gpt2), what caused it, and the fix.

## Execution order

### `OutOfOrderError`

One class, three messages. Which one you get depends on where the worker was
standing when the run ended, not on what the mistake was.

**Asked out of order, or never delivered.** The worker is not inside an
iteration loop:

```
'model.transformer.h.2.output.i0' was requested but the model already ran past it
```

You read a later module and then an earlier one, or the module never executed at
all — `tracer.stop()` before it, a `.skip()`ped module's children, an MoE expert
that was not routed to, a module whose `forward` another library replaced.

Fix: reorder into forward order; put independent orders in separate
`tracer.invoke(...)` blocks; use `tracer.cache()` when the order is unknown; for
gradients, read in reverse-forward order and only on the exact captured tensor.

**A loop with an end the run did not reach:**

```
'model.transformer.h.11.output.i3' was never reached: the loop asked for iteration 3
of 'model.transformer.h.11.output' and the run reached it 3 times, so the loop was
cut short and nothing after it ran. Bound the loop to the iterations the run makes
(`min_new_tokens=` holds a generation to a step count), or loop with `tracer.all()`
and put what follows the loop after the `with` block.
```

`tracer.iter[:8]`, `tracer.iter[2]`, and `tracer.iter[[0, 2, 7]]` all name an end,
and a run that makes fewer steps raises. The unwind takes the worker out at the
loop, so nothing after the loop runs — which is why this is an error and not a
note.

Fix: `min_new_tokens=N` alongside `max_new_tokens=N`, or `tracer.all()` with the
trailing statements moved past the `with` block.

**An open loop.** `tracer.iter[:]`, `tracer.iter[2:]`, and `tracer.all()` have no
end of their own, so outrunning the model is how they finish. That one is a
`UserWarning`, not an exception:

```
'model.transformer.h.11.output.i3' was never reached: an open `tracer.iter[:]` /
`tracer.all()` loop ends by asking for a step the run does not make. Values saved
inside the loop are kept; the statements after it did not run.
```

**The `.iN` suffix is the tell.** It is the occurrence index — `.i3` is the fourth
visit to that location, i.e. generation step 3. An occurrence *past anything the
loop selected* means the loop body reads locations out of order: each pass pushes
the next request one occurrence later, and the strand shows up at the end of the
run rather than at the line that caused it. Reorder the body.

## Context and setup

| Exception | Message | Cause → fix |
|---|---|---|
| `ValueError` | ``Cannot access `model.transformer.h.0.output` outside of interleaving`` | `.output`/`.input` read with no trace running → open a trace, or `model.scan(...)` for shapes |
| `ValueError` | ``save() was called outside a trace. `.save()` / nnsight.save(x) marks a value to return from the enclosing `with model.trace(...):` block, so it only works inside one — move the save into the trace block.`` | a save before or after the `with` → move it inside |
| `ValueError` | ``trace() needs an input, or at least one `with tracer.invoke(...)` block`` | `with model.trace():` with an empty body → pass an input or add an invoke. A body that also reads an envoy raises "Cannot access …" first, because the body runs before this check |
| `ValueError` | ``Cannot invoke while the model is already running.`` | nested `tracer.invoke`, or an invoke opened inside an `iter` loop → open all invokes at the top level of the trace |
| `AttributeError` | ``'NoneType' object has no attribute 'event'`` | a `with model.trace(...)` nested inside another trace → use `model.session()` for several traces, or sibling `tracer.invoke(...)` blocks for several inputs |
| `ValueError` | ``The body of a traced `with` must start on its own line; nnsight runs the body itself, and can only intercept it at the start of a line.`` | `with model.trace(x): out = ...` → put the body on the next line |
| `ValueError` | ``A traced `with` block cannot start with `try:`; nnsight intercepts the body at its first line, and a `try` there is the one statement Python gives it no way back out of. Put any statement above the `try`, or move the `try` outside the block.`` | the body's **first** statement is a `try` (with `except`, `finally`, or both) → put any statement above it, or wrap the whole `with` in the `try`. A `try` anywhere else in the body is ordinary code |
| `SyntaxError` | ``'return' outside function`` | a `return` inside the block — the body is compiled on its own, outside any function → save the value and `return` it after the `with` |
| `WithBlockNotFoundError` | *(no message)* | the block's source text isn't retrievable — a script piped to stdin (`python < s.py`), `exec(compile(str, ...))`, generated code → run the file by name. `python -c "..."` works |
| `NotImplementedError` | ``NNsight does not support batching multiple invokes`` | base `NNsight` with 2+ *input* invokes → use one invoke, or implement `_batch_size`/`_batch`; empty invokes always work |
| `AttributeError` | ``module 'nnsight' has no attribute 'list'`` (also `dict`, `apply`, `cond`, `iter`, `log`, `local`, `session`) | pre-0.8 code → [porting-pre-0.8.md](porting-pre-0.8.md) |

## Coordination

| Exception | Message | Cause → fix |
|---|---|---|
| `ValueError` | ``A barrier was never reached by every block it waits for; check the count it was created with`` | `tracer.barrier(n)` with `n` **larger** than the number of blocks that call it → pass the true count. This is the safe way to get the count wrong: it is loud, and it fails at run end without having released anything. A count that is too *small* releases early and says nothing — see the `NameError` row. **Call the barrier, don't wait on it**: `b = tracer.barrier(2)` then `b()`. There is no `b.wait()` |
| `ValueError` | ``A batched `.skip()` has to cover every row: skip the module in every invoke, or none — a shared forward can't run for only the rows an invoke left unskipped.`` | `.skip()` in some invokes but not others → skip in all of them or none |
| `ValueError` | ``A batched write has to keep its rows: this block owns rows 0:1 of 2, so the replacement must be (1, 7, 768), not (2, 7, 768).`` | a whole-tensor write inside one invoke of a batch, with a different leading dim → build the replacement from the activation you were served. A lone invoke *is* the batch and may reshape freely |
| `NameError` | ``name 'src' is not defined`` | **(a)** a value from another invoke read before its producer ran → `tracer.barrier(n)`, or park the reader past the producing module first. **(b)** a `tracer.barrier(n)` whose `n` is *smaller* than the number of blocks holding it: it releases as soon as `n` of them arrive, so the waiting blocks resume before the producer has bound its value. Nothing mentions the barrier — the name in the message is the tell. Count the blocks that call it. See `docs/usage/barrier.md` |

## Values and shapes

| Exception | Message | Cause → fix |
|---|---|---|
| `UnboundLocalError` | ``cannot access local variable 'h' where it is not associated with a value`` | forgot `.save()`, or the block unwound at a loop before reaching that line → save it; check for an over-running `iter` loop above it |
| `TypeError` | ``'tuple' object does not support item assignment`` | `attn.output[0] = x` → `attn.output[0][:] = x`, or rebuild the tuple and assign to `.output` |
| `AttributeError` | ``'TransformersModel' object (nor its module) has attribute 'model'`` | wrong module path for this architecture → `scripts/inspect_model.py <repo_id>`. The same message appears when an `eproperty`'s preprocess raises `AttributeError`, since a failing property getter falls through to `__getattr__` |
| `RuntimeError` | ``The expanded size of the tensor (10) must match the existing size (3) at non-singleton dimension 1.`` | a replacement tensor whose shape doesn't match the activation → build it from the activation (`torch.zeros_like(x)`, `x.shape`, `x.device`, `x.dtype`) |
| `NotImplementedError` | ``This tensor does not require grad, so a backward session cannot produce gradients: nothing the block reads can ever receive one. The forward ran without gradient tracking (e.g. under torch.no_grad()), or the tensor was created without requires_grad=True.`` | a `with metric.backward():` inside a `torch.no_grad()` region → drop the `no_grad` around the trace |
| `GuardOnDataDependentSymNode` | ``Could not guard on data-dependent expression ...`` | branching on **values** inside `model.scan(...)` — fake tensors have no data → branch on shapes only, or use a real trace |

## vLLM refusals

`scan` is refused up front:

```
NotImplementedError: scan is unavailable on vLLM: it runs the model's forward under
a fake-tensor mode to propagate shapes, and there is no forward here to run — the
engine's worker runs the real one, under torch.inference_mode. Trace a prompt and
read the shapes off the activations it serves.
```

Everything else raised by an intervention arrives at the client as a
`RuntimeError` whose message *begins* with the original type name and message,
once the request ends (or at construction for the last two). The original class is
not reconstructed across the process boundary — **match on the message, not the
class**:

<!-- test: skip nocompile -->
```python
except RuntimeError as error:
    if "A batched write has to keep its rows" in str(error):
        ...
```

| Message | Cause → fix |
|---|---|
| ``ValueError: A batched write has to keep its rows: …`` | a write that changes the request's row count → keep the leading dim. The refusal ends that request alone; the engine and its co-tenants survive |
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

## Warnings

| Category | When |
|---|---|
| `UserWarning: '<location>.iN' was never reached: an open …` | an open `iter[:]` / `all()` loop finished by outrunning the model. Expected; the statements after the loop did not run |
| `NNsightDeprecationWarning` | a deprecated name. A `FutureWarning`, so it shows wherever the call is — script, module, or notebook. Every message names its replacement; the full list is in [porting-pre-0.8.md](porting-pre-0.8.md) |

## Not errors, but they end the run

| Exception | Meaning |
|---|---|
| `EarlyStopException` | `tracer.stop()` — a clean early exit, swallowed by the interleaver. Don't catch it; do `.save()` before calling `stop()`, since nothing after it in that block runs. |
| `SourceNotAvailable` | `.source` can't instrument that callable: it calls a submodule (use that submodule's own `.source`), it's a builtin/C function, it's an assignment op (no callee), or you used recursive `.source` outside a trace. Decorated forwards are fine. |

## Reading a traceback

By default nnsight strips **its own** frames so the traceback points at your code;
the exception type is unchanged, so `except ValueError:` still works. Frames from
torch, transformers, and your own helpers all stay — so a failure caused by a
badly shaped write arrives with the model's stack intact, ending in torch, and the
line that made the write is not in it. The `with` line at the top is what names
your block.

To see nnsight's internals — when you suspect the bug is in the library rather
than your intervention:

```python
import nnsight
nnsight.CONFIG.APP.DEBUG = True
```

or `NNSIGHT_DEBUG=1 python script.py`, or pass `-v` on the command line. On GPT-2
that takes an in-block error from 4 frames to 14. Debug mode also makes remote runs
log payload/result sizes and every status update.
