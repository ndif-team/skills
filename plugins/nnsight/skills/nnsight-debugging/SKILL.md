---
name: nnsight-debugging
description: Diagnose and fix nnsight code that errors, hangs, returns nothing, or silently does the wrong thing — OutOfOrderError, save() outside a trace, "Cannot access outside of interleaving", WithBlockNotFoundError, empty saved lists, interventions with no effect, shape mismatches, batching and barrier errors. Also use when porting nnsight code written for older versions (0.4/0.5/0.6 idioms like .value, nnsight.list(), tracer.next(), LanguageModel, proxies), which is most of the nnsight code on the internet and fails in specific, recognizable ways on 0.8.
---

# Debugging nnsight

Most nnsight failures are one of a dozen known shapes. Match the symptom, apply
the fix. If the code came from a tutorial, a paper repo, or a model's card, check
[references/porting-pre-0.8.md](references/porting-pre-0.8.md) **first** — old
idioms are the single most common cause.

## Triage

| Symptom | Cause | Fix |
|---|---|---|
| `OutOfOrderError` | modules touched out of forward order | reorder, or split across `tracer.invoke`s |
| `UnboundLocalError` after the block | forgot `.save()` | save it, and bind it to a name |
| Saved list is empty / missing values | saved the elements, not the container | `nnsight.save([])` then append raw values |
| `ValueError: save() was called outside a trace` | save before/after the `with` | move it inside |
| `ValueError: Cannot access ... outside of interleaving` | `.output` read outside a trace | open a trace, or use `model.scan` for shapes |
| `WithBlockNotFoundError` (no message) | the block's source isn't on disk | run from a file/IPython, not `exec` of a string |
| `AttributeError: ... has attribute 'model'` | wrong module path for this architecture | `scripts/inspect_model.py` in the `nnsight` skill |
| `TypeError: 'tuple' object does not support item assignment` | assigning into a tuple output | `out[0][:] = x`, or rebuild and assign the tuple |
| Intervention has no effect | wrote to a copy, or indexed a tensor as if it were a tuple | see [Silent failures](#silent-failures) |
| Everything after a loop is missing | unbounded `tracer.iter[:]` / `tracer.all()` | bound it: `tracer.iter[:N]` |
| `NameError` on a value from another invoke | read before the producer ran | `tracer.barrier(n)` |
| `NotImplementedError: ... batching multiple invokes` | base `NNsight` with 2+ input invokes | one invoke, or implement `_batch_size`/`_batch` |
| Remote job errors or returns nothing | serialization / save rules differ remotely | the `nnsight-remote` skill |

Full catalogue with exact messages: [references/error-catalogue.md](references/error-catalogue.md).

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
```

## OutOfOrderError

```
OutOfOrderError: 'model.transformer.h.2.output.i0' was requested but the model
already ran past it
```

Your trace body is a worker that parks until the model produces each value. Ask
for something the model has already passed and nothing will ever wake you, so
nnsight raises at the end of the run.

<!-- test: expect-error OutOfOrderError -->
```python
with model.trace(prompt):
    late = model.transformer.h[8].output
    early = model.transformer.h[2].output      # already gone
```

Fixes, in order:

1. **Reorder** into forward order. Remember submodules come first: `h[0].ln_1` →
   `h[0].attn` → `h[0].mlp` → `h[0].output` → `h[1]...`.
2. **Split into invokes** — separate workers, independent order:

```python
with model.trace() as tracer:
    with tracer.invoke(prompt):
        late = model.transformer.h[8].output.save()
    with tracer.invoke(prompt):
        early = model.transformer.h[2].output.save()

print(late.shape, early.shape)
```

3. **Cache** if you want many modules regardless of order:
   `cache = tracer.cache()`.

The same error appears when you request a module the run never reached — after
`tracer.stop()`, on a `.skip()`ped module's children, or past the point where
generation ended.

## Nothing came back

Three distinct causes; check them in this order.

**Forgot `.save()`** — the block runs in another frame, so the name is undefined:

<!-- test: expect-error UnboundLocalError -->
```python
def get_hidden():
    with model.trace(prompt):
        hidden = model.transformer.h[0].output     # not saved
    return hidden

get_hidden()
```

**Saved the elements instead of the container** — this one is silent locally and
empty remotely:

```python
with model.trace(prompt):
    right = nnsight.save([])                       # save the container
    for block in model.transformer.h:
        right.append(block.output[0, -1])          # append raw values

print(len(right))                                  # 12
```

**Marked a value with no name to return it under.** `.save()` returns the value by
its *variable name*; a bare `model.output.logits.save()` on its own line marks
something with no local to carry it back, and it silently never appears.

## "Cannot access outside of interleaving"

```
ValueError: Cannot access `model.transformer.h.0.output` outside of interleaving
```

There is no value to read when the model isn't running.

<!-- test: expect-error ValueError -->
```python
hidden = model.transformer.h[0].output
```

If you wanted shapes rather than values, use `model.scan(prompt)` — no weights, no
compute. If you wanted values, open a trace.

## WithBlockNotFoundError

Raised with **no message**. nnsight reads the `with` block's source text to compile
it; if the source isn't retrievable, there is nothing to run. Triggers:
`exec(compile(source_string, ...))`, some bare REPLs, and dynamically generated
code. Run from a real file, IPython, or a notebook.

The related one:

```
ValueError: The body of a traced `with` must start on its own line; nnsight runs
the body itself, and can only intercept it at the start of a line.
```

Never write `with model.trace(x): out = ...` on one line.

## Silent failures

No exception, wrong science. These are the dangerous ones.

**Indexing a tensor as if it were a tuple.** Pre-0.8 examples write
`model.transformer.h[i].output[0]` because blocks used to return tuples. On 0.8 a
GPT-2 block returns a plain tensor, so `[0]` selects **batch row 0** — your code
runs, shapes look plausible, and you are analyzing the wrong slice.

```python
with model.trace(prompt):
    whole = model.transformer.h[5].output.save()          # [batch, seq, hidden]
    row0 = model.transformer.h[5].output[0].save()        # [seq, hidden] — a row!

print(whole.shape, row0.shape)
```

Check with `isinstance(x, tuple)` or `scripts/inspect_model.py --prompt ...`.

**An unbounded iteration loop swallowing trailing code.** `tracer.all()` and
`tracer.iter[:]` unwind at the final over-run step, dropping every later line in
that invoke:

```python
with model.generate(prompt, max_new_tokens=3) as tracer:
    seen = nnsight.save([])
    for step in tracer.all():
        seen.append(model.output.logits[0, -1].argmax(dim=-1))
    after_the_loop = nnsight.save("this line never runs")

print(len(seen))                             # 3 — the loop's own values survive
print("after_the_loop" in globals())         # False — the trailing line was dropped
```

Bound the loop (`tracer.iter[:3]`) and the trailing code runs.

**An intervention that never took effect.** Assigning to a local doesn't change
the model — you must write through the property:

```python
with model.trace(prompt):
    acts = model.transformer.h[5].output
    acts = acts * 0                        # rebinds a local; model unaffected
    unchanged = model.output.logits[0, -1].argmax().save()

with model.trace(prompt):
    model.transformer.h[5].output[:] = 0   # writes through the property
    changed = model.output.logits[0, -1].argmax().save()

print(model.tokenizer.decode(unchanged), "|", model.tokenizer.decode(changed))
```

**A metric that doesn't measure what you think.** Before trusting a result, check
that the *unmodified* run reproduces the expected baseline in the same trace, and
that a deliberately absurd intervention (zeroing everything) moves the metric.

## Diagnostics

**Print inside the trace.** The body runs real Python against real tensors:

```python
with model.trace(prompt):
    resid = model.transformer.h[5].output
    print("layer 5:", resid.shape, resid.dtype, resid.device, resid.detach().norm().item())
    logits = model.output.logits.save()
```

Remotely, `print` output comes back as log lines — far cheaper than saving a
tensor just to look at it.

**Check shapes without running:** `with model.scan(prompt):` gives fake tensors
with real shapes. Do not branch on their *values* — that raises
`GuardOnDataDependentSymNode`.

**See inside a forward:** `print(model.transformer.h[0].mlp.source)` lists every
hookable operation, outside a trace.

**Full tracebacks:** nnsight strips its own frames by default so errors point at
your code. To see everything:

```python
nnsight.CONFIG.APP.DEBUG = True
```

or `NNSIGHT_DEBUG=1 python script.py`. Turn it on when you suspect the bug is in
nnsight itself, not your intervention.

**Environment:** `python scripts/check_env.py --remote` (in the `nnsight` skill)
prints versions, GPU memory, NDIF key/host, deployed models, and the local-vs-NDIF
package diff — a version skew explains a lot of remote weirdness.

## Porting old code

If the code you are fixing uses `.value`, `nnsight.list()`, `tracer.next()`,
`with tracer.all():`, `LanguageModel`, `model.generator.output`, or talks about
"proxies", it targets a pre-0.8 nnsight. Do not patch it line by line — translate
it with [references/porting-pre-0.8.md](references/porting-pre-0.8.md), which maps
every removed idiom to its 0.8 form.

## Related skills

- `nnsight` — the API itself and how tracing works
- `nnsight-remote` — remote-specific failures and performance
