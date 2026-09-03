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
| `OutOfOrderError: … already ran past it` | modules touched out of forward order | reorder, or split across `tracer.invoke`s |
| `OutOfOrderError: … the loop asked for iteration N` | a bounded `tracer.iter[:N]` the run never reached | `min_new_tokens=N`, or `tracer.all()` with the trailing code past the `with` |
| `UnboundLocalError` after the block | forgot `.save()`, or a loop above unwound the rest of the block | save it and bind it; check the loop's bound |
| `ValueError: save() was called outside a trace` | save before/after the `with` | move it inside |
| `ValueError: Cannot access ... outside of interleaving` | `.output` read outside a trace | open a trace, or use `model.scan` for shapes |
| ``ValueError: … cannot start with `try:` `` | the body's first statement is a `try` | put any statement above it, or wrap the whole `with` |
| `SyntaxError: 'return' outside function` | a `return` inside the block | save it, return after the `with` |
| `WithBlockNotFoundError` (no message) | the block's source isn't on disk | run from a file/IPython, not stdin or `exec` of a string |
| `AttributeError: 'NoneType' object has no attribute 'event'` | `model.trace(...)` nested inside another trace | `model.session()`, or sibling invokes |
| `AttributeError: ... has attribute 'model'` | wrong module path for this architecture | `scripts/inspect_model.py` in the `nnsight` skill |
| `TypeError: 'tuple' object does not support item assignment` | assigning into a tuple output | `out[0][:] = x`, or rebuild and assign the tuple |
| `ValueError: A batched write has to keep its rows` | a whole-tensor write changed an invoke's row count | build the replacement from the activation you were served |
| Intervention has no effect | wrote to a copy, or indexed a tensor as if it were a tuple | see [Silent failures](#silent-failures) |
| Everything after a loop is missing | an open `tracer.iter[:]` / `tracer.all()` | bound it and pin the run, or move the trailing code past the `with` |
| Saved list is empty **when run remotely** | saved the elements, not the container | `nnsight.save([])` then append raw values |
| `NameError` on a value from another invoke | read before the producer ran | `tracer.barrier(n)`, then call it: `b()` |
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

assert late.shape == early.shape == (1, 10, 768)   # each invoke sees its own rows
```

3. **Cache** if you want many modules regardless of order:
   `cache = tracer.cache()`.

The same error appears when you request a module the run never reached — after
`tracer.stop()`, on a `.skip()`ped module's children, or past the point where
generation ended.

## A loop that outran the run

A loop with an end you named is a claim about the run, and a run that makes fewer
steps raises rather than warning — the worker is unwound *at the loop*, so every
statement after it is discarded, and a warning would leave those names holding
stale values:

<!-- test: expect-error OutOfOrderError -->
```python
with model.generate(prompt, max_new_tokens=3) as tracer:
    seen = nnsight.save([])
    for step in tracer.iter[:20]:              # the run makes 3 steps, not 20
        seen.append(model.output.logits[0, -1].argmax(dim=-1))
    ids = tracer.result.save()
```

Pin the run to the count the loop asks for:

```python
with model.generate(prompt, max_new_tokens=3, min_new_tokens=3) as tracer:
    seen = nnsight.save([])
    for step in tracer.iter[:3]:
        seen.append(model.output.logits[0, -1].argmax(dim=-1))
    ids = tracer.result.save()

assert len(seen) == 3
assert ids.shape[-1] == len(model.tokenizer.encode(prompt)) + 3
```

If the step count isn't knowable in advance, loop with `tracer.all()` and put
whatever follows the loop after the `with` block — an open loop warns instead of
raising, and the values saved inside it survive.

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

**Saved the elements instead of the container** — this one collects values locally
and comes back empty remotely:

```python
with model.trace(prompt):
    right = nnsight.save([])                       # save the container
    for block in model.transformer.h:
        right.append(block.output[0, -1])          # append raw values

assert len(right) == 12
```

**Marked a value with no name to return it under.** `save` records the object's
identity and the block returns every local bound to a marked object, so a bare
`model.output.logits.save()` on its own line marks something no local carries
back, and it never appears. Bind it.

The same identity rule means a saved value Python interns — a small int, `True`,
`None` — also brings back any unrelated local holding the same object. Harmless,
but it explains a name that is defined after the block when you never saved it.

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

## The block never ran at all

Three ways a `with` block is refused before anything executes.

**`WithBlockNotFoundError`**, raised with no message. nnsight reads the `with`
block's source text to compile it; if the source isn't retrievable, there is
nothing to run. Triggers: a script piped to the interpreter
(`python < s.py`, `cat s.py | python`), `exec(compile(source_string, ...))`, and
generated code. Run from a real file, IPython, or a notebook — `python -c "..."`
is fine.

**The body has to start on its own line**:

```
ValueError: The body of a traced `with` must start on its own line; nnsight runs
the body itself, and can only intercept it at the start of a line.
```

Never write `with model.trace(x): out = ...` on one line.

**The body cannot start with `try:`**:

```
ValueError: A traced `with` block cannot start with `try:`; nnsight intercepts the
body at its first line, and a `try` there is the one statement Python gives it no
way back out of. Put any statement above the `try`, or move the `try` outside the
block.
```

nnsight stops the interpreter running the body inline by raising at the body's
first line and catching that raise in the `with`; CPython gives the `try`
keyword's line no exception-table entry to route it through. Wrapping the whole
`with` in a `try` works, and so does any statement above it — only the first
statement is the cue:

```python
with model.trace(prompt):
    hidden = model.transformer.h[0].output      # any statement above the try
    try:
        head = hidden[0, -1]
    except IndexError:
        head = None
    kept = nnsight.save(head)

assert kept.shape == (768,)
```

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

assert whole.shape == (1, 10, 768)
assert row0.shape == (10, 768)
```

Check with `isinstance(x, tuple)` or `scripts/inspect_model.py --prompt ...`.

**An open iteration loop swallowing trailing code.** `tracer.all()` and
`tracer.iter[:]` unwind at the final over-run step, dropping every later line in
that invoke. Only a warning marks it:

```python
with model.generate(prompt, max_new_tokens=3) as tracer:
    seen = nnsight.save([])
    for step in tracer.all():
        seen.append(model.output.logits[0, -1].argmax(dim=-1))
    after_the_loop = nnsight.save("this line never runs")

assert len(seen) == 3                        # the loop's own values survive
assert "after_the_loop" not in globals()     # the trailing line was dropped
```

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

assert unchanged.item() != changed.item()
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

assert logits.shape == (1, 10, 50257)
```

Remotely, `print` output comes back as log lines — far cheaper than saving a
tensor just to look at it.

**Check shapes without running:** `with model.scan(prompt):` gives fake tensors
with real shapes. Do not branch on their *values* — that raises
`GuardOnDataDependentSymNode`. `scan` is unavailable on vLLM, which refuses it
with a `NotImplementedError` explaining why.

**See inside a forward:** `print(model.transformer.h[0].mlp.source)` lists every
operation you can reach, outside a trace.

**Full tracebacks:** nnsight strips its own frames by default so errors point at
your code. Frames from torch and transformers stay, so a failure caused by a
badly shaped write arrives with the model's stack under it and does *not* name the
line that wrote the value — read the `with` line at the top and check your writes.
To see everything:

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

Run it with warnings visible while you port: every name that still works under an
old spelling raises `nnsight.NNsightDeprecationWarning` naming its replacement,
and being a `FutureWarning` it shows from a package or helper module, not just
from `__main__`.

## Related skills

- `nnsight` — the API itself and how tracing works
- `nnsight-remote` — remote-specific failures and performance
