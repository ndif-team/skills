---
name: debugging
description: Diagnose and fix nnsight code that errors, hangs, returns nothing, or silently does the wrong thing — OutOfOrderError, NameError or UnboundLocalError after the trace, save() outside a trace, "Cannot access outside of interleaving", WithBlockNotFoundError, a nested trace inside another trace, empty saved lists, interventions with no effect, shape mismatches, batching and barrier errors. Also use when porting nnsight code written for older versions (0.4/0.5/0.6 idioms like .value, nnsight.list(), tracer.next(), LanguageModel, proxies), which is most of the nnsight code on the internet and fails in specific, recognizable ways on 0.8.
---

# Debugging nnsight

Most nnsight failures are one of a dozen known shapes. Match the symptom, apply
the fix. If the code came from a tutorial, a paper repo, or a model's card, check
[references/porting-pre-0.8.md](references/porting-pre-0.8.md) **first** — old
idioms are the single most common cause. Do not patch old code line by line;
translate it with that table, and run with warnings visible: every name that still
works under an old spelling raises `nnsight.NNsightDeprecationWarning` naming its
replacement.

## Triage

| Symptom | Cause | Fix |
|---|---|---|
| `OutOfOrderError: … already ran past it` | modules touched out of forward order | reorder, or split across `tracer.invoke`s |
| `UserWarning: … was never reached`; a short result; everything after a loop missing | a `tracer.iter` loop — bounded or open — asked for a step the run did not make and was cut short | [A loop that outran the run](#a-loop-that-outran-the-run) |
| `NameError` (module level) or `UnboundLocalError` (in a function) on a name after the block | forgot `.save()` — or a cut-short loop above the line dropped it | save it and bind it; see [Nothing came back](#nothing-came-back) |
| **No error**, but a name after the block holds an old value | forgot `.save()` on a name already bound (an earlier cell, a loop, a rerun) — it silently keeps the old object | `.save()` inside the block; see [Nothing came back](#nothing-came-back) |
| `NameError` **inside** the trace body, outside any invoke, on a name an invoke binds | trace-body code between invokes runs *before* the invoke workers do | reduce inside the invoke (before anything is transferred), or combine after the block on saved values. To collect across invokes, create `store = nnsight.save({})` in the body — the container must itself be saved or it never comes back — and write `store[k] = ...` inside each invoke (`nnsight` skill → references/batching.md, "Sharing values across invokes") |
| `NameError` inside an invoke on a value another invoke produces | read before the producer ran, or a `tracer.barrier(n)` whose `n` is too small released it early | `tracer.barrier(n)` with the true block count, then call it: `b()` |
| `ValueError: save() was called outside a trace` | save before/after the `with` | move it inside |
| `ValueError: Cannot access ... outside of interleaving` | `.output` read outside a trace | open a trace, or use `model.scan` for shapes |
| ``ValueError: … cannot start with `try:` `` / ``… must start on its own line`` | the body's first statement is a `try`, or the body is on the `with` line | see [The block never ran](#the-block-never-ran-at-all) |
| `SyntaxError: 'return' outside function` | a `return` inside the block | save it, return after the `with` |
| `WithBlockNotFoundError` (no message) | the block's source isn't on disk | run from a file/IPython, not stdin or `exec` of a string |
| `AttributeError: 'NoneType' object has no attribute 'event'` | `model.trace(...)` nested inside another trace | `model.session()`, or sibling invokes |
| `AttributeError: ... has attribute 'model'` (or `'<N>'` for an integer N) | wrong module path for this architecture, or an index past the last layer | `scripts/inspect_model.py` in the `nnsight` skill; `h[-1]` or stay within `len(...)` |
| `TypeError: 'tuple' object does not support item assignment` | assigning into a tuple output | `out[0][:] = x`, or rebuild and assign the tuple |
| Shape error deep in a later module of a batched trace | a whole-tensor write changed an invoke's row count — nothing checks it at the write | build the replacement from the activation you were served |
| Intervention has no effect | wrote to a copy, or indexed a tensor as if it were a tuple | see [Silent failures](#silent-failures) |
| Saved list is empty **when run remotely** | saved the elements, not the container | `nnsight.save([])` then append raw values |
| `NotImplementedError: ... batching multiple invokes` | base `NNsight` with 2+ input invokes | one invoke, or implement `_batch_size`/`_batch` |
| Remote job errors or returns nothing | serialization / save rules differ remotely | the `remote` skill |

Full catalogue with exact messages: [references/error-catalogue.md](references/error-catalogue.md).

<!-- test: setup -->
```python
import warnings
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
generation ended. Inside a `tracer.iter` loop an out-of-order body may not raise
at all — it shifts every request one step later; the error catalogue covers how
that shows up, including the `.source` variant.

## A loop that outran the run

A `tracer.iter` loop — bounded (`iter[:20]`) or open (`iter[:]`, `tracer.all()`) —
that asks for a step the run does not make is cut short there. The worker is
unwound *at the loop*, so every statement after it is discarded, and the only
signal is a warning. What the loop saved is kept, so the result looks complete
while holding fewer steps than the bound named:

```python
ids = None
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with model.generate(prompt, max_new_tokens=3) as tracer:
        seen = nnsight.save([])
        for step in tracer.iter[:20]:          # the run makes 3 steps, not 20
            seen.append(model.output.logits[0, -1].argmax(dim=-1))
        ids = tracer.result.save()

assert len(seen) == 3                          # short, and nothing raised
assert ids is None                             # the save after the loop never ran
assert any("was never reached" in str(w.message) for w in caught)
```

Pin the run to the count the loop asks for (`min_tokens=` / `ignore_eos=True` on
vLLM):

```python
with model.generate(prompt, max_new_tokens=3, min_new_tokens=3) as tracer:
    seen = nnsight.save([])
    for step in tracer.iter[:3]:
        seen.append(model.output.logits[0, -1].argmax(dim=-1))
    ids = tracer.result.save()

assert len(seen) == 3
assert ids.shape[-1] == len(model.tokenizer.encode(prompt)) + 3
```

If the step count isn't knowable in advance — or the run has a stop string, which
beats both minimums — loop with `tracer.all()` and put whatever follows the loop in
a separate `tracer.invoke()`, its own worker, which the unwind does not reach.
Moving it after the `with` instead does not work: `tracer.result` outside the block
raises ``Cannot access `result` outside of interleaving``.

```python
with model.generate(max_new_tokens=3) as tracer:
    with tracer.invoke(prompt):
        seen = nnsight.save([])
        for step in tracer.all():
            seen.append(model.output.logits[0, -1].argmax(dim=-1))
    with tracer.invoke():                      # a separate invoke always runs
        ids = tracer.result.save()

assert len(seen) == 3 and ids.shape[0] == 1
```

## Nothing came back

Four distinct causes; check them in this order.

**Forgot `.save()`** — the block runs in another frame, so the name is never
bound: `NameError` at module level, `UnboundLocalError` inside a function.
**Unless the name was already bound** — an earlier notebook cell, a previous loop
iteration, a rerun — in which case nothing raises and it silently keeps its *old*
value. Wrong science, no error. Check with `"name" in globals()` before the trace
or by comparing `id()`s, and save inside the block:

```python
stale = torch.zeros(3)                         # bound earlier
with model.trace(prompt):
    stale = model.transformer.h[0].output      # not saved
assert stale.shape == (3,)                     # still the old tensor
```

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

**The body has to start on its own line** — never write
`with model.trace(x): out = ...` on one line; it raises ``ValueError: The body of a
traced `with` must start on its own line``.

**The body cannot start with `try:`** — ``ValueError: A traced `with` block cannot
start with `try:` ``. nnsight intercepts the body by raising at its first line,
and CPython gives a `try` keyword's line no exception-table entry to route that
through. Wrapping the whole `with` in a `try` works, and so does any statement
above it — only the first statement is the cue:

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

**Print inside the trace.** The body runs real Python against real tensors, so
`print(resid.shape, resid.dtype, resid.device)` inside the block shows the live
value. Remotely, `print` output comes back as log lines — far cheaper than saving
a tensor just to look at it.

**Check shapes without running:** `with model.scan(prompt):` gives fake tensors
with real shapes. Do not branch on their *values* — that raises
`GuardOnDataDependentSymNode`. `scan` is unavailable on vLLM, which refuses it
with a `NotImplementedError` explaining why.

**See inside a forward:** `print(model.transformer.h[0].mlp.source)` lists every
operation you can reach, outside a trace.

**Full tracebacks:** nnsight strips most of its own frames; a few tracer frames
remain, and the last frame is your line. Frames from torch and transformers stay,
so a failure caused by a badly shaped write arrives with the model's stack under
it and does *not* name the line that wrote the value — read the `with` line at the
top and check your writes. To see everything, `nnsight.CONFIG.APP.DEBUG = True` or
`NNSIGHT_DEBUG=1 python script.py`; turn it on when you suspect the bug is in
nnsight itself, not your intervention.

**Environment:** `python scripts/check_env.py` (in the `nnsight` skill) prints
versions and GPU memory; `--remote` adds the local-vs-NDIF package diff.

## Related skills

- `nnsight` — the API itself and how tracing works
- `remote` — remote-specific failures and performance
