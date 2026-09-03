# Porting Pre-0.8 nnsight Code

Most nnsight code in tutorials, papers, blog posts, and model cards was written
for 0.4–0.7. Some of it errors immediately on 0.8; the dangerous part is what
*runs* and gives wrong numbers. Translate before debugging.

## Recognizing old code

Any of these means pre-0.8:

`.value` · `nnsight.list()` / `nnsight.dict()` · `nnsight.apply()` ·
`nnsight.cond()` · `nnsight.log()` · `nnsight.local()` · `nnsight.session()` ·
`tracer.next()` / `module.next()` · `with tracer.all():` · `with tracer.iter[...]:` ·
`LanguageModel(...)` · `model.generator.output` · `CONFIG.APP.CROSS_INVOKER` ·
the word "proxy" · `.output[0]` on every transformer block

## What you actually see

Keyed by the message, because that is what you have. Every row below was run on
0.8 against GPT-2 with transformers 5.15, from an imported module under untouched
warning filters.

### Removed — raises immediately

| Pre-0.8 | What 0.8 raises | Write instead |
|---|---|---|
| `nnsight.list()` | `AttributeError: module 'nnsight' has no attribute 'list'` | `nnsight.save([])` inside the trace |
| `nnsight.dict()` | `AttributeError: module 'nnsight' has no attribute 'dict'` | `nnsight.save({})` inside the trace |
| `nnsight.apply(fn, x)` | `AttributeError: module 'nnsight' has no attribute 'apply'` | `fn(x)` |
| `nnsight.cond(c)` | `AttributeError: module 'nnsight' has no attribute 'cond'` | `if c:` |
| `nnsight.iter(...)` | `AttributeError: module 'nnsight' has no attribute 'iter'` | `for … in …:` |
| `nnsight.log(x)` | `AttributeError: module 'nnsight' has no attribute 'log'` | `print(x)` |
| `nnsight.local()` | `AttributeError: module 'nnsight' has no attribute 'local'` | *(not ported)* |
| `nnsight.session()` | `AttributeError: module 'nnsight' has no attribute 'session'` | `model.session()` |
| `x.save()` … `x.value` | `AttributeError: 'Tensor' object has no attribute 'value'` | drop `.value` — the saved name **is** the value |
| `tracer.next()` | `AttributeError: 'InterleavingTracer' object has no attribute 'next'` | `for step in tracer.iter[:N]:` |
| `module.next()` | `AttributeError: 'Envoy' object (nor its module) has attribute 'next'` | `for step in tracer.iter[:N]:` |
| `tracer.local()` | `AttributeError: 'InterleavingTracer' object has no attribute 'local'` | *(not ported)* — use a remote session |
| `CONFIG.APP.CROSS_INVOKER` | `AttributeError: 'AppConfig' object has no attribute 'CROSS_INVOKER'` | nothing needed — invokes share values by default. A `tracer.barrier(n)` is only for the ordering case described in `docs/usage/invoke-and-batching.md` |
| `CONFIG.APP.CACHE_DIR` / `TRACE_CACHING` | `AttributeError: 'AppConfig' object has no attribute 'CACHE_DIR'` | nothing needed |
| `nnsight.save([])` before the `with` | `ValueError: save() was called outside a trace. …` | move the save inside the block |

None of these name a replacement, so grep for the old name rather than reading
the message.

### Deprecated — runs, and warns

Every one warns under `nnsight.NNsightDeprecationWarning`, a `FutureWarning`.
That category is shown by Python's default filters wherever the call sits — a
script, a helper module, a package, a notebook — so a warning missing from your
output means the idiom is not there, not that it was hidden.

| Pre-0.8 | The warning | Write instead |
|---|---|---|
| `LanguageModel(repo)` | `LanguageModel is deprecated; use TransformersModel(repo_id, task='text-generation') instead.` | `TransformersModel(repo, task="text-generation")` |
| `VisionLanguageModel(repo)` | `VisionLanguageModel is deprecated; use TransformersModel(repo_id, task='image-text-to-text') instead.` | `TransformersModel(repo, task="image-text-to-text")` |
| `model.iter[:3]` | `model.iter is deprecated; use tracer.iter instead.` | `tracer.iter[:3]` |
| `model.all()` | `model.all() is deprecated; use tracer.all() instead.` | `tracer.all()` |
| `with tracer.iter[:3]:` / `with tracer.all():` | ``The `with tracer.iter[...]:` / `with tracer.all():` block form is deprecated; use `for step in tracer.iter[...]:` instead.`` | `for step in tracer.iter[:3]:` |
| `model.generator.output` | `model.generator.output is deprecated; use tracer.result instead (model.generator.streamer.output still gives per-step tokens).` | `tracer.result` |
| `nnsight.ndif_status()` | `nnsight.ndif_status() is deprecated; use nnsight.status() instead.` | `nnsight.status()` |

`model.generator.output` and `tracer.result` carry the identical tensor — the
prompt's ids followed by the generated ones, `[1, 13]` for a 10-token prompt plus
3 new — so that row is a straight rename with nothing to re-index. Per-step
tokens are `model.generator.streamer.output`, which is not deprecated.

To silence nnsight's deprecations and no other library's:

```python
import warnings, nnsight
warnings.filterwarnings("ignore", category=nnsight.NNsightDeprecationWarning)
```

### Runs, says nothing, and may be wrong

| Pre-0.8 | What 0.8 does | Write instead |
|---|---|---|
| `block.output[0]` on a transformer block | Runs. `.output` is a plain `Tensor [1, 10, 768]`, so `[0]` is **batch row 0**, `[10, 768]` | `block.output` |
| `attn.output[0]` | Runs, and is still right — an attention submodule's `.output` really is a tuple | unchanged |
| `results = []` outside; `results.append(x.save())` | Runs locally and collects all 12 elements; returns nothing **remotely** | `results = nnsight.save([])` inside the trace, appending raw values |
| `hidden.retain_grad()`; `logits.sum().backward()`; read `hidden.grad` after | Runs, and gives the identical gradient | fine for reading one gradient; use `with metric.backward():` to edit one or read several in reverse order |
| `with tracer.all():` over N steps | Runs the loop, drops everything after it | bound the loop, or move the trailing code past the `with` |

## The three that silently change results

**1. `.output[0]` on transformer blocks.** Blocks used to return tuples. In
current `transformers` a GPT-2/Llama block returns a plain tensor, so `[0]` now
selects **batch row 0**. Shapes stay plausible, results are wrong.

```diff
- hidden = model.transformer.h[5].output[0]      # was: unwrap the tuple
+ hidden = model.transformer.h[5].output         # now: already a tensor
```

Attention submodules *do* still return tuples, so `attn.output[0]` is usually
still right. Confirm per model with `scripts/inspect_model.py --prompt ...`.

**2. Unbounded iteration.** Old nnsight bounded `tracer.all()` internally; 0.8
does not, and the over-run unwinds every line after the loop:

```diff
  with model.generate(prompt, max_new_tokens=5) as tracer:
-     for step in tracer.all():
+     for step in tracer.iter[:5]:
          collected.append(model.output.logits[0, -1].argmax(dim=-1))
      ids = tracer.result.save()     # only runs if the loop is bounded
```

A bound is a claim about the run, so a `tracer.iter[:5]` the run does not reach
raises rather than warning:

```
OutOfOrderError: 'model.output.i3' was never reached: the loop asked for iteration
3 of 'model.output' and the run reached it 3 times, so the loop was cut short and
nothing after it ran. …
```

Add `min_new_tokens=5` alongside `max_new_tokens=5` so the generation cannot stop
short of what the loop asks for.

**3. Saving list elements.** Old code often wrote `results.append(x.save())` into
a list created outside the trace. It collects values locally and returns nothing
remotely. Save the container instead:

```diff
- results = []
  with model.trace(prompt):
+     results = nnsight.save([])
      for block in model.transformer.h:
-         results.append(block.output.save())
+         results.append(block.output[0, -1])
```

## Gradients

Old code enabled grads by hand and read `.grad` after the trace:

```python-legacy
with model.trace(prompt):
    hidden = model.transformer.h[-1].output[0].save()
    hidden.retain_grad()
    logits = model.lm_head.output.save()
    logits.sum().backward()
print(hidden.grad)
```

That still runs on 0.8 and gives the same gradient, so it is not the first thing
to change — drop the `[0]` (point 1 above) and it is correct. The 0.8 form is what
you need in order to *edit* a gradient mid-backward, or to read `.grad` on several
tensors, which must happen in reverse-forward order inside the block. No
`retain_grad()` / `requires_grad_()` needed:

<!-- test: setup -->
```python
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
```

```python
with model.trace(prompt):
    hidden = model.transformer.h[-1].output
    metric = model.output.logits.sum()
    with metric.backward():
        grad = hidden.grad.clone().save()

assert grad.shape == (1, 10, 768)
assert grad.abs().sum() > 0
```

## `save()` now raises outside a trace

A pattern that used to be a silent no-op is now an error — build accumulators
*inside* the trace:

```diff
- buffer = nnsight.save([])          # ValueError on 0.8
  with model.trace(prompt):
+     buffer = nnsight.save([])
      buffer.append(model.transformer.h[0].output[0, -1])
```

## Porting checklist

1. Replace `LanguageModel` → `TransformersModel`; drop every `.value`.
2. Replace `nnsight.list/dict/apply/cond/log/session` with plain Python /
   `model.session()`.
3. Convert `with tracer.all():` / `tracer.next()` to a **bounded**
   `for step in tracer.iter[:N]:`, and add `min_new_tokens=N` so the run reaches
   the bound.
4. Re-check every `.output[0]` against the real output type.
5. Move accumulator creation inside the trace; save containers, not elements.
6. Swap `model.generator.output` for `tracer.result` — same tensor, no re-indexing.
7. If it generated text, decide whether you want `generate` (ids) or `pipe` (text).
8. Run it with warnings on. Anything still deprecated says so and names its
   replacement.
9. Sanity-check: does the unmodified baseline reproduce the expected output, and
   does an extreme intervention actually move the metric?
