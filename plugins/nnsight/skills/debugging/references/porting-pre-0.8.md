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

Keyed by the message, because that is what you have.

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
| `CONFIG.APP.CROSS_INVOKER` | `AttributeError: 'AppConfig' object has no attribute 'CROSS_INVOKER'` | nothing needed — invokes share values by default. A `tracer.barrier(n)` is only for the ordering case in the `nnsight` skill's batching reference |
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

`model.generator.output` → `tracer.result` is a straight rename: the same tensor
(prompt ids then generated ids), nothing to re-index. To silence nnsight's
deprecations and no other library's,
`warnings.filterwarnings("ignore", category=nnsight.NNsightDeprecationWarning)`.

### Runs, says nothing, and may be wrong

| Pre-0.8 | What 0.8 does | Write instead |
|---|---|---|
| `block.output[0]` on a transformer block | Runs. `.output` is a plain `Tensor [1, 10, 768]`, so `[0]` is **batch row 0**, `[10, 768]` | `block.output` |
| `attn.output[0]` | Runs, and is still right — an attention submodule's `.output` really is a tuple | unchanged |
| `results = []` outside; `results.append(x.save())` | Runs locally and collects all 12 elements; returns nothing **remotely** | `results = nnsight.save([])` inside the trace, appending raw values |
| `hidden.retain_grad()`; `logits.sum().backward()`; read `hidden.grad` after | Runs, and gives the identical gradient | fine for reading one gradient; use `with metric.backward():` to edit one or read several in reverse order |
| `with tracer.all():` over N steps | Runs the loop, drops everything after it | bound the loop with `min_new_tokens=N`, or move the trailing code into a separate `tracer.invoke()` — not past the `with`, where `tracer.result` raises |

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
to change — drop the `[0]` (the silent table above) and it is correct. The 0.8
form is what you need in order to *edit* a gradient mid-backward, or to read
`.grad` on several tensors, which must happen in reverse-forward order inside the
block. No `retain_grad()` / `requires_grad_()` needed:

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

## Porting checklist

1. Work through the three tables above: every removed name, every deprecation
   warning, every silent row.
2. Re-check every `.output[0]` against the real output type
   (`scripts/inspect_model.py --prompt ...` in the `nnsight` skill) — on 0.8 an
   attention submodule still returns a tuple, a block does not.
3. For every `tracer.iter` loop, decide what bounds the run: `min_new_tokens=N`
   for a bounded loop, `tracer.all()` plus a separate invoke where a stop string
   can end it. 0.8 does not bound `tracer.all()`.
4. If it generated text, decide whether you want `generate` (ids) or `pipe` (text).
5. Sanity-check: does the unmodified baseline reproduce the expected output, and
   does an extreme intervention actually move the metric?
