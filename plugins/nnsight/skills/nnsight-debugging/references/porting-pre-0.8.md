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

## Translation table

| Pre-0.8 | 0.8 | Note |
|---|---|---|
| `LanguageModel(repo)` | `TransformersModel(repo)` | old name warns; `VisionLanguageModel` → `TransformersModel(task="image-text-to-text")` |
| `x = m....output.save()` … `x.value` | `x = m....output.save()` … `x` | **no `.value`** — the saved name *is* the value |
| "the proxy" | a real `torch.Tensor` | values inside a trace are real; `.item()`, `.shape`, `if` all work |
| `nnsight.list()` / `nnsight.dict()` | `nnsight.save([])` / `nnsight.save({})` | plain Python types; save the container |
| `nnsight.apply(fn, x)` | `fn(x)` | just call it |
| `nnsight.cond(c)` / `nnsight.iter(...)` | `if` / `for` | ordinary Python in the trace body |
| `nnsight.log(x)` | `print(x)` | remote prints come back as log lines |
| `nnsight.session()` | `model.session()` | there is no top-level session |
| `tracer.next()` / `module.next()` | `for step in tracer.iter[:N]:` | manual stepping is gone |
| `with tracer.all():` | `for step in tracer.all():` | the `with` form is deprecated; prefer a **bounded** `tracer.iter[:N]` |
| `with tracer.iter[a:b]:` | `for step in tracer.iter[a:b]:` | same |
| `model.iter` / `model.all()` | `tracer.iter` / `tracer.all()` | deprecated on the model |
| `model.generator.output` | `tracer.result` | still works for per-step `.streamer.output` |
| `generate(...)` → decoded text | `model.pipe(...)` | 0.8 `generate` returns **token ids** |
| `nnsight.ndif_status()` | `nnsight.status()` | old name is a deprecated alias |
| `CONFIG.APP.CROSS_INVOKER` | *(gone)* | cross-invoke sharing is automatic; use `tracer.barrier(n)` when writing across the same module |
| `CONFIG.CACHE_DIR`, `TRACE_CACHING` | *(gone)* | trace caching is always on |
| `tracer.local()` hybrid streaming | *(not ported)* | use a remote session |
| `hidden.retain_grad()` + `loss.backward()` | `with metric.backward():` … `hidden.grad` | see below |

## The three that silently change results

**1. `.output[0]` on transformer blocks.** Blocks used to return tuples. In current
`transformers` a GPT-2/Llama block returns a plain tensor, so `[0]` now selects
**batch row 0**. Shapes stay plausible, results are wrong.

```diff
- hidden = model.transformer.h[5].output[0]      # was: unwrap the tuple
+ hidden = model.transformer.h[5].output         # now: already a tensor
```

Attention submodules *do* still return tuples, so `attn.output[0]` is usually
still right. Confirm per model with `scripts/inspect_model.py --prompt ...`.

**2. Unbounded iteration.** Old nnsight bounded `tracer.all()` internally; 0.8 does
not, and the over-run unwinds every line after the loop.

```diff
  with model.generate(prompt, max_new_tokens=5) as tracer:
-     for step in tracer.all():
+     for step in tracer.iter[:5]:
          collected.append(model.output.logits[0, -1].argmax(dim=-1))
      ids = tracer.result.save()     # only runs if the loop is bounded
```

**3. Saving list elements.** Old code often wrote `results.append(x.save())` into a
list created outside the trace. It appeared to work locally and returns nothing
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

0.8 interleaves the backward pass, so `.grad` is read **inside** a
`with metric.backward():` block, in reverse-forward order, on the tensor you
captured (not a slice of it). No `retain_grad()` / `requires_grad_()` needed:

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
   `for step in tracer.iter[:N]:`.
4. Re-check every `.output[0]` against the real output type.
5. Move accumulator creation inside the trace; save containers, not elements.
6. Convert gradient code to `with metric.backward():`.
7. If it generated text, decide whether you want `generate` (ids) or `pipe` (text).
8. Run it, then sanity-check: does the unmodified baseline reproduce the expected
   output, and does an extreme intervention actually move the metric?
