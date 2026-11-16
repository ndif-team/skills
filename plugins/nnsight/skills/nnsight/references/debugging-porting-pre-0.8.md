# Porting Code Written for nnsight Before 0.8

Most nnsight code in tutorials, papers and model cards predates 0.8. Some of it
raises at once; the dangerous part runs and gives wrong numbers. Translate the
whole snippet with these tables before debugging any one line.

Any of these marks old code: `.value`, `nnsight.list()` / `nnsight.dict()`,
`nnsight.apply()`, `nnsight.cond()`, `nnsight.log()`, `nnsight.session()`,
`tracer.next()` / `module.next()`, `with tracer.all():`, `with tracer.iter[...]:`,
`LanguageModel(...)`, `model.generator.output`, the word "proxy", and `.output[0]`
on every transformer block.

## Runs, says nothing, and may be wrong

| Old code | What happens now | Write |
|---|---|---|
| `block.output[0]` on a transformer block | `.output` is a plain `(batch, seq, hidden)` tensor, so `[0]` is **batch row 0** | `block.output` |
| `attn.output[0]` | still right: an attention module returns a tuple | unchanged |
| `results = []` outside the trace; `results.append(x.save())` inside | fills locally, comes back empty from a remote run | `results = nnsight.save([])` inside the trace, append raw values |
| `with tracer.all():` then code after it | the statements after the loop are dropped | `for step in tracer.iter[:N]:` with `min_new_tokens=N`, or the trailing code in a separate `tracer.invoke()` |
| `model.generate(...)` expected to return text | it returns token ids; sampling follows the checkpoint's `generation_config` | `model.tokenizer.decode(ids[0])`, or `model.pipe(...)` for decoded records; `do_sample=False` |
| `LanguageModel(repo, device_map="cpu")` | lands on the GPU when there is one | `TransformersModel(repo, task="text-generation", device="cpu")` |
| `hidden.retain_grad()`; `loss.backward()`; `hidden.grad` after the trace | works, same gradient | keep it for reading; `with metric.backward():` to edit a gradient or read only a few |

## Removed: raises at once

None of these messages names the replacement.

| Old code | Error | Write |
|---|---|---|
| `x.value` | `AttributeError: 'Tensor' object has no attribute 'value'` | drop `.value`; the saved name is the value |
| `nnsight.list()` / `nnsight.dict()` | `AttributeError: module 'nnsight' has no attribute 'list'` | `nnsight.save([])` / `nnsight.save({})` inside the trace |
| `nnsight.apply(fn, x)`, `nnsight.cond(c)`, `nnsight.iter(...)`, `nnsight.log(x)` | the same `AttributeError` | `fn(x)`, `if c:`, `for ... in ...:`, `print(x)` |
| `nnsight.session()` | the same `AttributeError` | `model.session()` |
| `nnsight.local()`, `tracer.local()` | `AttributeError` | not available; use a remote session |
| `tracer.next()`, `module.next()` | `AttributeError: ... has no attribute 'next'` | `for step in tracer.iter[:N]:` |
| `module.nns_output`, `module.nns_input` | `AttributeError` | `module.E_output`, `module.E_input` |
| `CONFIG.APP.CROSS_INVOKER`, `CACHE_DIR`, `TRACE_CACHING` | `AttributeError: 'AppConfig' object has no attribute ...` | delete the line; invokes share values by default |
| a save before or after the `with` | `ValueError: save() was called outside a trace` | move it inside |

## Deprecated: runs and warns

Each warns under `nnsight.NNsightDeprecationWarning`, a `FutureWarning`, with a
message that names the replacement. Silence nnsight's alone with
`warnings.filterwarnings("ignore", category=nnsight.NNsightDeprecationWarning)`.

| Old code | Write |
|---|---|
| `LanguageModel(repo)` | `TransformersModel(repo, task="text-generation")` |
| `VisionLanguageModel(repo)` | `TransformersModel(repo, task="image-text-to-text")` |
| `model.iter[:3]`, `model.all()` | `tracer.iter[:3]`, `tracer.all()` |
| `with tracer.iter[:3]:` | `for step in tracer.iter[:3]:` |
| `model.generator.output` | `tracer.result` (the same tensor) |
| `nnsight.ndif_status()` | `nnsight.status()` |

## A port, start to finish

```python
# before
from nnsight import LanguageModel
model = LanguageModel("gpt2", device_map="auto")
with model.trace("The Eiffel Tower is in"):
    h = model.transformer.h[5].output[0].save()
print(h.value.shape)
```

```python
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
with model.trace("The Eiffel Tower is in"):
    h = model.transformer.h[5].output.save()       # a tensor, not a tuple: no [0]

assert tuple(h.shape) == (1, 7, 768)               # no .value
```

Then check the port: every `.output[0]` against the real output type
(`scripts/inspect_model.py <repo_id> --prompt "..."`), every loop over generation
steps against the run's length, and that the unmodified baseline still gives the
expected output.
