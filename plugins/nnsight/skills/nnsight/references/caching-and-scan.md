# Caching Many Modules, and Scanning for Shapes

`tracer.cache()` records many modules in one trace with no ordering to respect.
`model.scan()` gives shapes without running the model or loading its weights.

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
```

## tracer.cache()

Open it first thing in the trace and name the modules. An unfiltered cache
records every module in the model:

```python
with model.trace(prompt) as tracer:
    cache = tracer.cache(modules=[block.mlp for block in model.transformer.h])

mlp3 = cache["model.transformer.h.3.mlp"].output        # by path
assert mlp3 is cache.transformer.h[3].mlp.output         # or by navigation
assert tuple(mlp3.shape) == (1, 10, 768) and not mlp3.requires_grad
assert len(cache.keys()) == 12
```

`modules=` takes envoys or path strings. The other arguments, with their
defaults: `device=torch.device("cpu")` (`None` leaves tensors where they are),
`dtype=None`, `detach=True`, `include_output=True`, `include_inputs=False`
(set it to read `entry.input` / `entry.inputs`).

| Rule | Consequence |
|---|---|
| Open the cache before reading or writing any activation | otherwise `ValueError: tracer.cache() must be declared before reading or modifying a model value` |
| It records values **after** the block's edits | cache and edit in one trace, and the cache holds the edited value |
| A module that ran several times gives a **list** | under `generate`, `entry.output` is a list with one tensor per step: `(1, prompt_len, hidden)`, then `(1, 1, hidden)` |
| `cache.keys()` is in the order the model reached them | not the order of `modules=`; do not zip the two |
| Inside an invoke it holds that invoke's rows, padded to the batch's length | mask before reducing ([batching](batching.md#positions-shift-with-the-batch)); for the whole batch, open the cache in an empty `tracer.invoke()` |
| A module you call yourself is recorded only with `hook=True` | an attached SAE or adapter is otherwise missing from the cache |
| `detach=False, device=None` keeps the graph | `cache[...].output.sum().backward()` then works, and the forward graph lives as long as the cache |

```python
with model.generate(prompt, max_new_tokens=3, min_new_tokens=3, do_sample=False) as tracer:
    steps = tracer.cache(modules=[model.transformer.h[-1]])

per_step = steps["model.transformer.h.11"].output
assert isinstance(per_step, list) and len(per_step) == 3
assert per_step[0].shape[1] == 10 and per_step[1].shape[1] == 1
```

A cache pickles without the model, so `torch.save(cache, path)` and
`torch.load(path, weights_only=False)` work. On the loaded object use path keys;
navigation (`cache.transformer.h[3]`) needs the model.

A cache costs what a hand-written loop of saves over the same modules costs. Use
it to avoid ordering the reads, not for speed.

## model.scan()

`scan` runs the forward under fake tensors. Shapes and dtypes are real, values
are not, and the model is never dispatched:

```python
meta = TransformersModel("openai-community/gpt2", task="text-generation")   # no dispatch

with meta.scan(prompt):
    attn_kind = nnsight.save(type(meta.transformer.h[0].attn.output).__name__)
    resid_shape = nnsight.save(tuple(meta.transformer.h[-1].output.shape))

assert resid_shape == (1, 10, 768) and attn_kind == "tuple"
assert not meta.dispatched
```

It is a tracing context like any other: saves, forward order and invokes apply.

- **Shapes only.** Branching on a value (`if x.mean() > 0:`) raises
  `GuardOnDataDependentSymNode: Could not guard on data-dependent expression`.
- `scan` belongs to the classes that load from a repo id. A plain
  `NNsight(module)` has none (`AttributeError: 'NNsight' object (nor its module)
  has attribute 'scan'`), and `VLLM` refuses it. Trace an input there.
- Some operations have no fake-tensor kernel and some vision-language
  preprocessing takes a size from tensor data. When `scan` raises on a model
  that traces fine, use a real trace.

`scripts/inspect_model.py --prompt` is a scan over every block child.
