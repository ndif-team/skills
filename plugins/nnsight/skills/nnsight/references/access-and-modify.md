# Access and Modify: Tuples, Inputs, Heads, Repeated Modules

[SKILL.md](../SKILL.md) covers reading `.output`, in-place and replacement edits,
and calling a module on a value. This page is the cases past that.

```python
import torch
import nnsight
from nnsight import NNsight, TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
```

## Tuple outputs

An attention module returns a tuple. Assigning into it, `attn.output[0] = x`, is
`tuple.__setitem__` and raises `TypeError: 'tuple' object does not support item
assignment`. Write into the element, or rebuild the tuple and assign it:

```python
with model.trace(prompt):
    model.transformer.h[0].attn.output[0][:] = 0                     # into the element

    attn = model.transformer.h[1].attn
    attn.output = (torch.zeros_like(attn.output[0]),) + tuple(attn.output[1:])   # a new tuple

    kinds = nnsight.save([type(model.transformer.h[2].attn.output).__name__,
                          type(model.transformer.h[2].output).__name__])

assert kinds == ["tuple", "Tensor"]
```

Replacement is the only way to change a value that is not a tensor. A replacement
built from a fresh tensor (`torch.zeros_like`) cuts the autograd graph at that
point; see [gradients](gradients.md#replacements-and-the-graph) when a backward
pass follows.

## Inputs

`.input` is the first positional argument (the first keyword argument when there
are none) and assigning to it repacks the call. Everything else goes through
`.inputs`, the `(args, kwargs)` pair, assigned whole:

```python
with model.trace(prompt):
    block = model.transformer.h[3]
    args, kwargs = block.inputs
    block.inputs = ((args[0] * 0.5, *args[1:]), kwargs)
    names = nnsight.save(sorted(kwargs))
    halved = model.output.logits.save()

print(names)
```

`model.inputs` is the whole model's call: `kwargs["input_ids"]` and
`kwargs["attention_mask"]` as the forward received them, padding included. Read
it before any module.

## Attention heads

Heads are separate only at the **input of the attention's output projection**:
`attn.c_proj.input` on GPT-2, `self_attn.o_proj.input` on Llama-style models
(`scripts/inspect_model.py <repo_id> --grep attn` prints the name for any other
family). Head `i` is the slice `[..., i * head_dim : (i + 1) * head_dim]` of that
tensor. `attn.output` is past the projection: the same slice there mixes every
head, raises nothing, and ranks different "heads".

```python
attn = model.transformer.h[5].attn
head_dim = attn._module.head_dim          # read it off the module
head = 4

with model.trace() as tracer:
    out = nnsight.save({})
    with tracer.invoke(prompt):
        out["clean"] = model.output.logits[0, -1]
    with tracer.invoke(prompt):
        attn.c_proj.input[:, :, head * head_dim:(head + 1) * head_dim] = 0     # ablate head 4
        out["ablated"] = model.output.logits[0, -1]

assert not torch.equal(out["clean"], out["ablated"])
```

- **Read `head_dim` from the module.** `hidden_size // num_attention_heads` is
  wrong on some families: on `google/gemma-2-2b` it gives 288 and the module's
  `head_dim` is 256. A slice with the wrong width raises nothing.
- **Grouped-query attention** (Llama-3, Qwen, SmolLM2): the projection input has
  one slice per *query* head, `num_attention_heads` of them. Only `k_proj` and
  `v_proj` outputs are indexed by `num_key_value_heads`.
- A head's write to the residual stream is its slice through the projection's
  weight: `z[..., lo:hi] @ c_proj.weight[lo:hi]` on GPT-2's `Conv1D` (weight is
  `[in, out]`), `z[..., lo:hi] @ o_proj.weight[:, lo:hi].T` on an `nn.Linear`.

For a reusable per-head view, define a custom value:
[modules-and-architectures](modules-and-architectures.md#custom-values-eproperty-and-envoys).
Attention probabilities are a different object:
[attention-analysis](attention-analysis.md).

## A module that runs more than once

`.output` is the module's **first** call in the forward pass. Reading it twice
returns that first value twice, so a weight-tied or looped module looks as if
every pass computed the same thing. `tracer.iter` steps through the calls, the
same way it steps through generation steps:

```python
class Looped(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = torch.nn.Linear(4, 4)

    def forward(self, x):
        for _ in range(3):
            x = torch.relu(self.block(x)) + x
        return x

net = NNsight(Looped())
x = torch.ones(1, 4)

with net.trace(x) as tracer:
    norms = nnsight.save([])
    for call in tracer.iter[:3]:                 # call 0, 1, 2 of every module read below
        norms.append(net.block.output.norm())

with net.trace(x) as tracer:
    for call in tracer.iter[1]:                  # the second call only
        net.block.output[:] = 0
    edited = net.output.save()

assert len(norms) == 3 and len({round(float(n), 4) for n in norms}) == 3
```

A loop that asks for more calls than the forward makes is cut short the way a
generation loop is ([generation](generation.md#the-one-rule-for-iteration-loops)).
Under `model.generate`, a module called `k` times per forward makes `k`
occurrences per step, so `tracer.iter[n]` no longer means step `n` for it.

## Calling a module with `hook=True`

`model.lm_head(x)` inside a trace runs the module with nnsight stood down: the
call serves no values and its submodules cannot be read. Pass `hook=True` when
the called module's own internals should be readable, which is how an attached
SAE or adapter is observed:
[modules-and-architectures](modules-and-architectures.md#attaching-a-module).

## A child named `output` or `input`

A submodule named like an nnsight property (BERT's attention has a child called
`output`) is reached as `.E_output` / `.E_input`; `.output` stays the module's own
value. nnsight warns at load, and `print(model)` labels the child `E_output/output`.
Its path in `named_modules()` and in cache keys is unchanged.
