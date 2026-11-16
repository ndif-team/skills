# Source Tracing: Values Inside a Forward

`.output` and `.input` reach values at module boundaries. For a value computed
inside a `forward` and never returned (attention scores, a pre-activation, an
intermediate projection) use `.source`: every call and every assignment in the
forward becomes a location with the same `.output` / `.input` / `.inputs` /
`.skip` interface as a module.

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True,
                          attn_implementation="eager")
prompt = "The cat sat on the"
```

## Find the operation: print the source

`print(module.source)` works outside a trace and lists the forward with a label
on every operation. **Read names off this listing; never guess them**, they
change with the model class and the transformers version.

```python
print(model.transformer.h[0].mlp.source)
```

```
                     * def forward(self, hidden_states: tuple[torch.FloatTensor] | None) -> torch.FloatTensor:
 self_c_fc_0     ->  0     hidden_states = self.c_fc(hidden_states)
 hidden_states_0 ->  +     ...
 self_act_0      ->  1     hidden_states = self.act(hidden_states)
 hidden_states_1 ->  +     ...
```

A call is named by its dotted callee joined with `_` plus an occurrence index
(`self.act(...)` is `self_act_0`, a second `torch.relu` is `torch_relu_1`). An
assignment is named after its target (`hidden_states_1`), and its `.output` is the
assigned value, which need not be a tensor. Calls and assignments share one
counter per name.

```python
with model.trace(prompt):
    mlp = model.transformer.h[0].mlp
    post_gelu = mlp.source.self_act_0.output.save()
    mlp.source.self_c_proj_0.output[:] = 0            # ablate the write
    logits = model.output.logits.save()

assert post_gelu.shape[-1] == 4 * model.config.n_embd
```

## Attention scores and probabilities

Load with `attn_implementation="eager"`: the default kernel never builds the
probability matrix. The attention module's forward calls the eager attention
function (`attention_interface_1` here); drill into that call with a second
`.source`, inside the trace, to reach what it assigns:

```python
attn = model.transformer.h[0].attn
_ = attn.source                                   # instrument once, before the trace (see Limits)

with model.trace(prompt):
    inner = attn.source.attention_interface_1.source          # ask for the drill first
    scores = inner.attn_weights_0.output.save()               # q @ k^T * scale, before the mask
    probs = inner.attn_weights_2.output.save()                # after the mask and the softmax
    out, weights = attn.source.attention_interface_1.output   # then the call's own result
    returned = weights.save()

assert tuple(probs.shape) == (1, model.config.n_head, 5, 5)                   # batch, heads, query, key
assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-4)
assert not torch.allclose(scores.sum(-1), torch.ones_like(scores.sum(-1)))    # pre-softmax
assert torch.equal(returned, probs)
```

Llama-style models call the same function with the same assignment names:
`model.model.layers[i].self_attn.source.attention_interface_1.source` has
`attn_weights_0` (scaled scores), `attn_weights_1` (mask added), `attn_weights_2`
(softmax). The scores are taken before the causal mask is added, so future keys
hold real numbers there.

When all you need is the probabilities, `attn.output[1]` is the same tensor with
no `.source` at all: [attention-analysis](attention-analysis.md).

Every layer in one trace:

```python
with model.trace(prompt):
    patterns = nnsight.save([block.attn.source.attention_interface_1.output[1]
                             for block in model.transformer.h])

assert len(patterns) == model.config.n_layer and not torch.equal(patterns[0], patterns[-1])
```

## Loops and generation

An operation inside a Python loop is one location that fires once per iteration,
so `tracer.iter[k]` selects the k-th. Under `model.generate` its counter runs per
call, not per step. In a generation loop read an operation *before* the output of
the module that contains it; after it, the read binds to the next step and the
list comes back one short
([generation](generation.md#source-operations-inside-a-loop)).

## Limits

| Situation | What happens |
|---|---|
| The **first** `.source` on a module is asked after that module's forward has begun: a block or the model itself, once any of its children was read | `OutOfOrderError: ... already ran past it` on an operation that has not run yet. Instrumenting rewrites the forward, so do `_ = module.source` before the trace |
| `op.source` asked after `op.output` on the same operation | `OutOfOrderError` on `...{op}.fn`; the drill is served just before the call runs, so ask for it first |
| An operation on a branch this model never takes | `OutOfOrderError`, worded as though you were late. The listing shows every branch; GPT-2's `transpose_0` is the cross-attention key transpose that never runs, `transpose_2` the one that does. Look for the live twin further down |
| Operations asked out of execution order | `OutOfOrderError`, as for modules |
| The operation calls a **submodule** | `SourceNotAvailable`; use that submodule's own `.source` |
| The operation is an **assignment**, a builtin or C function | no source to drill into; read its `.output` |
| The function names itself in its body (every `torch.nn.functional` entry point) | `nn_functional_softmax_0.source` raises `KeyError: 'softmax'`; read its `.output` |
| Recursive `.source` outside a trace | raises; the callee is resolved from the live value |
| Recursive `.source` on NDIF | may raise `SourceNotAvailable`; read the outer operation's `.output` |

Prefer a real submodule when one exposes the value (`mlp.output` over
`source.self_c_proj_0.output`): it is cheaper and survives a transformers upgrade.
For a writable per-head view of the projection input with no `.source`, see
[modules-and-architectures](modules-and-architectures.md#custom-values-eproperty-and-envoys).
