# Source Tracing — Values Inside a Forward

`.output` and `.input` only reach values at module boundaries. When the value you
want is computed *inside* a `forward` and never returned — attention
probabilities, a pre-activation, an intermediate projection — use `.source`.

nnsight rewrites the module's forward AST so every call site becomes hookable,
with the same `.input` / `.output` / `.skip` interface one level finer.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True,
                          attn_implementation="eager")
prompt = "The cat sat on the"
```

## Discovering operations

`print(module.source)` renders the forward with each call site labelled. It works
**outside** a trace, so use it to explore before writing anything:

```python
print(model.transformer.h[0].mlp.source)
```

```
                    * def forward(self, hidden_states: tuple[torch.FloatTensor] | None) -> torch.FloatTensor:
 self_c_fc_0    ->  0     hidden_states = self.c_fc(hidden_states)
 self_act_0     ->  1     hidden_states = self.act(hidden_states)
 self_c_proj_0  ->  2     hidden_states = self.c_proj(hidden_states)
 self_dropout_0 ->  3     hidden_states = self.dropout(hidden_states)
                    4     return hidden_states
```

Names are the dotted callee joined by `_`, plus an occurrence index in execution
order: `self.act(...)` → `self_act_0`, `torch.relu(...)` → `torch_relu_0`, a second
`relu` → `torch_relu_1`. **Do not guess these names** — print the source. They
change with the `transformers` version.

## Reading and replacing an operation

```python
with model.trace(prompt):
    post_gelu = model.transformer.h[0].mlp.source.self_act_0.output.save()
    model.transformer.h[0].mlp.source.self_c_proj_0.output[:] = 0    # ablate the write
    logits = model.output.logits.save()

print(post_gelu.shape)
```

`.input` is the first argument, `.inputs` is `(args, kwargs)`, and `.skip(value)`
bypasses the call entirely — same semantics as a module.

## The main use: attention patterns

The attention module returns the value-weighted result, not the probability
matrix. The probabilities come from the attention-interface call:

```python
with model.trace(prompt):
    attn_out, attn_weights = (
        model.transformer.h[0].attn.source.attention_interface_0.output.save()
    )

print(attn_weights.shape)            # [batch, heads, q_seq, k_seq]
print(attn_weights[0, 0].sum(-1))    # each row sums to 1
```

**You must load the model with `attn_implementation="eager"`.** The default `sdpa`
kernel never materializes the probability matrix and returns `None` for the
weights.

All layers in one pass:

```python
with model.trace(prompt):
    patterns = nnsight.save([])
    for block in model.transformer.h:
        _, weights = block.attn.source.attention_interface_0.output
        patterns.append(weights)

print(len(patterns), patterns[0].shape)
```

A quick induction-ish diagnostic — how much each head attends to the previous
token:

```python
with model.trace(prompt):
    _, weights = model.transformer.h[5].attn.source.attention_interface_0.output
    prev_token_score = weights.diagonal(offset=-1, dim1=-2, dim2=-1).mean(-1).save()

for head, score in enumerate(prev_token_score[0].tolist()):
    print(f"L5H{head:<2} previous-token attention {score:.3f}")
```

## Drilling deeper

An operation that calls a plain Python function can itself be traced — chain
`.source` again. This only works **inside** a trace, because the callee is
resolved from the live value at run time:

```python
with model.trace(prompt):
    inner = (model.transformer.h[0].attn
             .source.attention_interface_0
             .source.attn_output_transpose_0.output.save())

print(inner.shape)
```

## Iteration

Source ops work with `tracer.iter[...]`, and their counter is **per invocation**,
not per forward pass — an op inside a loop within one forward fires many times in
one step:

```python
with model.generate(prompt, max_new_tokens=3) as tracer:
    per_step = nnsight.save([])
    for step in tracer.iter[:3]:
        _, weights = model.transformer.h[0].attn.source.attention_interface_0.output
        per_step.append(weights.shape[-1])

print(per_step)      # key length grows as generation proceeds
```

## Limits

| Limit | What to do |
|---|---|
| The op calls a **submodule** | Refused (`SourceNotAvailable`) — use that submodule's own `.source` |
| The op calls a **builtin / C function** | No Python source exists; not drillable |
| The `forward` is **decorated** | Rejected — the decorator is load-bearing |
| Recursive `.source` outside a trace | Raises; open a trace first |
| Ops requested out of execution order | `OutOfOrderError`, same as modules |

Prefer a real submodule when one exposes the value (`mlp.output` beats
`source.self_c_proj_0.output`) — it is cheaper and stable across library versions.

## Related

- [access-and-modify.md](access-and-modify.md)
- [modules-and-architectures.md](modules-and-architectures.md) — per-head slicing without `.source`
- [execution-model.md](execution-model.md)
