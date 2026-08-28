# Source Tracing — Values Inside a Forward

`.output` and `.input` only reach values at module boundaries. When the value you
want is computed *inside* a `forward` and never returned — attention
probabilities, a pre-activation, an intermediate projection — use `.source`.

nnsight rewrites the module's forward AST so every call site *and every assignment*
becomes hookable, with the same `.input` / `.output` / `.skip` interface one level finer.

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

`print(module.source)` renders the forward with each call site and assignment labelled. It works
**outside** a trace, so use it to explore before writing anything:

```python
print(model.transformer.h[0].mlp.source)
```

```
                     * def forward(self, hidden_states: tuple[torch.FloatTensor] | None) -> torch.FloatTensor:
 self_c_fc_0     ->  0     hidden_states = self.c_fc(hidden_states)
 hidden_states_0 ->  +     ...
 self_act_0      ->  1     hidden_states = self.act(hidden_states)
 hidden_states_1 ->  +     ...
 self_c_proj_0   ->  2     hidden_states = self.c_proj(hidden_states)
 hidden_states_2 ->  +     ...
 self_dropout_0  ->  3     hidden_states = self.dropout(hidden_states)
 hidden_states_3 ->  +     ...
                     4     return hidden_states
```

Names are the dotted callee joined by `_`, plus an occurrence index in execution
order: `self.act(...)` → `self_act_0`, `torch.relu(...)` → `torch_relu_0`, a second
`relu` → `torch_relu_1`. Every assignment is an operation too, named after its
target: `hidden_states = self.act(hidden_states)` is `hidden_states_1`, and its
`.output` is the assigned value. Calls and assignments share one counter per name,
so where a forward binds a name and then calls it, the binding is `_0` and the call
`_1` — GPT-2's attention call is `attention_interface_1`. **Do not guess these
names** — print the source. They change with the `transformers` version.

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

## Values that are not a call's result

A product, a slice, a state carried through a loop — anything a forward *assigns*
but never returns from a call — is reachable by the name the forward gives it.
Inside GPT-2's eager attention, the pre-softmax scores are `attn_weights_0`
(`q @ k^T * scale`) and the probabilities `attn_weights_2` (after the mask and
softmax):

```python
with model.trace(prompt):
    inner = model.transformer.h[0].attn.source.attention_interface_1.source
    scores = inner.attn_weights_0.output.save()
    probs = inner.attn_weights_2.output.save()

print(scores.shape, probs[0, 0].sum(-1))   # scores are pre-softmax; probability rows sum to 1
```

An assignment inside a Python loop is one location that fires once per iteration,
so `tracer.iter[k]` selects the k-th iteration; the loop's bound is usually a call
in the same source (`range_0`, `nonzero_0`), so `len(op.output)` gives the count
in-trace.

## The main use: attention patterns

The attention module returns the value-weighted result, not the probability
matrix. The probabilities come from the attention-interface call:

```python
with model.trace(prompt):
    # Bind the save itself — unpacking inside the block would name the elements,
    # and a save comes back keyed by the name bound to the saved object.
    attn = model.transformer.h[0].attn.source.attention_interface_1.output.save()

attn_out, attn_weights = attn
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
        _, weights = block.attn.source.attention_interface_1.output
        patterns.append(weights)

print(len(patterns), patterns[0].shape)
```

A quick induction-ish diagnostic — how much each head attends to the previous
token:

```python
with model.trace(prompt):
    _, weights = model.transformer.h[5].attn.source.attention_interface_1.output
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
             .source.attention_interface_1
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
        _, weights = model.transformer.h[0].attn.source.attention_interface_1.output
        per_step.append(weights.shape[-1])

print(per_step)      # key length grows as generation proceeds
```

## Limits

| Limit | What to do |
|---|---|
| The op calls a **submodule** | Refused (`SourceNotAvailable`) — use that submodule's own `.source` |
| The op calls a **builtin / C function** | No Python source exists; not drillable |
| The `forward` is **decorated** | Instrumented: a wrapper that calls the function is peeled and rebuilt; a dispatcher that picks an implementation at run time shows the dispatch — drill into that op for what ran |
| The op is an **assignment** | No callee — `.source` on it raises `SourceNotAvailable` |
| Recursive `.source` outside a trace | Raises; open a trace first |
| Ops requested out of execution order | `OutOfOrderError`, same as modules |

Prefer a real submodule when one exposes the value (`mlp.output` beats
`source.self_c_proj_0.output`) — it is cheaper and stable across library versions.

## Related

- [access-and-modify.md](access-and-modify.md)
- [modules-and-architectures.md](modules-and-architectures.md) — per-head slicing without `.source`
- [execution-model.md](execution-model.md)
