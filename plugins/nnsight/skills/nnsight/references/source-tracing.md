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
`_1` — GPT-2's attention call is `attention_interface_1`.

**Do not guess these names** — print the source and read the label off the line you
want. Every label in this file was read off `transformers` 5.15, and a version that
renames an internal variable renames its operation. An assignment's `.output` is
whatever the right-hand side produced, which need not be a tensor:
`attention_interface_0` binds the *implementation function*, `hidden_shape_0` a
shape tuple. Saving one of those succeeds and hands you the object, so check what
you got rather than assuming a tensor.

## Reading and replacing an operation

```python
with model.trace(prompt):
    post_gelu = model.transformer.h[0].mlp.source.self_act_0.output.save()
    model.transformer.h[0].mlp.source.self_c_proj_0.output[:] = 0    # ablate the write
    logits = model.output.logits.save()

# GPT-2's MLP is 4x hidden, and this is the value after the GELU.
assert post_gelu.shape[-1] == 4 * model.config.n_embd
assert (post_gelu > 0).any() and (post_gelu < 0).any()
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

assert scores.shape == probs.shape                       # [batch, heads, q, k]
assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-4)
assert not torch.allclose(scores.sum(-1), torch.ones_like(scores.sum(-1)))  # pre-softmax
print(scores.shape, probs[0, 0].sum(-1))
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
n_tokens = len(model.tokenizer(prompt)["input_ids"])
assert attn_weights.shape == (1, model.config.n_head, n_tokens, n_tokens)
assert torch.allclose(attn_weights.sum(-1), torch.ones_like(attn_weights.sum(-1)), atol=1e-4)
print(attn_weights.shape)            # [batch, heads, q_seq, k_seq]
```

**You must load the model with `attn_implementation="eager"`.** The default `sdpa`
kernel never materializes the probability matrix and returns `None` for the
weights.

All layers in one pass. Instrument every attention *before* opening the trace: the
first `.source` on a module has to land before that module's forward runs, and inside
a sweep it would not for any layer after the first read. A bare attribute access does
it, with no forward pass:

```python
for block in model.transformer.h:
    _ = block.attn.source                  # instrument now, once per module

with model.trace(prompt):
    patterns = nnsight.save([])
    for block in model.transformer.h:
        _, weights = block.attn.source.attention_interface_1.output
        patterns.append(weights)

assert len(patterns) == model.config.n_layer
assert all(p.shape == patterns[0].shape for p in patterns)
assert not torch.equal(patterns[0], patterns[-1])     # real per-layer patterns
print(len(patterns), patterns[0].shape)
```

A quick induction-ish diagnostic — how much each head attends to the previous
token:

```python
with model.trace(prompt):
    _, weights = model.transformer.h[5].attn.source.attention_interface_1.output
    prev_token_score = weights.diagonal(offset=-1, dim1=-2, dim2=-1).mean(-1).save()

assert prev_token_score.shape == (1, model.config.n_head)
assert ((prev_token_score >= 0) & (prev_token_score <= 1)).all()
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

# The per-head attention output, transposed back to [batch, seq, heads, head_dim].
assert inner.shape[-1] == model.config.n_embd // model.config.n_head
assert inner.shape[-2] == model.config.n_head
print(inner.shape)
```

Ask for the drill **before** the operation's own `.output`: `op.source` is served
just as the call is about to run, one step earlier than the value it returns.

A drill needs Python source at the other end, and one common case has none — a
function that refers to its own name in its body, which every
`torch.nn.functional` entry point does to reach the torch-function dispatcher. So
`nn_functional_softmax_0.source` raises `KeyError: 'softmax'`. Its `.output` works,
and for softmax that is the tensor you were after anyway.

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

assert len(per_step) == 3
assert per_step == sorted(per_step) and per_step[0] < per_step[-1]   # key length grows
print(per_step)
```

## Limits

| Limit | What to do |
|---|---|
| The op calls a **submodule** | Refused (`SourceNotAvailable`) — use that submodule's own `.source` |
| The op calls a **builtin / C function** | No Python source exists; not drillable |
| The op calls a function that **names itself** (`F.softmax` and friends) | Not drillable — `KeyError` on the function's own name. Read its `.output` instead |
| The `forward` is **decorated** | Instrumented: a wrapper that calls the function is peeled and rebuilt; a dispatcher that picks an implementation at run time shows the dispatch — drill into that op for what ran |
| The op is an **assignment** | No callee — `.source` on it raises `SourceNotAvailable` |
| Recursive `.source` outside a trace | Raises; open a trace first |
| Ops requested out of execution order | `OutOfOrderError`, same as modules |
| **First** `.source` on a module, after something else in the block was read | `OutOfOrderError`. Instrumenting rewrites the forward, so it must happen before that forward runs — do `_ = module.source` outside the trace, once per module |
| `op.source` asked for after `op.output` on the same op | `OutOfOrderError` on `...{op}.fn` — the drill is served one step earlier; ask for it first |
| The op is on a **branch this config never takes** | `OutOfOrderError`, worded as though you were late. See below |

## Operations on branches that never run

`print(module.source)` and `.names` list every operation in the `forward`, including
the ones under an `if` this model's config makes false. Asking for one parks a
request at a location the model never reaches, and the trace fails with the *ordering*
error — so the message points at a problem you do not have.

GPT-2's attention lists 50 operations and runs 28. The other 22 are the
cross-attention and cache-hit paths, and their labels sit right beside the live ones:
`transpose_0` is the cross-attention key transpose, `transpose_2` the key transpose
GPT-2 actually performs.

<!-- test: expect-error OutOfOrderError -->
```python
with model.trace(prompt):
    dead = model.transformer.h[0].attn.source.transpose_0.output.save()
# OutOfOrderError: '...attn.source.transpose_0.output.i0' was requested but the
#                  model already ran past it
```

```python
with model.trace(prompt):
    key = model.transformer.h[0].attn.source.transpose_2.output.save()

head_dim = model.config.n_embd // model.config.n_head
assert key.shape == (1, model.config.n_head, len(model.tokenizer(prompt)["input_ids"]), head_dim)
print(key.shape)     # [batch, heads, seq, head_dim]
```

When an operation you can see in the listing reports as run past, before suspecting
your request order, read the listing around it: an operation under a false condition
(`is_cross_attention`, an `is_updated` cache hit, `use_parallel_residual`) is dead,
and there is usually a live twin further down with the next occurrence index.
Requesting one operation per trace tells you which labels answer.

Prefer a real submodule when one exposes the value (`mlp.output` beats
`source.self_c_proj_0.output`) — it is cheaper and stable across library versions.

## Related

- [access-and-modify.md](access-and-modify.md)
- [modules-and-architectures.md](modules-and-architectures.md) — per-head slicing without `.source`
- [execution-model.md](execution-model.md)
