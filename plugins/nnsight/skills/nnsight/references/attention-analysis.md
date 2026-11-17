# Attention analysis

Two things are called attention. The **pattern** is the softmax over keys, a
`(batch, heads, query, key)` matrix that says where each head looks. A head's
**write** is what it adds to the residual stream. A pattern is routing, not
content: a head can look at a position and write nothing useful from it, so a
claim about what a head does needs an [ablation](ablation.md) or a
[patch](activation-patching.md) as well.

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True,
                          attn_implementation="eager")

prompt = "When Mary and John went to the store, John gave a drink to"
tokens = [model.tokenizer.decode(i) for i in model.tokenizer(prompt).input_ids]
n_layers = len(model.transformer.h)
n_heads = model.config.n_head
```

## Patterns

**Load with `attn_implementation="eager"`.** Under the default implementation
(`sdpa`) the probability matrix is never built: `attn.output[1]` is `None`, and
`model.output.attentions` is `None`, or `()` with `output_attentions=True`. None
of them raises.

```python
with torch.no_grad(), model.trace(prompt):
    patterns = nnsight.save([block.attn.output[1] for block in model.transformer.h])

assert len(patterns) == n_layers
assert tuple(patterns[0].shape) == (1, n_heads, 14, 14)                 # (batch, heads, query, key)
```

Check that what you hold is a probability matrix before scoring it:

```python
first_layer = patterns[0]
later = torch.triu(torch.ones(14, 14, dtype=torch.bool), diagonal=1)   # key after query
assert torch.allclose(first_layer.sum(-1), torch.ones_like(first_layer.sum(-1)))
assert (first_layer[..., later] == 0).all()                            # causal mask: exactly zero
```

The scores before the softmax, and every other value inside the attention
forward, are read with `.source`: [source-tracing](source-tracing.md).

## Scoring heads

Reduce each head's `(query, key)` matrix to a number, then rank:

```python
P = torch.cat(patterns)                                        # (layers, heads, query, key) for one prompt
entropy = -(P * (P + 1e-9).log()).sum(-1).mean(-1)             # low: the head looks at one or two positions
previous = P.diagonal(offset=-1, dim1=-2, dim2=-1).mean(-1)    # weight on the token before the query
current = P.diagonal(dim1=-2, dim2=-1).mean(-1)                # weight on the query's own token
first = P[..., 1:, 0].mean(-1)                                 # weight on position 0, from queries 1 and up

def best(score, k=3, largest=True):
    values, flat = score.flatten().topk(k, largest=largest)
    return "   ".join(f"L{i // n_heads}H{i % n_heads} {v:.2f}" for v, i in zip(values.tolist(), flat.tolist()))

print("previous token ", best(previous))
print("own token      ", best(current))
print("position 0     ", best(first))
print("lowest entropy ", best(entropy, largest=False))

resting = first > 0.5
assert int(resting.sum()) == 100                               # of 144 heads
assert int(resting.flatten()[entropy.flatten().topk(10, largest=False).indices].sum()) == 9
```

```
previous token  L4H11 1.00   L2H2 0.54   L3H2 0.46
own token       L0H1 0.90   L0H3 0.82   L0H5 0.73
position 0      L5H1 1.00   L7H2 1.00   L6H9 0.99
lowest entropy  L4H11 0.00   L5H1 0.01   L7H2 0.03
```

**Position 0 is where a head puts its weight when it has nothing to match.** On
this prompt 100 of 144 heads put more than half of it there, and they are nine
of the ten lowest-entropy heads. Leave them out (`score[~resting]`) before
ranking by anything else. Score position 0 from queries 1 and up, because query
0 can only attend to itself.

A score from one prompt describes that prompt. L5H1, L7H2 and L6H9 look idle
here and are three of the five strongest induction heads below.

## Finding a head type: induction

Build an input on which only the behavior under test can score, and a control
without it. A random token sequence repeated twice has no meaning to use, so a
head that attends from a token in the second copy to the token after its first
occurrence is matching by content:

```python
generator = torch.Generator().manual_seed(0)
length = 20
start = torch.tensor([model.tokenizer.bos_token_id])
half = torch.randint(1000, 10000, (length,), generator=generator)
repeated = torch.cat([start, half, half]).unsqueeze(0)
control = torch.cat([start, torch.randint(1000, 10000, (2 * length,), generator=generator)]).unsqueeze(0)

def induction_scores(sequence):
    with torch.no_grad(), model.trace(sequence.to(model.device)):
        layers = nnsight.save([block.attn.output[1] for block in model.transformer.h])
    query = torch.arange(length, 2 * length - 1) + 1           # positions in the second copy
    key = query - length + 1                                   # the token after the first occurrence
    return torch.stack([layer[0, :, query, key].mean(-1) for layer in layers])   # (layers, heads)

scores, null = induction_scores(repeated), induction_scores(control)
top = scores.flatten().topk(5).indices.tolist()
for i in top:
    print(f"L{i // n_heads}H{i % n_heads:<3} repeated {scores.flatten()[i]:.3f}   control {null.flatten()[i]:.3f}")

assert {(i // n_heads, i % n_heads) for i in top} == {(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)}
assert float(null.flatten()[top].max()) < 0.01
```

```
L5H5   repeated 0.884   control 0.003
L7H10  repeated 0.860   control 0.004
L5H1   repeated 0.839   control 0.000
L6H9   repeated 0.815   control 0.001
L7H2   repeated 0.753   control 0.004
```

The control column makes the score a claim: the same heads at the same
positions score nothing on a sequence that does not repeat. Report it beside
every detector score.

## One head, in text

```python
head = patterns[5][0, 5]                                       # L5H5: (query, key)
for query, token in enumerate(tokens):
    key = int(head[query].argmax())
    if key != 0:                                               # every other query rests on position 0
        print(f"{token!r} at {query} -> {tokens[key]!r} at {key}  {float(head[query, key]):.2f}")

assert tokens[int(head[9].argmax())] == " went"
```

```
' John' at 9 -> ' went' at 4  0.82
```

The second ` John` looks at the token that followed the first one.

## What a head writes

The attention layer's output is the sum of one write per head. Head `h` is
columns `h * head_dim` to `(h + 1) * head_dim` of the output projection's input,
and its write is that slice through its rows of the projection
([access-and-modify](access-and-modify.md) has the convention):

```python
attn = model.transformer.h[9].attn
head_dim = attn._module.head_dim

with torch.no_grad(), model.trace(prompt):
    z = attn.c_proj.input.save()                               # (batch, tokens, heads * head_dim)
    layer_write = attn.output[0].save()                        # (batch, tokens, hidden), after c_proj

with torch.no_grad():
    weight, bias = attn.c_proj.weight, attn.c_proj.bias        # GPT-2's Conv1D weight is (in, out)
    writes = torch.stack([z[..., h * head_dim:(h + 1) * head_dim] @ weight[h * head_dim:(h + 1) * head_dim]
                          for h in range(n_heads)])            # (heads, batch, tokens, hidden)
    assert torch.allclose(writes.sum(0) + bias, layer_write, atol=1e-4)
    norms = writes[:, 0, -1].norm(dim=-1)

print("largest writes at the last token:",
      "   ".join(f"L9H{h} {norms[h]:.1f}" for h in norms.argsort(descending=True)[:3].tolist()))
```

```
largest writes at the last token: L9H6 22.4   L9H9 19.4   L9H8 14.9
```

To test whether a write matters, replace the head's slice of `c_proj.input`:
[ablation](ablation.md) sweeps every head in one trace, and
[circuit-discovery](circuit-discovery.md) follows a write to the components
that read it.

## During generation

After the prefill, a step computes one query row, over every token so far:

```python
with torch.no_grad(), model.generate(prompt, max_new_tokens=3, min_new_tokens=3, do_sample=False) as tracer:
    shapes = nnsight.save([])
    for step in tracer.iter[:3]:
        shapes.append(tuple(model.transformer.h[5].attn.output[1].shape))

assert shapes == [(1, 12, 14, 14), (1, 12, 1, 15), (1, 12, 1, 16)]
```

## Other architectures

- On Llama, Qwen and Gemma-2 the module is `model.model.layers[i].self_attn`,
  the pattern is `self_attn.output[1]` under eager, and the output projection is
  `self_attn.o_proj`.
- **`o_proj` is an `nn.Linear`, whose weight is `(out, in)`.** A head's write is
  `z[..., lo:hi] @ o_proj.weight[:, lo:hi].T`. The GPT-2 form `weight[lo:hi]`
  runs without error where the projection is square and is not the write.
- **Grouped-query attention**: several query heads share one key and value head
  (14 and 2 on `Qwen/Qwen2.5-0.5B`; `scripts/inspect_model.py` prints both
  counts). The pattern and `o_proj.input` still have one slot per query head.
  Only the `k_proj` and `v_proj` outputs are indexed by key-value head, and
  query head `h` reads key-value head
  `h // (num_attention_heads // num_key_value_heads)`.
- **Gemma-2 caps the attention scores** with a tanh
  (`config.attn_logit_softcapping`, 50.0 on `google/gemma-2-2b`) before the
  softmax, so a pattern recomputed by hand from queries and keys is not the
  model's.
- nnterp's `StandardizedTransformer(..., attn_implementation="eager")` serves the
  pattern as `model.layers[i].self_attn.attention_probabilities` under one name
  on every family it supports; see the `nnterp` skill.
