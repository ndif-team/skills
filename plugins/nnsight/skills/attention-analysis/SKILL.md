---
name: attention-analysis
description: Extract and analyze attention patterns and per-head behavior — attention probability matrices via .source, per-head metrics (entropy, previous-token, attention-sink, induction), automatic head-type detection, per-head reading and editing of the output. Use to find out what a head attends to, to identify induction/copy/previous-token heads, to visualize where information moves, or to pick candidate heads before patching or ablating them. Requires attn_implementation="eager"; covers grouped-query attention and the head-slicing convention.
---

# Attention Analysis

Two different objects get called "attention", and you need both:

- **The pattern** — `softmax(QK^T/√d)`, a `[batch, heads, query, key]` matrix
  saying *where each head looks*. Not returned by the attention module; reach it
  with `.source`.
- **The per-head output** — what each head *writes* into the residual stream. A
  slice of the output projection's input.

Patterns tell you about routing; outputs tell you about content. A head with a
striking pattern that writes nothing is not doing anything.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

# eager is required — sdpa/flash never materialize the probability matrix
model = TransformersModel("openai-community/gpt2", dispatch=True,
                          attn_implementation="eager")

prompt = "When Mary and John went to the store, John gave a drink to"
n_layers = len(model.transformer.h)
n_heads = model.config.n_head
head_dim = model.config.n_embd // n_heads
tokens = [model.tokenizer.decode([i]) for i in model.tokenizer(prompt).input_ids]
```

## Getting patterns

```python
with model.trace(prompt):
    attn_out, weights = model.transformer.h[0].attn.source.attention_interface_0.output
    first_layer = weights.detach().save()

print(first_layer.shape)              # [batch, heads, query, key]
print(first_layer[0, 0].sum(-1))      # each row sums to 1
```

If `weights` is `None`, the model was not loaded with `attn_implementation="eager"`.
The operation name (`attention_interface_0`) is version-dependent — confirm with
`print(model.transformer.h[0].attn.source)` rather than assuming. See the `nnsight`
skill → source tracing.

All layers in one pass:

```python
with model.trace(prompt):
    patterns = nnsight.save([])
    for block in model.transformer.h:
        _, weights = block.attn.source.attention_interface_0.output
        patterns.append(weights.detach())

print(len(patterns), patterns[0].shape)
```

## Per-head metrics

Reduce each `[query, key]` matrix to one number per head, then rank. These four
cover most of what people look for:

```python
def head_metrics(pattern):
    """pattern: [heads, query, key] for one layer."""
    heads = pattern.shape[0]
    seq = pattern.shape[-1]
    eps = 1e-9
    entropy = -(pattern * (pattern + eps).log()).sum(-1).mean(-1)
    previous = pattern.diagonal(offset=-1, dim1=-2, dim2=-1).mean(-1)
    self_attn = pattern.diagonal(dim1=-2, dim2=-1).mean(-1)
    to_first = pattern[..., 0].mean(-1)                  # attention sink / BOS
    return entropy, previous, self_attn, to_first

print(f"{'head':<8}{'entropy':>9}{'prev-tok':>10}{'self':>8}{'->pos0':>9}")
for layer in [0, 5, 11]:
    entropy, previous, self_attn, to_first = head_metrics(patterns[layer][0])
    for head in range(0, n_heads, 4):
        print(f"L{layer}H{head:<5}{entropy[head]:>9.2f}{previous[head]:>10.3f}"
              f"{self_attn[head]:>8.3f}{to_first[head]:>9.3f}")
```

Reading them:

- **Low entropy** — the head attends to one or two positions; likely doing
  something specific. High entropy is diffuse/uniform, often uninformative.
- **High previous-token score** — a previous-token head, the first half of an
  induction circuit.
- **High attention to position 0** — an attention sink, the "do nothing" state.
  Very common and rarely meaningful; exclude it before ranking heads by anything.

## Finding induction heads

Induction heads implement "I saw `AB` earlier, now I see `A`, so predict `B`". The
clean detector is a **repeated random sequence**: with real text a head can score
well by using semantics, but random tokens can only be matched by position.

```python
generator = torch.Generator().manual_seed(0)
length = 20
random_seq = torch.randint(1000, 10000, (length,), generator=generator)
repeated = torch.cat([random_seq, random_seq]).unsqueeze(0).to(model.device)

with model.trace(repeated):
    repeat_patterns = nnsight.save([])
    for block in model.transformer.h:
        _, weights = block.attn.source.attention_interface_0.output
        repeat_patterns.append(weights.detach())

# In the second copy, position length+i should attend to position i+1
query_positions = torch.arange(length, 2 * length - 1)
key_positions = query_positions - length + 1

scores = {}
for layer, weights in enumerate(repeat_patterns):
    per_head = weights[0, :, query_positions, key_positions].mean(-1)
    for head in range(n_heads):
        scores[(layer, head)] = float(per_head[head])

for (layer, head), score in sorted(scores.items(), key=lambda kv: -kv[1])[:5]:
    print(f"L{layer}H{head:<2} induction score {score:.3f}")
```

```
L7H10 induction score 0.902
L5H5  induction score 0.901
L5H1  induction score 0.880
L6H9  induction score 0.833
L7H2  induction score 0.801
```

Those are GPT-2 small's known induction heads, recovered from scratch. Use the
same shape of detector for any hypothesis: construct an input where only the
behavior you care about can produce a high score.

## Where a head looks, in text

For a specific head, print the strongest source position for each destination:

```python
LAYER, HEAD = 5, 5

with model.trace(prompt):
    _, weights = model.transformer.h[LAYER].attn.source.attention_interface_0.output
    head_pattern = weights[0, HEAD].detach().save()

for position, token in enumerate(tokens):
    best = head_pattern[position].argmax().item()
    weight = head_pattern[position, best].item()
    print(f"{token!r:<10} -> {tokens[best]!r:<10} ({weight:.2f})")
```

## What a head writes

The pattern says where a head looks; this says what it contributes. Head `h`
occupies columns `[h*head_dim : (h+1)*head_dim]` of the output projection's input:

```python
LAYER = 9

with model.trace(prompt):
    proj_in = model.transformer.h[LAYER].attn.c_proj.input.detach().save()

norms = torch.stack([
    proj_in[0, -1, head * head_dim:(head + 1) * head_dim].norm()
    for head in range(n_heads)
])
for head in norms.argsort(descending=True)[:4]:
    print(f"L{LAYER}H{int(head):<2} writes norm {norms[head]:.3f} at the last position")
```

Ablate or patch those same slices to test whether the writing matters — see the
`ablation` and `activation-patching` skills.

## Across generation

Patterns change every step as the sequence grows:

```python
with model.generate(prompt, max_new_tokens=3) as tracer:
    per_step = nnsight.save([])
    for step in tracer.iter[:3]:
        _, weights = model.transformer.h[5].attn.source.attention_interface_0.output
        per_step.append(weights.shape[-1])

print("key length per step:", per_step)
```

## Other architectures

The recipe is the same; the names are not.

- The `.source` operation name differs by model class and `transformers` version —
  always `print(block.attn.source)` first.
- **Grouped-query attention** (Llama-3, Qwen, Gemma): `num_key_value_heads` is
  smaller than `num_attention_heads`, so several query heads share one KV head.
  The pattern still has one row per *query* head, but head slices of the KV
  projections do not line up one-to-one. Check both counts —
  `scripts/inspect_model.py` in the `nnsight` skill prints them.
- Some models have no accessible probability matrix at all under their default
  attention kernel; `attn_implementation="eager"` is required everywhere.

## Cautions

**A pattern is not an explanation.** Attention weight is not information flow —
a head can attend strongly to a position and write nothing useful from it. Confirm
with ablation or patching before claiming a head "does" something.

**Exclude the sink before ranking.** Position-0 attention dominates many heads and
will sort your table for you if you let it.

**Average over inputs.** Head behavior is input-dependent; a single prompt gives a
single anecdote. Score over a set of prompts and report the distribution.

**One layer's pattern hides composition.** Induction needs a previous-token head
in an earlier layer feeding a matching head later. Interpreting either alone
misses the circuit — see the `circuit-discovery` skill.

## Ablating a head you found

Cut the head **before** `c_proj` — after the projection the hidden dimension no
longer decomposes per head, so slicing `attn.output` is not head ablation (measured:
21x smaller effect, no error). Either of these is correct and they agree exactly:

```python
# by the projection's input
with model.trace(prompt):
    lo, hi = HEAD * head_dim, (HEAD + 1) * head_dim
    model.transformer.h[LAYER].attn.c_proj.input[:, :, lo:hi] = 0

# or by the source op, if you also want to read the per-head tensor
with model.trace(prompt):
    op = model.transformer.h[LAYER].attn.source.attention_interface_0
    per = op.output[0].clone()
    per[:, :, HEAD, :] = 0
    op.output = (per,) + tuple(op.output[1:])
```

Establish a null distribution before believing the number. On GPT-2 at a middle
layer, ablating the head that attends *most* to a token is often indistinguishable
from ablating one that attends to it with weight ~0.0006 — per-head zero-ablation
deltas there are dominated by generic off-distribution shock. Sweep every head at the
layer, and prefer resample ablation (patch in a donor prompt's head) over zeroing.
See `nnsight/docs/patterns/per-head-attention.md` and the `ablation` skill.

## Related skills

- `nnsight` — `.source`, batching, module paths
- `ablation` — testing whether a head's output matters
- `activation-patching` — per-head causal attribution
- `circuit-discovery` — composing heads into circuits
