---
name: attention-analysis
description: Extract and analyze attention patterns and per-head behavior — attention probability matrices via .source, per-head metrics (entropy, previous-token, attention-sink, induction), automatic head-type detection, per-head reading and editing of the output. Use to find out what a head attends to, to identify induction/copy/previous-token heads, to visualize where information moves, or to pick candidate heads before patching or ablating them. Requires attn_implementation="eager"; covers grouped-query attention and the head-slicing convention.
---

# Attention Analysis

Two different objects get called "attention", and you need both:

- **The pattern** — `softmax(QK^T/√d)`, a `[batch, heads, query, key]` matrix
  saying *where each head looks*. Under `eager` the attention module hands it
  back as `attn.output[1]`; `.source` reaches the raw softmax inside.
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
head_dim = model.transformer.h[0].attn._module.head_dim   # not n_embd // n_heads
tokens = [model.tokenizer.decode([i]) for i in model.tokenizer(prompt).input_ids]
```

## Getting patterns

```python
with torch.no_grad():                     # a trace runs with autograd on
    with model.trace(prompt):
        first_layer = model.transformer.h[0].attn.output[1].save()

print(first_layer.shape)              # [batch, heads, query, key]
print(first_layer[0, 0].sum(-1))      # each row sums to 1
```

If the weights are `None`, the model was not loaded with
`attn_implementation="eager"` — `sdpa` and `flash_attention_2` never materialize
the matrix, and `model.output.attentions` is the empty tuple rather than an error.

**Anchor before you score.** Two properties are cheap and both must hold, or you
are holding something that is not a probability matrix:

```python
seq = first_layer.shape[-1]
upper = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)

assert torch.allclose(first_layer.sum(-1), torch.ones_like(first_layer.sum(-1)))
assert (first_layer[..., upper] == 0).all()      # causal mask: exactly zero, not small
```

Exactly zero rather than merely small is the tell: the eager path adds `-inf`
before the softmax. A `1e-5` up there means pre-softmax scores, or a
bidirectional model.

All layers in one pass — `model.output.attentions` needs
`output_attentions=True` and gives every layer; the loop lets you pick:

```python
with torch.no_grad():
    with model.trace(prompt):
        patterns = nnsight.save([])
        for block in model.transformer.h:
            patterns.append(block.attn.output[1])

print(len(patterns), patterns[0].shape)
```

The raw `softmax(QK^T/√d)`, before the dtype cast and dropout, is one level in:
`block.attn.source.attention_interface_1.source.nn_functional_softmax_0.output`.
On a float32 model it is the same tensor as `attn.output[1]`; on bf16 the two
differ by the cast (`2e-3` on Qwen2.5-0.5B). Those operation names are
`transformers` 5.15 — confirm with `print(model.transformer.h[0].attn.source)`,
and note that the listing includes operations on branches this model never runs
(22 of GPT-2 attention's 50), which raise `OutOfOrderError` if you ask for them.
See the `nnsight` skill → source tracing.

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
    to_first = pattern[..., 1:, 0].mean(-1)              # attention sink / BOS
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
  Score it over queries `q >= 1` only: `A[0, 0]` is 1 by construction, so
  including it inflates every head — over GPT-2's 144 heads on this 14-token
  prompt the mean is 0.642 with `q = 0` and 0.615 without, and the gap grows as
  `1/seq`.

## Finding induction heads

Induction heads implement "I saw `AB` earlier, now I see `A`, so predict `B`". The
clean detector is a **repeated random sequence**: with real text a head can score
well by using semantics, but random tokens can only be matched by position.

```python
generator = torch.Generator().manual_seed(0)
length = 20
bos = model.config.eos_token_id                  # gpt2 has no dedicated BOS

random_seq = torch.randint(1000, 10000, (length,), generator=generator)
repeated = torch.cat([torch.tensor([bos]), random_seq, random_seq]).unsqueeze(0)

# control: same length, same distribution, no repeat
flat = torch.randint(1000, 10000, (2 * length,), generator=generator)
control_seq = torch.cat([torch.tensor([bos]), flat]).unsqueeze(0)

def induction_scores(sequence):
    with torch.no_grad():
        with model.trace(sequence.to(model.device)):
            captured = nnsight.save([b.attn.output[1] for b in model.transformer.h])

    # in the second copy, position 1+length+i attends to position 1+i+1
    query_positions = torch.arange(length, 2 * length - 1) + 1
    key_positions = query_positions - length + 1
    return {
        (layer, head): float(w[0, head, query_positions, key_positions].mean())
        for layer, w in enumerate(captured)
        for head in range(n_heads)
    }

scores = induction_scores(repeated)
control = induction_scores(control_seq)

top = sorted(scores.items(), key=lambda kv: -kv[1])[:5]
for (layer, head), score in top:
    print(f"L{layer}H{head:<2} induction {score:.3f}   control {control[(layer, head)]:.3f}")

assert {head for head, _ in top} == {(5, 1), (5, 5), (6, 9), (7, 2), (7, 10)}
assert max(control[head] for head, _ in top) < 0.01
```

```
L5H5  induction 0.884   control 0.003
L7H10 induction 0.860   control 0.004
L5H1  induction 0.839   control 0.000
L6H9  induction 0.815   control 0.001
L7H2  induction 0.753   control 0.004
```

The **control column is what turns the number into a claim.** The same heads, the
same positions, on a sequence that does not repeat, score at nothing — so the
behavior is caused by the repetition and not by the position. A detector without
one measures nothing.

Those five are GPT-2 small's induction heads as ARENA's induction sweep reports
them (5.1, 5.5, 6.9, 7.2, 7.10), recovered from scratch. Use the same shape of
detector for any hypothesis: construct an input where only the behavior you care
about can produce a high score. A different published list circulates — the IOI
paper's 5.5, 5.8, 5.9, 6.9 — because it scores behavior on natural prompts rather
than on repeated random tokens. A head is an induction head relative to a probe;
report the probe with the score.

## Where a head looks, in text

For a specific head, print the strongest source position for each destination:

```python
LAYER, HEAD = 5, 5

with torch.no_grad():
    with model.trace(prompt):
        head_pattern = model.transformer.h[LAYER].attn.output[1][0, HEAD].save()

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

with torch.no_grad():
    with model.trace(prompt):
        proj_in = model.transformer.h[LAYER].attn.c_proj.input.save()

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
        per_step.append(model.transformer.h[5].attn.output[1].shape[-1])

print("key length per step:", per_step)
```

## Other architectures

The recipe is the same; the names are not.

- The module path changes (`model.model.layers[i].self_attn`), the pattern handle
  does not: `self_attn.output[1]` under eager, on every family tested. Source
  operation names differ by model class and `transformers` version — always
  `print(block.attn.source)` first.
- **Grouped-query attention** (Llama-3, Qwen, Gemma): `num_key_value_heads` is
  smaller than `num_attention_heads`, so several query heads share one KV head.
  The pattern still has one row per *query* head, and per-head slicing of the
  output projection's input still works unchanged —
  `o_proj.input.view(B, S, num_attention_heads, head_dim)` is `torch.equal` to the
  attention implementation's per-head output on Qwen2.5-0.5B (14 query / 2 KV) and
  gemma-2-2b (8 / 4). Only slices of `k_proj` / `v_proj` are indexed by
  `num_key_value_heads`. Check both counts — `scripts/inspect_model.py` in the
  `nnsight` skill prints them.
- **`head_dim` is not `hidden_size // num_attention_heads`.** Read
  `attn._module.head_dim`. On gemma-2-2b the arithmetic gives 288 and the true
  value is 256, and the wrong `.view()` succeeds.
- Some models have no accessible probability matrix at all under their default
  attention kernel; `attn_implementation="eager"` is required everywhere.
- **Gemma-2's eager attention is not plain softmax attention.** It softcaps the
  attention logits (`attn_logit_softcapping = 50.0`) between the matmul and the
  softmax, visible as `torch_tanh_0` in the implementation's operation list.
- `nnterp`'s `StandardizedTransformer(..., enable_attention_probs=True)` gives
  `model.attention_probabilities[i]` with no per-architecture names at all — see
  the `nnterp` skill.

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
    by_projection = model.output.logits[0, -1].save()

# or by the source op, if you also want to read the per-head tensor
with model.trace(prompt):
    op = model.transformer.h[LAYER].attn.source.attention_interface_1
    per = op.output[0].clone()
    per[:, :, HEAD, :] = 0
    op.output = (per,) + tuple(op.output[1:])
    by_source = model.output.logits[0, -1].save()

assert torch.equal(by_projection, by_source)
```

For a `.heads` accessor you can read and write like any other activation, see
`nnsight/docs/patterns/per-head-attention.md` — an `eproperty` on `c_proj`'s
input reproduces the ablation exactly, and one on the attention module's output
does not.

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
