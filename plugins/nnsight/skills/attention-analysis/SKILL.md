---
name: attention-analysis
description: Extract and analyze attention patterns and per-head behavior — attention probability matrices from attn.output[1] or .source, the anchors that check they really are probabilities, per-head metrics (entropy, previous-token, attention-sink, induction), automatic head-type detection, and what each head writes. Use to find out what a head attends to, to identify induction/copy/previous-token heads, to visualize where information moves, or to pick candidate heads before patching or ablating them (the sweep itself lives in the ablation skill). Requires attn_implementation="eager"; covers grouped-query attention and the head-slicing convention.
---

# Attention Analysis

Two different objects get called "attention", and you need both:

- **The pattern** — `softmax(QK^T/√d)`, a `[batch, heads, query, key]` matrix
  saying *where each head looks*. Under `eager` the attention module hands it
  back as `attn.output[1]`; `.source` reaches the raw softmax inside.
- **The per-head output** — what each head *writes* into the residual stream. A
  slice of the output projection's input.

Patterns tell you about routing; outputs tell you about content.

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

Every layer in one pass (`model.output.attentions` also works, but only with
`output_attentions=True`; the loop lets you pick layers):

```python
with torch.no_grad():                     # a trace runs with autograd on
    with model.trace(prompt):
        patterns = nnsight.save([block.attn.output[1] for block in model.transformer.h])

print(len(patterns), patterns[0].shape)   # 12 x [batch, heads, query, key]
```

If the weights are `None`, the model was not loaded with
`attn_implementation="eager"` — `sdpa` and `flash_attention_2` never materialize
the matrix, and `model.output.attentions` is the empty tuple rather than an error.

**Anchor before you score.** Two properties are cheap and both must hold, or you
are holding something that is not a probability matrix:

```python
first_layer = patterns[0]
seq = first_layer.shape[-1]
upper = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)

assert torch.allclose(first_layer.sum(-1), torch.ones_like(first_layer.sum(-1)))
assert (first_layer[..., upper] == 0).all()      # causal mask: exactly zero, not small
```

The raw `softmax(QK^T/√d)`, before the dtype cast and dropout, is one level in:
`block.attn.source.attention_interface_1.source.nn_functional_softmax_0.output`.
Those operation names are `transformers` 5.15 — print
`model.transformer.h[0].attn.source` first; the listing includes operations on
branches this model never runs, which raise `OutOfOrderError` if you ask for them.
See the `nnsight` skill → source tracing.

## Per-head metrics

Reduce each `[query, key]` matrix to one number per head, then rank. These four
cover most of what people look for:

```python
def head_metrics(pattern):
    """pattern: [heads, query, key] for one layer."""
    entropy = -(pattern * (pattern + 1e-9).log()).sum(-1).mean(-1)
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
  Very common and rarely meaningful; exclude it before ranking heads by anything,
  or it will sort your table for you. Score it over queries `q >= 1` only:
  `A[0, 0]` is 1 by construction and inflates every head.

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

Use the same shape of detector for any hypothesis: construct an input where only
the behavior you care about can produce a high score. A head is an induction head
relative to a probe — natural-text probes rank different heads — so report the
probe with the score.

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

Cut a head **before** `c_proj`, as here — after the projection the hidden
dimension no longer decomposes per head. Slicing `attn.output[0]` instead raises
no error and is not head ablation: the measured effect on P(' Paris') flips sign
for 61 of GPT-2's 144 heads (L0H10: 0.067 sliced wrong, 0.0059 correct).

To test whether the writing matters, the `ablation` skill sweeps every head in one
trace; `activation-patching` patches them. A per-head `.heads` accessor built
with `eproperty` is in the `nnsight` skill's
[modules-and-architectures.md](../nnsight/references/modules-and-architectures.md).

## Across generation

Patterns change every step as the sequence grows:

```python
with model.generate(prompt, max_new_tokens=3) as tracer:
    per_step = nnsight.save([])
    for step in tracer.iter[:3]:
        per_step.append(model.transformer.h[5].attn.output[1].shape[-1])

print("key length per step:", per_step)
```

Inside a generation loop, read a `.source` pattern before its module's `.output`;
in the wrong order it silently binds to the next step and the list comes up one
short (the `nnsight` skill's [generation.md](../nnsight/references/generation.md)).
The key length is the check: step `k` has `prompt_len + k` keys.

## Other architectures

The recipe is the same; the names are not.

- The module path changes (`model.model.layers[i].self_attn`), the pattern handle
  does not: `self_attn.output[1]` under eager, on every family tested. Source
  operation names differ by model class and `transformers` version — always
  `print(block.attn.source)` first.
- **Grouped-query attention** (Llama-3, Qwen, Gemma): several query heads share
  one KV head. The pattern still has one row per *query* head, and
  `o_proj.input.view(B, S, num_attention_heads, head_dim)` still splits the output
  per query head (`torch.equal` to the implementation's per-head output on
  Qwen2.5-0.5B and gemma-2-2b). Only `k_proj` / `v_proj` slices are indexed by
  `num_key_value_heads`; `scripts/inspect_model.py` in the `nnsight` skill prints
  both counts.
- **`head_dim` is not `hidden_size // num_attention_heads`.** Read
  `attn._module.head_dim`. On gemma-2-2b the arithmetic gives 288 and the true
  value is 256, and the wrong `.view()` succeeds.
- **Gemma-2's eager attention is not plain softmax attention.** It softcaps the
  attention logits (`attn_logit_softcapping = 50.0`) between the matmul and the
  softmax, visible as `torch_tanh_0` in the implementation's operation list.
- nnterp's `StandardizedTransformer(..., attn_implementation="eager")` gives
  `model.layers[i].self_attn.attention_probabilities` with no per-architecture
  names at all, on every family it knows — see the nnterp plugin's
  `nnterp:nnterp` skill.

## Cautions

**A pattern is not an explanation.** Attention weight is not information flow —
a head can attend strongly to a position and write nothing useful from it. Confirm
with ablation or patching before claiming a head "does" something, and establish
a null distribution first: sweep every head at the layer, and prefer resample
ablation over zeroing, whose per-head deltas are dominated by off-distribution
shock.

**Average over inputs.** Head behavior is input-dependent; a single prompt gives a
single anecdote. Score over a set of prompts and report the distribution.

**One layer's pattern hides composition.** Induction needs a previous-token head
in an earlier layer feeding a matching head later. Interpreting either alone
misses the circuit — see the `circuit-discovery` skill.

## Related skills

- `nnsight` — `.source`, batching, module paths
- `ablation` — testing whether a head's output matters
- `activation-patching` — per-head causal attribution
- `circuit-discovery` — composing heads into circuits
