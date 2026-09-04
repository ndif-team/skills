---
name: atp-star
description: Correct two blind spots in attribution patching (AtP) before trusting a ranking, then verify survivors for real. QK correction exactly recomputes the local attention softmax for a patched query or key instead of linearizing through it, recovering effects that the plain (clean − noise) · gradient score misses when a softmax row is saturated (probability near 0 or 1). GradDrop reruns backward with one downstream residual write's gradient zeroed at a time, exposing direct/indirect gradient-path cancellation that a single ordinary backward pass hides — a component can score near zero under plain AtP and still have a large, real causal effect. Use when an attention Q/K node's AtP score looks tiny despite a near-saturated softmax row, when a component you suspect matters scores near zero under attribution patching, or before acting on any AtP top-K ranking — always patch the top candidates for real first. Builds on attribution-patching; needs gradients and exact activation patching.
---

# AtP*

Attribution patching (`attribution-patching` skill) scores a component with one
linear approximation: `(clean − noise) · ∇metric`. Two things break that
approximation in ways that are easy to miss because the score doesn't look
broken — it just looks small:

- **A saturated attention softmax.** If a query/key pair already sits at
  probability ≈ 0 or ≈ 1, the softmax's *local slope* there is tiny, so the
  gradient-based score is tiny too — even though swapping in a genuinely
  different key or query can move the probability a long way. The AtP score
  isn't wrong about the derivative; the derivative just isn't the quantity
  that matters here.
- **Residual cancellation.** A component's effect can reach the metric through
  more than one additive path (straight down the residual stream, and again
  through whatever later layers compute from it). A single backward pass sums
  every path's gradient into the same `.grad` before you get to read it. If
  two paths partly cancel, the component can score near zero while still
  having a large, real, single-path effect.

Both failure modes are corrected with more compute at the point of failure,
not with a new full forward/backward for every component. The corrected score
is still a screen — finish by patching real candidates, which is item 3 below
and the point this page keeps returning to.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True, attn_implementation="eager")

clean = "The Eiffel Tower is in the city of"       # -> " Paris"
corrupt = "The Colosseum is in the city of"         # -> " Rome"
paris = model.tokenizer.encode(" Paris")[0]
rome = model.tokenizer.encode(" Rome")[0]
n_layers = len(model.transformer.h)
n_heads = model.config.n_head
head_dim = model.config.n_embd // n_heads
assert len(model.tokenizer(clean).input_ids) == len(model.tokenizer(corrupt).input_ids)

for block in model.transformer.h:
    _ = block.attn.source          # instrument every attention before any trace touches it
```

Same clean/corrupt pair as the `attribution-patching` and `activation-patching`
skills, so scores below are directly comparable to those pages. `eager`
attention is required — `sdpa` never materializes the probability matrix
`.source` needs. Put the model in evaluation mode before relying on any of the
identities below: attention dropout invalidates them, and eval is the
default for a model loaded this way (`model.transformer.h[0].attn.training`
is `False` here) — check it explicitly if you loaded the model differently.

## 1. QK correction: recompute the local softmax, don't linearize it

GPT-2's real attention call is reachable at
`block.attn.source.attention_interface_1`, called as
`attention_interface(self, query_states, key_states, value_states, mask, ...)`
— `self` is the non-tensor first argument, so the query/key/value tensors are
`call.inputs[0][1:4]`. **Read `.inputs` before `.output`** on the same call;
the reverse order raises `OutOfOrderError` because the output is served one
step later than the input snapshot.

```python
LAYER, HEAD, QUERY_POS, KEY_POS = 4, 11, 4, 3   # picked below for a saturated row

with torch.no_grad():
    with model.trace(clean):
        clean_key = model.transformer.h[LAYER].attn.source.key_states_2.output[0, HEAD, KEY_POS, :].save()

with model.trace(corrupt):
    key_ref = model.transformer.h[LAYER].attn.source.key_states_2.output
    key_ref.requires_grad_(True)
    noise_key = key_ref[0, HEAD, KEY_POS, :].save()

    call = model.transformer.h[LAYER].attn.source.attention_interface_1
    args, _ = call.inputs
    noise_query = args[1][0, HEAD, QUERY_POS, :].save()
    noise_value_at_key = args[3][0, HEAD, KEY_POS, :].save()

    attn_out, probs = call.output
    attn_out.requires_grad_(True)
    noise_prob = probs[0, HEAD, QUERY_POS, KEY_POS].save()
    # GPT-2's attention_interface_1 returns [batch, seq, heads, head_dim] here,
    # transposed relative to query's [batch, heads, seq, head_dim] — check, don't assume.
    transposed = nnsight.save(attn_out.shape != args[1].shape)
    noise_output_row = (
        attn_out[0, QUERY_POS, HEAD, :] if transposed else attn_out[0, HEAD, QUERY_POS, :]
    ).detach().save()

    logits = model.output.logits[0, -1]
    metric = logits[paris] - logits[rome]
    noise_metric = metric.save()
    with metric.backward():
        full_output_grad = attn_out.grad.clone().save()     # later forward op: read first
        key_grad = key_ref.grad[0, HEAD, KEY_POS, :].clone().save()

output_grad_row = (
    full_output_grad[0, QUERY_POS, HEAD, :] if transposed else full_output_grad[0, HEAD, QUERY_POS, :]
)

print(f"probability at L{LAYER}H{HEAD}, query {QUERY_POS} -> key {KEY_POS}: {noise_prob.item():.7f}")
assert noise_prob.item() > 0.999   # a genuinely saturated row, not a contrived one
```

```
probability at L4H11, query 4 -> key 3: 0.9999992
```

That row is essentially deterministic — query 4 puts almost all of its mass on
key 3. Now correct it: instead of differentiating through the softmax, solve
for the exact patched probability using the log-odds form (`sigmoid` and its
inverse), which is exact for a one-key change to one row regardless of how
saturated it starts:

```python
scale = head_dim ** -0.5
delta_score = torch.dot(noise_query, clean_key - noise_key) * scale
log_odds = torch.log(noise_prob) - torch.log1p(-noise_prob)
patched_prob = torch.sigmoid(log_odds + delta_score)

# Treat probability == 1 explicitly rather than dividing by (1 - 1).
denominator = 1.0 if noise_prob.item() == 1 else (1 - noise_prob)
probability_scale = 0.0 if noise_prob.item() == 1 else (patched_prob - noise_prob) / denominator
output_delta = probability_scale * (noise_value_at_key - noise_output_row)

naive_atp = torch.dot(clean_key - noise_key, key_grad)
qk_corrected = torch.dot(output_delta, output_grad_row)

print(f"naive gradient-based AtP for this key:  {naive_atp.item():.3e}")
print(f"QK-corrected local attribution:         {qk_corrected.item():.4f}")
```

```
naive gradient-based AtP for this key:  -3.720e-06
QK-corrected local attribution:         -0.1802
```

Six orders of magnitude apart. Verify which one is telling the truth by
patching the key for real — this is item 3 already, one call early because
it's the fastest way to settle the question:

```python
with torch.no_grad():
    with model.trace(corrupt):
        model.transformer.h[LAYER].attn.source.key_states_2.output[0, HEAD, KEY_POS, :] = clean_key
        patched_logits = model.output.logits[0, -1]
        patched_metric = (patched_logits[paris] - patched_logits[rome]).save()

exact_effect = (patched_metric - noise_metric).item()
print(f"exact effect (real patch):     {exact_effect:+.4f}")
print(f"naive AtP recovers             {naive_atp.item() / exact_effect:6.2%} of it")
print(f"QK-corrected score recovers    {qk_corrected.item() / exact_effect:6.2%} of it")

assert abs(naive_atp.item() / exact_effect) < 0.01
assert abs(qk_corrected.item() / exact_effect - 1) < 0.2
```

```
exact effect (real patch):     -0.2071
naive AtP recovers               0.00% of it
QK-corrected score recovers     86.99% of it
```

The naive gradient-based score isn't a rounding error away from the truth —
it recovers **0.0018%** of a real, six-tenths-of-a-logit effect, because the
softmax's local slope at probability `0.9999992` is itself about `6e-7`. The
correction recovers 87% of it in one local recompute plus the same downstream
gradient AtP already needed, with no extra model pass. The remaining gap is
the downstream half of the estimate — dotting the local output delta with a
gradient — which stays first-order; only the softmax step is exact.

## 2. GradDrop: repeat the backward pass, zero one path at a time

A component's gradient is the *sum* over every path from it to the metric.
Ordinary backprop only ever hands you that sum. To see one path at a time,
rerun backward from the same graph and zero the gradient flowing into a
specific downstream residual write before it propagates further back — the
write must execute after the component you're scoring, so its gradient is
encountered first:

```python
with torch.no_grad():
    with model.trace(clean):
        clean_heads = nnsight.save([model.transformer.h[l].attn.c_proj.input for l in range(n_layers)])

with model.trace(corrupt):
    head_refs = [model.transformer.h[l].attn.c_proj.input for l in range(n_layers)]
    for ref in head_refs:
        ref.requires_grad_(True)
    noise_heads = nnsight.save([ref.detach() for ref in head_refs])

    logits = model.output.logits[0, -1]
    metric = logits[paris] - logits[rome]
    corrupt_metric = metric.save()
    grads = nnsight.save([])
    with metric.backward():
        for l in reversed(range(n_layers)):        # reverse forward order
            grads.append(head_refs[l].grad.clone())
grads = grads[::-1]

atp_scores = torch.zeros(n_layers, n_heads)
for l in range(n_layers):
    delta = (clean_heads[l] - noise_heads[l]) * grads[l]
    for h in range(n_heads):
        lo, hi = h * head_dim, (h + 1) * head_dim
        atp_scores[l, h] = delta[..., lo:hi].sum()

CANCEL_LAYER, CANCEL_HEAD = 5, 10
lo, hi = CANCEL_HEAD * head_dim, (CANCEL_HEAD + 1) * head_dim
print(f"L{CANCEL_LAYER}H{CANCEL_HEAD} ordinary AtP score: {atp_scores[CANCEL_LAYER, CANCEL_HEAD]:+.5f}")
```

```
L5H10 ordinary AtP score: +0.00034
```

That head's AtP score is close enough to zero to discard. Run GradDrop before
you do: zero the gradient entering each later block's attention and MLP write
in turn (one downstream write per backward pass, `retain_graph=True` on every
pass but the last), and re-read the head's gradient each time:

```python
downstream = [(l, kind) for l in range(CANCEL_LAYER + 1, n_layers) for kind in ("mlp", "attn")]

with model.trace(corrupt):
    comp_ref = model.transformer.h[CANCEL_LAYER].attn.c_proj.input
    comp_ref.requires_grad_(True)
    noise_slice = comp_ref[:, :, lo:hi].detach().save()

    down_refs = {}
    for l in range(CANCEL_LAYER, n_layers):
        a = model.transformer.h[l].attn.output[0]
        a.requires_grad_(True)
        down_refs[(l, "attn")] = a
        m = model.transformer.h[l].mlp.output
        m.requires_grad_(True)
        down_refs[(l, "mlp")] = m

    logits = model.output.logits[0, -1]
    metric = logits[paris] - logits[rome]
    with metric.backward(retain_graph=True):
        for l in reversed(range(CANCEL_LAYER, n_layers)):
            down_refs[(l, "mlp")].grad             # mlp runs after attn within a block
            down_refs[(l, "attn")].grad
        ordinary_grad = comp_ref.grad[:, :, lo:hi].clone().save()

    dropped = nnsight.save({})
    for i, (l, kind) in enumerate(downstream):
        with metric.backward(retain_graph=(i < len(downstream) - 1)):
            for l2 in reversed(range(CANCEL_LAYER, n_layers)):
                for kind2 in ("mlp", "attn"):
                    if (l2, kind2) == (l, kind):
                        down_refs[(l2, kind2)].grad = torch.zeros_like(down_refs[(l2, kind2)].grad)
                    else:
                        down_refs[(l2, kind2)].grad
            dropped[(l, kind)] = comp_ref.grad[:, :, lo:hi].clone()

ordinary_score = float(((clean_heads[CANCEL_LAYER][:, :, lo:hi] - noise_slice) * ordinary_grad).sum())
drop_estimates = torch.tensor([
    float(((clean_heads[CANCEL_LAYER][:, :, lo:hi] - noise_slice) * dropped[key]).sum())
    for key in downstream
])
graddrop_score = drop_estimates.abs().sum() / (len(downstream) - 1)   # Equation 11

print(f"L{CANCEL_LAYER}H{CANCEL_HEAD} ordinary AtP score:      {ordinary_score:+.5f}")
print(f"L{CANCEL_LAYER}H{CANCEL_HEAD} GradDrop score (Eq. 11): {graddrop_score.item():+.5f}")
```

```
L5H10 ordinary AtP score:      +0.00034
L5H10 GradDrop score (Eq. 11): +0.05686
```

Dropping any single downstream write's gradient — one at a time, never all at
once — moves this head's estimate by two orders of magnitude, and summing the
absolute per-drop estimates (Equation 11 of the AtP* paper, scaled by
`1 / (num_downstream_writes − 1)`) turns twelve noisy one-path estimates into
one score. Confirm against a real patch:

```python
with torch.no_grad():
    with model.trace(corrupt):
        model.transformer.h[CANCEL_LAYER].attn.c_proj.input[:, :, lo:hi] = clean_heads[CANCEL_LAYER][:, :, lo:hi]
        patched_logits = model.output.logits[0, -1]
        cancel_patched_metric = (patched_logits[paris] - patched_logits[rome]).save()

cancel_exact = (cancel_patched_metric - corrupt_metric).item()
print(f"exact effect (real patch):  {cancel_exact:+.5f}")
print(f"ordinary AtP recovers       {ordinary_score / cancel_exact:6.2%} of it")
print(f"GradDrop recovers           {graddrop_score.item() / cancel_exact:6.2%} of it")

assert abs(ordinary_score / cancel_exact) < 0.05
assert abs(graddrop_score.item() / cancel_exact - 1) < 0.3
```

```
exact effect (real patch):  +0.06136
ordinary AtP recovers         0.56% of it
GradDrop recovers            92.66% of it
```

L5H10 has a real effect worth over a fifth of a logit, and ordinary AtP
reports 0.56% of it because this head's direct contribution to the residual
stream and its effect through later layers land with close to opposite sign.
Neither path is small; they cancel in the single sum a plain backward pass
gives you. GradDrop is one extra backward pass per downstream write, on a
graph you already built — no new forward pass, no new noise run.

## 3. Exact verification: a low rank is not the same as negligible

Section 1 and 2 already each ended with a real patch — that habit is the
point of this section, generalized to a ranking instead of one candidate.
Ordinary AtP ranks all 144 heads; patch the top few for real before reporting
anything, and check where the head just found by GradDrop actually sits in
that ranking:

```python
flat_scores = atp_scores.flatten().abs()
rank_by_atp = int((flat_scores > flat_scores[CANCEL_LAYER * n_heads + CANCEL_HEAD]).sum())
print(f"L{CANCEL_LAYER}H{CANCEL_HEAD} ranks #{rank_by_atp + 1} of {n_heads * n_layers} by |ordinary AtP score|")

top5 = flat_scores.topk(5).indices.tolist()
exact_top5 = []
for idx in top5:
    l, h = divmod(idx, n_heads)
    lo_h, hi_h = h * head_dim, (h + 1) * head_dim
    with torch.no_grad():
        with model.trace(corrupt):
            model.transformer.h[l].attn.c_proj.input[:, :, lo_h:hi_h] = clean_heads[l][:, :, lo_h:hi_h]
            patched_logits = model.output.logits[0, -1]
            patched_metric = (patched_logits[paris] - patched_logits[rome]).save()
    real_effect = (patched_metric - corrupt_metric).item()
    exact_top5.append(real_effect)
    print(f"L{l}H{h:<2} atp={atp_scores[l, h]:+.4f}  exact={real_effect:+.4f}")

assert all(abs(a - b) / max(abs(b), 1e-6) < 0.3 for a, b in zip(atp_scores.flatten()[top5].tolist(), exact_top5))
assert cancel_exact > min(abs(e) for e in exact_top5) * 0.1
```

```
L5H10 ranks #141 of 144 by |ordinary AtP score|
L9H8  atp=+1.2402  exact=+1.4311
L8H11 atp=+1.0232  exact=+1.0714
L6H4  atp=+0.6209  exact=+0.7206
L10H0  atp=+0.4775  exact=+0.5129
L11H3  atp=+0.2266  exact=+0.2249
```

Ordinary AtP agrees closely with the exact effect for its own top five — that
part of the ranking is trustworthy, and verifying it was five extra forward
passes. But L5H10 ranks 141st out of 144 by that same score, and its real
effect (`+0.0614`) is more than a quarter of the smallest verified top-5
effect (`+0.2249`) — not a component you'd want to have written off. A
top-K cutoff chosen from the raw AtP magnitude would have silently dropped it.
Two independent, cheap checks catch this before it becomes a wrong claim:
GradDrop on any candidate whose plain score looks suspiciously small for a
component you have a specific reason to suspect (feeds a head or path you
already care about, sits at a position the task depends on), and a real patch
on whatever you're about to report — top-K by score, plus anything GradDrop
or QK correction flagged, not top-K by score alone.

## Gotchas

- **Q, K, and V must come from the same call.** Read `.inputs` before
  `.output` on the same `attention_interface` call — the output is served one
  step after the inputs, so the reverse order raises `OutOfOrderError`.
- **Read gradients in reverse forward order inside one `backward()`.** A later
  forward op's `.grad` must be read before an earlier one's, in every backward
  session, including the repeated ones GradDrop uses.
- **Grab `.grad` off the tensor you called `requires_grad_()` on, not a slice
  of it.** Index into the gradient *after* the backward pass, not into the
  tensor before it — nnsight tracks the object registered during the forward
  pass.
- **Treat probability exactly 0 or exactly 1 as its own case.** The log-odds
  form divides by `1 − probability`; clamping instead of branching corrupts
  valid, merely-small probabilities elsewhere in the same row.
- **Only GPT-2's `attention_interface_1` call and `key_states_2` /
  `attention_interface_1`'s own output layout were verified here, on this
  `transformers` version.** A different architecture or version renames these
  — print `.source` and read the label off the line you want; see the
  `nnsight` skill's source-tracing reference.
- **GradDrop drops one downstream write per backward pass, never the whole
  set at once.** Dropping everything downstream in a single pass answers a
  different question (the pure direct-path effect with every later layer's
  own transformation removed), not the per-layer decomposition Equation 11
  aggregates.
- **This page stops at exact verification of individual candidates.**
  Estimating a confidence bound on everything you *didn't* verify (paired
  subset sampling, the Welch bound) is a further step with its own
  statistical machinery and is out of scope here — see the nnsight-side
  `atp-star` pattern doc if you need it.

## Related skills

- `attribution-patching` — the baseline score these two corrections sit on
  top of, and where the per-head ranking pattern comes from
- `circuit-discovery` — once a corrected ranking is verified, this is where
  edge-level attribution and path patching turn it into a circuit claim
