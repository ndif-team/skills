# Distributed alignment search

When no single component patches cleanly, the variable may lie along a direction
spread over the residual stream. Distributed alignment search (DAS) learns that
direction: swap only the activation's component along a unit vector from the
clean run into the corrupt run, and train the vector so the swap transfers the
answer. It continues [activation-patching](activation-patching.md) and uses its
pair and its metric.

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)

clean = "The Eiffel Tower is in the city of"        # answer " Paris"
corrupt = "The Colosseum is in the city of"         # answer " Rome", same token count
paris, rome, tokyo = (model.tokenizer.encode(word, add_special_tokens=False)[0]
                      for word in (" Paris", " Rome", " Tokyo"))
SUBJECT = slice(1, 5)                               # the tokens that differ
LAYER = 6

with model.trace(clean):
    donor = model.transformer.h[LAYER].output.detach().save()
    logits = model.output.logits[0, -1]
    clean_paris = (logits[paris] - logits[rome]).save()
    clean_tokyo = (logits[tokyo] - logits[rome]).save()
```

The direction is a leaf tensor you own. Each step is one trace on the corrupt
prompt: the swap is ordinary tensor arithmetic, the empty backward block fills
`direction.grad`, and the optimizer steps outside the trace
([gradients](gradients.md)).

```python
def train(target, other, steps=20):
    """Learn a direction whose swap at SUBJECT raises logit[target] - logit[other]."""
    generator = torch.Generator().manual_seed(0)
    direction = torch.nn.Parameter(torch.randn(768, generator=generator).to(model.device) * 0.02)
    optimizer = torch.optim.Adam([direction], lr=0.05)
    for step in range(steps):
        with model.trace(corrupt):
            unit = direction / direction.norm()
            acts = model.transformer.h[LAYER].output
            delta = donor[:, SUBJECT, :] - acts[:, SUBJECT, :]
            acts[:, SUBJECT, :] = acts[:, SUBJECT, :] + (delta @ unit).unsqueeze(-1) * unit
            logits = model.output.logits[0, -1]
            loss = -(logits[target] - logits[other])
            with loss.backward():
                pass
            diff = nnsight.save(-loss.item())
        optimizer.step()
        optimizer.zero_grad()
    return direction.detach(), diff

direction, to_paris = train(paris, rome)
_, to_tokyo = train(tokyo, rome)                    # an answer the donor prompt does not hold

print(f"swap trained toward ' Paris'  {to_paris:+.1f}   clean run {clean_paris.item():+.2f}")
print(f"swap trained toward ' Tokyo'  {to_tokyo:+.1f}   clean run {clean_tokyo.item():+.2f}")
assert to_paris > 10 * clean_paris.item() and to_tokyo > 10
```

```
swap trained toward ' Paris'  +68.2   clean run +2.42
swap trained toward ' Tokyo'  +48.6   clean run -1.15
```

**Neither number shows a representation.** A one-dimensional swap drives the
logit difference to 28 times what the clean run reaches, and the same recipe
makes the Eiffel Tower donor produce `' Tokyo'`. Given one pair and a free
direction, the optimizer finds a direction that pushes the output, whether or not
the model uses it. A DAS claim needs three nulls:

- **An arbitrary target**, as above: train the same swap toward an answer the
  donor does not hold. It must fail where the real target succeeds.
- **Held-out pairs.** Train on a set of pairs (a list of corrupt prompts and a
  donor with one row per prompt, all of one token count) and report the swap on
  pairs the direction never saw.
- **Untrained directions of the same size**: shift by the learned swap's
  per-position magnitude along random unit vectors.

```python
unit = direction / direction.norm()
generator = torch.Generator().manual_seed(1)
random_units = torch.nn.functional.normalize(torch.randn(5, 768, generator=generator), dim=-1).to(model.device)

with torch.no_grad(), model.trace() as tracer:
    untrained = nnsight.save({})
    for index in range(5):
        with tracer.invoke(corrupt):
            acts = model.transformer.h[LAYER].output
            size = (donor[:, SUBJECT, :] - acts[:, SUBJECT, :]) @ unit
            acts[:, SUBJECT, :] = acts[:, SUBJECT, :] + size.unsqueeze(-1) * random_units[index]
            logits = model.output.logits[0, -1]
            untrained[index] = logits[paris] - logits[rome]

print("random directions, same size: " + " ".join(f"{untrained[index].item():+.2f}" for index in range(5)))
assert max(abs(untrained[index].item()) for index in range(5)) < 5
```

```
random directions, same size: -2.23 -1.29 -0.79 -1.15 -1.22
```

The untrained shifts stay within one logit of the corrupt run's -1.49, so this
null passes here while the arbitrary target fails. Each null catches a different
failure, and one of them is not enough.

For a subspace of `k` dimensions, make the parameter a `(768, k)` matrix,
orthonormalize it inside the trace with `Q, _ = torch.linalg.qr(basis)`, and swap
`(delta @ Q) @ Q.T`. Report the smallest `k` that passes the nulls.
