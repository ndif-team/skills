# Distributed alignment search (learned subspaces)

Sometimes no single component patches cleanly because the variable is encoded in a
**direction** spread across the residual stream. DAS learns that subspace: patch
only the component of the activation along a learned direction, and optimize the
direction to make the transfer work.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

clean = "The Eiffel Tower is in the city of"      # → " Paris"
corrupt = "The Colosseum is in the city of"       # same token length
paris = model.tokenizer.encode(" Paris")[0]
rome = model.tokenizer.encode(" Rome")[0]
SUBJECT = slice(1, 5)                             # the tokens that differ
assert len(model.tokenizer(clean).input_ids) == len(model.tokenizer(corrupt).input_ids)
```

Train one direction at layer 6 so that swapping the subject's component along it,
from the clean run into the corrupt one, raises the Paris − Rome logit difference:

```python
LAYER = 6
with model.trace(clean):
    donor_acts = model.transformer.h[LAYER].output.detach().save()

generator = torch.Generator().manual_seed(0)
direction = torch.nn.Parameter(torch.randn(768, generator=generator).to(model.device) * 0.02)
optimizer = torch.optim.Adam([direction], lr=0.05)

for step in range(20):
    with model.trace(corrupt):
        unit = direction / direction.norm()
        acts = model.transformer.h[LAYER].output
        delta = donor_acts[:, SUBJECT, :] - acts[:, SUBJECT, :]
        acts[:, SUBJECT, :] = acts[:, SUBJECT, :] + (delta @ unit).unsqueeze(-1) * unit
        logits = model.output.logits[0, -1]
        loss = -(logits[paris] - logits[rome])
        with loss.backward():
            pass
        tracked = nnsight.save(loss.item())
    optimizer.step()
    optimizer.zero_grad()
    if step % 5 == 0:
        print(f"step {step:2d}  logit diff {-tracked:+.3f}")
```

```
step  0  logit diff -1.491
step  5  logit diff +31.455
step 10  logit diff +53.149
step 15  logit diff +62.827
```

**Read that result skeptically — it is the standard criticism of DAS.** A learned
direction drove the logit difference to +63 when the *real* clean run only
achieves `+2.425`. The optimizer found a direction that pushes the output, not
necessarily the one the model uses. Keep the intervention on the scale of the
natural activation difference, prefer the smallest subspace that works, and run
three nulls before claiming a subspace:

- **held-out prompts** — report transfer on pairs the direction was not trained on
- **a random direction** trained the same way — if it does nearly as well, the
  subspace claim is empty. That is DAS with another seed, though, and it inherits
  the optimizer's freedom. The cleaner null is untrained: shift the activation by
  the *same per-pair magnitude* as the learned swap, `|delta @ unit|`, along a
  random unit vector. On subject-number pairs that shift moved recovery by at most
  0.08 at layers 7–11, where the real direction recovered 0.46–0.97 (the probing
  skill's interchange runs it)
- **an arbitrary target** — train the same subspace to emit an unrelated answer,
  or from a donor prompt that does not contain the fact. A subspace that can be
  pointed at any answer is not carrying this one

Used with those controls, DAS answers questions single-component patching cannot:
whether a variable exists as a linear feature at all, and how many dimensions it
occupies.
