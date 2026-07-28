---
name: model-editing-and-lora
description: Change what a model knows or how it behaves, persistently — direct weight edits, ROME-style rank-one updates, persistent activation edits via model.edit, loading PEFT/LoRA adapters, and training a LoRA or adapter through a frozen model with nnsight's interleaved backward. Use to install or overwrite a fact, to bake an intervention into a model everyone shares, to fine-tune a small adapter while watching internals, and to evaluate an edit for specificity and generalization rather than just checking that the target prompt changed.
---

# Model Editing and LoRA

Three levels of permanence, in increasing order of commitment:

| Level | What changes | Reversible |
|---|---|---|
| `model.edit()` | activations, replayed on every run | `model.clear_edits()` |
| adapter / LoRA | added parameters routed into the forward | detach the module |
| weight edit | the model's own parameters | only if you saved a copy |

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

torch.manual_seed(0)
model = TransformersModel("openai-community/gpt2", dispatch=True)

prompt = "The Eiffel Tower is in the city of"
rome = model.tokenizer.encode(" Rome")[0]

with model.trace(prompt):
    original = model.output.logits[0, -1].argmax().save()

print(f"original prediction: {model.tokenizer.decode(original)!r}")
```

## Persistent activation edits

The lightest option: capture an intervention once and have it replay on every
future run. Nothing about the weights changes.

```python
with model.edit(inplace=True):
    model.transformer.h[9].output[:, -1, :] *= 1.5

with model.trace(prompt):
    edited = model.output.logits[0, -1].argmax().save()

print(f"with edit: {model.tokenizer.decode(edited)!r}")
model.clear_edits()
```

Use `model.edit()` without `inplace=True` to get an edited *copy* sharing the same
weights — the safe default when you want to compare against the original. See the
`nnsight` skill → control flow.

## Direct weight editing

Weights are ordinary parameters; edit them outside a trace with plain torch. The
ROME-style move is a **rank-one update** to an MLP output projection: write a
key-value association into the weights, where the key is the activation pattern at
the subject and the value is the direction of the token you want.

```python
weight = model.transformer.h[9].mlp.c_proj.weight        # Conv1D: [in, out]
backup = weight.detach().clone()

# key: what the MLP's hidden layer computes at the last position
with model.trace(prompt):
    key = model.transformer.h[9].mlp.source.self_act_0.output[0, -1].detach().save()

# value: the unembedding direction of the target token
value = model.lm_head.weight[rome].detach()

with torch.no_grad():
    update = torch.outer(key / key.norm(), value / value.norm())
    weight += 2.0 * update * float(weight.norm()) / 50

with model.trace(prompt):
    after = model.output.logits[0, -1].argmax().save()

print(f"after rank-one edit: {model.tokenizer.decode(after)!r}")

with torch.no_grad():
    weight.copy_(backup)

with model.trace(prompt):
    restored = model.output.logits[0, -1].argmax().save()

print(f"after restore: {model.tokenizer.decode(restored)!r}")
```

```
original prediction: ' Paris'
after rank-one edit: ' Rome'
after restore: ' Paris'
```

**Always keep a backup.** A weight edit is global: every prompt, every experiment
in the process, and anything else holding a reference to the model sees it.

Real ROME solves for the update rather than eyeballing a scale — it computes the
key from the subject's activations at the layer a causal trace identified, and
solves a constrained least-squares problem so the edit changes the target
association while minimizing movement elsewhere. Use the `causal-tracing` skill to
find the layer first; editing a layer the trace does not implicate mostly produces
collateral damage.

## Loading a PEFT / LoRA adapter

`TransformersModel` takes a `peft=` argument and grafts the adapter at load:

<!-- test: skip -->
```python
adapted = TransformersModel(
    "meta-llama/Llama-3.1-8B",
    peft="username/my-lora-adapter",
    dispatch=True,
)

with adapted.trace("Hello"):
    hidden = adapted.model.layers[16].output[0].save()
```

The adapter's own modules appear in the envoy tree, so you can read *inside* the
adapter — which is the point of doing this in nnsight rather than plain PEFT.
Inspect the paths with `scripts/inspect_model.py --grep lora` from the `nnsight`
skill.

## Training an adapter through a frozen model

nnsight's interleaved backward means an adapter inserted at any layer trains with
ordinary torch. The model's parameters never move; only the adapter's do.

```python
class LoRA(torch.nn.Module):
    def __init__(self, dim, rank=4):
        super().__init__()
        self.down = torch.nn.Linear(dim, rank, bias=False)
        self.up = torch.nn.Linear(rank, dim, bias=False)
        torch.nn.init.zeros_(self.up.weight)         # start as a no-op

    def forward(self, x):
        return self.up(self.down(x))

lora = LoRA(model.config.n_embd).to(model.device)
optimizer = torch.optim.Adam(lora.parameters(), lr=1e-3)

for step in range(30):
    with model.trace(prompt):
        hidden = model.transformer.h[9].output
        model.transformer.h[9].output = hidden + lora(hidden)      # replacement, not [:]=
        loss = -model.output.logits[0, -1].log_softmax(-1)[rome]
        with loss.backward():
            pass
        tracked = nnsight.save(loss.item())
    optimizer.step()
    optimizer.zero_grad()
    if step % 10 == 0:
        print(f"step {step:2d}  loss {tracked:.3f}")

with model.trace(prompt):
    hidden = model.transformer.h[9].output
    model.transformer.h[9].output = hidden + lora(hidden)
    trained = model.output.logits[0, -1].argmax().save()

print(f"with trained LoRA: {model.tokenizer.decode(trained)!r}")
```

```
step  0  loss 5.084
step 10  loss 0.54     (varies with initialization)
step 20  loss 0.000
with trained LoRA: ' Rome'
```

**Use replacement assignment, not in-place, when gradients flow through the
intervention.** Writing `model.transformer.h[9].output[:] = hidden + lora(hidden)`
modifies the tensor autograd needs and raises:

```
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation
```

`output = ...` hands the model a new tensor and leaves the graph intact. This is
the single most common failure when training anything through nnsight.

To make the adapter permanent, move the routing into an edit:

```python
model.transformer.h[9].adapter = lora

with model.edit(inplace=True):
    hidden = model.transformer.h[9].output
    model.transformer.h[9].output = hidden + model.transformer.h[9].adapter(hidden)

with model.generate(prompt, max_new_tokens=5) as tracer:
    ids = tracer.result.save()

print(model.tokenizer.decode(ids[0]))
model.clear_edits()
```

## Evaluating an edit

Changing the target prompt is the easy part and proves nothing. Three tests:

**Efficacy** — does the target prompt produce the new answer? (Shown above.)

**Generalization** — do paraphrases follow? An edit that only fires on the exact
string is a lookup table, not knowledge:

```python
paraphrases = [
    "The Eiffel Tower is located in the city of",
    "You can visit the Eiffel Tower in",
    "The Eiffel Tower stands in",
]

with model.trace() as tracer:
    outputs = nnsight.save([])
    for text in paraphrases:
        with tracer.invoke(text):
            hidden = model.transformer.h[9].output
            model.transformer.h[9].output = hidden + lora(hidden)
            outputs.append(model.output.logits[0, -1].argmax())

for text, token in zip(paraphrases, outputs):
    print(f"{text!r:<45} -> {model.tokenizer.decode(token)!r}")
```

**Specificity** — is everything *else* unchanged? This is the test that fails most
often and the one people skip:

```python
unrelated = [
    "The Colosseum is in the city of",
    "The capital of Japan is",
    "Water freezes at a temperature of",
]

with model.trace() as tracer:
    clean_outputs = nnsight.save([])
    for text in unrelated:
        with tracer.invoke(text):
            clean_outputs.append(model.output.logits[0, -1].argmax())

with model.trace() as tracer:
    edited_outputs = nnsight.save([])
    for text in unrelated:
        with tracer.invoke(text):
            hidden = model.transformer.h[9].output
            model.transformer.h[9].output = hidden + lora(hidden)
            edited_outputs.append(model.output.logits[0, -1].argmax())

for text, before, after in zip(unrelated, clean_outputs, edited_outputs):
    flag = "" if int(before) == int(after) else "   <-- COLLATERAL DAMAGE"
    print(f"{text!r:<40} {model.tokenizer.decode(before)!r} -> "
          f"{model.tokenizer.decode(after)!r}{flag}")
```

```
'The Eiffel Tower is located in the city of'   -> ' Rome'
'You can visit the Eiffel Tower in'            -> ' Rome'
'The Eiffel Tower stands in'                   -> ' Rome'

'The Colosseum is in the city of'      ' P'     -> ' Rome'   <-- COLLATERAL DAMAGE
'The capital of Japan is'              ' the'   -> ' Rome'   <-- COLLATERAL DAMAGE
'Water freezes at a temperature of'    ' about' -> ' -'      <-- COLLATERAL DAMAGE
```

That is the whole lesson in one output. The LoRA generalizes perfectly — every
paraphrase says " Rome" — and it is a catastrophe: the Colosseum now says " Rome",
Japan's capital says " Rome", and water freezes at " -". Trained on a single
prompt with no constraint on anything else, the adapter learned "always say Rome".

The fixes are the same ones real editing methods use: train on a **set** of
prompts, add a KL or L2 term that pins the model's behavior on unrelated inputs,
use the smallest rank and the fewest layers that achieve the target, and stop at
the efficacy you need rather than driving the loss to zero.

Report all three numbers. An edit with high efficacy, no generalization, and
broken specificity is a bug you have installed on purpose.

## Choosing a method

| Goal | Method |
|---|---|
| test an intervention across many runs | `model.edit()` |
| overwrite one factual association | rank-one weight edit, at a layer a causal trace implicated |
| change behavior across many inputs | train an adapter / LoRA |
| ship a model with the change baked in | weight edit or merged LoRA |
| study what an existing fine-tune changed | load it with `peft=` and diff activations against the base |

## Related skills

- `causal-tracing` — finding the layer worth editing
- `nnsight` — `edit`, attaching modules, gradients
- `model-steering` — inference-time behavior change without touching weights
- `nnsight-debugging` — the in-place/autograd error above and others
