---
name: model-editing-and-lora
description: Change what a model knows or how it behaves, persistently — direct weight edits, ROME-style rank-one updates, persistent activation edits via model.edit, loading PEFT/LoRA adapters, and training a LoRA or adapter through a frozen model with nnsight's interleaved backward. Use to install or overwrite a fact, to bake an intervention into a model everyone shares, to fine-tune a small adapter while watching internals, and to evaluate an edit for specificity and generalization rather than just checking that the target prompt changed.
---

# Model Editing and LoRA

Three levels of permanence, in increasing order of commitment:

| Level | What changes | Reversible | Use it to |
|---|---|---|---|
| `model.edit()` | activations, replayed on every run | `model.clear_edits()` | test an intervention across many runs |
| adapter / LoRA | added parameters routed into the forward | detach the module | change behavior across many inputs; study an existing fine-tune (`peft=`) |
| weight edit | the model's own parameters | only if you saved a copy | overwrite one association at a layer a causal trace implicated; ship the change baked in |

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

Capture an intervention once and it replays on every future run; the weights
never change.

```python
with model.edit(inplace=True):
    model.transformer.h[9].output[:, -1, :] = 0

with model.trace(prompt):
    edited = model.output.logits[0, -1].argmax().save()

print(f"with edit: {model.tokenizer.decode(edited)!r}")
assert int(edited) != int(original)
model.clear_edits()
```

```
original prediction: ' Paris'
with edit: ' and'
```

Without `inplace=True`, `model.edit()` returns an edited *copy* sharing the same
weights — the safe default when comparing against the original.

### Two boundaries an edit does not cross

**An edit belongs to the envoy you stored it on.** A trace replays only the edits
of the envoy it is rooted at, so an edit stored on a layer does nothing when you
trace the model, and the model's `clear_edits()` does not remove it:

```python
with model.transformer.h[9].edit(inplace=True):
    model.transformer.h[9].output[:] = 0

with model.trace(prompt):
    unaffected = model.output.logits[0, -1].argmax().save()

print(f"root trace: {model.tokenizer.decode(unaffected)!r}")   # ' Paris'
assert int(unaffected) == int(original)

model.clear_edits()
assert len(model.transformer.h[9]._edits) == 1                 # still there
model.transformer.h[9].clear_edits()                           # this clears it
```

Store the edit on the model and write the layer's path inside it instead.

**An edit applies through the interleaver, not to the module.** `model.trace(...)`,
`model.generate(...)`, `model.pipe(...)` and `model.trace(**ids, trace=False)` all
replay stored edits. A direct call on the wrapped `torch.nn.Module` does not:

```python
ids = model.tokenizer(prompt, return_tensors="pt").to(model.device)

with model.edit(inplace=True):
    model.transformer.h[9].output[:] = 0

through_trace = int(model.trace(**ids, trace=False).logits[0, -1].argmax())
direct_call = int(model(**ids).logits[0, -1].argmax())

print(model.tokenizer.decode(through_trace), "|", model.tokenizer.decode(direct_call))
assert through_trace != direct_call and direct_call == int(original)
model.clear_edits()
```

So a HuggingFace `Trainer`, an eval harness, or anything else holding
`model._module` runs the unedited model. Route those through `model.trace(...)`,
or make the change in the weights.

## Direct weight editing

Weights are ordinary parameters; edit them outside a trace with plain torch. The
ROME-style move is a **rank-one update** to an MLP output projection: the key is
the MLP's hidden activation at the subject, the value the target token's direction.

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
after rank-one edit: ' Rome'
after restore: ' Paris'
```

**Always keep a backup.** A weight edit is global: everything holding the model
sees it, including a direct `model(**ids)` call that an activation edit misses.

On a tensor-parallel model, sharded weights are `DTensor`s and this code needs
`.to_local()` — see the `tensor-parallel` skill.

Real ROME solves a constrained least-squares problem for the update instead of
eyeballing a scale, at the layer a causal trace identified — find it first with the
`causal-tracing` skill; editing a layer the trace does not implicate mostly
produces collateral damage.

## Loading a PEFT / LoRA adapter

`TransformersModel` takes a `peft=` argument (a hub repo id or a local path) and
grafts the adapter onto the model **in place**: each targeted module becomes the
adapter's layer, and nothing else moves. Every path you wrote for the base model
still names the same module, and the adapter's own modules join the envoy tree,
so you can read *inside* the adapter:

<!-- test: requires peft -->
```python
import tempfile
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

adapter_dir = tempfile.mkdtemp()             # a throwaway adapter; a repo id works the same
get_peft_model(
    AutoModelForCausalLM.from_pretrained("openai-community/gpt2"),
    LoraConfig(r=4, target_modules=["c_attn"], fan_in_fan_out=True, init_lora_weights=False),
).save_pretrained(adapter_dir)

adapted = TransformersModel("openai-community/gpt2", peft=adapter_dir, dispatch=True)

with adapted.trace(prompt):
    lora_out = adapted.transformer.h[0].attn.c_attn.lora_B.default.output.save()
    adapted_logits = adapted.output.logits[0, -1].detach().save()
with model.trace(prompt):
    base_logits = model.output.logits[0, -1].save()

print(type(adapted._module).__name__, tuple(lora_out.shape))
print([p for p, _ in adapted.named_modules() if "lora_B" in p][:2])
print(f"max logit shift vs base: {float((adapted_logits - base_logits).abs().max()):.2f}")
assert type(adapted._module).__name__ == "GPT2LMHeadModel"      # not re-rooted
```

To study what a fine-tune changed, diff activations like this against the base
model at every layer, not just the logits.

## Training an adapter through a frozen model

nnsight's interleaved backward means an adapter inserted at any layer trains with
ordinary torch. The model's parameters never move; only the adapter's do.

**Training against a *remote* model** needs a different recipe — the adapter and
optimizer are built inside `model.session(remote=True)`. The `remote` skill
covers it.

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

def route_lora(layer):
    hidden = layer.output
    layer.output = hidden + lora(hidden)          # replacement, not [:]=

for step in range(30):
    with model.trace(prompt):
        route_lora(model.transformer.h[9])
        loss = -model.output.logits[0, -1].log_softmax(-1)[rome]
        with loss.backward():
            pass
        tracked = nnsight.save(loss.item())
    optimizer.step()
    optimizer.zero_grad()
    if step % 10 == 0:
        print(f"step {step:2d}  loss {tracked:.3f}")

with model.trace(prompt):
    route_lora(model.transformer.h[9])
    trained = model.output.logits[0, -1].argmax().save()

print(f"with trained LoRA: {model.tokenizer.decode(trained)!r}")
```

```
step  0  loss 5.084
step 10  loss 0.969    (varies with initialization)
step 20  loss 0.000
with trained LoRA: ' Rome'
```

**Use replacement assignment, not in-place, when gradients flow through the
intervention** — the most common failure when training through nnsight.
`layer.output[:] = hidden + lora(hidden)` modifies the tensor autograd needs and
raises `RuntimeError: one of the variables needed for gradient computation has
been modified by an inplace operation`; `layer.output = ...` keeps the graph.

To make the adapter permanent, move the routing into an edit. A plain edit fires
at each location's **first** occurrence, which under `generate` is prefill alone —
so put the routing under `tracer.iter[:]`, and count the steps the adapter ran
whenever an edit is supposed to hold across a generation:

```python
model.transformer.h[9].adapter = lora

with model.edit(inplace=True) as tracer:
    for _ in tracer.iter[:]:
        hidden = model.transformer.h[9].output
        model.transformer.h[9].output = hidden + model.transformer.h[9].adapter(hidden, hook=True)

with model.generate(prompt, max_new_tokens=5, do_sample=False) as tracer:
    ran = tracer.cache(modules=[model.transformer.h[9].adapter])
    ids = tracer.result.save()

print(model.tokenizer.decode(ids[0]))
print(f"adapter ran {len(ran['model.transformer.h.9.adapter'])} of 5 steps")
assert len(ran["model.transformer.h.9.adapter"]) == 5      # 1 without the loop
model.clear_edits()
```

`hook=True` is what puts the adapter in the cache at all: a module you call
yourself is not recorded otherwise.

## Evaluating an edit

Changing the target prompt is the easy part and proves nothing. Three tests:

**Efficacy** — does the target prompt produce the new answer? (Shown above.)

**Generalization** — do paraphrases follow? An edit that only fires on the exact
string is a lookup table, not knowledge.

**Specificity** — is everything *else* unchanged? The test that fails most often
and the one people skip.

```python
def predictions(texts, edited):
    with model.trace() as tracer:
        out = nnsight.save([])
        for text in texts:
            with tracer.invoke(text):
                if edited:
                    route_lora(model.transformer.h[9])
                out.append(model.output.logits[0, -1].argmax())
    return [model.tokenizer.decode(token) for token in out]

paraphrases = [
    "The Eiffel Tower is located in the city of",
    "You can visit the Eiffel Tower in",
    "The Eiffel Tower stands in",
]
unrelated = [
    "The Colosseum is in the city of",
    "The capital of Japan is",
    "Water freezes at a temperature of",
]

for texts in (paraphrases, unrelated):
    for text, before, after in zip(texts, predictions(texts, False), predictions(texts, True)):
        flag = "   <-- COLLATERAL DAMAGE" if texts is unrelated and before != after else ""
        print(f"{text!r:<45} {before!r:<9} -> {after!r}{flag}")
```

```
'The Eiffel Tower is located in the city of'  ' Paris'  -> ' Rome'
'You can visit the Eiffel Tower in'           ' London' -> ' Rome'
'The Eiffel Tower stands in'                  ' the'    -> ' Rome'
'The Colosseum is in the city of'             ' P'      -> ' Rome'   <-- COLLATERAL DAMAGE
'The capital of Japan is'                     ' the'    -> ' Rome'   <-- COLLATERAL DAMAGE
'Water freezes at a temperature of'           ' about'  -> ' -'   <-- COLLATERAL DAMAGE
```

The LoRA generalizes perfectly and is a catastrophe: trained on one prompt with
no constraint on anything else, it learned "always say Rome". The fixes are the
ones real editing methods use — train on a **set** of prompts, add a KL or L2 term
pinning behavior on unrelated inputs, use the smallest rank and fewest layers
that work, and stop at the efficacy you need. Report all three numbers.

## Related skills

- `causal-tracing` — finding the layer worth editing
- `nnsight` — `edit`, attaching modules, gradients
- `model-steering` — inference-time behavior change without touching weights
- `debugging` — the in-place/autograd error above and others
- `remote` — training against NDIF
- `tensor-parallel` — editing sharded (`DTensor`) weights
