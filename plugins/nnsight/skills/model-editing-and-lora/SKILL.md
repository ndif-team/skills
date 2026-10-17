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
ROME-style move is a **rank-one update** to an MLP output projection `W`, chosen so
that one key `k` maps to a new value: `W += u·dvᵀ / (uᵀk)`, which gives
`k @ ΔW == dv` exactly. Three choices decide whether it hits only the target:

- **the key is the `c_proj` input at the subject's last token** (`' Tower'`), the
  place the fact is looked up. A key taken at the prompt's last token turns the
  update into "say Rome after any prompt like this". Measured on GPT-2 at layer 9,
  every unrelated prompt flipped to `' Rome'` (Statue of Liberty p=0.999, capital
  of Japan 0.743, Obama 0.457).
- **`u` is the key whitened by a key covariance** (`u = C⁻¹k`, `C` from ~20 ordinary
  sentences, regularized), so that keys unlike this one barely move.
- **`dv` is optimized**, not guessed: the change to the MLP output at the subject
  that makes the target likely, found through nnsight's backward.

```python
model.requires_grad_(False)          # only the edit's own tensors take gradients

SUBJECT = 4                          # ' Tower', the subject's last token
assert model.tokenizer.decode(model.tokenizer(prompt).input_ids[SUBJECT]) == " Tower"

corpus = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "In 1905, Albert Einstein published four papers that changed physics forever.",
    "The stock market fell sharply on Monday after the central bank raised rates.",
    "She opened the window and listened to the rain falling on the old roof.",
    "Python is a programming language that emphasizes readability and simplicity.",
    "The Great Wall of China stretches thousands of miles across northern China.",
    "Mount Everest is the highest mountain on Earth, located in the Himalayas.",
    "The Golden Gate Bridge is a suspension bridge in San Francisco, California.",
    "Shakespeare wrote many plays, including Hamlet, Macbeth and King Lear.",
    "The Amazon rainforest covers much of northern Brazil and several neighbors.",
    "Neural networks learn representations by adjusting weights through gradient descent.",
    "The Louvre Museum in Paris houses the Mona Lisa and thousands of other works.",
    "Basketball was invented by James Naismith in Springfield, Massachusetts.",
    "The Sydney Opera House is one of the most recognizable buildings in Australia.",
    "Tokyo is the capital of Japan and one of the largest cities in the world.",
    "The Leaning Tower of Pisa is famous for its unintended tilt.",
    "Coffee is one of the most widely consumed beverages in the world.",
    "The Statue of Liberty was a gift from the people of France to the United States.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The Taj Mahal is a white marble mausoleum in Agra, India.",
]

def rank_one_edit(layer, target=rome, stop_at=0.9, lam=1e-2):
    """The weight and the rank-one delta that map the subject's key to a new value."""
    mlp = model.transformer.h[layer].mlp
    with model.trace() as tracer:                  # key covariance over every corpus token
        keys = nnsight.save([])
        for text in corpus:
            with tracer.invoke(text):
                keys.append(mlp.c_proj.input[0].detach())
    K = torch.cat(list(keys))
    C = K.T @ K / len(K)
    C += lam * C.diagonal().mean() * torch.eye(len(C), device=C.device)

    with model.trace(prompt):                      # the key at the subject
        k = mlp.c_proj.input[0, SUBJECT].detach().save()

    dv = torch.zeros(model.config.n_embd, device=model.device, requires_grad=True)
    optimizer = torch.optim.Adam([dv], lr=0.5)
    for _ in range(200):                           # the value: an MLP output change
        with model.trace(prompt):
            mlp.output[:, SUBJECT, :] += dv
            logp = model.output.logits[0, -1].log_softmax(-1)
            loss = -logp[target] + 1e-3 * (dv ** 2).sum()
            with loss.backward():
                pass
            p = nnsight.save(logp[target].exp().item())
        if p >= stop_at:
            break
        optimizer.step()
        optimizer.zero_grad()

    u = torch.linalg.solve(C, k)                   # the whitened key
    return mlp.c_proj.weight, torch.outer(u, dv.detach()) / (u @ k)   # Conv1D: [in, out]
```

Apply it, then run the specificity check from
[Evaluating an edit](#evaluating-an-edit) before believing it:

```python
def top(texts):
    """Per text: (argmax token, p(' Rome')) at the last position."""
    with model.trace() as tracer:
        out = nnsight.save({})
        for text in texts:
            with tracer.invoke(text):
                probs = model.output.logits[0, -1].softmax(-1)
                out[text] = (probs.argmax().item(), probs[rome].item())
    return out

targets = [prompt, "The Eiffel Tower is located in the city of",
           "You can visit the Eiffel Tower in"]
others = ["The Statue of Liberty is in the city of", "The capital of Japan is",
          "Barack Obama was born in", "The Colosseum is in the city of",
          "Big Ben is in the city of"]

before = top(targets + others)
weight, delta = rank_one_edit(layer=4)
backup = weight.detach().clone()
with torch.no_grad():
    weight += delta
after = top(targets + others)

for text in targets + others:
    (b, _), (a, p_rome) = before[text], after[text]
    print(f"{text!r:<44} {model.tokenizer.decode(b)!r:>9} -> "
          f"{model.tokenizer.decode(a)!r:<9} p(Rome) {p_rome:.3f}")

assert after[prompt][0] == rome
assert all(after[t][0] == before[t][0] for t in others)      # specificity holds

with torch.no_grad():
    weight.copy_(backup)
assert top(targets + others) == before                          # restored exactly
```

```
'The Eiffel Tower is in the city of'          ' Paris' -> ' Rome'   p(Rome) 0.981
'The Eiffel Tower is located in the city of'  ' Paris' -> ' Rome'   p(Rome) 0.968
'You can visit the Eiffel Tower in'          ' London' -> ' Rome'   p(Rome) 0.525
'The Statue of Liberty is in the city of'       ' New' -> ' New'    p(Rome) 0.044
'The capital of Japan is'                       ' the' -> ' the'    p(Rome) 0.000
'Barack Obama was born in'                      ' New' -> ' New'    p(Rome) 0.000
'The Colosseum is in the city of'                 ' P' -> ' P'      p(Rome) 0.019
'Big Ben is in the city of'                     ' New' -> ' New'    p(Rome) 0.004
```

The target and both paraphrases move; none of the five unrelated prompts changes
its answer.

**Always keep a backup.** A weight edit is global: everything holding the model
sees it, including a direct `model(**ids)` call that an activation edit misses.

On a tensor-parallel model, sharded weights are `DTensor`s and this code needs
`.to_local()` — see the `tensor-parallel` skill.

### Which layer

Edit at the layer a causal trace identified; find it with the `causal-tracing`
skill. A small model may not show one: that skill reports GPT-2 small has no
subject site on this prompt. Then sweep candidate layers and let the specificity
check choose:

```python
for layer in (4, 6, 8):
    weight, delta = rank_one_edit(layer)
    backup = weight.detach().clone()
    with torch.no_grad():
        weight += delta
    after = top(targets + others)
    with torch.no_grad():
        weight.copy_(backup)
    changed = sum(after[t][0] != before[t][0] for t in others)
    print(f"layer {layer}: p(Rome) {after[prompt][1]:.3f}, "
          f"unrelated changed {changed}/{len(others)}")
```

```
layer 4: p(Rome) 0.981, unrelated changed 0/5
layer 6: p(Rome) 0.924, unrelated changed 0/5
layer 8: p(Rome) 0.922, unrelated changed 0/5
```

All three flip the target cleanly here; when several pass, add paraphrases and
unrelated prompts until the sweep tells them apart. Real ROME estimates `C` from
far more text (Wikipedia statistics, not 20 sentences); this is the same move at
small scale.

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
ordinary torch. **Freeze the base first.** Without `model.requires_grad_(False)`
the backward still fills `.grad` on every base weight; the base would not move,
because the optimizer holds only the adapter's parameters, but those gradients
cost a model's worth of memory and end up in anything that later calls `.step()`
on `model.parameters()`.

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

model.requires_grad_(False)
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

### Train and export a real PEFT adapter

An adapter loaded with `peft=` trains through the same backward, with two traps.
**It loads frozen**: PEFT sets `inference_mode`, so its `lora_` parameters arrive
with `requires_grad=False` and must be switched on. **`save_pretrained` on the
wrapped module writes a full model**: `adapted._module.save_pretrained(dir)` is
the transformers method and silently saves every base weight next to the LoRA
ones. Write the adapter files yourself:

<!-- test: requires peft -->
```python
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

lora_params = [p for name, p in adapted._module.named_parameters() if "lora_" in name]
assert not any(p.requires_grad for p in lora_params)          # loaded for inference
for p in lora_params:
    p.requires_grad_(True)
optimizer = torch.optim.Adam(lora_params, lr=1e-3)

for _ in range(5):
    with adapted.trace(prompt):
        loss = -adapted.output.logits[0, -1].log_softmax(-1)[rome]
        with loss.backward():
            pass
    optimizer.step()
    optimizer.zero_grad()

export_dir = tempfile.mkdtemp()
state = get_peft_model_state_dict(adapted._module)
save_file({f"base_model.model.{k}": v.detach().contiguous() for k, v in state.items()},
          f"{export_dir}/adapter_model.safetensors")
adapted._module.peft_config["default"].save_pretrained(export_dir)

reloaded = TransformersModel("openai-community/gpt2", peft=export_dir, dispatch=True)
c_attn = reloaded.transformer.h[0].attn.c_attn
with adapted.trace(prompt):
    trained_logits = adapted.output.logits.detach().save()
with reloaded.trace(prompt):
    base_part = c_attn.base_layer.output.detach().save()
    lora_part = c_attn.lora_B.default.output.detach().save()
    total = c_attn.output.detach().save()
    reloaded_logits = reloaded.output.logits.detach().save()

assert torch.equal(trained_logits, reloaded_logits)           # bit-identical round trip
scaling = reloaded._module.transformer.h[0].attn.c_attn.scaling["default"]  # alpha / r
assert torch.allclose(total, base_part + scaling * lora_part, atol=1e-5)
```

`lora_B.default.output` is the adapter's contribution **before** scaling: the
module's output is `base_layer.output + (lora_alpha / r) · lora_B.output`. Scale it
before comparing it with anything in the residual stream.

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
no constraint on anything else, it learned "always say Rome". The targeted
rank-one edit above passes the same test because its key is specific to the
subject. For a trained adapter the fixes are the ones real editing methods use — train on a **set** of prompts, add a KL or L2 term
pinning behavior on unrelated inputs, use the smallest rank and fewest layers
that work, and stop at the efficacy you need. Report all three numbers.

## Related skills

- `causal-tracing` — finding the layer worth editing
- `nnsight` — `edit`, attaching modules, gradients
- `model-steering` — inference-time behavior change without touching weights
- `debugging` — the in-place/autograd error above and others
- `remote` — training against NDIF
- `tensor-parallel` — editing sharded (`DTensor`) weights
