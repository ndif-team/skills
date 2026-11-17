# Model editing and LoRA

Three ways to make a change that outlasts one trace:

| Change | Where it lives | Undo | Use it to |
|---|---|---|---|
| activation edit (`model.edit`) | an intervention nnsight replays on every run | `model.clear_edits()` | hold an intervention across many runs |
| adapter / LoRA | added parameters routed into the forward | stop routing it, or `model.disable_adapters()` for a PEFT adapter | change behavior across many inputs, or study an existing fine-tune |
| weight edit | the model's own parameters | only from a copy you kept | overwrite one association and ship it in the weights |

An activation edit is stored on one envoy and replayed only by runs through
nnsight, so a direct call on the wrapped module (a `Trainer`, an eval harness
holding `model._module`) runs the unedited model:
[control-flow](control-flow.md). A weight edit and a PEFT adapter are in the
module itself and reach those callers.

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
rome = model.tokenizer.encode(" Rome", add_special_tokens=False)[0]
```

## Three tests for any edit

That the target prompt changed proves nothing. An edit is judged on three sets of
prompts:

- **Efficacy**: the target prompt gives the new answer.
- **Generalization**: paraphrases follow. An edit that fires only on the exact
  string is a lookup table.
- **Specificity**: unrelated prompts keep their answers. It is the test the
  edits below fail first.

```python
targets = [prompt, "The Eiffel Tower is located in the city of",
           "You can visit the Eiffel Tower in"]
others = ["The Statue of Liberty is in the city of", "The capital of Japan is",
          "Barack Obama was born in", "The Colosseum is in the city of",
          "Big Ben is in the city of"]

def top(texts, route=None):
    """Per text: (argmax token id, p(' Rome')) at the last position. One forward pass."""
    with torch.no_grad(), model.trace() as tracer:
        out = nnsight.save({})
        for text in texts:
            with tracer.invoke(text):
                if route is not None:
                    route()                          # an intervention under test
                probs = model.output.logits[0, -1].softmax(-1)
                out[text] = (probs.argmax().item(), probs[rome].item())
    return out

def show(after):
    for text in targets + others:
        (b, p_before), (a, p_after) = before[text], after[text]
        print(f"{text!r:<44} {model.tokenizer.decode(b)!r:>9} -> "
              f"{model.tokenizer.decode(a)!r:<8} p(Rome) {p_before:.3f} -> {p_after:.3f}")

before = top(targets + others)
```

## A rank-one weight edit

Weights are ordinary parameters: edit them outside a trace with plain torch, and
keep a copy, because nothing else undoes the edit. The ROME-style edit is a
rank-one update to an MLP output projection `W`, built so that one key `k` maps
to a new value: `ΔW = u·dvᵀ / (uᵀk)` gives `k @ ΔW == dv` exactly. Three choices
decide whether it changes only the target:

- **The key is the `c_proj` input at the subject's last token** (` Tower`). Taken
  at the prompt's last token, the edit becomes "say Rome after `in the city of`":
  the three unrelated prompts with that ending flip to ` Rome` (p 0.94 to 0.99),
  and the paraphrase with another ending does not follow.
- **`u` is the key whitened by a key covariance** (`u = C⁻¹k`, with `C` taken
  over the tokens of ordinary text and regularized), so that other keys barely
  move. With `u = k`, three of the five unrelated prompts flip.
- **`dv` is optimized**: the change to the MLP output at the subject that makes
  the target likely, found with a backward pass.

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
    # GPT-2's Conv1D weight is [in, out]. An nn.Linear weight is [out, in]: transpose the delta.
    return mlp.c_proj.weight, torch.outer(u, dv.detach()) / (u @ k)

weight, delta = rank_one_edit(layer=4)
backup = weight.detach().clone()
with torch.no_grad():
    weight += delta
after = top(targets + others)
show(after)

assert all(after[text][0] == rome for text in targets)                 # efficacy, generalization
assert all(after[text][0] == before[text][0] for text in others)       # specificity

with torch.no_grad():
    weight.copy_(backup)
assert top(targets + others) == before                                 # restored exactly
```

```
'The Eiffel Tower is in the city of'          ' Paris' -> ' Rome'  p(Rome) 0.006 -> 0.981
'The Eiffel Tower is located in the city of'  ' Paris' -> ' Rome'  p(Rome) 0.005 -> 0.968
'You can visit the Eiffel Tower in'          ' London' -> ' Rome'  p(Rome) 0.009 -> 0.525
'The Statue of Liberty is in the city of'       ' New' -> ' New'   p(Rome) 0.047 -> 0.044
'The capital of Japan is'                       ' the' -> ' the'   p(Rome) 0.000 -> 0.000
'Barack Obama was born in'                      ' New' -> ' New'   p(Rome) 0.000 -> 0.000
'The Colosseum is in the city of'                 ' P' -> ' P'     p(Rome) 0.014 -> 0.019
'Big Ben is in the city of'                     ' New' -> ' New'   p(Rome) 0.004 -> 0.004
```

The covariance needs enough text. With the corpus one sentence shorter, the
Colosseum prompt, whose four top candidates lie between 0.014 and 0.019, flips
to ` Rome`; with the first six sentences p(Rome) on the Statue of Liberty prompt
rises to 0.11. ROME estimates `C` from a large corpus.

Edit at the layer a causal trace implicates
([causal-tracing](causal-tracing.md)). When it implicates none, sweep `layer` and
keep the layers that pass the asserts: here 4, 6 and 8 all do, and a longer list
of paraphrases and unrelated prompts is what separates them. On a
tensor-parallel model the weights are sharded and need a different write:
[tensor-parallel](tensor-parallel.md).

## Training an adapter through a frozen model

An adapter added at any layer trains with ordinary torch, because the backward
pass runs through the intervention. Freeze the base first: without
`model.requires_grad_(False)` each backward fills `.grad` on every base weight.

```python
class LoRA(torch.nn.Module):
    def __init__(self, dim, rank=4):
        super().__init__()
        self.down = torch.nn.Linear(dim, rank, bias=False)
        self.up = torch.nn.Linear(rank, dim, bias=False)
        torch.nn.init.zeros_(self.up.weight)         # starts as a no-op

    def forward(self, x):
        return self.up(self.down(x))

torch.manual_seed(0)
lora = LoRA(model.config.n_embd).to(model.device)
model.requires_grad_(False)
optimizer = torch.optim.Adam(lora.parameters(), lr=1e-3)

def route_lora():
    layer = model.transformer.h[9]
    hidden = layer.output
    layer.output = hidden + lora(hidden)             # a replacement, not layer.output[:] = ...

losses = []
for step in range(30):
    with model.trace(prompt):
        route_lora()
        loss = -model.output.logits[0, -1].log_softmax(-1)[rome]
        with loss.backward():
            pass
        tracked = nnsight.save(loss.item())
    optimizer.step()
    optimizer.zero_grad()
    losses.append(tracked)

after = top(targets + others, route=route_lora)
print("loss at steps 0, 10, 20:", [round(losses[step], 3) for step in (0, 10, 20)])
print("say Rome:", sum(after[text][0] == rome for text in targets), "of 3 targets,",
      sum(after[text][0] == rome for text in others), "of 5 unrelated prompts")
assert all(after[text][0] == rome for text in targets + others)
```

```
loss at steps 0, 10, 20: [5.084, 0.538, 0.0]
say Rome: 3 of 3 targets, 5 of 5 unrelated prompts
```

The adapter passes efficacy and generalization and fails specificity on every
prompt: trained on one prompt with nothing holding the rest in place, it learned
to say Rome. A penalty in the loss is not enough by itself either: a KL term to
the base model's outputs on six other prompts brings their drift to 0.005 and
still leaves three of the five unrelated prompts saying Rome. Measure specificity
on prompts the training never saw.

The routing assigns a replacement. Written in place (`layer.output[:] = ...`) it
raises `RuntimeError: ... modified by an inplace operation`
([gradients](gradients.md)). Training against a model on NDIF puts the adapter
and the optimizer loop inside one `model.session(remote=True)`:
[remote-sessions-and-jobs](remote-sessions-and-jobs.md).

### Keeping the adapter on every run

Attach the adapter to the module tree and move the routing into an edit. Under
`generate` a plain edit runs on the first forward pass only, so the routing goes
under `tracer.iter[:]`. Count the steps it ran whenever an edit must hold across
a generation:

```python
model.transformer.h[9].adapter = lora

with model.edit(inplace=True) as tracer:
    for _ in tracer.iter[:]:
        hidden = model.transformer.h[9].output
        model.transformer.h[9].output = hidden + model.transformer.h[9].adapter(hidden, hook=True)

with model.generate(prompt, max_new_tokens=5, min_new_tokens=5, do_sample=False) as tracer:
    ran = tracer.cache(modules=[model.transformer.h[9].adapter])

assert len(ran["model.transformer.h.9.adapter"]) == 5      # 1 without the loop
model.clear_edits()
```

`hook=True` is what makes the call visible to the cache; without it the cache
comes back empty ([modules-and-architectures](modules-and-architectures.md)).

## PEFT adapters

The examples in this section need the `peft` package. `peft=` takes the repo id
or local path of an adapter that adds layers (LoRA and its relatives) and
attaches it to the model in place: each targeted module becomes the adapter's
layer and holds the original as `base_layer`. No other module moves, so every
path into the base model still names the same module, and the adapter's own
modules are in the tree to read. `model.load_adapter(peft_id)` attaches or swaps
an adapter on a loaded model, and `model.load_adapter(None)` removes it.

For the base and the adapted side of one experiment, keep the adapter attached
and switch it between traces: `disable_adapters()` runs the base model's forward
bit for bit, `enable_adapters()` the adapted one, and `adapters_enabled` holds
the state.

```python
adapter_id = "peft-internal-testing/gpt2-lora-random"     # a LoRA for this base model: repo id or path
adapted = TransformersModel("openai-community/gpt2", task="text-generation",
                            peft=adapter_id, dispatch=True)
c_attn = adapted.transformer.h[0].attn.c_attn

with torch.no_grad(), adapted.trace(prompt):
    base_part = c_attn.base_layer.output.save()
    lora_part = c_attn.lora_B.default.output.save()
    total = c_attn.output.save()
    adapted_logits = adapted.output.logits[0, -1].save()

adapted.disable_adapters()                    # the same tree, the base model's forward
with torch.no_grad(), adapted.trace(prompt):
    base_logits = adapted.output.logits[0, -1].save()
adapted.enable_adapters()

with torch.no_grad(), model.trace(prompt):    # the plain gpt2 from the first block
    plain_logits = model.output.logits[0, -1].save()

scaling = c_attn.scaling["default"]           # lora_alpha / r
assert type(adapted._module).__name__ == "GPT2LMHeadModel"        # attached in place, not wrapped
assert torch.allclose(total, base_part + scaling * lora_part, atol=1e-5)
assert torch.equal(base_logits, plain_logits) and adapted.adapters_enabled
print(f"largest logit change from the adapter: {float((adapted_logits - base_logits).abs().max()):.2f}")
```

```
largest logit change from the adapter: 5.22
```

- **`lora_B.default.output` is the adapter's contribution before scaling.** The
  module's output is `base_layer.output + (lora_alpha / r) · lora_B.output`.
- **While the adapter is off its own modules do not run.** Reading
  `lora_B.default.output` then raises `OutOfOrderError`. `base_layer` runs in
  both states.
- **Switch between traces.** Called inside a trace body, `disable_adapters()`
  stays in force after the trace. For a base row and an adapted row in one
  forward pass, assign `base_layer.output` to `output` on every adapted layer
  inside the base row's invoke.
- **Attaching changes no `requires_grad` flag of the base model**, and neither
  does switching or removing, so gradients with respect to activations work as
  on the base model. The adapter's own parameters arrive frozen.
- **On a half-precision model the adapter is held in float32**, so
  `lora_B.default.output` is float32 where the layer's output is in the model's
  dtype. `autocast_adapter_dtype=False`, on the constructor or `load_adapter`,
  holds the adapter in the model's dtype.

### Train and export

An attached adapter trains through the same backward block. Set the flags first,
on the loaded model (flags set before the weights load are lost). The adapter's
parameters arrive frozen, and left that way the loop runs without an error and
the adapter never moves. The base model's are as you left them, and left on,
every backward fills `.grad` on every base weight. To save the result, write the
adapter files yourself: `adapted._module.save_pretrained(dir)` is the
transformers method and writes the whole model, base weights included.

```python
import tempfile
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

for name, parameter in adapted.named_parameters():
    parameter.requires_grad_("lora_" in name)                 # train the adapter and nothing else
optimizer = torch.optim.Adam([p for p in adapted.parameters() if p.requires_grad], lr=1e-3)

losses = []
for _ in range(5):
    with adapted.trace(prompt):
        loss = -adapted.output.logits[0, -1].log_softmax(-1)[rome]
        with loss.backward():
            pass
        tracked = nnsight.save(loss.item())
    optimizer.step()
    optimizer.zero_grad()
    losses.append(tracked)
assert losses[-1] < losses[0]

export_dir = tempfile.mkdtemp()
state = get_peft_model_state_dict(adapted._module)
save_file({f"base_model.model.{key}": value.detach().contiguous() for key, value in state.items()},
          f"{export_dir}/adapter_model.safetensors")
adapted._module.peft_config["default"].save_pretrained(export_dir)

reloaded = TransformersModel("openai-community/gpt2", task="text-generation",
                             peft=export_dir, dispatch=True)
with torch.no_grad(), adapted.trace(prompt):
    trained_logits = adapted.output.logits.save()
with torch.no_grad(), reloaded.trace(prompt):
    reloaded_logits = reloaded.output.logits.save()
assert torch.equal(trained_logits, reloaded_logits)           # the export is the trained adapter
```

The exported directory also loads with `PeftModel.from_pretrained`. Switching the
adapter off and on keeps what was trained; removing it and loading it again
reads the weights from the id.
