# Models, Module Paths, and Attaching Your Own Modules

The single most common way agent-written nnsight code fails is a **wrong module
path** — `model.transformer.h[i]` copied onto a Llama model, or `.output[0]` on
something that returns a tensor. This file is how you avoid that.

## Model classes

| Class | Use for | Import |
|---|---|---|
| `TransformersModel` | **anything from HuggingFace** — text, vision, audio, multimodal | `from nnsight import TransformersModel` |
| `NNsight` | any `torch.nn.Module` you built yourself | `from nnsight import NNsight` |
| `DiffusionModel` | `diffusers` pipelines | `from nnsight import DiffusionModel` |
| `VLLM` | high-throughput / tensor-parallel serving | `from nnsight.modeling.vllm import VLLM` |

`LanguageModel` and `VisionLanguageModel` still exist but **warn on
construction** — they are deprecated aliases for `TransformersModel`. Use
`TransformersModel(repo_id, task="text-generation")` in new code.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel, NNsight

model = TransformersModel("openai-community/gpt2", dispatch=True)
```

Useful constructor arguments:

```python
# dispatch=False (default) builds on `meta` — no weights until the first run.
lazy = TransformersModel("openai-community/gpt2")
print(lazy.dispatched)                       # False

# attn_implementation="eager" is required to read attention probabilities
eager = TransformersModel("openai-community/gpt2", dispatch=True,
                          attn_implementation="eager")
print(eager.config.model_type)
```

Others worth knowing: `task=` (pipeline task, inferred if omitted), `device_map=`,
`torch_dtype=`, `revision=`, `peft=<adapter repo id>`, `rename=` (below), and
anything else HuggingFace accepts — it is forwarded.

## Finding module paths

**Run the inspector rather than guessing:**

```
python scripts/inspect_model.py meta-llama/Llama-3.1-8B --prompt "The capital of France is"
```

It builds the model on `meta` (no weights downloaded — a 27B model takes ~8s),
then prints the layer list path, each block child **in forward-pass order**, and
whether each `.output` is a tensor or a tuple. `--grep attn` filters paths;
`--depth 2` prints the tree.

In code, the equivalent moves are:

```python
print(model)                                  # the torch module tree
print([path for path, _ in model.named_modules()][:5])
print(model.get("transformer.h.0.mlp").path)  # fetch by dotted path
```

## Path reference (verified)

| Family | Layers | Attention | MLP | Embeddings | Final norm | Unembed |
|---|---|---|---|---|---|---|
| GPT-2 | `model.transformer.h[i]` | `.attn` | `.mlp` | `transformer.wte` | `transformer.ln_f` | `lm_head` |
| Llama / Mistral / Qwen / SmolLM | `model.model.layers[i]` | `.self_attn` | `.mlp` | `model.embed_tokens` | `model.norm` | `lm_head` |
| GPT-NeoX / Pythia | `model.gpt_neox.layers[i]` | `.attention` | `.mlp` | `gpt_neox.embed_in` | `gpt_neox.final_layer_norm` | `embed_out` |
| Gemma-3 (multimodal) | `model.model.language_model.layers[i]` | `.self_attn` | `.mlp` | — | — | `lm_head` |
| BERT | `model.bert.encoder.layer[i]` | `.attention` | `.intermediate` | `bert.embeddings` | — | — |

Two notes from that table:

- **Multimodal checkpoints nest the LM.** Gemma-3's text stack is under
  `model.language_model`, not at the root. Always inspect a VLM before writing
  paths.
- **BERT blocks have a child module literally named `output`.** The child wins, so
  nnsight's property moves to `.nns_output` on that module (it warns at load).

## Block internals run in a fixed order

Registration order (what `print(model)` shows) is **not** execution order. On
Llama, `self_attn` is registered first but `input_layernorm` runs first. Access
them in execution order or get `OutOfOrderError`:

```
gpt2       ln_1  ->  attn  ->  ln_2  ->  mlp  ->  block output
llama      input_layernorm -> self_attn -> post_attention_layernorm -> mlp -> block
gpt-neox   input_layernorm -> attention -> post_attention_dropout ->
           post_attention_layernorm -> mlp -> post_mlp_dropout -> block
```

`inspect_model.py --prompt ...` prints this for any model, because it reads it off
real forward hooks rather than assuming.

## Tensor or tuple?

Across the families above, in current `transformers`:

- a **block**'s `.output` is a plain `Tensor (batch, seq, hidden)`
- an **attention** submodule's `.output` is a `tuple(Tensor, ...)` — often with
  `None` in second place unless `attn_implementation="eager"`
- an **MLP**'s `.output` is a plain `Tensor`

Do not port `.output[0]` from old examples without checking; on a tensor it
silently selects batch row 0.

## Writing architecture-portable code

Read the paths off the config once, then use variables:

```python
def layer_list(model):
    """The block ModuleList for common decoder-only families."""
    for path in ("transformer.h", "model.layers", "gpt_neox.layers",
                 "model.language_model.layers"):
        try:
            return model.get(path)
        except Exception:
            continue
    raise ValueError("unknown architecture — run inspect_model.py")

layers = layer_list(model)
with model.trace("The Eiffel Tower is in the city of"):
    mid = layers[len(layers) // 2].output[0, -1].save()

print(len(layers), mid.shape)
```

Or install aliases at load with `rename=`, so one script works everywhere:

```python
aliased = TransformersModel("openai-community/gpt2", dispatch=True,
                            rename={"transformer.h": "layers"})

with aliased.trace("The Eiffel Tower is in the city of"):
    resid = aliased.layers[5].output.save()          # alias
    same = aliased.transformer.h[5].output.save()    # original still works

print(torch.equal(resid, same))
```

An alias points at the *same* envoy, so cache keys and iteration are unaffected.
For a maintained version of this idea across many architectures — with
`layers_output[i]`, `attentions[i]`, `mlps[i]` and model validation — see the
`nnterp` skill.

## Any PyTorch module

`NNsight` wraps anything; the tree mirrors your module names.

```python
net = torch.nn.Sequential(
    torch.nn.Linear(8, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 2),
)
wrapped = NNsight(net)

with wrapped.trace(torch.randn(4, 8)):
    hidden = wrapped[0].output.save()
    wrapped[2].output[:] = 0
    out = wrapped.output.save()

print(hidden.shape, out.abs().sum().item())
```

Base `NNsight` supports one input invoke; batching several requires implementing
`_batch_size` / `_batch`. Empty invokes always work.

## Attaching your own module (SAE, probe, adapter)

Assign it into the tree, then route activations through it:

```python
class TinyProbe(torch.nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.up = torch.nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.up(x)

model.transformer.h[6].probe = TinyProbe(768).to(model.device)   # in the tree now

with model.trace("The Eiffel Tower is in the city of"):
    acts = model.transformer.h[6].output
    model.transformer.h[6].output[:] = model.transformer.h[6].probe(acts)
    logits = model.output.logits.save()

print(logits.shape)
```

To make the attached module's **own internals** observable, the routing has to
live in an `edit` and the read in the trace — the routing call and the read must
be different workers, or the read is out of order. Pass `hook=True` so the
module's full `__call__` runs and its submodules fire:

```python
with model.edit(inplace=True):
    acts = model.transformer.h[6].output
    model.transformer.h[6].output[:] = model.transformer.h[6].probe(acts, hook=True)

with model.trace("The Eiffel Tower is in the city of"):
    inner = model.transformer.h[6].probe.up.output.save()    # observable
    logits = model.output.logits.save()

print(inner.shape)
model.clear_edits()
```

Trying to call the module and read its internals in the same trace body raises
`OutOfOrderError`: your worker is *inside* the call when the submodule fires, so
it can never be parked waiting for it.

This is the mechanism behind SAE analysis, LoRA/adapter interpretability, and
trained-probe insertion.

## Related

- [access-and-modify.md](access-and-modify.md) — what to do once you have the path
- [control-flow.md](control-flow.md) — `edit` semantics
- [source-tracing.md](source-tracing.md) — values with no module to attach to
