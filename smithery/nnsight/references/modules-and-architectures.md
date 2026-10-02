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

Others worth knowing: `task=` (pipeline task, inferred if omitted — inference
asks the Hub, so pass it when offline), `device_map=`, `dtype=`, `revision=`,
`peft=<adapter repo id>`, `rename=` (below), and anything else HuggingFace
accepts — it is forwarded.

`dtype=` also takes a quantization name (`"nf4"`, `"int8"`, ...) for a checkpoint
too big for the GPU you have; load the `quantization` guide (`quantization.md`) before using one, since
three of the formats fail silently rather than raising.

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
| Mixtral / Qwen3-MoE / OLMoE / DeepSeek-V3 / GPT-OSS | `model.model.layers[i]` | `.self_attn` | `.mlp` (sparse — table below) | `model.embed_tokens` | `model.norm` | `lm_head` |
| Mamba / Mamba2 | `model.backbone.layers[i]` | `.mixer` (no attention) | — | `backbone.embeddings` | `backbone.norm_f` | `lm_head` |
| RWKV-4 | `model.rwkv.blocks[i]` | `.attention` (the time mixer) | `.feed_forward` | `rwkv.embeddings` | `rwkv.ln_out` | `head` |
| Hybrids: Falcon-H1, Jamba, Qwen3-Next, Qwen3.5, Nemotron-H | `model.model.layers[i]` | `.self_attn`, `.mamba`, `.linear_attn` — *per layer* | `.mlp` / `.feed_forward` | `model.embed_tokens` | `model.norm` | `lm_head` |

Verified on transformers 5.15 with `scripts/inspect_model.py`.

Four notes from that table:

- **Multimodal checkpoints nest the LM.** Gemma-3's text stack is under
  `model.language_model`, not at the root. Always inspect a VLM before writing
  paths.
- **BERT blocks have a child module literally named `output`.** nnsight's `.output`
  wins, so the child moves to `.E_output` on that module (it warns at load). Its
  path is unchanged, so `named_modules()` still lists it as `...attention.output`.
  RWKV-4 does the same at `rwkv.blocks[i].attention.output`, and warns once per
  block on load.
- **A hybrid's block children vary by layer.** Falcon-H1 runs `mamba` *and*
  `self_attn` in the same block; Jamba, Qwen3-Next and Qwen3.5 alternate mixer
  types between layers, and Qwen3.5's is `linear_attn`. Never assume the child
  names you found on layer 0 exist on layer 20 — inspect the layer you are about
  to touch.
- **A state-space model's recurrent state is not an activation.** It lives in a
  cache object threaded through the layers, and the pure-torch fallback kernels
  update it *in place*, so a read taken before the next module fires and one taken
  after give different numbers. Clone what you read. The per-token state exists
  only in those fallbacks: with `mamba_ssm` / `causal_conv1d` / `fla` installed the
  fast path takes over and the intermediate handles disappear.

### MoE blocks

The sparse `.mlp` is a container; the pieces you intervene on are inside it.

| repo family | router | experts | shared expert | note |
|---|---|---|---|---|
| Mixtral, Qwen3-MoE, Qwen3-Next, OLMoE, DeepSeek-V3, GLM-4, Ernie-4.5 | `.mlp.gate` | `.mlp.experts` | `.mlp.shared_expert(s)` where present | DeepSeek-V3, GLM-4 and Ernie make the first layers dense |
| GPT-OSS, Phi-MoE, Jamba, Gemma-4 | `.mlp.router` | `.mlp.experts` | — | Gemma-4 hangs both off the decoder layer itself; GPT-OSS's block output is `(hidden, router_scores)` |

Two things about the router's value, both of which turn a plausible-looking
intervention into a no-op or a wrong index:

- **The router returns `(logits, weights, index)`, and the block uses `[1]` and
  `[2]`.** Masking `gate.output[0]` is a silent no-op on `TransformersModel` —
  the logits have already been through softmax and top-k by then. Write the
  routing **weights** (`[1]`) or the expert rows instead. (The `vllm` guide (`vllm.md`)'s
  `mlp.gate.output[0][:, e] = -inf` recipe is correct **for vLLM**, whose router
  returns `(logits, bias)`; it does nothing here.)
- **Everything below the router is flat `(B*T, ...)`.** The block reshapes with
  `hidden_states.view(-1, hidden_dim)` before routing, so `gate.output`,
  `experts.output` and every `.source` op under them lose the batch axis. There is
  no `[:, -1]` to take: the last token of the batch's first row is `[T-1]`, and
  inside an invoke you index the invoke's own `(rows*seq, ...)` block. nnsight
  scopes those rows — a leading dim that is a whole multiple of the batch size is
  narrowed `seq` rows per row of batch — so per-invoke reads and writes are correct;
  see [batching.md](batching.md#which-values-are-scoped-to-your-rows) for the rule
  and for the warning you get when a layout it cannot read is written to.

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

There is no rule that holds across families — check, per module, on the version
you are running. Measured on transformers 5.15:

- a decoder **block**'s `.output` is a plain `Tensor (batch, seq, hidden)` on
  GPT-2, Llama, GPT-NeoX, Mamba and MoE families; a **1-tuple** on Falcon-H1; a
  **3-tuple** `(hidden, None, None)` on RWKV-4; and `(hidden, router_scores)` on a
  GPT-OSS sparse block
- an **attention** submodule's `.output` is a `tuple(Tensor, ...)` — often with
  `None` in second place unless `attn_implementation="eager"`
- an **MLP**'s `.output` is a plain `Tensor`, and so is a Mamba `.mixer`'s, but
  RWKV-4's `.attention` and `.feed_forward` are both tuples
- an MoE **router**'s `.output` is a 3-tuple `(logits, weights, index)`

Do not port `.output[0]` from old examples without checking; on a tensor it
silently selects batch row 0 and nothing raises. `inspect_model.py --prompt ...`
prints the type of every block child, which is faster than reading modeling code.

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
`layers_output[i]`, `attentions[i]`, `mlps[i]` and model validation — see the `nnterp` guide (`nnterp.md`).

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
`_batch_size` / `_batch`. Empty invokes always work. It also has no `.scan()` —
that belongs to the wrappers that build a model from a repo id.

### Wrapping changes the module, and the change sticks

`NNsight(net)` does not copy `net`; `wrapped._module is net`, and the first trace
installs nnsight's controller into `net`'s own submodules
(`module.__dict__["forward"]`). Everything else holding a reference to `net` sees
that, in traces and outside them, for the rest of the process. Three consequences
worth planning around:

- **Finish the module before you wrap it.** `deepcopy`, `pickle` and
  `torch.compile` all have to deal with the installed controller, and the safe
  order is to do them first and wrap the result. `NNsight(net)` *then*
  `torch.compile(net)` works; compiling first and wrapping the `OptimizedModule`
  does not.
- **Do not replace a `forward` after wrapping.** Your function takes the same
  instance slot the controller is in, so nnsight is silently switched off for that
  module — no warning, and the next read of it fails far away as
  `OutOfOrderError: 'model.0.output.i0' was requested but the model already ran
  past it`. Patch `forward` before wrapping, or patch it through a trace.
- **Mutate the tree through the envoy, not the module.** The envoy's children are
  built once. `net.append(nn.Linear(4, 2))` after wrapping leaves the torch module
  with 2 children and the envoy with 1, and `wrapped[1]` raises `IndexError: list
  index out of range` rather than saying anything about the tree. Assign through
  the envoy (`wrapped.layer = new_module`), or wrap again.

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
    model.transformer.h[6].output = model.transformer.h[6].probe(acts)
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
    model.transformer.h[6].output = model.transformer.h[6].probe(acts, hook=True)

with model.trace("The Eiffel Tower is in the city of"):
    inner = model.transformer.h[6].probe.up.output.save()    # observable
    logits = model.output.logits.save()

print(inner.shape)
model.clear_edits()
```

Trying to call the module and read its internals in the same trace body raises
`OutOfOrderError`: your worker is *inside* the call when the submodule fires, so
it can never be parked waiting for it.

Both examples **replace** the output rather than writing into it. `output[:] =
probe(acts)` writes the attachment's result into the tensor that was its own
input, and a later `.backward()` raises `RuntimeError: one of the variables
needed for gradient computation has been modified by an inplace operation` —
which breaks training the attachment, the main reason to insert one.

This is the mechanism behind SAE analysis, LoRA/adapter interpretability, and
trained-probe insertion.

## Adding your own served value (`eproperty`, `envoys=`)

`.input`, `.inputs` and `.output` are not special-cased in the tracer — they are
`eproperty` descriptors on `Envoy`, and you can define more. An `eproperty` is a
**preprocess** (what a read of that location hands your block) and, optionally, a
**transform** (how an edit to that view is mapped back into the model's layout,
spliced in once the block is done with the read). That is the supported way to
give a module a reshaped, writable view — per head, per expert, per channel —
instead of reshaping by hand at every call site and hoping the write lands.

A custom `Envoy` subclass reaches a **chosen** module through the `envoys=`
constructor argument, which maps a module type or a dotted path suffix to the
subclass. Everything not named stays the base `Envoy`:

```python
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.eproperty import eproperty
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

class Heads(Envoy):
    @eproperty(key="output", description="attention output split per head")
    def heads(self, value):                        # read: (B, S, H) -> (B, nh, S, hd)
        hidden = value[0]                          # GPT-2 attention returns a tuple
        b, s, _ = hidden.shape
        return hidden.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    @heads.transform
    def heads(self, value):                        # write: back to the model's layout
        b, nh, s, hd = value.shape
        return (value.transpose(1, 2).reshape(b, s, nh * hd), None)

headed = TransformersModel("openai-community/gpt2", dispatch=True,
                           envoys={GPT2Attention: Heads})   # or {"attn": Heads} by suffix

with headed.trace("The Eiffel Tower is in the city of"):
    per_head = headed.transformer.h[5].attn.heads.clone().save()
    headed.transformer.h[5].attn.heads[:, 7] = 0             # zero head 7
    after = headed.transformer.h[5].attn.output[0].save()

print(per_head.shape, type(headed.transformer.h[5].attn).__name__,
      type(headed.transformer.h[5].mlp).__name__)
assert after.view(1, -1, 12, 64)[:, :, 7].abs().max() == 0   # the transform landed
assert after.view(1, -1, 12, 64)[:, :, 6].abs().max() > 0
```

```
torch.Size([1, 12, 10, 64]) Heads Envoy
```

Four things that bite:

- **`self.<name>` inside a preprocess falls through to the module**, which is how
  `self.num_heads` and `self.head_dim` resolve above. Nothing checks they exist —
  GPT-NeoX spells the second one `head_size`.
- **The transform must return a value shaped like the *location*.** The read
  above indexes `value[0]`, so the write has to hand back the whole tuple.
- **A preprocess that raises `AttributeError` is swallowed**, because an
  `eproperty` is a `property` and a raising getter falls through to
  `__getattr__`. You get `'Heads' object (nor its module) has attribute 'heads'`
  — which blames the name you asked for, not the line that failed. If a custom
  eproperty reports itself missing, the bug is inside its preprocess.
- **`key="input"` serves the raw `(args, kwargs)` pair**, not a bare tensor.
  Destructure it (`(x,), _ = value`) and repack it in the transform.

Without `envoys=`, a custom `eproperty` has to live on a class that is already an
envoy — your `NNsight` subclass, or the tracer. That is how the `VLLM` wrapper
adds `model.logits` and `model.samples`.

## Related

- [access-and-modify.md](access-and-modify.md) — what to do once you have the path
- [control-flow.md](control-flow.md) — `edit` semantics
- [source-tracing.md](source-tracing.md) — values with no module to attach to
