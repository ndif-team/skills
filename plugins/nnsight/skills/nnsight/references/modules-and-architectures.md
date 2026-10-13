# Models, Module Paths, and Attaching Your Own Modules

The single most common way agent-written nnsight code fails is a **wrong module
path** — `model.transformer.h[i]` copied onto a Llama model, or `.output[0]` on
something that returns a tensor. This file is how you avoid that.

## Model classes

`TransformersModel` for anything from HuggingFace, `NNsight` for a module you
built, `DiffusionModel` for `diffusers`, `VLLM` for the vLLM engine — imports and
constructor in [api-reference.md](api-reference.md#model-classes).
`LanguageModel` is a deprecated alias that warns; write
`TransformersModel(repo_id, task="text-generation")`.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel, NNsight

model = TransformersModel("openai-community/gpt2", dispatch=True)
```

Constructor arguments worth knowing: `dispatch=False` (the default) builds on
`meta` with no weights until the first run; `attn_implementation="eager"` is
required to read attention probabilities; `task=` is inferred by asking the Hub,
so pass it when offline; plus `device=`, `device_map=`, `dtype=`, `revision=`,
`peft=<adapter repo id>`, `rename=` (below). Anything else HuggingFace accepts is
forwarded.

**Pick one device with `device=`, not `device_map=`.** The model loads through
`transformers.pipeline(...)`, which moves a model that sits on a single device to
its own `device`, defaulting to the first GPU. Measured on transformers 5.17 with
one GPU:

| Argument | Model ends up on |
|---|---|
| none | `cuda:0` (CPU only without an accelerator) |
| `device="cpu"` | `cpu` |
| `device="cuda"` / `device=0` | `cuda:0` |
| `device_map="cpu"` / `device_map={"": "cpu"}` | `cuda:0`, silently |
| `device_map="cuda"` / `device_map="auto"` | `cuda:0` |
| `device_map=` dict over two or more devices | as mapped |

`device_map=` is for spreading a model across devices (`"auto"` on several GPUs,
or a dict). `DiffusionModel` is the exception: diffusers honours
`device_map="cuda"` and ignores `device=`.

```python
cpu_model = TransformersModel("openai-community/gpt2", device="cpu", dispatch=True)
assert next(cpu_model.parameters()).device.type == "cpu"
```

`dtype=` also takes a quantization name (`"nf4"`, `"int8"`, ...) for a checkpoint
too big for the GPU you have; load the `quantization` skill before using one, since
three of the formats fail silently rather than raising.

## Finding module paths

**Run the inspector rather than guessing:**

```
python scripts/inspect_model.py meta-llama/Llama-3.1-8B --prompt "The capital of France is"
```

It builds the model on `meta` (no weights downloaded — a 27B model takes ~8s),
then prints the layer list path, each block child **in forward-pass order**, and
whether each `.output` is a tensor or a tuple. `--grep attn` filters paths;
`--depth 2` prints the tree; `--task` and `--trust-remote-code` pass through. It
prints the task and the class it built — the task decides the class, and the class
decides the tree. In code:

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

Verified on transformers 5.15 with `scripts/inspect_model.py`. Notes:

- **Multimodal checkpoints nest the LM** (Gemma-3: `model.language_model`).
  Always inspect a VLM before writing paths.
- **BERT blocks have a child module literally named `output`**, as does RWKV-4's
  `attention` — it moves to `.E_output`; see
  [api-reference.md](api-reference.md#envoy-properties-any-module).
- **A hybrid's block children vary by layer.** Falcon-H1 runs `mamba` *and*
  `self_attn` in one block; Jamba, Qwen3-Next and Qwen3.5 (`linear_attn`) alternate
  mixer types between layers. Inspect the layer you are about to touch.
- **A state-space model's recurrent state is not an activation.** It lives in a
  cache object the pure-torch fallback kernels update *in place*, so clone what you
  read. With `mamba_ssm` / `causal_conv1d` / `fla` installed the fast path takes
  over and the per-token intermediate handles disappear.

### MoE blocks

The sparse `.mlp` is a container; the pieces you intervene on are inside it.

| repo family | router | experts | shared expert | note |
|---|---|---|---|---|
| Mixtral, Qwen3-MoE, Qwen3-Next, OLMoE, DeepSeek-V3, GLM-4, Ernie-4.5 | `.mlp.gate` | `.mlp.experts` | `.mlp.shared_expert(s)` where present | DeepSeek-V3, GLM-4 and Ernie make the first layers dense |
| GPT-OSS, Phi-MoE, Jamba, Gemma-4 | `.mlp.router` | `.mlp.experts` | — | Gemma-4 hangs both off the decoder layer itself; GPT-OSS's block output is `(hidden, router_scores)` |

Two router traps that turn a plausible intervention into a no-op or a wrong index:

- **The router returns `(logits, weights, index)`, and the block uses `[1]` and
  `[2]`.** Masking `gate.output[0]` is a silent no-op on `TransformersModel` —
  softmax and top-k already ran. Write the routing **weights** (`[1]`) or the
  expert rows. (The `vllm` skill's `gate.output[0][:, e] = -inf` recipe is right
  only for vLLM, whose router returns `(logits, bias)`.)
- **Everything below the router is flat `(B*T, ...)`** — `gate.output`,
  `experts.output` and every `.source` op under them lose the batch axis, so the
  last token of row 0 is `[T-1]`, not `[:, -1]`. nnsight still scopes those rows
  per invoke; see [batching.md](batching.md#which-values-are-scoped-to-your-rows).

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

`inspect_model.py --prompt ...` prints this for any model, read off real forward
hooks.

## Tensor or tuple?

No rule holds across families — check per module, on your version. Measured on
transformers 5.15:

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
silently selects batch row 0 and nothing raises.

## Writing architecture-portable code

Install aliases at load with `rename=`, so one script works everywhere:

```python
aliased = TransformersModel("openai-community/gpt2", dispatch=True,
                            rename={"transformer.h": "layers"})

with aliased.trace("The Eiffel Tower is in the city of"):
    resid = aliased.layers[5].output.save()          # alias
    same = aliased.transformer.h[5].output.save()    # original still works

print(torch.equal(resid, same))
```

An alias points at the *same* envoy, so cache keys and iteration are unaffected.
For a maintained version of this idea across many architectures — one set of
names (`model.layers[i].self_attn`, `.mlp`, `model.norm`) and standard values
(`layer_output`, `attention_output`, `mlp_output`) on every family — see the
nnterp plugin's `nnterp:nnterp` skill.

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

`NNsight(net)` does not copy `net`; the first trace installs nnsight's controller
into `net`'s own submodules (`module.__dict__["forward"]`), and everything holding
a reference to `net` sees that for the rest of the process. So:

- **Finish the module before you wrap it.** `deepcopy`, `pickle` and
  `torch.compile` go first. `NNsight(net)` *then* `torch.compile(net)` works;
  wrapping a compiled `OptimizedModule` does not.
- **Do not replace a `forward` after wrapping.** Your function takes the
  controller's instance slot, so nnsight is silently off for that module and the
  next read fails far away as `OutOfOrderError: 'model.0.output.i0' was requested
  but the model already ran past it`. Patch `forward` before wrapping.
- **Mutate the tree through the envoy, not the module.** The envoy's children are
  built once: `net.append(...)` after wrapping leaves the envoy a child short, and
  `wrapped[1]` raises a bare `IndexError`. Assign through the envoy
  (`wrapped.layer = new_module`), or wrap again.

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

Calling the module and reading its internals in the same trace body raises
`OutOfOrderError`: your worker is *inside* the call when the submodule fires.

Both examples **replace** the output rather than writing into it. `output[:] =
probe(acts)` overwrites the attachment's own input, and a later `.backward()`
raises `RuntimeError: ... modified by an inplace operation` — which breaks
training the attachment.

## Adding your own served value (`eproperty`, `envoys=`)

`.input`, `.inputs` and `.output` are `eproperty` descriptors on `Envoy`, and you
can define more: a **preprocess** (what a read hands your block) and optionally a
**transform** (how an edit to that view maps back into the model's layout). That
is the supported way to give a module a reshaped, writable view — per head, per
expert, per channel. The `envoys=` constructor argument maps a module type or a
dotted path suffix to your `Envoy` subclass; everything not named stays the base
`Envoy`.

Per-head work has to happen where the heads are still separate: the **input** of
the attention's output projection (`attn.c_proj` on GPT-2, `o_proj` on Llama).
`attn.output` is already past that projection, so zeroing a 64-wide slice of it
mixes every head and ablates none.

```python
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.eproperty import eproperty

class Heads(Envoy):
    n_heads, head_dim = 12, 64                     # a Conv1D carries neither, so set them here

    @eproperty(key="input", description="c_proj input split per head")
    def heads(self, value):                        # read: (B, S, H) -> (B, nh, S, hd)
        (x,), _ = value                            # key="input" serves (args, kwargs)
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

    @heads.transform
    def heads(self, value, raw):                   # write: back to (args, kwargs)
        b, nh, s, hd = value.shape
        return ((value.transpose(1, 2).reshape(b, s, nh * hd),), {})

headed = TransformersModel("openai-community/gpt2", dispatch=True,
                           envoys={"attn.c_proj": Heads})   # not mlp.c_proj
prompt = "The Eiffel Tower is in the city of"
proj = headed.transformer.h[5].attn.c_proj

with headed.trace(prompt):
    per_head = proj.heads.clone().save()
    proj.heads[:, 7] = 0                                     # ablate head 7
    ablated = headed.output.logits.save()

with headed.trace(prompt):                                   # the same ablation by hand
    proj.input[..., 7 * 64:8 * 64] = 0
    by_hand = headed.output.logits.save()

with headed.trace(prompt):
    clean = headed.output.logits.save()

print(per_head.shape, type(proj).__name__, type(headed.transformer.h[5].mlp.c_proj).__name__)
assert torch.equal(ablated, by_hand)                         # the transform landed
assert not torch.equal(ablated, clean)                       # and it changed something
```

```
torch.Size([1, 12, 10, 64]) Heads Envoy
```

Four things that bite:

- **`self.<name>` inside a preprocess falls through to the module** when the
  class does not define it — `self.num_heads` works on an attention module. Nothing
  checks it exists: GPT-NeoX spells `head_dim` as `head_size`, and a projection like
  `c_proj` has neither, hence the class attributes above.
- **The transform must return a value shaped like the *location*.** It is
  called with the edited view and `raw`, the value as served before the
  preprocess. For `key="output"` on a tuple-returning module, index `value[0]` in
  the preprocess and rebuild the whole tuple from `raw` in the transform.
- **A preprocess that raises `AttributeError` is swallowed**, because an
  `eproperty` is a `property` and a raising getter falls through to
  `__getattr__`. You get `'Heads' object (nor its module) has attribute 'heads'`
  — which blames the name you asked for, not the line that failed. If a custom
  eproperty reports itself missing, the bug is inside its preprocess.
- **`key="input"` serves the raw `(args, kwargs)` pair**, not a bare tensor.
  Destructure it (`(x,), _ = value`) and repack it in the transform, as above.

Without `envoys=`, a custom `eproperty` has to live on a class that is already an
envoy — your `NNsight` subclass, or the tracer. That is how the `VLLM` wrapper
adds `model.logits` and `model.samples`.

## Related

- [access-and-modify.md](access-and-modify.md) — what to do once you have the path
- [control-flow.md](control-flow.md) — `edit` semantics
- [source-tracing.md](source-tracing.md) — values with no module to attach to
