# Models, Module Paths, and Your Own Modules

```python
import torch
import nnsight
from nnsight import NNsight, TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
```

## Loading

`TransformersModel(repo_id, *, task=None, dispatch=False, revision=None,
peft=None, rename=None, envoys=None, **hf_kwargs)`. [SKILL.md](../SKILL.md)
covers `task`, `dispatch`, `device` and `dtype`. The rest:

- `dtype=` is the keyword; `torch_dtype=` is deprecated and warns. A quantization
  name (`"nf4"`, `"int8"`) goes in the same slot: [quantization](quantization.md).
- `attn_implementation="eager"` is required to read attention probabilities.
- `peft=<adapter id or path>` attaches a LoRA adapter:
  [model-editing-and-lora](model-editing-and-lora.md).
- `trust_remote_code=True` and any other HuggingFace keyword are passed through.
- The task decides the class and the class decides the tree. A multimodal
  checkpoint loaded with `task="text-generation"` is still the multimodal class.
- A model already loaded with transformers goes in place of the repo id:
  `TransformersModel(hf_model, tokenizer=tokenizer)`. The task is inferred from
  its class.

**One device: `device=`. Several: `device_map=`.** The model loads through
`transformers.pipeline`, which moves a single-device model to its own default:

| Argument | Model ends up on |
|---|---|
| none | `cuda:0` (CPU when there is no GPU) |
| `device="cpu"` | `cpu` |
| `device="cuda"` / `device=0` | `cuda:0` |
| `device_map="cpu"` / `device_map={"": "cpu"}` | `cuda:0`, silently |
| `device_map="auto"` | `cuda:0` on one GPU, spread across several |

`DiffusionModel` is the exception: diffusers honours `device_map="cuda"` and
ignores `device=`.

## Finding module paths

Run the inspector. It builds the model without weights and prints the layer
list, each block child in forward-pass order, and each output's type:

```
python scripts/inspect_model.py HuggingFaceTB/SmolLM2-135M-Instruct --task text-generation --prompt "The capital of France is"

layers        model.model.layers[i]          (30 blocks)
embeddings    model.model.embed_tokens
final norm    model.model.norm
unembed       model.lm_head
  model.model.layers[0].input_layernorm           Tensor(1, 5, 576)
  model.model.layers[0].self_attn                 tuple(Tensor(1, 5, 576), NoneType)
  model.model.layers[0].post_attention_layernorm  Tensor(1, 5, 576)
  model.model.layers[0].mlp                       Tensor(1, 5, 576)
  model.model.layers[0]                           Tensor(1, 5, 576)
```

`--grep attn` filters paths, `--depth 2` prints the tree, `--trust-remote-code`
passes through. In code: `print(model)`, `model.named_modules()` (absolute paths
such as `model.transformer.h.0`), and `model.get("transformer.h.0.mlp")` to fetch
by path.

What the inspector reports for common families:

| Family | Layers | Attention / mixer | Final norm | Unembed |
|---|---|---|---|---|
| GPT-2 | `model.transformer.h[i]` | `.attn` | `transformer.ln_f` | `lm_head` |
| Llama, Mistral, Qwen, SmolLM, Gemma text-only | `model.model.layers[i]` | `.self_attn` | `model.norm` | `lm_head` |
| GPT-NeoX / Pythia | `model.gpt_neox.layers[i]` | `.attention` | `gpt_neox.final_layer_norm` | `lm_head` |
| Gemma-3 multimodal (4b and larger) | `model.model.language_model.layers[i]` | `.self_attn` | inspect | `lm_head` |
| BERT | `model.bert.encoder.layer[i]` | `.attention` | none | task head |
| Mamba2 | `model.backbone.layers[i]` | `.mixer` (no attention) | inspect | `lm_head` |

- **Registration order is not execution order.** `print(model)` lists a Llama
  block's `self_attn` first, and `input_layernorm` runs first. Read in the
  inspector's order.
- **Tensor or tuple is per module and per family.** A decoder block returns a
  plain tensor on the families above; attention modules return a tuple whose
  second element is `None` unless the model was loaded eager. Other families
  return tuples from blocks or MLPs, so check before indexing.
- **A hybrid's block children vary by layer** (attention on one layer, a
  state-space or linear-attention mixer on the next). Inspect the layer you touch.

### MoE blocks

The sparse `.mlp` is a container: a router (`.mlp.gate` or `.mlp.router`) and
`.mlp.experts`. On OLMoE the inspector prints

```
  model.model.layers[0].mlp.gate     tuple(Tensor(5, 64), Tensor(5, 8), Tensor(5, 8))
  model.model.layers[0].mlp.experts  Tensor(5, 2048)
```

- **The router returns `(logits, weights, index)` and the block uses the last
  two.** Masking `gate.output[0]` is a silent no-op: softmax and top-k already
  ran. Write the routing weights (`[1]`) or the expert rows. The vLLM engine's
  router is laid out differently:
  [vllm-throughput-and-architectures](vllm-throughput-and-architectures.md).
- **Everything below the router is flat `(batch * tokens, ...)`.** The last token
  of row 0 is `[T - 1]`, not `[:, -1]`. Invokes still scope these rows
  ([batching](batching.md#which-values-are-scoped-to-an-invoke)).
- An expert that was not routed to never runs; reading it is `OutOfOrderError`.

## Aliasing paths with `rename`

```python
aliased = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True,
                            rename={"transformer.h": "layers", "mlp": ["ffn", "feed_forward"]})

with aliased.trace(prompt):
    a = aliased.layers[5].ffn.output.save()
    b = aliased.transformer.h[5].mlp.output.save()     # original paths still work

assert torch.equal(a, b)
```

A dotted key mounts that subtree on the root under the alias. A single-component
key (`"mlp"`) binds wherever a module has that child. `*` stands for one path
component (`{"*.h": "layers"}` works whatever the parent is called), and a module
class as the key aliases the child of that type on every module that has one. A
value may be a list of aliases. Aliases are extra names for the same envoy: `named_modules()` and cache
path keys keep the original names. For one maintained vocabulary across families
(`model.layers[i].self_attn`, `.mlp`, standard values), use the `nnterp` skill.

## Wrapping your own module

`NNsight(module)` wraps any `torch.nn.Module`. Inputs are whatever its `forward`
takes; the tree mirrors its attribute names, and `Sequential` children are
indexed.

```python
net = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
wrapped = NNsight(net)

with wrapped.trace(torch.randn(4, 10)):
    acts = wrapped[1].output.clone().save()      # after the ReLU
    wrapped[1].output[:, :16] = 0                # zero half of them
    out = wrapped.output.save()

assert tuple(acts.shape) == (4, 32) and tuple(out.shape) == (4, 2)
```

- It takes a module, not a repo id, and has no `dispatch`, `device` or `dtype`:
  move the module yourself, or call `wrapped.to(device)`.
- No `.scan()`, `.generate()` or tokenizer. Gradients, `edit`, `session`, `skip`,
  `cache` and `.source` work as on any model.
- One input per trace. A second input invoke raises `NotImplementedError: NNsight
  does not support batching multiple invokes`; pass a batched tensor, or use
  empty invokes for several orderings over the same input.
- A module called more than once per forward:
  [access-and-modify](access-and-modify.md#a-module-that-runs-more-than-once).
- **Wrapping changes the module itself, for the rest of the process.** nnsight
  installs itself on the module's submodules, not on a copy, so everything that
  holds the module sees it. Do not assign a new `forward` to a submodule
  afterwards: nnsight is silently off for it and the next read is an
  `OutOfOrderError`.

## Attaching a module

Assign a module into the tree and route an activation through it. This is how an
SAE, a probe or an adapter is added:

```python
probe = torch.nn.Sequential(torch.nn.Linear(768, 768), torch.nn.Tanh()).to(model.device)
model.transformer.h[6].probe = probe

with model.trace(prompt):
    block = model.transformer.h[6]
    block.output = block.probe(block.output)          # replace, do not write into
    logits = model.output.logits.save()
```

To read the attached module's **own internals**, the routing call and the read
must be different workers: put the call in an `edit` with `hook=True`, and read
in a trace.

```python
with model.edit(inplace=True):
    block = model.transformer.h[6]
    block.output = block.probe(block.output, hook=True)

with model.trace(prompt):
    inner = model.transformer.h[6].probe[0].output.save()      # the Linear inside the probe
model.clear_edits()

assert tuple(inner.shape) == (1, 10, 768)
```

- Without `hook=True` the later read raises `OutOfOrderError: '...probe.0.output.i0'
  was requested but the model already ran past it`, though the submodule never
  ran. On an attached module, read that message as "is it hooked?".
- Calling the module and reading its internals in one trace body is the same
  error: the worker is inside the call when the submodule fires.
- A module you call yourself appears in `tracer.cache()` only with `hook=True`.
- Calling it in the edit and discarding the result observes without changing the
  forward. Assigning the result to `.output` splices it in.

## Custom values: `eproperty` and `envoys=`

`.output`, `.input` and `.inputs` are `eproperty` descriptors, and you can define
more: a reshaped, writable view of a location. The getter receives what the
location serves; an optional `transform` maps an edited view back into the
location's layout. `envoys=` maps a module type or a path suffix to your `Envoy`
subclass.

```python
from nnsight.intervention.envoy import Envoy
from nnsight.intervention.eproperty import eproperty

class Heads(Envoy):
    n_heads, head_dim = 12, 64

    @eproperty(key="input", description="c_proj input split per head")
    def heads(self, value):                        # read: (B, S, H) -> (B, heads, S, head_dim)
        (x,), _ = value                            # key="input" serves (args, kwargs)
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)

    @heads.transform
    def heads(self, value, raw):                   # write: back to (args, kwargs)
        b, nh, s, hd = value.shape
        return ((value.transpose(1, 2).reshape(b, s, nh * hd),), {})

headed = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True,
                           envoys={"attn.c_proj": Heads})      # not mlp.c_proj
proj = headed.transformer.h[5].attn.c_proj

with headed.trace(prompt):
    per_head = proj.heads.clone().save()
    proj.heads[:, 7] = 0                                       # ablate head 7
    ablated = headed.output.logits.save()

with headed.trace(prompt):
    proj.input[..., 7 * 64:8 * 64] = 0                         # the same ablation by hand
    by_hand = headed.output.logits.save()

assert tuple(per_head.shape) == (1, 12, 10, 64) and torch.equal(ablated, by_hand)
```

- `key="input"` serves the `(args, kwargs)` pair; `key="output"` serves the
  return value. The transform returns a value shaped like the **location** (the
  whole pair, or the whole tuple rebuilt from `raw`), not like the view.
- `self.<name>` inside the getter falls through to the wrapped module when the
  class does not define it (`self.num_heads` on an attention module). Nothing
  checks that it exists.
- **A getter that raises `AttributeError` is swallowed**, because a failing
  property falls through to `__getattr__`. The error becomes `'Heads' object (nor
  its module) has attribute 'heads'`, blaming the name you asked for. When a
  custom value reports itself missing, the bug is inside its getter.
- Without `envoys=`, an `eproperty` lives on a class that is already an envoy:
  your `NNsight` subclass. Subclassing `NNsight` and implementing `_batch_size`
  and `_batch` is also how a custom model gains multi-invoke batching.
