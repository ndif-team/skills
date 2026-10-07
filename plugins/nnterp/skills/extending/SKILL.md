---
name: extending
description: Extend nnterp when a checkpoint or value is outside what it ships. Load for UnsupportedFamily on a model_type; a family whose attention_output, mlp_output or pattern the base classes locate wrongly (sandwich norms, residual added inside, own attention arithmetic, scaled residual); adding your own value to a block, attention or MLP; finding the .source op names an EProperty path needs; registering a family or overriding a shipped one; a root size or logit step the family spells its own way (`def <size>(model)`, `def project_on_vocab(model, hidden)`); MoE, recurrent-mixer or vision envoys in a new family; a transformers upgrade that moved op names. Covers the family module (RENAME incl. class keys, ENVOYS, Moe/RecurrentMixer/Vision subclasses, ImageScatter, handing values to children), EProperty paths/select/unavailable/sourced, DerivedEProperty, TokenEProperty, StandardizedProperty/Capability, per-module sizes, layouts, envoys=, register() and FamilySuite. To use the standard values, load the nnterp skill.
---

# Extending nnterp

nnterp gives every transformer family one vocabulary (`model.layers[i].self_attn`, `.mlp`,
`.linear_attn`, `model.norm`, ...) and standard values (`layer_output`,
`attention_output`, `attention_probabilities`, `router_logits`, ...) through one module
per family under `nnterp/families/`, named after `config.model_type`
(`len(nnterp.families.known())` is 98). This skill is for when that is not enough: a
`model_type` nnterp does not ship, a forward that puts a value where the base classes do
not look, or a value nnterp does not have. Verified on nnsight 0.8.0 / transformers
5.17.0 against `openai-community/gpt2` and `HuggingFaceTB/SmolLM2-135M-Instruct`. The
nnterp repo's pages: `docs/extending/` (`adding-a-family.md`, `overriding-values.md`,
`custom-values.md`, `finding-source-ops.md`, `registering.md`),
`docs/developing/eproperty-internals.md`, `docs/developing/recurrent-mixer-internals.md`
and `docs/reference/families.md` (every family's overrides).

## What breaks

1. **Operation names are version-bound.** A release that renames a local or adds a
   binding before an op renames it; `print(envoy.source)` on the version you run.
2. **Import order.** `import nnterp` (or `nnsight`) before any
   `from transformers.models... import`; the reverse segfaults at import on this stack.
3. **`hasattr(envoy, value)` never answers `False`.** Unavailable: it raises
   `nnterp.Unavailable`. Available, outside a trace: a value at the module boundary or
   one `source` level deep raises nnsight's `ValueError: Cannot access ... outside of
   interleaving`; one inside a call (`attention_probabilities`) raises
   `SourceNotAvailable: recursive .source is only available inside a trace`. Use `support()`.
4. **An out-of-order read inside a call raises `OutOfOrderError` naming the call's
   `.fn`**, not the value: `layer_output` then `attention_probabilities` fails with
   `'...attn.source.attention_interface_1.fn.i0' was requested but the model already
   ran past it`. Read a module's interior values before its boundary values.
5. **After `../`, native names only.** `EProperty("../ln_2.output")` on a GPT-2 attention
   works; `"../post_attention_layernorm.output"` builds a location nothing serves and
   the trace fails with `OutOfOrderError` on `...h.0.post_attention_layernorm.output.i0`.
   A child segment below the host (`"embed_tokens.output"`) does take aliases.
6. **`envoys=` matches by module type or native path, never by alias, and type keys
   win.** `envoys={"attn": X}` on GPT-2 loses to the family's `GPT2Attention` key.
7. **A root size or `project_on_vocab` override is a module-level function.** The root
   looks up `model.family.<name>`; a `head_dim` on the family's `Attention` class is the
   per-module size, which the root never reads.
8. **A child never finds its parent through `interleaver.envoys`.** The family's
   `Layer.__init__` hands down what the child lacks (`self.mlp.router = self.router`).
9. **`inputs` serves `(args, kwargs)`; `input` the first argument.** `select=` names one
   element of `inputs` (int: positional, str: keyword) and repacks a write.
10. **`register()` is process-wide, silent, and read at load.** A model built before it
    keeps the shipped family; undo with `del families.REGISTRY[model_type]`. With no
    types given it registers the one the family's `__name__` ends in, so a variant named
    `my_gpt2` covers `my_gpt2`, not `gpt2`: name the types, `register(variant, "gpt2")`.

<!-- test: setup -->
```python
import torch
import nnterp
from nnterp import StandardizedTransformer, families
from nnterp.families import gpt2

model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"

assert model.family is gpt2
print(type(model.layers[0]).__name__, type(model.layers[0].self_attn).__name__, type(model.layers[0].mlp).__name__)
# Layer Attention Mlp
```

## The shape of a family module

One file, `nnterp/families/<model_type>.py`; the file name is the registry: `lookup`
imports the module named after the type, and the module declares no list of types. A
Llama-like family changes nothing but the container names:

<!-- test: skip nocompile -->
```python
"""<Family> (``<X>ForCausalLM``). <Where the residual is added; interface or own arithmetic; what returns a tuple.>"""
from transformers.models.<module>.modeling_<module> import <X>Attention, <X>DecoderLayer, <X>MLP
from ..components import Attention, Layer, Mlp   # + Moe, LinearAttention, StateSpace, ... as needed

RENAME = {
    "model.embed_tokens": "embed_tokens",         # multi-component keys bind at the root
    "model.layers": "layers",
    "model.norm": "norm",
    # "attn": "self_attn",                        # single-component: every envoy with that child
    # <X>Mamba2Mixer: "linear_attn",              # class key: every envoy with exactly one child of that class
}

class Layer(Layer):
    """<Family>'s decoder block; returns a bare tensor, so the base holds."""   # returns_tuple = True otherwise

class Attention(Attention):
    """<Family>'s attention; the shared eager forward and the residual added in the block, so the base holds."""

class Mlp(Mlp):
    """<Family>'s MLP; the residual is added in the block, so the base holds."""

ENVOYS = {<X>DecoderLayer: Layer, <X>Attention: Attention, <X>MLP: Mlp}
# def intermediate_size(model): return model.config.<key>              # only where the config spells a size its own way
```

- **`RENAME`**: a key that resolves nowhere is skipped; an alias that would shadow an
  existing name raises at construction; a class key is for one native name with several
  meanings (Nemotron-H's `mixer` is `linear_attn`, `self_attn` or `mlp` by class), and a
  class two children of one envoy share raises `ValueError` there.
- **`Layer`, `Attention`, `Mlp`** exist even when nothing changes (the suite asserts the
  block envoys are exactly the family's classes). Override `off_interface()` for a config
  flag that routes the attention off transformers' interface (GPT-2); redefine values for
  own arithmetic (GPT-J, BLOOM, MPT, Falcon), a residual added inside (BLOOM, MPT), a
  post-norm (Gemma-2/3/4, OLMo-2/3) or a scaled add (Granite's `residual_multiplier`).
- **A mixture of experts** is an `nnterp.components.Moe` keyed on the MoE module class:
  `class Mlp(Moe)` where every MLP is one (Mixtral), `class Moe(Moe, Mlp)` beside a
  dense `Mlp` (DeepSeek-V3). Alias the router `router` and the shared expert
  `shared_experts`; set `SCORING` (`"softmax"`, `"sigmoid"`, ...). Relocated routing
  values are `TokenEProperty`s (below).
- **A recurrent mixer** (DeltaNet, Mamba-1, Mamba-2) is a `RecurrentMixer` subclass:
  `LinearAttention`, `SelectiveScan` or `StateSpace` with a docstring when transformers'
  pure-torch kernel is the one, else a new subclass setting `CHUNK_KERNEL`,
  `RECURRENT_KERNEL`, `STATE_OP` and declaring values at `kernel("inputs")`
  (`nnterp.components.recurrent.kernel`). Key it in `ENVOYS` so `route_kernels` finds it;
  `docs/developing/recurrent-mixer-internals.md` is the contract.
- **A block hands its children what their modules lack**, in `Layer.__init__` after
  `super().__init__`: Gemma-4 hands `self.mlp.router = self.router` (the mixture's router
  is the block's child), the Granite families hand `residual_multiplier` to their `Mlp`
  and mixers. Handed envoys survive the weight swap. Runnable on GPT-2:
  [references/descriptors.md](references/descriptors.md).

Shipped modules quoted, the rules, the test file: [references/family-module-recipe.md](references/family-module-recipe.md).

### Sizes and the logit lens

The root's `num_layers`, `hidden_size`, `vocab_size`, `num_heads`, `num_kv_heads`,
`head_dim`, `qk_head_dim`, `intermediate_size` are each a `StandardizedProperty`
(`nnterp.standardized`): the plain Llama-style rule over the text config unless the
family module defines `def <size>(model)`, which wins on read. `project_on_vocab` is a
`StandardizedCapability`, the same for a method: `def project_on_vocab(model, hidden)`
in the family module is bound in the root's place, so the lens on the last block still
equals `logits` (Cohere/Cohere-2 `* logit_scale`, Granite and its five relatives
`/ logits_scaling`, Falcon-H1, HyperCLOVA X, the Mamba families, Nemotron-H, DeepSeek-V4). Both are
read-only; assigning raises `AttributeError` naming the function to write. Which
families define which size: `docs/reference/families.md`, "Logits, scales and sizes".

Each block's own sizes are plain properties on its envoys, read off the module:
`layers[i].self_attn.num_heads`, `.num_kv_heads`, `.head_dim`, `.qk_head_dim` and
`layers[i].mlp.intermediate_size` (one routed expert's on a mixture). They differ from
the root's on Gemma-4, MiMo-V2-Flash and Laguna; a module that keeps a size under
another name overrides the property on the family's subclass (JetMoE's `Mlp`).

## The override toolkit

Keep the name, the layout annotation and the description; change only the location.

| The forward does | Write | Shipped example |
|---|---|---|
| adds a post-sublayer norm's output | `@EProperty("../post_attention_layernorm.output", ...)` on `attention_output` (native name after `../`) | gemma2.py |
| adds the residual inside the module | `@EProperty("source.dropout_add_0.input", ...)` | bloom.py |
| its own attention arithmetic | `EProperty("source.<op>.inputs", select=n)` / `("source.<op>.output", select=i)`; pattern at the dropout after its softmax | bloom.py, falcon.py, gptj.py |
| has no such value | `attention_scores = unavailable(NOT_ON_INTERFACE)` | DBRX `expert_outputs` (`unavailable(...)`) |
| routes off the interface on a config flag | `def off_interface(self): return "<reason>" if flag else super().off_interface()` | gpt2.py |
| picks its operation by a config flag | a key function of the envoy returning the path: `EProperty(by_alibi("F_softmax_0", "F_softmax_1", "input"))` | falcon.py |
| the value is a binding in the *block's* forward, read after the block starts | `@EProperty("../source.<op>.output")` on the child, `sourced = True` on the block's class | llama4_text.py |
| keeps heads first | `return seq_first(value)` in the preprocess and in `@value.postprocess` | mpt.py, falcon.py |
| mutates the served tensor in place later | preprocess `.clone()`, plus `@value.transform` returning `value.clone()` | falcon.py `mlp_output` |
| scales a sublayer's output when adding it | preprocess `* multiplier`, postprocess `/ multiplier` + `rewrap` | granite.py |
| holds the value flat over tokens `[batch*seq, ...]` | `TokenEProperty(...)` with the base's layout | hunyuan_v1_moe.py `router_logits` |

Each descriptor's arguments, a run of each, and its traps: [references/descriptors.md](references/descriptors.md).

## Finding the operation names

Outside a trace, `print(model.layers[0].self_attn.source)`: the labels on the left are
the names a `source` segment takes. A call is `<callee>_<n>` with the dotted chain
joined by `_`; a binding is an operation too, sharing one counter per name, so
`attention_interface_0` is the binding and `attention_interface_1` the interface *call*.
Operations inside that call exist only under `attention_interface_1.source`, only
inside a trace. Read in forward order: the call's `.inputs`, the drill, operations
inside, the call's `.output`:

```python
attn = model.layers[0].self_attn
with model.trace(prompt):
    args = attn.source.attention_interface_1.inputs.save()          # 1. the call's arguments
    inner = attn.source.attention_interface_1.source                # 2. the drill, from the live callee
    names = [op.name for op in inner].save()
    probs = inner.nn_functional_dropout_0.output.save()             # 3. an operation inside it
    returned = attn.source.attention_interface_1.output.save()      # 4. the call's own return

print(names)
assert len(args[0]) == 5                       # (module, query, key, value, attention_mask)
assert torch.equal(returned[1], probs)         # the pattern the module returns is the dropout's output
```

```
['repeat_kv_0', 'key_states_0', 'repeat_kv_1', 'value_states_0', 'key_states_transpose_0', 'torch_matmul_0', 'attn_weights_0', 'attn_weights_1', 'nn_functional_softmax_0', 'to_0', 'attn_weights_2', 'nn_functional_dropout_0', 'attn_weights_3', 'torch_matmul_1', 'attn_output_0', 'attn_output_transpose_0', 'contiguous_0', 'attn_output_1']
```

The base `Attention` pins `attention_scores` at `nn_functional_softmax_0`'s input,
`attention_probabilities` at `nn_functional_dropout_0`'s output, head outputs at the
call's output element 0, q/k/v at its `inputs` 1, 2, 3. A wrong name raises
`SourceNotAvailable` listing every operation; a name under a branch this config never
takes fails as `OutOfOrderError`. Naming rules, the SmolLM2 listing, dead branches, read
order: [references/finding-source-ops.md](references/finding-source-ops.md).

## A value of your own

Subclass the family's class (not `nnterp.Attention`, so the family's overrides stay), add
a descriptor, install it with `envoys=` keyed on the module type:

```python
from jaxtyping import Float
from torch import Tensor
from nnterp import DerivedEProperty, EProperty
from nnterp.components import Pattern, interface_reason
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention     # after import nnterp

def attention_entropy(self) -> Float[Tensor, "batch heads query"]:    # a new shape: no named layout has it
    probs = self.attention_probabilities                              # reads another value, inside the trace
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)

class MyAttention(gpt2.Attention):
    @EProperty("source.attention_interface_1.source.nn_functional_softmax_0.output",
               description="The softmax output before the dropout",
               unavailable=interface_reason)                         # states the eager requirement for you
    def attention_softmax(self, value) -> Pattern:                    # the pattern's own layout, by name
        return value

    attention_entropy = DerivedEProperty(attention_entropy, description="Per-row entropy of the pattern", unavailable=interface_reason)

custom = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager",
                                 envoys={GPT2Attention: MyAttention})     # the type key displaces the family's
attn = custom.layers[0].self_attn
assert type(attn) is MyAttention and custom.family is gpt2

with custom.trace(prompt):
    soft = attn.attention_softmax.save()
    entropy = attn.attention_entropy.save()

assert tuple(soft.shape) == tuple(entropy.shape) + (soft.shape[-1],) and (entropy >= 0).all()
assert "(attention_softmax) -> Pattern [batch heads query key]: The softmax output before the dropout" in repr(attn)
assert custom.support()["self_attn.attention_entropy"] is None                      # listed off the tree
assert "self_attn.attention_entropy" not in model.support()                         # the plain load has no envoy carrying it
assert MyAttention.attention_softmax.layout is Pattern is MyAttention.attention_probabilities.layout
```

Annotate with the named layout the shape already has (`Residual`, `Pattern`, `Keys`,
`RouterLogits`, ... : `nnterp.components` exports 28, `nnterp.standardized` adds `Logits`,
`NextTokenProbs`, `Tokens`); `layout` and `dims` are read off the annotation and the
repr line is `(name) -> Layout [axes]: description`. A shape no standard value has gets
an inline `jaxtyping` type. A value that must run on NDIF lives in an installed module,
since remote traces carry envoy classes by reference.

## A family whose checkpoints are vision-language wrappers

The family is chosen from `text_config.model_type`, so a wrapper (`LlavaForConditionalGeneration`
around a Llama) loads through the text model's family, and that family names the wrapper's
parts too. What it adds, quoted from `llama.py` and `gemma3_text.py`:

- **The wrapper's text spellings in `RENAME`**, beside the plain ones, so whichever the tree
  has binds: `"model.language_model.layers": "layers"` (Llava, DeepSeek-VL, Gemma 3),
  `"model.text_model.layers": "layers"` (Idefics 3, SmolVLM), and the same for
  `embed_tokens` and `norm`.
- **The tower's root and projector, keyed from the model root**:
  `"model.vision_tower": "vision"`, `"model.multi_modal_projector": "projector"`
  (`llama.py` also has `"model.vision_model": "vision"`, `"model.aligner": "projector"`,
  `"model.connector": "projector"`). `projector` is the last module before the scatter.
- **The tower's inner names, keyed relative to the tower**: multi-component keys no text
  model has (`"encoder.layers": "layers"`, `"embeddings.patch_embedding": "patch_embed"`)
  and single names no text block has (`"post_layernorm": "norm"`,
  `"layer_norm1": "input_layernorm"`, `"layer_norm2": "post_attention_layernorm"`). A bare
  name a text block also has is never a tower key.
- **`Vision`, `VisionLayer`, `VisionAttention`, `VisionMlp`** from `nnterp.components`, keyed
  in `ENVOYS` on the tower's module types:
  `SiglipVisionModel: Vision, SiglipEncoderLayer: VisionLayer, SiglipAttention: VisionAttention, SiglipMLP: VisionMlp`.
  A tower that differs (a sandwich block, a final norm a rename key cannot tell apart) gets a
  subclass in the family file (`llama.py`'s `SiglipVision`).
- **`ImageScatter` keyed on the wrapper's model type**, whose forward writes the image
  features into the token embeddings: `Gemma3Model: ImageScatter`, `LlavaModel: ImageScatter`.
  That is where `vision.image_features` is read. A wrapper that writes them through a helper
  keys a subclass naming the call and its argument (`llama.py`'s `InputsMerger`). A wrapper
  whose root forward scatters has no inner model to key, so the family sets
  `ROOT_SCATTER = "inputs_embeds_masked_scatter_0"` (`llama4_text.py`).
- **A `VisionSuite` subclass** in the family's test file, per wrapper (`TestLlavaVision` in
  `tests/families/test_llama.py`).

On `trl-internal-testing/tiny-LlavaForConditionalGeneration` (family `llama`, CLIP tower):

```python
from nnterp.components import ImageScatter

vlm = StandardizedTransformer("trl-internal-testing/tiny-LlavaForConditionalGeneration", task="image-text-to-text",
                              device="cpu", dispatch=True)
assert vlm.family is families.llama
assert vlm.get("model.vision_tower") is vlm.vision                      # a root key, bound from the model root
assert vlm.get("model.multi_modal_projector") is vlm.projector
assert vlm.get("model.vision_tower.encoder.layers.0") is vlm.vision.layers[0]   # an inner key, bound on the tower
assert type(vlm.vision).__name__ == "Vision"
assert [type(e).__name__ for e in (vlm.vision.layers[0], vlm.vision.layers[0].self_attn, vlm.vision.layers[0].mlp)] == [
    "VisionLayer", "VisionAttention", "VisionMlp"]
assert isinstance(vlm.get("model"), ImageScatter)                       # LlavaModel: where image_features is read
assert "norm" not in vlm.vision._aliases                                # CLIP's post_layernorm norms the CLS token only
assert vlm.get("model.language_model.layers.0") is vlm.layers[0]        # the wrapper's text spelling: the root stays the text model's
```

The rules, the per-tower exceptions, `ROOT_SCATTER` and the suite's attributes:
[references/family-module-recipe.md](references/family-module-recipe.md#a-vision-tower-in-a-family).

## Registering a family from outside the package

`families.register(family, *model_types)` puts any module or object with `RENAME` and
`ENVOYS` into `REGISTRY` under those types, and `lookup` consults `REGISTRY` before the
shipped modules. With no types given it takes the one the family's `__name__` ends in
(`register(my_package.zamba)` covers `zamba`); a `types.SimpleNamespace` has no
`__name__` and raises `TypeError`, so it always names them. It returns the family. Size
and `project_on_vocab` functions on it are read like a shipped module's:

```python
import types

variant = types.ModuleType("my_gpt2")                    # or: import my_package.my_family
variant.RENAME = {**gpt2.RENAME, "mlp": ["mlp", "ffn"]}
variant.ENVOYS = {**gpt2.ENVOYS, GPT2Attention: MyAttention}
variant.hidden_size = lambda model: 999                  # a size spelled by the family: wins over config.hidden_size
variant.intermediate_size = gpt2.intermediate_size       # a variant carries only what it defines
variant.project_on_vocab = lambda model, hidden: model.lm_head(model.norm(hidden)) * 2
assert families.register(variant, "gpt2") is variant      # the name says my_gpt2: the type is passed
try:
    registered = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager")
    assert registered.family is variant and type(registered.layers[0].self_attn) is MyAttention
    assert registered.hidden_size == 999 and registered.head_dim == 999 // registered.num_heads   # the root's rules read through it
    assert registered.intermediate_size == 4 * 999                     # gpt2's n_inner function, carried over
    assert registered.layers[0].ffn is registered.layers[0].mlp
    with registered.trace(prompt):
        lens = registered.project_on_vocab(registered.layers[-1].layer_output).save()
        logits = registered.logits.save()
    assert torch.allclose(lens, 2 * logits, atol=1e-4)                 # the family's function, bound in the root's place
    assert model.family is gpt2                                        # loaded before register: keeps what it resolved
    assert "gpt2" in families.REGISTRY and len(families.known()) == 98  # known() lists the shipped modules only
finally:
    del families.REGISTRY["gpt2"]

assert families.lookup("gpt2") is gpt2
```

A family that will ship is the same module saved as `nnterp/families/<model_type>.py`
with the `register` line removed. `rename=` / `envoys=` on one load are the per-model alternative.

## The test file

`tests/families/test_<model_type>.py`: a `FamilySuite` subclass with `REPO` (a tiny,
offline-cached checkpoint), `FAMILY` and `NATIVE = rows(...)`, run from the nnterp root
with `HF_HUB_OFFLINE=1 pytest tests/families/test_<model_type>.py`. It checks the
aliases, the envoy classes, `support()` against what reads, the contribution identity,
every source value on every attention block, root and per-module sizes against the
weights, `project_on_vocab(layers[-1].layer_output) == logits`, and every value against
its layout. Attributes and test groups: [references/family-module-recipe.md](references/family-module-recipe.md).

## References

| File | Covers |
|---|---|
| [references/family-module-recipe.md](references/family-module-recipe.md) | `known()`, `UnsupportedFamily`; llama, gpt2, gemma2, bloom, falcon quoted from source; MoE, recurrent-mixer, class-key and `project_on_vocab` excerpts; `RENAME`/`ENVOYS` rules; a vision tower in a family (tower keys, `ImageScatter`, `ROOT_SCATTER`, `no_tower_run`, per-tower exceptions, `VisionSuite`); `FamilySuite` |
| [references/descriptors.md](references/descriptors.md) | `EProperty` paths, `select`, `unavailable`, `sourced`, `DerivedEProperty`, `TokenEProperty`, handing values down, per-module sizes, `StandardizedProperty`/`StandardizedCapability`, helpers |
| [references/finding-source-ops.md](references/finding-source-ops.md) | operation naming, the GPT-2 and SmolLM2 listings, read order and the `.fn` error, dead branches, what cannot be drilled, what a release moves |

## Related skills

- `nnterp`: using the standard values, `support()`, MoE and recurrent-mixer values, the per-family quirks
- `patterns`: the recipes written against the standard values
- `nnsight`: `rename=`, `envoys=` and `.source` in general
- `debugging`: `OutOfOrderError`, an empty result, a trace cut short
