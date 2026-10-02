# The Family Module Recipe

A family is one module under `nnterp/families/` named after `config.model_type`, plus
one test file under `tests/families/`. This file quotes five complete modules as they
ship (the two plain ones and the three classic kinds of override), excerpts for a
mixture of experts, a recurrent mixer, class-keyed names and a family logit step, and
the test file with every `FamilySuite` attribute. Quoted family code carries relative imports
(`from ..components import ...`) and is not executed here; the nnterp repo's
`docs/extending/adding-a-family.md` and `docs/developing/testing.md` are the pages it
mirrors.

## Step 0: the model type and the registry

```python
import nnterp
from nnterp import families
from transformers import AutoConfig                      # after import nnterp

print(AutoConfig.from_pretrained("openai-community/gpt2").model_type)
print(AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct").model_type)
print(len(families.known()), families.known()[:4])
```

```
gpt2
llama
92 ['afmoe', 'apertus', 'arcee', 'bamba']
```

`families.known()` is the sorted list of shipped modules, read off the package directory
without importing them; `docs/reference/families.md` in the nnterp repo is the table of
all of them with their quirks. A multimodal config nests the text model's type under
`text_config`; that nested type is what `StandardizedTransformer` resolves. A type with
no module and nothing registered (Zamba and Zamba2 are deliberately unshipped: their
hybrid blocks add a shared transformer's output to the mixer's input, not to the
residual stream, so the standard contributions mean nothing there):

<!-- test: expect-error UnsupportedFamily -->
```python
families.lookup("zamba2")
# UnsupportedFamily: no standardization for model_type 'zamba2'; known: ['afmoe', 'apertus', ...].
# Add nnterp/families/zamba2.py with MODEL_TYPES, RENAME and ENVOYS, or pass a module to nnterp.families.register().
```

## The recipe

1. `AutoConfig.from_pretrained(repo).model_type`; create `nnterp/families/<model_type>.py`.
   The file name is what `lookup` imports, so `gemma3_text.py` covers `gemma3_text` and
   nothing else.
2. `MODEL_TYPES = ("<model_type>",)`. The registry test asserts every shipped module
   covers exactly the type it is named after.
3. Write `RENAME`. Print the raw model (`TransformersModel(repo)`) or its
   `named_modules()` for the native tree; map the containers and any block names that
   differ from Llama's.
4. Subclass `Layer`, `Attention`, `Mlp` (and `Moe` for a mixture, a `RecurrentMixer`
   subclass for a hybrid), overriding only what the forward spells differently.
5. Key them in `ENVOYS` on the transformers module classes.
6. When the config spells a root size its own way (`n_inner`, `ffn_dim`, `v_head_dim`),
   define `def <size>(model)` at module level; when the model does something to the
   head's output (a scale, a cast), define `def project_on_vocab(model, hidden)`. The
   suite's `test_sizes_match_the_model` and `test_project_on_vocab_is_the_logit_lens`
   check them.
7. Add `tests/families/test_<model_type>.py` and run it.

## Two complete modules as they ship

`llama.py`, the family the vocabulary is taken from. Mistral, Qwen2/3, Gemma 1, OLMo 1,
Phi-3, SmolLM3 and StableLM are this file with the class names swapped:

<!-- test: skip -->
```python
"""Llama (``LlamaForCausalLM``), the family the standard vocabulary is taken from.

Block names (``input_layernorm``, ``self_attn``, ``post_attention_layernorm``,
``mlp``) and ``lm_head`` are already the standard ones. The only change is
lifting the containers out of ``model.model``: ``model.layers`` instead of
``model.model.layers``.
"""

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaMLP

from ..components import Attention, Layer, Mlp

MODEL_TYPES = ("llama",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Llama's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Llama's attention; the shared eager forward and the residual added in the block, so the base holds."""


class Mlp(Mlp):
    """Llama's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {LlamaDecoderLayer: Layer, LlamaAttention: Attention, LlamaMLP: Mlp}
```

`gpt2.py` shows the rest of the template in use: block-level single-component aliases,
container keys anchored at the root, a config flag that takes the attention off the
shared interface, and a size the config spells its own way:

<!-- test: skip -->
```python
"""GPT-2 (``GPT2LMHeadModel``).

Its tree is ``transformer.{wte, wpe, drop, h[i].{ln_1, attn, ln_2, mlp}, ln_f}``
plus ``lm_head``. The container-level keys are anchored at the root
(``transformer.h``) so the aliases land on the root envoy: ``model.layers``,
not ``model.model.layers``. ``wpe`` (learned absolute positions) and ``drop``
have no standard name; they stay reachable under their own.

A checkpoint with ``reorder_and_upcast_attn`` set takes GPT-2's own
``_upcast_and_reordered_attn`` path instead of the shared eager forward, so
``attention_probabilities`` is unavailable there.
"""

from typing import TYPE_CHECKING

from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2MLP

from ..components import Attention, Layer, Mlp

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

MODEL_TYPES = ("gpt2",)

RENAME = {
    "transformer.wte": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "ln_1": "input_layernorm",
    "attn": "self_attn",
    "ln_2": "post_attention_layernorm",
}


class Layer(Layer):
    """GPT-2's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """GPT-2's attention; the shared eager forward and the residual added in the block, so the base holds.

    A checkpoint with ``reorder_and_upcast_attn`` takes GPT-2's own upcast
    path instead, where nothing on the interface is reachable.
    """

    def off_interface(self):
        if self._module.config.reorder_and_upcast_attn:
            return "this checkpoint sets reorder_and_upcast_attn, which takes GPT-2's own upcast attention path"
        return super().off_interface()


class Mlp(Mlp):
    """GPT-2's MLP; the residual is added in the block, so the base holds."""


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {GPT2Block: Layer, GPT2Attention: Attention, GPT2MLP: Mlp}


# -- sizes: what GPT-2's config calls them ------------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP width is ``n_inner``, ``None`` meaning four times the hidden size; the config's ``intermediate_size`` is never read by the model."""
    return model.config.n_inner or 4 * model.hidden_size
```

### The `RENAME` rules

nnsight resolves every key relative to every envoy in the tree (the `nnsight` skill's
`modules-and-architectures.md` covers `rename=` in general):

- A key with several components (`"transformer.h"`, `"model.layers"`,
  `"norm_attn_norm.attn"`) binds where it resolves from: the root. That lifts the
  containers out of the inner model so the alias is `model.layers`, not
  `model.model.layers`. DBRX's `"norm_attn_norm.attn": "self_attn"` resolves from each
  block, so the block reads like any other.
- A single-component key (`"attn"`, `"ln_1"`, `"ffn"`) binds on every envoy with a child
  of that name: every block gets `self_attn`.
- A key that resolves nowhere is skipped, not an error. GPT-NeoX's `RENAME` carries both
  `"embed_out": "lm_head"` and the containers; on current transformers the head is
  already `lm_head`, the `embed_out` key binds nothing, and the family covers both.
- An alias that would shadow an existing name (a sibling module, an `Envoy` attribute
  such as `output`, an `nn.Module` attribute such as `config`) raises at construction.
  OPT keeps its block-level `final_layer_norm` native: a single-component alias for it
  would also bind on the decoder's final norm.
- A value may be a list (`"mlp": ["mlp", "ffn"]`) to bind several aliases.
- A key may be a module **class**: nnsight binds it on every envoy with exactly one
  direct child of that class, for a native name whose meaning depends on the block.
  Nemotron-H's one `mixer` per block is keyed by class:

<!-- test: skip -->
```python
# nnterp/families/nemotron_h.py
MIXER_NAMES = {
    NemotronHMamba2Mixer: "linear_attn",
    NemotronHAttention: "self_attn",
    NemotronHMoE: "mlp",
    NemotronHMLP: "mlp",
}

RENAME = {
    "model.embeddings": "embed_tokens",
    "model.layers": "layers",
    "model.norm_f": "norm",
    # One native name, four meanings: the standard name follows the mixer's class.
    **MIXER_NAMES,
    "gate": "router",
}
```

A class two children of one envoy share is an error there; key those by name. On GPT-2:

```python
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP        # after import nnterp
from nnterp import StandardizedTransformer

by_class = StandardizedTransformer("openai-community/gpt2", device="cpu", rename={GPT2MLP: "ffn"})
assert by_class.layers[0].ffn is by_class.layers[0].mlp

try:
    StandardizedTransformer("openai-community/gpt2", device="cpu", rename={nn.LayerNorm: "a_norm"})
except ValueError as error:
    print(error)
```

```
`rename` key LayerNorm matches 2 children of `model.transformer.h.0` (ln_1, ln_2); a class key binds only where one child is of that class. Key those by name.
```

The standard names are `embed_tokens`, `layers`, `norm`, `lm_head`, and on blocks
`self_attn`, `mlp`, `input_layernorm`, `post_attention_layernorm`, `linear_attn`. Bind
the norms only where the family has a module in that position; their meaning varies,
and the suite checks the sublayer inputs against the family's own norm
(`ATTENTION_NORM`, `MLP_NORM` below), not the alias.

### The `ENVOYS` rules

- Import the modeling module at the top of the family module and nowhere else in nnterp:
  `import nnterp` loads no transformers modeling code because families are imported on
  first use (`test_import_is_lazy` runs that check in a subprocess).
- Several module types may share one envoy class (a dense MLP and a shared expert of the
  same class); a mixture gets its own (`DeepseekV2MLP: Mlp, DeepseekV2Moe: Moe`), and the
  suite accepts either class on a block.
- `envoys=` matches by type (walking the module's MRO) or by native path suffix, never by
  alias, and type keys are tried before path keys. A user displaces a family's envoy by
  keying on the same type.
- Define `Layer`, `Attention` and `Mlp` even when a module type has no such module; the
  suite reads `FAMILY.Attention`. OPT keys no `Mlp` in `ENVOYS`, and `support()`, which
  walks the tree, then lists no `mlp` value at all (`"mlp.mlp_output" not in
  model.support()`); the suite skips its MLP tests there.

### The size and logit functions

The root's sizes (`num_layers`, `hidden_size`, `vocab_size`, `num_heads`, `num_kv_heads`,
`head_dim`, `qk_head_dim`, `intermediate_size`) each read the plain config key; each is a
`StandardizedProperty`, which on read prefers a function of the same name in the family
module. So a family states its own spelling as a module-level `def <size>(model)` and
nothing else: no registration, no subclass, and the function may read the other sizes
off the model (`4 * model.hidden_size`). Falcon's two are quoted below; a family whose
config uses the plain keys defines none (`llama.py`). Which of the 92 define which size
is in the nnterp repo's `docs/reference/families.md`, "Logits, scales and sizes"
(DeepSeek-V2/V3/V3.2, GLM-MoE-DSA, GLM4-MoE-Lite, Youtu, MiMo-V2-Flash, GPT-BigCode,
GPT-2, GPT-J, GPT-Neo, CodeGen, OPT, XGLM, GPT-NeoX-Japanese, MPT, BLOOM, Mamba-2,
Gemma-4, GraniteMoE-Hybrid, ZAYA, Falcon, and DBRX and Llama 4's `intermediate_size`,
which that table omits). A module re-exports another's with an import
(`from .deepseek_v2 import head_dim, qk_head_dim` in `deepseek_v3.py`).

`project_on_vocab` is a `StandardizedCapability`: the root's is `lm_head(norm(hidden))`
plus `final_logit_softcapping` when the config sets it, and a family whose model does
something else to the head's output defines `def project_on_vocab(model, hidden)`,
bound in the root's place, so the lens on the last block still equals `logits`:

<!-- test: skip -->
```python
# nnterp/families/cohere.py
def project_on_vocab(model: "StandardizedTransformer", hidden: torch.Tensor) -> torch.Tensor:
    """The logit lens as the model makes its logits: the final norm, ``lm_head``, then times ``logit_scale``."""
    return model.lm_head(model.norm(hidden)) * model.config.logit_scale
```

The others: Cohere-2 (imports Cohere's), Granite (`/ logits_scaling`; GraniteMoE,
GraniteMoE-Shared, GraniteMoE-Hybrid, Granite-SWA, GraniteMoE-SWA import it), Falcon-H1
(`* lm_head_multiplier`), HyperCLOVA X (`* logits_scaling`), Mamba, Falcon-Mamba,
Mamba-2 and Nemotron-H (the head in its own dtype, float32 logits), DeepSeek-V4 (collapses its parallel streams with `hc_head` first).
Each block's own sizes are not module functions but properties on the family's envoy
subclasses ([descriptors.md](descriptors.md)).

## Override 1: a sandwich block (gemma2.py)

Gemma-2's block is `x + post_attention_layernorm(attn(input_layernorm(x)))` and then
`+ post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))`. What the block adds
is the sibling norm's output, so the contribution points there. Gemma-3 (text), OLMo-2
and OLMo-3 are the same override; OLMo-2/3 have only the post-norms, so `self_attn.input`
is the block input there.

<!-- test: skip -->
```python
"""Gemma 2 (``Gemma2ForCausalLM``).

Llama's names plus a *sandwich* block: every sublayer is normed before and
after, ``x + post_attention_layernorm(attn(input_layernorm(x)))`` and then
``+ post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))``. What the
block adds is the post-norm's output, not the module's, so the contributions
point at the sibling norms. ``final_logit_softcapping`` applies to the logits after ``lm_head``.
"""

from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention, Gemma2DecoderLayer, Gemma2MLP

from ..components import Attention, EProperty, Layer, Mlp, Residual

MODEL_TYPES = ("gemma2",)

RENAME = {
    "model.embed_tokens": "embed_tokens",
    "model.layers": "layers",
    "model.norm": "norm",
}


class Layer(Layer):
    """Gemma-2's decoder block; returns a bare tensor, so the base holds."""


class Attention(Attention):
    """Gemma-2's attention: the shared eager forward, but what reaches the residual stream is the post-attention norm's output."""

    @EProperty(
        "../post_attention_layernorm.output",
        description="What the attention adds to the residual stream: the post-attention norm's output",
    )
    def attention_output(self, value) -> Residual:
        return value


class Mlp(Mlp):
    """Gemma-2's MLP: what reaches the residual stream is the post-feedforward norm's output."""

    @EProperty(
        "../post_feedforward_layernorm.output",
        description="What the MLP adds to the residual stream: the post-feedforward norm's output",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {Gemma2DecoderLayer: Layer, Gemma2Attention: Attention, Gemma2MLP: Mlp}
```

Every read, write and in-place edit of `layers[i].self_attn.attention_output` now goes to
`layers[i].post_attention_layernorm.output`, and the contribution identity the suite
checks holds. `../` steps to the parent by name arithmetic, so the rest of the key is
native names; it works here because Gemma-2's native name is the standard one (on GPT-2
the same key fails, [descriptors.md](descriptors.md)). The override
annotates `-> Residual`, the same named layout as the
base value (`Residual` comes in the same `from ..components import ...` line as the
envoys), so `layout` and `dims` stay the base's rather than a retyped copy.

## Override 2: the residual added inside the module (bloom.py)

BLOOM's sublayers take the residual as an argument and add it inside
(`dropout_add(x, residual, ...)`), so the module's output is a residual-stream state.
The contribution is the first argument of that call, `"source.dropout_add_0.input"`.
BLOOM's attention also does its own arithmetic whatever `attn_implementation` says, so
every interior value is redefined on its own operations and none carries an eager
predicate:

<!-- test: skip -->
```python
"""BLOOM (``BloomForCausalLM``).

``transformer.{word_embeddings, word_embeddings_layernorm, h[i].{input_layernorm,
self_attention, post_attention_layernorm, mlp}, ln_f}`` and ``lm_head``. Both
sublayers take the residual as an argument and add it *inside* the module
(``dropout_add``), so the module outputs are residual-stream states; the
contributions are the first argument of each ``dropout_add`` call. The
attention does its own arithmetic whatever ``attn_implementation`` says, so
the pattern is its dropout's output and needs no eager load.
The embedding norm has no standard name.
"""

from typing import TYPE_CHECKING

from transformers.models.bloom.modeling_bloom import BloomAttention, BloomBlock, BloomMLP

from ..components import Attention, EProperty, HeadOutputs, Keys, Layer, Mlp, Pattern, Queries, Residual, Values

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

MODEL_TYPES = ("bloom",)

RENAME = {
    "transformer.word_embeddings": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "self_attention": "self_attn",
}


class Layer(Layer):
    """BLOOM's block; returns a tuple, which the base unwraps."""

    returns_tuple = True


class Attention(Attention):
    """BLOOM's attention adds the residual inside: the contribution is what enters ``dropout_add``."""

    # ``_reshape`` splits the fused projection into ``(query, key, value)``,
    # heads first (the keys are transposed for the score matmul only later);
    # the scores are the softmax's input after the mask; the head outputs are
    # the ``bmm`` result, ``[batch * heads, seq, head_dim]``.

    @EProperty("source.self__reshape_0.output", select=0, description=Attention.attention_queries.description)
    def attention_queries(self, value) -> Queries:
        return value

    @EProperty("source.self__reshape_0.output", select=1, description=Attention.attention_keys.description)
    def attention_keys(self, value) -> Keys:
        return value

    @EProperty("source.self__reshape_0.output", select=2, description=Attention.attention_values.description)
    def attention_values(self, value) -> Values:
        return value

    @EProperty("source.F_softmax_0.input", description=Attention.attention_scores.description)
    def attention_scores(self, value) -> Pattern:
        return value

    @EProperty("source.torch_bmm_0.output", description=Attention.attention_head_outputs.description)
    def attention_head_outputs(self, value) -> HeadOutputs:
        batch_heads, seq, head_dim = value.shape
        heads = self._module.num_heads
        return value.view(batch_heads // heads, heads, seq, head_dim).transpose(1, 2)

    @attention_head_outputs.postprocess
    def attention_head_outputs(self, value):
        batch, seq, heads, head_dim = value.shape
        return value.transpose(1, 2).reshape(batch * heads, seq, head_dim)

    @EProperty(
        "source.dropout_add_0.input",
        description="What the attention adds to the residual stream: the tensor entering dropout_add",
    )
    def attention_output(self, value) -> Residual:
        return value

    @EProperty(
        "source.self_attention_dropout_0.output",
        description="The attention pattern the values are mixed with",
    )
    def attention_probabilities(self, value) -> Pattern:
        return value


class Mlp(Mlp):
    """BLOOM's MLP adds the residual inside: the contribution is what enters ``dropout_add``."""

    @EProperty(
        "source.dropout_add_0.input",
        description="What the MLP adds to the residual stream: the tensor entering dropout_add",
    )
    def mlp_output(self, value) -> Residual:
        return value


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {BloomBlock: Layer, BloomAttention: Attention, BloomMLP: Mlp}


# -- sizes: BLOOM's config does not say ---------------------------------------------

def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP is four times the hidden size wide; the config has no key for it."""
    return 4 * model.hidden_size
```

Four things to copy from it: `description=Attention.attention_queries.description`
reuses the base's text (the class attribute is the descriptor itself), and each
redefinition annotates with the base's named layout (`-> Queries`, `-> Pattern`,
`-> HeadOutputs`, imported from `..components` beside the envoys); a key ending in
`.input` on `dropout_add_0` means "the first argument", and assigning it replaces only
that argument, keeping the residual; `select=0/1/2` on `"source.self__reshape_0.output"`
picks one of the three tensors the call returns, and a write repacks the tuple; the head
outputs are a *view* on read (`view` and `transpose`) so in-place edits still land, with
a `postprocess` reversing both for an assignment. MPT's MLP adds the residual inside
too; its contribution is `EProperty("source.F_dropout_0.output", ...)`, the dropout's
output just before the add.

## Override 3: own arithmetic, read order and a clone with a transform (falcon.py)

Falcon's 7B layout is a parallel block (one norm feeds both sublayers) whose attention
does its own arithmetic on one of two branches picked by `config.alibi`, with different
operations for every interior value, and whose block adds the attention output *into
the MLP's output tensor in place*:

<!-- test: skip -->
```python
"""Falcon (``FalconForCausalLM``): the 7B layout (parallel attention), with or without alibi, and the 40B layout.

``transformer.{word_embeddings, h[i].{input_layernorm, self_attention, mlp},
ln_f}`` and ``lm_head``. One norm feeds both sublayers and the block sums
``x + attn + mlp``, but it does so by adding the attention output *into the
MLP's output tensor in place*, so ``mlp_output`` reads a copy taken as the
MLP returns, and a transform carries edits to it back into the model.

The attention does its own arithmetic, and ``config.alibi`` picks one of two
branches of it with different operations, so each interior value names its
operation by that flag. Without alibi the queries and keys leave
``apply_rotary_pos_emb`` and the pattern is the first softmax, which has no
dropout after it; with alibi there is no rotary, the pattern is the dropout
after the second softmax, and the head outputs are flattened over batch and
heads. The 40B layout (``ln_attn`` / ``ln_mlp``, ``new_decoder_architecture``)
runs the same forward with its key/value heads already broadcast.
"""

from typing import TYPE_CHECKING

from transformers.models.falcon.modeling_falcon import FalconAttention, FalconDecoderLayer, FalconMLP

from ..components import (
    Attention, EProperty, HeadOutputs, Keys, Layer, Mlp, Pattern, Queries, Residual, Values,
    first_tensor, needs_eager, rewrap, seq_first,
)

if TYPE_CHECKING:
    from ..standardized import StandardizedTransformer

MODEL_TYPES = ("falcon",)

RENAME = {
    "transformer.word_embeddings": "embed_tokens",
    "transformer.h": "layers",
    "transformer.ln_f": "norm",
    "self_attention": "self_attn",
}


class Layer(Layer):
    """Falcon's parallel block; returns a tuple, which the base unwraps."""

    returns_tuple = True


def alibi(envoy) -> bool:
    return bool(envoy._module.config.alibi)


def by_alibi(without: str, with_alibi: str, attribute: str = "output"):
    """A key at one of two operations, chosen by the checkpoint's ``alibi`` flag, a config value read at load."""

    def choose(envoy):
        return f"source.{with_alibi if alibi(envoy) else without}.{attribute}"

    choose.__name__ = f"{without}|{with_alibi}"
    return choose


class Attention(Attention):
    """Falcon's attention: the residual is added in the block; the pattern and the interior on its own ops, per ``alibi``."""

    # Without alibi, queries and keys are ``apply_rotary_pos_emb``'s two
    # returns and the values the binding just before it (so read the values
    # before the queries or keys in one trace: they bind first); with alibi
    # there is no rotary and all three are the reshaped bindings, in forward
    # order. Keys and values are ``num_kv_heads`` wide (1 under multi-query).
    # The scores are the softmax's input after the mask, the pattern the
    # softmax itself (no alibi) or the dropout after it (alibi). The head
    # outputs are the ``scores @ values`` product: heads first without alibi,
    # flattened over batch and heads with it; both are served ``[batch, seq,
    # heads, head_dim]`` as a view, so in-place edits land.

    @EProperty(by_alibi("apply_rotary_pos_emb_0", "query_layer_0"), description=Attention.attention_queries.description, unavailable=needs_eager)
    def attention_queries(self, value) -> Queries:
        return value if alibi(self) else value[0]

    @attention_queries.postprocess
    def attention_queries(self, value):
        if alibi(self):
            return value
        _, keys = self.source.apply_rotary_pos_emb_0.output
        return value, keys

    @EProperty(by_alibi("apply_rotary_pos_emb_0", "key_layer_0"), description=Attention.attention_keys.description, unavailable=needs_eager)
    def attention_keys(self, value) -> Keys:
        return value if alibi(self) else value[1]

    @attention_keys.postprocess
    def attention_keys(self, value):
        if alibi(self):
            return value
        queries, _ = self.source.apply_rotary_pos_emb_0.output
        return queries, value

    @EProperty("source.value_layer_0.output", description=Attention.attention_values.description, unavailable=needs_eager)
    def attention_values(self, value) -> Values:
        return value

    @EProperty(by_alibi("F_softmax_0", "F_softmax_1", "input"), description=Attention.attention_scores.description, unavailable=needs_eager)
    def attention_scores(self, value) -> Pattern:
        return value

    @EProperty(by_alibi("attn_output_1", "flatten_0"), description=Attention.attention_head_outputs.description, unavailable=needs_eager)
    def attention_head_outputs(self, value) -> HeadOutputs:
        if alibi(self):  # [batch * heads, seq, head_dim] -> a [batch, seq, heads, head_dim] view
            heads = self._module.num_heads
            return value.view(-1, heads, *value.shape[1:]).transpose(1, 2)
        return seq_first(value)

    @attention_head_outputs.postprocess
    def attention_head_outputs(self, value):
        if alibi(self):
            return value.transpose(1, 2).reshape(-1, *value.shape[1:2], value.shape[3])
        return seq_first(value)

    @EProperty(
        by_alibi("F_softmax_0", "self_attention_dropout_0"),
        description="The attention pattern the values are mixed with",
        unavailable=needs_eager,
    )
    def attention_probabilities(self, value) -> Pattern:
        return value


class Mlp(Mlp):
    """Falcon's MLP: the block later adds the attention into this tensor in place, so read a copy.

    The copy keeps a saved read honest. So that in-place edits to it still
    reach the model, a transform hands a copy of the edited copy back to be
    swapped in once the block is done with the read; the second copy is what
    keeps the user's tensor clean when the block then adds into it.
    """

    @EProperty(key="output", description="What the MLP adds to the residual stream (a copy, since the block adds the attention into the live tensor in place)")
    def mlp_output(self, value) -> Residual:
        return first_tensor(value).clone()

    @mlp_output.postprocess
    def mlp_output(self, value):
        return rewrap(self, value)

    @mlp_output.transform
    def mlp_output(self, value, raw):
        # Fires on the model side, after the read. The module returns a bare
        # tensor, so ``raw`` needs no rebuilding around the edited copy.
        return value.clone()


#: Module type -> Envoy subclass, for nnsight's ``envoys=``.
ENVOYS = {FalconDecoderLayer: Layer, FalconAttention: Attention, FalconMLP: Mlp}


# -- sizes: what Falcon's config calls them --------------------------------------

def num_kv_heads(model: "StandardizedTransformer") -> int:
    """``num_kv_heads`` on the 40B layout (``new_decoder_architecture``); 1 under ``multi_query``; else every head."""
    config = model.config
    if config.new_decoder_architecture:
        return config.num_kv_heads
    return 1 if config.multi_query else model.num_heads


def intermediate_size(model: "StandardizedTransformer") -> int:
    """The MLP width is ``ffn_hidden_size``."""
    return model.config.ffn_hidden_size
```

What each part answers:

- **Values from a binding, not a call.** `value_layer_0`, `attn_output_1` and the alibi
  branch's `query_layer_0`, `key_layer_0` and `flatten_0` are assignments in the forward;
  a binding is an operation whose `.output` is the bound value. `attn_output_0` is the
  sdpa branch that does not run under eager; nnsight numbers every call in the source,
  executed or not, so both branches' names are in one listing.
- **A path picked per checkpoint.** `EProperty`'s key may be a function of the envoy
  returning the path, called at each read; `by_alibi(without, with_alibi, attribute="output")`
  builds one off `config.alibi`, `"source.<op>.<attribute>"`, so one class covers both
  branches: queries
  `apply_rotary_pos_emb_0` return 0 / `query_layer_0`, keys `apply_rotary_pos_emb_0`
  return 1 / `key_layer_0`, values `value_layer_0` on both, scores `F_softmax_0` /
  `F_softmax_1` input, pattern `F_softmax_0` output / `self_attention_dropout_0` output,
  head outputs `attn_output_1` / `flatten_0`. The preprocess and postprocess branch on
  the same flag where the served objects differ (the rotary's tuple, the flattened head
  outputs). Every one of the six is guarded by `needs_eager` alone.
- **Read order.** Without alibi the values bind before the rotary that produces the
  queries and keys, so a trace reading two of them reads `attention_values` first; with
  alibi there is no rotary and the three are the reshaped bindings in forward order:
  queries, keys, values. The suite reads interior values one per trace for this reason;
  `test_falcon.py` pins the first order in `test_values_bind_before_the_rotary`. A value
  read at a `source` path is read inside the module's forward whichever branch runs, so
  the suite's `test_every_source_value_resolves_on_every_layer` selects them with
  `inside_forward()`, which is `True` for a key function.
- **The pattern.** No dropout follows the first softmax, so without alibi the pattern is
  `F_softmax_0`'s output; the alibi branch runs a second softmax with a dropout after
  it, so there the pattern is `self_attention_dropout_0`'s output. Wherever a dropout
  does follow, read the pattern there.
- **`seq_first`** on read and again in `postprocess`: a transpose is a view of the same
  storage, so in-place edits land; the transpose is its own inverse. The alibi branch's
  head outputs come out `[batch * heads, seq, head_dim]` and are served as the same
  `[batch, seq, heads, head_dim]` view (`view` then `transpose`), the postprocess
  reversing both, so in-place edits land on that branch too.
- **Clone plus transform.** A plain `mlp_output` would be a live tensor the block later
  mutates, so a saved read would silently become `mlp + attn`. The preprocess returns a
  clone; a clone is invisible to the model, so `@mlp_output.transform` hands the edited
  copy back to be swapped in after the read. `transform(self, value, raw)` receives the
  raw served value so a tuple-returning module could rebuild its container
  (`(edited.clone(), *raw[1:])`). A transform is needed only when the preprocess returns
  something other than the served object and in-place edits must still reach the
  model; GPT-2's MLP output is never mutated later, so the base holds there. Its two
  Falcon tests: `mlp.output == mlp_output + attention_output`, and `mlp_output[:] = 0`
  moves the logits while the saved copy stays zero.
- **The sizes.** Falcon's config has no `num_key_value_heads` and no `intermediate_size`:
  the key/value head count is `num_kv_heads` on the 40B layout, `1` under `multi_query`
  and every head otherwise, and the MLP width is `ffn_hidden_size`. The two module-level
  functions are the whole override; the root's `StandardizedProperty` finds them by name.
- **The layouts.** Every redefined value annotates with the base's named layout
  (`-> Queries`, `-> Keys`, `-> Values`, `-> Pattern`, `-> HeadOutputs`, `-> Residual`),
  imported from `..components`, so `falcon.Attention.attention_keys.layout is
  Keys` whichever branch serves it; the alibi branch's head outputs are reshaped *to*
  that layout rather than annotated with their own.

## Mixtures, recurrent mixers, and a block handing values down

The newer envoy kinds are subclasses keyed in `ENVOYS` like the rest; most families
need only a docstring. Excerpts from the current source (the rest of each module is the
Llama shape). A mixture where every MLP is one (Mixtral) makes `Moe` its `Mlp`; one with
dense blocks too (DeepSeek-V3) adds a `Moe` that inherits the family's `mlp_output`:

<!-- test: skip -->
```python
# nnterp/families/mixtral.py
class Mlp(Moe):
    """A mixture of experts: the module returns the routed hidden states (a bare tensor on this transformers), so the base holds."""

ENVOYS = {MixtralDecoderLayer: Layer, MixtralAttention: Attention, MixtralSparseMoeBlock: Mlp}


# nnterp/families/deepseek_v3.py
class Moe(Moe, Mlp):
    """DeepSeek-V3's mixture of experts: a sigmoid router with a selection bias and group-limited top-k, routed experts and shared experts."""

    SCORING = "sigmoid"

ENVOYS = {DeepseekV3DecoderLayer: Layer, DeepseekV3Attention: Attention, DeepseekV3MLP: Mlp, DeepseekV3MoE: Moe}
```

The base `Moe` holds when the router computes its logits with `F.linear` (`LOGITS`),
the experts module takes `(hidden, top_k_index, top_k_weights)` under transformers'
`@use_experts_implementation`, and the shared expert's output is what the mixture adds;
`RENAME` aliases the router `router` (`"gate": "router"`) and the shared expert
`shared_experts`. Otherwise redefine the value as a `TokenEProperty` with the base's
layout (Hunyuan's `"router.wg.output"`, [descriptors.md](descriptors.md)), or
`unavailable(...)` where no tensor holds it (DBRX's `expert_outputs`). The nnterp repo's
`docs/usage/mixture-of-experts.md` is the user side.

A recurrent mixer whose forward is transformers' pure-torch kernel needs only a
docstring on the shipped base (`LinearAttention` for a gated DeltaNet, `SelectiveScan`
for Mamba-1, `StateSpace` for Mamba-2):

<!-- test: skip -->
```python
# nnterp/families/qwen3_next.py
class LinearAttention(LinearAttention):
    """Qwen3-Next's gated DeltaNet mixer; transformers' pure-torch chunked rule, so the base holds."""

ENVOYS = {Qwen3NextDecoderLayer: Layer, Qwen3NextAttention: Attention, Qwen3NextGatedDeltaNet: LinearAttention, Qwen3NextMLP: Mlp, Qwen3NextSparseMoeBlock: Moe}
```

A mixer with other kernels is a new `RecurrentMixer` subclass that sets `CHUNK_KERNEL`,
`RECURRENT_KERNEL` and `STATE_OP` and declares its values at `kernel("inputs")`
([descriptors.md](descriptors.md), and the nnterp repo's
`docs/developing/recurrent-mixer-internals.md` for the occurrence arithmetic). Keep it
keyed in `ENVOYS`: `route_kernels(model.family, ...)` finds the mixer class there.

When a child's value needs something its module lacks, the block hands it down in
`Layer.__init__`; Gemma-4's mixture lives on the block, not in its `mlp`:

<!-- test: skip -->
```python
# nnterp/families/gemma4_text.py
class Layer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp.has_experts = bool(self._module.enable_moe_block)
        if self.mlp.has_experts:
            self.mlp.router = self.router
            self.mlp.experts = self.experts
```

The GraniteMoE families call `hand_residual_multiplier(self)` there, setting the
block's `residual_multiplier` on its `Mlp` and `RecurrentMixer` children, whose modules
keep no config. A family never looks its parent up through the interleaver's envoys.

## The test file

One file per family under `tests/families/`, subclassing `FamilySuite`
(`tests/families/suite.py`). Llama's is the minimum, GPT-2's states one quirk and adds
one family-specific test, Falcon's has three classes for one family (the 7B layout, the
same checkpoint with `alibi` switched on in a patched copy of its config, and the 40B
layout):

<!-- test: skip -->
```python
"""Llama, end to end: the family the vocabulary is taken from."""

from suite import FamilySuite, LLAMA_ROWS

from nnterp.families import llama


class TestLlama(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    FAMILY = llama
    NATIVE = LLAMA_ROWS
```

<!-- test: skip -->
```python
"""GPT-2, end to end."""

from suite import FamilySuite, rows

from nnterp.families import gpt2


class TestGPT2(FamilySuite):
    REPO = "hf-internal-testing/tiny-random-gpt2"
    FAMILY = gpt2
    NATIVE = rows("transformer", "h", "wte", "ln_f", attn="attn", ln1="ln_1", ln2="ln_2")
    REFUSES_IN_PLACE_QKV = True  # q/k/v are split views of one c_attn tensor

    def test_reorder_and_upcast_makes_the_interface_unavailable(self, model):
        config = model.layers[0].self_attn._module.config
        config.reorder_and_upcast_attn = True
        try:
            support = model.support()
            assert all("reorder_and_upcast_attn" in support[f"self_attn.{name}"][0] for name in ("attention_probabilities", "attention_queries"))
        finally:
            config.reorder_and_upcast_attn = False
```

<!-- test: skip -->
```python
"""Falcon (7B layout), end to end: parallel block, multi-query, in-place add into the MLP output."""

import glob
import json
import os
import tempfile

import torch
from suite import FamilySuite, rows, PROMPT

from nnterp.families import falcon


class TestFalcon(FamilySuite):
    REPO = "Rocketknight1/tiny-random-falcon-7b"
    FAMILY = falcon
    NATIVE = rows("transformer", "h", "word_embeddings", "ln_f", attn="self_attention", ln2=None)
    MLP_NORM = "input_layernorm"             # parallel (7B layout)

    def test_mlp_output_is_a_copy_the_block_does_not_touch(self, model):
        """The block adds the attention into the MLP's live tensor in place; the value is a copy."""
        with model.trace(PROMPT):
            attn = model.layers[0].self_attn.attention_output.save()
            mlp = model.layers[0].mlp.mlp_output.save()
            live = model.layers[0].mlp.output.save()
        torch.testing.assert_close(live, mlp + attn)

    def test_in_place_mlp_edit_reaches_the_model_through_the_transform(self, model):
        with model.trace(PROMPT):
            clean = model.logits.save()
        with model.trace(PROMPT):
            model.layers[0].mlp.mlp_output[:] = 0
            kept = model.layers[0].mlp.mlp_output.save()
            edited = model.logits.save()
        assert not torch.equal(clean, edited)
        assert torch.equal(kept, torch.zeros_like(kept))  # the user's copy stays what they made it

    def test_values_bind_before_the_rotary(self, model):
        """In one trace the values must be read before the queries or keys."""
        with model.trace(PROMPT):
            v = model.layers[0].self_attn.attention_values.save()
            q = model.layers[0].self_attn.attention_queries.save()
        assert v.shape[1] == 1 and q.shape[1] == model.num_heads  # multi-query



def _alibi_checkpoint(repo="Rocketknight1/tiny-random-falcon-7b"):
    """The 7B tiny checkpoint with ``alibi`` switched on in its config: the same weights, the other attention branch."""
    snapshot = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"))[0]
    patched = tempfile.mkdtemp(prefix="falcon-alibi-")
    for name in os.listdir(snapshot):
        if name != "config.json":
            os.symlink(os.path.realpath(os.path.join(snapshot, name)), os.path.join(patched, name))
    config = json.load(open(os.path.join(snapshot, "config.json")))
    config["alibi"] = True
    json.dump(config, open(os.path.join(patched, "config.json"), "w"))
    return patched


class TestFalconAlibi(FamilySuite):
    """The 7B layout with alibi: no rotary, the pattern at the dropout after the second softmax, flattened head outputs."""

    REPO = _alibi_checkpoint()
    FAMILY = falcon
    NATIVE = rows("transformer", "h", "word_embeddings", "ln_f", attn="self_attention", ln2=None)
    MLP_NORM = "input_layernorm"

    def test_alibi_branch(self, model):
        assert model.layers[0].self_attn._module.config.alibi
        with model.trace(PROMPT):
            probs = model.layers[0].self_attn.attention_probabilities.save()
            raw = model.layers[0].self_attn.source.self_attention_dropout_0.output.save()
        assert torch.equal(probs, raw)


class TestFalcon40B(FamilySuite):
    """The 40B layout: ``new_decoder_architecture``, with ``ln_attn`` / ``ln_mlp`` in place of one norm."""

    REPO = "Rocketknight1/tiny-random-falcon-40b"
    FAMILY = falcon
    NATIVE = rows("transformer", "h", "word_embeddings", "ln_f", attn="self_attention", ln1=None, ln2=None)
    KV_HEADS_EXPANDED = True  # the new layout broadcasts its 8 kv heads to all 128 before the rotary
    ATTENTION_NORM = "ln_attn"
    MLP_NORM = "ln_mlp"
    MLP_NORM_BEFORE_ATTENTION = True   # both norms are taken from the block input before either sublayer runs

    def test_new_decoder_architecture(self, model):
        assert model.config.new_decoder_architecture
        assert hasattr(model.layers[0], "ln_attn") and hasattr(model.layers[0], "ln_mlp")
```

`rows(container, layers, embed, norm, attn="self_attn", mlp="mlp", ln1="input_layernorm", ln2="post_attention_layernorm")`
builds the standard-path to native-path dict; pass `None` for a module the family does
not have (`ln2=None` on a parallel block, `mlp=None` on OPT). `LLAMA_ROWS` is
`rows("model", "layers", "embed_tokens", "norm")`. The class-scoped `model` fixture loads
`REPO` with `dispatch=True, attn_implementation="eager"` plus `LOAD_KWARGS`, and
`raw_model` is the same checkpoint as a plain `TransformersModel`.

### Every class attribute

| attribute | meaning |
| --- | --- |
| `REPO` | The pinned tiny checkpoint, offline-cached. |
| `FAMILY` | The family module the checkpoint must resolve to (`model.family is FAMILY`). |
| `NATIVE` | Standard path to native path; each pair must be the same envoy, and every `layers.0.*` name must exist on every block. |
| `EXPECTED_UNAVAILABLE` | `support()` key to a substring of the reason, for values some block lacks (a hybrid: `"self_attn.attention_output": "no self_attn module"` for its linear blocks); every other value must report `None`. A module no block has (OPT's `mlp`) is not in `support()` and needs no entry. Default `{}`. |
| `REFUSES_IN_PLACE_QKV` | torch refuses in-place edits on q/k/v that are views out of a `split`/`chunk` (GPT-2, MPT); the suite expects a `RuntimeError` matching `view` and skips the in-place query edit. |
| `ATTENTION_SINK` | The pattern's rows sum to less than one (GPT-OSS). |
| `KV_HEADS_EXPANDED` | Keys and values are read already expanded to `num_heads` (latent attention; Falcon's 40B layout). |
| `MLP_WIDTH_KEY` | A config key naming the first block's MLP width when it is not `intermediate_size` (an all-MoE family's `moe_intermediate_size`). |
| `LOAD_KWARGS` | Extra load arguments the checkpoint needs (`{"dtype": torch.float32}` on DBRX's degenerate tiny checkpoint). |
| `QUERY_GATED` | `q_proj` produces the query and a gate side by side, twice the width (Qwen3.5). |
| `ATTENTION_NORM` | The block's own module whose output enters the attention; `None` when the block input enters directly (OLMo-2/3). Default `"input_layernorm"`. |
| `MLP_NORM` | The block's own module whose output enters the MLP, whatever the family calls it. Default `"post_attention_layernorm"`; `"input_layernorm"` on a parallel block. |
| `MLP_NORM_BEFORE_ATTENTION` | The MLP's norm runs before the attention does (Falcon's 40B layout norms both inputs up front). |
| `MOE_UNAVAILABLE` | Mixture value to a substring of the reason, for values this checkpoint's mixture lacks (no entry needed for `shared_expert_output` without a shared expert). |
| `ROUTER_EXTRA_CLASSES` | Router columns beyond the experts (ZAYA's skip class). |

`pattern_from_scores(self, model, scores)` is the one method a subclass may override:
what the softmax makes of `attention_scores`; a sink family appends its column.

### What the suite asserts

| group | one line each |
|---|---|
| Names | `model.family is FAMILY`; every `NATIVE` row resolves to the same envoy; nothing is bound at `model.model`; every `layers.0.*` name exists on every block; every block, attention and MLP is exactly the family's class and a subclass of the component base; a trace through the standard names gives `hidden_size` / `vocab_size` widths. |
| Availability | `support()` keys equal the standard value set (plus `linear_attn.*` on a hybrid, minus `mlp.*` where no block has an MLP); every `EXPECTED_UNAVAILABLE` key reports the substring and every other key `None`; on block 0 an available value is an `EProperty` on its class, a missing module says `no ... module`, an unavailable value raises `Unavailable` with the reason. |
| Boundary values | `layer_output` equals `.output` (or its first element); `attention_output`, `mlp_output`, `layer_output` read `hidden_size`-wide tensors on every block; `input + attention_output + mlp_output == layer_output` on every block at 8 ulp; `self_attn.input` and `mlp.input` equal `ATTENTION_NORM`'s and `MLP_NORM`'s outputs (or the block input); block 0 and `lm_head.output` equal the raw `TransformersModel`; assigning each contribution times zero, and zeroing `layer_output[:]`, moves the logits. |
| The pattern | shape `[batch, num_heads, seq, seq]`, `lm_head.weight`'s dtype, rows sum to one (or lie in `(0, 1)` with `ATTENTION_SINK`), lower-triangular; first and last blocks differ and two traces agree; a random pattern and a zeroed head both move the logits (a read can be causally inert); every available value on `FAMILY.Attention` whose key reads inside the forward (`inside_forward()`) reads a tensor on every attention block, one trace per value. |
| The interior | q `[b, heads, s, qk_head_dim]`, k/v `[b, kv_heads, s, ·]`, scores and pattern `[b, heads, s, s]`, head outputs `[b, s, heads, ·]`, `pattern_from_scores(scores) == pattern`; zeroing each of the five interior values moves the logits and zeroed head outputs give a position-independent contribution; in-place zeroing of scores and head outputs moves the logits, queries too unless `REFUSES_IN_PLACE_QKV`. |
| Methods | `skip_layers(1, last)` hands block 0's output straight through and the logits equal `project_on_vocab` of it; `skip_with=zeros` zeroes block 0's output and block 1's input; `steer` moves the last position by `2 * vector` and nothing else; `project_on_vocab` on the last block equals `logits`, `get_topk_closest_tokens(k=3)` returns one dict of three entries summing to at most one. |
| Layouts | every value with a `layout` on the root, a block, its attention, its MLP and one mixer is an instance of its named layout (`Residual`, `Pattern`, ...: a `jaxtyping` type) with each named axis matching the model's sizes; one value per trace; at least 11 checked. |
| The input | `input_ids`, `input_size`, `attention_mask` are `[1, n]`, the mask all ones, `token_embeddings` `n` long, the three names in the repr; assigning another prompt's ids and mask reproduces its logits; assigning `input_size` raises `AttributeError`. |
| The root | `logits` equals `model.output.logits` and softcapped `lm_head.output`; assigning `logits * 0` zeroes `tracer.result.logits`; `token_embeddings` equals `embed_tokens.output` and assigning it moves the logits; `next_token_probs` equals `logits[:, -1].softmax(-1)`, is in the repr and refuses assignment; `num_layers`, `num_heads`, `hidden_size`, `num_kv_heads`, `q_proj` / `o_proj` widths and the MLP width match the tensors; each block's `self_attn` and `mlp` sizes match that block's weights (`test_per_module_sizes_match_each_block`); the block repr lists the values. |
| Mixture (a `Moe` on some block) | `support()` matches `MOE_UNAVAILABLE`; `num_experts` / `top_k` are the modules'; `expert_outputs.sum(2) == routed_output` and `routed_output + shared_expert_output` is the mixture's output; zeroing a slot, ablating an expert, rerouting a token (against a hand computation) and writing `router_logits` act as stated; values and assignments are per invoke; `expert_outputs` is unavailable without a grouped experts implementation. |

Run it from the nnterp repo root (`git rev-parse --show-toplevel` must print the nnterp
checkout): `HF_HUB_OFFLINE=1 pytest tests/families/test_<model_type>.py`. A failure in
`test_every_source_value_resolves_on_every_layer` names the operation the forward does
not have; [finding-source-ops.md](finding-source-ops.md) is how to find the one it does.

The template above, written as a standalone module for `gpt2` and passed to
`nnterp.families.register()` at the top of a test file that subclasses `FamilySuite`
with GPT-2's `NATIVE` and `REFUSES_IN_PLACE_QKV`, passes the whole suite on
`hf-internal-testing/tiny-random-gpt2` with `model.family` being the standalone module.

## Gotchas

- **The file name is the registry.** `MODEL_TYPES` must be `("<file stem>",)`.
- **Import the modeling module only inside the family module.** An import at
  `nnterp/__init__.py` or in `components/` would load transformers modeling code on
  `import nnterp`.
- **Import nnterp (or nnsight) before any `transformers.models...` import** in a script
  or test; the reverse order segfaults at import on this stack. nnterp's `conftest.py` is
  the one line `import nnsight` for that reason.
- **Every family test loads eager.** Values still unavailable on that checkpoint go in
  `EXPECTED_UNAVAILABLE` with a substring of the reason.
- **Interior values bind at different points of the forward.** A family-specific test
  that reads two in one trace must read them in forward order.
- **The class-scoped `model` fixture is shared.** A test that mutates the config
  (`config.reorder_and_upcast_attn = True` in `test_gpt2.py`) restores it in a `finally`.
  A flag that changes which operations the forward runs (Falcon's `alibi`) gets its own
  suite class on a patched copy of the checkpoint (`_alibi_checkpoint` in
  `test_falcon.py`), so every test runs on that branch.
- **`known()` and `all_families()` list the shipped modules only.** A registered family
  is reached through `lookup` and `model.family`, and appears in the `UnsupportedFamily`
  message's list.
- **No version conditionals in a family module.** One module targets the transformers
  nnterp is developed on; a release that moves an op is handled by updating the string
  (the procedure is in [finding-source-ops.md](finding-source-ops.md)).

## Related

- [descriptors.md](descriptors.md): what each descriptor in these modules takes.
- [finding-source-ops.md](finding-source-ops.md): where the operation names come from.
- nnterp repo: `docs/extending/adding-a-family.md`, `docs/developing/testing.md`,
  `docs/reference/families.md` (every shipped family's overrides in one table).
