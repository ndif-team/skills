---
name: nnterp
description: Write interpretability code once on nnterp's StandardizedTransformer and run it on its 98 transformer families (GPT-2, Llama, Qwen, Gemma, OLMo, Granite, DeepSeek, DeltaNet and Mamba hybrids, 39 MoE families) and on vision-language models loaded with task="image-text-to-text" (model.vision, the tower blocks vision.layers[i], model.projector, vision.image_token_mask, vision.image_features): one vocabulary (model.layers[i].self_attn / .linear_attn / .mlp, model.norm, model.lm_head) and standard values (layer_output, attention_output, mlp_output, layer_mid, mlp_neurons, attention_probabilities, logits), plus support(), skip_layers, steer, project_on_vocab. Load it for code that reads or edits the residual stream, a sublayer's contribution, attention, expert routing, recurrent state or an image's path into the text model on more than one architecture, or that would branch on per-family module paths; and when a result on Gemma-4, Granite, Doge, ZAYA or DeepSeek-V4 looks wrong (their blocks are not a plain sum).
---

# nnterp

nnterp is a thin layer on nnsight 0.8. `StandardizedTransformer` is a
`TransformersModel` whose envoy tree also answers to one set of names on every
family, and whose blocks carry *standard values* (`layer_output`,
`attention_output`, `mlp_output`, `attention_probabilities`, `logits`, ...) that
mean the same thing everywhere. Everything nnsight does (`trace`, `generate`,
`.save()`, `tracer.invoke`, `tracer.iter`, `.source`, `remote=True`) works
unchanged; this skill covers only what nnterp adds. Load the `nnsight` skill for
the underlying API.

Verified on nnterp `v2.0.0` at `3fb2cc0` (98 families) with nnsight 0.8.2 and transformers 5.20.2.
nnterp needs nnsight 0.8 or later: the families alias `o_proj` and `down_proj` with dotted
rename keys (`"attn.c_proj": "attn.o_proj"` on GPT-2), which older nnsight cannot bind.

## Orientation

<!-- test: setup -->
```python
import warnings
import torch
import nnsight
import nnterp
from nnterp import StandardizedTransformer, Unavailable

# device= picks the device; device_map="cpu" is ignored by the pipeline nnsight loads through
model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"

ids = model.tokenizer(" Paris", add_special_tokens=False).input_ids
assert len(ids) == 1, ids                  # one token, or it is not one logit
paris = ids[0]
```

```python
with model.trace(prompt):
    x = model.layers[5].input.save()                            # residual stream entering block 5
    attn = model.layers[5].self_attn.attention_output.save()    # what attention adds to it
    mlp = model.layers[5].mlp.mlp_output.save()                 # what the MLP adds
    resid = model.layers[5].layer_output.save()                 # the stream leaving the block
    logits = model.logits.save()

assert model.family.__name__ == "nnterp.families.gpt2" and len(nnterp.families.known()) == 99   # 98 families + kimi_k25
assert (model.num_layers, model.hidden_size, model.num_heads) == (12, 768, 12)
assert x.shape == attn.shape == mlp.shape == resid.shape == (1, 10, 768)
assert torch.equal(x + attn + mlp, resid)                       # the contribution identity, exact in fp32
assert logits[0, -1].argmax() == paris
```

`model.layers[5]` *is* `model.transformer.h[5]` (an alias, not a copy), so native
paths keep working.

## The things that break agent-written nnterp code

**1. The target token picks BOS.** `tokenizer.encode(" Paris")[0]` is the BOS
token on Llama-3 and Gemma tokenizers, and every recipe then runs "fine" and
prints 0.000; `input_ids[-1]` picks the wrong piece of a multi-token word. Take
`model.tokenizer(" Paris", add_special_tokens=False).input_ids[0]` and assert the
list has one element (above). On a Llama-2 / Mistral sentencepiece tokenizer
`" Paris"` is `['▁', 'Paris']` and `[0]` is the lone space: tokenize `"Paris"`
there (`['▁Paris']`). A word that splits (`' Par', 'is'` on Granite) is not one
logit; pick another word or use `nnterp.prompt_utils.get_first_tokens`.

**2. The attention interior needs `attn_implementation="eager"`.** The default
(`sdpa` on most families) leaves `attention_probabilities`, queries, keys,
values, scores and head outputs unavailable. `support()` reads the config, so it
answers without loading weights:

```python
plain = StandardizedTransformer("openai-community/gpt2")        # meta device, transformers' default
reason = plain.support(layer=0)["self_attn.attention_probabilities"]
assert reason == "read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'"
assert plain.support(layer=0)["self_attn.attention_output"] is None    # boundary values never need eager
assert plain.support(layer=0)["self_attn.attention_premix"] is None    # the heads at o_proj: a module boundary
```

**3. `support()`, not `hasattr`.** `hasattr(envoy, value)` is never `False`: it
raises `nnterp.Unavailable` for an unavailable value, and outside a trace an
available one raises `ValueError` (or `SourceNotAvailable` for one inside the
attention call). Ask `model.support()` outside the trace.

<!-- test: expect-error Unavailable -->
```python
hasattr(plain.layers[0].self_attn, "attention_probabilities")     # Unavailable, not False
```

**4. The values are tensors. Never index `[0]` to unwrap.** `layer_output`,
`attention_output` and `mlp_output` are the tensor on every family, including
tuple blocks (GPT-J, BLOOM, MPT, Falcon) and every attention module, whose raw
`.output` is `(attn_output, attn_weights)`. `[0]` selects batch row 0, silently:

```python
with model.trace(prompt):
    raw = model.layers[3].self_attn.output.save()
    wrong = model.layers[3].layer_output[0].save()

assert isinstance(raw, tuple) and wrong.shape == (10, 768)       # batch row 0, not "the tensor"
```

**5. `attention_output` is the contribution, and not every block is a plain sum.**
`layers[i].input + attention_output + mlp_output == layer_output` defines the two
contributions on sequential, parallel and post-norm blocks (on Gemma-2/3,
OLMo-2/3, EXAONE-4 they are the post-norms' outputs; on BLOOM/MPT the tensor
before an add inside the module). Five families break the plain sum, and naive
direct logit attribution there is wrong by orders of magnitude:

| family | the block | what a term from block `i` reaches the last stream as |
|---|---|---|
| Gemma-4 | `(input + attn + mlp [+ per_layer_output]) * layer_scalar` | times every `layer_scalar` from `i` on (0.005 to 0.99 each) |
| Doge, ZAYA | the stream itself is rescaled per channel by each merge's learned parameters | times the gates of every later merge |
| Granite line, HyperCLOVA X | contributions are the scaled terms (`* residual_multiplier`) | the plain sum holds; the head scales the logits (`logits_scaling`) |
| DeepSeek-V4 | `layer_output` is `[batch, seq, streams, hidden]` (`Streams`) | mixed by `attention_comb` / `mlp_comb`; only the stream mean is additive |

The recipes (stream weights for Gemma-4 and Doge, the stream formula for
DeepSeek-V4): [references/values.md](references/values.md).

**6. Reads follow the forward, within one invoke.** `input_ids` before any
block; within a block `layer_input`, `attention_norm_output`, the attention's
interior (the queries and keys before the rotary, then the queries, keys, values and
`interface`, scores, `attention_probabilities`, `attention_premix`),
`attention_output`, `layer_mid`, `mlp_norm_output`, `mlp_input`, `mlp_activation`,
`mlp_neurons`, `mlp_output`, `layer_output` (`model.order(i)` lists a block's
values in forward order); block 3 before block 5; `final_norm_output`,
`lm_head_output`, `logits` and `next_token_probs` last. A late read raises
`OutOfOrderError`, which names an internal location (`...attention_interface_1.fn.i0`),
not the value:

<!-- test: expect-error OutOfOrderError -->
```python
with model.trace(prompt):
    late_attn = model.layers[3].self_attn.attention_output.save()
    late_pattern = model.layers[3].self_attn.attention_probabilities.save()   # inside the attention: too late
```

When code decides which values to read or edit (a list of sites, a sweep over several
values per block, a port of code that hooked modules in its own order), sort the sites by
`model.rank(name, layer)` before the trace instead of keeping a table of what comes first:
that table differs per family (on Falcon without alibi the values come before the queries;
a mixture's shared expert runs first on some families and last on others), and `rank` is
measured on the loaded checkpoint. `model.order(i)` lists block `i`'s values in that order,
equal ranks sharing one location. Both are asked outside the trace, by `support()`'s
dotted names; `rank` raises `Unavailable` for a value the checkpoint lacks:

```python
sites = [("layer_output", 3), ("self_attn.attention_probabilities", 3), ("mlp.mlp_output", 1)]
sites.sort(key=lambda site: model.rank(*site))                 # (block, r): forward order on any family
assert sites == [("mlp.mlp_output", 1), ("self_attn.attention_probabilities", 3), ("layer_output", 3)]
assert model.order(3)["self_attn.attention_probabilities"] < model.order(3)["self_attn.attention_output"]

def value_at(name, layer):                                     # "self_attn.attention_probabilities" -> (envoy, attribute)
    host = model.layers[layer]
    *path, attr = name.split(".")
    for part in path:
        host = getattr(host, part)
    return host, attr

saved = []
with model.trace(prompt):
    for name, layer in sites:                                  # issued in rank order: no OutOfOrderError
        host, attr = value_at(name, layer)
        saved.append(getattr(host, attr).save())

assert [tuple(s.shape) for s in saved] == [(1, 10, 768), (1, 12, 10, 10), (1, 10, 768)]
```

An edit lands at its value's location as a read does, so a list of edits sorts the same
way. The ranks also explain a failure after the fact: compare them for the reads around an
`OutOfOrderError` or a shifted `tracer.iter` list.

Three variants do not raise. Under `generate` + `tracer.iter`, an out-of-order
read *binds the next step's value*, so per-step lists come back shifted by one. A
read pinned to a step the run never reaches (past `max_new_tokens`, or that
shifted read on the last step) cuts the block short with a `UserWarning`: later
names are unbound, so look for the warning when a saved name is missing. On a Mamba-1 decode step, `attention_head_outputs` read before
`state_output` returns the next step's state ([references/recurrent-mixers.md](references/recurrent-mixers.md)).
`skip_layers` consumes `layers[start].input`: read it first or not at all.

**7. On BLOOM and XGLM an attention-interior edit in one invoke reaches every invoke.**
Their interior is served off `[batch * num_heads, ...]` tensors, so a one-forward sweep with
one invoke per head gives wrong numbers there (and on XGLM a read returns the whole batch):
run one trace per head. On the other families two invokes of one trace can both read and
edit `attention_probabilities` and `attention_scores`; at one location the invoke written
first is served first, so an empty `tracer.invoke()` after an invoke that edits the scores
reads the edited pattern.

**8. Overwriting `attention_scores` lifts the causal mask.** The scores are read
after masking, so `scores[:] = 0` attends to the future. Add to them
(`scores[..., k] += c`, or `-inf` to block a key) instead.

**9. GPT-2's and GPT-BigCode's queries, and MPT's queries, keys and values, are split
views torch will not edit in place**; so are Mamba-1's, and before the rotary CodeGen's,
DBRX's and GPT-NeoX's queries and keys, JetMoE's keys and latent attention's shared rope
key. torch refuses an in-place edit with grad on; assign instead (or edit under
`torch.no_grad()`):

<!-- test: expect-error RuntimeError -->
```python
with model.trace(prompt):
    model.layers[0].self_attn.attention_queries[:, 0] = 0      # RuntimeError: ... is a view ...
```

```python
with model.trace(prompt):
    model.layers[0].self_attn.attention_queries = model.layers[0].self_attn.attention_queries * 0
    edited = model.logits.save()

assert not torch.allclose(edited, logits)
```

GPT-2's and GPT-BigCode's keys and values take an in-place edit, and it lands.

**10. Recurrent mixers route kernels before the first trace.** On Mamba,
Falcon-Mamba and Jamba with `mamba_ssm` installed, a CPU run crashes with
`Expected u.is_cuda()` (even for `layer_output`) until
`nnterp.route_kernels(model.family, "torch")`; on DeltaNet hybrids the per-token
`state`/`states` need the same call; on Mamba-2 families `states` needs
`nnterp.chunk_per_token(model)`.

**11. Hybrids: decide `self_attn` vs `linear_attn` outside the trace**, with
`getattr(layer, "self_attn", None) is not None`; `getattr` inside a trace can trip
a served value. `linear_attn` is a DeltaNet, a Mamba-1 scan or a Mamba-2 mixer;
the same names mean different things on each.

**12. `envoys=` keys match a module type first, then a path (native, or any alias
spelling of it).** A family keys its envoys on module types, so a path key such as
`"layers.0.mlp"` binds but loses to the family's `GPT2MLP` key; displace a family envoy
by keying on its type.

**13. A silent segfault at import is pyarrow's, not nnterp's.** On glibc 2.31 (Ubuntu
20.04) with pyarrow installed, `import transformers.modeling_layers` before nnsight crashes
in pyarrow's jemalloc thread. Set `JE_ARROW_MALLOC_CONF=background_thread:false`, or import
`nnterp` (or `nnsight`) first.

**14. A vision-language wrapper loaded without `task="image-text-to-text"` sees no image.**
The default `task="text-generation"` load has no processor, so the tower never runs:
`model.vision.support()` is `{}`, `model.support()` has no `vision.` rows, and every tower
value raises `Unavailable`, inside a trace too, with the reason `a text-only load: no
processor, so no image reaches the model; load with task='image-text-to-text'`.

**15. Read `vision.image_token_mask` first and `vision.image_features` after the tower;
one image-carrying invoke per trace.** The mask comes off the inputs, like `input_ids`;
`image_features` is read at the scatter, after the tower's values and before
`layers[0].input`. Under `generate` on transformers 5.18 and later the images are encoded
before the first forward, so the tower's values come before the mask; mask then
`image_features` holds on every version. Two invokes of `model.trace(prompt, images=[...])` raise
`NotImplementedError: Can't batch these inputs`. Chat-message inputs that embed the image do
batch, and then `vision.image_features` is flat over the whole batch's image tokens in each
invoke, so a write in one invoke reaches every row. Put several images in one invoke, and
run clean and ablated as two traces.

**16. Boolean indexing with the mask flattens the batch.** `out[mask]` is
`[image_tokens, hidden]` over every row in row-major order (the order of `image_features`),
not `[batch, ...]`. Index one row (`out[0, mask[0]]`) or mask instead of indexing
(`out.masked_fill(~mask[..., None], 0)`) to keep the batch.

**17. A query or key moved to another position goes in before the rotary.**
`attention_queries` and `attention_keys` are rotated for the positions they ran at: a
rotated query written at another position or into another prompt is attended with the
wrong position. Write `attention_queries_prerope` / `attention_keys_prerope` instead; the
block rotates what you wrote for the position it lands at
([Before the rotary embedding](#before-the-rotary-embedding)). Hand-rotating
`self_attn.q_proj.output` is wrong too: Phi-3 has no `q_proj`, and Qwen3, OLMo-2/3 and
Gemma norm between the projection and the rotary. `self_attn.rotate` is the rotary
alone: HunYuan and Llama 4 norm the queries and keys after it, JetMoE tiles its keys,
Ministral 3 scales its queries, so there `attention_queries` is not `rotate(...)` itself.

**18. Read `self_attn.interface` once per forward, and call it before the scores.**
Reading it takes the attention call's arguments, so it goes where the queries, keys and
values are, in any order with them; calling it takes the function the forward is about to
run, so call it before reading or writing the scores, the pattern or the head outputs. Call
the result as often as you need. A second read after a call, or a call after those reads,
asks for a point the forward has passed and raises `OutOfOrderError`, under `tracer.iter`
too; a first read placed after them under `tracer.iter` binds the next step's call without
an error, as any late read does there (trap 6). Within one invoke, a padded batch included,
it reproduces the model bit for bit; across invokes only to rounding (about `1e-8` in
float32), so put prompts in one invoke (`model.trace([a, b])`) when that matters
([Calling the attention function again](#calling-the-attention-function-again-interface)).

## The vocabulary

Llama's block names, with the containers lifted out of the inner `.model`:

| standard name | GPT-2 | Llama | GPT-NeoX |
|---|---|---|---|
| `model.embed_tokens` | `transformer.wte` | `model.embed_tokens` | `gpt_neox.embed_in` |
| `model.layers[i]` | `transformer.h[i]` | `model.layers[i]` | `gpt_neox.layers[i]` |
| `model.layers[i].self_attn` | `...h[i].attn` | same | `...layers[i].attention` |
| `model.layers[i].self_attn.o_proj` | `...attn.c_proj` | same | `...attention.dense` |
| `model.layers[i].mlp` | same | same | same |
| `model.layers[i].mlp.act_fn` | `...mlp.act` | same | `...mlp.act` |
| `model.layers[i].mlp.down_proj` | `...mlp.c_proj` | same | `...mlp.dense_4h_to_h` |
| `model.norm` | `transformer.ln_f` | `model.norm` | `gpt_neox.final_layer_norm` |
| `model.lm_head` | same | same | same |

`self_attn.input` is what enters the attention and `mlp.input` what enters the
MLP, on every family. The norm aliases (`input_layernorm`, `post_attention_layernorm`)
mean different things per family (a sandwich block's `post_attention_layernorm` follows
the attention); read the boundary values `attention_norm_output`, `layer_mid` and
`mlp_norm_output` instead, which each family points at its own modules. `o_proj`,
`act_fn` and `down_proj` are the output projection, the activation and the down
projection on every family that has them. A recurrent block has `linear_attn` instead
of `self_attn`; OPT and XGLM have no `mlp` module; Mamba and Mamba-2 blocks have
neither attention nor MLP.

## The standard values

| value | on | layout | notes |
|---|---|---|---|
| `layer_input` | `layers[i]` | `Residual` | the stream entering the block, `layers[i].input` |
| `attention_norm_output` | `layers[i]` | `Residual` | the normed stream the attention is called with (`input_layernorm.output`) |
| `layer_mid` | `layers[i]` | `Residual` | the stream between the sublayers, `layer_input + attention_output`; none on a parallel block |
| `mlp_norm_output`, `mlp_input` | `layers[i]` | `Residual` | the MLP's pre-norm output; what the MLP is called with (`mlp.input`) |
| `layer_output` | `layers[i]` | `Residual` | the stream leaving the block; `Streams` on DeepSeek-V4 |
| `attention_output` | `self_attn` / `linear_attn` | `Residual` | the contribution |
| `attention_premix` | `self_attn` | `HeadOutputs` | `o_proj.input` split by head, after any gate; any `attn_implementation` |
| `mlp_activation`, `mlp_neurons` | `mlp` | `Neurons` | `act_fn.output`; `down_proj.input` (`act(gate) * up` on a gated MLP, the activation on a plain one); none on a routed mixture |
| `mlp_output` | `mlp` | `Residual` | the contribution |
| `attention_queries_prerope`, `attention_keys_prerope` | `self_attn` | `Queries`, `Keys` | what the rotary embedding receives: after the projection and any q/k norm, before the rotation; any `attn_implementation`; [below](#before-the-rotary-embedding) |
| `attention_queries`, `_keys`, `_values` | `self_attn` | `Queries`, `Keys`, `Values` | seq on axis 2; queries and keys after the rotary; keys/values before `repeat_kv`; eager |
| `attention_scores` | `self_attn` | `Pattern` | masked, scaled, pre-softmax; eager |
| `attention_probabilities` | `self_attn` | `Pattern` | the tensor the values are mixed with; eager |
| `attention_head_outputs` | `self_attn` | `HeadOutputs` | `[batch, seq, heads, head_dim]`, before the output projection; eager |
| `interface` | `self_attn` | a function, `f(Queries, Keys, Values) -> (HeadOutputs, Pattern)` | the attention function the block called, bound to this call's mask, scale, dropout and sliding window; read-only; eager; [below](#calling-the-attention-function-again-interface) |
| `router_logits`, `expert_weights`, `expert_indices`, `expert_outputs`, `routed_output`, `shared_expert_output` | `mlp` on a `Moe` | `RouterLogits`, `ExpertWeights`, ... | [references/mixture-of-experts.md](references/mixture-of-experts.md) |
| `state_input`, `state_output`, `state`, `states`, `decays`, `betas`, ... | `linear_attn` | `State`, `Gates`, `Scan*`, `SSD*` | [references/recurrent-mixers.md](references/recurrent-mixers.md) |
| `per_layer_output` | `layers[i]` (Gemma-4 E2B/E4B) | `Residual` | the block's third add |
| `attention_post`, `attention_comb`, `mlp_post`, `mlp_comb` | `layers[i]` (DeepSeek-V4) | `StreamWeights`, `StreamMixing` | the hyper-connection weights |
| `final_norm_output` | root | `Residual` | `norm.output`: the stream the head projects |
| `lm_head_output` | root | `Logits` | `lm_head.output`: the head's logits before a softcap or scale |
| `logits` | root | `Logits` | the output's `.logits` (softcapped, scaled); `lm_head_output` is raw |
| `next_token_probs` | root | `NextTokenProbs` | `logits[:, -1].softmax(-1)`, in the model dtype; read-only |
| `token_embeddings` | root | `Residual` | `embed_tokens.output`; `layers[0].input` is what enters block 0 |
| `input_ids`, `attention_mask`, `input_size` | root | `Tokens` | read first; the first two assignable |
| `layer_output`, `attention_output`, `mlp_output` | `vision.layers[i]` (`.self_attn`, `.mlp`) | `Patches` | the tower's stream and contributions, `[images, patches, vision_hidden]`; the interior as on a text block |
| `patch_embeddings`, `tower_output` | `vision` | `Patches` | the patch embedding's output; the last block's stream after `vision.norm`, before pooling or adapter |
| `image_token_mask` | `vision` | `ImageTokenMask` | `[batch, seq]` bool, `input_ids == image_token_id`; read first; read-only |
| `image_features` | `vision` | `ImageFeatures` | `[image_tokens, hidden]`, flat: what the wrapper scatters in, `layers[0].input[mask] == image_features` |

Every layout is a named `jaxtyping` alias (`nnterp.components.Residual`, `Pattern`,
...; the root's from `nnterp.standardized`): `value.layout` is that alias,
`value.dims` its axes, and `print(envoy)` lists each value as
`(name) -> Layout [axes]: description`:

```python
from nnterp.components import Pattern

assert type(model.layers[0].self_attn).attention_probabilities.layout is Pattern
print(type(model.layers[0].self_attn).attention_probabilities)
# (attention_probabilities) -> Pattern [batch num_heads query key]: The attention pattern the values are mixed with
```

An axis that is a fact about the model is named after the size that holds it (`hidden_size`,
`num_heads`, `num_kv_heads`, `head_dim`, `vocab_size`, `num_experts`; the constants `HIDDEN`,
`HEADS`, ... in `nnterp.components`), so `nnterp.axis_size(envoy, axis)` resolves it to a
number for a value served on that envoy, the block's own count where blocks differ. Use it
to check or allocate shapes without naming a family's config key; `batch`, `seq`, `query`
and `key` are the tensor's own and raise `ValueError`:

```python
from nnterp import axis_size

assert type(model.layers[0].self_attn).attention_probabilities.dims == ("batch", "num_heads", "query", "key")
assert axis_size(model.layers[0].self_attn, "num_heads") == 12 and axis_size(model.layers[0], "hidden_size") == 768
```

**Between and inside the sublayers.** The module boundaries are values too, so a recipe
that wants the stream after the attention, the normed input of a sublayer, a neuron or a
head's output at the projection never names a per-family module:

```python
with model.trace(prompt):
    block = model.layers[5]
    x = block.layer_input.save()
    normed = block.attention_norm_output.save()          # what the attention is called with
    premix = block.self_attn.attention_premix.save()     # o_proj's input by head: [batch, seq, heads, head_dim]
    attn = block.self_attn.attention_output.save()
    mid = block.layer_mid.save()                         # the stream between the sublayers
    neurons = block.mlp.mlp_neurons.save()               # down_proj's input: [batch, seq, intermediate_size]
    head = model.lm_head_output.save()                   # lm_head's own output, before any softcap

assert torch.equal(mid, x + attn) and normed.shape == x.shape
assert premix.shape == (1, 10, 12, 64) and neurons.shape == (1, 10, 3072)
assert torch.equal(head, logits)                         # GPT-2's head applies nothing after lm_head
```

For one head, `attention_premix` is the value to ablate or attribute when the model is not
loaded eager: it is served under any `attn_implementation`, and on a plain attention it is
the same tensor as `attention_head_outputs`, which needs eager (on Qwen3-Next and Laguna the
premix is after the gate, the head outputs before it). For one neuron, `mlp_neurons` is the
per-neuron coefficient the down projection sums into the stream; `mlp_activation` is
`act_fn`'s output, which on a gated MLP is not yet multiplied by the up projection.

Each is defined on the pre-norm block; a family whose block is built otherwise points it
at its own module or marks it unavailable with the reason (a parallel block has no
`layer_mid`, a routed mixture no `mlp_activation` / `mlp_neurons`). Per block structure,
gated vs plain MLPs, and the root pair: [references/values.md](references/values.md#the-module-boundaries).

**Sizes.** The root's (`num_layers`, `hidden_size`, `vocab_size`, `num_heads`,
`num_kv_heads`, `head_dim`, `qk_head_dim`, `intermediate_size`) are the config's,
spelled per family. Where blocks differ (Gemma-4's full blocks, MiMo-V2-Flash,
Laguna, Gemma-4 E2B's double-wide MLPs) read the block's own:
`layers[i].self_attn.num_heads` / `.num_kv_heads` / `.head_dim` / `.qk_head_dim`
and `layers[i].mlp.intermediate_size` (one expert's on a mixture).

## The same script on another family

<!-- test: setup -->
```python
llama = StandardizedTransformer("HuggingFaceTB/SmolLM2-135M-Instruct", device="cpu", dispatch=True, attn_implementation="eager")
lparis = llama.tokenizer(" Paris", add_special_tokens=False).input_ids[0]
```

```python
with llama.trace(prompt):
    x = llama.layers[5].input.save()
    k = llama.layers[5].self_attn.attention_keys.save()
    pattern = llama.layers[5].self_attn.attention_probabilities.save()
    attn = llama.layers[5].self_attn.attention_output.save()
    mlp = llama.layers[5].mlp.mlp_output.save()
    resid = llama.layers[5].layer_output.save()
    ll = llama.logits.save()

n = len(llama.tokenizer(prompt).input_ids)
assert (llama.num_layers, llama.hidden_size, llama.num_heads, llama.num_kv_heads) == (30, 576, 9, 3)
assert resid.shape == (1, n, 576) and resid.dtype == torch.bfloat16     # the checkpoint's dtype
assert k.shape == (1, 3, n, 64) and pattern.shape == (1, 9, n, n)       # kv_heads vs query heads
assert torch.equal(x + attn + mlp, resid)
assert ll[0, -1].argmax() == lparis
```

Same body, different family, no `[0]`. In bf16 the pattern's rows sum to 1
within about 4e-3, and `next_token_probs` is a bf16 softmax (off by up to ~2e-3):
compare with `atol`, and take `logits[:, -1].float().softmax(-1)` for metrics.

## Vision-language models

<!-- test: setup -->
```python
from PIL import Image

vlm = StandardizedTransformer("trl-internal-testing/tiny-LlavaForConditionalGeneration",
                              task="image-text-to-text", device="cpu", dispatch=True)   # the processor comes with the task
image = Image.new("RGB", (64, 64), "red")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]}]
vprompt = vlm.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)   # places the image token
```

```python
with vlm.trace(vprompt, images=[image]):
    mask = vlm.vision.image_token_mask.save()            # [batch, seq] bool; first: it comes off the inputs
    patches = vlm.vision.layers[0].layer_output.save()   # Patches: [images, patches, vision_hidden]
    features = vlm.vision.image_features.save()          # [image_tokens, hidden], flat
    first = vlm.layers[0].input.save()
    out = vlm.layers[0].layer_output.save()
    vlm.layers[1].layer_output[mask] = 0                 # a one-sided edit: the image positions only
    edited = vlm.logits.save()

assert vlm.family is nnterp.families.llama               # the family is the text model's
assert mask.shape == (1, 592) and mask.sum() == 576
assert patches.shape == (1, 577, 16)                     # CLIP: the CLS token first, then 576 patches
assert features.shape == (576, 16)
assert torch.equal(first[mask], features)                # the scatter identity, exact
assert out[mask].shape == (576, 16) and out[~mask].shape == (16, 16)   # image rows, text rows
assert edited.shape == (1, 592, 32064)
```

`model.vision` is the tower (a `Vision`), `vision.layers[i]` its blocks with
`self_attn` and `mlp` and the same block values (the module boundaries too), laid out `Patches`,
and `model.projector` the last module before the image enters the text model;
`model.layers` stays the text model's. The tower's sizes are on `model.vision`
(`num_layers`, `vision_hidden_size`, `num_heads`, `patch_size`, ...). The one rule for
`vision.image_features`: it is read at the scatter, the tensor the wrapper writes
into the token embeddings, so `layers[0].input[mask] == image_features` on every
wrapper; ablate or patch the image there. `projector.output` differs from it on
LLaVA-NeXT, LLaVA-OneVision, Gemma 4 unified and Qwen2.5-VL. Names per tower (SigLIP, CLIP,
Pixtral, the Qwen ViT, Llama 4, Gemma 4), the Qwen ViT's packed row and DeepStack,
`generate`, availability and the 16 host families:
[references/vision.md](references/vision.md).

## Writing values

In-place edits reach the model because the value is the live tensor;
assignment replaces it. On a tuple block or an attention module the other
elements are put back for you:

```python
with model.trace(prompt):
    model.layers[5].self_attn.attention_output[:, -1] = 0       # ablate attention at the last position
    model.layers[5].mlp.mlp_output[:] = 0                        # ablate the MLP everywhere
    model.layers[8].layer_output[:, -1, :] *= 2                  # scale the stream leaving block 8
    changed = model.logits.save()

with model.trace(prompt):
    model.layers[5].layer_output = model.layers[5].layer_output * 0     # assign a new tensor
    zeroed = model.logits.save()

assert not torch.allclose(changed, logits) and not torch.allclose(zeroed, logits)
```

On Granite (and GraniteMoE, ZAYA) a contribution is a scaled copy divided back
on write: in bf16 an edit at one position shifts every position by up to half the
edit's own effect; compare such edits in float32. On Gemma-4 `attention_keys` and
`attention_values` are served as a copy private to each block: an edit, in place or by
assignment, changes what that block attends with and nothing else; to change what every
borrower of a KV-sharing source block attends with, edit the source block's `k_proj` /
`v_proj` output.

## Before the rotary embedding

`attention_queries` and `attention_keys` are rotated for the positions they run at.
`self_attn.attention_queries_prerope` and `self_attn.attention_keys_prerope` are the same
queries and keys one step earlier, as the block's rotary embedding receives them: after the
projection and any query or key norm (Qwen3's `q_norm` / `k_norm`, OLMo-2/3, Gemma-3/4),
before the rotation, laid out `Queries` and `Keys` (the keys before the cache and
`repeat_kv`). `self_attn.rotate(x, positions)` applies that block's own rotary embedding to
any tensor laid out like them, at the absolute positions you give it, and needs no trace:
`rotate(attention_queries_prerope, range(seq))` is `attention_queries` bit for bit. Both
values are read on the attention module's own forward, not inside the eager interface, so
they are served under any `attn_implementation`, and they come before `attention_queries`
in a trace (`model.order(i)` ranks them first).

Use them instead of `attention_queries` / `attention_keys`:

- **To move a query or key to another position or prompt.** A rotated query carries its
  position in its rotation; written elsewhere it is attended with the wrong one. Write the
  value before the rotary instead: the block rotates what you wrote for the position it
  lands at, so the transplant is one assignment.
- **For an edit that must come before the rotation**: a direction found on the unrotated
  queries or keys (a probe, a steering vector), which the rotary would turn by a different
  angle at every position.
- **To recompute attention with keys from elsewhere**: `rotate` the source keys at the
  destination positions, outside the trace.

And instead of `self_attn.q_proj.output`, whose layout nnterp does not promise (Phi-3 and
GPT-NeoX fuse the queries, keys and values into one projection; Phi-3 has no `q_proj`), and
which on Qwen3, OLMo-2/3 and Gemma misses the norm between the projection and the rotary.
GPT-2 has no rotary, so this runs on SmolLM2 (`llama`, loaded above):

```python
attn = llama.layers[1].self_attn

source, target = "The capital of France is Paris", "My favourite city in the whole world is Rome"
with llama.trace(source):
    q_src = attn.attention_queries_prerope.save()       # [1, num_heads, seq, head_dim], not rotated yet
    k_src = attn.attention_keys_prerope.save()          # [1, num_kv_heads, seq, head_dim]
with llama.trace(target):
    attn.attention_queries_prerope[:, :, -1] = q_src[:, :, -1]   # the source's last query, rotated for the target's last position
    moved = attn.attention_queries.save()
    logits = llama.logits.save()

seq = k_src.shape[2]
assert torch.equal(moved[:, :, -1:], attn.rotate(q_src[:, :, -1:], [moved.shape[2] - 1]))
later = attn.rotate(k_src, range(5, 5 + seq))           # the source's keys as if five positions later
assert q_src.shape == (1, 9, seq, 64) and later.shape == k_src.shape == (1, 3, seq, 64)
```

On a block without a rotary embedding both values are `Unavailable`, and `rotate` raises
`Unavailable` with the same reason, which `support()` gives per block:

```python
assert plain.support(layer=0)["self_attn.attention_queries_prerope"] == (
    "no rotary embedding: GPT-2 adds learned absolute position embeddings (wpe) to the stream")
```

That is GPT-2, GPT-BigCode, GPT-Neo, OPT and XGLM (absolute positions), BLOOM, MPT and Falcon
with `alibi` (ALiBi), the attention blocks of Jamba, Nemotron-H and Kimi-Linear, and the NoPE
blocks: the full-attention blocks of Cohere 2, AFMoE and a hybrid EXAONE 4, the blocks
SmolLM3's and Llama 4's `no_rope_layers` mark, a Granite-SWA block whose `layer_rope_theta`
is 0. Under latent attention (DeepSeek-V2/V3, Kimi-K2, GLM-5, ...) `attention_keys_prerope`
is the one rope key per position that every head shares, `[batch, 1, seq, qk_rope_head_dim]`
(`SharedRopeKeys`). Assignment reaches the model on every family; an in-place edit is
refused on the split views of trap 9.

On `StandardizedVLLM` both values are served (the arguments of vLLM's own
`self.rotary_emb(positions, q, k)` call, private copies, rotated for where a write lands),
but **`rotate` runs inside a trace only** there: it calls the engine's rotary embedding, and
outside a trace raises `RuntimeError`. Neither value is served on the families vLLM runs
through its transformers backend (OLMo, OLMo 2, SmolLM3, StarCoder2) or under its latent
attention (DeepSeek-V2/V3, Kimi-K2, GLM-4.7-Flash): read them, and rotate, on a
`StandardizedTransformer` there. Where a family's values differ, positions under rope
scaling and `generate`, and the vLLM recipe:
[references/attention-interior.md](references/attention-interior.md#before-the-rotary-embedding-attention_queries_prerope-attention_keys_prerope-rotate).

## Calling the attention function again: `interface`

To recompute a block's attention with other keys or values, or to patch what some heads
read without touching the KV cache, call the block's own attention function.
`model.layers[i].self_attn.interface` is the function this block's forward called
(`eager_attention_forward`, or the family's own: Gemma-2's with its logit softcap, GPT-OSS's
with its sink column; the `_attn` method on GPT-J, GPT-Neo, CodeGen and GPT-NeoX-Japanese),
bound to everything else the call received this time: the mask, the scale, the dropout, the
sliding window. `attn.interface(q, k, v)` takes tensors in the layouts of
`attention_queries`, `attention_keys` and `attention_values` (keys and values `num_kv_heads`
wide: it repeats them for each group itself), with the live call's query and key lengths,
and returns `(head_outputs, pattern)` in the layouts of `attention_head_outputs` and
`attention_probabilities`; on the served tensors it is those two values bit for bit. It
computes and returns: nothing reaches the model, the cache, the other heads or the other
rows until you write a result. GPT-2 serves it (`model`, loaded eager above); its queries are
split views (trap 9), so edit a clone, as below:

```python
attn = model.layers[5].self_attn
fact = "The capital of France is"

with model.trace(fact):
    q, k, v = attn.attention_queries, attn.attention_keys, attn.attention_values
    heads, pattern = attn.interface(q, k, v)         # the function this call ran, run again on the same tensors
    probs = attn.attention_probabilities.save()      # read after the call: the pattern comes later in the forward
    served = attn.attention_head_outputs.save()
    heads, pattern = heads.save(), pattern.save()

assert torch.equal(heads, served) and torch.equal(pattern, probs)   # the same function on the same shapes
assert heads.shape == (1, 5, 12, 64) and pattern.shape == (1, 12, 5, 5)
```

Use it instead of writing `attention_keys` or `attention_values`:

- **For an edit of keys or values that must not reach the KV cache or the other heads of a
  group.** Writing `attention_keys` changes the keys every query head of the group reads, at
  every query row, and under `generate` an in-place edit stays in the cache for every later
  step (an assignment replaces the argument of that one forward). Call `interface` on
  edited copies instead and write only the heads you chose.
- **For a per-row patch**: compute with the edited tensors, then write the rows you want of
  `attention_head_outputs` (the last one, for the prediction).
- **For a recompute that must match the model bit for bit.** A hand-written
  `softmax(q @ k.T * scale)` needs the module's scale, misses Gemma-2's softcap and a sink
  column, and rounds differently in bf16 when its matmuls have other shapes than the
  model's; the call runs the model's function on the model's shapes.

Read it once and call the result as often as you need (trap 18). Under grouped-query
attention a key/value head edited in a copy moves the whole group in the result, and only
the heads you write reach the model; SmolLM2 has three query heads per key/value head
(`llama`, loaded above):

```python
attn = llama.layers[14].self_attn

with llama.trace(fact):
    interface = attn.interface                       # read once, with the queries, keys and values
    q, k, v = attn.attention_queries, attn.attention_keys, attn.attention_values
    k2 = k.clone()                                   # a copy: k is what the model attends with (and, under generate, the cache)
    k2[:, 1, 0] = 0                                  # key/value head 1, position 0
    clean, _ = interface(q, k, v)
    edited, _ = interface(q, k2, v)                  # called again: same mask, same scale
    attn.attention_head_outputs[:, -1, 4] = edited[:, -1, 4]   # head 4, last row: the only write
    moved = edited.ne(clean).flatten(0, 1).any(0).any(-1).save()

assert moved.nonzero().flatten().tolist() == [3, 4, 5]   # the result moves for the whole group; only head 4 is written
```

A key or query edited before the rotary goes in rotated for its position,
`interface(q, attn.rotate(k_prerope, positions), v)`, with `attention_keys_prerope` read
first, since the forward reaches it before the interface
([Before the rotary embedding](#before-the-rotary-embedding)). Inside `tracer.iter` it is
that step's call: on a decode step the queries have one row, the keys and values are the
whole cache, and the mask is the step's. On a model split tensor-parallel it takes the
gathered queries, keys and values and runs every head on every rank. In training mode with
a dropout the call draws its own mask. Chosen heads reading another run's keys and values
over a span, on the prompt and on every `generate` step, is the `patterns` skill's
head-selective K/V patching.

`support()` reports `self_attn.interface` unavailable where the values it works with are
(any `attn_implementation` but `"eager"`, with the same reason) and, with
`nnterp.components.OWN_ARITHMETIC` (`the queries, keys and values are not the arguments of
one call in this forward that returns the head outputs ...`), where no such call exists:
BLOOM, MPT, Falcon and XGLM do their own arithmetic, and DeepSeek-V4 rotates the
interface's output back before its projection. On a vision tower it is served on SigLIP,
CLIP, Pixtral, Gemma 4 and Llama 4; the Qwen ViT and RADIO make one call per image and
report that. `StandardizedVLLM` does not serve it: the engine's attention is a kernel that
writes the step's keys and values into its cache as it runs.

```python
eager_reason = plain.support(layer=0)["self_attn.attention_probabilities"]
assert plain.support(layer=0)["self_attn.interface"] == eager_reason   # sdpa: the reason of the values it works with
```

Everything else (decode steps, padded batches, the per-family calls, the cache under
`generate`):
[references/attention-interior.md](references/attention-interior.md#calling-the-attention-function-again-interface).

## Methods over the values

```python
torch.manual_seed(0)
direction = torch.randn(model.hidden_size)

with model.trace(prompt):
    before = model.layers[1].layer_output.save()
    model.skip_layers(2, 3)                                      # blocks 2 and 3 do not run (inclusive)
    after = model.layers[3].layer_output.save()
    model.steer(6, direction, factor=3.0, token_positions=-1)    # add to the stream leaving block 6, in place
    mid = model.layers[9].layer_output.save()
    lens = model.project_on_vocab(mid).save()                    # logit lens: the model's own head
    last = model.layers[-1].layer_output.save()
    out = model.logits.save()

assert torch.equal(before, after)
assert torch.equal(model.project_on_vocab(last), out)            # exact at the last block, on every family
print(model.get_topk_closest_tokens(mid[0, -1], k=3))            # [{token: probability}]
```

`project_on_vocab` is the family's whole head: norm, `lm_head`, then the softcap
(Gemma-2/4), `logit_scale` (Cohere), `/ logits_scaling` (Granite) or `hc_head`
(DeepSeek-V4). `steer` moves `vector` to the stream's device and dtype; scale the
factor to the stream's norm (`layer_output[0, -1].norm()`, which grows with depth).
`get_topk_closest_tokens` takes a residual-stream tensor, not logits, and keys on
the decoded string: byte tokens that decode to `'�'` collapse into one entry.

## `support()`

`model.support()` is every root and block value by dotted name: `None` when
available on every block, else `{block: reason}`. `model.support(layer=i)` is one
block, flat; `envoy.support()` one envoy. A module no block has (OPT's `mlp`) has
no key: `support().get("mlp.mlp_output", "absent")` on a family you do not know.
Reading an unavailable value raises `nnterp.Unavailable` with the same reason,
before the model runs. Reasons, `SourceNotAvailable`, guarding a script:
[references/availability-and-support.md](references/availability-and-support.md).

Before writing against an unfamiliar checkpoint, run the script (meta device, no
weights): it prints the family, the standard-name to native-path table, the
sizes, `support()` and a block's repr (with `--task image-text-to-text`, a
vision-language wrapper's tower names, sizes and `model.vision.support()` too):

```
python scripts/inspect_family.py <repo_id> [--layer N] [--eager] [--task image-text-to-text]
```

## Families

98 families; `nnterp.families.known()` lists 99 modules, counting `kimi_k25` (Kimi K2's
family under its wrapper's name). 16 of the families also host a vision tower on their
image-text-to-text wrappers (the family is the text model's: Llava resolves to `llama`);
the wrappers table is in [references/vision.md](references/vision.md#which-wrappers).
What a recipe must survive is grouped by quirk in
[references/api-reference.md](references/api-reference.md); the full per-family table is
nnterp's `docs/reference/families.md`.

A `model_type` with no family loads with the best-effort `nnterp.families.default` and a
`UserWarning` (`nnterp has no family for model_type ...`): it serves `layer_output`, the
module boundaries its aliases reach, the root values, the sizes and the attention
interior, not `attention_output`, `mlp_output`, `layer_mid`, `mlp_norm_output`, a
mixture's interior or `project_on_vocab`, and raises `UnsupportedFamily` when it cannot
standardize the checkpoint (no blocks, embedding, final norm or head under a name it
knows, or a stream of the wrong shape). `nnterp.families.covers(config.model_type)` says
before a load whether a shipped or registered family covers a checkpoint (`False` means
the default), and `nnterp.families.resolve(config)` is the family the load would use;
the `extending` skill adds a family, registers one, or passes one to a single load with
`family=`.

## The vLLM engine and several GPUs

`nnterp.StandardizedVLLM(repo_id, dispatch=True, ...)` is the same vocabulary, values
and layouts on nnsight's `VLLM` engine (60 families). Pick it for throughput: many
concurrent prompts, long generations, a served engine. Stay on `StandardizedTransformer`
for gradients, an edit of the attention pattern or a decode step's pattern,
`self_attn.interface`, `model.order()` / `rank()`, and the values the engine does not have (`lm_head_output`, a
gated MLP's `mlp_activation`, and `attention_mask`, since a request is one unpadded
sequence). Both are `nnterp.Standardized`, so a helper written against the standard
names takes either. What differs, from the repo's `docs/usage/vllm.md`:

<!-- test: skip -->
```python
from nnterp import StandardizedVLLM

vmodel = StandardizedVLLM("HuggingFaceTB/SmolLM2-135M-Instruct", dispatch=True, gpu_memory_utilization=0.2, max_model_len=1024)
layer = vmodel.layers[12]
with vmodel.trace("The Eiffel Tower is in the city of", temperature=0.0, max_tokens=1):   # sampling settings on trace
    stream_in = layer.layer_input.save()          # never layers[i].input: natively the positions
    stream = layer.layer_output.save()            # [1, tokens, hidden_size]: one prompt per invoke, a private copy
    vlogits = vmodel.logits.save()                # [1, 1, vocab_size]: the last position only
```

The pattern and scores are recomputed from the queries and keys (read-only, prefill only),
`self_attn.rotate` runs inside a trace only (it calls the engine's rotary embedding), and an
assigned value must keep the shape it was served with. vLLM itself is the `nnsight` plugin's
`vllm` skill.

A model too big for one GPU loads sharded with
`distributed_config=DistributedConfig(tp_size=N)` under `torchrun`: every value reads whole
on every rank (the attention interior and `mlp_activation` gathered where a rank holds a
piece, edits split back), except a split mixture's `expert_outputs`, a per-rank partial sum
(read `routed_output`). Read values unconditionally, never under a branch on the rank
(`docs/usage/loading.md`; the `nnsight` plugin's `tensor-parallel` skill).

## References

| File | Covers |
|---|---|
| [references/values.md](references/values.md) | the boundary values and the identity, tuple blocks, where families relocate a contribution, the non-additive families with their stream-weight recipes, root values, sizes, layouts |
| [references/attention-interior.md](references/attention-interior.md) | queries/keys/values/scores/probabilities/head outputs, family caveats (GPT-2, MPT, Falcon, sinks, latent attention, Gemma-4 borrowed keys), what edits do to the mask, under `generate`; the queries and keys before the rotary and `rotate` (moving a query or key to another position, per-family differences, rope scaling, on vLLM); `interface`, the block's attention function called again (edits of keys or values that must not reach the cache or the rest of a group, per-row patches, exact recomputes; where it is unavailable) |
| [references/mixture-of-experts.md](references/mixture-of-experts.md) | a `Moe`'s six values, usage / entropy / ablation / rerouting recipes with pad masking, bf16 and invoke caveats, per-family gaps |
| [references/recurrent-mixers.md](references/recurrent-mixers.md) | `LinearAttention` (DeltaNet), `SelectiveScan` (Mamba-1), `StateSpace` (Mamba-2): what each name means on each, kernels, per-token state, read-order traps |
| [references/availability-and-support.md](references/availability-and-support.md) | `support()` forms, the reasons, `Unavailable` vs `SourceNotAvailable`, guarding a script |
| [references/vision.md](references/vision.md) | vision-language models: the tower names per tower, the image values and `Patches`, where the image meets the text model, image vs text positions, inputs and read order, `generate`, the Qwen ViT, availability, the wrappers |
| [references/generation-and-helpers.md](references/generation-and-helpers.md) | values under `generate` with `tracer.iter`, `nnterp.prompt_utils`, `nnterp.nnsight_utils`, padding, `remote=True` |
| [references/api-reference.md](references/api-reference.md) | every exported symbol, and the 98 families grouped by quirk |

The nnterp repo's pages go deeper: `docs/usage/*.md` (one page per feature),
`docs/reference/families.md`, `docs/patterns/*.md`.

## Related skills

- `nnsight`: the underlying API (`.save()`, invokes, `tracer.iter`, `.source`, `rename`)
- `patterns`: logit lens, attribution, ablation, patching, steering, probing, sweeps written once against these values
- `extending`: a checkpoint nnterp does not know, overriding a value, a value of your own
- `debugging`: an error, a hang, an empty result
- `remote`: running on NDIF; nnterp must be installed server-side
