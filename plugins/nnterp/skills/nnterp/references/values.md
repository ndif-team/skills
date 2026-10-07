# The standard values

Every value below is an nnsight `eproperty` on a `Layer`, `Attention`, `Mlp`
(`Moe`), recurrent mixer or the root, read or written **inside a trace body**
like `.output`. This file is the per-value contract: what it is, its layout, how
it reads and writes, and where a family relocates it so the name still means the
same thing. The attention interior has its own file,
[attention-interior.md](attention-interior.md); a mixture's routing
[mixture-of-experts.md](mixture-of-experts.md); recurrent mixers
[recurrent-mixers.md](recurrent-mixers.md).

<!-- test: setup -->
```python
import torch
import nnsight
import nnterp
from nnterp import Attention, Layer, StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"
```

## The three block values and the identity

| value | what it is |
|---|---|
| `model.layers[i].layer_output` | the residual stream leaving block `i` |
| `model.layers[i].self_attn.attention_output` | what the attention sublayer adds to the stream (`linear_attn.attention_output` on a recurrent block) |
| `model.layers[i].mlp.mlp_output` | what the MLP sublayer adds to the stream |

The two contributions are *defined* by one identity, which holds on a sequential
block (GPT-2, Llama), a parallel block (GPT-NeoX, Phi, GPT-J, Falcon, Cohere) and
a post-norm block alike:

```
layers[i].input + attention_output + mlp_output == layers[i].layer_output
```

```python
with model.trace(prompt):
    x = model.layers[5].input.save()
    attn = model.layers[5].self_attn.attention_output.save()
    mlp = model.layers[5].mlp.mlp_output.save()
    out = model.layers[5].layer_output.save()

torch.testing.assert_close(x + attn + mlp, out)      # max abs diff 0.0 in fp32
```

Read the four in forward order. The identity is exact in fp32 and bf16 on
sequential blocks; a parallel block sums in another order and lands within a few
ulps of its dtype, so compare with a tolerance there. Gemma-4, Doge, ZAYA and
DeepSeek-V4 carry extra factors ([below](#blocks-that-are-not-a-plain-sum)). A
Mamba / Mamba-2 block is `input + linear_attn.attention_output`; a Nemotron-H block
is the input plus its one sublayer; Falcon-H1's has four terms.

## Tensor blocks and tuple blocks

A Llama, GPT-2 or GPT-NeoX block returns `hidden_states` alone; GPT-J, GPT-Neo,
CodeGen, GPT-NeoX-Japanese, BLOOM, MPT, Falcon, GLM-5, Bamba, Falcon-H1 and ZAYA
return a tuple with it first (`Layer.returns_tuple`). `layer_output` is the tensor
either way, the same object the block returned:

```python
with model.trace(prompt):
    raw = model.layers[2].output.save()
    out = model.layers[2].layer_output.save()

assert type(model.layers[2]).returns_tuple is False
assert torch.equal(raw, out)                          # GPT-2: a tensor block
```

<!-- test: skip -->
```python
bloom = StandardizedTransformer("bigscience/bloom-560m", device="cpu", dispatch=True)

with bloom.trace(prompt):
    raw = bloom.layers[2].output.save()               # a tuple on BLOOM
    out = bloom.layers[2].layer_output.save()         # the tensor
    bloom.layers[2].layer_output = out * 0            # replaces the first element, keeps the rest

assert isinstance(raw, tuple) and torch.equal(raw[0], out)
```

The same holds for `attention_output` on an attention module returning
`(attn_output, attn_weights)`: the value is the first tensor, an assignment
rewraps it. Skip a tuple block with `skip_layers` or `Layer.skip_with`, not
nnsight's `.skip(tensor)`, which hands a bare tensor where a tuple is expected.

## Reading, editing in place, assigning

In-place edits reach the model because the value is the live tensor; assignment
replaces it:

```python
with model.trace(prompt):
    clean = model.logits.save()

with model.trace(prompt):
    model.layers[5].self_attn.attention_output[:, -1] = 0       # ablate attention at the last position
    model.layers[5].mlp.mlp_output[:] = 0                        # ablate the MLP everywhere
    model.layers[6].layer_output[:, -1] += 1.0                   # steer the stream leaving block 6
    edited = model.logits.save()

with model.trace(prompt):
    model.layers[4].layer_output = model.layers[3].layer_output * 2     # assign a new tensor
    assigned = model.logits.save()

assert not torch.allclose(edited, clean) and not torch.allclose(assigned, clean)
```

Clone when you need the "before": a read after an in-place edit returns the
edited tensor. `model.layers[i].input` is the stream entering the block on every
family.

## Where the families relocate a contribution

The family's `Attention` or `Mlp` subclass points the value at what the block
adds; the identity is what you rely on, not `self_attn.output[0]`.

- **Post-norm (sandwich) blocks**: `gemma2`, `gemma3_text`, `gemma4_text`,
  `gemma4_unified_text`, `olmo2`, `olmo3`, `exaone4`, `flex_olmo`, `afmoe`, `glm4`,
  `hyperclovax`, and OLMo-Hybrid's attention blocks. The stream receives a
  post-sublayer norm's output, so `attention_output` is that norm's output
  (`post_attention_layernorm` on most; `post_self_attn_layernorm` on GLM-4, where
  `post_attention_layernorm` is the pre-MLP norm) and comes *after*
  `self_attn.output` in the forward. On OLMo-2/3, EXAONE-4 and FlexOlmo there is no
  pre-norm: `self_attn.input` is the block input.
- **Residual added inside the module**: BLOOM (both sublayers) and MPT (the MLP)
  return `input + contribution`; the value is the tensor before that add, read
  *before* the module's own `.output`. DBRX adds outside the attention module, so
  the base holds.
- **Falcon's copy**: the block adds the attention into the MLP's output tensor in
  place, so `mlp_output` is a copy taken as the MLP returns; a transform carries
  in-place edits and assignments back.
- **GPT-NeoX-Japanese**: on the block whose attention returns a `dense_bias`,
  `attention_output` is the output plus that bias (edits are carried back).
- **Granite line** (`granite`, `granitemoe`, `granitemoeshared`, `granitemoehybrid`,
  `granite_swa`, `granitemoe_swa`) and **HyperCLOVA X**: the block adds
  `h * residual_multiplier` (0.22 on granite-4.1-3b), so the contributions are the
  scaled terms and the plain identity holds. They are computed copies: an
  assignment is divided back, an in-place edit is carried back by a transform, a
  read with no edit leaves the forward bit-identical. In bf16 dividing the whole
  copy back shifts *every* position by up to half an edit's own effect (1e-7 in
  fp32): run position-specific edits in float32. `self_attn.output` /
  `mlp.output` stay unscaled, and so do a mixture's `routed_output` /
  `expert_outputs`.
- **Falcon-H1** scales each mixer's output by its µP multiplier before the add;
  the values are the block's own bindings of the products.

The skipped blocks below show the post-norm and residual-inside forms on public
checkpoints:

<!-- test: skip -->
```python
gemma = StandardizedTransformer("google/gemma-2-2b", device="cpu", dispatch=True)

with gemma.trace(prompt):
    raw_attn = gemma.layers[0].self_attn.output.save()            # (tensor, weights): before the post-norm
    post = gemma.layers[0].post_attention_layernorm.output.save()
with gemma.trace(prompt):
    attn = gemma.layers[0].self_attn.attention_output.save()

assert torch.equal(attn, post) and not torch.equal(attn, raw_attn[0])

with bloom.trace(prompt):
    x = bloom.layers[2].input.save()
    attn = bloom.layers[2].self_attn.attention_output.save()      # an op inside the module: before .output
with bloom.trace(prompt):
    raw_attn = bloom.layers[2].self_attn.output.save()

assert torch.allclose(raw_attn[0], x + attn)                     # the module returns input + contribution
```

## Blocks that are not a plain sum

On these families a term added in block `i` does not reach the last stream as
itself, and naive direct logit attribution ranks the wrong terms (on gemma-4-E2B
it named block 0's attention the top term, off by ~7,700%; on Doge-320M ~18x).
nnterp serves the contributions unscaled and computes no weights for you; the
recipes below do. Both are skipped here (no such checkpoint in the executed set)
and were verified on the pinned tiny checkpoints with the scales moved away from
one, in float32: the weighted sum equals the last `layer_output` to 1e-7, the
naive sum is off by 11 on a stream of norm ~1.

**Gemma-4** (`gemma4_text`, `gemma4_unified_text`). The block is Gemma-3's
sandwich, then on E2B/E4B a third add, `layers[i].per_layer_output` (unavailable
on 26B-A4B and 31B), then `*= layer_scalar`, a per-block buffer far from one
(0.005 to 0.99 on the released weights):
`(input + attention_output + mlp_output [+ per_layer_output]) * layer_scalar == layer_output`.
A term from block `i` reaches the last stream times the scalars of blocks `i`
through the last:

<!-- test: skip -->
```python
model = StandardizedTransformer("google/gemma-4-E2B", device="cpu", dispatch=True, dtype=torch.float32)
has_ple = model.support().get("per_layer_output", "absent") is None        # decided outside the trace

scalars = torch.stack([layer._module.layer_scalar.float().reshape(()) for layer in model.layers])
reach = scalars.flip(0).cumprod(0).flip(0)        # reach[i]: product of the scalars of blocks i..last

terms = {}
with model.trace(prompt):
    base = model.layers[0].input.save()
    for i, layer in enumerate(model.layers):
        terms["attn", i] = layer.self_attn.attention_output.save()
        terms["mlp", i] = layer.mlp.mlp_output.save()
        if has_ple:
            terms["ple", i] = layer.per_layer_output.save()
    final = model.layers[-1].layer_output.save()

weighted = {key: reach[key[1]] * t for key, t in terms.items()}           # what each term contributes to `final`
torch.testing.assert_close(reach[0] * base + sum(weighted.values()), final, rtol=1e-4, atol=1e-4)
```

Attribute with `weighted` (through `lm_head`, as in the `patterns` skill).
Steering a contribution at block `i` moves the last stream `reach[i]` times as
much; `steer` writes `layer_output`, after the scalar. The logit lens is
unaffected: the final RMS norm divides the scale back out.

**Doge.** The block gates the *stream*, per channel, with learned parameters
(`layers[i]._module.input_residual`, `.post_attention_residual`, both starting at
one) and adds the modules' outputs unscaled: `h = input_residual * input +
attention_output`, `layer_output = post_attention_residual * h + mlp_output`:

<!-- test: skip -->
```python
gate_in = [layer._module.input_residual.float() for layer in model.layers]          # [hidden] each
gate_post = [layer._module.post_attention_residual.float() for layer in model.layers]
after = [torch.ones_like(gate_in[0]) for _ in range(model.num_layers + 1)]
for i in reversed(range(model.num_layers)):
    after[i] = gate_in[i] * gate_post[i] * after[i + 1]     # what the stream entering block i is scaled by

weight = {("attn", i): gate_post[i] * after[i + 1] for i in range(model.num_layers)}
weight |= {("mlp", i): after[i + 1] for i in range(model.num_layers)}
# final == after[0] * base + sum(weight[key] * terms[key])   (terms read as in the Gemma-4 recipe)
```

No released Doge checkpoint loads in transformers 5.17 (size mismatches); the
tiny one does. **ZAYA** merges each sublayer's output `o` into the stream `r` as
`(o + hidden_states_bias) * hidden_states_scale + (r + residual_bias) * residual_scale`
(each merge's parameters on `layers[i].post_attention_residual_scale._module` /
`post_mlp_residual_scale._module`); `attention_output` / `mlp_output` are the first
term, and the stream term rescales and biases what came before. Unroll it the way
the Doge loop does.

**DeepSeek-V4**: parallel streams. The residual between blocks is `hc_mult` (4)
copies of the stream: `layer_output` and `layers[i].input` are `[batch, seq,
streams, hidden]` (layout `Streams`), the block's own tensor, so writes land.
Each sublayer reads a weighted collapse and returns `[batch, seq, hidden]`;
`attention_output` / `mlp_output` are those outputs, unscaled. Four values on the
block carry the hyper-connection weights, float32 and writable:
`attention_post` / `mlp_post` (`StreamWeights`, `[batch, seq, streams]`, in (0, 2))
and `attention_comb` / `mlp_comb` (`StreamMixing`, `[batch, seq, streams,
streams]`, doubly stochastic, applied transposed). The identity is the block's
formula, exact in float32:

```
h            = attention_combᵀ · input + attention_post ⊗ attention_output
layer_output = mlp_combᵀ · h + mlp_post ⊗ mlp_output
```

Only the stream mean is additive (`out.mean(2) == input.mean(2) +
attention_post.mean(-1, keepdim=True) * attention_output + mlp_post.mean(-1,
keepdim=True) * mlp_output`, within ~2e-6 relative). Code that assumes rank 3
(`resid[:, -1] @ W`, `lm_head(norm(resid))`) runs and silently answers per stream;
the plain identity raises a shape error at most lengths. `project_on_vocab`
collapses the streams with the model's `hc_head` and equals `logits` at the last
block; `skip_layers` and `steer` work unchanged (a `[hidden]` vector adds to every
stream).

## Root values

The root answers for the whole run. Read `input_ids` / `attention_mask` /
`input_size` first (they are the model's input), `token_embeddings` next,
`lm_head.output`, `logits` and `next_token_probs` last:

```python
with model.trace(prompt):
    ids = model.input_ids.save()                 # [batch, seq]
    mask = model.attention_mask.save()           # [batch, seq]; zeros are padding
    size = torch.tensor(model.input_size).save() # a torch.Size bound in the block does not survive it: save a tensor
    emb = model.token_embeddings.save()          # [batch, seq, hidden]: embed_tokens.output
    entering = model.layers[0].input.save()      # what block 0 receives
    raw = model.lm_head.output.save()            # the raw projection
    logits = model.logits.save()                 # [batch, seq, vocab]
    probs = model.next_token_probs.save()        # [batch, vocab]

assert ids.shape == mask.shape == (1, 10) and size.tolist() == [1, 10]
assert not torch.equal(emb, entering)            # GPT-2 adds wpe between the two
assert torch.equal(raw, logits)                  # GPT-2's head applies nothing more
assert torch.allclose(probs, logits[:, -1].softmax(-1))
```

**`logits`** is the `.logits` of the model's output: softcapped on Gemma-2/4 and
VaultGemma, times `logit_scale` on Cohere, `/ logits_scaling` on the Granite line,
times `logits_scaling` on HyperCLOVA X, times `lm_head_multiplier` on Falcon-H1,
float32 on Mamba-2 and Nemotron-H. `lm_head.output` is the raw projection;
`project_on_vocab` applies the family's step. Assigning replaces the logits in the
output:

```python
with model.trace(prompt) as tracer:
    model.logits = model.logits * 0
    result = tracer.result.logits.save()

assert result.abs().max() == 0
```

**`token_embeddings`** is whatever the embedding module returns: before GPT-2's
positional `wpe`, before BLOOM's `word_embeddings_layernorm`, before Granite's
`embedding_multiplier` (12 on granite-3), but *with* Gemma's `sqrt(hidden)` scale,
which sits inside Gemma's embedding module. `layers[0].input` is what enters
block 0 on every family; use it as the base of a decomposition. Assign
`token_embeddings` to replace what the embedding hands on.

**`next_token_probs`** is `logits[:, -1].softmax(-1)` in the model's dtype,
derived and read-only. Position `-1` is every row's last token only under left
padding, which nnsight's tokenizer sets for causal models. On a bf16 model it is
a bf16 softmax (off by up to ~2e-3): take `logits[:, -1].float().softmax(-1)` for
a metric. It can hold exact zeros, so a KL needs `torch.special.xlogy` or
`F.kl_div` on log-softmax, never `p * (p.log() - q.log())` (NaN on Pythia).

<!-- test: expect-error AttributeError -->
```python
with model.trace(prompt):
    model.next_token_probs = model.next_token_probs * 0     # AttributeError: assign model.logits instead
```

**`input_ids` and `attention_mask`** are assignable, and the model then runs on
what you set. Assigning ids of another length does not resize the mask; assign
both:

```python
with model.trace("Paris is the capital of"):
    other_ids = model.input_ids.save()
    other_logits = model.logits.save()

with model.trace(prompt):
    model.input_ids = other_ids.clone()
    model.attention_mask = torch.ones_like(other_ids)
    swapped = model.logits.save()

assert torch.allclose(swapped, other_logits)      # the second trace ran on the first prompt's ids
```

`input_size` is read-only (`AttributeError`: assign `input_ids`).

## Sizes

**The root's** are read off the config before any trace, without `dispatch`:

| size | plain rule |
|---|---|
| `num_layers` | `len(model.layers)` |
| `hidden_size`, `vocab_size`, `num_heads` | `config.hidden_size`, `config.vocab_size`, `config.num_attention_heads` |
| `num_kv_heads` | `config.num_key_value_heads`, else `num_heads` |
| `head_dim` | `config.head_dim` when the config says (Qwen3, Gemma), else `hidden_size // num_heads` |
| `qk_head_dim` | `head_dim` |
| `intermediate_size` | `config.intermediate_size` (the dense MLP; experts are `moe_intermediate_size` wide) |

Each is a `StandardizedProperty`: a function of the same name in the family module
(`def intermediate_size(model)`) wins on read. Families that spell one their own
way: GPT-2/GPT-J/CodeGen (`n_inner`), Falcon (`ffn_hidden_size`, `num_kv_heads`
by layout), OPT/XGLM (`ffn_dim`), MPT, BLOOM, the latent-attention families
(`head_dim` = `v_head_dim`, `qk_head_dim` = nope + rope), Mamba-2 (`num_heads` the
SSD heads, `intermediate_size` the mixer's inner width), Gemma-4, GraniteMoE-Hybrid,
ZAYA; the full list is in nnterp's `docs/reference/families.md` ("Logits, scales
and sizes"). A pure Mamba config has no attention heads: `num_heads` and
`head_dim` have nothing to read.

**A block's own** are on its modules, read outside or inside a trace, and differ
from the root's where blocks differ (Gemma-4's full blocks, MiMo-V2-Flash's
sliding blocks, Laguna's per-block head counts, Gemma-4 E2B's double-wide MLPs):
`layers[i].self_attn.num_heads`, `.num_kv_heads`, `.head_dim`, `.qk_head_dim`;
`layers[i].mlp.intermediate_size` (one routed expert's width on a mixture).

```python
from nnterp.standardized import StandardizedProperty

assert (model.num_layers, model.hidden_size, model.num_heads, model.num_kv_heads) == (12, 768, 12, 12)
assert (model.head_dim, model.qk_head_dim, model.vocab_size, model.intermediate_size) == (64, 64, 50257, 3072)
assert model.config.n_inner is None and model.family.intermediate_size(model) == model.intermediate_size
assert isinstance(StandardizedTransformer.intermediate_size, StandardizedProperty)
assert (model.layers[3].self_attn.num_heads, model.layers[3].mlp.intermediate_size) == (12, 3072)
```

## Layouts

Every value's return annotation is a named `jaxtyping` alias, defined beside the
envoy that serves it and re-exported by `nnterp.components` (the root's
`Logits`, `NextTokenProbs`, `Tokens` from `nnterp.standardized`). `value.layout` is
that alias itself, `value.dims` its axes, and the envoy repr prints each value as
`(name) -> Layout [axes]: description`. Layouts differ between values, not
between families; the one exception is `layer_output` on DeepSeek-V4 (`Streams`).

| layout | axes | values |
|---|---|---|
| `Residual` | `batch seq hidden` | `layer_output`, `attention_output`, `mlp_output`, `token_embeddings`, `per_layer_output`, `routed_output`, `shared_expert_output` |
| `Streams`, `StreamWeights`, `StreamMixing` | `batch seq streams hidden`, `batch seq streams`, `batch seq streams streams` | DeepSeek-V4's stream and hyper-connection weights |
| `Logits`, `NextTokenProbs`, `Tokens` | `batch seq vocab`, `batch vocab`, `batch seq` (`Int`) | the root values |
| `Queries`, `Keys`, `Values` | `batch heads seq qk_head_dim`, `batch kv_heads seq qk_head_dim`, `batch kv_heads seq head_dim` | softmax attention |
| `Pattern`, `HeadOutputs` | `batch heads query key`, `batch seq heads head_dim` | scores and pattern; head outputs |
| `RouterLogits`, `ExpertWeights`, `ExpertIndices`, `ExpertOutputs` | `batch seq experts`, `batch seq top_k` (x2, indices `Int`), `batch seq top_k hidden` | a `Moe` |
| `LinearQK`, `LinearV`, `Gates`, `State`, `States` | `batch seq heads key_dim`, `batch seq heads value_dim`, `batch seq heads`, `batch heads key_dim value_dim`, `batch seq heads key_dim value_dim` | DeltaNet (`State`/`States` also Mamba-2's) |
| `ScanQK`, `ScanValues`, `ScanSteps`, `ScanDecays`, `ScanState`, `ScanStates` | `batch seq groups state_dim`, `batch seq channels` (x2), `batch seq channels state_dim`, `batch channels state_dim`, `batch seq channels state_dim` | Mamba-1 |
| `SSDQueries`, `SSDKeys`, `SSDValues`, `SSDHeadOutputs` | `batch seq groups state_dim` (x2), `batch seq heads head_dim` (x2) | Mamba-2 |

```python
from nnterp.components import Keys, Pattern, Queries, Residual
from nnterp.standardized import Logits
from nnterp.families import falcon

assert Attention.attention_keys.layout is Keys and falcon.Attention.attention_keys.layout is Keys
assert Layer.layer_output.layout is Residual and StandardizedTransformer.logits.layout is Logits
assert Layer.layer_output.dims == ("batch", "seq", "hidden")

with model.trace(prompt):
    q = model.layers[0].self_attn.attention_queries.save()
    pattern = model.layers[0].self_attn.attention_probabilities.save()

assert isinstance(pattern, Pattern) and isinstance(q, Queries) and not isinstance(q[0], Queries)   # rank and dtype
```

`isinstance` checks rank and dtype, not axis sizes. Batch is axis 0 everywhere;
the sequence axis is 1 except on softmax attention's queries, keys and values,
where it is 2. "The last token" is `value[:, -1]` for a residual value,
`value[:, :, -1]` for queries/keys/values, `pattern[:, :, -1, :]` for the last
query row.

A vision-language wrapper loaded with `task="image-text-to-text"` adds the tower's
values: `layer_output`, `attention_output` and `mlp_output` on `vision.layers[i]`,
and `patch_embeddings` and `tower_output` on `model.vision`, all laid out
`Patches` (`images patches vision_hidden`; a row is an image, a crop, a tile or a
packed run of images, per tower); `vision.image_token_mask` (`ImageTokenMask`,
`batch seq`, bool) and `vision.image_features` (`ImageFeatures`,
`image_tokens hidden`, flat over the batch). The text values cover the image
positions too. What each means per tower: [vision.md](vision.md#the-values).

## Gotchas

- Forward order within one trace: `input`, `attention_output`, `mlp_output`,
  `layer_output`; on BLOOM/MPT a contribution comes before the module's `.output`,
  on post-norm families after it.
- Never `[0]` a standard value to unwrap it; it indexes the batch.
- `mlp_output` does not exist on OPT or XGLM (no `mlp` module); `support()` then has
  no `mlp.mlp_output` key, so `.get` it. `layers[i].fc2.output` is what that block
  adds.
- Nothing bound inside a trace survives without `.save()`; save each operand of a
  difference you compute later.

The nnterp repo pages behind this file: `docs/usage/residual-stream.md`,
`docs/usage/root-values.md`, `docs/usage/layouts.md`, `docs/reference/families.md`.
