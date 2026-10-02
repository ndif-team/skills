---
name: nnterp
description: Write interpretability code once against nnterp's StandardizedTransformer and run it on all 92 transformer families it knows (GPT-2, Llama, Qwen, Gemma-2/3/4, OLMo, Granite, Falcon, DeepSeek, Qwen3.5/Qwen3-Next DeltaNet hybrids, Mamba/Mamba-2/Jamba, 36 mixture-of-experts families): one vocabulary (model.layers[i].self_attn / .linear_attn / .mlp, model.norm, model.lm_head) and standard values (layer_output, attention_output, mlp_output, attention_probabilities and the attention interior, a mixture's expert_weights / expert_indices, a recurrent mixer's state, logits, next_token_probs), plus support(), skip_layers, steer, project_on_vocab. Load it for code that reads or edits the residual stream, a sublayer's contribution, an attention pattern, expert routing or a recurrent state on more than one architecture, or that would otherwise branch on per-family module paths and unwrap tuples; and when a result on Gemma-4, Granite, Doge, ZAYA or DeepSeek-V4 looks wrong (their blocks are not a plain sum).
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

Verified on nnterp `5aba2f4` (branch `0.8-refactor`, 92 families), nnsight 0.8 dev, transformers 5.17.0.

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

assert model.family.__name__ == "nnterp.families.gpt2" and len(nnterp.families.known()) == 92
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
block; a block's interior (queries, scores, `attention_probabilities`) before
its `attention_output`, before `mlp_output`, before `layer_output`; block 3
before block 5; `logits` and `next_token_probs` last. A late read raises
`OutOfOrderError`, which names an internal location (`...attention_interface_1.fn.i0`),
not the value:

<!-- test: expect-error OutOfOrderError -->
```python
with model.trace(prompt):
    late_attn = model.layers[3].self_attn.attention_output.save()
    late_pattern = model.layers[3].self_attn.attention_probabilities.save()   # inside the attention: too late
```

Three variants do not raise. Under `generate` + `tracer.iter`, an out-of-order
read *binds the next step's value*, so per-step lists come back shifted by one. A
read pinned to a step the run never reaches (past `max_new_tokens`, or that
shifted read on the last step) cuts the block short with a `UserWarning`: later
names are unbound, so look for the warning when a saved name is missing. On a Mamba-1 decode step, `attention_head_outputs` read before
`state_output` returns the next step's state ([references/recurrent-mixers.md](references/recurrent-mixers.md)).
`skip_layers` consumes `layers[start].input`: read it first or not at all.

**7. Patterns in two invokes of one trace raise.** Reading `attention_probabilities`
or `attention_scores` in two invokes of one trace fails with
`TypeError: 'NoneType' object is not subscriptable` (an nnsight bug). Batch the
prompts in one invoke, or use one trace per prompt.

**8. Overwriting `attention_scores` lifts the causal mask.** The scores are read
after masking, so `scores[:] = 0` attends to the future. Add to them
(`scores[..., k] += c`, or `-inf` to block a key) instead.

**9. GPT-2, GPT-BigCode and MPT queries/keys/values are split views**; so are
Mamba-1's. torch refuses an in-place edit with grad on; assign instead:

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

**10. Recurrent mixers route kernels before the first trace.** On Mamba,
Falcon-Mamba and Jamba with `mamba_ssm` installed, a CPU run crashes with
`Expected u.is_cuda()` (even for `layer_output`) until
`nnterp.route_kernels(model.family, "torch")`; on DeltaNet hybrids the per-token
`state`/`states` need the same call; on Mamba-2 families `states` needs
`nnterp.chunk_per_token(model)`.

**11. Hybrids: decide `self_attn` vs `linear_attn` outside the trace**, with
`getattr(layer, "self_attn", None) is not None`. `getattr` inside a trace can trip
a served value, and `a or b` over envoys truth-tests the module's `__len__`
(`TypeError`). `linear_attn` is a DeltaNet, a Mamba-1 scan or a Mamba-2 mixer;
the same names mean different things on each.

**12. `envoys=` keys are module types or native paths, never aliases.** Displace
a family envoy by keying on its type; `"layers.0.mlp"` matches nothing.

**13. Import `nnterp` (or `nnsight`) before any `transformers.models...` module**;
the reverse order segfaults at import.

## The vocabulary

Llama's block names, with the containers lifted out of the inner `.model`:

| standard name | GPT-2 | Llama | GPT-NeoX |
|---|---|---|---|
| `model.embed_tokens` | `transformer.wte` | `model.embed_tokens` | `gpt_neox.embed_in` |
| `model.layers[i]` | `transformer.h[i]` | `model.layers[i]` | `gpt_neox.layers[i]` |
| `model.layers[i].self_attn` | `...h[i].attn` | same | `...layers[i].attention` |
| `model.layers[i].mlp` | same | same | same |
| `model.norm` | `transformer.ln_f` | `model.norm` | `gpt_neox.final_layer_norm` |
| `model.lm_head` | same | same | same |

`self_attn.input` is what enters the attention and `mlp.input` what enters the
MLP, on every family; prefer them to the norm aliases (`input_layernorm`,
`post_attention_layernorm`), whose meaning differs per family. A recurrent block
has `linear_attn` instead of `self_attn`; OPT and XGLM have no `mlp` module;
Mamba and Mamba-2 blocks have neither attention nor MLP.

## The standard values

| value | on | layout | notes |
|---|---|---|---|
| `layer_output` | `layers[i]` | `Residual` | the stream leaving the block; `Streams` on DeepSeek-V4 |
| `attention_output` | `self_attn` / `linear_attn` | `Residual` | the contribution |
| `mlp_output` | `mlp` | `Residual` | the contribution |
| `attention_queries`, `_keys`, `_values` | `self_attn` | `Queries`, `Keys`, `Values` | seq on axis 2; keys/values before `repeat_kv`; eager |
| `attention_scores` | `self_attn` | `Pattern` | masked, scaled, pre-softmax; eager |
| `attention_probabilities` | `self_attn` | `Pattern` | the tensor the values are mixed with; eager |
| `attention_head_outputs` | `self_attn` | `HeadOutputs` | `[batch, seq, heads, head_dim]`, before the output projection; eager |
| `router_logits`, `expert_weights`, `expert_indices`, `expert_outputs`, `routed_output`, `shared_expert_output` | `mlp` on a `Moe` | `RouterLogits`, `ExpertWeights`, ... | [references/mixture-of-experts.md](references/mixture-of-experts.md) |
| `state_input`, `state_output`, `state`, `states`, `decays`, `betas`, ... | `linear_attn` | `State`, `Gates`, `Scan*`, `SSD*` | [references/recurrent-mixers.md](references/recurrent-mixers.md) |
| `per_layer_output` | `layers[i]` (Gemma-4 E2B/E4B) | `Residual` | the block's third add |
| `attention_post`, `attention_comb`, `mlp_post`, `mlp_comb` | `layers[i]` (DeepSeek-V4) | `StreamWeights`, `StreamMixing` | the hyper-connection weights |
| `logits` | root | `Logits` | the output's `.logits` (softcapped, scaled); `lm_head.output` is raw |
| `next_token_probs` | root | `NextTokenProbs` | `logits[:, -1].softmax(-1)`, in the model dtype; read-only |
| `token_embeddings` | root | `Residual` | `embed_tokens.output`; `layers[0].input` is what enters block 0 |
| `input_ids`, `attention_mask`, `input_size` | root | `Tokens` | read first; the first two assignable |

Every layout is a named `jaxtyping` alias (`nnterp.components.Residual`, `Pattern`,
...; the root's from `nnterp.standardized`): `value.layout` is that alias,
`value.dims` its axes, and `print(envoy)` lists each value as
`(name) -> Layout [axes]: description`:

```python
from nnterp.components import Pattern

assert type(model.layers[0].self_attn).attention_probabilities.layout is Pattern
print(type(model.layers[0].self_attn).attention_probabilities)
# (attention_probabilities) -> Pattern [batch heads query key]: The attention pattern the values are mixed with
```

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
edit's own effect; compare such edits in float32. On Gemma-4 an in-place edit of a
KV-sharing source block's keys or values reaches every block that borrows them;
assign to keep it local.

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
sizes, `support()` and a block's repr:

```
python scripts/inspect_family.py <repo_id> [--layer N] [--eager]
```

## Families

92 families, `nnterp.families.known()`; an unknown `model_type` raises
`UnsupportedFamily` before any weights load (the `extending` skill adds
one). What a recipe must survive is grouped by quirk in
[references/api-reference.md](references/api-reference.md); the full per-family
table is nnterp's `docs/reference/families.md`.

## References

| File | Covers |
|---|---|
| [references/values.md](references/values.md) | the boundary values and the identity, tuple blocks, where families relocate a contribution, the non-additive families with their stream-weight recipes, root values, sizes, layouts |
| [references/attention-interior.md](references/attention-interior.md) | queries/keys/values/scores/probabilities/head outputs, family caveats (GPT-2, MPT, Falcon, sinks, latent attention, Gemma-4 borrowed keys), what edits do to the mask, under `generate` |
| [references/mixture-of-experts.md](references/mixture-of-experts.md) | a `Moe`'s six values, usage / entropy / ablation / rerouting recipes with pad masking, bf16 and invoke caveats, per-family gaps |
| [references/recurrent-mixers.md](references/recurrent-mixers.md) | `LinearAttention` (DeltaNet), `SelectiveScan` (Mamba-1), `StateSpace` (Mamba-2): what each name means on each, kernels, per-token state, read-order traps |
| [references/availability-and-support.md](references/availability-and-support.md) | `support()` forms, the reasons, `Unavailable` vs `SourceNotAvailable`, guarding a script |
| [references/generation-and-helpers.md](references/generation-and-helpers.md) | values under `generate` with `tracer.iter`, `nnterp.prompt_utils`, `nnterp.nnsight_utils`, padding, `remote=True` |
| [references/api-reference.md](references/api-reference.md) | every exported symbol, and the 92 families grouped by quirk |

The nnterp repo's pages go deeper: `docs/usage/*.md` (one page per feature),
`docs/reference/families.md`, `docs/patterns/*.md`.

## Related skills

- `nnsight`: the underlying API (`.save()`, invokes, `tracer.iter`, `.source`, `rename`)
- `patterns`: logit lens, attribution, ablation, patching, steering, probing, sweeps written once against these values
- `extending`: a checkpoint nnterp does not know, overriding a value, a value of your own
- `debugging`: an error, a hang, an empty result
- `remote`: running on NDIF; nnterp must be installed server-side
