# API reference (nnterp on nnsight 0.8)

Every name nnterp exports, with the value rows' layout, assignability and
availability; then the 98 families grouped by the quirks a cross-family recipe
must survive. Everything nnsight's `TransformersModel` offers (`trace`,
`generate`, `session`, `edit`, `tracer.iter`, `.save()`, `remote=`) is inherited
unchanged; the `nnsight` skill covers it. nnterp's own page is
`docs/reference/api-quick-reference.md`.

```python
import torch
import nnterp
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager")
prompt = "The Eiffel Tower is in"
n = len(model.tokenizer(prompt).input_ids)       # 7

with model.trace(prompt):
    x = model.layers[3].input.save()
    pattern = model.layers[3].self_attn.attention_probabilities.save()   # inside the attention: before its output
    attn = model.layers[3].self_attn.attention_output.save()
    mlp = model.layers[3].mlp.mlp_output.save()
    resid = model.layers[3].layer_output.save()

torch.testing.assert_close(x + attn + mlp, resid)
assert pattern.shape == (1, 12, n, n)
assert set(nnterp.__all__) >= {"StandardizedTransformer", "Moe", "SelectiveScan", "StateSpace", "Vision", "route_kernels", "chunk_per_token"}
assert len(nnterp.families.known()) == 99      # 98 families + kimi_k25, Kimi K2's under its wrapper's name
```

## Top-level names

| name | what it is |
|---|---|
| `StandardizedTransformer` | the model class: a `TransformersModel` renamed to the standard vocabulary and wrapped in the family's envoys |
| `StandardizedVLLM` | the same over nnsight's `VLLM` engine, families from `nnterp.families.vllm` ([SKILL.md](../SKILL.md#the-vllm-engine-and-several-gpus)) |
| `Standardized` | what the two share (sizes, `support()`, the methods); `isinstance(model, Standardized)` accepts either engine |
| `Layer`, `Attention`, `Mlp`, `Moe` | the base envoys a family subclasses; the hosts of the standard values |
| `RecurrentMixer`, `LinearAttention`, `SelectiveScan`, `StateSpace` | the recurrent-mixer base and its DeltaNet, Mamba-1 and Mamba-2 subclasses |
| `Vision` | a vision-language wrapper's tower root, `model.vision`; its block and wrapper envoys are in `nnterp.components` (below) |
| `Standard` | the envoy base of them all: `values()`, `support()`, `sourced` |
| `EProperty`, `DerivedEProperty`, `unavailable` | the descriptors a value is made of, and a value a family lacks (the `extending` skill); `TokenEProperty` is in `nnterp.components` |
| `route_kernels`, `route_delta_rule`, `chunk_per_token` | the recurrent kernel switch (process-wide), its DeltaNet spelling, Mamba-2's per-token chunking (per model) |
| `Unavailable`, `UnsupportedFamily` | the two exceptions nnterp raises itself |
| `axis_size(envoy, axis)` | the size of a layout axis that is a fact about the model (`num_heads`, `hidden_size`, ...) for a value served on `envoy` |

`nnterp.families`, `nnterp.components`, `nnterp.standardized`, `nnterp.prompt_utils`
and `nnterp.nnsight_utils` are modules.

## `StandardizedTransformer`

```
StandardizedTransformer(repo_id, *args, family=None, rename=None, envoys=None, tokenizer_kwargs=None, **kwargs)
```

| argument | meaning |
|---|---|
| `repo_id` | a Hub repo id, or an already-loaded `torch.nn.Module` (its own `config` is read) |
| `family` | the family for this load instead of the one looked up: a module or object with `RENAME` and `ENVOYS` (the `extending` skill); nothing is registered |
| `rename` | extra nnsight aliases merged over the family's `RENAME`; a key given here wins |
| `envoys` | extra `envoys=` entries merged over the family's `ENVOYS`; keys are module **types** or paths (native, or any alias spelling); type keys are tried first, so a family's type key wins over a path key |
| `tokenizer_kwargs` | attributes set on the loaded tokenizer: `{"padding_side": "left", "pad_token": ...}` |
| `**kwargs` | to `TransformersModel`: `dispatch=True`, `device="cpu"` / `"cuda"` (`device_map="cpu"` is ignored by the text-generation pipeline), `attn_implementation="eager"`, `experts_implementation=`, `dtype=`, `revision=`, `trust_remote_code=`, `distributed_config=DistributedConfig(tp_size=N)` under `torchrun` |

The constructor reads the config first (a multimodal config's `text_config`) and
resolves the family (`nnterp.families.resolve`); a `model_type` with no family loads
with the best-effort `default` family and a `UserWarning`, and `UnsupportedFamily` is
raised before any weights load only when the default cannot standardize it. Neither
`attn_implementation` nor `experts_implementation` is forced.

### Root values (inside a trace)

| value | layout | assignable | what |
|---|---|---|---|
| `model.logits` | `Logits` | yes: replaces `output.logits` | the model's final logits (softcapped, scaled per family); `lm_head_output` is the raw projection |
| `model.final_norm_output` | `Residual` | yes | `norm.output`: the stream the head projects |
| `model.lm_head_output` | `Logits` | yes | `lm_head.output`: the head's logits before the family's softcap or scale; on `StandardizedVLLM` the logits themselves where the logits processor applies no softcap or scale, unavailable naming which where it does |
| `model.token_embeddings` | `Residual` | yes | `embed_tokens.output`: whatever the embedding module returns |
| `model.next_token_probs` | `NextTokenProbs` | no (`AttributeError`: assign `logits`) | `logits[:, -1].softmax(-1)` in the model dtype |
| `model.input_ids`, `model.attention_mask` | `Tokens` | yes: the model runs on what you set | the call's ids and mask; read before any block |
| `model.input_size` | a `torch.Size` | no | `[batch, seq]` of the current call |

### Methods

| method | where | what |
|---|---|---|
| `skip_layers(start, end, skip_with=None)` | inside a trace, before block `start` runs | blocks `start..end` (inclusive) do not run; consumes `layers[start].input`; must cover every row of a batch (one invoke of several raises `ValueError`) |
| `steer(layers, vector, factor=1.0, token_positions=None, batch_index=None)` | inside a trace; `layers` an int or ascending list | `layer_output += factor * vector` in place at those positions / that row |
| `project_on_vocab(hidden)` | inside (live) or outside (saved) | the model's own path to its logits: norm, `lm_head`, then the family's step (softcap, `logit_scale`, `/ logits_scaling`, `hc_head`, ZAYA's cast; on OPT-350m `project_out` in place of the missing final norm); equals `logits` on the last block |
| `get_topk_closest_tokens(hidden, k=5)` | outside, on a saved `[..., hidden]` tensor | `project_on_vocab`, softmax, `{decoded token: probability}` per position; byte tokens decoding alike collapse |
| `probs_to_dict(probs, k=5)` | outside | the `k` most likely tokens of one `[vocab]` distribution, same keying |
| `support(layer=None)` | outside; nothing runs | every root and block value, `None` or `{layer: reason}`; with `layer`, that block flat |
| `order(layer=None)` | outside; one probe `scan` on first use, then kept | the available values in forward order, `{name: rank}`; equal ranks are one location ([availability-and-support.md](availability-and-support.md#forward-order-order-and-rank)) |
| `rank(name, layer=None)` | outside; as `order` | a sortable `(side, r)`: sort `(value, block)` sites by it to put reads and edits in forward order; `Unavailable` / `KeyError` for a value the checkpoint lacks / no value of that name |
| `layers[i].self_attn.rotate(x, positions)` | outside or inside a trace; inside only on `StandardizedVLLM` | `x` (`[..., seq, qk_head_dim]`, laid out like the values before the rotary) rotated by that block's own rotary embedding at absolute `positions` (`[seq]`, or `[batch, seq]`): `rotate(attention_queries_prerope, positions)` is `attention_queries` bit for bit; `Unavailable` with `no_rotary()`'s reason on a block without one ([attention-interior.md](attention-interior.md#before-the-rotary-embedding-attention_queries_prerope-attention_keys_prerope-rotate)) |
| `layers[i].self_attn.value_weight(kv)`, `.output_weight(j)`; `layers[i].attention_input_gain` | outside or inside a trace, on a dispatched model | key/value head `kv`'s slice of the value projection `[head_dim, hidden_size]`, head `j`'s slice of the output projection `[hidden_size, head_dim]`, and the pre-attention norm's per-channel gain `[hidden_size]` (ones on a post-norm block), read off the modules; the fused and transposed layouts read for you, and under a PEFT adapter the weight the LoRA layer applies; `Unavailable` with `no_head_weights()`'s reason (latent attention, DeepSeek-V4, Gemma-4, JetMoE, ZAYA, the default family, towers, tensor parallelism, vLLM) ([attention-interior.md](attention-interior.md#the-weights-per-head)) |
| `layers[i].self_attn.interface(q, k, v)` | inside a trace: read with the queries, keys and values, once per forward; called before the scores, the pattern or the head outputs | the attention function the block's call ran (`eager_attention_forward` or the family's own; `_attn` on GPT-J, GPT-Neo, CodeGen, GPT-NeoX-Japanese), bound to the call's mask, scale, dropout and sliding window; takes `Queries`, `Keys`, `Values` (`num_kv_heads` wide) and returns `(HeadOutputs, Pattern)`, `attention_head_outputs` and `attention_probabilities` bit for bit on the served tensors; nothing reaches the model until you write a result; a `DerivedEProperty`, read-only ([attention-interior.md](attention-interior.md#calling-the-attention-function-again-interface)) |

### Sizes

`num_layers`, `hidden_size`, `vocab_size`, `num_heads`, `num_kv_heads`,
`head_dim`, `qk_head_dim`, `intermediate_size`: each a `StandardizedProperty`
(`nnterp.standardized`) reading the text config by the plain rule unless the family
module defines `def <size>(model)`. `project_on_vocab` is a
`StandardizedCapability` the same way (`def project_on_vocab(model, hidden)`).
A block's own sizes are plain properties on its modules:
`layers[i].self_attn.num_heads` / `num_kv_heads` / `head_dim` / `qk_head_dim`,
`layers[i].mlp.intermediate_size`; a mixture's `num_experts`, `top_k`, `SCORING`.
Details: [values.md](values.md#sizes).

### Other attributes

| attribute | what |
|---|---|
| `model.family` | the family module (`nnterp.families.llama`); what `route_kernels` takes |
| `model.layers`, `model.embed_tokens`, `model.norm`, `model.lm_head` | the vocabulary, as envoys |
| `model.layers[i].self_attn` / `.linear_attn` / `.mlp` | the family's `Attention`, recurrent mixer, `Mlp` or `Moe`; absent where the block has none |
| `model.layers[i].self_attn.o_proj`, `.mlp.act_fn`, `.mlp.down_proj` | the output projection, the activation and the down projection under their standard names, aliased from each family's spelling |
| `model.add_prefix_false_tokenizer` | the tokenizer with `add_prefix_space=False`, loaded on first use |
| `model.vision`, `model.projector` | a vision-language wrapper's tower (a `Vision`) and the last module before the scatter; present on the 16 host families' wrappers, absent on a text-only checkpoint ([vision.md](vision.md)) |
| `model.processor` | the wrapper's processor, loaded with `task="image-text-to-text"`; `None` on the default `text-generation` load |

## Envoys and their values

| host | values | page |
|---|---|---|
| `Layer` | `layer_input`, `attention_norm_output`, `layer_mid`, `mlp_norm_output`, `mlp_input` (`Residual`; `FlatResidual` for OPT's `layer_mid` / `mlp_norm_output`); `layer_output` (`Residual`; `Streams` on DeepSeek-V4, as are its `layer_input` and `layer_mid`); `per_layer_output` (Gemma-4); `attention_post` / `attention_comb` / `mlp_post` / `mlp_comb` (DeepSeek-V4); `skip_with(hidden)`; `attention_input_gain`; `returns_tuple` | [values.md](values.md) |
| `Attention` | `attention_output`; `attention_premix` (`HeadOutputs`, `o_proj.input`, any implementation); `attention_queries_prerope` / `attention_keys_prerope` (`Queries` / `Keys`, `SharedRopeKeys` under latent attention; before the rotary, any implementation); `attention_queries` / `_keys` / `_values` / `_scores` / `_probabilities` / `_head_outputs` (eager); `interface` (eager; `Unavailable` with `OWN_ARITHMETIC` where no one call takes the queries, keys and values and returns the head outputs); `rotate(x, positions)`; `value_weight(kv)`, `output_weight(j)`; `SINK`; `off_interface()`, `no_rotary()`, `no_head_weights()` | [attention-interior.md](attention-interior.md) |
| `Mlp` | `mlp_output`; `mlp_activation`, `mlp_neurons` (`Neurons`: `act_fn.output`, `down_proj.input`) | [values.md](values.md#inside-the-sublayers) |
| `Moe(Mlp)` | `router_logits`, `expert_weights`, `expert_indices`, `expert_outputs`, `routed_output`, `shared_expert_output` (each a `TokenEProperty`); `num_experts`, `top_k`, `SCORING`, `no_mixture()`; `mlp_activation` / `mlp_neurons` unavailable (routed experts) | [mixture-of-experts.md](mixture-of-experts.md) |
| `RecurrentMixer` | `attention_output`, `state`, `states`, `state_after(t)`, `set_state_after(t, v)`; subclasses add `attention_queries` / `_keys` / `_values`, `decays`, `betas`, `state_input`, `attention_head_outputs`, `state_output` | [recurrent-mixers.md](recurrent-mixers.md) |
| `Vision` | `image_token_mask` (`ImageTokenMask`, read-only), `patch_embeddings`, `tower_output` (`Patches`), `image_features` (`ImageFeatures`, assignable); sizes `num_layers`, `vision_hidden_size`, `num_heads`, `head_dim`, `intermediate_size`, `patch_size`, `image_size`; `support(layer=None)`, `no_images()` | [vision.md](vision.md) |
| `PixtralVision(Vision)`, `QwenVision(Vision)` | Pixtral's tower (`patch_embeddings` is the packed row entering `ln_pre`); the Qwen ViT's (packed `[1, patches, vision_hidden]`, sizes add `spatial_merge_size`, `window_size`) | [vision.md](vision.md#the-qwen-vit) |
| `VisionLayer(Layer)`, `VisionAttention(Attention)`, `VisionMlp(Mlp)` | a tower block's `layer_output`, `attention_output`, `mlp_output` and the four module boundaries as `Patches`; the attention interior, `attention_premix`, `mlp_activation` and `mlp_neurons` as on a text block | [vision.md](vision.md#the-values) |
| `QwenVisionAttention(VisionAttention)` | the Qwen ViT's attention: queries, keys, values whole before the per-image split, `attention_head_outputs` concatenated after it; scores and pattern unavailable (`PER_IMAGE`) | [vision.md](vision.md#the-qwen-vit) |
| `ImageScatter` | keyed on the wrapper's inner model: the module whose forward writes the image features into the token embeddings (`scatter`, `scatter_argument`); where `image_features` is read. No values of its own | [vision.md](vision.md#availability) |

The vision layouts are `Patches` (`[images, patches, vision_hidden]`), `ImageTokenMask`
(`[batch, seq]`, bool) and `ImageFeatures` (`[image_tokens, hidden]`), all in
`nnterp.components` beside the text layouts.

Every value's descriptor exposes `.layout` (the named alias), `.dims`,
`.description`, `.reason(envoy)`; `str(descriptor)` is its repr line,
`(name) -> Layout [axes]: description`.

## `nnterp.families`

| name | what |
|---|---|
| `lookup(model_type)` | the family module: a registered one, else `nnterp.families.<model_type>` imported on first use, else `default` with a warning |
| `resolve(config)` | the family a load of that config uses: its own `model_type`'s where a family covers it, else its text config's |
| `covers(model_type)` | whether a shipped or registered family covers the type, without importing it; `False` means a load gets the best-effort default |
| `default` | the best-effort family for a `model_type` with none: a name guess checked at load; no contributions or `project_on_vocab` |
| `register(family, *model_types)` | add a family (anything with `RENAME` and `ENVOYS`) for the given model types, or for the type its module name ends in; consulted before the shipped modules, so it also overrides one |
| `known()` | the 99 shipped model types: 98 families, plus `kimi_k25`, which re-exports `kimi_k2` (registered ones are not listed) |
| `all_families()` | every shipped family, imported |
| `REGISTRY` | `model_type -> family` for what `register` added |

## `nnterp.prompt_utils` and `nnterp.nnsight_utils`

| name | what |
|---|---|
| `get_first_tokens(words, model_or_tokenizer)` | the first token of `word` and of `" word"` for each word, deduplicated |
| `Prompt.from_strings(prompt, targets, model)`, `Prompt.has_no_collisions()`, `Prompt.get_target_probs(probs)` | a prompt with named sets of target ids |
| `run_prompts(model, prompts, batch_size=32, get_probs_func=None, func_kwargs=None, remote=False)` | each target's mass per prompt, `[num_prompts, layers]` |
| `TokenizationError` | a word has no standalone first token |
| `get_token_activations(model, prompts, layers=None, get_activations=None, remote=False, idx=None, tracer=None)` | `[num_layers, num_prompts, hidden]` at one position, on the CPU |
| `collect_token_activations_batched(...)`, `collect_last_token_activations_session(...)` | the same over batches; inside one `model.session` for a remote run |
| `compute_next_token_probs(model, prompt, remote=False)` | `[num_prompts, vocab]` on the CPU |

## Exceptions

| exception | raised when |
|---|---|
| `nnterp.Unavailable` (`RuntimeError`) | a value this checkpoint lacks is read or written, before the model runs; `hasattr` raises it too |
| `nnterp.UnsupportedFamily` (`ValueError`) | the `model_type` has no family and the best-effort default cannot standardize it (no blocks, embedding, final norm or head under a name it knows, or a stream of the wrong shape) |
| `nnsight...SourceNotAvailable` | a source-located value's operation is not in this run's forward (another transformers release, an unexpected branch); also `hasattr` on a nested-source value outside a trace |
| `nnsight...OutOfOrderError` | a value read after the model ran past it; the message names an internal location, not the value; also a second `self_attn.interface` read after a call, or a call after the scores, the pattern or the head outputs (`'...attention_interface_1.input.i0'`, `'...attention_interface_1.fn.i0'`), under `tracer.iter` too |
| `AttributeError` | assigning a read-only value (`next_token_probs`, `input_size`, `states`, any `DerivedEProperty`); a mixture's `num_experts` / `top_k` / `SCORING` on a dense block |
| `RuntimeError` (torch view error) | an in-place edit of a split view with grad on: GPT-2 / GPT-BigCode queries, MPT queries (and keys and values on a call without a cache), Mamba-1 / Mamba-2 kernel arguments; before the rotary, DBRX / GPT-NeoX queries and keys (CodeGen's with one head per `mp_num` group), JetMoE keys, latent attention's shared rope key |
| `RuntimeError: Expected u.is_cuda()` | an unrouted Mamba-1 model on CPU with `mamba_ssm` installed |

## The 98 families, by what a recipe must survive

`nnterp.families.known()` lists them; nnterp's `docs/reference/families.md` has one
row per family (native names, public and pinned checkpoints, relocations, what
`support()` reports) and the full quirk prose. A family not in a group below is
Llama-shaped: the standard names and the base values hold as they are (`llama`,
`mistral`, `ministral`, `ministral3`, `qwen2`, `qwen3`, `gemma`, `olmo`, `phi3`,
`smollm3`, `glm`, `helium`, `arcee`, `apertus`, `bitnet`, `ernie4_5`,
`hunyuan_v1_dense`, `seed_oss`, `starcoder2`, `nemotron`, `persimmon`,
`vaultgemma`, ...).

| quirk | families | what changes |
|---|---|---|
| tuple block | `gptj`, `gpt_neo`, `codegen`, `gpt_neox_japanese`, `bloom`, `mpt`, `falcon`, `glm_moe_dsa`, `bamba`, `falcon_h1`, `zaya` | `.output` is a tuple; the values are tensors |
| post-norm (sandwich) | `gemma2`, `gemma3_text`, `gemma4_text`, `gemma4_unified_text`, `olmo2`, `olmo3`, `exaone4`, `flex_olmo`, `afmoe`, `glm4`, `hyperclovax`, OLMo-Hybrid's attention blocks | contributions are the post-norms' outputs, after `self_attn.output`; per-head sums are pre-norm |
| residual inside the module | `bloom`, `mpt` (MLP) | contributions read before the module's `.output` |
| parallel block | `gpt_neox`, `phi`, `gptj`, `codegen`, `stablelm`, `cohere`, `cohere2`, `falcon` | `mlp.input` is the block input's norm; the identity holds within a few ulps |
| scaled contributions | `granite`, `granitemoe`, `granitemoeshared`, `granitemoehybrid`, `granite_swa`, `granitemoe_swa`, `hyperclovax`, `falcon_h1` | values are the scaled terms, computed copies divided back on write |
| not a plain sum | `gemma4_text`, `gemma4_unified_text` (`layer_scalar`), `doge`, `zaya` (stream gates), `deepseek_v4` (`Streams`) | weight each term before attributing ([values.md](values.md#blocks-that-are-not-a-plain-sum)) |
| own attention arithmetic | `gptj`, `gpt_neo`, `codegen`, `gpt_neox_japanese`, `xglm`, `bloom`, `mpt`, `falcon` | same six interior names on the family's ops; different scale placement; `out_proj` / `dense`; `self_attn.interface` calls `_attn` on the first four |
| attention sink | `gpt_oss`, `deepseek_v4`, `mimo_v2_flash` (sliding blocks) in the pattern (`SINK`); `granite_swa`, `granitemoe_swa` scaling the head outputs | pattern rows sum below one, or the pattern does not control the output |
| latent attention | `deepseek_v2`, `deepseek_v3`, `kimi_k2`, `deepseek_v32`, `glm_moe_dsa`, `glm4_moe_lite`, `youtu`, `kimi_linear` (attention blocks) | `qk_head_dim` != `head_dim`; keys arrive `num_heads` wide; `attention_keys_prerope` is the shared rope key (`SharedRopeKeys`, `[batch, 1, seq, qk_rope_head_dim]`) on all but `kimi_linear` |
| no rotary embedding | `gpt2`, `gpt_bigcode`, `gpt_neo`, `opt`, `xglm` (absolute positions), `bloom`, `mpt`, `falcon` with `alibi` (ALiBi), the attention blocks of `jamba`, `nemotron_h`, `kimi_linear`; NoPE blocks of `cohere2`, `afmoe`, a hybrid `exaone4`, `smollm3`, `llama4_text`, `granite_swa` / `granitemoe_swa` (`layer_rope_theta` 0); every block of a `granitemoehybrid` or `olmo_hybrid` checkpoint whose config gives no rotary | `attention_queries_prerope`, `attention_keys_prerope` and `rotate` are `Unavailable` (on those blocks) |
| a step after the rotary | `hunyuan_v1_dense`, `hunyuan_v1_moe`, `llama4_text` (norms), `jetmoe` (keys tiled), `ministral3` (queries scaled past `original_max_position_embeddings`) | `rotate` is the rotary alone: `attention_queries` is that step applied to its result |
| no attention function to call again | `xglm`, `bloom`, `mpt`, `falcon` (own arithmetic), `deepseek_v4` (the head outputs are the interface's output rotated back) | `self_attn.interface` is `Unavailable` (`OWN_ARITHMETIC`) |
| no per-head value or output slice | latent attention (`deepseek_v2`, `deepseek_v3`, `kimi_k2`, `deepseek_v32`, `glm_moe_dsa`, `glm4_moe_lite`, `youtu`, `kimi_linear`), `deepseek_v4`, `gemma4_text` / `gemma4_unified_text` (`v_norm`), `jetmoe` (per-expert output projection), `zaya` (delayed values) | `value_weight` / `output_weight` are `Unavailable` (`no_head_weights()`) |
| sparse attention | `deepseek_v32`, `glm_moe_dsa` | the pattern is zero outside the indexer's selection (dense under 2048 tokens) |
| borrowed keys / values | `gemma4_text`, `gemma4_unified_text` | `attention_keys` / `_values` are a copy private to each block, so an edit stays on that block; edit the source block's `k_proj` / `v_proj` output to reach every borrower; skipping a source block fails |
| gated query | `qwen3_next`, `qwen3_5_text`, `qwen3_5_moe_text` | `q_proj` is twice as wide; `attention_queries` is not |
| DeltaNet hybrid | `qwen3_next`, `qwen3_5_text`, `qwen3_5_moe_text`, `olmo_hybrid`, `kimi_linear` | `linear_attn` on three blocks in four; Kimi-Linear's `decays` are `ChannelGates` (`[batch, seq, heads, key_dim]`, one decay per key channel) |
| Mamba-1 | `mamba`, `falcon_mamba`, `jamba` | `linear_attn` is a `SelectiveScan`; `route_kernels` first |
| Mamba-2 | `mamba2`, `nemotron_h`, `bamba`, `falcon_h1`, `granitemoehybrid` | `linear_attn` is a `StateSpace`; Nemotron-H names its one sublayer by class |
| mixture of experts (39) | `mixtral`, `qwen2_moe`, `qwen3_moe`, `qwen3_next`, `qwen3_5_moe_text`, `qwen3_vl_moe_text`, `olmoe`, `flex_olmo`, `gpt_oss`, `deepseek_v2`/`v3`/`v32`/`v4`, `kimi_k2`, `kimi_linear`, `glm4_moe`, `glm4_moe_lite`, `glm_moe_dsa`, `dots1`, `solar_open`, `mimo_v2_flash`, `ernie4_5_moe`, `minimax_m2`, `phimoe`, `hunyuan_v1_moe`, `jetmoe`, `dbrx`, `jamba`, `laguna`, `afmoe`, `llama4_text`, `gemma4_text`, `nemotron_h`, `granitemoe`, `granitemoe_swa`, `granitemoeshared`, `granitemoehybrid`, `zaya`, `doge` | `mlp` is a `Moe` (on some blocks); [mixture-of-experts.md](mixture-of-experts.md) |
| no MLP module | `opt`, `xglm`, `mamba`, `falcon_mamba`, `mamba2` | no `mlp.*` key in `support()`; OPT/XGLM's `fc2.output` is what the block adds |
| split-view q/k/v | `gpt2`, `gpt_bigcode` (queries; their keys and values take in-place edits), `mpt` (queries, keys, values); before the rotary `codegen`, `dbrx`, `gpt_neox` (queries, keys), `jetmoe` (keys), latent attention (the shared rope key) | assign, do not edit in place |
| logits scaled after the head | softcap: `gemma2`, `gemma3_text` (when set), `gemma4_text`, `gemma4_unified_text`, `vaultgemma`; `cohere`, `cohere2`, `granite` line, `hyperclovax`, `falcon_h1`, `mamba2` / `nemotron_h` (float32), `deepseek_v4` (`hc_head`) | `logits` != `lm_head.output`; `project_on_vocab` applies the step |
| per-block sizes | `gemma4_text`, `mimo_v2_flash`, `laguna` | read `layers[i].self_attn.*` / `.mlp.intermediate_size` |
| multimodal rotary (M-RoPE) | `qwen2_vl_text`, `qwen2_5_vl_text`, `qwen3_vl_text`, `qwen3_vl_moe_text` | folded into one `cos`/`sin` before the blocks: `attention_queries` / `_keys` are the rotated ones, as on any rotary family |
| DeepStack | `qwen3_vl_text`, `qwen3_vl_moe_text` | the text model adds image features at the image positions after blocks 0-2, outside the blocks: `layers[k].deepstack_output`, and `layers[k+1].input != layers[k].layer_output` there |
| vision tower host | `gemma3_text`, `gemma`, `qwen2`, `cohere2`, `llama`, `mistral`, `ministral3`, `qwen2_vl_text`, `qwen2_5_vl_text`, `qwen3_vl_text`, `qwen3_vl_moe_text`, `qwen3_5_text`, `qwen3_5_moe_text`, `llama4_text`, `gemma4_text`, `gemma4_unified_text` | their image-text-to-text wrappers have `model.vision` and `model.projector` under `task="image-text-to-text"`; which wrappers and towers: [vision.md](vision.md#which-wrappers) |

Not shipped: `zamba`, `zamba2` (a shared transformer's output is added to the
mixer's input, not the stream, so the standard contributions mean nothing there).
