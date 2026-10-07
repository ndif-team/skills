# Vision-language models

An image-text-to-text checkpoint (Gemma 3, Llava, Qwen3-VL, Mistral 3, Llama 4, ...) is a
text model plus a vision tower, a projector, and a step that scatters the projected image
features into the text stream at the image tokens. nnterp keeps the text model's standard
names, and the family is the text model's (`gemma3_text`, `llama`, ...). It adds names for
the vision side: the tower, its blocks, the projector, and two values of the tower for where
the image enters the text model.

The blocks here run on `trl-internal-testing/tiny-LlavaForConditionalGeneration` (a CLIP
tower and a Llama text model, every width 16, random weights), so they assert shapes and
identities, never values. nnterp's own pages: `docs/usage/vision.md` and
`docs/patterns/image-pathway.md`.

<!-- test: setup -->
```python
import torch
import nnsight
import nnterp
from PIL import Image
from nnterp import StandardizedTransformer, Unavailable

vlm = StandardizedTransformer(
    "trl-internal-testing/tiny-LlavaForConditionalGeneration",
    task="image-text-to-text", device="cpu", dispatch=True, attn_implementation="eager",
)
image = Image.new("RGB", (64, 64), "red")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}]}]
vprompt = vlm.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)   # places the image token
```

```python
with vlm.trace(vprompt, images=[image]):
    mask = vlm.vision.image_token_mask.save()                                # first: it comes off the inputs
    embedded = vlm.vision.patch_embeddings.save()
    vpattern = vlm.vision.layers[0].self_attn.attention_probabilities.save()
    patches = vlm.vision.layers[0].layer_output.save()
    tower = vlm.vision.tower_output.save()
    features = vlm.vision.image_features.save()
    first = vlm.layers[0].input.save()
    vlogits = vlm.logits.save()

assert vlm.family is nnterp.families.llama and isinstance(vlm.vision, nnterp.Vision)
assert mask.shape == (1, 592) and mask.sum() == 576
assert embedded.shape == (1, 576, 16)                    # one row per patch, before CLIP's CLS token
assert vpattern.shape == (1, 4, 577, 577)                # [images, heads, patches, patches]: CLS first
assert patches.shape == tower.shape == (1, 577, 16)      # [images, patches, vision_hidden]
assert features.shape == (576, 16)                       # [image_tokens, hidden]: flat
assert torch.equal(first[mask], features)                # the features are what enters the text model
assert vlogits.shape == (1, 592, 32064)
```

## The names

`model.vision` is the tower's root (a `Vision`), `vision.layers[i]` its blocks (a
`VisionLayer`, with `self_attn` a `VisionAttention` and `mlp` a `VisionMlp`),
`vision.patch_embed` the patch embedding, `vision.norm` the final norm over the patches
where the tower has one, and `model.projector` the last module before the scatter. The
native names they alias, per tower:

| tower | `vision` | `patch_embed` | `layers[i]` | `self_attn`, `mlp` | `input_layernorm`, `post_attention_layernorm` | `norm` | `projector` |
|---|---|---|---|---|---|---|---|
| SigLIP (Gemma 3, PaliGemma, llava-interleave, LLaVA-OneVision, Aya Vision, Cohere2-Vision, DeepSeek-VL, Idefics 3, SmolVLM) | `model.model.vision_tower` (DeepSeek-VL, Idefics 3, SmolVLM: `model.model.vision_model`) | `embeddings.patch_embedding` | `encoder.layers[i]` | native | `layer_norm1`, `layer_norm2` | `post_layernorm` | `model.model.multi_modal_projector` (Gemma 3's pools 4096 patches to 256 tokens; Aya Vision's and Cohere2-Vision's pixel-shuffle); DeepSeek-VL: `model.model.aligner`; Idefics 3, SmolVLM: `model.model.connector` |
| CLIP (Llava 1.5, VipLlava, LLaVA-NeXT, BakLLaVA) | `model.model.vision_tower` | `embeddings.patch_embedding` | `encoder.layers[i]` | native | `layer_norm1`, `layer_norm2` | none: CLIP's `post_layernorm` norms only the pooled CLS token | `model.model.multi_modal_projector` |
| Pixtral (Mistral 3, Pixtral-12B), a `PixtralVision` | `model.model.vision_tower` | `patch_conv` | `transformer.layers[i]` | `attention`, `feed_forward` | `attention_norm`, `ffn_norm` | none | `model.model.multi_modal_projector` (merges each 2x2 block of patches on Mistral 3) |
| the Qwen ViT (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3.5, Qwen3.5-MoE), a `QwenVision` | `model.model.visual` | `patch_embed` | `blocks[i]` | `attn` (a `QwenVisionAttention`), `mlp` | `norm1`, `norm2` | none: the merger norms its own input | `visual.merger`, inside the tower (folds each 2x2 block of patches into one token) |
| Llama 4's ViT | `model.vision_model` | `patch_embedding` (unfold + linear) | `model.layers[i]` | native | native | `layernorm_post` | `model.multi_modal_projector` (a linear) |
| Gemma 4's ViT | `model.model.vision_tower` | `patch_embedder.input_proj` (a linear; the 2D position embedding comes after) | `encoder.layers[i]` | native; the contributions are the post-norms' outputs (a sandwich block) | native: `post_attention_layernorm` follows the attention | none | `model.model.embed_vision` (an RMS norm and a linear) |
| Gemma 4 unified's encoder-free embedder | `model.model.embed_vision` | `patch_dense` | none: no blocks | none | none | none | `embed_vision.multimodal_embedder` (an RMS norm and a linear) |

`embed_tokens`, `layers`, `norm` and `lm_head` stay the language model's
(`model.language_model.*` on the wrapper; `language_model.model.*` on Llama 4's), and native
names keep working. `model.num_layers` and `model.hidden_size` are the text model's; the
tower's sizes are on `model.vision`. Gemma 4's audio tower and its embedder keep their native
names (`model.model.audio_tower`, `model.model.embed_audio`).

```python
assert vlm.vision.path == "model.model.vision_tower"
assert vlm.vision.layers[0].path == "model.model.vision_tower.encoder.layers.0"
assert vlm.vision.layers[0].input_layernorm.path.endswith("layer_norm1")
assert vlm.projector.path == "model.model.multi_modal_projector"
assert getattr(vlm.vision, "norm", None) is None                     # CLIP: no norm over the patches
assert (vlm.vision.num_layers, vlm.vision.hidden_size, vlm.vision.num_heads, vlm.vision.head_dim) == (2, 16, 4, 4)
assert (vlm.vision.image_size // vlm.vision.patch_size) ** 2 == 576  # 336 // 14 = 24 patches a side
```

## The values

| value | host | layout | meaning |
|---|---|---|---|
| `layer_output` | `vision.layers[i]` | `Patches` | the tower's stream leaving the block |
| `attention_output`, `mlp_output` | `vision.layers[i].self_attn`, `.mlp` | `Patches` | what each sublayer adds: `input + attention_output + mlp_output == layer_output` |
| `attention_probabilities`, `attention_queries`, ... | `vision.layers[i].self_attn` | `Pattern`, `Queries`, ... | as on a text block, the `batch` axis being the tower's rows; no causal mask (Pixtral masks between packed images, Gemma 4 masks its padded keys); needs eager |
| `patch_embeddings` | `vision` | `Patches` | the patch embedding's output, one row per patch, before any position embedding, CLS token or pre-norm |
| `tower_output` | `vision` | `Patches` | the last block's stream after `vision.norm` where there is one, before any pooling, CLS dropping or adapter |
| `image_token_mask` | `vision` | `ImageTokenMask` `[batch seq]` bool | `input_ids == config.image_token_id`, read off the model's inputs; read-only |
| `image_features` | `vision` | `ImageFeatures` `[image_tokens hidden]` | what the wrapper scatters into the token embeddings, flat over every image token in row-major order; assignable, in-place edits land |

`Patches` is `[images, patches, vision_hidden]`. What a row and the patch axis are, per tower:

| tower | rows | the patch axis |
|---|---|---|
| SigLIP | one per image (per crop on LLaVA-OneVision, per tile on Idefics 3 and SmolVLM) | the patches, in raster order |
| CLIP | one per image (per crop on LLaVA-NeXT) | the CLS token first, then the patches |
| Pixtral | one, packed: every image's patches, image after image | each image has `(height // patch_size) * (width // patch_size)` patches, its `image_sizes` entry from the processor; `patch_embeddings` is the packed row as it enters `ln_pre` |
| the Qwen ViT | one, packed (the tower runs on `[patches, vision_hidden]`, served with a leading 1) | every image's patches in merge-block order; on Qwen2.5-VL in window order inside the tower |
| Llama 4's ViT | one per image tile | the patches, then the CLS token last (`patches + 1` rows); the tower drops it after `vision.norm` |
| Gemma 4's ViT | one per image | the patches padded to `max_soft_tokens * pooling_kernel_size**2` rows (2520 by default); the padded rows are masked as keys but run through every block, so they are rows of `layer_output`, with values; the pooler zeroes and strips them |
| Gemma 4 unified's embedder | one per image | `patch_embeddings` and `tower_output` only, padded to `max_soft_tokens` rows (280) the same way |

**Sizes.** `model.vision` has `num_layers`, `hidden_size`, `num_heads`, `head_dim`,
`intermediate_size`, `patch_size`, `image_size`, read off the tower's own config. The Qwen
ViT, Pixtral, Gemma 4 and Gemma 4 unified take images of any resolution, so their
`image_size` raises `Unavailable` saying where each image's grid is (`image_grid_thw`,
`image_sizes`, `image_position_ids`). The encoder-free embedder has `num_layers == 0`,
`hidden_size` its `mm_embed_dim` and `patch_size` the 48-pixel merged patch it embeds; its
`num_heads`, `head_dim` and `intermediate_size` raise `Unavailable`.

Two images in one invoke give one `Patches` row each on CLIP, and the features stay flat:

```python
two = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "Compare them."}]}]
two_prompt = vlm.processor.apply_chat_template(two, add_generation_prompt=True, tokenize=False)

with vlm.trace(two_prompt, images=[image, Image.new("RGB", (64, 64), "blue")]):
    mask2 = vlm.vision.image_token_mask.save()
    rows2 = vlm.vision.layers[0].layer_output.save()
    features2 = vlm.vision.image_features.save()
    first2 = vlm.layers[0].input.save()

assert mask2.sum() == 1152                               # 576 image tokens per image
assert rows2.shape == (2, 577, 16) and features2.shape == (1152, 16)
assert torch.equal(first2[mask2], features2)
```

## Where the image meets the text model

`layers[0].input[vision.image_token_mask] == vision.image_features` holds exactly:
`image_features` is read at the scatter, the tensor the wrapper's forward writes into the
token embeddings at the image tokens, and nothing touches it before block 0. It is the place
to ablate, patch or steer the image as the text model sees it. On most wrappers it is the
projector's output, reshaped. Where the wrapper changes that output before scattering it,
`projector.output != image_features`:

- LLaVA-NeXT and LLaVA-OneVision unpad the projector's output and add a newline token per
  row (`model.model.image_newline`), so `projector.output` has another row count.
- Gemma 4 unified runs the projector on the padded rows too and strips them:
  `image_features == projector.output[valid]`, where `valid` is
  `(image_position_ids != -1).all(-1)`.
- Qwen2.5-VL restores the merge-block order after the merger, so `projector.output` is in
  window order once an image spans more than one 112-pixel window.

What feeds the projector differs per host: Gemma 3 pools `tower_output`; Llava takes
`vision.layers[-2].layer_output` without its CLS token (`vision_feature_layer=-2`), so a
write to `tower_output` or to the last block does not reach Llava's text model; VipLlava
concatenates several blocks' streams. `model.projector.input` is what the projector
receives. On Llama 4 the tower runs a pixel-shuffle adapter after `tower_output`
(`vision.vision_adapter`) whose output, flattened over the tiles, is `projector.input`; on
Gemma 4 the tower's `pooler` average-pools 3x3 patches of `tower_output` into each soft token
and strips the padding.

```python
with vlm.trace(vprompt, images=[image]):
    second_to_last = vlm.vision.layers[-2].layer_output.save()
    projected_in = vlm.projector.input.save()
    projected = vlm.projector.output.save()
    features = vlm.vision.image_features.save()

assert torch.equal(projected_in, second_to_last[:, 1:])       # Llava: block -2's stream, CLS dropped
assert torch.equal(projected.reshape(-1, 16), features)       # Llava 1.5 scatters the projector's output as it is
```

On `llava-hf/llava-1.5-7b-hf` asked the color of a red square (`Red` at 0.990): zeroing
`image_features` leaves `Red` at 0.010 with no color token on top; zeroing `tower_output`
changes nothing; zeroing `vision.layers[-2].layer_output` makes `</s>` the top token
(`Red` 0.001); and assigning the red run's features in a blue image's run answers `Red` at
0.990.

## Image positions and text positions

Every text-model value is over the whole sequence, image tokens and text tokens alike:
`layers[i].layer_output` is `[batch, seq, hidden]` with 576 of its 592 rows the image here,
and the text blocks' pattern has the image tokens as ordinary key positions.
`vision.image_token_mask` splits them:

```python
with vlm.trace(vprompt, images=[image]):
    m = vlm.vision.image_token_mask                              # the proxy works as an index; read it first
    mask = m.save()
    pattern = vlm.layers[0].self_attn.attention_probabilities.save()
    out = vlm.layers[0].layer_output.save()
    h = vlm.layers[1].layer_output
    h[m] = 0                                                     # a one-sided edit: the image positions only
    edited_out = h.save()
    edited = vlm.logits.save()

assert out[mask].shape == (576, 16) and out[~mask].shape == (16, 16)   # image rows, text rows
to_image = pattern[0, :, -1, mask[0]].sum(-1)                    # [heads]: the last token's mass on the image
assert to_image.shape == (vlm.num_heads,)
assert (edited_out[mask] == 0).all() and not (edited_out[~mask] == 0).all()
assert not torch.equal(edited, vlogits)
```

- Boolean indexing flattens the batch: `out[mask]` is `[image_tokens, hidden]` over every
  row of the batch in row-major order, the order of `vision.image_features`. To keep the
  batch shape, mask instead of indexing (`out.masked_fill(~mask[..., None], 0)`), or index
  one row (`out[0, mask[0]]`).
- A one-sided edit is an in-place write through the mask (`h[mask] = 0`, or
  `h[mask] = h[mask].mean(0)` to mean-ablate the image positions); the text positions are
  untouched.
- On `llava-hf/llava-1.5-7b-hf`, zeroing the image rows of `layer_output` after block
  4 / 8 / 16 / 24 leaves `Red` at 0.012 / 0.114 / 0.384 / 0.994: the text positions have read
  the image out by the middle of the stack. The last token's mean mass on the image is 0.73
  at block 0, under 0.1 from block 3, and 0.1 to 0.26 across blocks 10 to 24, where single
  heads put up to 0.95 of their mass on it. There `out[mask]` is `(576, 4096)` and
  `out[~mask]` `(23, 4096)`, and an eager trace under `torch.no_grad()` peaks at 14.6 GB in
  float16.

## Inputs and read order

- `model.trace(prompt, images=[image])`, the image placeholder in the prompt (the
  processor's chat template puts it there); or `model.trace(encoding)` with an encoding
  built by `model.processor`.
- `model.trace("text")` on the wrapper is a text-only trace: the mask is all false and
  `vision.image_features` is never reached. PaliGemma's processor demands an image; pass
  `dict(model.tokenizer(text, return_tensors="pt"))` there.
- **One image-carrying invoke per trace.** Several images go in one invoke, as lists. Two
  invokes of `model.trace(prompt, images=[...])` raise `NotImplementedError: Can't batch
  these inputs`. Chat-message inputs that embed the image do batch, and then
  `vision.image_features` is flat over the whole batch's image tokens in each invoke, so a
  write in one invoke reaches every row, while the mask is per invoke. Clean and ablated
  runs are two traces.
- Llama 4's image processor returns bfloat16 pixels, which a float32 tower refuses: load
  Llama 4 in bfloat16, or cast `pixel_values` in an encoding.
- Gemma 3's tower attends over 4096 patches; an eager trace keeps every block's
  16 x 4096 x 4096 pattern for autograd unless it runs under `torch.no_grad()`.

Read order is the forward's: `vision.image_token_mask` first (it comes off the inputs, like
`input_ids`), then `vision.patch_embeddings`, the tower's blocks (a block's attention
interior before its `layer_output`), `vision.tower_output`, `projector`, then
`vision.image_features`, then the text model's values. A late read raises
`OutOfOrderError`.

<!-- test: expect-error NotImplementedError -->
```python
with vlm.trace() as tracer:
    with tracer.invoke(vprompt, images=[image]):
        a = vlm.vision.image_features.save()
    with tracer.invoke(vprompt, images=[image]):
        b = vlm.vision.image_features.save()
```

## Under `generate`

The tower runs on the prompt call only. `vision.image_token_mask` is `[batch, prompt_len]`
on step 0 and `[batch, 1]`, all false, on every decode step; `image_features` and the
tower's values have one occurrence, step 0's, so read them under `tracer.iter[0]` (the mask
first). An edit before any step lands on the prompt call. The per-step table is in
[generation-and-helpers.md](generation-and-helpers.md#images-under-generate).

```python
with vlm.generate(vprompt, images=[image], max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[0]:
        step0_mask = vlm.vision.image_token_mask.save()
        step0_features = vlm.vision.image_features.save()
    ids = tracer.result.save()

with vlm.generate(vprompt, images=[image], max_new_tokens=3, do_sample=False) as tracer:
    vlm.vision.image_features[:] = 0                      # the model generates without the image
    blind = tracer.result.save()

assert step0_mask.shape == (1, 592) and step0_features.shape == (576, 16)
assert ids.shape == blind.shape == (1, 595) and not torch.equal(ids, blind)
```

## The Qwen ViT

Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-VL-MoE, Qwen3.5 and Qwen3.5-MoE share one tower at
`model.visual`. It runs on `[patches, vision_hidden]`, every image of the invoke
concatenated, and its attention calls the interface once per image (once per window on
Qwen2.5-VL's windowed blocks).

- Its `Patches` values are `[1, patches, vision_hidden]`: the packed tensor with a leading
  images axis of 1, a view, so in-place edits land; assign the same shape.
  `vision.layers[i].output` stays the native `[patches, vision_hidden]`.
- The processor's `image_grid_thw` (`[t, h, w]` per image, in patches) splits the row:
  image `j` has `t * h * w` patches, in merge-block order (each 2x2 block the merger folds is
  consecutive), not raster order. On Qwen2.5-VL the block values and `tower_output` are in
  window order.
- `attention_queries`, `attention_keys` and `attention_values` are the whole
  `[1, heads, patches, head_dim]` tensors before the module splits them (queries and keys
  after the 2D rotary embedding), and `attention_head_outputs` is the per-image outputs
  concatenated back, `[1, patches, heads, head_dim]`, under any implementation but flash.
  `attention_scores` and `attention_probabilities` are `Unavailable` under every
  implementation. Split the queries and keys at the attention's `cu_seqlens` argument
  (`self_attn.inputs[1]["cu_seqlens"]`) to compute a per-image pattern.
- The sizes add `spatial_merge_size` and `window_size` (Qwen2.5-VL's, in pixels; `None`
  elsewhere); `hidden_size` is the tower's width (`embed_dim` on Qwen2-VL).

**DeepStack.** On `qwen3_vl_text` and `qwen3_vl_moe_text` the tower also taps three of its
blocks, and the text model adds each tap's merged features at the image positions after
text blocks 0, 1 and 2, outside the blocks. `layers[k].deepstack_output`
(`[image_tokens, hidden]`, assignable) is what is added after block `k`:
`layers[k+1].input[mask] == layers[k].layer_output[mask] + layers[k].deepstack_output`. On
the other blocks it is `Unavailable`. On `Qwen/Qwen3-VL-4B-Instruct`, asked the color of a
red square on white, zeroing `image_features` alone still answers "Red"; zeroing
`deepstack_output` on blocks 0-2 too answers "White".

<!-- test: skip -->
```python
qwen = StandardizedTransformer("Qwen/Qwen3-VL-4B-Instruct", task="image-text-to-text", dispatch=True, dtype=torch.bfloat16)
content = [{"type": "image"}, {"type": "text", "text": "What color is the square?"}]
qprompt = qwen.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)

with torch.no_grad(), qwen.trace(qprompt, images=[image]):
    qwen.vision.image_features[:] = 0
    for k in range(3):
        qwen.layers[k].deepstack_output[:] = 0         # in forward order: after block k
    ablated = qwen.logits.save()
```

**M-RoPE.** The text blocks of the Qwen VL families use multimodal rotary embeddings
(temporal, height and width position streams), which the model folds into one `cos`/`sin`
before the blocks; the attention applies it before the interface, so `attention_queries` and
`attention_keys` are the rotated queries and keys, as on any rotary family.

## Availability

`model.vision.support()` lists the tower's values and its block values the way
`model.support()` lists the text blocks', and `model.support()` carries the same rows under
`vision.` (`"vision.image_features"`, `"vision.self_attn.attention_probabilities"`). A
text-only checkpoint has no `model.vision` at all: guard on
`getattr(model, "vision", None)` first and on `support()` second.

```python
rows = {k for k in vlm.support() if k.startswith("vision.")}
assert {"vision.image_token_mask", "vision.image_features", "vision.layer_output",
        "vision.self_attn.attention_probabilities", "vision.mlp.mlp_output"} <= rows
assert vlm.vision.support()["image_features"] is None and vlm.vision.support()["layer_output"] is None
```

The tower runs only where an image can reach the model: a wrapper loaded with its processor.
Gemma 3, Gemma 4 and Gemma 4 unified build the wrapper for the default
`task="text-generation"`; so do the Llava-class and Qwen-VL wrappers when dispatched. That
load has no processor. There `model.vision.support()` is empty, `model.support()` has no
`vision.` rows, and every tower value raises `Unavailable`, inside a trace too:

<!-- test: expect-error Unavailable -->
```python
text_only = StandardizedTransformer("trl-internal-testing/tiny-LlavaForConditionalGeneration", device="cpu", dispatch=True)
assert text_only.vision.support() == {} and not any(k.startswith("vision.") for k in text_only.support())
text_only.vision.image_features
# Unavailable: model.model.vision_tower.image_features is not available: a text-only load: no processor,
#   so no image reaches the model; load with task='image-text-to-text'
```

| reason (exact, or its pattern) | value(s) | when |
|---|---|---|
| `a text-only load: no processor, so no image reaches the model; load with task='image-text-to-text'` | every value of `model.vision` and its blocks | a wrapper loaded under the default `task="text-generation"` |
| `the <family> family names no projector on this wrapper` | every tower value | the family does not name where the image enters the text model |
| `the <family> family keys no ImageScatter on the '<model_type>' wrapper, so where its image features enter the text stream is unknown` | `vision.image_features` | the tower's names bind but the family does not say where the wrapper writes the features in |
| `the config names no image_token_id` | `vision.image_token_mask`, `vision.image_features` | the config has no image token id |
| `<tower>.image_size is not available: the tower takes images of any resolution, each cut into its own patch grid by the processor; read the grid off the processor's output (image_grid_thw, image_sizes, image_position_ids)` | `vision.image_size` (a size, raised at the read; not a `support()` row) | the Qwen ViT, Pixtral, Gemma 4, Gemma 4 unified |
| `the Qwen ViT's attention makes one interface call per image (attention_interface_2; ...` | the Qwen ViT's `attention_scores`, `attention_probabilities` | always |
| `flash attention runs every image in one call over cu_seqlens, with no concatenation to read; load with attn_implementation='eager' or 'sdpa'` | the Qwen ViT's `attention_head_outputs` | a flash load |

`vision.image_features` is read where the family says the wrapper writes the features in: an
`ImageScatter` keyed on the wrapper's model, or, on Llama 4, whose top-level forward
scatters, the family's `ROOT_SCATTER`.

## Which wrappers

16 families host a tower:

| family | wrapper (`model_type`) | tower |
|---|---|---|
| `gemma3_text` | Gemma 3 (`gemma3`) | SigLIP |
| `gemma` | PaliGemma (`paligemma`) | SigLIP |
| `qwen2` | llava-interleave (`llava`), LLaVA-OneVision (`llava_onevision`) | SigLIP |
| `cohere2` | Aya Vision (`aya_vision`), Cohere2-Vision (`cohere2_vision`) | SigLIP |
| `llama` | Llava 1.5 (`llava`), VipLlava (`vipllava`), LLaVA-NeXT (`llava_next`) | CLIP |
| `llama` | DeepSeek-VL (`deepseek_vl`) | SigLIP |
| `llama` | Idefics 3 (`idefics3`), SmolVLM (`smolvlm`) | their SigLIP-shaped ViT, one row per tile |
| `mistral` | LLaVA-NeXT (`llava_next`, `llava-v1.6-mistral`), BakLLaVA (`llava`; no tiny checkpoint, so untested: the keys are LLaVA-NeXT's tower and Llava's scatter) | CLIP |
| `mistral` | Mistral 3 (`mistral3`, Mistral Small 3.1 / 3.2), Pixtral-12B (`llava`) | Pixtral |
| `ministral3` | Mistral 3 (`mistral3`, Ministral 3) | Pixtral |
| `qwen2_vl_text`, `qwen2_5_vl_text`, `qwen3_vl_text`, `qwen3_vl_moe_text` | Qwen2-VL (`qwen2_vl`), Qwen2.5-VL (`qwen2_5_vl`), Qwen3-VL (`qwen3_vl`), Qwen3-VL-MoE (`qwen3_vl_moe`) | the Qwen ViT |
| `qwen3_5_text`, `qwen3_5_moe_text` | Qwen3.5 (`qwen3_5`), Qwen3.5-MoE (`qwen3_5_moe`) | the Qwen ViT |
| `llama4_text` | Llama 4 (`llama4`) | Llama 4's ViT |
| `gemma4_text` | Gemma 4 (`gemma4`) | Gemma 4's ViT |
| `gemma4_unified_text` | Gemma 4 unified (`gemma4_unified`) | the encoder-free embedder |

`python scripts/inspect_family.py <repo_id> --task image-text-to-text` prints a wrapper's
tower names, sizes and `model.vision.support()` on the meta device.

## What is not covered

- The towers of EXAONE 4.5 and LightOnOCR: the text names bind on their wrappers, but the
  towers and projectors are native-only, and there is no `model.vision` there.
- Mllama: its text model is a type nnterp has no family for.
- Video and audio values; batching several image-carrying invokes in one trace.
