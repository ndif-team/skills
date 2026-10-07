# The image pathway

Ablate the image where it enters the text model or inside the tower, patch one image's
features into another image's run, measure each head's attention onto the image, and edit
the image positions of a text block without touching the text. nnterp pages:
`docs/patterns/image-pathway.md`, `docs/usage/vision.md`, `docs/usage/layouts.md`,
`docs/usage/availability.md`, `docs/usage/generation.md`.

## What this is for

An image enters a vision-language model once: the tower turns it into patches, the
projector maps them into the text model's width, and the wrapper scatters the result into
the token embeddings at the image tokens. From there on the image is a run of positions in
the text model's sequence, and every text-block value covers it. nnterp gives the pathway
three handles that mean the same thing on every wrapper:

- `vision.image_features`, `[image_tokens, hidden]`: what the text model receives at the
  image tokens, read at the scatter, so `layers[0].input[mask] == image_features` exactly.
  Ablate or patch here to change *the image as the text model sees it*.
- `vision.image_token_mask`, `[batch, seq]`: which positions are the image. Index any
  text-block value with it to split the image rows from the text rows.
- `vision.layers[i]`: the tower's blocks, with `layer_output`, `attention_output`,
  `mlp_output` and the attention interior over the patches. Ablate here to change *what the
  tower computes*.

The same code runs on every wrapper in the table under "Which wrappers" in nnterp's
`docs/usage/vision.md` (Llava 1.5, LLaVA-NeXT, Gemma 3, PaliGemma, the Qwen-VL and Qwen3.5
wrappers, Mistral 3, Llama 4, Gemma 4 and the rest). The executed blocks below run on
`trl-internal-testing/tiny-LlavaForConditionalGeneration` (random weights, every width 16,
a 2-block CLIP tower, a 2-block Llama text model, float16), so they assert shapes and
identities. The numbers in prose are nnterp's, from `llava-hf/llava-1.5-7b-hf` with a red
square on white and the question "What color is the square? Answer with one word.",
answered `Red` at 0.990.

## Canonical pattern

Load with `task="image-text-to-text"` (the default `text-generation` load has no processor,
and every tower value raises `Unavailable`), build the prompt with the processor's chat
template (it places the image token), and read the mask before anything else: it comes off
the inputs.

<!-- test: setup -->
```python
import torch
from PIL import Image
from nnterp import StandardizedTransformer

model = StandardizedTransformer("trl-internal-testing/tiny-LlavaForConditionalGeneration", task="image-text-to-text",
                                device="cpu", dispatch=True, attn_implementation="eager")
red, blue = Image.new("RGB", (64, 64), "red"), Image.new("RGB", (64, 64), "blue")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the square? Answer with one word."}]}]
prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
```

```python
with model.trace(prompt, images=[red]):
    mask = model.vision.image_token_mask.save()           # [1, seq], 576 true
    red_features = model.vision.image_features.save()     # [576, hidden]
    first = model.layers[0].input.save()
    clean = model.logits.save()

with model.trace(prompt, images=[red]):
    model.vision.image_features[:] = 0                    # the text model receives zeros at the image tokens
    zeroed = model.logits.save()

assert mask.shape == (1, clean.shape[1]) and int(mask.sum()) == 576
assert red_features.shape == (576, model.hidden_size)
assert torch.equal(first[mask], red_features)             # the features are what enters the text model
assert not torch.equal(zeroed, clean)
```

On the 7B, with the image gone the model has nothing to answer about: `Red` falls from
0.990 to 0.010, `''` is on top, and no color token takes its place. One image-carrying
invoke per trace: the clean and the ablated runs are two traces, and `red_features` is an
ordinary tensor by the time the second runs.

## Ablate inside the tower

A tower block's contribution is zeroed like a text block's. Which block reaches the text
model depends on what the wrapper feeds the projector: Llava takes block -2's stream
without its CLS token (`vision_feature_layer=-2`), so the last block and `tower_output`
are computed and discarded, while Gemma 3 pools `tower_output` itself.

```python
with model.trace(prompt, images=[red]):
    model.vision.tower_output[:] = 0
    no_tower_output = model.logits.save()

with model.trace(prompt, images=[red]):
    model.vision.layers[-1].layer_output[:] = 0
    no_last_block = model.logits.save()

with model.trace(prompt, images=[red]):
    model.vision.layers[-2].layer_output[:] = 0
    no_block_m2 = model.logits.save()

with model.trace(prompt, images=[red]):
    read = model.vision.layers[-2].layer_output.save()    # [1, 577, vision_hidden]: CLS first
    projected = model.projector.input.save()              # what the projector actually receives

assert torch.equal(no_tower_output, clean)                # Llava never reads it
assert torch.equal(no_last_block, clean)
assert not torch.equal(no_block_m2, clean)                # this is what the projector reads
assert torch.equal(projected, read[:, 1:])                # block -2's patches, CLS dropped
```

```
tower_output zeroed               top='Red'  P(Red)=0.990
tower block -2 zeroed             top='</s>' P(Red)=0.001
tower block 0 MLP zeroed          top='Red'  P(Red)=0.932
```

Those are the 7B's (24 tower blocks). `model.projector.input` is what the projector
receives on every wrapper; when a tower edit does nothing, compare it with the value you
edited.

## Patch one image into another's run

Activation patching across images: save `image_features` from the red run (done above) and
assign it in the blue run. The text model then sees the red image under the blue image's
prompt.

```python
with model.trace(prompt, images=[blue]):
    blue_clean = model.logits.save()

with model.trace(prompt, images=[blue]):
    model.vision.image_features[:] = red_features         # same shape: both images give 576 tokens
    patched = model.logits.save()

assert not torch.equal(blue_clean, clean)
assert torch.equal(patched, clean)                        # same prompt, same features: the red run, bit for bit
```

On the 7B the blue run answers `Blue` at 0.989, and with the red features assigned `Red`
at 0.990. The two runs must have the same number of image tokens for the assignment to
fit: on a fixed-resolution tower (CLIP, SigLIP) every image does; on a variable-resolution
one (the Qwen ViT, Pixtral, Gemma 4) use images of the same size. To patch part of the
image, index the features by patch, in the tower's row order (CLIP's and SigLIP's are
raster order, 24 x 24 patches on Llava 1.5):

```python
top_half = slice(0, 12 * 24)                              # the first 12 rows of the 24 x 24 grid
with model.trace(prompt, images=[blue]):
    model.vision.image_features[top_half] = red_features[top_half]
    half_patched = model.logits.save()

assert not torch.equal(half_patched, blue_clean) and not torch.equal(half_patched, clean)
```

## Attention onto the image

The text blocks' pattern is `[batch, heads, query, key]` over the whole sequence, so the
mass a head puts on the image is the pattern summed over the image key columns. Per block,
from the last token:

```python
masses = []                                   # made outside the block: a name bound inside does not survive it
with model.trace(prompt, images=[red]):
    image = model.vision.image_token_mask     # first in the trace: it comes off the inputs
    for layer in model.layers:
        probabilities = layer.self_attn.attention_probabilities
        masses.append(probabilities[0, :, -1, image[0]].sum(-1).save())   # [heads]: the last token's mass on the image

stacked = torch.stack(masses).float()         # [layers, heads]
mean, peak = stacked.mean(-1), stacked.amax(-1)   # per block: the average head, the head that looks most
assert stacked.shape == (model.num_layers, model.layers[0].self_attn.num_heads)
assert ((stacked >= 0) & (stacked <= 1 + 1e-3)).all()
```

On the 7B's red square the mean mass is 0.73 at block 0 and 0.42 at block 1, under 0.1
from block 3 on, then between 0.1 and 0.26 across blocks 10 to 24, where single heads put
0.75 to 0.95 of their mass on the image (blocks 12, 14, 18, 19, 22). Those are the heads to
take to `references/patching-and-attention.md`. In a two-image prompt, a head's mass on
one image is the same expression with that image's slice of the mask.

## Edit the image positions of a text block

Boolean indexing with the mask writes the image rows and leaves the text rows alone, so an
edit at one text block asks when the text positions have finished reading the image:

```python
rows = {}
for k in range(model.num_layers):
    with model.trace(prompt, images=[red]):
        image = model.vision.image_token_mask
        h = model.layers[k].layer_output
        h[image] = 0                                      # the image rows leaving block k; text rows untouched
        rows[k] = model.logits.save()

with model.trace(prompt, images=[red]):
    image = model.vision.image_token_mask
    h = model.layers[0].layer_output
    h[image] = h[image].mean(0)                           # mean-ablate: every image row replaced by their average
    mean_ablated = model.logits.save()

with model.trace(prompt, images=[red]):
    mask = model.vision.image_token_mask.save()
    out = model.layers[0].layer_output.save()

assert all(not torch.equal(rows[k], clean) for k in rows) and not torch.equal(mean_ablated, clean)
assert out[mask].shape == (576, model.hidden_size)         # the image rows, the batch flattened
assert out[~mask].shape == (out.shape[1] - 576, model.hidden_size)   # the text rows
```

On the 7B:

```
image rows zeroed after block 4    top=''      P(Red)=0.012 P(Blue)=0.008
image rows zeroed after block 8    top='Black' P(Red)=0.114 P(Blue)=0.086
image rows zeroed after block 16   top='Red'   P(Red)=0.384 P(Blue)=0.013
image rows zeroed after block 24   top='Red'   P(Red)=0.994 P(Blue)=0.000
```

The image is read out of its positions by the middle of the stack: after block 24 the image
rows no longer matter. Mean-ablating instead (the image's identity removed but not its
presence) gives `White` at 0.27 after block 4 and `Red` at 0.62 after block 16. There
`layers[16].layer_output[mask]` is `[576, 4096]` and `[~mask]` is `[23, 4096]`. To keep
the batch shape, mask instead of indexing (`out.masked_fill(~mask[..., None], 0)`), or
index one row (`out[0, mask[0]]`).

## Under generate

The tower runs on the prompt call only. Read the mask and the features under
`for step in tracer.iter[0]:`, the mask first; on every decode step the mask is `[batch, 1]`, all
false. An edit before any step lands on the prompt call:

```python
with model.generate(prompt, images=[red], max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[0]:
        step0_mask = model.vision.image_token_mask.save()
        step0_features = model.vision.image_features.save()
    plain_ids = tracer.result.save()

with model.generate(prompt, images=[red], max_new_tokens=3, do_sample=False) as tracer:
    model.vision.image_features[:] = 0                    # the model generates without the image
    blind_ids = tracer.result.save()

assert torch.equal(step0_mask, mask) and step0_features.shape == (576, model.hidden_size)
assert not torch.equal(plain_ids, blind_ids)
```

## Gotchas

- Read `vision.image_token_mask` first in every trace that uses it, and `image_features`
  after the tower's values and before the text model's; a later read raises
  `OutOfOrderError` in a plain trace.
- One image-carrying invoke per trace, so clean and ablated runs are separate traces;
  several images go in one invoke as a list, and the mask covers them all.
  `model.trace(prompt, images=[...])` refuses to batch two image-carrying invokes.
  Chat-message inputs that embed the image (`{"type": "image", "image": img}`) do run as
  one batch, but there `image_features` is flat over the whole batch's image tokens in
  every invoke, and a write in one invoke reaches every row, while the mask is per invoke.
- `image_features` is zeroed before block 0, but it is not always the only way in: Qwen3-VL
  re-adds the image after text blocks 0 to 2 (`layers[k].deepstack_output`), so zero those
  too. On `Qwen/Qwen3-VL-4B-Instruct`, zeroing `image_features` alone still answers "Red";
  zeroing `deepstack_output` on blocks 0-2 too answers "White".
- A tower edit that changes nothing is usually at a block the wrapper does not read (Llava's
  last block, `tower_output`); check `model.projector.input`.
- Patching the whole of `image_features` needs the same image-token count in both runs: a
  fixed-resolution tower (CLIP, SigLIP), or same-size images on a variable-resolution one.
- Eager attention over a big tower holds every block's pattern for autograd: wrap the trace
  in `torch.no_grad()` (Gemma 3's 4096 patches ran out of a 48 GB card without it; the 7B
  Llava recipe peaked at 14.6 GB in float16 with it).
- On a text-only load (the default `task="text-generation"`) every tower value raises
  `Unavailable` and `model.support()` has no `vision.` rows.
