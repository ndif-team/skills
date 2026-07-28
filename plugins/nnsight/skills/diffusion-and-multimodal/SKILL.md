---
name: diffusion-and-multimodal
description: Trace and intervene on models that are not text-only — vision-language models (image+text via TransformersModel with a processor), diffusion pipelines via DiffusionModel including the diffusion lens and per-timestep interventions, and audio/vision encoders. Use to find where image information enters the language stream, to patch or ablate image tokens versus text tokens, to read what a diffusion model has drawn at intermediate denoising steps, or to work with any HuggingFace task beyond text generation.
---

# Diffusion and Multimodal Models

nnsight's API does not change for these models — `trace`, `.output`, `save`,
invokes, gradients all work the same. What changes is **how you build the input**
and **where the module tree puts things**.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel
from PIL import Image
```

## Vision-language models

Load with `task="image-text-to-text"`; the pipeline brings a **processor** rather
than a bare tokenizer.

```python
vlm = TransformersModel("llava-hf/llava-interleave-qwen-0.5b-hf",
                        task="image-text-to-text", dispatch=True)

print(type(vlm._module).__name__, type(vlm.processor).__name__)
```

Build the input with the processor's chat template, then pass the encoded batch
straight to `trace`:

```python
image = Image.new("RGB", (224, 224), (120, 30, 200))       # a solid purple square

messages = [{
    "role": "user",
    "content": [{"type": "image"}, {"type": "text", "text": "What color is this?"}],
}]
text = vlm.processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = vlm.processor(images=image, text=text, return_tensors="pt")

with vlm.trace(inputs):
    hidden = vlm.model.language_model.layers[10].output.save()
    logits = vlm.output.logits.save()

print(hidden.shape, logits.shape)
```

```
torch.Size([1, 743, 1024]) torch.Size([1, 743, 152000])
```

**743 positions for a short question.** Almost all of them are image patch tokens.
This is the first thing to internalize about VLM interpretability: the sequence is
dominated by vision tokens, and "position -1" is the only text position you can
index without counting.

### Where things live

A VLM has two stacks, and the language stack is nested:

```python
paths = [name for name, _ in vlm.named_modules()]
print([p for p in paths if "vision" in p][:3])
print([p for p in paths if p.endswith("language_model")][:2])
```

Typical layout: `model.model.vision_tower...` for the encoder,
`model.model.multi_modal_projector` for the adapter that maps vision features into
the language embedding space, and `model.model.language_model.layers[i]` for the
text stack. Confirm with `scripts/inspect_model.py <repo> --grep vision` from the
`nnsight` skill — the nesting differs across VLM families and changes between
`transformers` versions.

### Reading the vision path

The projector output is the interesting boundary — it is literally what the image
becomes inside the language model:

```python
with vlm.trace(inputs):
    projected = vlm.model.multi_modal_projector.output.save()

print("image features entering the language stream:", projected.shape)
```

### Separating image tokens from text tokens

Most VLM interpretability questions are "does this come from the image or the
text?" Find the image span from the processor's special token, then index it:

```python
input_ids = inputs["input_ids"][0]
image_token_id = vlm.config.image_token_id
image_positions = (input_ids == image_token_id).nonzero().flatten()

print(f"{len(image_positions)} image positions, "
      f"{len(input_ids) - len(image_positions)} text positions")
```

Once you have the span, every technique in the other skills applies to it: ablate
the image positions and see whether the answer survives, patch them from a
different image, or run a logit lens over them.

```python
with vlm.trace(inputs):
    clean = vlm.output.logits[0, -1].argmax().save()

with vlm.trace(inputs):
    vlm.model.language_model.layers[10].output[:, image_positions, :] = 0
    no_image = vlm.output.logits[0, -1].argmax().save()

print(f"with image {vlm.processor.tokenizer.decode(clean)!r}   "
      f"image ablated {vlm.processor.tokenizer.decode(no_image)!r}")
```

Two separate traces, not two invokes — **a processor's encoded output cannot be
batched across invokes**:

```
NotImplementedError: Can't batch these inputs; pass text or token ids.
```

nnsight can batch text and token ids, but not a multimodal payload with pixel
values attached. So the sweep pattern from the `nnsight` skill (one invoke per
variant, one forward pass) does not apply to VLM image inputs: each condition is
its own forward pass. Plan experiments accordingly — a 12-layer ablation sweep on
a VLM is 12 passes, not 1.

Note `vlm.processor.tokenizer` — on a VLM, `model.tokenizer` may be `None`, since
the tokenizer lives inside the processor.

## Diffusion models

> The diffusion examples below are **not executed by this repo's test suite** —
> the tiny pipelines available offline have broken configs and real ones are large
> downloads. They follow the nnsight 0.8 `DiffusionModel` API; verify against your
> own pipeline.

`DiffusionModel` wraps any `diffusers` pipeline. Its components — `unet` (or
`transformer`), `vae`, `text_encoder`, `scheduler` — are envoys.

<!-- test: skip -->
```python
from nnsight import DiffusionModel

sd = DiffusionModel("stabilityai/stable-diffusion-2-1-base", dispatch=True)

with sd.generate("a photograph of an astronaut riding a horse",
                 num_inference_steps=20) as tracer:
    mid = sd.unet.mid_block.output.save()          # the bottleneck
    image = tracer.result.save()

print(mid.shape)
image.images[0].save("out.png")
```

### Per-timestep interventions

Denoising runs the UNet once per step, so timesteps are `tracer.iter`
occurrences — exactly like generation steps in a language model:

<!-- test: skip -->
```python
with sd.generate(prompt, num_inference_steps=20) as tracer:
    trajectory = nnsight.save([])
    for step in tracer.iter[:20]:
        trajectory.append(sd.unet.mid_block.output.mean().cpu())
        if step > 10:
            sd.unet.mid_block.output[:] *= 1.1      # amplify late steps only
    image = tracer.result.save()
```

Bound the loop, as always — an unbounded `tracer.all()` drops everything after it.

### The diffusion lens

The analogue of the logit lens: decode an intermediate UNet state through the VAE
to see what the model has "drawn" so far.

<!-- test: skip -->
```python
with sd.generate(prompt, num_inference_steps=20) as tracer:
    for step in tracer.iter[:20]:
        latents = sd.unet.conv_out.output
        if step in (0, 5, 10, 19):
            preview = sd.vae.decode(latents / sd.vae.config.scaling_factor).sample.cpu().save()
```

Early steps show layout and colour blocks; later steps sharpen detail. Comparing
that trajectory across prompts is the diffusion-model equivalent of asking when a
prediction is decided.

### Cross-attention: where the prompt acts

Text conditioning enters through cross-attention in the UNet blocks. Those are the
modules to ablate or patch when asking which words drove which part of the image:

<!-- test: skip -->
```python
with sd.generate(prompt, num_inference_steps=20) as tracer:
    for step in tracer.iter[:20]:
        sd.unet.mid_block.attentions[0].transformer_blocks[0].attn2.output[:] = 0
    ablated = tracer.result.save()
```

Path names vary between UNet and DiT architectures — inspect first.

## Other modalities

Everything HuggingFace can build a pipeline for works through `TransformersModel`
with the right `task=`: `image-classification` (`image_processor`),
`automatic-speech-recognition` and `audio-classification` (`feature_extractor`),
`fill-mask`, `text-classification`. The preprocessor attribute that is populated
depends on the task; the others are `None`.

<!-- test: skip -->
```python
vision = TransformersModel("facebook/dinov3-vits16-pretrain-lvd1689m",
                           task="image-feature-extraction", dispatch=True)

with vision.trace(image):
    patch_features = vision.encoder.layer[6].output.save()
```

## What to watch for

**Sequence length is dominated by non-text tokens.** Position indices copied from
a text-only experiment will be wrong. Compute spans from the input ids.

**`model.tokenizer` may be `None`.** Use `model.processor.tokenizer`.

**Module nesting is deeper and less stable.** VLM layouts move between
`transformers` releases more than text models do — inspect, do not assume.

**`scan` often fails on multimodal pipelines**, because preprocessing branches on
tensor data that fake tensors cannot supply. Use a real trace for shapes.

**Diffusion has no single "output token".** Metrics are image-space or
latent-space, so decide what you are measuring before intervening — the
`interp-experiment-design` skill covers choosing one.

## Related skills

- `nnsight` — the API, module paths, iteration
- `logit-lens` — the text-model analogue of the diffusion lens
- `activation-patching`, `ablation` — apply directly to image-token spans
- `attention-analysis` — cross-attention in diffusion, vision attention in VLMs
