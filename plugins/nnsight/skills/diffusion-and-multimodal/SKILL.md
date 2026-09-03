---
name: diffusion-and-multimodal
description: Trace and intervene on models that are not text-only — vision-language models (image+text via TransformersModel with a processor), diffusion pipelines via DiffusionModel, and audio/vision encoders. Use to find where image information enters the language stream, to patch or ablate image tokens versus text tokens, to run the diffusion lens over a text encoder or decode a partly-denoised latent, or to work with any HuggingFace task beyond text generation, including the chunked ones (ASR, NER, zero-shot).
---

# Diffusion and Multimodal Models

`trace`, `.output`, `save` and gradients work on these models exactly as they do
on a language model. What changes is **how you build the input**, **where the
module tree puts things**, and one hard limit: a multimodal input carries pixel
or audio tensors, and those cannot be padded into a batch, so each condition in a
sweep is its own forward pass.

<!-- test: setup -->
```python
import numpy as np
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

print(type(vlm.processor).__name__)          # LlavaProcessor
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
assert hidden.shape[1] == inputs["input_ids"].shape[1]
```

```
torch.Size([1, 743, 1024]) torch.Size([1, 743, 152000])
```

**743 positions for a short question.** Almost all of them are image patch
tokens, so a position index copied from a text-only experiment points at the
wrong thing, and `-1` is the only text position you can index without counting.
Sequence length is not uniformly image-dominated either: the same 224x224 image
is 729 tokens on Llava and 75 *total* tokens on Qwen3-VL, whose patch grid
adapts.

### Where things live

A VLM has two stacks, and the language stack is nested inside the model. The
names are not shared across families. Measured on transformers 5.15:

| repo | vision stack | adapter | text stack |
|---|---|---|---|
| `llava-hf/llava-interleave-qwen-0.5b-hf` | `model.vision_tower` | `model.multi_modal_projector` | `model.language_model` |
| `trl-internal-testing/tiny-LlavaForConditionalGeneration` | `model.vision_tower` | `model.multi_modal_projector` | `model.language_model` |
| `hf-internal-testing/tiny-random-Idefics3ForConditionalGeneration` | `model.vision_model` | *(none exposed)* | *(no `language_model`)* |
| `Qwen/Qwen3-VL-4B-Instruct` | `model.visual` (+ `merger`) | *(none)* | `model.language_model` |

Check before you index. `scripts/inspect_model.py <repo> --grep vision` from the
`nnsight` skill prints the paths.

### Reading the vision path

The projector output is the boundary worth watching: it is what the image
becomes inside the language model.

```python
with vlm.trace(inputs):
    projected = vlm.model.multi_modal_projector.output.save()

print("image features entering the language stream:", projected.shape)
assert projected.shape[1] == 729
```

### Separating image tokens from text tokens

Most VLM questions are "does this come from the image or the text?". The image
positions come from the processor's special token:

```python
input_ids = inputs["input_ids"][0]
image_positions = (input_ids == vlm.config.image_token_id).nonzero().flatten()

print(f"{len(image_positions)} image positions, "
      f"{len(input_ids) - len(image_positions)} text positions")
assert (image_positions.diff() == 1).all()        # one contiguous span
```

Those positions exist only because the **processor** built the ids. The template
string holds a single `<image>`, and the processor expands it into 729 copies;
the tokenizer on its own leaves the one. Ids from the tokenizer paired with
pixels from the processor fail deep in modeling code with `ValueError: Image
features and image tokens do not match, tokens: 1, features: 746496`.

```python
tokenizer_only = vlm.processor.tokenizer(text, return_tensors="pt")["input_ids"][0]
print("processor:", len(input_ids), " tokenizer alone:", len(tokenizer_only))
assert (tokenizer_only == vlm.config.image_token_id).sum() == 1
```

With the span in hand, every technique in the other skills applies to it: ablate
the image positions, patch them from a different image, run a logit lens over
them.

### Choosing a metric that can move

Zeroing the image positions mid-stack leaves the top-1 token where it was, so an
`argmax` comparison shows nothing and makes a working intervention look broken.
Read the distribution instead:

```python
def top5(row):
    probs = row.detach().softmax(-1)
    values, ids = probs.topk(5)
    return [(vlm.tokenizer.decode(i), round(float(v), 3)) for i, v in zip(ids, values)]

with vlm.trace(inputs):
    clean = vlm.output.logits[0, -1].save()

with vlm.trace(inputs):
    vlm.model.language_model.layers[10].output[:, image_positions, :] = 0
    ablated = vlm.output.logits[0, -1].save()

print("clean  ", top5(clean))
print("ablated", top5(ablated))
assert top5(clean)[0][0] == top5(ablated)[0][0]        # same top-1 either way
assert abs(top5(clean)[1][1] - top5(ablated)[1][1]) > 0.2
```

```
clean   [('The', 0.605), ('This', 0.324), ('It', 0.056), ('I', 0.004), ('You', 0.004)]
ablated [('The', 0.805), ('A', 0.031), ('This', 0.026), ('If', 0.02), ('It', 0.015)]
```

Generation is the other metric that moves. Zeroing the projector cuts every
image feature at once, and the answer changes even though the first token does
not:

```python
with vlm.generate(inputs, max_new_tokens=12, do_sample=False) as tracer:
    ids_clean = tracer.result.save()

with vlm.generate(inputs, max_new_tokens=12, do_sample=False) as tracer:
    vlm.model.multi_modal_projector.output[:] = 0
    ids_blind = tracer.result.save()

prompt_length = inputs["input_ids"].shape[-1]
answer = lambda ids: vlm.tokenizer.decode(ids[0, prompt_length:], skip_special_tokens=True)
print("with image:", repr(answer(ids_clean)))
print("projector zeroed:", repr(answer(ids_blind)))
assert answer(ids_clean) != answer(ids_blind)
```

```
with image: 'The color of the image is not specified in the caption.'
projector zeroed: 'The color is this.'
```

### One condition, one forward pass

Two separate traces above, not two invokes. A processor's encoding cannot be
batched:

<!-- test: expect-error NotImplementedError -->
```python
with vlm.trace() as tracer:
    with tracer.invoke(inputs):
        pass
    with tracer.invoke(inputs):
        pass
# NotImplementedError: Can't batch these inputs; pass text or token ids.
```

nnsight batches text and token ids, not a payload with pixel values attached. The
sweep pattern from the `nnsight` skill (one invoke per variant, one forward pass)
does not apply here: a 12-layer ablation sweep on a VLM is 12 passes.

## Diffusion models

`DiffusionModel` wraps any `diffusers` pipeline. Its `nn.Module` components —
`unet` (or `transformer`), `vae`, `text_encoder` — are envoys. The scheduler is
not a module, so reach it on `sd.pipeline`.

<!-- test: gpu -->
```python
import logging
from nnsight import DiffusionModel

logging.getLogger("diffusers").setLevel(logging.ERROR)

sd = DiffusionModel("stabilityai/sd-turbo", torch_dtype=torch.float16,
                    safety_checker=None, dispatch=True, device_map="cuda")
sd.pipeline.set_progress_bar_config(disable=True)

PROMPT = "a red apple on a wooden table"
STEPS = 6
SETTINGS = dict(num_inference_steps=STEPS, guidance_scale=0.0, seed=17, output_type="np")

with sd.generate(PROMPT, **SETTINGS) as tracer:
    mid = sd.unet.mid_block.output.save()          # the bottleneck
    full = tracer.result.save()

print(mid.shape, full.images[0].shape)
```

Precision goes by `torch_dtype=`, diffusers' spelling. `dtype=`, which is what
`TransformersModel` takes, is accepted and ignored, leaving the pipeline in
float32.

### A traced run denoises once

`with sd.generate(...)` is a *traced* run, and a traced run defaults to
`num_inference_steps=1`. The same call means one denoising step inside a `with`
and the pipeline's default outside it. On SD 1.4 that is 1 denoiser forward
against 51, and a single step gives a noise blob rather than an image with
nothing printed to say so. Name the step count on every traced generation whose
image you intend to look at.

### Per-timestep interventions

The denoiser runs once per iteration of the pipeline's loop, and `tracer.iter`
repeats the body for each one. **Bound the loop by the number of denoiser calls,
which the scheduler decides** — it is not always `num_inference_steps`. SD 1.x
defaults to `PNDMScheduler`, which calls the denoiser `num_inference_steps + 1`
times (50 steps, 51 forwards); DDIM and Euler call it once per step.
`len(scheduler.timesteps)` after `set_timesteps` is the count:

<!-- test: gpu -->
```python
sd.pipeline.scheduler.set_timesteps(STEPS)
CALLS = len(sd.pipeline.scheduler.timesteps)
print(f"{STEPS} steps -> {CALLS} denoiser calls")

with sd.generate(PROMPT, **SETTINGS) as tracer:
    trajectory = nnsight.save([])
    for step in tracer.iter[:CALLS]:
        trajectory.append(sd.unet.mid_block.output.mean().cpu())
        if step > 3:
            sd.unet.mid_block.output[:] *= 1.1      # amplify late steps only
    amplified = tracer.result.save()

assert len(trajectory) == CALLS
```

`step` is a plain `int`, so `if step > 3:` really branches. Three ways to get
this wrong:

- A bound past the calls the run makes raises `OutOfOrderError`, naming the
  iteration asked for and the count reached.
- An open `tracer.iter[:]` warns instead, and every statement after the loop is
  dropped — `tracer.result.save()` among them, so the name is unbound after the
  block.
- An open `tracer.iter[:]` whose body touches no module at all never returns.

Order matters inside the loop as well: reading a later component and then writing
an earlier one raises `OutOfOrderError` rather than parking the write until the
next step. Read `sd.unet.inputs` before `sd.unet.conv_out.output`, not after.

### The diffusion lens

The diffusion lens (Toker et al., ACL 2024) probes the **text encoder**, not the
denoiser: read an intermediate encoder layer, push it straight into the encoder's
final layer norm so the remaining layers are skipped, and let the denoiser render
whatever that early embedding means. Sweeping the intercept point shows the
prompt assembling layer by layer.

<!-- test: gpu -->
```python
distances = {}
for layer in (0, 6, 12, 21):
    with sd.generate(PROMPT, **SETTINGS) as tracer:
        sd.text_encoder.final_layer_norm.input = sd.text_encoder.encoder.layers[layer].output
        lensed = tracer.result.save()
    distances[layer] = float(np.abs(lensed.images[0] - full.images[0]).mean())

print({k: round(v, 4) for k, v in distances.items()})
assert distances[0] > distances[21]
```

```
{0: 0.2296, 6: 0.1796, 12: 0.1513, 21: 0.0926}
```

Deeper layers land closer to the image the full encoder produces. The images
themselves are the result worth looking at: Toker et al. report that CLIP
encoders settle on a compound prompt's first noun early and add the second later,
while T5 encoders do the reverse. The `diffusion_lens.ipynb` tutorial reproduces
that on SD 1.5, Deep Floyd, SDXL and FLUX, with the module paths for each.

Module paths for the lens differ by encoder: CLIP layers are
`text_encoder.encoder.layers[i]` and return a tensor; T5 blocks are
`text_encoder.encoder.block[i]` and return a tuple, so take `.output[0]`. SDXL
and FLUX read a *penultimate* hidden state from their second encoder, so write
there rather than into the norm.

### Previewing a partly-denoised latent

A different question from the diffusion lens, and often confused with it: what
does the image look like partway through denoising? Decode the latent the
denoiser is working on, which is its **input** — `sd.unet.conv_out.output` is the
predicted noise, and decoding that shows something that gets *less* like the
final image as the run proceeds:

<!-- test: gpu -->
```python
with sd.generate(PROMPT, **SETTINGS) as tracer:
    latents = nnsight.save([])
    noise = nnsight.save([])
    for step in tracer.iter[:CALLS]:
        latents.append(sd.unet.inputs[0][0].clone())     # input before output
        noise.append(sd.unet.conv_out.output.clone())
    result = tracer.result.save()

def decode(latent):
    with torch.no_grad():
        image = sd.pipeline.vae.decode(latent / sd.vae.config.scaling_factor).sample
    return (image / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()

final = result.images[0]
correlation = lambda x: np.corrcoef(decode(x).ravel(), final.ravel())[0, 1]
for step in range(CALLS):
    print(f"step {step}: latent {correlation(latents[step]):.3f}   "
          f"conv_out {correlation(noise[step]):.3f}")

assert correlation(latents[-1]) > correlation(latents[0])
assert correlation(latents[-1]) > correlation(noise[-1])
```

```
step 0: latent 0.362   conv_out 0.333
step 1: latent 0.400   conv_out 0.329
step 2: latent 0.467   conv_out 0.317
step 3: latent 0.565   conv_out 0.296
step 4: latent 0.691   conv_out 0.261
step 5: latent 0.838   conv_out 0.219
```

`.clone()` matters: the pipeline reuses the latent buffer, so references saved
across steps all end up holding the last step's data.

Guidance doubles the denoiser's batch. `guidance_scale=0.0` above keeps it at one
row; above 1 the rows are the unconditional half followed by the conditional
half, so row 0 decodes to the empty prompt's image, not the prompt's.

### Cross-attention: where the prompt acts

Text conditioning enters through cross-attention in the denoiser blocks — `attn1`
is self-attention, `attn2` is cross-attention. Those are the modules to ablate
when asking which words drove which part of the image:

<!-- test: gpu -->
```python
attn2 = sorted((name for name, _ in sd.unet.named_modules() if name.endswith(".attn2")))
print(len(attn2), "cross-attention layers;", attn2[0])

target = sd.unet.get(attn2[0].removeprefix("model.unet."))
with sd.generate(PROMPT, **SETTINGS) as tracer:
    for _ in tracer.iter[:CALLS]:
        target.to_out[0].input[:] = 0
    ablated = tracer.result.save()

change = float(np.abs(ablated.images[0] - full.images[0]).mean())
print("mean pixel change:", round(change, 4))
assert change > 0.01
```

Cut at `to_out[0].input` — the post-attention, pre-projection activation. That
removes the layer's contribution while letting attention itself run. The
`cross-attention-ablation.ipynb` tutorial sweeps all 16 layers of SD 1.4 this
way; one of them turns out to carry the prompt's binding to a specific painting.

Path names differ between UNet and DiT pipelines, so list them rather than
hard-coding.

## Other tasks

Any task the `transformers` pipeline API supports works through
`TransformersModel` with the right `task=`. Which preprocessor is populated
depends on the task, and the others are `None`:

| task | populated |
|---|---|
| `image-text-to-text` | `processor`, `tokenizer` (from the processor) |
| `image-classification`, `image-feature-extraction` | `image_processor` |
| `automatic-speech-recognition` | `feature_extractor`, `tokenizer` |
| `audio-classification` | `feature_extractor` |
| `fill-mask`, `text-classification`, `token-classification` | `tokenizer` |

`model.tokenizer` is populated on a VLM. It is `None` for the tasks with no text
side, which is where a `tokenizer.decode` call falls over.

```python
vision = TransformersModel("facebook/dinov3-vits16-pretrain-lvd1689m",
                           task="image-feature-extraction", dispatch=True)

with vision.trace(image):
    patches = vision.model.layer[6].output.save()

print(patches.shape, "| tokenizer:", vision.tokenizer)
assert vision.tokenizer is None
```

```
torch.Size([1, 201, 384]) | tokenizer: None
```

The layer path is `vision.model.layer[i]`: the root envoy's child is the
`DINOv3ViTModel`, whose blocks are `layer`. `vision.encoder` does not exist and
raises `AttributeError: 'TransformersModel' object (nor its module) has attribute
'encoder'`.

### Chunked tasks

Some tasks split one input into several encodings and forward each separately:
token windows past the model's length limit, one entailment pair per candidate
label, a long recording's windows. Those become **rows of the trace's single
forward**, in the order the task yields them, so a read inside the block sees one
row per chunk.

```python
ner = TransformersModel("hf-internal-testing/tiny-random-BertForTokenClassification",
                        task="token-classification", dispatch=True)

with ner.trace("John lives in Paris"):
    short = ner.output.logits.save()

with ner.trace(" ".join(["John lives in Paris"] * 200), stride=16):
    windows = ner.output.logits.save()

print(short.shape, windows.shape)
assert short.shape[0] == 1 and windows.shape[0] == 7
```

```
torch.Size([1, 18, 2]) torch.Size([7, 512, 2])
```

Four things to know about these tasks:

- A chunked invoke is the whole batch. The row count belongs to the task, so
  putting one next to another invoke is refused with `task=... splits this invoke
  into N forward rows`.
- `document-question-answering` and `zero-shot-object-detection` take the task's
  own input dict (`{"image": ..., "question": ...}`), which goes through the
  task's preprocessing rather than being read as a model encoding.
- An encoder-decoder task needs the decoder's side of the forward:
  `whisper` traces with `decoder_input_ids=` and gives one row per audio chunk
  (70 s at `chunk_length_s=30` is 3 rows of `[1500, 384]` at the encoder).
- `mask-generation` is refused. Its preprocessing runs the model to embed the
  image before yielding one input per batch of candidate points, so there is no
  single forward to assemble. Use `model.pipe(image)`, or build an encoding with
  `model.image_processor` and pass the points as `input_points=`.

```python
zero_shot = TransformersModel("hf-internal-testing/tiny-random-DistilBertForSequenceClassification",
                              task="zero-shot-classification", dispatch=True)

with zero_shot.trace("one day I will see the world",
                     candidate_labels=["travel", "cooking", "dancing"]):
    per_label = zero_shot.output.logits.save()

print(per_label.shape)         # one row per candidate label
assert per_label.shape[0] == 3
```

## What to watch for

**Sequence length is dominated by non-text tokens.** Compute spans from the input
ids rather than reusing indices from a text-only experiment.

**Module nesting is deeper and less stable.** VLM layouts differ across families
and move between `transformers` releases. Inspect first.

**`scan` fails on some VLM families** with
`GuardOnDataDependentSymNode`: preprocessing takes a size that depends on tensor
*data*, which fake tensors cannot supply. It raises on Idefics3 and Qwen3-VL and
succeeds on both Llava variants. Run a real trace when `scan` refuses.

**Diffusion has no single "output token".** Metrics are image-space or
latent-space, so decide what you are measuring before intervening; the
`interp-experiment-design` skill covers choosing one.

## Related skills

- `nnsight` — the API, module paths, iteration
- `logit-lens` — the text-model technique the diffusion lens is named after
- `activation-patching`, `ablation` — apply directly to image-token spans
- `attention-analysis` — cross-attention in diffusion, vision attention in VLMs
