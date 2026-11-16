# Vision-language models and diffusion pipelines

`trace`, `.output`, `save` and gradients work on these models as in
[SKILL.md](../SKILL.md). What changes is how the input is built, where the
module tree puts things, and what batches. Image inputs need Pillow installed,
and diffusion needs `diffusers`. Whisper, vision encoders and classifiers are in
[other tasks](diffusion-and-multimodal-other-tasks.md).

## Vision-language models

On a family the `nnterp` skill supports, prefer it: it serves the image-token
mask, the image features and the language layers under the same names on every
vision-language family. What follows is the raw path, for a model nnterp does not
know or a value it does not name.

Load with `task="image-text-to-text"` and pass chat messages with the image
inside them. The task's processor builds the model's input:

```python
import numpy as np
import torch
import nnsight
from nnsight import TransformersModel
from PIL import Image

vlm = TransformersModel("llava-hf/llava-interleave-qwen-0.5b-hf",
                        task="image-text-to-text", dispatch=True)
image = Image.new("RGB", (224, 224), (120, 30, 200))          # a solid purple square

def chat(*images, question="What color is this?"):
    return [{"role": "user", "content": [*({"type": "image", "image": i} for i in images),
                                         {"type": "text", "text": question}]}]

with vlm.trace(chat(image)):
    input_ids = vlm.inputs[1]["input_ids"][0].save()           # the ids the processor built
    hidden = vlm.model.language_model.layers[10].output.save()
    clean = vlm.output.logits[0, -1].detach().save()

image_positions = (input_ids == vlm.config.image_token_id).nonzero().flatten()
print(tuple(hidden.shape), "|", len(image_positions), "image positions")
assert hidden.shape[1] == len(input_ids) == 743 and len(image_positions) == 729
```

```
(1, 743, 1024) | 729 image positions
```

- **Most positions are image patches**, so a position index from a text-only
  experiment points at the wrong thing and `-1` is the only text position you
  can index without counting. Index the image with `image_positions`, built from
  the ids as above: the count belongs to the model (the same image is 64
  positions on Qwen3-VL-4B) and every further image adds as many again.
- `vlm.processor(images=image, text=text, return_tensors="pt")`, with `text`
  from `vlm.processor.apply_chat_template(...)`, builds the same encoding outside
  a trace, and `vlm.trace(encoding)` or `vlm.trace(text, images=image)` runs it
  with identical logits. The tokenizer alone leaves the template's single
  `<image>` unexpanded, and its ids with the processor's pixels raise `ValueError:
  Image features and image tokens do not match`.
- `vlm.scan(...)` raises `GuardOnDataDependentSymNode` on some families (Idefics3,
  Qwen3-VL). Run a real trace there.

### Where the stacks live

A VLM has a vision stack, an adapter and a language stack nested in the model,
and the names are not shared across families:

| family | vision stack | adapter | language stack |
|---|---|---|---|
| Llava | `model.vision_tower` | `model.multi_modal_projector` | `model.language_model` |
| Idefics3 | `model.vision_model` | `model.connector` | `model.text_model` |
| Qwen3-VL | `model.visual` | `model.visual.merger` **and** `model.visual.deepstack_merger_list` | `model.language_model` |

Check before indexing: `python scripts/inspect_model.py <repo_id> --task
image-text-to-text --depth 2` prints the tree. The adapter's output is what the
image becomes inside the language model, `(images, 729, 1024)` here.

**Qwen3-VL has more than one adapter.** Image features reach the language stream
through `visual.merger`, and again through the three modules of
`visual.deepstack_merger_list`, whose outputs are added to the image positions
after language layers 0 to 2. On Qwen3-VL-4B, zeroing `visual.merger` alone, or
the image span at `layers[0].input`, leaves the answer below at `Purple`, which
makes a correct intervention look broken. Zero every deepstack merger and then
the merger, in that order, because the deepstack mergers run first:

```python
qwen = TransformersModel("Qwen/Qwen3-VL-4B-Instruct", task="image-text-to-text", dispatch=True)
one_word = chat(image, question="What color is this? Answer in one word.")

with qwen.pipe(one_word, max_new_tokens=8, do_sample=False) as tracer:
    for merger in qwen.model.visual.deepstack_merger_list:
        merger.output[:] = 0
    qwen.model.visual.merger.output[:] = 0
    records = tracer.result.save()

print(records[0]["generated_text"][-1])
```

```
{'role': 'assistant', 'content': 'white'}
```

That block needs a GPU for a 4B model, and `torchvision` for Qwen3-VL's
processor.

### A metric that can move

Zeroing the image positions mid-stack leaves the top-1 token where it was, so an
`argmax` comparison shows nothing. Read the distribution, or generate:

```python
def top3(row):
    values, ids = row.float().softmax(-1).topk(3)
    return [(vlm.tokenizer.decode(i), round(float(v), 2)) for i, v in zip(ids, values)]

with vlm.trace(chat(image)):
    vlm.model.language_model.layers[10].output[:, image_positions, :] = 0
    ablated = vlm.output.logits[0, -1].detach().save()

with vlm.generate(chat(image), max_new_tokens=12, do_sample=False) as tracer:
    seeing = tracer.result.save()

with vlm.generate(chat(image), max_new_tokens=12, do_sample=False) as tracer:
    vlm.model.multi_modal_projector.output[:] = 0              # cut every image feature
    blind = tracer.result.save()

answer = lambda ids: vlm.tokenizer.decode(ids[0, len(input_ids):], skip_special_tokens=True)
print("clean  ", top3(clean))
print("ablated", top3(ablated))
print(repr(answer(seeing)), "->", repr(answer(blind)))

assert top3(clean)[0][0] == top3(ablated)[0][0] == "The"       # same top-1 either way
assert top3(ablated)[0][1] - top3(clean)[0][1] > 0.1
assert answer(seeing) != answer(blind)
```

```
clean   [('The', 0.61), ('This', 0.32), ('It', 0.06)]
ablated [('The', 0.81), ('A', 0.03), ('This', 0.03)]
'The color of the image is not specified in the caption.' -> 'The color is this.'
```

### What batches, and the leak on the vision side

Chat messages batch across invokes, so a sweep is one forward pass. A processor
encoding and the `text, images=` form do not: a second invoke raises
`NotImplementedError: Can't batch these inputs; pass text or token ids.`

```python
with vlm.trace() as tracer:
    swept = nnsight.save({})
    for layer in (4, 8, 12):
        with tracer.invoke(chat(image)):
            vlm.model.language_model.layers[layer].output[:] = 0
            swept[layer] = vlm.output.logits[0, -1].argmax()

assert len(swept) == 3
```

**Only the language side is scoped to an invoke's rows.** A vision-tower or
adapter activation is batched over *images*, which lines up with the prompt rows
only when every invoke carries the same number of images
([batching](batching.md) has the rule). Put a two-image invoke beside a
one-image invoke and each sees all three images at the adapter, so an edit there
reaches the other invoke with no warning:

```python
other = Image.new("RGB", (224, 224), (20, 200, 40))

def pair(edit):
    with vlm.trace() as tracer:
        with tracer.invoke(chat(image)):
            untouched = vlm.output.logits[0, -1].float().detach().save()
        with tracer.invoke(chat(image, other)):
            shape = nnsight.save(tuple(vlm.model.multi_modal_projector.output.shape))
            edit()
    return untouched, shape

def zero_adapter():
    vlm.model.multi_modal_projector.output[:] = 0

def zero_language_layer():
    vlm.model.language_model.layers[4].output[:] = 0

reference, shape = pair(lambda: None)
vision_edit, _ = pair(zero_adapter)
language_edit, _ = pair(zero_language_layer)

assert shape == (3, 729, 1024)                                  # all three images, in a two-image invoke
assert (language_edit - reference).abs().max() == 0             # scoped
assert (vision_edit - reference).abs().max() > 1                # leaked into the other invoke
```

Batch language-side sweeps, and give a vision-side intervention its own trace
unless the image counts match. A "does batched match unbatched?" check on the
language side does not catch this.

## Diffusion pipelines

`DiffusionModel` wraps a `diffusers` pipeline. Its `nn.Module` components
(`unet` or `transformer`, `vae`, `text_encoder`) are envoys. The scheduler is not
a module: reach it on `sd.pipeline`.

```python
import logging
from nnsight import DiffusionModel

logging.getLogger("diffusers").setLevel(logging.ERROR)

sd = DiffusionModel("stabilityai/sd-turbo", torch_dtype=torch.float16,
                    safety_checker=None, dispatch=True, device_map="cuda")
sd.pipeline.set_progress_bar_config(disable=True)
assert sd.unet.dtype == torch.float16 and sd.unet.device.type == "cuda"

PROMPT = "a red apple on a wooden table"
STEPS = 6
SETTINGS = dict(num_inference_steps=STEPS, guidance_scale=0.0, seed=17, output_type="np")

with sd.generate(PROMPT, **SETTINGS) as tracer:
    mid = sd.unet.mid_block.output.save()          # the bottleneck, first denoising step
    full = tracer.result.save()                    # the pipeline's output object

print(tuple(mid.shape), full.images[0].shape)
```

```
(1, 1280, 8, 8) (512, 512, 3)
```

- **Load keywords go to diffusers untouched**, and it ignores the ones it does
  not know. Placement is `device_map="cuda"`: `device=` is ignored and leaves the
  pipeline on the CPU. Precision is `torch_dtype=`, and a spelling the installed
  diffusers does not take leaves it in float32. Assert both, as above.
- **A traced run denoises once.** Inside a `with`, `sd.generate(...)` and
  `sd.trace(...)` default to `num_inference_steps=1`, which gives a noise blob
  with nothing printed to say so. Called without `with`, `sd.generate(...)` uses
  the pipeline's own default. Name the step count on every traced run whose
  image you look at.
- `seed=` makes a run reproducible. Without it two runs differ.
- Every Stable Diffusion repo declares the text-to-image pipeline, which ignores
  `image=` without an error. For image-to-image or inpainting from the same
  weights, name the class: `DiffusionModel(repo, automodel="AutoPipelineForImage2Image", ...)`.
- To run one component's forward alone, trace its envoy on a dispatched model,
  with inputs on its device and dtype:
  `with sd.unet.trace(sample, timestep, encoder_hidden_states=states):`.

### Per-step interventions

The denoiser runs once per iteration of the pipeline's loop, and `tracer.iter`
repeats the body for each. **Bound the loop by the number of denoiser calls,
which the scheduler decides.** SD 1.x's default `PNDMScheduler` makes 51 calls
for 50 steps, where DDIM and Euler make 50. `len(scheduler.timesteps)` after
`set_timesteps` is the count:

```python
sd.pipeline.scheduler.set_timesteps(STEPS)
CALLS = len(sd.pipeline.scheduler.timesteps)

with sd.generate(PROMPT, **SETTINGS) as tracer:
    trajectory = nnsight.save([])
    for step in tracer.iter[:CALLS]:
        trajectory.append(sd.unet.mid_block.output.mean().cpu())
        if step > 3:
            sd.unet.mid_block.output[:] *= 1.1      # amplify late steps only
    amplified = tracer.result.save()

assert len(trajectory) == CALLS == 6
```

A bound past the calls the run makes cuts the loop short with a warning and
drops the statements after it, `tracer.result.save()` included.

### The diffusion lens

The diffusion lens (Toker et al., 2024) probes the **text encoder**: feed an
intermediate encoder layer to the encoder's final layer norm, so the remaining
layers are skipped, and let the denoiser render what that early embedding means.

```python
distances = {}
for layer in (0, 6, 12, 21):
    with sd.generate(PROMPT, **SETTINGS) as tracer:
        sd.text_encoder.final_layer_norm.input = sd.text_encoder.encoder.layers[layer].output
        lensed = tracer.result.save()
    distances[layer] = float(np.abs(lensed.images[0] - full.images[0]).mean())

print({k: round(v, 2) for k, v in distances.items()})
assert distances[0] > distances[21]
```

```
{0: 0.23, 6: 0.18, 12: 0.15, 21: 0.09}
```

Deeper layers land closer to the image the full encoder produces. CLIP layers
are `text_encoder.encoder.layers[i]` and return a tensor. T5 blocks are
`text_encoder.encoder.block[i]` and return a tuple: take `.output[0]`.

**Find the read point in the pipeline's own `encode_prompt`.** Not every
pipeline uses the final layer norm's output: SDXL takes hidden state `[-2]` of
each encoder, and FLUX.2 klein stacks hidden states 9, 18 and 27 of a Qwen3
encoder, so the write above is a silent no-op there and looks like a lens that
does nothing. `inspect.getsource(sd.pipeline.encode_prompt)` shows it.

### Previewing a partly-denoised latent

Decode the latent the denoiser is working on, which is its **input**.
`sd.unet.conv_out.output` is the predicted noise, and decoding it gives
something that gets *less* like the final image as the run proceeds:

```python
with sd.generate(PROMPT, **SETTINGS) as tracer:
    latents = nnsight.save([])
    noise = nnsight.save([])
    for step in tracer.iter[:CALLS]:
        latents.append(sd.unet.inputs[0][0].clone())     # the input comes before the output
        noise.append(sd.unet.conv_out.output.clone())
    result = tracer.result.save()

def decode(latent):
    with torch.no_grad():
        decoded = sd.pipeline.vae.decode(latent / sd.vae.config.scaling_factor).sample
    return (decoded / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).float().cpu().numpy()

final = result.images[0]
correlation = lambda x: np.corrcoef(decode(x).ravel(), final.ravel())[0, 1]
print("latent  ", [round(float(correlation(x)), 2) for x in latents])
print("conv_out", [round(float(correlation(x)), 2) for x in noise])

assert correlation(latents[-1]) > correlation(latents[0])
assert correlation(latents[-1]) > correlation(noise[-1])
```

```
latent   [0.36, 0.4, 0.47, 0.57, 0.69, 0.84]
conv_out [0.33, 0.33, 0.32, 0.3, 0.26, 0.22]
```

`.clone()` matters: the pipeline reuses the latent buffer, so references saved
across steps all end up holding the last step's data.

### Guidance and several prompts

Every example above runs at `guidance_scale=0.0`, which is not the default (7.5
on Stable Diffusion). Above 1 the denoiser's batch doubles: the unconditional rows, then the
conditional rows, so row 0 belongs to the empty prompt. Several prompts batch as
invokes, each seeing its own rows of both halves, and the pipeline's output is
read in an empty invoke:

```python
def two_prompts(edit):
    with sd.generate(num_inference_steps=2, guidance_scale=2.0, seed=5, output_type="np") as tracer:
        with tracer.invoke(PROMPT):
            shape = nnsight.save(tuple(sd.unet.mid_block.output.shape))
            edit()
        with tracer.invoke("a blue car"):
            pass
        with tracer.invoke():
            result = tracer.result.save()
    return shape[0], result.images

def in_place():
    sd.unet.mid_block.output[:] = 0

def assigned():
    sd.unet.mid_block.output = torch.zeros_like(sd.unet.mid_block.output)

rows, reference = two_prompts(lambda: None)
_, lost = two_prompts(in_place)
_, landed = two_prompts(assigned)

assert rows == 2 and reference.shape[0] == 2                     # uncond and cond rows; one image per prompt
assert np.abs(lost - reference).max() == 0                       # the in-place write never reached the model
assert np.abs(landed[0] - reference[0]).mean() > 0.01            # the assignment did
assert np.abs(landed[1] - reference[1]).max() == 0               # and stayed in its invoke
```

**With several invokes under guidance, assign: `output = x`.** An invoke's rows
are then two separate ranges, the value the block sees is a copy stitched from
both, and `output[:] = x` writes into that copy with no error. With one prompt
an in-place write lands as usual. Reading `tracer.result` inside a prompt's own
invoke raises a `TypeError`.

### Cross-attention: where the prompt acts

Text conditioning enters the denoiser through cross-attention: in each block
`attn1` is self-attention and `attn2` is cross-attention. Path names differ
between UNet and DiT pipelines, so list them:

```python
cross = sorted((path, envoy) for path, envoy in sd.unet.named_modules() if path.endswith(".attn2"))
print(len(cross), "cross-attention layers;", cross[0][0])

target = cross[0][1]
with sd.generate(PROMPT, **SETTINGS) as tracer:
    for step in tracer.iter[:CALLS]:
        target.to_out[0].input[:] = 0               # after attention, before its output projection
    cut = tracer.result.save()

change = float(np.abs(cut.images[0] - full.images[0]).mean())
assert change > 0.01
```

```
16 cross-attention layers; model.unet.down_blocks.0.attentions.0.transformer_blocks.0.attn2
```

Cutting at `to_out[0].input` removes the layer's contribution while the
attention itself still runs.
