---
name: diffusion-and-multimodal
description: Trace and intervene on models that are not text-only — vision-language models (image+text via TransformersModel with a processor), diffusion pipelines via DiffusionModel, and audio/vision encoders. Use to find where image information enters the language stream, to patch or ablate image tokens versus text tokens, to run the diffusion lens over a text encoder or decode a partly-denoised latent, or to work with any HuggingFace task beyond text generation, including the chunked ones (ASR, NER, zero-shot).
---

# Diffusion and Multimodal Models

`trace`, `.output`, `save` and gradients work on these models exactly as they do
on a language model. What changes is **how you build the input**, **where the
module tree puts things**, and which inputs batch.

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

print(hidden.shape, logits.shape)    # [1, 743, 1024] [1, 743, 152000]
assert hidden.shape[1] == inputs["input_ids"].shape[1]
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
| `hf-internal-testing/tiny-random-Idefics3ForConditionalGeneration` | `model.vision_model` | *(none exposed)* | *(no `language_model`)* |
| `Qwen/Qwen3-VL-4B-Instruct` | `model.visual` | `model.visual.merger` **and** `model.visual.deepstack_merger_list` | `model.language_model` |

Check before you index. `scripts/inspect_model.py <repo> --grep vision` from the
`nnsight` skill prints the paths.

**Qwen3-VL has more than one adapter.** Image features reach the language stream
twice: through `visual.merger`, and again through `visual.deepstack_merger_list`,
whose outputs are injected into the residual after language layers 0-2 (visible
at `language_model.source.self__deepstack_process_0`). Zeroing `visual.merger`
alone — or the image span at `layers[0].input` — leaves the answer unchanged and
makes a correct intervention look broken. Zero the merger *and* every deepstack
merger.

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

One contiguous span is a fact about a single-image prompt. A prompt with text between
images gives one span per image, so treat the positions as a mask, not as a slice
`[start:end]`. (The `nnterp` skill serves this mask and the image features as named
values on every VLM family it wraps.)

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

On llava-interleave-qwen-0.5b, "The" stays on top while its probability moves
from 0.61 to 0.81 and "This" drops from 0.32 to 0.03.

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

### What batches, and what a batched VLM trace still gets wrong

Two separate traces above, not two invokes — a **processor encoding** cannot be
batched, and neither can the `prompt, images=` form:

<!-- test: expect-error NotImplementedError -->
```python
with vlm.trace() as tracer:
    with tracer.invoke(inputs):
        pass
    with tracer.invoke(inputs):
        pass
# NotImplementedError: Can't batch these inputs; pass text or token ids.
```

**Chat messages with the image embedded do batch.** nnsight sends those through
the task pipeline's own preprocessing, which collates `pixel_values` along dim 0,
so the sweep pattern from the `nnsight` skill applies after all: one invoke per
variant, one forward pass.

```python
def chat(img, question="What color is this?"):
    return [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": question}]}]

with vlm.trace() as tracer:
    swept = nnsight.save([])
    for layer in (4, 8, 12):
        with tracer.invoke(chat(image)):
            vlm.model.language_model.layers[layer].output[:] = 0
            swept.append(vlm.output.logits[0, -1].argmax())

print("ablating one layer per invoke, one pass:", [int(t) for t in swept])
assert len(swept) == 3
```

**Only the language side is reliably scoped per invoke.** nnsight decides an
invoke's rows from a value's leading dimension — the batch size, or a whole
multiple of it (the `nnsight` skill's batching reference, "Which values are scoped
to your rows"). A vision-tower or projector activation is batched over **images**, which lines up
with the prompt rows only when every invoke carries the same number of images. Put
a two-image invoke beside a one-image invoke and
`vlm.model.multi_modal_projector.output` is `(3, 729, 1024)` in *both* of them:
zeroing it in the second moves the first invoke's untouched logits by 8.4
(llava-interleave-qwen-0.5b, bf16). The write warns when this happens; the read
does not.

So batch **language-side** sweeps, and give every vision-side intervention its
own trace unless the image counts match. The obvious "does batched match single?"
control does not catch this: the language side agrees exactly while the vision
side leaks.

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
# diffusers places the pipeline with device_map=; device= is ignored here
sd.pipeline.set_progress_bar_config(disable=True)

PROMPT = "a red apple on a wooden table"
STEPS = 6
SETTINGS = dict(num_inference_steps=STEPS, guidance_scale=0.0, seed=17, output_type="np")

with sd.generate(PROMPT, **SETTINGS) as tracer:
    mid = sd.unet.mid_block.output.save()          # the bottleneck
    full = tracer.result.save()

print(mid.shape, full.images[0].shape)
```

Two silent defaults. Precision goes by `torch_dtype=`, diffusers' spelling —
`dtype=` is accepted and ignored, leaving the pipeline in float32. And **a traced
run denoises once**: inside a `with`, `sd.generate(...)` defaults to
`num_inference_steps=1` (outside it, the pipeline's default — 51 forwards on SD
1.4), which gives a noise blob with nothing printed to say so. Name the step count
on every traced generation whose image you intend to look at.

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

`step` is a plain `int`, so `if step > 3:` really branches. Two ways to get
this wrong:

- A loop that asks for a call the run does not make — a bound past the calls the
  run makes, or an open `tracer.iter[:]` reaching its natural end — is cut short
  with a warning, and every statement after the loop is dropped:
  `tracer.result.save()` among them, so the name is unbound after the block.
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
assert distances[0] > distances[21]      # {0: 0.23, 6: 0.18, 12: 0.15, 21: 0.09}
```

Deeper layers land closer to the image the full encoder produces. The images
themselves are the result worth looking at: Toker et al. report that CLIP
encoders settle on a compound prompt's first noun early and add the second later,
while T5 encoders do the reverse.

Module paths for the lens differ by encoder: CLIP layers are
`text_encoder.encoder.layers[i]` and return a tensor; T5 blocks are
`text_encoder.encoder.block[i]` and return a tuple, so take `.output[0]`.

**Find the read point in the pipeline's own `encode_prompt`, do not assume one.**
SDXL and FLUX.1 take a *penultimate* hidden state from their second encoder, so
`[-2]` is where to write; FLUX.2 uses a Qwen3 encoder and reads hidden states 9,
18 and 27, which makes a write to `[-2]` a silent no-op that looks like a lens
that does nothing. `inspect.getsource(sd.pipeline.encode_prompt)` settles it in
one line.

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

Over six steps the latent's correlation with the final image climbs from 0.36 to
0.84 while `conv_out`'s falls from 0.33 to 0.22.

`.clone()` matters: the pipeline reuses the latent buffer, so references saved
across steps all end up holding the last step's data.

**Guidance doubles the denoiser's batch, and changes how writes land.** Every
example here runs at `guidance_scale=0.0`, which is *not* the diffusers default
(7.5). Above 1 the rows are the unconditional half followed by the conditional
half, so row 0 decodes to the empty prompt's image. One prompt then owns two
non-adjacent row ranges, so the value your block sees is stitched from both — a
copy, not a view. Reads are unaffected, but `output[:] = x` can write into that
copy and never reach the model, with no error; `output = x` always lands. Check
that the image actually moved.

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
removes the layer's contribution while letting attention itself run. Path names
differ between UNet and DiT pipelines, so list them rather than hard-coding.

## Other tasks and chunked tasks

Any `transformers` pipeline task works through `TransformersModel` with the right
`task=` — image classification and feature extraction, ASR, NER, zero-shot. Which
preprocessor is populated depends on the task (`model.tokenizer` is `None` when
there is no text side), and some tasks — long-text NER, zero-shot classification,
long audio — split one input into several forward rows, one per chunk, which makes
a chunked invoke the whole batch. Whisper also runs its encoder twice.
[references/other-tasks.md](references/other-tasks.md) has the preprocessor table,
a vision-encoder trace, the chunked tasks and the two tasks a trace refuses.

## What to watch for

**`scan` fails on some VLM families** with
`GuardOnDataDependentSymNode`: preprocessing takes a size that depends on tensor
*data*, which fake tensors cannot supply. It raises on Idefics3 and Qwen3-VL and
succeeds on both Llava variants. Run a real trace when `scan` refuses.

## Related skills

- `nnsight` — the API, module paths, iteration
- `logit-lens` — the text-model technique the diffusion lens is named after
- `activation-patching`, `ablation` — apply directly to image-token spans
- `attention-analysis` — cross-attention in diffusion, vision attention in VLMs
