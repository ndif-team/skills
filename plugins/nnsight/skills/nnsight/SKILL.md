---
name: nnsight
description: "Read and edit the internals of any PyTorch model with nnsight: model.trace, .save(), interventions, batched invokes, gradients, .source, generation, and remote execution on NDIF (remote=True) for models too large to run locally. Load whenever a request names nnsight, and before writing, fixing or explaining any nnsight code, however short: idioms from before 0.8 (.value, LanguageModel, proxies) are widespread and silently wrong. Use for an nnsight error such as OutOfOrderError, a hang or a wrong result, porting old code, a model family nnterp does not support, arbitrary module or .source access, diffusion and vision-language models, vLLM, tensor parallelism and 4/8-bit loading, and wherever PyTorch forward hooks would otherwise be written. Bundles a guide per technique: logit lens, patching, causal tracing, ablation, attention analysis, circuit discovery, probing, SAEs, steering, editing and LoRA. When no library is named, a standard experiment on a supported transformer goes to nnspec first, then nnterp."
---

# nnsight

nnsight reads, replaces and backpropagates through any intermediate value of a
PyTorch model's forward pass, locally or on NDIF. This file covers loading,
reading and editing activations, batching, generation and gradients. For
anything else, open the [guide](#guides) that names the task before writing code.

## Which library

nnsight is the base of three libraries. **A request that names a library gets that library**, even when a higher row would fit, and a
request for a Python script is not a request for an nnspec document. Otherwise take the highest row
that fits, and take a row only when its skill is available in this session:

| Use | When |
|---|---|
| `nnspec` skill | A standard experiment on a transformer language model (lens, patching, ablation, attention, steering, probes, SAEs, DAS) that fits a JSON document. No intervention code to write. |
| `nnterp` skill | The experiment needs Python that a document cannot express, on a family nnterp knows. One module vocabulary on every family. |
| nnsight (this skill) | nnterp does not know the family, or the work needs access below nnterp's standard values: arbitrary modules, `.source`, your own or non-transformer models, diffusion, vLLM, tensor parallelism. Also existing nnsight code, and any error an nnsight trace raises. |

## Load, read, edit

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt):
    clean = model.output.logits.save()                     # read the model's output

with model.trace(prompt):
    resid = model.transformer.h[5].output.save()           # read an activation
    model.transformer.h[8].output[:, -1, :] *= 2           # edit one
    edited = model.output.logits.save()                    # see what the edit changed

assert tuple(resid.shape) == (1, 10, 768)                  # (batch, tokens, hidden)
assert model.tokenizer.decode(clean[0, -1].argmax()) == " Paris"
assert model.tokenizer.decode(edited[0, -1].argmax()) == " London"
```

Requires nnsight 0.8 with transformers 5 or later (`pip install nnsight`).

- **`task=`** names the transformers pipeline task. Left out, it is looked up on
  the Hub, and offline (`HF_HUB_OFFLINE=1`) the load fails with `RuntimeError: You
  cannot infer task automatically within pipeline when using offline mode`.
- **`dispatch=True`** loads the weights now. Without it they load at the first
  run.
- **Device**: the first GPU, or the CPU when there is none. Choose with
  `device="cpu"` or `device="cuda:1"`. `device_map="cpu"` still lands on the GPU.
- **dtype**: the one the checkpoint declares, which is bfloat16 for most recent
  models, on CPU too. Pass `dtype=torch.float32` (`torch_dtype=` is deprecated)
  when a check compares numbers: in bfloat16 two near-tied logits swap between
  batch shapes.
- A model you built yourself is `NNsight(module)`, with the same API.

Every module has `.output` (what its forward returns), `.input` (its first
argument) and `.inputs` (`(args, kwargs)`), each readable and assignable inside a
trace. `model.output` and `model.inputs` are the whole model's.

## How a trace runs

The body of `with model.trace(...)` does not run where it is written. nnsight
compiles it and runs it as a worker that takes turns with the forward pass:
reading `module.output` parks the worker until the model produces that value,
which arrives as a real `torch.Tensor`. So `if`, `for`, `print` and `.item()`
work inside the body, a value leaves it only through a save, modules must be
touched in the order the model runs them, and one trace is one forward pass.

## Traps

**1. Save, and bind the save.** A name assigned in the body exists afterwards
only if its value was saved: `x = value.save()` or `x = nnsight.save(value)`.
Unsaved, it is a `NameError` after the block (`UnboundLocalError` in a function),
or, when the name already existed (a notebook rerun, a loop), its **old value**
with no error. A bare `value.save()` line returns nothing, and so does
`float(x.save())`: bind the save and convert after the block. To collect, create
the container in the block and save the container:

```python
with model.trace(prompt):
    per_layer = nnsight.save([])                  # save the list
    for block in model.transformer.h:
        per_layer.append(block.output[0, -1])     # append raw values

assert len(per_layer) == 12
```

"My list is empty (or undefined) after the trace" is this rule: a list created
inside the block and not saved is gone, and one created outside and appended to
fills locally but comes back empty from a remote run.

A saved activation carries the forward graph. `float()` or `.item()` on it then
warns `Converting a tensor with requires_grad=True to a scalar may lead to
unexpected behavior`, and holding it keeps the graph in memory: save
`value.detach()`, or wrap pure collection in `torch.no_grad()`. It also aliases
the live tensor, so `.clone()` it if the block writes to that location later.

**2. Forward order.** Within one body, touch modules in the order the model runs
them: layer 2 before layer 8, a block's submodules (`ln_1`, `attn`, `ln_2`, `mlp`
on GPT-2) before the block's own `.output`, `model.inputs` first, `model.output`
and `tracer.result` last. Out of order raises `OutOfOrderError: '...h.2.output.i0'
was requested but the model already ran past it`. Writes follow the same rule
silently: an in-place write to a tensor made after a later module was read
changes nothing, because the model already consumed it. Reorder, give each
ordering its own `tracer.invoke`, or collect with `tracer.cache()`. For a module
the forward calls several times (a looped or weight-tied net), `.output` is its
**first** call however often you read it; step through the calls with
`tracer.iter` ([access-and-modify](references/access-and-modify.md)).

**3. Do not guess paths or output types.** GPT-2 blocks are
`model.transformer.h[i]`, Llama, Qwen and Mistral `model.model.layers[i]`, a
multimodal Gemma-3 `model.model.language_model.layers[i]`. A block's `.output` is
a plain tensor on these and an attention module's is a tuple, so `.output[0]`
copied from an older example silently selects batch row 0 of a block. Run
`python scripts/inspect_model.py <repo_id> --prompt "..."`: without downloading
weights it prints the layer path, each block child in execution order, and
whether each output is a tensor or a tuple.

**4. Write through the model.** `acts = block.output; acts = acts * 0` rebinds a
local and changes nothing. Write in place (`block.output[:] = 0`) or assign the
property (`block.output = block.output * 0`).

**5. Target token ids.** `tokenizer.encode(" Paris")[0]` is the BOS token on
tokenizers that prepend one (Llama-3, Gemma), and every probability read at it
is near zero. Pass `add_special_tokens=False`, as below. A string prompt is
tokenized with the tokenizer's defaults, so on those models position 0 of every
activation is BOS.

**6. Code written for nnsight before 0.8 is wrong here.** `.value`,
`nnsight.list()`, `tracer.next()`, `with tracer.all():`, `LanguageModel` and
"proxies" all mark it. Translate it with the
[porting guide](references/debugging-porting-pre-0.8.md) before debugging it.

## Editing

```python
with model.trace(prompt):
    before = model.transformer.h[0].output.clone().save()   # clone: the next line overwrites it
    model.transformer.h[0].output[:, -1, :] = 0             # in place, part of a tensor
    model.transformer.h[1].attn.output[0][:] = 0            # tuple output: write into its element
    mlp = model.transformer.h[1].mlp
    mlp.output = mlp.output * 0.5                           # replacement with a new value
    mid = model.transformer.h[5].output
    lens = model.lm_head(model.transformer.ln_f(mid))[0, -1].argmax().save()

assert before[0, -1].abs().sum() > 0
```

Calling a module inside a trace, as the last line does, applies it to a value
without disturbing the forward pass or the ordering rule. Build new tensors on
the activation's `device` and `dtype`. Inputs past the first argument, attention
heads and the tuple rules are in
[access-and-modify](references/access-and-modify.md).

## Several prompts in one pass

A list is one batch with the same intervention on every row. Prompts are
left-padded, so `[:, -1]` is each prompt's last token and any other position
index shifts with the batch:

```python
with model.trace(["The Eiffel Tower is in", "Rome is in"]):
    last = model.output.logits[:, -1].save()

assert last.shape[0] == 2
```

For a different intervention per row, open the trace with no input and give each
variant a `tracer.invoke`. Inside an invoke, index as if its input were alone:

```python
paris = model.tokenizer.encode(" Paris", add_special_tokens=False)[0]

with model.trace() as tracer:
    scores = nnsight.save({})                     # keyed: invokes finish in model order
    with tracer.invoke(prompt):
        scores["clean"] = model.output.logits[0, -1, paris]
    for layer in range(len(model.transformer.h)):
        with tracer.invoke(prompt):
            model.transformer.h[layer].output[:, -1, :] = 0
            scores[layer] = model.output.logits[0, -1, paris]

assert len(scores) == 13                          # 12 ablations and a baseline, one forward pass
```

Write results into a saved dict from inside each invoke. A list appended to from
several invokes comes back in the order the model reached them, and code in the
trace body outside the invokes runs before all of them, so it cannot read what
an invoke binds. N traces for N variants are N forward passes. A mean or max over
tokens needs the padding masked, and passing a value between invokes has its own
rules: [batching](references/batching.md).

## Generation

```python
with model.generate(prompt, max_new_tokens=3, min_new_tokens=3, do_sample=False) as tracer:
    picks = nnsight.save([])
    for step in tracer.iter[:3]:                               # one step per generated token
        model.transformer.h[6].output[:, -1, :] *= 1.5         # an edit at every step
        picks.append(model.output.logits[0, -1].argmax(dim=-1))
    ids = tracer.result.save()                                 # token ids: prompt, then new

assert len(picks) == 3 and tuple(ids.shape) == (1, 13)
print(model.tokenizer.decode(ids[0]))
```

- **Without the loop, an intervention applies to the first forward pass only.**
- **Step 0 is the prefill** over the whole prompt, `(batch, prompt_len, hidden)`.
  Each later step is one position, `(batch, 1, hidden)`. Step *k* produces
  generated token *k*; `tracer.iter[1:3]` and `tracer.iter[[0, 2]]` select steps.
- **A loop must not ask for a step the run does not make.** It is cut short with
  `UserWarning: '...' was never reached: the loop asked for a step the run did
  not make`, what it saved is kept, and the statements after the loop,
  `tracer.result.save()` included, never run. `max_new_tokens` is only an upper
  bound, so pass `min_new_tokens=` to hold the count and check `len()` of what you
  collected. The same warning appears when the loop body reads modules out of
  forward order.
- **Sampling follows the checkpoint's `generation_config`**, and most instruct
  checkpoints sample. Pass `do_sample=False`. Keywords go to the model's
  `generate`.
- **Chat models**: pass the messages list
  (`[{"role": "user", "content": "..."}]`) as the input and the checkpoint's
  template is applied.
- `model.pipe(prompt, ...)` returns the pipeline's decoded records in place of
  ids. Keywords such as `max_new_tokens` on `model.trace` are ignored by its one
  forward pass. Gradients are unavailable inside `generate`.

Unknown step counts, streaming and the step-order traps:
[generation](references/generation.md).

## Gradients

```python
with model.trace(prompt):
    embeds = model.transformer.wte.output          # 1. capture in the forward
    hidden = model.transformer.h[-1].output
    metric = model.output.logits[0, -1, paris]     # 2. a scalar
    with metric.backward():                        # 3. a backward pass, interleaved
        grad = hidden.grad.clone().save()          # 4. read last layer first
        saliency = (embeds * embeds.grad).sum(-1).save()   # input x gradient per token

assert tuple(grad.shape) == (1, 10, 768) and tuple(saliency.shape) == (1, 10)
```

Capture tensors before the backward block, read `.grad` on the exact tensor you
captured (a slice such as `block.output[0]` is another tensor), in reverse
forward order, inside the block. Break either rule and the error names an object
id: `OutOfOrderError: '139853884760112.grad.i0' was requested but the model
already ran past it`. Outside the block `.grad` is `None`. Activations need no
`requires_grad_()`.
Editing gradients, optimizing a vector through the model and the replacements
that silently cut the graph: [gradients](references/gradients.md).

## Run methods

| You want | Use | `tracer.result` |
|---|---|---|
| one forward pass | `model.trace(x)` | the forward's output object |
| generated tokens | `model.generate(x, max_new_tokens=N)` | token ids |
| decoded text or labels | `model.pipe(x, ...)` | the pipeline's records |
| shapes with no compute or weights | `model.scan(x)` | fake tensors (not on plain `NNsight`) |
| several traces sharing values | `model.session()` | values flow between its traces |
| an intervention on every later run | `model.edit(inplace=True)` | undone by `model.clear_edits()` |

Without `with`, `generate` and `pipe` run and return their result, and
`model.trace(x)` runs nothing (`model.trace(x, trace=False)` is a plain forward).

## Remote

The same trace runs on NDIF against a model you cannot host. Build the model
without `dispatch=True`, add `remote=True`, and reduce before saving: every save
is a download.

```python
with model.trace(prompt, remote=True):
    top = model.output.logits[0, -1].argmax().cpu().save()
```

Several traces belong in one `model.session(remote=True)`. Read the
[remote guide](references/remote.md) before writing remote code.

## Scripts

Run them; do not read them. `scripts/inspect_model.py <repo_id> [--task T]
[--prompt P] [--grep attn]` prints module paths, execution order and
tensor-or-tuple without weights. `scripts/check_env.py [--remote]` prints
versions, GPUs, the NDIF key and host, and deployed models.

## Guides

The API, one topic each:

| To | Read |
|---|---|
| edit a tuple output, an input or keyword argument, or one attention head (GPT-2 or Llama-style); read a module that runs several times per forward | [access-and-modify](references/access-and-modify.md) |
| run a sweep in one pass, mask padding, pass a value from one prompt into another | [batching](references/batching.md) |
| generate with an unknown step count, stream tokens, fix a per-step list that is short or shifted | [generation](references/generation.md) |
| edit a gradient, run several backward passes, train a vector or adapter through the model | [gradients](references/gradients.md) |
| collect many modules at once, or get shapes without loading weights | [caching-and-scan](references/caching-and-scan.md) |
| skip a module, stop a pass early, chain traces, keep an intervention on every run | [control-flow](references/control-flow.md) |
| read a value computed inside a forward: attention scores, a pre-activation, any call or assignment | [source-tracing](references/source-tracing.md) |
| find module paths for a model family, place a model on devices, alias paths, wrap your own module, attach an SAE or probe, define a custom value | [modules-and-architectures](references/modules-and-architectures.md) |
| look up a signature, a `CONFIG` key or an exception | [api-reference](references/api-reference.md) |
| diagnose an error, a hang, an empty or wrong result | [debugging](references/debugging.md) |
| port code written for nnsight before 0.8 | [porting](references/debugging-porting-pre-0.8.md) |
| run on NDIF: setup, sessions, download size, remote-only failures | [remote](references/remote.md) |

Techniques:

| To | Read |
|---|---|
| see at which layer an answer becomes decodable | [logit-lens](references/logit-lens.md) |
| find which layer, position or head carries a behavior, with a clean and a corrupt prompt | [activation-patching](references/activation-patching.md) |
| rank every component at once with gradients | [attribution-patching](references/attribution-patching.md) |
| locate a fact from one prompt by noising the subject and restoring states | [causal-tracing](references/causal-tracing.md) |
| test whether a component is needed: zero, mean or resample ablation, every head in one trace | [ablation](references/ablation.md) |
| read attention patterns, score heads, find induction heads | [attention-analysis](references/attention-analysis.md) |
| find and validate the set of components a task uses | [circuit-discovery](references/circuit-discovery.md) |
| train a probe on activations, test its direction causally, erase a concept | [probing](references/probing.md) |
| attach, evaluate, train or steer with a sparse autoencoder | [sae-and-dictionary-learning](references/sae-and-dictionary-learning.md) |
| steer generation with a direction or a function vector | [model-steering](references/model-steering.md) |
| edit weights (ROME-style), load or train a LoRA adapter | [model-editing-and-lora](references/model-editing-and-lora.md) |
| choose a metric, controls and prompt sets before any of these | [interp-experiment-design](references/interp-experiment-design.md) |

Runtimes and other model types:

| To | Read |
|---|---|
| run interventions inside vLLM for throughput, streaming or serving | [vllm](references/vllm.md) |
| shard a model across GPUs and keep gradients and `.source` | [tensor-parallel](references/tensor-parallel.md) |
| load a model in 4 or 8 bits | [quantization](references/quantization.md) |
| trace a vision-language model or a diffusion pipeline | [diffusion-and-multimodal](references/diffusion-and-multimodal.md) |
| trace Whisper, a vision encoder, a classifier or another pipeline task | [other tasks](references/diffusion-and-multimodal-other-tasks.md) |
