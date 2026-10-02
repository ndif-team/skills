---
name: nnsight
description: Read, modify, and analyze the internals of neural networks with nnsight 0.8 — tracing activations, intervening on modules, batching interventions, gradients, caching, generation, and remote execution on NDIF for models too large to run locally. Use for any task that touches model internals rather than just outputs: activation extraction, logit lens, activation and attribution patching, causal tracing, ablation, attention-head analysis, circuit discovery, probing, sparse autoencoders, steering vectors, model editing and LoRA, vLLM and tensor-parallel runs, quantized and multimodal models, and debugging or porting existing nnsight code. Load this before writing any nnsight code: idioms from older versions (.value, nnsight.list(), LanguageModel, proxies) are widespread and silently wrong on 0.8. Bundles one guide per technique in references/ and scripts that print a model's module paths without downloading weights.
license: MIT
metadata:
  author: ndif-team
  source: https://github.com/ndif-team/skills
  nnsight: "0.8"
  transformers: ">=5"
---

# nnsight

nnsight gives you access to every intermediate value in a model's forward pass —
reading them, replacing them, backpropagating through them — for models you run
locally and for models too large to fit on your machine (via NDIF).

This file is the general-purpose reference. Techniques built on it (logit lens,
activation patching, steering, …) each have a guide under `references/`; see
[Technique and runtime guides](#technique-and-runtime-guides) at the bottom.

## Orientation

```python
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt):
    clean = model.output.logits.save()                     # read the output

with model.trace(prompt):
    resid = model.transformer.h[5].output.save()           # read an activation
    model.transformer.h[8].output[:, -1, :] *= 2           # modify one
    edited = model.output.logits.save()                    # see what it changed

assert tuple(resid.shape) == (1, 10, 768)
assert model.tokenizer.decode(clean[0, -1].argmax()) == " Paris"
assert model.tokenizer.decode(edited[0, -1].argmax()) == " London"
```

Install: `pip install nnsight` (needs `torch` and `transformers`).

## The five things that break agent-written nnsight code

**1. `.save()` or it never existed.** Assignments inside a trace body do not escape
it — the body runs in a different frame. Bind what you save (`x = ....save()`; a
bare `....save()` on its own line returns nothing), and when collecting, save the
**container** and append raw values:

```python
with model.trace("The Eiffel Tower is in the city of"):
    per_layer = nnsight.save([])                  # save the list…
    for block in model.transformer.h:
        per_layer.append(block.output[0, -1])     # …append raw values

assert len(per_layer) == 12
```

There is no `.value` in 0.8 — the saved variable *is* the tensor.

**2. Access modules in forward-pass order.** Reading layer 8 then layer 2 raises
`OutOfOrderError`: your code is a worker that parks until the model produces each
value, and the model has already gone past. Within a block, the submodules
(`ln_1`, `attn`, `mlp`) come before the block's own `.output`. This binds writes
too — an edit at layer 0 goes above a read at layer 11, not below it.

**3. Don't guess module paths or output types.** `model.transformer.h[i]` is GPT-2;
Llama is `model.model.layers[i]`; Gemma-3 is `model.model.language_model.layers[i]`.
A block's `.output` is a plain tensor, but an attention submodule's is a tuple —
so `.output[0]` copied from an old example silently selects batch row 0. Run:

```
python scripts/inspect_model.py <repo_id> --prompt "your prompt"
```

It builds the model on `meta` (a 27B model takes ~8s, no weights downloaded) and
prints the layer path, every block child **in execution order**, and whether each
output is a tensor or a tuple.

**4. One trace is one forward pass — structure by input, not by activation.**
Everything you want from one input comes out of one trace. N traces to fetch N
layers is N forward passes (and, remotely, N network round-trips).

**5. Old nnsight code is everywhere and it is wrong here.** `.value`,
`nnsight.list()`, `tracer.next()`, `with tracer.all():`, `LanguageModel`,
"proxies" — all pre-0.8. If you are adapting code from a tutorial or a paper repo,
convert it first; see the `debugging` guide (`references/debugging.md`).

## Picking a run method

| You want | Use | You get back |
|---|---|---|
| one forward pass | `model.trace(x)` | the model's output object |
| generated tokens | `model.generate(x, max_new_tokens=N)` | token ids on `tracer.result` (the checkpoint's `generation_config` decides sampling — pass `do_sample=False`) |
| decoded text / labels | `model.pipe(x, ...)` | pipeline records (often sampled — pass `do_sample=False`) |
| shapes, no compute | `model.scan(x)` | fake tensors: shapes and dtypes only — it cannot see devices or values. Loader-backed models only (`TransformersModel`, `DiffusionModel`, `VLLM`); plain `NNsight(module)` has no `.scan` |
| several traces sharing values | `model.session()` | values flow between traces |
| a permanent intervention | `model.edit(inplace=True)` | replayed on every later run |

## Core moves

**Batch a sweep into one pass.** Loop inside the trace, one `tracer.invoke` per
variant, no input on `trace()`:

```python
prompt = "The Eiffel Tower is in the city of"
paris = model.tokenizer(" Paris").input_ids[0]

with model.trace() as tracer:
    scores = nnsight.save([])
    for layer in range(len(model.transformer.h)):
        with tracer.invoke(prompt):
            model.transformer.h[layer].output[:, -1, :] = 0
            scores.append(model.output.logits[0, -1, paris])

print([round(s.item(), 2) for s in scores])     # 12 ablations, one forward pass
```

Inside an invoke, index as if that input were alone — `[:, -1, :]` means "this
invoke's rows, last position".

**Grab many modules at once** with `tracer.cache()` (call it first thing; it
observes post-intervention values, and returns a *list* per module when the module
runs more than once, as in generation):

```python
with model.trace(prompt) as tracer:
    cache = tracer.cache(modules=[model.transformer.h[0], model.transformer.h[11]])

print(cache["model.transformer.h.0"].output.shape)
```

**Gradients** — capture in the forward, read in reverse order inside
`with metric.backward():`:

```python
with model.trace(prompt):
    hidden = model.transformer.h[-1].output
    metric = model.output.logits[0, -1, paris]
    with metric.backward():
        grad = hidden.grad.clone().save()

print(grad.shape)
```

**Generation** — a `tracer.iter` loop must not ask for a step the run does not
make. A bound the run meets keeps the code after the loop; a loop that outruns
the run warns, keeps what it saved, and drops the statements after it — the
result looks complete while being short. `max_new_tokens` is an upper bound, so
pass `min_new_tokens=` when the bound has to hold, and check the `len()` of what
you collected:

```python
with model.generate(prompt, max_new_tokens=3, min_new_tokens=3) as tracer:
    picks = nnsight.save([])
    for step in tracer.iter[:3]:
        picks.append(model.output.logits[0, -1].argmax(dim=-1))
    ids = tracer.result.save()

assert len(picks) == 3
print(model.tokenizer.decode(ids[0]))
```

## Running it somewhere else

The same trace runs on NDIF against a model you can't fit locally — add
`remote=True` and reduce metrics *before* saving, since every `.save()` is a
download:

```python
with model.trace(prompt, remote=True):
    last = model.transformer.h[-1].output[:, -1].detach().cpu().save()

print(last.shape)
```

Anything beyond one trace (loops, sweeps, multi-step experiments) should be a
`model.session(remote=True)` — one job instead of N round-trips. See the `remote` guide (`references/remote.md`) before writing remote code.

## References

Read the one that matches the task before writing code.

| File | Covers |
|---|---|
| [references/execution-model.md](references/execution-model.md) | how tracing actually works: deferred execution, interleaving, ordering, `.save()` semantics. **Read this once.** |
| [references/access-and-modify.md](references/access-and-modify.md) | `.output` / `.input` / `.inputs`, in-place vs replacement, tuples, heads, positions, devices |
| [references/batching.md](references/batching.md) | `tracer.invoke`, empty invokes, sweeps, cross-invoke values, `tracer.barrier` |
| [references/generation.md](references/generation.md) | `generate` vs `pipe`, `tracer.iter`, per-step interventions, streaming, chat templates |
| [references/gradients.md](references/gradients.md) | `with tensor.backward():`, saliency, input×grad, gradient editing, optimizing through a frozen model |
| [references/caching-and-scan.md](references/caching-and-scan.md) | `tracer.cache()`, `model.scan()`, fake-tensor rules |
| [references/control-flow.md](references/control-flow.md) | `skip`, `tracer.stop`, `session`, `edit`, conditionals |
| [references/source-tracing.md](references/source-tracing.md) | `.source` — attention patterns and other values inside a forward |
| [references/modules-and-architectures.md](references/modules-and-architectures.md) | model classes, per-family module paths (MoE, SSM, hybrids included), `rename`, wrapping your own module, attaching SAEs/probes, custom served values with `eproperty` / `envoys=` |
| [references/api-reference.md](references/api-reference.md) | every method, property, config key, and exception in tables |

Scripts (run them, don't read them):

- `scripts/inspect_model.py <repo_id> [--prompt P] [--grep attn] [--depth 2]
  [--task text-generation] [--trust-remote-code]` — module paths, execution order,
  tensor-vs-tuple and MoE router/expert paths, without loading weights. It prints
  the task and the class it built: the task decides the class, and the class
  decides the tree
- `scripts/check_env.py [--remote]` — versions, GPUs, NDIF key/host, deployed
  models, and the local-vs-NDIF package diff

## Before you run an experiment

- Module paths and output types confirmed with `inspect_model.py`, not assumed
- Every value you need after the block is `.save()`d, containers not elements
- Module accesses in forward order; sweeps batched into invokes
- A control condition in the same trace (unmodified run) to compare against
- For generation: bounded `tracer.iter[:N]`, and `do_sample=False` if you want
  reproducibility from `pipe`
- For large models: reduce metrics *inside* the trace before saving

## Technique and runtime guides

Everything the NDIF team ships as separate skills is bundled here as one guide
each. Read the one that matches the task before writing code; each is
self-contained and its examples run against real models in CI.

**Foundation**

| Guide | Use when... |
|---|---|
| [debugging](references/debugging.md) | an error, a hang, an empty result, or pre-0.8 code to port |
| [remote](references/remote.md) | running on NDIF: sessions, request batching, download size, non-blocking jobs |

**Techniques**

| Guide | Use when... |
|---|---|
| [interp-experiment-design](references/interp-experiment-design.md) | choosing a metric, controls, and sanity checks before running anything |
| [logit-lens](references/logit-lens.md) | decoding what each layer predicts; tracking where an answer emerges |
| [activation-patching](references/activation-patching.md) | locating the layers, positions, or heads that carry a behavior; DAS |
| [attribution-patching](references/attribution-patching.md) | scaling patching to whole models with a gradient approximation |
| [causal-tracing](references/causal-tracing.md) | corrupt-and-restore factual localization (ROME-style) |
| [ablation](references/ablation.md) | testing necessity: zero, mean, resample, and noise ablation |
| [attention-analysis](references/attention-analysis.md) | attention patterns, per-head metrics, induction/copy head detection |
| [circuit-discovery](references/circuit-discovery.md) | finding and validating the subgraph behind a task (IOI-style) |
| [probing](references/probing.md) | training classifiers on activations; what is linearly decodable, and whether it is used |
| [sae-and-dictionary-learning](references/sae-and-dictionary-learning.md) | feature-level analysis; attaching, training, and evaluating SAEs |
| [model-steering](references/model-steering.md) | steering vectors, function vectors, persistent behavioral edits |
| [model-editing-and-lora](references/model-editing-and-lora.md) | weight edits, ROME-style updates, adapters trained through a frozen model |

**Runtimes and tooling**

| Guide | Use when... |
|---|---|
| [nnterp](references/nnterp.md) | one script that runs unchanged across GPT-2, Llama, Qwen, Gemma |
| [vllm](references/vllm.md) | throughput, continuous batching, CUDA-graph taps, `model.edit()` sweeps, nnsight-serve |
| [tensor-parallel](references/tensor-parallel.md) | a model too big for one GPU, sharded with transformers TP under `torchrun` |
| [quantization](references/quantization.md) | a model too big for one GPU, held in 4 or 8 bits (`dtype="nf4"`, `"int8"`) |
| [diffusion-and-multimodal](references/diffusion-and-multimodal.md) | VLMs, diffusion pipelines, the diffusion lens, non-text tasks |
