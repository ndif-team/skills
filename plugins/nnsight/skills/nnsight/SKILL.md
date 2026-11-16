---
name: nnsight
description: "Read and edit the internals of any PyTorch model with nnsight: model.trace, .save(), interventions, batched invokes, gradients, .source, generation, and remote execution on NDIF (remote=True) for models too large to run locally. Load before writing, fixing or explaining any nnsight code, however short: idioms from before 0.8 (.value, LanguageModel, proxies) are widespread and silently wrong. Use for an nnsight error such as OutOfOrderError, a hang or a wrong result, porting old code, a model family nnterp does not support, arbitrary module or .source access, diffusion and vision-language models, vLLM, tensor parallelism and 4/8-bit loading, and wherever PyTorch forward hooks would otherwise be written. Bundles a guide per technique: logit lens, activation and attribution patching, causal tracing, ablation, attention analysis, circuit discovery, probing, SAEs, steering, editing and LoRA. The base layer under nnterp and nnspec: for a standard experiment on a supported transformer prefer nnspec, then nnterp."
---

# nnsight

nnsight reads, replaces and backpropagates through every intermediate value in a
model's forward pass, locally or on NDIF for models too large for your machine.
This file is the core API. Each technique and runtime has its own guide, listed
under [Guides](#guides): read the one that matches the task before writing code.

## Which library

nnsight is the base of three libraries. When the user names a library, use it. Otherwise take the highest row that fits,
and take a row only when its skill is available in this session:

| Use | When |
|---|---|
| `nnspec` skill | A standard experiment on a transformer language model (lens, patching, ablation, attention, steering, probes, SAEs, DAS) that fits a JSON document. No intervention code to write. |
| `nnterp` skill | The experiment needs Python that a document cannot express, on a family nnterp knows. One module vocabulary on every family. |
| nnsight (this skill) | nnterp does not know the family, or the work needs access below nnterp's standard values: arbitrary modules, `.source`, your own or non-transformer models, diffusion, vLLM, tensor parallelism. Also existing nnsight code, and any error an nnsight trace raises. |

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

Install: `pip install nnsight` (needs `torch` and `transformers`). Pick a device
with `device="cpu"` / `device="cuda:1"`, not `device_map=` — `device_map="cpu"`
still lands on the GPU; see the device table in
[references/modules-and-architectures.md](references/modules-and-architectures.md).

## The five things that break agent-written nnsight code

**1. `.save()` or it never existed.** Assignments inside a trace body do not escape
it — the body runs in a different frame. Bind what you save (`x = ....save()`; a
bare `....save()` on its own line returns nothing; neither does `float(x.save())`,
because the call around it still runs in the trace, so bind the saved value and
convert it after the block), and when collecting, save the
**container** and append raw values:

```python
with model.trace("The Eiffel Tower is in the city of"):
    per_layer = nnsight.save([])                  # save the list…
    for block in model.transformer.h:
        per_layer.append(block.output[0, -1])     # …append raw values

assert len(per_layer) == 12
```

There is no `.value` in 0.8 — the saved variable *is* the tensor. Saved
activations arrive with `requires_grad=True` and a live `grad_fn` that keeps the
forward graph alive: `.detach()` anything you keep or reuse (full story: the
[sae-and-dictionary-learning guide](references/sae-and-dictionary-learning.md) and
[references/gradients.md](references/gradients.md)).

**2. Access modules in forward-pass order.** Reading layer 8 then layer 2 raises
`OutOfOrderError`: your code is a worker that parks until the model produces each
value, and the model has already gone past. Within a block, the submodules
(`ln_1`, `attn`, `mlp`) come before the block's own `.output`. This binds writes
too — an edit at layer 0 goes above a read at layer 11, not below it. Inside a
`tracer.iter` loop a wrong-order read binds to the *next* step instead of raising
— see [references/generation.md](references/generation.md).

**3. Don't guess module paths or output types.** `model.transformer.h[i]` is GPT-2;
Llama is `model.model.layers[i]`; Gemma-3 is `model.model.language_model.layers[i]`.
A block's `.output` is a plain tensor, but an attention submodule's is a tuple —
so `.output[0]` copied from an old example silently selects batch row 0. Run
`python scripts/inspect_model.py <repo_id> --prompt "..."`: without downloading
weights, it prints every block child **in execution order** and whether each
output is a tensor or a tuple.

**4. One trace is one forward pass — structure by input, not by activation.**
N traces to fetch N layers is N forward passes (remotely, N round-trips).

**5. Old nnsight code is everywhere and it is wrong here.** `.value`,
`nnsight.list()`, `tracer.next()`, `with tracer.all():`, `LanguageModel`,
"proxies" — all pre-0.8. If you are adapting code from a tutorial or a paper repo,
convert it first; see the [debugging guide](references/debugging.md).

## Picking a run method

| You want | Use | You get back |
|---|---|---|
| one forward pass | `model.trace(x)` | the model's output object |
| generated tokens | `model.generate(x, max_new_tokens=N)` | token ids on `tracer.result` (the checkpoint's `generation_config` decides sampling — pass `do_sample=False`) |
| decoded text / labels | `model.pipe(x, ...)` | pipeline records (often sampled — pass `do_sample=False`) |
| shapes, no compute | `model.scan(x)` | fake tensors: shapes and dtypes only, no values (not on plain `NNsight`) |
| several traces sharing values | `model.session()` | values flow between traces |
| a permanent intervention | `model.edit(inplace=True)` | replayed on every later run |

## Core moves

**Batch a sweep into one pass.** Loop inside the trace, one `tracer.invoke` per
variant, no input on `trace()`:

```python
prompt = "The Eiffel Tower is in the city of"
paris = model.tokenizer(" Paris").input_ids[0]

with model.trace() as tracer:
    scores = nnsight.save({})                   # keyed: invokes finish in model order
    for layer in range(len(model.transformer.h)):
        with tracer.invoke(prompt):
            model.transformer.h[layer].output[:, -1, :] = 0
            scores[layer] = model.output.logits[0, -1, paris]

assert sorted(scores) == list(range(12))        # 12 ablations, one forward pass
```

Inside an invoke, index as if that input were alone — `[:, -1, :]` means "this
invoke's rows, last position". That holds for the last position only: inputs are
left-padded to the longest, so a full-sequence or absolute index includes the pads;
slice `[:, -len(ids):]` or see
[references/batching.md](references/batching.md#positions-shift-with-the-batch).
Trace-body code *outside* the invokes runs before
any of them, so it raises `NameError` on a name an invoke binds — combine across
invokes through a `nnsight.save({})` written from inside each one, as above.

**Grab many modules at once** with `tracer.cache(modules=[...])`, opened first thing
in the trace — see [references/caching-and-scan.md](references/caching-and-scan.md).

**Gradients** — capture in the forward, read in reverse order inside
`with metric.backward():`:

```python
with model.trace(prompt):
    hidden = model.transformer.h[-1].output
    metric = model.output.logits[0, -1, paris]
    with metric.backward():
        grad = hidden.grad.clone().save()
```

**Generation** — a `tracer.iter` loop must not ask for a step the run does not
make: one that outruns the run warns, keeps what it saved, and silently drops the
statements after the loop. `max_new_tokens` is only an upper bound, so pass
`min_new_tokens=` when the bound has to hold, and check `len()` of what you collected:

```python
with model.generate(prompt, max_new_tokens=3, min_new_tokens=3) as tracer:
    picks = nnsight.save([])
    for step in tracer.iter[:3]:
        picks.append(model.output.logits[0, -1].argmax(dim=-1))
    ids = tracer.result.save()

assert len(picks) == 3 and ids.shape[1] == 13
```

## Running it somewhere else

The same trace runs on NDIF against a model you can't fit locally — add
`remote=True` and reduce metrics *before* saving, since every `.save()` is a
download:

```python
with model.trace(prompt, remote=True):
    last = model.transformer.h[-1].output[:, -1].detach().cpu().save()
```

Anything beyond one trace should be a `model.session(remote=True)` — one job
instead of N round-trips. Load the [remote guide](references/remote.md) before writing remote code.

## References

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
| [references/modules-and-architectures.md](references/modules-and-architectures.md) | constructor and device placement, per-family module paths (MoE, SSM, hybrids included), `rename`, wrapping your own module, attaching SAEs/probes, custom served values with `eproperty` / `envoys=` |
| [references/api-reference.md](references/api-reference.md) | model classes and every method, property, config key, and exception in tables |

Scripts (run them, don't read them): `scripts/inspect_model.py <repo_id>
[--prompt P] [--grep attn]` — module paths, execution order, tensor-vs-tuple,
without weights; `scripts/check_env.py [--remote]` — versions, GPUs, NDIF key and
host, deployed models.

## Guides

Read the guide for the task before writing code. Each is self-contained, and
carries the traps specific to its technique.

Techniques:

- [ablation](references/ablation.md) — remove a component (zero, mean, resample) to test necessity
- [activation-patching](references/activation-patching.md) — copy activations between runs to localize a behavior
- [attribution-patching](references/attribution-patching.md) — gradient approximation of patching, for screening
- [causal-tracing](references/causal-tracing.md) — ROME-style corrupt-and-restore to find where a fact lives
- [attention-analysis](references/attention-analysis.md) — attention patterns, per-head metrics, head types
- [logit-lens](references/logit-lens.md) — decode intermediate residual streams through the unembedding
- [probing](references/probing.md) — classifiers on activations: what is decodable, and where
- [circuit-discovery](references/circuit-discovery.md) — find and validate the subgraph a task uses
- [model-steering](references/model-steering.md) — add directions to the residual stream at inference
- [model-editing-and-lora](references/model-editing-and-lora.md) — persistent weight/activation edits, ROME, LoRA
- [sae-and-dictionary-learning](references/sae-and-dictionary-learning.md) — sparse autoencoders: attach, read, train
- [interp-experiment-design](references/interp-experiment-design.md) — metrics, prompt sets, controls, baselines

Infrastructure:

- [remote](references/remote.md) — running on NDIF: sessions, request batching, download size
- [debugging](references/debugging.md) — an error, a hang, an empty result, or pre-0.8 code to port
- [vllm](references/vllm.md) — nnsight inside the vLLM engine: throughput, serving, taps
- [tensor-parallel](references/tensor-parallel.md) — a model sharded across GPUs with transformers TP
- [quantization](references/quantization.md) — 4/8-bit loading and what it changes in a trace
- [diffusion-and-multimodal](references/diffusion-and-multimodal.md) — VLMs, diffusion pipelines, audio/vision encoders
