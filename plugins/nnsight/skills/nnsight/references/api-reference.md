# API Reference (nnsight 0.8)

Every public surface in one place. `model` is an `NNsight` / `TransformersModel` /
`DiffusionModel` / `VLLM`; `tracer` is what `as tracer:` binds.

## Model classes

| Class | Import | For |
|---|---|---|
| `TransformersModel` | `from nnsight import TransformersModel` | **primary** — any HuggingFace checkpoint, any task |
| `NNsight` | `from nnsight import NNsight` | any `torch.nn.Module` |
| `DiffusionModel` | `from nnsight import DiffusionModel` | `diffusers` pipelines |
| `VLLM` | `from nnsight.modeling.vllm import VLLM` | vLLM engine; `mode="sync"｜"async"` |
| `LanguageModel`, `VisionLanguageModel` | — | **deprecated**, warn on construction |

Constructor: `TransformersModel(repo_id, *, task=None, tokenizer=None, processor=None,
image_processor=None, feature_extractor=None, peft=None, revision=None,
dispatch=False, rename=None, **hf_kwargs)`.

## Run methods

Each is usable as `with model.<method>(...) as tracer:`, or called directly to just
run. Pass input to the method for one implicit invoke; pass nothing and define the
batch with `tracer.invoke(...)` blocks.

| Method | Runs | `tracer.result` |
|---|---|---|
| `model.trace(*inputs, **kw)` | one forward pass | the forward's return value |
| `model.generate(*inputs, max_new_tokens=N)` | the model's `generate` (greedy default) | **token ids** `[batch, seq]` |
| `model.pipe(*inputs, **kw)` | the whole task pipeline | pipeline **records** (text, labels) |
| `model.scan(*inputs)` | one forward under fake tensors — no weights, no compute | (read shapes inside) |
| `model.edit(*, inplace=False)` | captures interventions as replayed defaults | `as (tracer, edited)` / `as tracer` |
| `model.session(*, remote=False)` | a scope enclosing several traces | (only saves survive) |
| `with tensor.backward(...):` | a backward pass, interleaved | — |

`model.trace(..., trace=False)` bypasses tracing for a plain forward.

## Tracer

| Member | Signature | Does |
|---|---|---|
| `tracer.invoke` | `invoke(*args, **kwargs)` | add one batched input group; empty = whole batch |
| `tracer.result` | — | the traced call's return value; served after the forward, so read it *after* `model.output` |
| `tracer.iter` | `iter[slice｜int｜list]` | target occurrences; loop `for step in tracer.iter[:3]:` |
| `tracer.all` | `all()` | `tracer.iter[:]` — every occurrence (unbounded; drops trailing code) |
| `tracer.cache` | `cache(modules=None, device=cpu, dtype=None, detach=True, include_output=True, include_inputs=False, non_blocking=False)` | record many modules at once |
| `tracer.barrier` | `barrier(n) -> Barrier` | cross-invoke meeting point; call it, don't enter it |
| `tracer.stop` | `stop()` | end the forward pass now |

`tracer.next()` does not exist in 0.8.

## Envoy properties (any module)

| Member | Is |
|---|---|
| `.output` | the module's forward return value (read/assign) |
| `.input` | first positional arg, or first kwarg (read/assign) |
| `.inputs` | `(args, kwargs)` (read/assign — the only way to edit past the first argument) |
| `.source` | operation-level handle into the forward |
| `.device` / `.devices` | device(s) of its parameters |
| `.path` | its dotted path in the tree |

If a child module shadows one of these names, the property moves to `.nns_output` /
`.nns_input` on that module.

## Envoy methods

| Member | Signature | Does |
|---|---|---|
| `envoy.skip` | `skip(replacement)` | bypass this module's forward |
| `envoy(...)` | `envoy(*args, hook=False, **kwargs)` | apply the module to a value; `hook=True` lets *nnsight* watch the call, so its submodules become addressable (the module's own PyTorch hooks fire either way) |
| `envoy.get` | `get("transformer.h.0.mlp")` | fetch a descendant by dotted path |
| `envoy.modules` | `modules(include_fn=None, names=False)` | list descendants |
| `envoy.named_modules` | `named_modules(include_fn=None)` | `(path, envoy)` pairs (**absolute** paths) |
| `envoy.to` / `.cpu` / `.cuda` | `to(device)` | move the underlying module |
| `envoy.clear_edits` | `clear_edits()` | drop stored edits |
| `envoy[i]`, `len(envoy)`, iteration | — | index / iterate direct children |
| `envoy.source.<op>` | e.g. `.source.self_act_0` | one call site inside the forward; `print(envoy.source)` lists them |

## Top-level functions

| Member | Signature | Does |
|---|---|---|
| `nnsight.save` | `save(obj) -> obj` | mark a value to survive the outermost trace. **Raises outside a trace.** |
| `nnsight.register` | `register(module｜"name")` | ship a local module's source with remote requests |
| `nnsight.status` | `status(raw=False)` | query NDIF; `print()` shows deployed models |
| `nnsight.is_model_running` | `is_model_running(repo_id, revision="main")` | is it RUNNING on NDIF |
| `nnsight.compare` | `compare()` | local vs NDIF package/python diff |
| `nnsight.CONFIG` | — | the config singleton |

Removed in 0.8: `nnsight.list/dict/int/float/bool`, `nnsight.apply`, `nnsight.cond`,
`nnsight.log`, `nnsight.local`, `nnsight.session`. Use plain Python and
`model.session()`.

## Model-specific handles

| Member | On | Is |
|---|---|---|
| `model.tokenizer` / `.processor` / `.image_processor` / `.feature_extractor` | `TransformersModel` | whatever the task loaded (any may be `None`) |
| `model.pipeline` | `TransformersModel`, `DiffusionModel` | the underlying pipeline |
| `model.config`, `.repo_id`, `.revision`, `.dispatched` | `TransformersModel` | metadata |
| `model.generator.streamer.output` | `TransformersModel` | per-step tokens during decoding |
| `model.generator.output` | `TransformersModel` | **deprecated** — use `tracer.result` |
| `model.logits`, `model.samples` | `VLLM` | this step's pre-sampling logits / drawn ids |

## Remote

`remote=` on `trace` / `generate` / `session`:

| Value | Behavior |
|---|---|
| `True` | run on `CONFIG.API.HOST`; `blocking=True` (default) waits, `blocking=False` returns a job to `poll()` |
| `"local"` | serialize/deserialize and run in-process — an offline dry run |
| `"http://host:port"` | same as `True` against that host |

See the `nnsight-remote` skill.

## Config

`nnsight.CONFIG`, loaded from package defaults → `~/.config/nnsight/config.yaml` →
environment.

| Setting | Default | Does |
|---|---|---|
| `CONFIG.API.HOST` | `https://api.ndif.us` | NDIF base URL (`NDIF_HOST`) |
| `CONFIG.API.APIKEY` | `None` | NDIF key (`NDIF_API_KEY`, or `CONFIG.set_default_api_key(...)`) |
| `CONFIG.API.COMPRESS` | `True` | compress payloads and results |
| `CONFIG.APP.DEBUG` | `False` | keep nnsight's internal frames in tracebacks (`NNSIGHT_DEBUG`) |
| `CONFIG.APP.REMOTE_LOGGING` | `True` | stream remote `print()` back as log events |
| `CONFIG.APP.PYMOUNT` | `True` | mount `.save()` on every object; when off use `nnsight.save(x)` |

`CONFIG.save()` writes to the user file.

## Exceptions

| Exception | Means |
|---|---|
| `OutOfOrderError` | a location was requested after the model ran past it |
| `EarlyStopException` | `tracer.stop()` — a clean exit, swallowed by the interleaver |
| `SourceNotAvailable` | no Python source to instrument (builtin/C), a submodule call, or an assignment op |
| `WithBlockNotFoundError` | the trace body's source could not be read |
| `NotImplementedError` (batching) | this model can't batch multiple input invokes |
| `ValueError: save() was called outside a trace` | move the save inside |
| `ValueError: Cannot access ... outside of interleaving` | `.output` read outside a trace — also what an input-less `trace()` with no `invoke` says |

Full diagnosis: the `nnsight-debugging` skill.
