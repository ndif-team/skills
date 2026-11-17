# API Reference

Signatures and names, for looking up. `model` is any model class below; `tracer`
is what `as tracer:` binds.

## Model classes

| Class | Import | For |
|---|---|---|
| `TransformersModel` | `from nnsight import TransformersModel` | any HuggingFace checkpoint, any pipeline task |
| `NNsight` | `from nnsight import NNsight` | any `torch.nn.Module` you already built |
| `DiffusionModel` | `from nnsight import DiffusionModel` | a `diffusers` pipeline |
| `VLLM` | `from nnsight.modeling.vllm import VLLM` | the vLLM engine, `mode="sync"` or `"async"` |

`TransformersModel(repo_id, *, task=None, tokenizer=None, processor=None,
image_processor=None, feature_extractor=None, peft=None,
autocast_adapter_dtype=True, revision=None, dispatch=False, rename=None,
envoys=None, **hf_kwargs)`. `hf_kwargs` carries
`device`, `device_map`, `dtype`, `attn_implementation`, `trust_remote_code` and
anything else transformers accepts.

`NNsight(module, rename=None, envoys=None)`.

## Run methods

| Method | Signature |
|---|---|
| `model.trace` | `trace(*inputs, trace=True, remote=False, blocking=True, backend=None, **forward_kwargs)` |
| `model.generate` | `generate(*inputs, max_new_tokens=N, remote=False, **generate_kwargs)` (`TransformersModel`, `DiffusionModel`) |
| `model.pipe` | `pipe(*inputs, **pipeline_kwargs)` (`TransformersModel`) |
| `model.scan` | `scan(*inputs, **kwargs)` (not on plain `NNsight`) |
| `model.edit` | `edit(*, inplace=False)`; binds `(tracer, edited_copy)`, or `tracer` when `inplace=True` |
| `model.session` | `session(remote=False, blocking=True)` |
| `tensor.backward` | `with tensor.backward(retain_graph=False):` |
| `model.dispatch` | `dispatch()` loads the weights now |

## Tracer

| Member | Signature | Does |
|---|---|---|
| `tracer.invoke` | `invoke(*inputs, **kwargs)` | one batched input group; with no input, the whole batch |
| `tracer.result` | property | the traced call's return value; the last thing a block can read |
| `tracer.iter` | `iter[int \| slice \| list]` | select occurrences (generation steps, repeated calls): `for step in tracer.iter[:3]:` |
| `tracer.all` | `all()` | `tracer.iter[:]` |
| `tracer.cache` | `cache(modules=None, device=cpu, dtype=None, detach=True, include_output=True, include_inputs=False, non_blocking=False)` | record many modules |
| `tracer.barrier` | `barrier(n)` | a meeting point for `n` invokes; call the object it returns |
| `tracer.stop` | `stop()` | end the forward pass now |
| `tracer.backend` | attribute | after a remote block: the job (`job_id`, `status`, `meta`, callable to poll) |

## Envoy (every module)

| Member | Is |
|---|---|
| `.output` | the forward's return value, read or assign |
| `.input` | first positional argument (or first keyword), read or assign |
| `.inputs` | `(args, kwargs)`, read or assign |
| `.source` | operations inside the forward; `print(envoy.source)` lists them |
| `.skip(replacement)` | do not run the module; use `replacement` as its output |
| `envoy(*args, hook=False, **kwargs)` | apply the module to a value; `hook=True` makes its internals readable |
| `.get("transformer.h.0.mlp")` | a descendant by dotted path |
| `.modules(include_fn=None, names=False)`, `.named_modules(include_fn=None)` | descendants; names are absolute (`model.transformer.h.0`) |
| `envoy[i]`, `len(envoy)`, iteration | children of a `ModuleList` or `Sequential` |
| `.to(device)`, `.cpu()`, `.cuda()` | move the module; returns the envoy |
| `.device`, `.devices` | device, or set of devices, of its parameters |
| `.path` | its dotted path |
| `._module` | the wrapped `torch.nn.Module` itself |
| `.edit(inplace=False)`, `.clear_edits()` | store or drop persistent interventions |
| `.E_output`, `.E_input` | a child module literally named `output` / `input` |

## Model handles

| Member | On | Is |
|---|---|---|
| `model.tokenizer`, `.processor`, `.image_processor`, `.feature_extractor` | `TransformersModel` | what the task loaded; the others are `None` |
| `model.pipeline` | `TransformersModel`, `DiffusionModel` | the underlying pipeline |
| `model.config`, `.repo_id`, `.revision`, `.task`, `.dispatched` | `TransformersModel` | metadata |
| `model.generator.streamer.output` | `TransformersModel` | this step's tokens during `generate` |
| `model.load_adapter(peft_id)` | `TransformersModel` | attach or swap a PEFT adapter; `None` removes it |
| `model.disable_adapters()`, `model.enable_adapters()`, `model.adapters_enabled` | `TransformersModel` | switch an attached adapter off and on; the base model's forward while off |
| `model.to_model_key()` | remote-capable models | the identity a remote request carries: repo id, revision, task |
| `model.logits`, `model.samples` | `VLLM` | this step's pre-sampling logits and sampled ids |

## Top-level functions

| Member | Signature | Does |
|---|---|---|
| `nnsight.save` | `save(value)` | mark a value to leave the outermost trace; same as `value.save()`. Raises `ValueError` outside a trace |
| `nnsight.register` | `register(module)` | ship a local Python module's source with remote requests |
| `nnsight.status` | `status(raw=False)` | deployed models on NDIF and their state |
| `nnsight.is_model_running` | `is_model_running(repo_id, revision="main", task=None, trust_remote_code=None)` | whether that deployment is RUNNING |
| `nnsight.compare` | `compare(host=None)` | local against server Python and package versions |
| `nnsight.CONFIG` | object | settings, below |
| `nnsight.NNsightDeprecationWarning` | `FutureWarning` subclass | the category of every nnsight deprecation |

Command line: `nnsight login` stores an NDIF key, `nnsight whoami` shows whose it
is, `nnsight-serve <repo_id>` serves a vLLM engine.

## Remote

`remote=` on `trace`, `generate` and `session`:

| Value | Behavior |
|---|---|
| `True` | run on `CONFIG.API.HOST`; with `blocking=False`, submit and return a job to poll |
| `"local"` | serialize, deserialize and run in this process: a dry run of the remote path |
| `"http://host:port"` | as `True`, against that host |

`RemoteBackend(model_key, host=None, api_key=None, blocking=True, job_id=None)`
and `AsyncRemoteBackend` are in `nnsight.intervention.backends.remote`, with
`RemoteError`. See [remote](remote.md).

## CONFIG

Loaded from the package defaults, then `~/.config/nnsight/config.yaml` (or
`$NNSIGHT_CONFIG`), then the environment. `CONFIG.save()` writes the user file.

| Setting | Default | Does | Environment |
|---|---|---|---|
| `CONFIG.API.HOST` | `https://api.ndif.us` | NDIF base URL | `NDIF_HOST` |
| `CONFIG.API.APIKEY` | `None` | NDIF key; `CONFIG.set_default_api_key(key)` sets and saves it | `NDIF_API_KEY` |
| `CONFIG.API.COMPRESS` | `True` | compress request and result payloads | |
| `CONFIG.APP.DEBUG` | `False` | keep nnsight's frames in tracebacks; log remote payload sizes | `NNSIGHT_DEBUG` |
| `CONFIG.APP.REMOTE_LOGGING` | `True` | show remote status and the block's `print()` output | |
| `CONFIG.APP.PYMOUNT` | `True` | mount `.save()` on every object; when off, use `nnsight.save(x)` | |
| `CONFIG.APP.DISABLE_CPP_BACKTRACE` | `True` | keep a torch C++ error inside a trace from crashing the process | `NNSIGHT_DISABLE_CPP_BACKTRACE` |

## Exceptions

| Exception | Raised when |
|---|---|
| `OutOfOrderError` | a location was requested after the model ran past it, or never ran |
| `SourceNotAvailable` | `.source` cannot drill into that operation |
| `WithBlockNotFoundError` | the trace body's source text is not available |
| `RemoteError` | a remote job failed; the message ends with the original error |
| `ValueError: save() was called outside a trace` | a save before or after the `with` |
| `ValueError: Cannot access ... outside of interleaving` | `.output` read with no trace running |
| `NotImplementedError: ... does not support batching multiple invokes` | plain `NNsight` with two input invokes |

Diagnosis by message: [debugging](debugging.md).
