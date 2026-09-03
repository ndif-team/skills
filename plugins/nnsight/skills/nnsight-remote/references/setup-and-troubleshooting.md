# NDIF Setup and Troubleshooting

## Getting access

1. Get a key at [login.ndif.us](https://login.ndif.us).
2. Set it once — it persists to `~/.config/nnsight/config.yaml`:

<!-- test: skip -->
```python
from nnsight import CONFIG
CONFIG.set_default_api_key("YOUR_KEY")
```

Or export `NDIF_API_KEY`. For gated checkpoints (Llama, some Gemma) also export
`HF_TOKEN` — the server needs it to load the weights on your behalf.

3. Check what is deployed before writing code against a model:

<!-- test: skip -->
```python
import nnsight

print(nnsight.status())                                        # all deployments
print(nnsight.is_model_running("meta-llama/Llama-3.1-70B"))    # one model
```

`RUNNING` means ready. `COLD` means the deployment exists but is not loaded — your
job will queue until it is. If a model is not listed at all, it is not available
on that host.

## Hosts

| Setting | Where |
|---|---|
| default | `CONFIG.API.HOST` = `https://api.ndif.us` |
| environment | `NDIF_HOST` |
| per call | `remote="http://localhost:8001"` |

A URL must start with `http://` or `https://`. Local or internal deployments may
not require an API key; the public host does.

## Version skew

The server runs its own package set. A mismatch in `nnsight`, `torch`, or
`transformers` is a real source of "works locally, fails remotely" — module paths
and output types differ across `transformers` versions in particular.

<!-- test: skip -->
```python
import nnsight
print(nnsight.compare())     # local vs remote python and package versions
```

Anything flagged CRITICAL there is worth resolving before debugging your
intervention. `scripts/check_env.py --remote` in the `nnsight` skill prints this
together with your key, host, and GPU state.

## What can run inside a remote trace

Imports are restricted to a whitelist: `builtins`, `torch`, `numpy`, `einops`,
`collections`, `math`, `time`, `sympy`, `typing`, `nnterp`.

Your own code is not on it. Two options:

<!-- test: skip -->
```python
# 1. ship your module's source with the request (cloudpickle by value)
import nnsight, my_utils
nnsight.register(my_utils)          # before using anything from it

with model.trace(prompt, remote=True):
    vec = my_utils.normalize(model.transformer.h[5].output).save()
```

<!-- test: skip -->
```python
# 2. inline the helper inside the trace
with model.trace(prompt, remote=True):
    hidden = model.transformer.h[5].output
    vec = (hidden / hidden.norm(dim=-1, keepdim=True)).save()
```

Registration usually happens automatically for local modules; call it explicitly
when it doesn't (editable installs are the common case). Keep registered modules
small — their source ships with **every** request.

## Verify offline before spending a job

`remote="local"` runs the full serialize → deserialize → execute path in-process,
with your local modules hidden the way the server sees them. It catches unshipped
helpers, non-whitelisted imports, and unsaved containers with no queue and no GPU.

```python
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in the city of", remote="local"):
    top = model.output.logits[0, -1].argmax().cpu().save()

print(model.tokenizer.decode(top))
```

## Failure modes

| Symptom | Likely cause | Check |
|---|---|---|
| Job sits in the queue | model is COLD, or the queue is deep | `nnsight.status()` |
| `RemoteError` at submission | bad key, wrong host, unknown model id | `check_env.py --remote` |
| `ModuleNotFoundError` on the worker | a local helper wasn't shipped | `nnsight.register(...)`, reproduce with `remote="local"` |
| Runs locally, errors remotely | version skew, or a non-whitelisted import | `nnsight.compare()` |
| Result is missing a key | the save wasn't bound to a name | bind it: `x = ....save()` |
| Saved list is empty | saved the elements instead of the container | `nnsight.save([])` inside the trace |
| `RuntimeError` with a foreign traceback | a worker-side exception re-raised at the client | read the embedded original traceback |
| Job killed around an hour in | the per-request wall-clock limit | chunk into several sessions |

## Watching what a job actually does

`print()` inside a remote trace comes back as log lines — the cheapest possible
debugging channel:

<!-- test: remote -->
```python
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2")

with model.trace("The Eiffel Tower is in the city of", remote=True):
    hidden = model.transformer.h[5].output
    print(f"shape {tuple(hidden.shape)} dtype {hidden.dtype}")
    out = hidden[0, -1].mean().cpu().save()

print(round(float(out), 5))
```

Set `CONFIG.APP.DEBUG = True` to also log payload and result byte sizes plus each
status transition — the direct way to confirm a transfer really shrank. Disable
log streaming entirely with `CONFIG.APP.REMOTE_LOGGING = False`.

## Related

- [sessions-and-jobs.md](sessions-and-jobs.md)
- the `nnsight-debugging` skill for non-remote errors
