# NDIF Setup and Troubleshooting

Key, status and `remote="local"` setup are in the remote skill's SKILL.md; this
page covers the rest.

<!-- test: setup -->
```python
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2")
```

## Hosts and config

| Setting | Where |
|---|---|
| default | `CONFIG.API.HOST` = `https://api.ndif.us` |
| environment | `NDIF_HOST` |
| per call | `remote="http://host:port"` |

A URL must start with `http://` or `https://`. Self-hosted deployments may not
require an API key; the public host does. `CONFIG.set_default_api_key(...)`
persists to `~/.config/nnsight/config.yaml`.

## Version skew

The server runs its own package set. A mismatch in `nnsight`, `torch`, or
`transformers` is a real source of "works locally, fails remotely" — module paths
and output types differ across `transformers` versions in particular.
`nnsight.compare()` prints local versus remote python and package versions for
every package the server has — one missing from the table is not installed there.
Resolve anything flagged
CRITICAL before debugging your intervention.

## Shipping your own code

A helper from your own module is not installed on the server. Two options:

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
small — their source ships with **every** request. Reproduce a `ModuleNotFoundError`
on the worker with `remote="local"`.

## Catching a worker-side exception

A worker-side exception does not keep its class across the network. Whatever the
block raised — `IndexError`, `ValueError`, `OutOfOrderError` — reaches you as
`RemoteError`, so `except IndexError:` around a remote trace never fires. Catch
`RemoteError` and read the original off the end of its message:

<!-- test: remote -->
```python
from nnsight.intervention.backends.remote import RemoteError

try:
    with model.trace("The Eiffel Tower is in the city of", remote=True):
        past_the_end = model.transformer.h[5].output[0, 99].save()   # 10 tokens
except RemoteError as error:
    original = str(error).strip().splitlines()[-1]

print(original)      # IndexError: index 99 is out of bounds for dimension 1 with size 10
assert original.startswith("IndexError")
```

## Related

- [sessions-and-jobs.md](sessions-and-jobs.md)
- the `debugging` skill for non-remote errors
