# Sessions and Job Management

How to structure a remote experiment so it costs one queue wait instead of many,
and how to run jobs without blocking.

<!-- test: setup -->
```python
import time
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2")
prompt = "The Eiffel Tower is in the city of"
```

## What a session is

`model.session(remote=True)` captures its whole body — traces, loops, ordinary
Python — serializes it once, and runs it on the server as a single job. You pay
queue latency once and download once.

<!-- test: remote -->
```python
with model.session(remote=True):
    layer_means = nnsight.save([])
    for layer in range(4):
        with model.trace(prompt):
            layer_means.append(model.transformer.h[layer].output[0, -1].mean().cpu())

print([round(float(m), 4) for m in layer_means])
```

Three rules:

- `remote=True` on the **session**, never on the inner traces.
- Values move between traces for free; only `.save()` sends something to you.
- The loop is ordinary Python captured with the block — it is not a server-side
  `for` in any special sense, it just runs there.

## The pattern this exists for

A patching experiment is three traces that must share a tensor. Locally you would
`.save()` it; remotely that would mean downloading an activation and re-uploading
it. In a session it never leaves the server:

<!-- test: remote -->
```python
with model.session(remote=True):
    with model.trace("The Eiffel Tower is in"):
        donor = model.transformer.h[5].output[:, -1]           # stays remote

    with model.trace("The Colosseum is in"):
        baseline = model.output.logits[0, -1].argmax().cpu().save()

    with model.trace("The Colosseum is in"):
        model.transformer.h[5].output[:, -1] = donor
        patched = model.output.logits[0, -1].argmax().cpu().save()

print(model.tokenizer.decode(baseline), "->", model.tokenizer.decode(patched))
```

Two small scalars come back; the `[1, 768]` (or `[1, 8192]`) activation never
crosses the network.

## Limits to design around

| Limit | Consequence |
|---|---|
| One hour per request, session included | chunk long experiments into back-to-back sessions |
| One failing trace aborts the whole session | isolate risky work; don't put a 200-condition sweep in one job |
| Outer-scope variables are unavailable inside | construct everything inside the block |
| Sessions cut queue and transport time, not GPU time | a 5-minute computation is still 5 minutes |

Chunking looks like this — each session is an independent job, so a failure costs
one chunk rather than everything:

<!-- test: remote -->
```python
def sweep(layers):
    with model.session(remote=True):
        out = nnsight.save([])
        for layer in layers:
            with model.trace(prompt):
                model.transformer.h[layer].output[:, -1, :] = 0
                out.append(model.output.logits[0, -1].max().cpu())
    return [float(x) for x in out]

results = sweep(range(0, 3)) + sweep(range(3, 6))
print(len(results))
```

Inside a single session you can still batch with `tracer.invoke(...)` — that
collapses a sweep into one *forward pass* as well as one request. Combining both
is the fastest form.

## Non-blocking jobs

`blocking=False` submits and returns; the backend polls.

<!-- test: remote -->
```python
with model.trace(prompt, remote=True, blocking=False) as tracer:
    top = model.output.logits[0, -1].argmax().cpu().save()

backend = tracer.backend
print("submitted", backend.job_id, backend.status)

while True:
    result = backend()               # None until COMPLETED
    if result is not None:
        break
    time.sleep(0.5)

print(result.keys(), model.tokenizer.decode(result["top"]))
```

The result is a dict keyed by the **variable names** you saved — with
`blocking=False` the values are not pushed back into your locals, because the
`with` block exited long before the job finished. Read them out of the dict.

`backend()` raises `RemoteError` if the job errored, and returns `None` both while
running and before the first status lands.

## Async jobs

To wait on an event loop instead of polling — or to run several jobs concurrently
— use `AsyncRemoteBackend`. Submission is still synchronous; only the wait is
async.

<!-- test: skip -->
```python
import asyncio
from nnsight.intervention.backends.remote import AsyncRemoteBackend

async def run(prompt):
    backend = AsyncRemoteBackend(model.to_model_key())
    with model.trace(prompt, backend=backend):
        out = model.output.logits[0, -1].argmax().cpu().save()
    result = await backend                       # the saves dict
    return model.tokenizer.decode(result["out"])

async def main():
    return await asyncio.gather(
        run("The Eiffel Tower is in"),
        run("The Colosseum is in"),
    )

asyncio.run(main())
```

`async for update in backend` streams raw status updates instead, ending with the
saves dict as the final item. That form does not raise on `ERROR` — an error
update simply ends the stream, so inspect it yourself.

## Choosing

| Situation | Use |
|---|---|
| One trace | `model.trace(..., remote=True)` |
| Several related traces | `model.session(remote=True)` |
| Long job, want your process back | `blocking=False` + poll |
| Many jobs at once, or async app | `AsyncRemoteBackend` |
| Developing / testing the remote path | `remote="local"` |

## Related

- [setup-and-troubleshooting.md](setup-and-troubleshooting.md)
- the `nnsight` skill → batching, for collapsing sweeps into one forward pass
