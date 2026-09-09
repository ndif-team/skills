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
| Everything the block reads is pickled into the request | keep closed-over objects small and picklable; they arrive on the CPU |
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

## Load the data on the server

Upload the minimum. A HuggingFace `Dataset` is memory-mapped, so pickling one puts
an arrow file **path** in the payload that only exists on your machine, and the
worker fails with `FileNotFoundError`. Import `datasets` inside the session and the
download happens server-side instead:

<!-- test: remote -->
```python
with model.session(remote=True):
    from datasets import load_dataset

    rows = load_dataset("nyu-mll/glue", "sst2", split="train[:40]")
    tally = nnsight.save({"n": 0, "hits": 0})

    for start in range(0, len(rows), 10):
        batch = rows[start : start + 10]
        with model.trace(batch["sentence"]):
            top = model.output.logits[:, -1].argmax(-1)
        tally["n"] += len(top)
        tally["hits"] += int((top > 0).sum())

print(tally)
```

Slice in the split string (`train[:40]`) rather than after loading — it is the
difference between the server materialising forty rows and the full 67k.

Download the minimum too: reduce inside the block and save the summary, not the
activations you reduced. A client-side tensor you fold in arrives on the CPU in
float32, so take device and dtype off an activation:

<!-- test: remote -->
```python
probe = torch.randn(768)

with model.session(remote=True):
    scores = nnsight.save([])
    for text in ["a great film", "utter garbage", "i loved it"]:
        with model.trace(text):
            hidden = model.transformer.h[6].output
            scores.append((hidden[0, -1] @ probe.to(hidden.device, hidden.dtype)).item())

print([round(s, 2) for s in scores])
```

## Training inside a session

The optimizer loop goes in the session too, so a 500-step run is one job rather
than 500. Parameters have to be created on the module's device — `torch.randn`
gives CPU float32 even when the code is running next to the weights — and only
plain tensors can come back, so hand back the trained weights rather than the
adapter that holds them.

<!-- test: remote -->
```python
module = model.transformer.h[-1].mlp

with model.session(remote=True):
    # Defined inside the block, so the class travels as source. Defined outside,
    # it is pickled by reference and the server has to be able to import its
    # module (see `nnsight.register`).
    class LoRA(torch.nn.Module):
        def __init__(self, module, dim, rank):
            # Named, not bare: a shipped class is recompiled outside any class
            # body, so `super()` has no __class__ cell to read.
            super(LoRA, self).__init__()
            self.module = module
            device = module.device
            self.WA = torch.nn.Parameter(torch.randn(dim, rank).to(device))
            self.WB = torch.nn.Parameter(torch.zeros(rank, dim).to(device))

        def __call__(self):
            hidden = self.module.input
            delta = torch.matmul(torch.matmul(hidden.to(self.WA.dtype), self.WA), self.WB)
            self.module.output = delta.to(hidden.dtype) + self.module.output

        def parameters(self):
            return [self.WA, self.WB]

    adapter = LoRA(module, 768, 4)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)

    for _ in range(3):
        with model.trace(prompt):
            adapter()
            loss = -model.output.logits[0, -1].log_softmax(-1)[6342]
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"loss {loss.item():.3f}")

    trained_WA = adapter.WA.detach().cpu().save()
    trained_WB = adapter.WB.detach().cpu().save()

print(trained_WA.shape, trained_WB.shape)
```

Rebuild the adapter from those weights to use it later. Two things that look fine
and aren't: `output[:] = ...` instead of `output = ...` breaks the autograd graph
("one of the variables needed for gradient computation has been modified"), and
`.save()` on `self.WA` inside `__init__` returns nothing, because saves come back
keyed by *variable name* and an attribute has none.

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
