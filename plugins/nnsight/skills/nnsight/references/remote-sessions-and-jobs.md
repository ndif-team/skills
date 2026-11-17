# Remote Sessions and Jobs

Structuring a larger remote experiment: chunked sweeps, data loaded on the
server, training inside a session, and jobs you do not wait for. Read
[remote](remote.md) first; its rules apply to everything here.

The examples need an NDIF key and a deployed model. The session examples were
run with `remote="local"`; the job examples were checked against the client's
source and are not run here.

```python
import time
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation")
prompt = "The Eiffel Tower is in the city of"
```

## Chunk a sweep

Each session is one job with its own time limit, and one failing trace aborts
the session. One session per chunk means a failure costs a chunk:

```python
def sweep(layers):
    with model.session(remote=True):
        with model.trace() as tracer:
            out = nnsight.save({})
            for layer in layers:
                with tracer.invoke(prompt):
                    model.transformer.h[layer].output[:, -1, :] = 0
                    out[layer] = model.output.logits[0, -1].max().cpu()
    return {layer: float(value) for layer, value in out.items()}

results = sweep(range(0, 6)) | sweep(range(6, 12))
assert sorted(results) == list(range(12))
```

## Load the data on the server

A HuggingFace `Dataset` is memory-mapped: pickling one ships a file **path** that
exists only on your machine, and the server raises `FileNotFoundError`. Import
`datasets` inside the session and the download happens there. Slice in the split
string so the server materializes only those rows:

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
```

## Train inside a session

Put the optimizer loop in the session, so 500 steps are one job. Build the
parameters **inside** the session, on the device and dtype of an activation; a
client-side optimizer over shipped parameters trains nothing. Only tensors come
back, so save the trained weights, not the object holding them.

```python
labels = [1.0, 0.0, 1.0, 0.0]
texts = ["a great film", "utter garbage", "i loved it", "awful, just awful"]

with model.session(remote=True):
    rows = []
    for text in texts:
        with model.trace(text):
            hidden = model.transformer.h[6].output[0, -1].detach().float()
        rows.append(hidden)
    features = torch.stack(rows)
    target = torch.tensor(labels, device=features.device)
    weight = torch.zeros(768, device=features.device, requires_grad=True)
    optimizer = torch.optim.SGD([weight], lr=1e-3)
    for _ in range(20):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(features @ weight, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    final_loss = nnsight.save(loss.item())
    probe = weight.detach().cpu().save()

assert tuple(probe.shape) == (768,) and final_loss < 0.7
```

An adapter that writes into the model trains the same way, with a trace per step.
Read the device from an activation, replace the output (a write into it breaks
the graph), and step the optimizer between traces:

```python
module = model.transformer.h[-1].mlp
rome = model.tokenizer.encode(" Rome", add_special_tokens=False)[0]

with model.session(remote=True):
    with model.trace(prompt):
        device = module.output.device
    WA = torch.nn.Parameter(torch.randn(768, 4, device=device) * 0.01)
    WB = torch.nn.Parameter(torch.zeros(4, 768, device=device))
    optimizer = torch.optim.AdamW([WA, WB], lr=1e-2)
    losses = nnsight.save([])
    for _ in range(3):
        with model.trace(prompt):
            delta = module.input @ WA @ WB               # input before output
            module.output = module.output + delta
            loss = -model.output.logits[0, -1].log_softmax(-1)[rome]
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    trained_WA = WA.detach().cpu().save()
    trained_WB = WB.detach().cpu().save()

assert losses[-1] < losses[0] and tuple(trained_WA.shape) == (768, 4)
```

A save on an attribute (`self.WA.save()` inside a class) returns nothing: saves
come back by variable name. Save the weights under names at the end.

## Submit without waiting

`blocking=False` submits the job and returns. `tracer.backend` is the job: call
it to poll. It returns `None` until the job has completed, then a dict of the
saves **keyed by variable name**. Your local names are not filled in, because
the `with` block exited long before the job finished.

```python
with model.trace(prompt, remote=True, blocking=False) as tracer:
    top = model.output.logits[0, -1].argmax().cpu().save()

backend = tracer.backend
print(backend.job_id, backend.status)

while (result := backend()) is None:      # one HTTP request per call
    time.sleep(1)

print(model.tokenizer.decode(result["top"]))
```

- `backend()` raises `RemoteError` when the job failed. Status changes only when
  you poll, and the block's `print()` output is not streamed.
- A session takes `blocking=False` the same way.
- To fetch a job from another process, keep `backend.job_id` and build
  `RemoteBackend(model.to_model_key(), blocking=False, job_id=job_id)` from
  `nnsight.intervention.backends.remote`; calling it polls. Fetch results
  promptly, the server does not keep them forever.

## Await jobs on an event loop

`AsyncRemoteBackend` waits without blocking a thread, which lets several jobs
run at once. Submission is still synchronous:

```python
import asyncio
from nnsight.intervention.backends.remote import AsyncRemoteBackend

async def run(text):
    backend = AsyncRemoteBackend(model.to_model_key())
    with model.trace(text, backend=backend):
        out = model.output.logits[0, -1].argmax().cpu().save()
    result = await backend                       # the saves dict
    return model.tokenizer.decode(result["out"])

async def main():
    return await asyncio.gather(run("The Eiffel Tower is in"), run("The Colosseum is in"))

print(asyncio.run(main()))
```

`async for update in backend` yields the raw status updates and the saves dict
last. That form does not raise on a failed job; the stream just ends.

## Choosing

| Situation | Use |
|---|---|
| one trace | `model.trace(..., remote=True)` |
| several related traces, a loop, a training run | `model.session(remote=True)` |
| a long job, and you want your process back | `blocking=False`, then poll |
| many jobs at once, or an async application | `AsyncRemoteBackend` |
| checking that the code survives the trip | `remote="local"` |
