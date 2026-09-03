---
name: nnsight-remote
description: Write and optimize nnsight code that runs on NDIF — remote=True, model.session(remote=True), non-blocking jobs, and models too large to run locally (Llama-70B/405B, DeepSeek). Use whenever remote execution is involved even if the user did not ask for optimization: naive remote code is routinely 100x slower and thousands of times larger in transfer than it needs to be, and both causes (too many requests, too much downloaded) have mechanical fixes. Also covers NDIF setup, API keys, model availability, the import whitelist, shipping local helper code, and diagnosing remote-only failures.
---

# Remote Execution on NDIF

NDIF runs models you can't host — you write the same nnsight code and it executes
on their hardware. Two things change, and both are about the network:

1. **Every request is a queue wait plus a round trip.** A loop that issues one
   request per forward pass is dominated by that, not by the model.
2. **Every `.save()` is a download.** A `[batch, seq, vocab]` logits tensor from a
   70B model is hundreds of megabytes; the scalar you actually wanted is 8 bytes.

Everything below follows from those two facts. The general API is unchanged — see
the `nnsight` skill.

## Setup

```
python scripts/check_env.py --remote      # in the nnsight skill
```

That prints your key, host, which models are deployed and whether they are
RUNNING, and the local-vs-NDIF package diff. In code:

<!-- test: skip -->
```python
from nnsight import CONFIG
import nnsight

CONFIG.set_default_api_key("YOUR_KEY")     # from login.ndif.us; persists to disk
print(nnsight.status())                    # deployed models and their state
print(nnsight.is_model_running("meta-llama/Llama-3.1-70B"))
```

The key also comes from `NDIF_API_KEY`; the host from `NDIF_HOST` or
`CONFIG.API.HOST` (or per call: `remote="http://host:port"`). Gated checkpoints
need `HF_TOKEN` set.

A model that is **COLD** rather than RUNNING will queue until it is loaded — check
before blaming your code for a hang.

<!-- test: setup -->
```python
import time
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2")     # no dispatch — it runs remotely
prompt = "The Eiffel Tower is in the city of"
```

## Principle 1 — one job, not N

`model.session(remote=True)` wraps any number of traces into a **single** request:
one queue wait, one connection, one result download. Only the session carries
`remote=True`; inner traces inherit it.

<!-- test: remote -->
```python
with model.session(remote=True):
    means = nnsight.save([])
    for layer in range(3):
        with model.trace(prompt):
            means.append(model.transformer.h[layer].output[0, -1].mean().cpu())

print([round(float(m), 4) for m in means])
```

Inside a session, values flow between traces **without leaving the server** — this
is what makes multi-step experiments (clean run → capture → patched run) cheap:

<!-- test: remote -->
```python
with model.session(remote=True):
    with model.trace("The Eiffel Tower is in"):
        donor = model.transformer.h[5].output[:, -1]      # never downloaded

    with model.trace("The Colosseum is in"):
        model.transformer.h[5].output[:, -1] = donor
        patched = model.output.logits[0, -1].argmax().cpu().save()

print(model.tokenizer.decode(patched))
```

**Do not put `remote=True` on the inner traces** — the session already provides the
backend.

Limits worth planning around: a single request, including a whole session, is
killed after **one hour**, and **one failing trace aborts the entire session**.
Chunk long or fault-prone experiments into several back-to-back sessions; the
per-session overhead is small compared to losing an hour of compute.

## Principle 2 — download the answer, not the data

Move every reduction inside the trace. If the client's next move on a saved tensor
is `argmax`, `topk`, an index, or a difference, that belongs on the server.

<!-- test: remote -->
```python
paris = model.tokenizer(" Paris").input_ids[0]
london = model.tokenizer(" London").input_ids[0]

with model.trace(prompt, remote=True):
    logits = model.output.logits[0, -1]                       # stays on the server
    logit_diff = (logits[paris] - logits[london]).cpu().save()  # 8 bytes come back

print(round(float(logit_diff), 3))
```

The same experiment written to save raw logits transfers ~200 KB for gpt2 and
~800 MB for a 70B-class model with a long prompt — for a number you then throw
away. For tensors you genuinely need, `.detach().cpu()` before saving.

## Remote-only rules

| Rule | Why |
|---|---|
| `.save()` is the **only** channel back | A client-side list appended to inside a trace stays empty locally — build accumulators inside and save the container |
| Bind every save to a name | Results come back keyed by variable name; an unbound `.save()` is simply absent |
| An import must be present on the server | Stdlib is fine; `torch`, `numpy`, `transformers`, `accelerate`, `diffusers`, `einops`, `peft`, `nnsight` are assumed installed (`_SERVER_MODULES`, what `remote="local"` simulates). Anything else, check with `nnsight.compare()` before relying on it |
| Your own helpers must be shipped | `nnsight.register(my_module)` (cloudpickle by value), or inline the helper |
| `print()` is your debugger | It comes back as log lines — far cheaper than saving a tensor to look at it (needs `CONFIG.APP.REMOTE_LOGGING`, on by default) |
| Variables from outside the block **are** captured | Every name the block reads is pickled into the payload. They must be picklable, they arrive on CPU, and edits stay server-side unless you save them |
| Match the server's device and dtype at runtime | You cannot know either client-side — read them off an activation: `vec.to(hidden.device, hidden.dtype)` |
| `super()` needs its class named | A shipped class is recompiled outside any class body, so bare `super()` raises `super(): __class__ cell not found`. Write `super(MyClass, self).__init__()` |

An outer variable comes back if you save it *inside* the block — the save pushes
by name and replaces the client's copy, so it's a fine way to accumulate:

<!-- test: remote -->
```python
acc = []                                   # empty on the client
with model.session(remote=True):
    for text in ["a", "b", "c"]:
        with model.trace(text):
            value = model.transformer.h[0].output.sum().item()
        acc.append(value)
    nnsight.save(acc)                      # ships the server's version home
print(acc)                                 # [61.39, 46.88, 54.30]
```

Without that `nnsight.save(acc)` the appends happen to the server's copy and the
client's list stays empty.

<!-- test: remote -->
```python
with model.trace(prompt, remote=True):
    hidden = model.transformer.h[5].output
    print(f"server-side shape {tuple(hidden.shape)}, norm {hidden.norm().item():.2f}")
    summary = hidden[0, -1].mean().cpu().save()

print(round(float(summary), 5))
```

## Submitting without waiting

`blocking=False` submits and returns immediately; call the backend to poll.

<!-- test: remote -->
```python
with model.trace(prompt, remote=True, blocking=False) as tracer:
    top = model.output.logits[0, -1].argmax().cpu().save()

backend = tracer.backend
print("job", backend.job_id, backend.status)

while True:
    result = backend()          # None until COMPLETED, then the saves dict
    if result is not None:
        break
    time.sleep(0.5)

print(result.keys(), model.tokenizer.decode(result["top"]))
```

The result dict is keyed by your variable names. Sessions accept `blocking=False`
too. For an event loop, `AsyncRemoteBackend` supports `await backend` for the saves
dict and `async for update in backend` for streamed status.

## Testing without burning a job

`remote="local"` runs the whole serialize → deserialize → execute path in your own
process. What it checks is that the block survives the round trip — your helpers
ship, your saves come back. It runs the model locally, so point it at a small
stand-in rather than the model you'll actually use:

```python
local_model = TransformersModel("openai-community/gpt2", dispatch=True)

with local_model.trace(prompt, remote="local"):
    check = local_model.output.logits[0, -1].argmax().cpu().save()

print(local_model.tokenizer.decode(check))
```

Develop against a small model locally, dry-run with `remote="local"`, then switch
the model id and `remote=True`.

## Reviewing remote code

Walk these in order; the first two are usually worth orders of magnitude.

| Signal | Fix |
|---|---|
| `for` loop around `model.trace(..., remote=True)` | wrap in `model.session(remote=True)`, drop the inner `remote=True` |
| `.save()` on logits/hidden states followed by a client-side reduction | move the reduction into the trace |
| Client-scope list appended to inside a trace | build it inside, `nnsight.save()` the container |
| Optimizer or `nn.Module` built outside a remote session | move both inside the session — a client-side optimizer over a shipped module trains nothing, silently |
| `.to(model.device)` on an undispatched model | that is `meta`, and `.to("meta")` drops the data with no error — read the device off `module.device` or an activation inside the block |
| Saved CUDA tensor without `.detach().cpu()` | add it |
| A helper from the user's own file called inside a trace | `nnsight.register(...)` or inline it |
| Session that could exceed an hour | split into several sessions |
| N forward passes for N layers/heads/positions | batch with `tracer.invoke` per variant (`nnsight` skill → batching) |
| Only layers 0..L matter | `tracer.stop()` after L |

Say what the fix buys — request count and megabytes — rather than just asserting
it. If the user is prototyping and wants simple code first, agree, and flag what
to apply before they scale up.

## When it goes wrong

| Symptom | Check |
|---|---|
| Hangs in the queue | is the model RUNNING or COLD? `nnsight.status()` |
| `RemoteError` on submission | API key, host, model id, `nnsight.compare()` version skew |
| Works locally, fails remotely | non-whitelisted import, or an unregistered local helper — reproduce with `remote="local"` |
| Result missing a value | the save wasn't bound to a name, or you saved elements instead of a container |
| Worker-side exception | re-raised at the client as `RuntimeError` carrying the original traceback |

`CONFIG.APP.DEBUG = True` adds payload/result byte sizes and per-status logging —
useful for confirming a transfer really did shrink.

## References

- [references/sessions-and-jobs.md](references/sessions-and-jobs.md) — session
  semantics, chunking, non-blocking and async job management, concurrency
- [references/setup-and-troubleshooting.md](references/setup-and-troubleshooting.md)
  — keys, hosts, status and availability, the whitelist, `nnsight.register`,
  version skew

## Related skills

- `nnsight` — the API these patterns are written in
- `nnsight-debugging` — errors that are not remote-specific
