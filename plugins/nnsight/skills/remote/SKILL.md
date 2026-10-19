---
name: remote
description: Write and optimize nnsight code that runs on NDIF — remote=True, model.session(remote=True), blocking=False jobs, and models too large to run locally (Llama-70B/405B, DeepSeek). Use whenever remote execution is involved even if the user did not ask for optimization — the two costs (too many requests, too much downloaded) have mechanical fixes. Also covers NDIF setup, API keys, COLD/RUNNING model status, nnsight.register for shipping local helper code, RemoteError, and diagnosing remote-only failures.
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
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2")     # no dispatch — it runs remotely
prompt = "The Eiffel Tower is in the city of"
```

## Principle 1 — one job, not N

`model.session(remote=True)` wraps any number of traces into a **single** request:
one queue wait, one connection, one result download. Only the session carries
`remote=True`; inner traces inherit it, so do not repeat it on them.

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

Sessions have a server-side time limit, and **one failing trace aborts the entire
session** — split long or fault-prone sweeps into several back-to-back sessions.
[references/sessions-and-jobs.md](references/sessions-and-jobs.md) has the chunking
pattern, server-side dataset loading, and training loops.

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
| An import must be present on the server | Stdlib is fine; `torch`, `numpy`, `transformers`, `accelerate`, `diffusers`, `einops`, `peft` and `nnsight` are assumed installed — the same set `remote="local"` simulates. Anything else, check with `nnsight.compare()` before relying on it |
| Your own helpers must be shipped | `nnsight.register(my_module)` (cloudpickle by value), or inline the helper |
| `print()` is your debugger | It comes back as log lines — far cheaper than saving a tensor to look at it (needs `CONFIG.APP.REMOTE_LOGGING`, on by default) |
| Variables from outside the block **are** captured | Every name the block reads is pickled into the payload. They must be picklable, they arrive on CPU, and edits stay server-side unless you save them |
| Take device and dtype from an activation | You cannot know either client-side: `vec.to(hidden.device, hidden.dtype)`. Not from a module — its metadata may not reflect the serving device |
| `super()` needs its class named | A shipped class is recompiled outside any class body, so bare `super()` raises `super(): __class__ cell not found`. Write `super(MyClass, self).__init__()` |

An outer variable comes back if you save it *inside* the block — the save pushes
by name and replaces the client's copy. Without the `nnsight.save(acc)` the
appends happen to the server's copy and the client's list stays empty:

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

## Numerics on the server

The deployment chooses the dtype, not you — a server model may run in bfloat16
where your local copy is float32, and some ops (norms, softmax, reductions) can
come back in a different dtype from the layer around them. Take the dtype from an
activation inside the block rather than assuming one. bf16 has about three
significant digits, so a next-token argmax whose top two logits nearly tie can flip
between a batched and a solo run, or between the server and a local float32 model.
Compare a logit difference or a top-k margin rather than asserting one exact token.

## Submitting without waiting

`model.trace(..., remote=True, blocking=False)` (or a session with it) submits and
returns immediately; `tracer.backend()` returns `None` until the job completes,
then a dict of saves keyed by variable name — your locals are **not** filled in.
`AsyncRemoteBackend` does the same on an event loop. Both are in
[references/sessions-and-jobs.md](references/sessions-and-jobs.md#non-blocking-jobs).

## Testing without burning a job

`remote="local"` runs the whole serialize → deserialize → execute path in your own
process, with your local modules hidden the way the server sees them. It checks
that the block survives the round trip — your helpers ship, your saves come back —
with no queue and no GPU. It runs the model locally, so point it at a small
stand-in rather than the model you'll actually use:

```python
local_model = TransformersModel("openai-community/gpt2", dispatch=True)

with local_model.trace(prompt, remote="local"):
    check = local_model.output.logits[0, -1].argmax().cpu().save()

print(local_model.tokenizer.decode(check))
```

Develop against a small model locally, dry-run with `remote="local"`, then switch
the model id and `remote=True`.

A forgotten `.save()` passes (client-side mutations still come back), and
`CONFIG.APP.DEBUG` prints no payload bytes. It does catch the round trip — an unpicklable capture fails only here:

<!-- test: expect-error TypeError -->
```python
import threading
lock = threading.Lock()
with local_model.trace(prompt):                   # plain local: fine
    busy = nnsight.save(lock.locked())
with local_model.trace(prompt, remote="local"):   # TypeError: cannot pickle '_thread.lock' object
    busy = nnsight.save(lock.locked())
```

## Reviewing remote code

Walk these in order; the first two are usually worth orders of magnitude.

| Signal | Fix |
|---|---|
| `for` loop around `model.trace(..., remote=True)` | wrap in `model.session(remote=True)`, drop the inner `remote=True` |
| `.save()` on logits/hidden states followed by a client-side reduction | move the reduction into the trace |
| Client-scope list appended to inside a trace | build it inside, `nnsight.save()` the container |
| Optimizer or `nn.Module` built outside a remote session | move both inside the session — a client-side optimizer over a shipped module trains nothing, silently |
| Saved CUDA tensor without `.detach().cpu()` | add it |
| A helper from the user's own file called inside a trace | `nnsight.register(...)` or inline it |
| One long session | split into several — sessions have a time limit |
| N forward passes for N layers/heads/positions | batch with `tracer.invoke` per variant (`nnsight` skill → batching) |
| Only layers 0..L matter | `tracer.stop()` after L — it halts the whole forward, so every invoke must be done by then (it is not per-invoke) |

Say what the fix buys — request count and megabytes — rather than just asserting
it. If the user is prototyping and wants simple code first, agree, and flag what
to apply before they scale up.

## When it goes wrong

| Symptom | Check |
|---|---|
| Hangs in the queue | is the model RUNNING or COLD? `nnsight.status()` |
| `RemoteError` on submission | API key, host, model id, `nnsight.compare()` version skew |
| Works locally, fails remotely | an import the server lacks, or an unregistered local helper — reproduce with `remote="local"` |
| Result missing a value | the save wasn't bound to a name, or you saved elements instead of a container |
| Worker-side exception | arrives as `RemoteError`, whose message is the worker's traceback ending in the original type and message. `except IndexError:` never matches — catch `RemoteError` and read the last line |
| `Could not deploy this model. This model runs code from its own repository (trust_remote_code=True)…` | the model is not deployed and your API key lacks the `trust_remote_code` (or `trusted`) tag — ask an NDIF admin to deploy it or grant the tag |
| RUNNING in `nnsight.status()`, yet your request deploys a new copy or waits | your `task=` / `trust_remote_code=` differ from the deployment's; both are in the model key, so match what `status()` shows |

`CONFIG.APP.DEBUG = True` adds payload/result byte sizes and per-status logging —
useful for confirming a transfer really did shrink.

## References

- [references/sessions-and-jobs.md](references/sessions-and-jobs.md) — chunking,
  server-side datasets, training a probe or LoRA in a session, non-blocking and
  async jobs, choosing between them
- [references/setup-and-troubleshooting.md](references/setup-and-troubleshooting.md)
  — hosts, version skew, `nnsight.register` versus inlining, catching `RemoteError`

## Related skills

- `nnsight` — the API these patterns are written in
- `debugging` — errors that are not remote-specific
