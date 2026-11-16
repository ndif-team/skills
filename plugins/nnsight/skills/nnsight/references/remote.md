# Remote Execution on NDIF

NDIF hosts models too large to run yourself. The trace is written the same way
and runs on their hardware. Two costs are new, and both are the network:

1. **Every request is a queue wait and a round trip.** A loop of remote traces
   pays that once per iteration.
2. **Every save is a download.** The logits of a 70B model over a long prompt are
   hundreds of megabytes; the number you wanted is eight bytes.

The examples need an NDIF API key and a deployed model. Each was also run with
`remote="local"`, the dry run [below](#test-without-a-job), which is how to check
your own code.

## Setup

```python
import torch
import nnsight
from nnsight import CONFIG, TransformersModel

CONFIG.set_default_api_key("YOUR_KEY")      # once per machine; from login.ndif.us
print(nnsight.status())                     # deployed models and their state

model = TransformersModel("openai-community/gpt2", task="text-generation")   # no dispatch: no local weights
prompt = "The Eiffel Tower is in the city of"
```

- The key also comes from the `NDIF_API_KEY` environment variable or the
  `nnsight login` command; the host from `NDIF_HOST`, `CONFIG.API.HOST`, or
  per call as `remote="http://host:port"`.
- Without `dispatch=True` the model is built on the `meta` device: every module
  path exists and no weights are loaded, so a laptop can write a trace for a 70B
  model.
- `python scripts/check_env.py --remote` prints key, host, deployed models and
  the local-versus-server package diff in one go.
- **A model that is not RUNNING queues until it is loaded.** Check
  `nnsight.is_model_running(repo_id)` before blaming your code for a hang. The
  deployment is keyed by repo id, revision and `task` (and `trust_remote_code`
  when set), so a `task=` that differs from the one `status()` shows is a
  different model: it waits or deploys a second copy.

## One request, not N

`model.session(remote=True)` wraps any number of traces in a single request.
Only the session carries `remote=True`. Values flow between its traces without
leaving the server:

```python
with model.session(remote=True):
    with model.trace("The Eiffel Tower is in"):
        donor = model.transformer.h[5].output[:, -1]            # never downloaded
    with model.trace("The Colosseum is in"):
        model.transformer.h[5].output[:, -1] = donor
        patched = model.output.logits[0, -1].argmax().cpu().save()

print(model.tokenizer.decode(patched))
```

A sweep over N variants of one prompt is one trace with N invokes
([SKILL.md](../SKILL.md)): the session saves queue time, the invokes save GPU
time. Sessions have a server-side time limit, and one failing trace aborts the
whole session, so split a long sweep into chunks. Chunking, datasets loaded on the
server, training loops and non-blocking jobs:
[remote-sessions-and-jobs](remote-sessions-and-jobs.md).

## Download the answer, not the data

Reduce inside the block. If the next thing you would do with a saved tensor is
`argmax`, `topk`, an index or a difference, do it on the server:

```python
paris = model.tokenizer.encode(" Paris", add_special_tokens=False)[0]
london = model.tokenizer.encode(" London", add_special_tokens=False)[0]

with model.trace(prompt, remote=True):
    logits = model.output.logits[0, -1]                          # stays on the server
    logit_diff = (logits[paris] - logits[london]).cpu().save()   # one number comes back

print(round(float(logit_diff), 3))
```

For a tensor you do need, `.detach().cpu()` before saving.

## Rules that only bite remotely

| Rule | Why |
|---|---|
| A save is the only way back | a list made outside the block and appended to inside stays empty on the client. Create it inside and save it, or save the outer name inside the block (`nnsight.save(acc)`) |
| Bind every save to a name | results come back keyed by variable name |
| Variables from outside the block are shipped | everything the block reads is pickled into the request: it must be picklable, it arrives on the CPU in its own dtype, and changes to it stay on the server |
| Take device and dtype from an activation | `vec.to(hidden.device, hidden.dtype)`. The deployment picks the dtype, often bfloat16, so near-tied logits can order differently from a local float32 run |
| Imports must exist on the server | the standard library, `torch`, `numpy`, `transformers`, `einops` and nnsight's dependencies do. Check anything else with `nnsight.compare()` |
| Your own helpers must be shipped | see [below](#shipping-your-own-code) |
| `print()` is the debugger | output comes back as log lines, far cheaper than saving a tensor to look at it |
| An optimizer or module built outside a session trains nothing | build both inside the session |

## Test without a job

`remote="local"` runs the whole serialize, ship, deserialize and execute path in
your own process, with your local modules hidden as the server would have them.
Develop on a small model, dry-run it, then switch the repo id and `remote=True`:

```python
local = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)

with local.trace(prompt, remote="local"):
    check = local.output.logits[0, -1].argmax().cpu().save()

assert local.tokenizer.decode(check) == " Paris"
```

It catches what does not survive the trip, such as an unpicklable object the
block reads (`TypeError: cannot pickle '_thread.lock' object`) or a helper module
that was not shipped. It does not catch a forgotten save on an outer list: in one
process the appends still land.

In a `remote="local"` **session**, a blank line between or inside the inner
traces makes a later inner trace fail with `WithBlockNotFoundError: nnsight found
no with statement at ...`. Write session bodies without blank lines.

## Shipping your own code

A function defined in your script, or in a module of your working directory, is
registered automatically and travels with the request. Call
`nnsight.register(my_module)` before the trace for a module nnsight does not see
as local, most often an editable install (`pip install -e .`):

```python
import my_utils                      # illustrative: your own module
nnsight.register(my_utils)

with model.trace(prompt, remote=True):
    vec = my_utils.normalize(model.transformer.h[5].output).save()
```

The module's source ships with every request, so keep it small, and whatever it
imports must exist on the server. For a one-line helper, inline it in the block.

## When a remote run fails

Whatever the block raises on the server reaches you as `RemoteError`, never as
its own class, so `except IndexError:` around a remote trace does not fire. Catch
`RemoteError` and read the original off the end of its message:

```python
from nnsight.intervention.backends.remote import RemoteError

try:
    with model.trace(prompt, remote=True):
        past_the_end = model.transformer.h[5].output[0, 99].save()    # 10 tokens
except RemoteError as error:
    original = str(error).strip().splitlines()[-1]

print(original)      # IndexError: index 99 is out of bounds for dimension 1 with size 10
```

| Symptom | Check |
|---|---|
| waits in the queue | is the model RUNNING? `nnsight.status()` |
| `RemoteError` at submission | API key, host, repo id, version skew (`nnsight.compare()`) |
| works locally, fails remotely | an import the server lacks or an unshipped helper; reproduce with `remote="local"` |
| a saved value is missing | the save was not bound to a name, or elements were saved in place of their container |
| `Could not deploy this model. This model runs code from its own repository ...` | the model is not deployed and your key may not deploy `trust_remote_code` models; ask the NDIF admins |
| out of memory, or you want to know what a job cost | `tracer.backend.meta` after the block: `runtime`, `max_mem_by_gpu`, and on an out-of-memory failure `alloc_shortfall_by_gpu`. Read with `.get()` |

`CONFIG.APP.DEBUG = True` logs payload and result sizes, which confirms that a
download really shrank.

## Reviewing remote code

| Signal | Fix |
|---|---|
| a `for` loop around `model.trace(..., remote=True)` | one `model.session(remote=True)`, no `remote=` on the inner traces |
| a saved logits or hidden-state tensor reduced on the client | move the reduction into the block |
| N traces for N layers, heads or positions of one prompt | one trace, one invoke per variant |
| a saved tensor without `.detach().cpu()` | add it |
| only layers up to L matter | `tracer.stop()` after L (it stops every invoke) |

Say what a fix buys in requests and megabytes.
