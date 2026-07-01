---
name: remote
description: Optimize nnsight code for remote execution on NDIF. Use whenever the user is writing, reviewing, or refactoring nnsight code that runs against NDIF (any usage of `remote=True`, `model.session`, or models too large to run locally like Llama-70B/405B), even if they don't ask for "optimization" explicitly — naive remote code is often 100x slower than it needs to be, and almost all of that comes from two mistakes: too many requests and too much downloaded.
---

# Remote Execution Optimization

When nnsight code runs against NDIF, every forward pass becomes a network round-trip and every `.save()` becomes a download. The same code that's fine locally can be 100x slower remotely if it isn't structured to minimize requests and result size.

This skill covers the two principles that are *specific* to remote execution: bundling traces into a single request, and keeping downloads small. The other techniques that compound with remote performance (caching activations in one pass, batching with `tracer.invoke()`, skipping layers with `.skip()`) are general nnsight intervention patterns — see the `nnsight-basics` skill. They apply locally too, but they amplify remotely because every forward pass you eliminate is also a request you don't have to wait on.

## When this matters

A piece of nnsight code is remote-execution code if any of these are true:
- It uses `remote=True` on `trace`, `session`, or `generate`
- It targets a model too large to fit on the user's hardware (e.g. Llama-3.1-70B, Llama-3.1-405B, DeepSeek-V3)
- The user has mentioned NDIF, `login.ndif.us`, or an NDIF API key

## Principle 1: Bundle traces in a session

Each NDIF request pays full network + queue latency. A loop that issues one request per forward pass is dominated by that overhead, not by the model.

`model.session(remote=True)` opens a single NDIF request inside which any number of `model.trace(...)` blocks run. Only the outer session needs `remote=True`; the inner traces inherit it.

```python
with model.session(remote=True):
    results = list().save()
    for layer in range(n_layers):
        with model.trace(prompt):  # no remote=True here
            ...
            results.append(model.output.logits)
```

Two things to know about session scope:
- Variables defined inside one trace are reachable from later traces in the same session — you don't need `.save()` to pass values between traces, only to return them to the client.
- Code outside the trace blocks (loops, list construction, light tensor ops) runs on the remote worker, not the client.

**Gotcha — mutating local objects from inside a trace doesn't work.** A list defined in client scope and `.append()`'d inside a remote trace gets populated on the server but stays empty locally. Always create accumulators inside the trace and `.save()` them:

```python
# WRONG: results stays empty locally even though appends happen on the server
results = []
with model.session(remote=True):
    for layer in range(n_layers):
        with model.trace(prompt):
            results.append(model.output.logits.save())
assert len(results) == n_layers  # FAILS — list is empty locally

# RIGHT: create inside, save the container itself
with model.session(remote=True):
    results = list().save()
    for layer in range(n_layers):
        with model.trace(prompt):
            results.append(model.output.logits)
```

**Wall-clock ceiling.** A single NDIF request — including the entire body of a `model.session` — is killed after one hour. Sessions are powerful enough that it's tempting to pack a whole experiment into one; resist that when the experiment is long-running. If a session is approaching the limit, chunk it into multiple back-to-back sessions rather than risking the whole job. The per-session queue overhead is small compared to losing an hour of compute.

**When reviewing:** any remote code with a Python `for` loop around `model.trace(..., remote=True)` is leaving large speedups on the table. Wrap it in a session and drop `remote=True` from the inner traces. Also flag any local-scope list/dict that's mutated inside the trace, and estimate whether the resulting session fits inside the 1-hour limit.

## Principle 2: Save only what you need

Every `.save()`'d tensor is serialized and sent back to the client. A `[batch, seq, vocab]` logits tensor for a 70B-class model is hundreds of MB; a single scalar metric is bytes.

If the only thing the local code does with a saved tensor is reduce it (top-k, argmax, indexing into target tokens, computing a logit-difference, etc.), move that reduction inside the trace and save the reduced result instead.

```python
# Bad: downloads full logits, then reduces locally
with model.trace(prompt):
    logits = model.output.logits.save()
metric = stats_from_logits(logits, entities)

# Good: reduces remotely, downloads only the metric
with model.trace(prompt):
    logits = model.output.logits
    metric = stats_from_logits(logits, entities).save()
```

This is the cheapest principle to apply — usually a one-line change — and easily reduces result size by orders of magnitude (the case study this skill is drawn from went from 3.11 GB to 113 KB, a ~27,000x reduction).

For tensors you genuinely do need to download, detach and move to CPU before saving so the server doesn't ship a CUDA tensor with extra metadata:

```python
hidden = model.model.layers[-1].output[0].detach().cpu().save()
```

**When reviewing:** scan every `.save()`. If the saved tensor is large (logits, hidden states, attention scores) and the next thing in client code is a reduction, push the reduction in. If a saved tensor is genuinely needed, make sure it's `.detach().cpu()`'d first.

## Reviewing remote code

Walk these signals in order — sessions and save-pruning are the highest-leverage fixes for remote-specific overhead:

| Signal | Apply |
| --- | --- |
| `for` loop around `model.trace(..., remote=True)` | Wrap in `model.session(remote=True)`, drop inner `remote=True` |
| Local-scope list/dict appended-to inside a remote trace | Move construction inside the trace, `.save()` the container |
| `.save()` on a large tensor followed by a reduction in client code | Move the reduction inside the trace |
| `.save()` on a CUDA tensor without `.detach().cpu()` | Add `.detach().cpu()` before `.save()` |
| Session approaching ~1 hour of work | Split into multiple sessions |

After applying these, look at the *intervention* pattern itself for further wins — caching, batching, and skipping (covered in `nnsight-basics`) often deliver another large multiplier on top of session bundling.

## Implementing remote code from scratch

Default structure:

1. Wrap the whole experiment in `with model.session(remote=True):`.
2. Build any caching forward passes first, with the layer loop inside the trace (see `nnsight-basics` → "Caching Activations in One Pass").
3. For sweeps over heads / positions / coefficients, use `tracer.invoke()` per variant inside one trace (see `nnsight-basics` → "Batched Processing with Invokers").
4. Reduce metrics inside the trace; save scalars or small tensors, not raw logits or hidden states.
5. If you're patching at layer L while computing layers 0..L-1 from scratch each pass, pre-cache them and use `.skip()` (see `nnsight-basics` → "Module Skipping").

A minimal remote-shaped trace looks like:

```python
with model.session(remote=True):
    metrics = list().save()
    with model.trace() as tracer:
        for variant in sweep:
            with tracer.invoke(prompt):
                # apply intervention parameterized by `variant`
                ...
                logits = model.output.logits
                metrics.append(reduce_to_scalar(logits))
```

One request, one forward pass per layer of intervention, only the small `metrics` list comes back.

## Operational caveats

A few things about NDIF that aren't optimizations but will silently break or surprise users if you don't account for them when writing remote code.

**Whitelisted modules only.** Code inside a remote trace can only import from a whitelist: `builtins`, `torch`, `numpy`, `einops`, `collections`, `math`, `time`, `sympy`, `typing`, `nnterp`. If your trace calls a helper from one of the user's own files (e.g. a `stats_from_logits` function defined in `utils.py`), serialization will fail or the helper won't resolve on the server. Two fixes:

```python
# Option 1: register the user's module so cloudpickle ships its source with the request
import cloudpickle
cloudpickle.register_pickle_by_value(my_utils_module)

# Option 2: inline the helper inside the session/trace where it's used
```

When reviewing remote code that calls helpers, check that they're either inlined or registered.

**Print for debugging, don't save.** `print(...)` inside a remote trace shows up in the request logs as `LOG` lines. Use it for sanity checks — shapes, scalar values, branch markers — instead of `.save()`'ing intermediate values just to inspect them.

```python
with model.trace(prompt, remote=True):
    h = model.model.layers[5].output[0]
    print(f"layer 5 hidden mean: {h.mean()}, shape: {h.shape}")
```

**`.generate()` remote uses `model.generator.output`.** For remote autoregressive generation, `tracer.result.save()` does not work — use `model.generator.output.save()` to retrieve the generated tokens.

```python
with model.generate("Hello", max_new_tokens=5, remote=True) as tracer:
    output = model.generator.output.save()  # not tracer.result
```

**Setup that has to be in place.** `CONFIG.set_default_api_key("...")` (key from `login.ndif.us`), `HF_TOKEN` env var for gated models, Python 3.12, `nnsight >= 0.5.13`. If a user's first remote call is failing for non-obvious reasons, check these before assuming it's their intervention code.

## What to push back on

If a user asks for help with remote code that doesn't apply these principles, suggest the optimizations even if they didn't ask. The cost of a naive remote loop isn't subtle — it's measured in dollars of compute and minutes-to-hours of wall clock per experiment, and the fixes are mostly mechanical. Show them the before/after request count and download size; that's usually persuasive.

If they push back because they're prototyping and want simple code first, that's fine — agree, but flag what to apply when they scale up.
