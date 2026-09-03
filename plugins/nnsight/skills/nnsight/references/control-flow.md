# Control Flow: skip, stop, session, edit

Four ways to change *what runs*, rather than what a value is.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
```

## module.skip(replacement) — don't run this module

The module's forward never executes; `replacement` becomes its output.

```python
with model.trace(prompt):
    layer0 = model.transformer.h[0].output
    model.transformer.h[1].skip(layer0)          # layer 1 never runs
    logits = model.output.logits.save()

print(model.tokenizer.decode(logits[0, -1].argmax()))
```

Feeding a module its own input turns it into a pass-through — the identity
ablation of a whole block. This works for a residual-stream block, whose input and
output are both the `(batch, seq, hidden)` hidden state; it is wrong for a module
that returns something shaped differently from what it takes:

```python
with model.trace(prompt):
    model.transformer.h[6].skip(model.transformer.h[6].input)   # block 6 is a no-op
    passthrough = model.output.logits[0, -1].argmax().save()

with model.trace(prompt):
    baseline = model.output.logits[0, -1].argmax().save()

assert model.tokenizer.decode(passthrough) == " London"     # baseline is " Paris"
assert passthrough.item() != baseline.item()
```

Zeroing the output instead is a different ablation — the block contributes nothing
rather than passing its input along:

```python
with model.trace(prompt):
    model.transformer.h[5].mlp.skip(torch.zeros_like(model.transformer.h[5].mlp.input))
    logits = model.output.logits.save()

assert logits.shape[-1] == model.config.vocab_size
```

Uses: ablate a sublayer, splice in a cached activation, or route around layers you
have already computed (skipping layers 0..L-1 with a cache saves real time on deep
models).

Rules:
- The replacement must match the module's real output in **structure, shape, dtype
  and device** — a tensor for a GPT-2 block, a tuple for a module that returns one.
  A mismatch is caught by the model, not by nnsight, so it arrives as a bare torch
  error from inside the next forward (`expected scalar type Double but found
  Float`, `Expected all tensors to be on the same device`) that names neither
  `skip` nor your module.
- A skipped module's submodules never run, so reading their `.output` is out of
  order.
- In a batched trace, a skip must be applied in **every** invoke or none.
- A skip is one-shot per module call; across generation steps, re-skip per step
  (`tracer.iter[...]`) or make it persistent with `edit`.

## tracer.stop() — abandon the rest of the pass

```python
with model.trace(prompt) as tracer:
    early = model.transformer.h[2].output.save()     # save BEFORE stopping
    tracer.stop()

print(early.shape)        # layers 3+ never ran
```

Everything after `stop()` in that block is unreachable, so save first. `stop()`
ends the whole run — for one module use `skip`. In generation it stops the entire
decode loop, not just the current step:

```python
with model.generate(prompt, max_new_tokens=20) as tracer:
    picks = nnsight.save([])
    for step in tracer.all():
        picks.append(model.output.logits[0, -1].argmax(dim=-1))
        if len(picks) == 3:
            tracer.stop()

assert len(picks) == 3          # the run ended at step 2, not step 19
```

Two things a stop takes with it:

- **The run's result is gone.** A stopped run never returns one, so
  `tracer.result.save()` after the `stop()` is unreachable, and from a separate
  empty invoke it raises `OutOfOrderError: 'result.i0'`. Save activations, not
  results.
- **Saving `tracer.result` *before* the `stop()` defeats it.** The worker parks on
  the result until the run finishes, so the whole forward runs and the stop fires
  afterwards — with a full logits tensor in hand as evidence it "worked".

```python
with model.trace(prompt) as tracer:
    early = model.transformer.h[2].output.save()
    result = tracer.result.save()      # parks here until the run is over
    tracer.stop()                      # too late to save anything

assert result.logits.shape[1] == len(model.tokenizer.encode(prompt))
```

Early stopping is the cheapest optimization in interpretability: if your metric
only needs layer 5 of a 80-layer model, stop there.

## model.session() — several traces, one scope

Inside a session, values flow from one trace to the next without `.save()`; only
what you save escapes the session.

```python
with model.session():
    with model.trace("The Eiffel Tower is in"):
        a = model.transformer.h[-1].output[0, -1]
    with model.trace("The Colosseum is in"):
        b = model.transformer.h[-1].output[0, -1]
        similarity = torch.nn.functional.cosine_similarity(a, b, dim=0).save()

print(round(similarity.item(), 3))
```

Ordinary Python surrounds the traces:

```python
with model.session():
    norms = nnsight.save([])
    for layer in range(4):
        with model.trace(prompt):
            norms.append(model.transformer.h[layer].output.norm())

print([round(n.item(), 1) for n in norms])
```

Sessions matter most **remotely**, where a session is one job instead of N — see
the `nnsight-remote` skill.

## model.edit() — interventions that persist

An edit captures a block and replays it on every future run of that model.

Non-in-place (default) stores the edit on a shallow copy, leaving the original
clean — the two share weights, so this costs nothing:

```python
with model.edit() as (tracer, edited):
    edited.transformer.h[5].output[:] = 0

with edited.trace(prompt):
    patched = edited.output.logits[0, -1].argmax().save()
with model.trace(prompt):
    original = model.output.logits[0, -1].argmax().save()

print(model.tokenizer.decode(patched), "|", model.tokenizer.decode(original))
```

In-place edits change the model everyone holds:

```python
with model.edit(inplace=True):
    model.transformer.h[5].output[:, -1, :] += 2.0

with model.trace(prompt):
    steered = model.output.logits[0, -1].argmax().save()

print(model.tokenizer.decode(steered))

model.clear_edits()          # remove them again
```

Edits stack in registration order and run **before** your invokes on each trace.
A plain edit applies at the location's first occurrence; to re-apply it at every
generation step, put it under `tracer.iter`:

```python
with model.edit(inplace=True) as tracer:
    for _ in tracer.iter[:]:
        model.transformer.h[5].output[:, -1, :] += 2.0

with model.generate(prompt, max_new_tokens=3) as tracer:
    ids = tracer.result.save()

print(model.tokenizer.decode(ids[0]))
model.clear_edits()
```

Use edits for: always-on steering vectors, permanently ablated heads, attaching an
SAE into the forward path (see
[modules-and-architectures.md](modules-and-architectures.md)). Use a plain trace
for one-off interventions — an edit you forget to clear will silently contaminate
every later experiment.

## Conditionals and loops

Ordinary Python works inside a trace body; you are operating on real tensors in a
worker greenlet.

```python
with model.trace(prompt):
    resid = model.transformer.h[6].output
    if resid[:, -1].norm() > 100:            # real tensor, real branch
        model.transformer.h[6].output[:, -1, :] *= 0.5
    logits = model.output.logits.save()

print(logits.shape)
```

One syntactic restriction: a trace body cannot **start** with `try:` — nnsight
intercepts the body at its first line, and a `try` there is the one statement
Python gives it no way back out of. Put any statement above it, or the `try`
outside the `with`.

The exceptions are `model.scan()` (fake tensors — branch on shapes only, not
values) and remote traces (the body is serialized, so helpers from your own files
need `nnsight.register(...)`).

## Related

- [batching.md](batching.md) — the skip-covers-every-row rule
- [caching-and-scan.md](caching-and-scan.md) — cache + skip for layer reuse
- [execution-model.md](execution-model.md)
