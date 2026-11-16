# Control Flow: skip, stop, session, edit

Four ways to change *what runs*, where an edit to `.output` changes a value.

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt):
    original = model.output.logits[0, -1].argmax().save()      # ' Paris'
```

## module.skip(replacement)

The module's forward never runs and `replacement` becomes its output. Feeding a
residual block its own input makes it a pass-through:

```python
with model.trace(prompt):
    block = model.transformer.h[6]
    block.skip(block.input)                       # block 6 contributes nothing
    skipped = model.output.logits[0, -1].argmax().save()

assert int(skipped) != int(original)
```

- For a sublayer that is *added* to the residual stream (an MLP, an attention),
  the pass-through value is zeros shaped like its output:
  `mlp.skip(torch.zeros_like(mlp.input))`.
- The replacement must match the real output in structure, shape, dtype and
  device (a tuple for a module that returns one). A mismatch surfaces as a bare
  torch error inside a later module that names neither `skip` nor yours.
- `skip` moves the run past the module, as reading its `.output` would. Read
  anything earlier above the `skip` line, and do not read the skipped module's
  submodules: both are `OutOfOrderError`.
- In a batched trace, skip the module in every invoke or in none. A skip covers
  one call: under `generate`, repeat it inside `tracer.iter` or store it in an
  `edit`.

## tracer.stop()

Ends the forward pass. Save first; nothing after it in the block runs:

```python
with model.trace(prompt) as tracer:
    early = model.transformer.h[2].output.save()
    tracer.stop()                                  # layers 3 and up never run

assert tuple(early.shape) == (1, 10, 768)
```

- The module the worker is parked on has already run: its value is served and an
  edit to it lands.
- **A stopped run has no result.** `tracer.result` after the stop is unreachable,
  and saving it *before* the stop defeats the stop: the worker waits on the result
  until the whole forward has run.
- It stops the whole batch and, under `generate`, the whole decode loop, not one
  invoke or one step.

## model.session()

Inside a session, values flow from one trace to the next without a save, and
ordinary Python runs between the traces. Only saved values leave the session:

```python
with model.session():
    with model.trace("The Eiffel Tower is in"):
        donor = model.transformer.h[5].output[:, -1]            # not saved
    with model.trace("The Colosseum is in"):
        model.transformer.h[5].output[:, -1] = donor
        patched = model.output.logits[0, -1].argmax().save()

print(model.tokenizer.decode(patched))
```

The same holds for any nested context: only the outermost one filters on saves,
so a value read inside `with metric.backward():` needs its save only to leave
the enclosing trace. Locally a session is a convenience. With `remote=True` it
turns N requests into one: [remote](remote.md).

## model.edit()

An edit stores a block and replays it on every later run. The default returns an
edited **copy** that shares the weights and leaves the original clean:

```python
with model.edit() as (tracer, steered):
    steered.transformer.h[9].output[:, -1, :] = 0

with steered.trace(prompt):
    changed = steered.output.logits[0, -1].argmax().save()
with model.trace(prompt):
    same = model.output.logits[0, -1].argmax().save()

assert int(changed) != int(original) and int(same) == int(original)
```

`model.edit(inplace=True)` stores it on the model itself, for everything that
holds the model, until `model.clear_edits()`:

```python
with model.edit(inplace=True) as tracer:
    for _ in tracer.iter[:]:                       # every generation step, not just the first
        model.transformer.h[9].output[:, -1, :] = 0

with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    ids = tracer.result.save()

model.clear_edits()
print(model.tokenizer.decode(ids[0]))
```

- **A plain edit fires at each location's first occurrence**, which under
  `generate` is the prompt pass alone. Put the body under `tracer.iter[:]` to
  apply it at every step. Every later run then ends with the `UserWarning: ...
  was never reached` of an open loop; that is expected here.
- Edits stack in the order they were stored and run before the trace's own
  invokes. An in-place edit you forget to clear changes every later experiment.
- **An edit belongs to the envoy it was stored on.** `model.transformer.h[9].edit(
  inplace=True)` does nothing when you trace `model`, and `model.clear_edits()`
  does not remove it. Store edits on the model and name the layer inside.
- **An edit applies through nnsight, not to the module.** `trace`, `generate`,
  `pipe` and `model.trace(**ids, trace=False)` replay it. A direct call on the
  wrapped module (`model(**ids)`, or anything holding `model._module`, such as a
  HuggingFace `Trainer`) runs the unedited model:

```python
ids = model.tokenizer(prompt, return_tensors="pt").to(model.device)

with model.edit(inplace=True):
    model.transformer.h[9].output[:] = 0

through_trace = int(model.trace(**ids, trace=False).logits[0, -1].argmax())
direct_call = int(model(**ids).logits[0, -1].argmax())
model.clear_edits()

assert through_trace != direct_call and direct_call == int(original)
```

  For a change every caller sees, edit the weights:
  [model-editing-and-lora](model-editing-and-lora.md).
- Edits travel with a remote request, so an edited model can be traced with
  `remote=True`.
- Attaching a module and reading its internals needs an edit plus `hook=True`:
  [modules-and-architectures](modules-and-architectures.md#attaching-a-module).
