# Debugging nnsight

Find the message or the symptom in the table, then read the section it names.
Code from a tutorial, a paper repository or a model card is probably written
for an older nnsight: translate it with the
[porting guide](debugging-porting-pre-0.8.md) before debugging it.

## By message

| Message | Cause | Fix |
|---|---|---|
| `OutOfOrderError: '<path>.output.i0' was requested but the model already ran past it` | a module read after a later one, or a module that never ran | [OutOfOrderError](#outofordererror) |
| `OutOfOrderError: '<number>.grad.i0' ...` | a gradient read in forward order, on a slice, or off the metric's path | [gradients](gradients.md#when-a-grad-read-fails) |
| `UserWarning: '<path>.iN' was never reached: the loop asked for a step the run did not make ...` | a `tracer.iter` loop outran the run, or its body reads modules out of order | [generation](generation.md#the-one-rule-for-iteration-loops) |
| `NameError` / `UnboundLocalError` on a name **after** the block | it was not saved, or a cut-short loop above it dropped the line | [Nothing came back](#nothing-came-back) |
| `NameError` **inside** the body, outside any invoke, on a name an invoke binds | trace-body code outside the invokes runs before all of them | write into a `nnsight.save({})` from inside each invoke; combine after the block |
| `NameError` inside an invoke on another invoke's value | the reader ran before the producer, or a barrier's count is too small | [batching](batching.md#sharing-values-across-invokes) |
| `RuntimeError: You cannot infer task automatically within pipeline when using offline mode` | no `task=` and no network | `TransformersModel(repo_id, task="text-generation", ...)` |
| `ValueError: Cannot access ... outside of interleaving` | `.output` read with no trace running; also an input-less `model.trace()` whose body reads a module outside any invoke | open a trace or an invoke; `model.scan` for shapes |
| `ValueError: trace() needs an input, or at least one tracer.invoke(...) block` | `with model.trace():` and no invoke | pass an input or open an invoke |
| `ValueError: Cannot invoke while the model is already running.` | an invoke under `model.trace(x)`, nested in another invoke, or inside an `iter` loop | open every invoke at the top level of an input-less trace |
| `AttributeError: 'NoneType' object has no attribute 'event'` | `model.trace(...)` nested inside another trace | `model.session()` for several traces; sibling invokes for several inputs |
| `AttributeError: '...' object (nor its module) has attribute 'model'` | wrong module path for this architecture; also a custom `eproperty` whose getter raised | `scripts/inspect_model.py <repo_id>` |
| `AttributeError: 'Envoy' object (nor its module) has attribute '12'` | an index past the last layer | `h[-1]`, or stay within `len(model.transformer.h)` |
| `AttributeError: module 'nnsight' has no attribute 'list'`; `'Tensor' object has no attribute 'value'` | code for an older nnsight | [porting guide](debugging-porting-pre-0.8.md) |
| `TypeError: 'tuple' object does not support item assignment` | `attn.output[0] = x` | `attn.output[0][:] = x`, or assign a rebuilt tuple |
| `RuntimeError: The expanded size of the tensor (10) must match the existing size (3) ...` | a written value whose shape differs from the activation | build it from the activation (`torch.zeros_like(x)`) |
| `RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation` | an in-place write that a later backward needs | [gradients](gradients.md#training-a-tensor-through-the-model) |
| `NotImplementedError: This tensor does not require grad, so a backward session cannot produce gradients` | backward inside `generate`, under `torch.no_grad()`, or on a frozen model | [gradients](gradients.md#where-gradients-are-unavailable) |
| `NotImplementedError: NNsight does not support batching multiple invokes` | plain `NNsight` with two input invokes | one invoke with a batched tensor |
| `ValueError: A barrier was never reached by every block it waits for` | `tracer.barrier(n)` with `n` above the number of blocks calling it | pass the true count |
| ``ValueError: A batched `.skip()` has to cover every row`` | a skip in some invokes only | skip in every invoke or none |
| `ValueError: save() was called outside a trace` | a save before or after the `with` | move it inside |
| `WithBlockNotFoundError: nnsight has no source for <stdin> ...` | the block's source cannot be read back: a script piped to the interpreter, `exec` of a string | run from a file, a notebook or `python -c` |
| `SyntaxError: 'return' outside function` | a `return` inside the trace body | save the value and return after the `with` |
| `GuardOnDataDependentSymNode` | branching on a value under `model.scan` | branch on shapes, or use a real trace |
| `SourceNotAvailable` | `.source` cannot drill into that operation | [source-tracing](source-tracing.md#limits) |
| a bare torch shape or dtype error from inside a later module | a `skip` replacement or a batched replacement of the wrong shape, dtype or device | check the writes in the block; the traceback does not name them |
| `RemoteError` | anything raised on NDIF | [remote](remote.md) |
| a `RuntimeError` whose message starts with another error's name, on a `VLLM` model | an error inside the vLLM worker | [vllm](vllm.md) |

Two messages say what to do and need no more: ``The body of a traced `with` must
start on its own line`` and ``A traced `with` block cannot start with `try:` ``.

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
```

## OutOfOrderError

The trace body is a worker that parks until the model produces each value it
asks for. Ask for something the model has already passed and nothing will wake
it, so nnsight raises when the run ends.

```python
with model.trace(prompt):
    late = model.transformer.h[8].output.save()
    early = model.transformer.h[2].output.save()      # the model is already past layer 2
# OutOfOrderError: 'model.transformer.h.2.output.i0' was requested but the model already ran past it
```

**Asked in the wrong order.** Reorder into forward order; a block's submodules
come before the block's own `.output`. When two orders are both needed, give each
its own invoke, or collect with `tracer.cache()`:

```python
with model.trace() as tracer:
    with tracer.invoke(prompt):
        late = model.transformer.h[8].output.save()
    with tracer.invoke(prompt):
        early = model.transformer.h[2].output.save()

assert late.shape == early.shape == (1, 10, 768)
```

**The module never ran.** The same message, with nothing read after it:

- a location after `tracer.stop()`, or `tracer.result` of a stopped run;
- a child of a module that was `.skip()`ped;
- an MoE expert that was not routed to;
- a `.source` operation on a branch this model does not take, or the first
  `.source` on a module whose forward had already begun
  ([source-tracing](source-tracing.md#limits));
- a submodule of an attached module called without `hook=True`
  ([modules-and-architectures](modules-and-architectures.md#attaching-a-module));
- a module whose `forward` was replaced after the model was wrapped.

**A collection iterated out of order.** A `set` of `(layer, head)` pairs iterates
in arbitrary order. Loop over `sorted(...)`.

The `.i0` suffix is the occurrence index: `.i3` is the fourth visit to that
location, generation step 3. A second read of the same module in one body is
still occurrence 0 and returns the first call's value
([access-and-modify](access-and-modify.md#a-module-that-runs-more-than-once)).

## Nothing came back

- **Not saved.** `NameError` at module level, `UnboundLocalError` inside a
  function. When the name already existed before the block, nothing raises and it
  keeps its **old value**:

```python
stale = torch.zeros(3)
with model.trace(prompt):
    stale = model.transformer.h[0].output          # not saved
assert tuple(stale.shape) == (3,)                  # still the old tensor
```

- **A list created inside the block is empty or undefined after it.** Save the
  container: `acts = nnsight.save([])`, then append raw values.
- **A list created outside and appended to is empty after a remote run.** Same
  fix.
- **A bare `value.save()` line**, or a save wrapped in a call
  (`float(x.save())`), binds nothing. Bind the save, convert after the block.
- **Everything after a loop is missing.** A `tracer.iter` loop was cut short:
  [generation](generation.md#the-one-rule-for-iteration-loops).

## No error, wrong result

| Symptom | Cause |
|---|---|
| shapes look right, numbers are for one batch row | `.output[0]` on a block whose output is a plain tensor |
| the intervention changed nothing | a local was rebound (`acts = acts * 0`) instead of written through the module |
| an in-place write changed nothing | it came after a later module was read in the same body |
| the whole block did nothing | `model.trace(x)` or `model.scan(x)` called without `with` |
| only one forward pass ran | `max_new_tokens` passed to `model.trace`; use `model.generate` |
| a module that runs several times gives identical values | `.output` is its first call; step through the calls with `tracer.iter` |
| a probability is near zero at every layer | the target id is the BOS token: `encode(text, add_special_tokens=False)` |
| a batched mean, max or position index is off | left padding: [batching](batching.md#positions-shift-with-the-batch) |
| two generations disagree, or the lens disagrees with the emitted token | sampling: pass `do_sample=False` |
| every later trace is changed | an `edit(inplace=True)` was never cleared, or a weight was edited |
| an upstream gradient is too small with no error | a replacement built from a fresh tensor: [gradients](gradients.md#replacements-and-the-graph) |
| `float(saved)` warns `Converting a tensor with requires_grad=True to a scalar` | saved activations carry the graph: `.detach()` before saving |

Before trusting a result, check that the unmodified run in the same trace
reproduces the known behavior, and that an absurd intervention (zero a whole
layer) moves the metric.

## Tools

- **Print inside the trace.** The body runs real Python on real tensors:
  `print(x.shape, x.dtype, x.device)` shows the live value. Remotely the output
  comes back as log lines.
- **Shapes without running:** `model.scan(prompt)`, or
  `scripts/inspect_model.py <repo_id> --prompt "..."`.
- **Operations inside a forward:** `print(module.source)`.
- **Tracebacks.** nnsight removes most of its own frames; the last frame is your
  line and the exception type is unchanged. Frames from torch and transformers
  stay, so a failure caused by a badly shaped write ends inside the model and does
  not show the line that wrote it. `nnsight.CONFIG.APP.DEBUG = True` (or
  `NNSIGHT_DEBUG=1`) keeps every frame.
- **Environment:** `python scripts/check_env.py` prints versions and GPU memory;
  `--remote` adds the NDIF key, host, deployed models and the package diff.
