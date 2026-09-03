# Multi-Token Generation

Three run methods, three different return values. Picking the wrong one is the
most common source of "why is my output empty / not text / not reproducible".

| Method | Runs | `tracer.result` is | Sampling |
|---|---|---|---|
| `model.trace(x)` | one forward pass | the forward's output object | n/a |
| `model.generate(x, max_new_tokens=N)` | the model's `generate` | **token ids** `[batch, seq]` | **greedy** unless asked |
| `model.pipe(x, ...)` | the whole task pipeline | the pipeline's **records** (decoded text, labels) | the checkpoint's `task_specific_params` — often sampled |

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
```

## generate — token ids

```python
with model.generate(prompt, max_new_tokens=3) as tracer:
    ids = tracer.result.save()

assert tuple(ids.shape) == (1, 13)      # 10 prompt tokens + 3 new
print(model.tokenizer.decode(ids[0]))   # The Eiffel Tower is in the city of Paris, and
```

`tracer.result` has to be read inside the block — it is served during the run, so
reading it afterwards raises `ValueError: Cannot access 'result' outside of
interleaving`. `model.generator.output` hands back the identical tensor and warns
that `tracer.result` is the way to ask for it.

Generation through the model is **greedy by default**, so it is reproducible
without passing anything. Ask for sampling explicitly (`do_sample=True, top_k=50`);
all kwargs are forwarded to the underlying `generate`.

Called without a `with` block it simply returns the ids: `ids = model.generate(prompt, max_new_tokens=3)`.

## pipe — decoded records

```python
with model.pipe(prompt, max_new_tokens=5, do_sample=False) as tracer:
    records = tracer.result.save()

assert records[0]["generated_text"].startswith(prompt)
print(records[0]["generated_text"])
```

Note `do_sample=False`:
gpt2's pipeline config asks for sampling, so pipe output is non-deterministic
unless you turn it off.

## Interventions during generation

The block's interventions apply to the decode loop's forward passes. Without an
iteration API they bind to the **first** forward only:

```python
with model.generate(prompt, max_new_tokens=5) as tracer:
    model.transformer.h[6].output[:, -1, :] = 0     # step 0 only
    ids = tracer.result.save()

print(model.tokenizer.decode(ids[0]))
```

To act on every step, loop over `tracer.iter[...]`:

```python
with model.generate(prompt, max_new_tokens=5) as tracer:
    for step in tracer.iter[:5]:
        model.transformer.h[6].output[:, -1, :] *= 0.5
    ids = tracer.result.save()

print(model.tokenizer.decode(ids[0]))
```

## Collecting per-step values

```python
with model.generate(prompt, max_new_tokens=5) as tracer:
    picks = nnsight.save([])
    for step in tracer.iter[:5]:
        picks.append(model.output.logits[0, -1].argmax(dim=-1))
    ids = tracer.result.save()

print([model.tokenizer.decode(p) for p in picks])
```

`step` is a real integer, so ordinary Python works inside the loop:

```python
with model.generate(prompt, max_new_tokens=4) as tracer:
    for step in tracer.iter[:4]:
        if step >= 2:
            model.transformer.h[8].output[:, -1, :] += 2.0
    ids = tracer.result.save()

print(model.tokenizer.decode(ids[0]))
```

## The one rule for iteration loops

**A loop must not ask for a step the run does not make.** A bound the run meets
is fine, and the code after the loop runs. A loop that outruns the run — bounded
or open — is cut short there with a warning: values saved inside it are kept,
and the statements after it do not run. The result looks complete while being
shorter than the bound, so check the `len()` of what you collected:

```python
import warnings

tail = None
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with model.generate(prompt, max_new_tokens=3) as tracer:
        seen = nnsight.save([])
        for step in tracer.iter[:10]:            # 10 steps of a 3-step run
            seen.append(model.transformer.h[6].output[:, -1, :].norm())
        tail = nnsight.save("after the loop")

assert len(seen) == 3                            # cut short at the run's end
assert tail is None                              # the statement after the loop never ran
assert any("was never reached" in str(w.message) for w in caught)
```

`max_new_tokens` is an upper bound: an EOS or a stop string ends generation
sooner, and then a bound matching `max_new_tokens` outruns the run.
`min_new_tokens=N` suppresses EOS until N tokens are generated, which is what
makes a bound of N hold:

```python
with model.generate(prompt, max_new_tokens=5, min_new_tokens=5) as tracer:
    picks = nnsight.save([])
    for step in tracer.iter[:5]:
        picks.append(model.output.logits[0, -1].argmax(dim=-1))
    ids = tracer.result.save()

assert len(picks) == 5 and ids.shape[1] == 5 + len(model.tokenizer.encode(prompt))
```

`min_new_tokens` holds off EOS only — a `stop_strings=` criterion still ends the
run wherever it matches.

### When you cannot know the step count

`tracer.all()` and `tracer.iter[:]` end *by* asking for a step the run does not
make, so they warn instead of raising. The same unwind still discards everything
the block has after the loop:

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with model.generate(prompt, max_new_tokens=3) as tracer:
        seen = nnsight.save([])
        for step in tracer.all():
            seen.append(model.output.logits[0, -1].argmax(dim=-1))
        after_the_loop = nnsight.save("this line never runs")

assert len(seen) == 3                            # the loop's own values survive
assert "after_the_loop" not in globals()         # the trailing line was dropped
assert "was never reached" in str(caught[0].message)
```

Put what has to happen afterwards in a separate empty invoke — its own worker, so
the loop's unwind does not reach it:

```python
with model.generate(max_new_tokens=3) as tracer:
    with tracer.invoke(prompt):
        seen = nnsight.save([])
        for step in tracer.iter[:]:
            seen.append(model.output.logits[0, -1].argmax(dim=-1))

    with tracer.invoke():             # a separate invoke always runs
        ids = tracer.result.save()

assert len(seen) == 3 and ids.shape[0] == 1
```

Reading `tracer.result` below the `with` block instead does not work: it is
served during the run, so outside the block it raises `ValueError: Cannot access
'result' outside of interleaving`.

### Two ways a loop hangs or lies

An open loop ends when a request in its body outruns the model, so a body that
reads no module never ends at all — a pure Python spin, no warning, no timeout:

<!-- test: skip -->
```python
# Hangs forever — nothing in the body parks the worker, so the loop never ends.
with model.generate(prompt, max_new_tokens=2) as tracer:
    n = nnsight.save([0])
    for step in tracer.all():
        n[0] = step
```

And inside a loop, a read followed by a write to a module *below* it parks the
write on the next step, so every intervention lands one step late. Order the body
the way the forward runs.

### Step 0 is the prefill

`tracer.iter` counts forward passes, not generated tokens, and the first pass is
the prefill over the whole prompt:

```python
with model.generate("The Eiffel Tower is in the city of", max_new_tokens=3) as tracer:
    shapes = nnsight.save([])
    for step in tracer.iter[:]:
        shapes.append(tuple(model.transformer.h[0].output.shape))

assert shapes == [(1, 10, 768), (1, 1, 768), (1, 1, 768)]   # prompt, then one token per step
```

So an intervention inside the loop applies to **every prompt position** on step 0
and to a single token afterwards. Skip it with `tracer.iter[1:]` if you only mean
the generated tokens.

Other iteration forms: `tracer.iter[2]` (one step), `tracer.iter[1:3]` (a range),
`tracer.iter[[0, 2, 4]]` (explicit list). Negative indices raise — there is no
"last step" shorthand. A `with tracer.iter[...]:` block is deprecated in
favour of the `for` loop.

## Streaming tokens as they are produced

```python
with model.generate(prompt, max_new_tokens=3) as tracer:
    chunks = nnsight.save([])
    for step in tracer.iter[:3]:
        chunks.append(model.generator.streamer.output)

assert [tuple(c.shape) for c in chunks] == [(1, 10), (1,), (1,)]
```

The prompt arrives as one block, then one token per step.

## Batched generation

```python
with model.generate(["The Eiffel Tower is in", "The Colosseum is in"],
                    max_new_tokens=3) as tracer:
    ids = tracer.result.save()

assert tuple(ids.shape) == (2, 10)      # left-padded to the longer prompt, + 3 new
print([model.tokenizer.decode(row, skip_special_tokens=True) for row in ids])
```

Prompts are left-padded to the longest, and the padding is still in
`tracer.result`, so decode with `skip_special_tokens=True` unless you want it.
Each invoke keeps its own step counter, so different invokes can iterate over
different ranges.

## Chat models

Instruct-tuned checkpoints need their chat template applied. Do it with the
tokenizer, then trace the formatted string:

```python
chat = TransformersModel("HuggingFaceTB/SmolLM2-135M-Instruct", dispatch=True)

text = chat.tokenizer.apply_chat_template(
    [{"role": "user", "content": "Name one city in France."}],
    tokenize=False,
    add_generation_prompt=True,
)

with chat.generate(text, max_new_tokens=10) as tracer:
    reply = tracer.result.save()

print(chat.tokenizer.decode(reply[0, -10:], skip_special_tokens=True))
```

Two things to watch: the template usually adds a system turn and special tokens,
so **position indices shift** (`-1` is still the last token, but "the 3rd token" is
not the 3rd word); and a chat model's layers are at `chat.model.layers[i]`, not
`chat.transformer.h[i]` — see [modules-and-architectures.md](modules-and-architectures.md).

```python
with chat.trace(text):
    resid = chat.model.layers[10].output.save()   # a tensor on transformers 5.x

print(resid.shape)
```

## Related

- [execution-model.md](execution-model.md) — why trailing code disappears after an unbounded loop
- [batching.md](batching.md) — per-invoke iteration counters
- [modules-and-architectures.md](modules-and-architectures.md) — module paths per architecture
