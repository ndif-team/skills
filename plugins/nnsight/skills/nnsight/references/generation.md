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

print(ids.shape)                        # [1, 13] — 10 prompt + 3 new
print(model.tokenizer.decode(ids[0]))
```

Generation through the model is **greedy by default**, so it is reproducible
without passing anything. Ask for sampling explicitly (`do_sample=True, top_k=50`);
all kwargs are forwarded to the underlying `generate`.

Called without a `with` block it simply returns the ids: `ids = model.generate(prompt, max_new_tokens=3)`.

## pipe — decoded records

```python
with model.pipe(prompt, max_new_tokens=5, do_sample=False) as tracer:
    records = tracer.result.save()

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

## The unbounded-iteration trap

`tracer.all()` and `tracer.iter[:]` run until generation stops — and the final
over-run unwinds the loop **and every line after it in that invoke**. Code placed
after an unbounded loop does not run:

```python
with model.generate(prompt, max_new_tokens=3) as tracer:
    seen = nnsight.save([])
    for step in tracer.all():
        seen.append(model.output.logits[0, -1].argmax(dim=-1))
    after_the_loop = nnsight.save("this line never runs")

print(len(seen))                             # 3 — the loop's own values survive
print("after_the_loop" in globals())         # False — the trailing line was dropped
```

**Bounding the loop is not a reliable fix.** `max_new_tokens` is a cap, not a
promise: if the model emits EOS (or hits a stop string) before step `N`, a bounded
`tracer.iter[:N]` parks on a step that never runs and drops the trailing code
exactly as the unbounded form does — it warns `'...' was never reached` and
carries on. Since you cannot know in advance how many steps a generation takes,
no bound can guarantee the loop completes.

The reliable fix is to put the trailing code in a **separate empty invoke**:

```python
with model.generate(prompt, max_new_tokens=3) as tracer:
    seen = nnsight.save([])
    for step in tracer.iter[:]:
        seen.append(model.output.logits[0, -1].argmax(dim=-1))

    with tracer.invoke():             # a separate invoke always runs
        ids = tracer.result.save()

print(len(seen), ids.shape)
```

Values from the steps that did happen are kept either way — only trailing code in
the *same* invoke is lost.

### Step 0 is the prefill

`tracer.iter` counts forward passes, not generated tokens, and the first pass is
the prefill over the whole prompt:

```python
with model.generate("The Eiffel Tower is in the city of", max_new_tokens=3) as tracer:
    shapes = nnsight.save([])
    for step in tracer.iter[:]:
        shapes.append(model.transformer.h[0].output.shape)
# [(1, 10, 768), (1, 1, 768), (1, 1, 768)]  <- 10 prompt tokens, then one per step
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

print([tuple(c.shape) for c in chunks])    # [(1, 10), (1,), (1,)]
```

The prompt arrives as one block, then one token per step.

## Batched generation

```python
with model.generate(["The Eiffel Tower is in", "The Colosseum is in"],
                    max_new_tokens=3) as tracer:
    ids = tracer.result.save()

print([model.tokenizer.decode(row) for row in ids])
```

Prompts are left-padded to the longest. Each invoke keeps its own step counter, so
different invokes can iterate over different ranges.

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
