# Generation: Steps, Loops and What Each Run Method Returns

[SKILL.md](../SKILL.md) has the basic `model.generate` loop and its rules. This
page covers what a step is, loops whose length is unknown, the ways a per-step
list comes out short or shifted, streaming, batches and chat prompts.

```python
import warnings
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
```

## What a step is, and where a write lands

`tracer.iter` counts forward passes. Step 0 is the prefill over the whole prompt;
each later step is one position. So a write inside the loop touches **every prompt
position** on step 0 unless it is sliced (`[:, -1, :]`), and `tracer.iter[1:]`
skips the prompt pass. A write placed above or outside the loop reaches step 0
only.

A write to layer L on step *k* changes that step's token. It reaches layers
above L on that pass and, through the key-value cache, on every later step.
Layers at or below L see it only from step *k* + 1, through the token it changed:

```python
def run(edit):
    with model.generate(prompt, max_new_tokens=3, min_new_tokens=3, do_sample=False) as tracer:
        early = nnsight.save([])
        for step in tracer.iter[:3]:
            early.append(model.transformer.h[3].output.clone())
            if edit and step == 1:
                model.transformer.h[6].output[:, -1, :] *= -5
        ids = tracer.result.save()
    return early, ids

base, base_ids = run(False)
edit, edit_ids = run(True)
assert tuple(base[0].shape) == (1, 10, 768) and tuple(base[1].shape) == (1, 1, 768)
assert torch.equal(base_ids[0, :11], edit_ids[0, :11])      # token 0 unchanged
assert base_ids[0, 11] != edit_ids[0, 11]                   # step 1's token changed
assert torch.equal(base[1], edit[1])                        # layer 3, step 1: untouched
assert not torch.equal(base[2], edit[2])                    # step 2 sees the new token
```

Per-step states have different lengths (prompt, then 1), so take `[:, -1]` of
each before stacking them.

## The one rule for iteration loops

A loop must not ask for a step the run does not make. When it does, the loop is
cut short there, what it saved is kept, the statements after it never run, and
the only signal is this warning:

```
UserWarning: 'model.transformer.h.6.output.i3' was never reached: the loop asked
for a step the run did not make, so it was cut short — values saved inside the
loop are kept, and the statements after it did not run. ...
```

The warning names the loop bound, but it has **two causes**:

1. **The run ended early.** `max_new_tokens` is an upper bound; an EOS token or a
   stop string ends the run sooner. `min_new_tokens=N` suppresses EOS until N
   tokens exist. It does not override `stop_strings=`.
2. **The loop body reads modules out of forward order.** In a loop that starts at
   step 0 that raises `OutOfOrderError`. In one that starts later (`iter[1:]`) it
   does not: each late read binds to the *next* step, so its list comes back one
   entry short and shifted, a write lands a step late, and the warning names an
   occurrence past the loop's end (`.i4` for a four-step run).

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with model.generate(prompt, max_new_tokens=4, min_new_tokens=4, do_sample=False) as tracer:
        late, early = nnsight.save([]), nnsight.save([])
        for step in tracer.iter[1:4]:
            late.append(model.transformer.h[8].output.shape[1])
            early.append(model.transformer.h[2].output.shape[1])   # layer 2 after layer 8
        tail = nnsight.save("after the loop")

assert len(late) == 3 and len(early) == 2                 # one short, no error
assert "tail" not in dir()                                # the trailing statement never ran
assert any(".i4' was never reached" in str(w.message) for w in caught)
```

A shifted body that stops short of the run's last step (`iter[1:3]` over four
steps) finishes with no warning at all. Order the body the way the forward runs,
and assert that parallel lists have equal length.

The `.iN` suffix in the warning and in `OutOfOrderError` is the occurrence index:
`.i3` is the fourth visit to that location, generation step 3.

### When the step count is unknown

`tracer.all()` and `tracer.iter[:]` end *by* asking for a step the run does not
make, so they always warn and always drop the statements after the loop. Put
what must happen afterwards, reading `tracer.result` included, in a separate
empty invoke. It is another worker, so the loop's end does not reach it:

```python
with model.generate(max_new_tokens=3, do_sample=False) as tracer:
    with tracer.invoke(prompt):
        seen = nnsight.save([])
        for step in tracer.all():
            seen.append(model.output.logits[0, -1].argmax(dim=-1))

    with tracer.invoke():
        ids = tracer.result.save()

assert len(seen) == 3 and tuple(ids.shape) == (1, 13)
```

An open loop ends when a read in its body outruns the model, so **an open loop
whose body reads no module never returns**. There is no warning and no timeout.

### `.source` operations inside a loop

A `.source` operation read after its module's own `.output` binds to the next
step even at step 0, when that module's `.source` was first touched inside the
loop: the list is one short and starts at step 1. Touch `.source` once before the
trace (`_ = attn.source`) and read operations before the module output that
contains them. Details: [source-tracing](source-tracing.md).

## `pipe` and generation keywords

`model.pipe` runs the whole task pipeline and puts its records on
`tracer.result`. It follows the pipeline's own task parameters, which sample for
gpt2, hence `do_sample=False`. Per-step loops work as under `generate`.

```python
with model.pipe(prompt, max_new_tokens=5, do_sample=False) as tracer:
    records = tracer.result.save()

assert records[0]["generated_text"].startswith(prompt)
```

Keywords `generate` does not consume go to the model's own `generate`, and what
that returns is `tracer.result`. Per-step scores without a loop:

```python
with model.generate(prompt, max_new_tokens=3, do_sample=False,
                    return_dict_in_generate=True, output_scores=True) as tracer:
    out = tracer.result.save()

assert tuple(out.sequences.shape) == (1, 13) and len(out.scores) == 3
```

`tracer.result` is read inside the block. Two runs that disagree, or a per-step
reading that disagrees with the emitted tokens, is sampling far more often than
a broken intervention.

## Streaming, batches and chat

Tokens as they are produced, the prompt first and then one token per step:

```python
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    chunks = nnsight.save([])
    for step in tracer.iter[:3]:
        chunks.append(model.generator.streamer.output)

assert [tuple(c.shape) for c in chunks] == [(1, 10), (1,), (1,)]
```

A batch is left-padded and the padding stays in `tracer.result`; decode with
`skip_special_tokens=True`. Each invoke keeps its own step counter.

```python
with model.generate(["The Eiffel Tower is in", "The Colosseum is in"],
                    max_new_tokens=3, do_sample=False) as tracer:
    ids = tracer.result.save()

assert tuple(ids.shape) == (2, 10)      # padded to the longer prompt, plus 3
print([model.tokenizer.decode(row, skip_special_tokens=True) for row in ids])
```

A chat checkpoint takes the message list as its input and applies its own
template, generation prompt included. The template adds a system turn and
special tokens, so only `-1` keeps its meaning as a position:

```python
chat = TransformersModel("HuggingFaceTB/SmolLM2-135M-Instruct", task="text-generation", dispatch=True)
messages = [{"role": "user", "content": "Name one city in France."}]

with chat.generate(messages, max_new_tokens=8, do_sample=False) as tracer:
    args, kwargs = chat.inputs
    n_prompt = nnsight.save(kwargs["input_ids"].shape[1])
    reply = tracer.result.save()

print(chat.tokenizer.decode(reply[0, n_prompt:], skip_special_tokens=True))
```

To see or change the formatted prompt, build it with
`chat.tokenizer.apply_chat_template(messages, tokenize=False,
add_generation_prompt=True)` and pass the string.

## Elsewhere

- `tracer.stop()` ends the whole generation, and the result with it:
  [control-flow](control-flow.md).
- Activations captured under `generate` do not require grad, so a backward block
  raises `NotImplementedError`: [gradients](gradients.md#where-gradients-are-unavailable).
- `tracer.cache()` across steps returns one entry per step with no loop:
  [caching-and-scan](caching-and-scan.md).
- A module called several times per forward makes several occurrences per step:
  [access-and-modify](access-and-modify.md#a-module-that-runs-more-than-once).
