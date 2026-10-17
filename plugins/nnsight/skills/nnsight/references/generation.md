# Multi-Token Generation

Three run methods, three return values. Picking the wrong one is the usual cause of
"why is my output empty / not text / not reproducible".

| Method | Runs | `tracer.result` is | Sampling |
|---|---|---|---|
| `model.trace(x)` | one forward pass | the forward's output object | n/a |
| `model.generate(x, max_new_tokens=N)` | the model's `generate` | **token ids** `[batch, seq]` | the checkpoint's `generation_config` — instruct models sample |
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

Generation uses the **checkpoint's own `generation_config`**. A base model that
ships no sampling defaults (gpt2, SmolLM2-135M: `generation_config.do_sample` is
`None`) decodes greedily; most instruct checkpoints *sample* (Qwen3-8B and
Llama-3.2-1B-Instruct ship `do_sample=True, temperature=0.6`). Pass
`do_sample=False` for determinism — kwargs are forwarded to `generate`. Two runs
disagreeing, or a per-step reading that disagrees with the emitted tokens, is
sampling far more often than a bug in the intervention.

Called without a `with` block it returns the ids: `ids = model.generate(prompt, max_new_tokens=3)`.

## pipe — decoded records

```python
with model.pipe(prompt, max_new_tokens=5, do_sample=False) as tracer:
    records = tracer.result.save()

assert records[0]["generated_text"].startswith(prompt)
print(records[0]["generated_text"])
```

gpt2's pipeline config asks for sampling, hence `do_sample=False`.

## Interventions during generation

The block's interventions apply to the decode loop's forward passes. Without an
iteration API they bind to the **first** forward only:

```python
with model.generate(prompt, max_new_tokens=5) as tracer:
    model.transformer.h[6].output[:, -1, :] = 0     # step 0 only
    ids = tracer.result.save()

print(model.tokenizer.decode(ids[0]))
```

To act on every step, loop over `tracer.iter[...]`. `step` is a real integer, so
ordinary Python works inside the loop:

```python
with model.generate(prompt, max_new_tokens=4) as tracer:
    for step in tracer.iter[:4]:
        model.transformer.h[6].output[:, -1, :] *= 0.5     # every step
        if step >= 2:
            model.transformer.h[8].output[:, -1, :] += 2.0  # steps 2 and 3
    ids = tracer.result.save()

print(model.tokenizer.decode(ids[0]))
```

Other forms: `tracer.iter[2]` (one step), `tracer.iter[1:3]` (a range),
`tracer.iter[[0, 2, 4]]` (a list). Negative indices raise — there is no "last
step" shorthand. A `with tracer.iter[...]:` block is deprecated in favour of the
`for` loop.

### What a step is

`tracer.iter` counts forward passes, and step *k*'s forward produces generated
token *k*. Step 0 is the prefill over the whole prompt (`[B, prompt_len, H]`);
every later step is one position (`[B, 1, H]`). So a loop intervention applies to **every prompt position** on step 0 and one
token afterwards — use `tracer.iter[1:]` for generated tokens only. A write at
layer L on step *k* reaches the layers above L on that pass (and, through the KV
cache, on every later step) and changes that step's token; it reaches layers at or below L
only from step *k*+1, through the token it changed. Until a generated token
differs, layers ≤ L stay bit-identical to the unedited run:

```python
def run(edit):
    with model.generate(prompt, max_new_tokens=3, min_new_tokens=3) as tracer:
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

`tracer.all()` and `tracer.iter[:]` always end *by* asking for a step the run
does not make, so they always warn and always drop the statements after the loop.
Put what has to happen afterwards — reading `tracer.result` included — in a
separate empty invoke, its own worker, so the loop's unwind does not reach it:

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

### Two ways a loop hangs or lies

An open loop ends when a request in its body outruns the model, so a body that
reads no module spins forever, with no warning and no timeout:

<!-- test: skip -->
```python
with model.generate(prompt, max_new_tokens=2) as tracer:
    n = nnsight.save([0])
    for step in tracer.all():
        n[0] = step
```

And inside a loop, a read followed by a write to a module *below* it parks the
write on the next step, so every intervention lands one step late. Order the body
the way the forward runs.

### A wrong-order `.source` read lands on the next step

A read the model already ran past raises `OutOfOrderError` at step 0 and binds to
the next step after that. A `.source` op whose module's `.source` is first touched
inside the loop binds to the next step **even at step 0**: reading `attn.output`
and then an op inside `attn` gives the op's steps 1, 2, …, one entry short. The
only signal is the closing "was never reached" warning, which blames the loop bound:

```python
import warnings

attn = model.transformer.h[3].attn          # its .source is not touched before this loop
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with model.generate(prompt, max_new_tokens=4, min_new_tokens=4) as tracer:
        outs, ops = nnsight.save([]), nnsight.save([])
        for step in tracer.iter[:4]:
            outs.append(attn.output[0])                              # fires last in attn
            ops.append(attn.source.attention_interface_1.output[0])  # already ran this step

assert len(outs) == 4 and len(ops) == 3     # one short...
assert ops[0].shape[1] == 1                 # ...and it starts at step 1, not the 10-token prefill
assert any("was never reached" in str(w.message) for w in caught)
```

Once the module's `.source` has been touched (here, by the loop above), the same
body raises at step 0:

<!-- test: expect-error OutOfOrderError -->
```python
with model.generate(prompt, max_new_tokens=4, min_new_tokens=4) as tracer:
    for step in tracer.iter[:4]:
        late = attn.output[0]
        early = attn.source.attention_interface_1.output[0]
```

So: read ops before the module output that contains them, touch `.source` once
outside the trace (`_ = attn.source`), and assert parallel lists have equal length.

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

The template usually adds a system turn and special tokens, so **position indices
shift** (`-1` is still the last token, but "the 3rd token" is not the 3rd word).
And a chat model's layers are at `chat.model.layers[i]`, not
`chat.transformer.h[i]` — see [modules-and-architectures.md](modules-and-architectures.md).

## Related

- [execution-model.md](execution-model.md) — why trailing code disappears after an unbounded loop
- [batching.md](batching.md) — per-invoke iteration counters
- [modules-and-architectures.md](modules-and-architectures.md) — module paths per architecture
