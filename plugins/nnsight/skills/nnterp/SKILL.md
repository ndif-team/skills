---
name: nnterp
description: Write architecture-agnostic interpretability code with nnterp's StandardizedTransformer, which wraps nnsight and gives every transformer the same accessors — layers_output[i], attentions_output[i], mlps_output[i], attention_probabilities[i], logits, next_token_probs, skip_layers — so one script runs unchanged on GPT-2, Llama, Qwen, Gemma, and others. Use when an experiment must generalize across model families, when you want normalized tuple-vs-tensor outputs, or when you would otherwise be writing per-architecture module-path branches by hand.
---

# nnterp

nnterp is a thin layer over nnsight that renames every transformer into one
schema. Instead of `model.transformer.h[5].output` for GPT-2 and
`model.model.layers[5].output[0]` for Llama, you write `model.layers_output[5]`
for both — and it returns a tensor either way.

Use it when portability matters. Use plain nnsight when you need something nnterp
does not expose, or when you are working with one model and want no extra layer.

<!-- test: setup -->
```python
import torch
import nnsight
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")
prompt = "The Eiffel Tower is in the city of"
```

`StandardizedTransformer` dispatches on construction and validates that the
renaming worked, so a model it cannot standardize fails loudly at load rather than
silently giving you the wrong module.

## The unified accessors

```python
with model.trace(prompt):
    resid = model.layers_output[5].save()          # residual stream after block 5
    attn = model.attentions_output[6].save()       # attention output, always a tensor
    mlp = model.mlps_output[8].save()              # MLP output
    logits = model.logits.save()

print(resid.shape, attn.shape, mlp.shape, logits.shape)
```

```
torch.Size([1, 10, 768]) torch.Size([1, 10, 768]) torch.Size([1, 10, 768]) torch.Size([1, 10, 50257])
```

Note what is *not* there: no `.output[0]`, no per-family conditional. nnterp
normalizes tuple-returning modules to tensors, which removes the single most
common source of wrong-but-plausible interpretability code.

Inputs have the matching form — `layers_input[i]`, `attentions_input[i]`,
`mlps_input[i]` — and the module objects themselves are `model.layers[i]`,
`model.attentions[i]`, `model.mlps[i]` when you need to reach further in.

Convenience values:

```python
with model.trace(prompt):
    probs = model.next_token_probs.save()

print(f"{model.tokenizer.decode(probs.argmax())!r}  layers={model.num_layers}  hidden={model.hidden_size}")
```

```
' Paris'  layers=12  hidden=768
```

## The same script on another architecture

```python
llama_style = StandardizedTransformer("HuggingFaceTB/SmolLM2-135M-Instruct")

with llama_style.trace(prompt):
    resid = llama_style.layers_output[5].save()

print(resid.shape)          # torch.Size([1, 9, 576]) — identical code, different family
```

That is the whole value proposition. In plain nnsight the same two runs need
`model.transformer.h[5].output` and `model.model.layers[5].output[0]` — different
paths *and* different unwrapping.

## Writing, not just reading

Assignment goes through the same accessors:

```python
with model.trace(prompt):
    model.layers_output[5] = model.layers_output[5] * 0
    ablated = model.logits.save()

print(f"layer 5 zeroed -> {model.tokenizer.decode(ablated[0, -1].argmax())!r}")
```

And `skip_layers` bypasses a contiguous range — the portable form of layer
ablation:

```python
with model.trace(prompt):
    model.skip_layers(0, 2)                 # layers 0..2 do not run
    skipped = model.logits.save()

print(skipped.shape)
```

## Attention probabilities

nnterp exposes `model.attention_probabilities[i]`, which finds the probability
matrix inside whatever attention implementation the model uses. Enable it at
construction — it also forces `attn_implementation="eager"`:

```python
with_probs = StandardizedTransformer("openai-community/gpt2", enable_attention_probs=True)

with with_probs.trace(prompt):
    pattern = with_probs.attention_probabilities[5].save()

print(pattern.shape)                          # [batch, heads, query, key]
print(pattern[0, 0].sum(-1)[:3])               # rows sum to 1
```

They are assignable too, which is the portable way to intervene on attention:

```python
with with_probs.trace(prompt):
    uniform = torch.ones_like(with_probs.attention_probabilities[5])
    with_probs.attention_probabilities[5] = uniform / uniform.shape[-1]
    flattened = with_probs.logits.save()

print(f"attention flattened at layer 5 -> {with_probs.tokenizer.decode(flattened[0, -1].argmax())!r}")
```

Construction validates the accessor (shape, rows summing to 1, and that editing
the probabilities changes the logits), so a model where it does not work fails at
load rather than returning something wrong. Check `model.attn_probs_available`
when handling arbitrary architectures.

**Version note.** nnterp locates this value by the *name of the operation* in the
model's attention forward, and those names follow whatever transformers writes.
GPT-2's dropout call changed from `module.attn_dropout(...)` (transformers ≤ 4.57)
to `nn.functional.dropout(...)` (transformers ≥ 5), which broke the accessor on
GPT-2 until nnterp learned both spellings. If `enable_attention_probs=True` raises
`AttributeError: ... has no operation ...` on some other architecture, that is the
same class of problem: run `model.attention_probabilities.print_source()` to see
the real operation names and pass the right one via
`RenameConfig(attn_prob_source=...)`. Plain nnsight `.source` is the fallback —
see the `attention-analysis` skill.

## Choosing between nnterp and plain nnsight

| Use nnterp when | Use plain nnsight when |
|---|---|
| the script must run on several model families | you are working with one model |
| you want tuple-vs-tensor normalized for you | you need a module nnterp does not name |
| you want load-time validation that paths resolved | you need `.source`, `scan`, custom `edit` plumbing, or vLLM/diffusion |
| writing a reusable tool or library | writing a one-off experiment |

The two compose: `StandardizedTransformer` **is** an nnsight model, so
`tracer.invoke`, `tracer.cache`, `tensor.backward()`, `session`, and `remote=True`
all work on it, and you can drop to raw module paths at any point.

```python
with model.trace() as tracer:
    with tracer.invoke(prompt):
        a = model.layers_output[3].save()          # nnterp accessor
    with tracer.invoke("The Colosseum is in the city of"):
        b = model.transformer.h[3].output.save()   # raw nnsight path, same module

print(torch.allclose(a, b, atol=1e-4), a.shape)
```

## Notes

- nnterp is on NDIF's remote import whitelist, so `StandardizedTransformer(...,
  remote=True)` works inside remote traces — see the `nnsight-remote` skill.
- It subclasses nnsight's `LanguageModel`, which warns as deprecated on
  construction; that warning is nnterp's internal usage, not a problem with your
  code.
- `check_renaming=True` (the default) is what makes it fail loudly on an
  unsupported architecture. Do not turn it off to make a load succeed — the
  accessors will silently point somewhere wrong.
- For an architecture nnterp does not know, pass a `rename_config`, or use
  nnsight's own `rename={...}` (see the `nnsight` skill → modules and
  architectures).

## Related skills

- `nnsight` — the underlying API, module paths, and `rename`
- `attention-analysis` — attention patterns via `.source` when the accessor is unavailable
- `logit-lens`, `activation-patching` — techniques worth writing portably
