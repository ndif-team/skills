---
name: nnterp
description: Write architecture-agnostic interpretability code with nnterp's StandardizedTransformer, which wraps nnsight and gives every transformer the same accessors — layers_output[i], attentions_output[i], mlps_output[i], attention_probabilities[i], logits, next_token_probs, skip_layers — so one script runs unchanged on GPT-2, Llama, Qwen, Gemma, and others. Use when an experiment must generalize across model families, when you want normalized tuple-vs-tensor outputs, or when you would otherwise be writing per-architecture module-path branches by hand.
---

# nnterp

nnterp is a thin layer over nnsight that renames every transformer into one
schema. Instead of `model.transformer.h[5].output` for GPT-2 and
`model.model.layers[5].output` for Llama, you write `model.layers_output[5]` for
both — and it returns a tensor either way, on families whose blocks return a
tensor and on families whose blocks return a tuple.

Use it when portability matters. Use plain nnsight when you need something nnterp
does not expose, or when you are working with one model and want no extra layer.

Everything below is verified on nnsight 0.8.0 and transformers 5.15. nnterp
requires nnsight >= 0.8.

<!-- test: setup -->
```python
import torch
import nnsight
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2")
prompt = "The Eiffel Tower is in the city of"
n_tokens = len(model.tokenizer.encode(prompt))
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

assert resid.shape == attn.shape == mlp.shape == (1, n_tokens, model.hidden_size)
assert logits.shape == (1, n_tokens, model.vocab_size)
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
`attentions_output[i]` is the value *after* the output projection, so per-head
work on it needs an explicit `view(batch, seq, num_heads, -1)`; the
`attention-analysis` skill covers the per-head conventions.

Convenience values:

```python
with model.trace(prompt):
    probs = model.next_token_probs.save()
    logits = model.logits.save()

assert probs.shape == (1, model.vocab_size)
assert torch.allclose(probs, logits[:, -1].softmax(-1))
print(f"{model.tokenizer.decode(probs.argmax())!r}  layers={model.num_layers}  hidden={model.hidden_size}")
```

```
' Paris'  layers=12  hidden=768
```

`model.project_on_vocab(x)` applies the final norm and the unembedding, so
`project_on_vocab(layers_output[num_layers - 1])` reproduces `logits` exactly
(max absolute difference 0.0 on GPT-2) — that is the logit lens, with no
per-family plumbing.

## The same script on another architecture

```python
llama_style = StandardizedTransformer("HuggingFaceTB/SmolLM2-135M-Instruct")

with llama_style.trace(prompt):
    resid = llama_style.layers_output[5].save()

assert resid.shape == (1, len(llama_style.tokenizer.encode(prompt)), llama_style.hidden_size)
print(resid.shape)          # torch.Size([1, 9, 576]) — identical code, different family
```

That is the whole value proposition. In plain nnsight the same two runs need
`model.transformer.h[5]` and `model.model.layers[5]` — different paths — and the
unwrapping is decided per family rather than per accessor. On transformers 5 a
modern decoder block returns a plain tensor (GPT-2, Llama, Qwen, GPT-NeoX,
Gemma), while a Bloom or GPT-J block returns a tuple whose first element is the
hidden state, and every attention module returns a `(output, weights)` pair.
Writing `.output[0]` where the block returns a tensor indexes the *sequence*
dimension: you get `torch.Size([10, 768])` where you wanted
`torch.Size([1, 10, 768])`, with no error and a shape that still looks
reasonable. `layers_output[i]` is the same expression on all of them.

## Writing, not just reading

Assignment goes through the same accessors:

```python
with model.trace(prompt):
    baseline = model.logits.save()

with model.trace(prompt):
    model.layers_output[5] = model.layers_output[5] * 0
    ablated = model.logits.save()

assert not torch.allclose(baseline, ablated)
print(f"layer 5 zeroed -> {model.tokenizer.decode(ablated[0, -1].argmax())!r}")
```

And `skip_layers` bypasses a contiguous range — the portable form of layer
ablation. Both ends are inclusive, so `skip_layers(0, 2)` wires `layers_input[0]`
straight to `layers_output[2]` and layers 0, 1 and 2 do not run:

```python
with model.trace(prompt):
    entry = model.layers_input[0].save()
    model.skip_layers(0, 2)
    exit_ = model.layers_output[2].save()
    skipped = model.logits.save()

assert torch.equal(entry, exit_)
assert not torch.allclose(baseline, skipped)
print(skipped.shape)
```

Read `layers_input[0]` *before* calling `skip_layers`: the call consumes that
location, and asking for it afterwards raises `OutOfOrderError`.

## Attention probabilities

nnterp exposes `model.attention_probabilities[i]`, which finds the probability
matrix inside whatever attention implementation the model uses. Enable it at
construction — it also forces `attn_implementation="eager"`:

```python
with_probs = StandardizedTransformer("openai-community/gpt2", enable_attention_probs=True)

with with_probs.trace(prompt):
    pattern = with_probs.attention_probabilities[5].save()

rows = pattern.sum(-1).detach()
assert pattern.shape == (1, with_probs.num_heads, n_tokens, n_tokens)   # [batch, head, query, key]
assert torch.allclose(rows, torch.ones_like(rows), atol=5e-3)           # each row is a distribution
assert pattern.triu(diagonal=1).abs().max() == 0                        # causal: nothing attends ahead
print(pattern.shape, rows.min().item(), rows.max().item())
```

```
torch.Size([1, 12, 10, 10]) 0.9999998211860657 1.0000001192092896
```

The head dimension counts **query** heads, not key/value heads: `repeat_kv` runs
inside the attention forward before the QK product, so `pattern[:, h]` is query
head `h` on a grouped-query model too (Qwen2.5-0.5B: 14 query heads, 2 kv heads,
`pattern.shape[1] == 14`). Use `atol` on the row sums — in bfloat16 they land
within about 3e-3 of 1, not 1e-6.

They are assignable, which is the portable way to intervene on attention. Keep
the replacement causal unless you mean to let tokens read the future:

```python
with with_probs.trace(prompt):
    mask = torch.ones_like(with_probs.attention_probabilities[5]).tril()
    with_probs.attention_probabilities[5] = mask / mask.sum(-1, keepdim=True)
    flattened = with_probs.logits.save()

with with_probs.trace(prompt):
    unflattened = with_probs.logits.save()

assert not torch.allclose(unflattened, flattened)
print(f"flattening layer 5 shifts the logits by {(flattened - unflattened).abs().max().item():.2f}")
```

### Order matters here

The probabilities are computed *inside* the attention forward, so they come
before the attention module's own output. Asking for them after it raises:

<!-- test: expect-error OutOfOrderError -->
```python
with with_probs.trace(prompt):
    attn = with_probs.attentions_output[5].save()
    pattern = with_probs.attention_probabilities[5].save()   # OutOfOrderError
```

The in-order sequence for one layer is `layers_input[i]` →
`attention_probabilities[i]` → `attentions_output[i]` → `mlps_output[i]` →
`layers_output[i]`, with `input_ids` and `token_embeddings` before all of them and
`logits` then `next_token_probs` after.

### Load-time validation

Construction checks the shape, that rows sum to 1, that editing the
probabilities changes the logits, and that the value the `.source` walk reaches
matches the weights the attention module returns — so a model where the accessor
would return the wrong tensor fails at load rather than handing you a plausible
matrix. Check `model.attn_probs_available` when handling arbitrary architectures.

Two constraints this validation carries:

- **It runs on layer 0 only**, on a three-token input
  (`AttentionProbabilitiesAccessor.check_source`, `nnterp/utils.py`
  `dummy_inputs`). On a model whose layers are heterogeneous — Gemma-2/3
  alternating sliding-window and full attention, for instance — layer 0 passing
  says nothing about layer *i*. Spot-check `probs.sum(-1)` at a mid-stack layer
  on a real prompt.
- **fp16 checkpoints are refused.** `enable_attention_probs=True` forces eager
  attention, which does the softmax in the checkpoint dtype where sdpa upcasts,
  and float16 overflows there. `EleutherAI/pythia-70m-deduped` raises
  `RenamingError: ... returns NaN logits with eager attention`. Load it with
  `dtype=torch.float32` or `dtype=torch.bfloat16`.

`check_renaming=False` does not withdraw the accessor; it skips this validation
and warns that the probabilities are unverified for the architecture. Leave it on.

### When the accessor cannot resolve

nnterp reaches the probabilities by drilling into the attention forward with
`.source` (`nnterp/rename_utils.py`, `default_attention_prob_source` and
`first_drillable_op`), and the
operation names there follow whatever transformers writes. If a transformers
release rewrites that line, construction fails with a `RenamingError` naming the
operations it tried. `.source` on the model prints the real names, and works
outside a trace:

```python
print(with_probs.layers[0].self_attn.source)      # the forward, with operation labels
```

Then supply your own `AttnProbFunction`:

```python
from nnterp.rename_utils import (
    ATTENTION_DROPOUT_OPS,
    AttnProbFunction,
    RenameConfig,
    first_available_op,
)


class MyAttnProbs(AttnProbFunction):
    def get_attention_prob_source(self, attention_module, return_module_source=False):
        source = attention_module.source.attention_interface_1.source
        if return_module_source:
            return source
        return first_available_op(source, *ATTENTION_DROPOUT_OPS)


custom = StandardizedTransformer(
    "openai-community/gpt2",
    enable_attention_probs=True,
    rename_config=RenameConfig(attn_prob_source=MyAttnProbs()),
)
assert custom.attn_probs_available
```

`RenameConfig(attn_prob_source=...)` takes an `AttnProbFunction` instance, not an
operation name. For reading patterns without nnterp at all, see the
`attention-analysis` skill.

## Choosing between nnterp and plain nnsight

| Use nnterp when | Use plain nnsight when |
|---|---|
| the script must run on several model families | you are working with one model |
| you want tuple-vs-tensor normalized for you | you need a module nnterp does not name |
| you want load-time validation that paths resolved | you need `.source`, `scan`, custom `edit` plumbing, or vLLM/diffusion |
| writing a reusable tool or library | writing a one-off experiment |

The two compose: `StandardizedTransformer` **is** an nnsight model, so
`tracer.invoke`, `tracer.cache`, `tensor.backward()`, `session`, and `remote=True`
all work on it, and you can drop to raw module paths at any point. The accessor
and the raw path name the same location:

```python
with model.trace() as tracer:
    with tracer.invoke(prompt):
        a = model.layers_output[3].save()          # nnterp accessor
        b = model.transformer.h[3].output.save()   # raw nnsight path, same module
    with tracer.invoke("The Colosseum is in the city of"):
        c = model.layers_output[3].save()

assert torch.equal(a, b)
assert not torch.equal(a, c)
print(a.shape, c.shape)
```

## Notes

- nnterp is on NDIF's remote import whitelist, so `StandardizedTransformer(...,
  remote=True)` works inside remote traces — see the `nnsight-remote` skill.
- Under eager attention the attention module returns `(output, weights)`, and
  `weights` is bit-identical to `attention_probabilities[i]` on every family
  tested. nnterp cross-checks against it at load. It is read-only: the weights
  are returned rather than consumed, so writing to `self_attn.output[1]` changes
  nothing downstream and reports no error. Edit through the accessor.
- `check_renaming=True` (the default) is what makes construction fail loudly on
  an architecture whose modules do not resolve. Turning it off does not fix a
  failing load, it only stops the checking — the accessors still point wherever
  they pointed, now unverified.
- For an architecture nnterp does not know, pass a `rename_config`, or use
  nnsight's own `rename={...}` (see the `nnsight` skill → modules and
  architectures).

## Related skills

- `nnsight` — the underlying API, module paths, and `rename`
- `attention-analysis` — attention patterns via `.source`, and per-head conventions
- `logit-lens`, `activation-patching` — techniques worth writing portably
