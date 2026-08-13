---
name: quantization
description: Load a model in 4 or 8 bits by naming the format where you would name a dtype — TransformersModel(..., dtype="nf4"). Covers the accepted names (nf4/int4/4bit, fp4, int8/8bit, fp8), what a trace sees (module paths and activations are unchanged; raw .weight is a packed blob), the compute dtype and why int8 differs, how much memory it actually saves versus what the arithmetic predicts, and the accuracy cost. Use when a checkpoint does not fit on the available GPU, when a user asks about bitsandbytes, load_in_4bit, load_in_8bit, BitsAndBytesConfig, quantization_config, NF4, or 8-bit inference, or when a quantized model's weights look like the wrong shape.
---

# Quantization

Hold each weight in fewer bits instead of splitting the model across GPUs. The
format goes in the dtype slot — there is no config to build and nothing to
import.

> Verified on nnsight 0.8, transformers 5.15, bitsandbytes 0.50, one A100.

## Loading

<!-- test: gpu -->
```python
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel(
    "openai-community/gpt2",
    task="text-generation",
    dtype="nf4",          # where you would write "bfloat16"
    dispatch=True,
    device=0,
)

print(type(model.transformer.h[0].attn.c_attn._module).__name__)   # Linear4bit
```

| Name | What you get | Bytes/weight |
|---|---|---|
| `nf4`, `int4`, `4bit` | bitsandbytes 4-bit, NF4 | 0.5 |
| `fp4` | bitsandbytes 4-bit, FP4 | 0.5 |
| `int8`, `8bit` | bitsandbytes LLM.int8() | 1.0 |
| `fp8` | transformers block-wise FP8 (H100+) | 1.0 |

Several names for one thing on purpose — someone reaching for 4-bit writes
whichever of `int4` / `4bit` / `nf4` they last read about. The unqualified names
mean **NF4**, what bitsandbytes recommends; `fp4` is reached only by asking for
it by name. A name that is neither a torch dtype nor one of these raises rather
than guessing at a width.

Passing your own `quantization_config=` still works, but not *together* with a
quantization name — two answers to how the weights are held, so it raises.

## Your intervention code does not change

The module tree is **identical**: a quantized linear is a different class holding
a packed weight, but it sits at the same path with the same children, so every
module reference, envoy and remote request lines up. Activations are ordinary
16-bit tensors of the usual shape.

<!-- test: gpu -->
```python
with model.trace("The Eiffel Tower is in the city of"):
    attn = model.transformer.h[5].attn.c_attn.output.save()
    logits = model.lm_head.output.save()

print(attn.shape, attn.dtype)      # (1, 11, 2304) torch.bfloat16 — as if unquantized
print(model.tokenizer.decode(logits[0, -1].argmax()))
```

**Raw weights are the exception.** A 4-bit weight really is stored packed, so
reading one gives a `uint8` blob rather than the matrix:

<!-- test: gpu -->
```python
weight = model.transformer.h[0].attn.c_attn._module.weight
print(weight.shape, weight.dtype)   # torch.Size([884736, 1]) torch.uint8
```

Read activations, not weights — or load unquantized when the weights themselves
are the object of study. Nothing errors here; the shape is simply not the one a
weight-space method assumes.

## The compute dtype

Everything the format leaves alone — norms, embeddings, the LM head — and
everything the model computes in is `bfloat16`, with one exception.

**`int8` computes in `float16`.** bitsandbytes implements LLM.int8() that way and
casts anything else on the way in, warning *once per matmul* as it does (over a
hundred lines for one short forward). So an `int8` model's activations arrive as
`float16`, not `bfloat16`. This is deliberate and also more accurate — see the
table below.

Override it when you need to:

<!-- test: gpu -->
```python
model32 = TransformersModel(
    "openai-community/gpt2", task="text-generation",
    dtype="nf4", compute_dtype="float32", dispatch=True, device=0,
)
with model32.trace("Hello"):
    out = model32.lm_head.output.save()
print(out.dtype)   # torch.float32
```

`bnb_4bit_compute_dtype=` is accepted as a synonym, for anyone arriving from the
bitsandbytes documentation.

## What it costs

Llama-3.2-1B, layer-5 hidden-state norm against the unquantized run:

| dtype | GPU allocated | norm | next token |
|---|---|---|---|
| `bfloat16` | 2.47 GB | 422.17 | ` Paris` |
| `int8` | 1.50 GB | 422.07 | ` Paris` |
| `nf4` | 1.07 GB | 408.76 | ` Paris` |
| `fp4` | 1.07 GB | 384.41 | ` Paris` |

**4-bit is a real perturbation**, a few percent on hidden-state norms and growing
with depth. Treat a quantized run as a *different model*, not a cheaper copy of
the same one — do not compare activations across widths, and do not report a
4-bit result as though it were the checkpoint's.

`int8` is nearly free in accuracy and saves less memory; `nf4` is the reverse.
`fp4` is both worse and no smaller than `nf4`, which is why the unqualified names
point at NF4.

**It saves less than the arithmetic predicts.** The format leaves embeddings,
norms and the LM head in 16 bits and stores a scale per block, none of which is
in a parameter count — so `nf4` really takes 1.07 GB where 0.5 bytes/weight
predicts 0.62. The gap is worst for models whose vocabulary is a large fraction
of them (small models, large tokenizers). Budget from measurement.

## Requirements

- **A GPU.** bitsandbytes cannot quantize on CPU or on meta.
- **`pip install bitsandbytes`** for everything except `fp8`.
- **H100 or later for `fp8`**, which is transformers' own quantizer rather than
  bitsandbytes. transformers checks the hardware and raises.

`dispatch=False` still works: the **meta build ignores the quantization** and
builds the architecture at the compute dtype. That is what makes the lazy path
work, and what lets a client model a checkpoint a server holds quantized. Weights
are quantized only when they are actually loaded.

## On NDIF

The same names configure a deployment (`ndif deploy <model> --dtype nf4`), so a
server can hold a model 4-bit with nothing client-side changing — a trace written
against an unquantized replica works against a quantized one.

A **client cannot request it**. A remote model key is the repo id and revision and
says nothing about how the weights are held, so `dtype=` on a `remote=True` model
shapes only your own meta build. The deployment decides.

## Diagnosing

| Symptom | Cause |
|---|---|
| `.weight` is a `(N, 1)` uint8 tensor | Expected — 4-bit weights are packed. Read activations instead |
| Activations are `float16` when you expected `bfloat16` | `int8`, which computes in float16 by design. Pass `compute_dtype="bfloat16"` if you must, and accept the per-matmul warnings |
| `MatMul8bitLt: inputs will be cast ...` repeated per matmul | An `int8` model given a non-float16 compute dtype |
| Model still takes far more GPU than 0.5 bytes/weight | Expected — embeddings, norms and the LM head stay 16-bit |
| `ValueError: dtype='nf4' and an explicit quantization_config ...` | Both were passed; keep one |
| `ValueError: Unknown dtype ...` | Not a torch dtype and not one of the names above — check the spelling |
| `ValueError: Cannot size a checkpoint held as 'int3' ...` | torch has `int1`–`int7` but nothing here loads them; use `int4`/`nf4` |
| Results differ from a published unquantized number | Expected at 4-bit; see the accuracy table |
| ImportError from bitsandbytes | `pip install bitsandbytes`, and check a CUDA GPU is visible |

## Choosing between this and tensor parallelism

Both make a model that does not fit, fit. **Quantization** costs accuracy and
needs one GPU; **tensor parallelism** (see the `tensor-parallel` skill) is
numerically faithful but needs N GPUs and `torchrun`, and puts two rules on your
intervention code. Prefer tensor parallelism when the activations are the result;
prefer quantization when you are constrained to one card, or when the model is so
large that N cards is not on offer. They are not mutually exclusive in principle,
but the combination is untested.
