---
name: quantization
description: Load a model in 4 or 8 bits by naming the format where you would name a dtype — TransformersModel(..., dtype="nf4"). Covers the accepted names (nf4/int4/4bit, fp4, int8/8bit, fp8), what a trace sees (module paths and activations are unchanged; raw .weight is a packed blob), the compute dtype and why int8 differs and is several times slower, how much memory it actually saves versus what the arithmetic predicts, the accuracy cost in KL and top-1 agreement, and the three silent failures — fp8 loading unquantized below compute capability 8.9, int3 loading float32, and MoE experts never being quantized at all. Use when a checkpoint does not fit on the available GPU, when a user asks about bitsandbytes, load_in_4bit, load_in_8bit, BitsAndBytesConfig, quantization_config, NF4, or 8-bit inference, when gradients through a quantized model come back NaN, or when a quantized model's weights look like the wrong shape.
---

# Quantization

Hold each weight in fewer bits instead of splitting the model across GPUs. The
format goes in the dtype slot — there is no config to build and nothing to
import.

> Verified on nnsight 0.8, transformers 5.15, bitsandbytes 0.50, one RTX A6000
> (compute capability 8.6). The `fp8` row of the table below cannot be verified
> on that card, or on an A100; see [`fp8` is silent](#fp8-is-silent-below-compute-capability-89).

## Loading

<!-- test: gpu -->
```python
import torch
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel(
    "openai-community/gpt2",
    task="text-generation",
    dtype="nf4",          # where you would write "bfloat16"
    dispatch=True,
    device=0,
)

assert type(model.transformer.h[0].attn.c_attn._module).__name__ == "Linear4bit"
```

| Name | What you get | Bytes/weight |
|---|---|---|
| `nf4`, `int4`, `4bit` | bitsandbytes 4-bit, NF4 | 0.5 |
| `fp4` | bitsandbytes 4-bit, FP4 | 0.5 |
| `int8`, `8bit` | bitsandbytes LLM.int8() | 1.0 |
| `fp8` | transformers block-wise FP8, compute capability 8.9+ | 1.0 |

Several names for one thing on purpose — someone reaching for 4-bit writes
whichever of `int4` / `4bit` / `nf4` they last read about. The unqualified names
mean **NF4**, what bitsandbytes recommends; `fp4` is reached only by asking for
it by name.

Passing your own `quantization_config=` still works, but not *together* with a
quantization name — two answers to how the weights are held, so it raises.

Requires `pip install bitsandbytes accelerate`. Neither is a dependency of
nnsight, so a clean install raises `ImportError` at the first quantized load.

### `fp8` is silent below compute capability 8.9

`fp8` is transformers' own quantizer rather than bitsandbytes, and it needs a
4090, an L40S, an H100 or later. On older hardware it does **not** raise. It logs
a warning, sets `dequantize` on the quantization config, and loads bfloat16 —
while leaving the `FineGrainedFP8HfQuantizer` attached, so anything that inspects
`hf_quantizer` is told the model is quantized.

Measured on Llama-3.2-1B on an A6000 (8.6): `Linear` rather than `FP8Linear`, the
same 2.30 GB as bfloat16, and a KL of 0.000000 against the bfloat16 run — the same
model, bit for bit, at twice the width the caller budgeted. An **A100 is 8.0 and
does not qualify either**, so a user asking for `fp8` on the most common
interpretability card gets an unquantized run reported as an 8-bit result.

<!-- test: gpu -->
```python
fp8 = TransformersModel(
    "openai-community/gpt2", task="text-generation", dtype="fp8",
    dispatch=True, device=0,
)

if torch.cuda.get_device_capability() < (8, 9):
    assert fp8._module.config.quantization_config.dequantize is True
    assert type(fp8.transformer.h[0].attn.c_attn._module).__name__ == "Conv1D"
```

`config.quantization_config.dequantize` is the check. `True` means the weights
are bfloat16 whatever the name asked for.

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

assert attn.shape == (1, 10, 2304) and attn.dtype == torch.bfloat16
print(model.tokenizer.decode(logits[0, -1].argmax()))
```

GPT-2 is a **float32** checkpoint, and that block shows what quantizing one does:
the activation comes back `bfloat16`, not the `float32` the unquantized model
gives. Quantizing moves the whole model to the compute dtype, norms and LM head
included. On a checkpoint that was already 16-bit, activations are unchanged.

**Raw weights are the exception.** A 4-bit weight really is stored packed, so
reading one gives a `uint8` blob rather than the matrix:

<!-- test: gpu -->
```python
weight = model.transformer.h[0].attn.c_attn._module.weight
assert weight.shape == (884736, 1) and weight.dtype == torch.uint8
```

Read activations, not weights — or load unquantized when the weights themselves
are the object of study. Nothing errors here; the shape is simply not the one a
weight-space method assumes. `sum(p.numel())` follows the storage too, so a 4-bit
model reports roughly half its real parameter count.

## The compute dtype

Everything the format leaves alone — norms, embeddings, the LM head — and
everything the model computes in is `bfloat16`, with one exception.

**`int8` computes in `float16`.** bitsandbytes implements LLM.int8() that way and
casts anything else on the way in, warning *once per matmul* as it does (over a
hundred lines for one short forward). So an `int8` model's activations arrive as
`float16`, not `bfloat16`.

Override it when you need to:

<!-- test: gpu -->
```python
model32 = TransformersModel(
    "openai-community/gpt2", task="text-generation",
    dtype="nf4", compute_dtype="float32", dispatch=True, device=0,
)
with model32.trace("Hello"):
    out = model32.lm_head.output.save()
assert out.dtype == torch.float32
```

`bnb_4bit_compute_dtype=` is accepted as a synonym, for anyone arriving from the
bitsandbytes documentation.

### `int8` gradients overflow to NaN

float16 has a much narrower exponent than bfloat16, and a backward pass over a
large-magnitude loss runs out of it. This kills attribution patching and every
other gradient-based method on an `int8` model, silently: nothing raises, and the
NaNs propagate into the attribution score.

<!-- test: gpu -->
```python
int8 = TransformersModel(
    "openai-community/gpt2", task="text-generation",
    dtype="int8", dispatch=True, device=0,
)

with int8.trace("The Eiffel Tower is in the city of"):
    activation = int8.transformer.h[0].output
    loss = int8.output.logits.sum()               # large magnitude
    with loss.backward():
        summed = activation.grad.save()

with int8.trace("The Eiffel Tower is in the city of"):
    activation = int8.transformer.h[0].output
    loss = torch.log_softmax(int8.output.logits[0, -1], -1).max()
    with loss.backward():
        normalized = activation.grad.save()

assert torch.isnan(summed).all()                  # every element
assert torch.isfinite(normalized).all()           # a normalized loss survives
```

Use a loss on a scale float16 can hold, or use `nf4`, which computes in bfloat16
and gives a finite gradient for either loss (1079000.0 for `logits.sum()` on this
model, against float32's 934879.56).

## What it costs

Llama-3.2-1B, over the 86 next-token distributions of a fixed passage. **KL** is
against the same model at `bfloat16` and **top-1 agreement** is how often the two
pick the same next token:

| dtype | weights | vs bf16 | mean KL | top-1 agreement | forward |
|---|---|---|---|---|---|
| `bfloat16` | 2.30 GB | 1.00x | — | — | 14–18 ms |
| `int8` | 1.40 GB | 0.61x | 0.011 | 95.3% | 50–67 ms |
| `nf4` | 1.00 GB | 0.43x | 0.143 | 87.1% | 20–24 ms |
| `fp4` | 1.00 GB | 0.43x | 0.182 | 82.4% | — |

**4-bit changes the argmax next token more than one time in ten.** Treat a
quantized run as a *different model*, not a cheaper copy of the same one — do not
compare activations across widths, and do not report a 4-bit result as though it
were the checkpoint's. A hidden-state norm is not the measure to quote here: it
moves about 3% at `nf4` while the argmax under it changes on one token in eight.

The damage shrinks as the model grows: the same measurement on Llama-3.1-8B gives
`nf4` 92.9% agreement against 87.1% on the 1B.

`fp4` is worse than `nf4` at exactly the same size, which is why the unqualified
names point at NF4.

**`int8` is the accurate format and the slow one.** It was at least 2.9x
bfloat16's forward in every run (the card is shared, so the exact ratio moves).
For a sweep over hundreds of forwards that usually decides it in `nf4`'s favor,
against `int8`'s better accuracy.

**It saves less than the arithmetic predicts.** The format leaves embeddings,
norms and the LM head in 16 bits and stores a scale per block, none of which is
in a parameter count — so `nf4` really takes 1.00 GB where 0.5 bytes/weight
predicts 0.58. Counting the embeddings at 2 bytes and the rest at the format's
width predicts 0.94. The gap is worst for models whose vocabulary is a large
fraction of them (small models, large tokenizers). Budget from measurement.

## What it will not do

**A mixture-of-experts model barely shrinks.** bitsandbytes swaps `nn.Linear` and
nothing else, and transformers 5 holds the experts as stacked 3-D parameters on
one module rather than as linears. Only the attention projections, the router and
the shared layers are quantized, and those are the minority of an MoE's weights.
Qwen1.5-MoE-A2.7B at `nf4` goes from 12.89 GiB to 12.53 GiB, under 3%, while
every routing decision downstream of the quantized attention is perturbed. For an
MoE that does not fit, use tensor parallelism (the `tensor-parallel` skill).

**A narrow torch dtype that nothing can load falls back to float32.**
`torch.int1` through `torch.int7` exist, so `dtype="int3"` is not rejected as a
name. transformers tries it, fails, logs `Falling back to torch.float32 because
loading with the original dtype failed on the target device`, and loads float32 —
4.60 GB on Llama-3.2-1B against bfloat16's 2.30. A request for something narrower
returns something twice as wide, with no exception. (`dispatch=False` does raise:
`... cannot be instantiated under dtype=torch.int3 as it's not a floating-point
dtype`.)

**`load_in_4bit=` / `load_in_8bit=` are not transformers 5 arguments.** They are
what every tutorial written against transformers 4 passes. In 5 they reach the
model class as a stray kwarg and surface fifteen lines down a nested traceback,
under a `ValueError` about loading classes:

```
TypeError: LlamaForCausalLM.__init__() got an unexpected keyword argument 'load_in_4bit'
```

Rewrite as `dtype="nf4"` or `dtype="int8"`.

**A GPU is not required**, contrary to what bitsandbytes' reputation suggests: on
0.50, `nf4` loads and runs on CPU, reaching the same layer-5 norm as the GPU to
five significant figures. What the quantizers reject is the `meta` device.

`dispatch=False` still works because of that: the **meta build ignores the
quantization** and builds the architecture at the compute dtype. That is what
makes the lazy path work, and what lets a client model a checkpoint a server
holds quantized. Weights are quantized only when they are actually loaded.

## On NDIF

The same names configure a deployment (`ndif deploy <model> --dtype nf4`), so a
server can hold a model 4-bit with nothing client-side changing — a trace written
against an unquantized replica works against a quantized one.

A **client cannot request it**. A remote model key is the repo id and revision and
says nothing about how the weights are held, so `dtype=` on a `remote=True` model
shapes only your own meta build. The deployment decides.

Placement uses the nominal bytes/weight from the table above, which undercounts by
the margin shown, so a quantized deployment needs more padding than NDIF's
default 0.15.

## Diagnosing

| Symptom | Cause |
|---|---|
| `.weight` is a `(N, 1)` uint8 tensor | Expected — 4-bit weights are packed. Read activations instead |
| Activations are `float16` when you expected `bfloat16` | `int8`, which computes in float16 by design. Pass `compute_dtype="bfloat16"` if you must, and accept the per-matmul warnings |
| `MatMul8bitLt: inputs will be cast ...` repeated per matmul | An `int8` model given a non-float16 compute dtype |
| Gradients through the model are all NaN | `int8` plus a large-magnitude loss overflowing float16. Use a normalized loss, or `nf4` |
| An `fp8` model is exactly as big as the bfloat16 one | Compute capability below 8.9. Check `config.quantization_config.dequantize` |
| An MoE saved almost nothing | Its experts are not `nn.Linear`, so bitsandbytes never touched them |
| `dtype="int3"` gave a model twice the expected size | float32 fallback. Use `int4` / `nf4` |
| `TypeError: ... unexpected keyword argument 'load_in_4bit'` | transformers 4 spelling; use `dtype="nf4"` |
| Model still takes far more GPU than 0.5 bytes/weight | Expected — embeddings, norms and the LM head stay 16-bit |
| `ValueError: dtype='nf4' and an explicit quantization_config ...` | Both were passed; keep one |
| `AttributeError: module 'torch' has no attribute 'nf8'` | A misspelled format name reaches torch as a dtype. Check it against the table |
| `ImportError` naming bitsandbytes or accelerate | `pip install bitsandbytes accelerate` |
| `ValueError: Unknown dtype ...` / `Cannot size a checkpoint held as 'int3' ...` | Server-side sizing, not loading — an NDIF deployment named a format `bytes_per_element` does not accept |

## Choosing between this and tensor parallelism

Both make a model that does not fit, fit. **Quantization** costs accuracy and
needs one GPU; **tensor parallelism** (see the `tensor-parallel` skill) is
numerically faithful but needs N GPUs and `torchrun`, and puts two rules on your
intervention code. Prefer tensor parallelism when the activations are the result,
and for any MoE; prefer quantization when you are constrained to one card, or
when the model is so large that N cards is not on offer.

**They compose**, and for a model that fits neither way alone they have to.
Verified on Llama-3.3-70B-Instruct at `nf4` across 4 A100s: transformers shards
the packed weights, and `layers[40].mlp.gate_proj.output` reads at its full 28672
rather than one rank's 7168 — so the gather works through the quantization. The
weights took 43.3 GB across the four cards against ~141 GB for bfloat16.
