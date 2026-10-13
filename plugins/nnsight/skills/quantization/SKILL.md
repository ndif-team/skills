---
name: quantization
description: Load a model in 4 or 8 bits by naming the format where you would name a dtype — TransformersModel(..., dtype="nf4"). Covers the accepted names (nf4/int4/4bit, fp4, int8/8bit, fp8), what a trace sees (module paths and activations are unchanged; raw .weight is a packed blob), the compute dtype and why int8 differs and is several times slower, how much memory it actually saves versus what the arithmetic predicts, the accuracy cost in KL and top-1 agreement, the fp8 refusal below compute capability 8.9, and the two silent failures — int3 loading float32, and MoE experts never being quantized at all. Use when a checkpoint does not fit on the available GPU, when a user asks about bitsandbytes, load_in_4bit, load_in_8bit, BitsAndBytesConfig, quantization_config, NF4, or 8-bit inference, when gradients through a quantized model come back NaN, or when a quantized model's weights look like the wrong shape.
---

# Quantization

Hold each weight in fewer bits instead of splitting the model across GPUs. The
format goes in the dtype slot — no config to build, nothing to import.

> The `fp8` row of the table below needs compute capability 8.9 or later; see
> [`fp8` is refused below compute capability 8.9](#fp8-is-refused-below-compute-capability-89).

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

The unqualified 4-bit names mean **NF4**, what bitsandbytes recommends; `fp4` only
by name.

Passing your own `quantization_config=` still works, but not *together* with a
quantization name — two answers to how the weights are held, so it raises.

Requires `pip install bitsandbytes accelerate`. Neither is a dependency of
nnsight, so a clean install raises `ImportError` at the first quantized load.

### `fp8` is refused below compute capability 8.9

`fp8` is transformers' own quantizer rather than bitsandbytes, and it needs a
4090, an L40S, an H100 or later — Ampere cards (8.0, 8.6) do not qualify.
Transformers alone would silently load bfloat16 there, at twice the width
budgeted; nnsight instead raises `ValueError` below compute capability 8.9, before
any weights are fetched.

<!-- test: gpu -->
```python
if torch.cuda.get_device_capability() < (8, 9):
    try:
        TransformersModel(
            "openai-community/gpt2", task="text-generation", dtype="fp8",
            dispatch=True, device=0,
        )
        raise AssertionError("fp8 was not refused below capability 8.9")
    except ValueError as e:
        assert "compute capability 8.9" in str(e)
```

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

GPT-2 is a **float32** checkpoint, yet the activation comes back `bfloat16`:
quantizing moves the whole model to the compute dtype, norms and LM head included.

**Raw weights are the exception.** A 4-bit weight really is stored packed, so
reading one gives a `uint8` blob rather than the matrix:

<!-- test: gpu -->
```python
weight = model.transformer.h[0].attn.c_attn._module.weight
assert weight.shape == (884736, 1) and weight.dtype == torch.uint8
```

Nothing errors; the shape is just not what a weight-space method assumes. Read
activations, or load unquantized when the weights are the object of study.
`sum(p.numel())` follows the storage too, reporting about half the real count.

## The compute dtype

Everything the format leaves alone (norms, embeddings, the LM head) and everything
the model computes in is `bfloat16` — except **`int8`, which computes in
`float16`**: bitsandbytes implements LLM.int8() that way and casts anything else on
the way in, warning *once per matmul*. Override the compute dtype with
`compute_dtype=` (or its bitsandbytes synonym `bnb_4bit_compute_dtype=`):

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

### `int8` gradients overflow to NaN

float16's narrow exponent overflows on a backward pass over a large-magnitude
loss. This silently kills attribution patching and every other gradient method on
an `int8` model:

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

Use a loss on a scale float16 can hold, or `nf4`, which computes in bfloat16 and
gives a finite gradient for either loss.

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
quantized run as a *different model*, not a cheaper copy — do not compare
activations across widths, and do not report a 4-bit result as the checkpoint's.
A hidden-state norm hides this: it moves about 3% at `nf4`. The damage shrinks
with size (Llama-3.1-8B: `nf4` 92.9% agreement).

**`int8` is the accurate format and the slow one** — about 3x bfloat16's forward,
which for a sweep over hundreds of forwards usually decides it in `nf4`'s favor.

**It saves less than the arithmetic predicts.** Embeddings, norms and the LM head
stay 16-bit and each block stores a scale, so `nf4` takes 1.00 GB where 0.5
bytes/weight predicts 0.58 — worst on small models with large vocabularies.
Budget from measurement.

## What it will not do

**A mixture-of-experts model barely shrinks.** bitsandbytes swaps `nn.Linear` only,
and transformers 5 holds the experts as stacked 3-D parameters, so only attention,
router and shared layers are quantized: Qwen1.5-MoE-A2.7B at `nf4` goes from 12.89
to 12.53 GiB, while every routing decision is perturbed. For an MoE that does not
fit, use the `tensor-parallel` skill.

**A narrow torch dtype that nothing can load falls back to float32.**
`torch.int1` through `torch.int7` exist, so `dtype="int3"` is not rejected as a
name: transformers logs `Falling back to torch.float32 because loading with the
original dtype failed on the target device` and loads float32 — twice bfloat16's
size, with no exception.

**`load_in_4bit=` / `load_in_8bit=` are not transformers 5 arguments.** The
transformers 4 tutorials' spelling surfaces deep in a nested traceback as
`TypeError: LlamaForCausalLM.__init__() got an unexpected keyword argument
'load_in_4bit'`. Rewrite as `dtype="nf4"` or `dtype="int8"`.

**A GPU is not required**: with bitsandbytes 0.50, `nf4` loads and runs on CPU.
What the quantizers reject is the `meta` device — so `dispatch=False` builds the
architecture at the compute dtype, ignoring the quantization, and weights are
quantized only when actually loaded.

## On NDIF

A **client cannot request a quantization**; the deployment decides. A remote model
key is the repo id, revision and task — nothing about how the weights are held —
so `dtype=` on a `remote=True` model shapes only your own meta build. Because the
module tree is unchanged, a trace written against an unquantized model runs
against a quantized deployment as is.

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
| `ValueError: dtype='nf4' and an explicit quantization_config ...` | Both were passed; keep one |
| `AttributeError: module 'torch' has no attribute 'nf8'` | A misspelled format name reaches torch as a dtype. Check it against the table |
| `ImportError` naming bitsandbytes or accelerate | `pip install bitsandbytes accelerate` |

## Choosing between this and tensor parallelism

**Quantization** costs accuracy and needs one GPU; **tensor parallelism** (the
`tensor-parallel` skill) is numerically faithful but needs N GPUs and `torchrun`,
and puts rules on your intervention code. Prefer tensor parallelism when the
activations are the result, and for any MoE; quantization when you are limited to
one card.

**They compose**, and for a model that fits neither way alone they have to:
transformers shards the packed weights, and the gather works through the
quantization — on Llama-3.3-70B-Instruct at `nf4` and tp=4,
`layers[40].mlp.gate_proj.output` reads at its full 28672, and the weights take
about 43 GB in total against ~141 GB for bfloat16.
