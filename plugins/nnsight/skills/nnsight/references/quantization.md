# Quantization

Hold each weight in 4 or 8 bits by naming the format where a dtype goes. It
needs `pip install bitsandbytes accelerate`: neither comes with nnsight, and
without them the first quantized load raises an `ImportError` that names the
missing one. Quantization changes the model's numbers and needs one device;
[tensor-parallel](tensor-parallel.md) keeps the numbers and needs one GPU per
rank.

## Loading

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation",
                          dtype="nf4", dispatch=True)       # where "bfloat16" would go
prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt):
    act = model.transformer.h[5].attn.c_attn.output.save()
    logits = model.lm_head.output.save()

c_attn = model.transformer.h[0].attn.c_attn._module
assert type(c_attn).__name__ == "Linear4bit"                        # same path, another class
assert act.shape == (1, 10, 2304) and act.dtype == torch.bfloat16   # activations: ordinary tensors
assert c_attn.weight.shape == (884736, 1) and c_attn.weight.dtype == torch.uint8   # weights: packed
print(model.tokenizer.decode(logits[0, -1].argmax()))
```

```
 Paris
```

| Name | Format | Bytes per weight |
|---|---|---|
| `nf4`, `int4`, `4bit` | bitsandbytes 4-bit, NF4 | 0.5 |
| `fp4` | bitsandbytes 4-bit, FP4 | 0.5 |
| `int8`, `8bit` | bitsandbytes LLM.int8() | 1.0 |
| `fp8` | transformers block-wise FP8 | 1.0 |

- `fp8` needs a GPU of compute capability 8.9 or later
  (`torch.cuda.get_device_capability()`). Below that nnsight raises `ValueError`
  before any weights are fetched, where transformers alone would load bfloat16.
- The bitsandbytes formats load and run on a CPU as well.
- A `quantization_config=` of your own is passed through. Together with a
  format name it raises.

## What a trace sees

**The module tree is the same.** A quantized linear is another class at the same
path, so every module reference, and a trace written for the unquantized model,
works unchanged.

**Activations arrive in the compute dtype**, which is `bfloat16` (`float16` for
`int8`) whatever the checkpoint declares: gpt2 is a float32 checkpoint and the
activation above is bfloat16, norms and LM head included. Set it with
`compute_dtype=`:

```python
model32 = TransformersModel("openai-community/gpt2", task="text-generation",
                            dtype="nf4", compute_dtype="float32", dispatch=True)
with model32.trace("Hello"):
    out = model32.lm_head.output.save()

assert out.dtype == torch.float32
```

**A raw `.weight` is the packed storage**, a `(N, 1)` `uint8` blob in place of
the matrix, and nothing raises when a weight-space method reads it.
`sum(p.numel() ...)` counts the storage too (82.0M for gpt2 at `nf4`, 124.4M
unquantized). Read activations, or load unquantized when the weights are the
object of study.

### `int8` computes in float16, and its gradients overflow

bitsandbytes implements LLM.int8() in float16. Its narrow exponent overflows on
a backward pass from a large-magnitude metric, and every gradient method then
returns NaN with no error. This example needs a GPU:

```python
int8 = TransformersModel("openai-community/gpt2", task="text-generation",
                         dtype="int8", dispatch=True)

with int8.trace(prompt):
    activation = int8.transformer.h[0].output
    metric = int8.output.logits.sum()                 # large magnitude
    with metric.backward():
        summed = activation.grad.save()

with int8.trace(prompt):
    activation = int8.transformer.h[0].output
    metric = torch.log_softmax(int8.output.logits[0, -1], -1).max()
    with metric.backward():
        normalized = activation.grad.save()

assert summed.dtype == torch.float16
assert torch.isnan(summed).all()                      # every element
assert torch.isfinite(normalized).all()               # a log-probability survives
```

Use a metric on a scale float16 can hold, or `nf4`, which computes in bfloat16
and gives a finite gradient for both. On a CPU a backward pass through an `int8`
model raises `RuntimeError: one of the variables needed for gradient computation
has been modified by an inplace operation`, whatever the metric.

## What it costs

Measured on Llama-3.2-1B against the same model at bfloat16, over the 87
next-token distributions of one passage:

- **4-bit is a different model.** `nf4` picks a different next token at 20% of
  the positions and `fp4` at 25%. Do not compare activations across formats, and
  do not report a 4-bit result as the checkpoint's.
- **`int8` is the accurate format and the slow one**: it agrees at 94% of the
  positions and takes 4.8 times the forward time, where `nf4` takes 1.2 times.
- **It saves less than the arithmetic says.** Embeddings, norms and the LM head
  stay 16-bit, so `nf4` takes 1.01 GB where 0.5 bytes per weight predicts 0.62,
  and `int8` 1.50 where 1.24. Measure with `model._module.get_memory_footprint()`.

**A mixture-of-experts model barely shrinks.** bitsandbytes swaps `nn.Linear`
and nothing else, and transformers holds the experts as stacked 3-D parameters
on one module. At `nf4` a Qwen3-MoE's `q_proj` becomes `Linear4bit` while the
experts' `gate_up_proj` and `down_proj` stay bfloat16. For an MoE that does not
fit, shard it: [tensor-parallel](tensor-parallel.md).

## A checkpoint that ships quantized

A repo whose `config.json` carries a `quantization_config` (a bitsandbytes
4-bit upload, an FP8 or MXFP4 release) is loaded by transformers with the
quantizer that config names. Pass no format name.

- **`dispatch=False` does not apply it.** The meta build has plain linears at
  the checkpoint's dtype, and the quantized classes appear when the weights
  load, at the same paths. `dtype="nf4"` on an ordinary checkpoint behaves the
  same way.
- **A format the machine cannot run is dequantized to bfloat16 with a log line
  and no error**: an FP8 checkpoint below compute capability 8.9 (`We will
  default to dequantizing the model to bf16`), an MXFP4 one without the `kernels`
  package. The model then takes its unquantized size. The `fp8` refusal above
  covers only the name `dtype="fp8"`.
- A format name on such a checkpoint is ignored with a `UserWarning` when it
  belongs to the same backend (an NF4 checkpoint loaded with `dtype="int8"` stays
  4-bit) and raises `ValueError: The model is quantized with ... but you are
  passing a ... config` otherwise.

After any quantized load, check what the weights are held in:

```python
import collections

held = collections.Counter(str(p.dtype) for p in model._module.parameters())
assert held == {"torch.bfloat16": 100, "torch.uint8": 48}      # gpt2 at nf4: 48 packed weights
```

## On NDIF

A client cannot request a quantization. The remote model key is the repo id,
revision, task and `trust_remote_code`, so `dtype=` on a remote model shapes only
your own meta build and the deployment decides how the weights are held. A trace
written against the unquantized tree runs against a quantized deployment as is.

## Diagnosing

| Symptom | Cause |
|---|---|
| `TypeError: ... got an unexpected keyword argument 'load_in_4bit'`, at the end of a `ValueError: Could not load model` | `load_in_4bit=` and `load_in_8bit=` are not accepted. Write `dtype="nf4"` or `dtype="int8"` |
| `AttributeError: module 'torch' has no attribute 'nf8'` | A misspelled format name reaches torch as a dtype. Check it against the table |
| `dtype="int3"` gave a model twice the bfloat16 size | `torch.int1` to `torch.int7` exist, so the name is not rejected. transformers logs `Falling back to torch.float32` and loads float32 |
| `MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16`, once per matmul | `int8` with a `compute_dtype` other than float16 |
| Gradients are all NaN | `int8` and a large-magnitude metric |
| A quantized checkpoint is as large as its bfloat16 original | It was dequantized at load, or it is an MoE |
