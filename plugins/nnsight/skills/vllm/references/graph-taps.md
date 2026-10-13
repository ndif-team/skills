# Graph taps

A replayed CUDA graph runs no Python, so the default eager engine gives up graphs
to serve every location. `taps=` names the locations to record *into* the graph
as breaks and serve on every replay; everything else is vanilla vLLM with graphs
on. Trace syntax, `tracer.iter`, `model.logits` and `model.samples` are unchanged.

<!-- test: gpu setup requires vllm -->
```python
import torch
import nnsight
from nnsight.modeling.vllm import VLLM

model = VLLM("HuggingFaceTB/SmolLM2-135M-Instruct", dispatch=True,
             gpu_memory_utilization=0.2, max_model_len=1024,
             taps=["model.layers.*.output", "model.layers.8.mlp.input"])
print(len(model.taps), model.taps[:2])   # 31 ('model.model.layers.0.output', ...)
assert "model.model.layers.8.mlp.input" in model.taps
```

`*` matches one path segment; the `model.` prefix is implied; a tap that names no
module is refused at construction. Keep the set small — each tap splits the graph.

<!-- test: gpu requires vllm -->
```python
torch.manual_seed(0)
v = torch.randn(576, dtype=torch.bfloat16, device="cuda")
v = 40.0 * v / v.norm()
prompt = "The capital of France is"

with model.trace(prompt, temperature=0.0, max_tokens=8) as tracer:
    plain = tracer.result.save()

with model.trace(prompt, temperature=0.0, max_tokens=8, ignore_eos=True) as tracer:
    hs = nnsight.save([])
    for _ in tracer.iter[:8]:                                   # every step, prefill included
        model.model.layers[6].output[0][:] += v                 # in place
        hs.append(sum(model.model.layers[12].output)[-1].clone())   # clone: graph memory
    steered = tracer.result.save()

assert len(hs) == 8 and not torch.equal(hs[1], hs[2])
assert steered.outputs[0].text != plain.outputs[0].text
```

What changes under graphs:

- **A tap can be a `.source` op** — `"model.layers.10.self_attn.source.qkv_split_0.output"`:
  served on replay, reads equal to eager, in-place edits land. Ops inside fused
  kernels are not locations. An unknown op name fails the engine build; the list
  of ops the forward *does* have is in the `(EngineCore pid=...)` log.
- **Only taps are served.** A read of any other module location fails when the
  request ends with `'...' is not a tap on this engine`. `model.logits`,
  `model.samples` and `tracer.result` always work.
- **Edits land in place.** `x[:] += v` is exactly right; a replacement
  (`layer.output = t`) is copied back into the graph's memory and must keep the
  block's rows — a short one can take the engine down, not just the request.
- **Clone what you keep.** The value served *is* the graph's memory, rewritten
  next step: an un-cloned list comes back with every decode entry holding the last
  step's values (the prefill entry, a separate buffer, survives — so it looks
  partly right), and nothing warns.
- **Steer inside the `iter` loop.** An edit written before the loop fires on the
  prefill only.
- **`torch.compile` is off** for a tapped engine; the switch is process-wide.
- **Hybrid and recurrent trunks replay graphs for decode only.** On a model vLLM
  reports as hybrid or attention-free (Qwen3.5, Qwen3.6, Mamba, ...) a tapped
  engine pins `cudagraph_mode="FULL_DECODE_ONLY"` (a full graph over a
  gated-delta-net layer miscomputes); tapped generation matches eager exactly.
  Your own `compilation_config` wins.

<!-- test: gpu requires vllm expect-error RuntimeError -->
```python
with model.trace(prompt, temperature=0.0, max_tokens=1):
    x = model.model.layers[3].self_attn.o_proj.input.save()      # not a tap
```

## Measured

Llama-3.1-8B and -70B, bf16, 512-token prompt, 128 new tokens, greedy, capturing
one layer every step. Tokens per second, with taps' share of vanilla in
parentheses:

| | vanilla | eager | taps |
|---|---:|---:|---:|
| 8B, 1 GPU | 92 | 85% | 89 (96%) |
| 8B, tp=4 | 229 | 29% | 213 (93%) |
| 8B, tp=8 | 313 | 20% | 284 (91%) |
| 70B, tp=8 | 61 | 46% | 58 (95%) |

An eager engine's rate depends on the host's spare CPU (a Python round trip per
module call), so that column is a share, not a rate; taps keep the graph, and the
graph is what scales — declare them whenever the GPUs outnumber one.
