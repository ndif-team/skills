# Graph taps

vLLM's decode throughput comes from replaying CUDA graphs, and a replayed graph runs
no Python — so the eager engine (the default) gives that up to serve every location.
`taps=` names the locations to record *into* the graph as breaks and serve on every
replay; everything else is vanilla vLLM with graphs on. The same trace syntax,
per-request scoping, `tracer.iter`, `model.logits` and `model.samples` work.

<!-- test: gpu setup -->
```python
import torch
import nnsight
from nnsight.modeling.vllm import VLLM

model = VLLM("HuggingFaceTB/SmolLM2-135M-Instruct", dispatch=True,
             gpu_memory_utilization=0.2, max_model_len=1024,
             taps=["model.layers.*.output", "model.layers.8.mlp.input"])
print(len(model.taps), model.taps[:2])
# 31 ('model.model.layers.0.output', 'model.model.layers.1.output')
assert "model.model.layers.8.mlp.input" in model.taps
```

`*` matches one path segment; the `model.` prefix is implied and `model.taps`
reports the resolved names; a tap that names no module is refused at construction.
Keep the set small — each tap splits the graph.

<!-- test: gpu -->
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

print(repr(plain.outputs[0].text), "->", repr(steered.outputs[0].text))
assert len(hs) == 8 and not torch.equal(hs[1], hs[2])
assert steered.outputs[0].text != plain.outputs[0].text
```

What changes under graphs:

- **A tap can be a `.source` op** — `"model.layers.10.self_attn.source.qkv_split_0.output"`:
  the worker instruments that forward before recording, and the op is served on
  replay (reads equal eager exactly; in-place edits land). Ops inside fused kernels
  are still not locations. An op name the forward does not have is refused while the
  engine builds, and the caller sees only `RuntimeError: Engine core initialization
  failed` — the message listing the ops it *does* have is in the `(EngineCore pid=...)`
  output above it.
- **Only taps are served.** A read of any other module location fails when the
  request ends with `'...' is not a tap on this engine`. `model.logits`,
  `model.samples` and `tracer.result` always work.
- **Edits land in place.** `x[:] += v` is exactly right; a replacement
  (`layer.output = t`) is copied back into the graph's memory and must keep the rows
  the block owns — a short one is refused with `A batched write has to keep its rows:
  ... must be (9, 576), not (2, 576)` before the model sees it.
- **Clone what you keep.** The value served *is* the graph's memory, rewritten
  next step. An un-cloned list still comes back as N separate tensors (each is
  copied at collect time), but every decode entry holds the last step's values and
  nothing warns. The prefill entry is a different buffer, sized to the prompt, so it
  survives — which is why a list of un-cloned captures looks partly right.
- **Steer inside the `iter` loop.** An edit written before the loop fires on the
  prefill only.
- **`torch.compile` is off** for a tapped engine; the switch is process-wide.
- **Hybrid and recurrent trunks replay graphs for decode only.** On any model
  vLLM reports as hybrid or attention-free (Qwen3.5, Qwen3.6, Mamba, ...) a tapped
  engine pins `cudagraph_mode="FULL_DECODE_ONLY"`: a full graph captured over a
  gated-delta-net layer silently miscomputes the other batch composition. Tapped
  generation matches eager exactly there; your own `compilation_config` wins.

<!-- test: gpu expect-error RuntimeError -->
```python
with model.trace(prompt, temperature=0.0, max_tokens=1):
    x = model.model.layers[3].self_attn.o_proj.input.save()      # not a tap
```

## Measured

Llama-3.1-8B, bf16, A100, 512-token prompt, 128 new tokens, greedy, capturing one
layer every step. Vanilla and taps in tokens per second, taps with its share of
vanilla in parentheses; the eager column is a share only:

| | vanilla | eager | taps |
|---|---:|---:|---:|
| 8B, 1 GPU | 92 | 85% | 89 (96%) |
| 8B, tp=4 | 229 | 29% | 213 (93%) |
| 8B, tp=8 | 313 | 20% | 284 (91%) |
| 70B, tp=8 | 61 | 46% | 58 (95%) |

The eager column is a share, not a rate, because an eager engine's rate is not a
property of the card: it spends a Python round trip per module call on the driver,
so it follows the host's spare CPU (8B plain generation: 86 tok/s on a quiet box,
54 on a busy one, where a tapped engine gives 89 on both). It is also not nnsight's
cost — plain `vllm.LLM(..., enforce_eager=True)` measures the same as `VLLM(...)`
generating with no trace (69.6 / 69.7, 52.1 / 54.1, and 39.3 / 39.0 at tp=2). What
taps buy is the graph, and the graph is what scales: declare them whenever the GPUs
outnumber one. On new architectures the gap to vanilla is wider (Qwen3.5-0.8B: taps
59% of vanilla, eager 11%) because vanilla's `torch.compile` pays there and a tapped
engine runs without it.
