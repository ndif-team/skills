# Tensor parallelism, MoE and hybrid trunks

## Tensor parallelism is transparent to the block

<!-- test: gpu slow setup -->
```python
import torch
import nnsight
from nnsight.modeling.vllm import VLLM

model = VLLM("Qwen/Qwen2.5-0.5B-Instruct", dispatch=True, tensor_parallel_size=2,   # 14 heads: splits in two
             gpu_memory_utilization=0.2, max_model_len=1024)

with model.trace("The capital of France is", temperature=0.0, max_tokens=1):
    layer = model.model.layers[10]
    o_in = layer.self_attn.o_proj.input            # reads in forward order: attention,
    d_in = layer.mlp.down_proj.input               # then MLP, then the layer's output
    out = layer.output
    shapes = nnsight.save({
        "o_proj.input": tuple(o_in.shape),
        "down_proj.input": tuple(d_in.shape),
        "layer.output": (tuple(out[0].shape), tuple(out[1].shape)),
        "o_proj.weight (this rank)": tuple(layer.self_attn.o_proj._module.weight.shape),
    })
    h = (out[0] + out[1])[-1:]
    lens = model.logits_processor(model.lm_head, model.model.norm(h)).argmax(-1).item().save()

print(shapes, model.tokenizer.decode(lens))
assert shapes["o_proj.input"][-1] == 896 and shapes["o_proj.weight (this rank)"][-1] == 448
```

nnsight gathers a column-parallel output or a row-parallel input into the whole
tensor before your code reads it and re-splits what you write, so every rank runs
identical code against the complete tensor: `qkv_proj.output`,
`gate_up_proj.output`, `o_proj.input`, `down_proj.input` read at full width;
layer outputs and `norm.output` are whole on every rank already. **Parameters are
not gathered** — `weight` is this rank's slice, which is how you tell what is
sharded (the block above sees an 896-wide `o_proj` input against a 448-wide weight).

- Every rank runs the block: no rank-dependent control flow, no partial
  collectives. A tensor referenced from outside travels to every rank; an in-block
  `torch.randn` agrees across ranks (vLLM seeds every worker alike).
- The client-side `print(model)` shows `tp_size=1` whatever you asked for; check
  `._module.weight.shape` inside the block.
- A fused projection (`qkv_proj`, `gate_up_proj`) gathers in rank order —
  `[q₀ k₀ v₀ | q₁ k₁ v₁]` — so slice it by head, not by `[:q_size]`.
- **The logit lens** goes through the model's logits path, which gathers the vocab
  shards: `model.logits_processor(model.lm_head, model.model.norm(h))`.
  `model.lm_head(h)` raises `LMHead's weights should be used in the sampler`.
  `norm(h)` returns a tensor; `norm(h, residual)` is the fused add and returns a pair.
- vLLM sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` itself once CUDA is initialized,
  which dispatching an engine does, and logs `Reasons: CUDA is initialized`. What
  that costs you is the `if __name__ == "__main__":` guard in the main SKILL — spawn
  re-imports the main module, at any `tensor_parallel_size`, including 1.
- Decode-context parallelism on MLA models is handled.

Throughput under TP is where `taps=` matters most — see
[graph taps](graph-taps.md).

## Mixture-of-experts

The router `mlp.gate` is a `ReplicatedLinear`, full and identical on every rank;
its `.output` is a pair `(logits, bias)` of shape `[tokens, num_experts]`. The
fused experts (`mlp.experts`) return the whole value on 0.27 (the all-reduce moved
inside the layer). Individual experts are not submodules — vLLM stacks them into
one grouped kernel — and neither is the top-k selection: recompute it from the
logits, matching the checkpoint's `num_experts_per_tok` / `norm_topk_prob`.

<!-- test: skip -->
```python
model = VLLM("Qwen/Qwen1.5-MoE-A2.7B", dispatch=True, gpu_memory_utilization=0.5)

with model.trace("The theory of relativity was developed by", temperature=0.0, max_tokens=6) as tracer:
    tops = nnsight.save([])
    for _ in tracer.iter[:6]:
        logits, _bias = model.model.layers[5].mlp.gate.output
        tops.append(logits[-1].topk(2).indices.clone())
    result = tracer.result.save()

print(result.outputs[0].text, [t.tolist() for t in tops])
# ' Albert Einstein. It is a' [[17, 26], [8, 36], [5, 22], [22, 42], [10, 3], [45, 40]]
```

To ablate an expert, mask its router logit: `mlp.gate.output[0][:, e] = -inf`.
Qwen-MoE also carries `mlp.shared_expert` and `mlp.shared_expert_gate` as
ordinary submodules.

## Hybrid (linear-attention) trunks

Qwen3-Next / Qwen3.5 / Qwen3.6 interleave gated-delta-net layers with full
attention. Both are ordinary decoder-layer envoys with the same `(hidden,
residual)` output; tell them apart by the child they carry:

<!-- test: skip -->
```python
model = VLLM("Qwen/Qwen3.5-0.8B", dispatch=True, gpu_memory_utilization=0.4)
layers = model.language_model.model.layers          # a checkpoint with a vision tower
kinds = {i: "linear" if hasattr(layers[i]._module, "linear_attn") else "attention"
         for i in range(len(layers))}
print([i for i, k in kinds.items() if k == "attention"])
# [3, 7, 11, 15, 19, 23]
```

The recurrent state lives in vLLM's state cache, not in any module output, so it
is not a location; the layer outputs are. A tapped engine on these models pins
decode-only graphs and matches eager exactly — [graph taps](graph-taps.md).
Vision-language checkpoints load and trace on text prompts; image inputs are
not accepted.
