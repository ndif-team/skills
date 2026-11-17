# vLLM: throughput and architectures

Depth for the [vLLM guide](vllm.md): CUDA-graph taps, tensor parallelism,
mixture-of-experts routers and hybrid linear-attention trunks. Every example
needs a CUDA GPU and `pip install "nnsight[vllm]"`, and builds its own engine.

## Graph taps

A replayed CUDA graph runs no Python, so an engine with no `taps` runs eagerly to
serve every location and gives up graph replay. `taps=` names the locations to
record into the graph and serve on every replay. Declare taps when decode
throughput matters and you know the locations up front; stay eager to explore.
Trace syntax, `tracer.iter`, `model.logits`, `model.samples` and `tracer.result`
do not change.

```python
import torch
import nnsight
from nnsight.modeling.vllm import VLLM

if __name__ == "__main__":
    model = VLLM("HuggingFaceTB/SmolLM2-135M-Instruct", dispatch=True,
                 gpu_memory_utilization=0.2, max_model_len=1024,
                 taps=["model.layers.*.output", "model.layers.8.mlp.input"])
    assert len(model.taps) == 31 and "model.model.layers.8.mlp.input" in model.taps

    torch.manual_seed(0)
    v = torch.randn(576)
    v = 40.0 * v / v.norm()
    prompt = "The capital of France is"

    with model.trace(prompt, temperature=0.0, max_tokens=8, ignore_eos=True) as tracer:
        plain = tracer.result.save()

    with model.trace(prompt, temperature=0.0, max_tokens=8, ignore_eos=True) as tracer:
        hs = nnsight.save([])
        for _ in tracer.iter[:8]:                                    # every step, prefill included
            h = model.model.layers[6].output[0]
            h[:] += v.to(h.device, h.dtype)                          # in place
            hs.append(model.model.layers[12].output[1][-1].clone())  # clone: the graph rewrites this memory
        steered = tracer.result.save()

    assert len(hs) == 8 and not torch.equal(hs[1], hs[2])
    assert steered.outputs[0].text != plain.outputs[0].text
```

- **Tap patterns.** A module path from the root ending in `.input` or `.output`;
  `*` stands for one path segment. `model.taps` holds the resolved set, each entry
  with the root's `model.` prefix. A pattern that matches no module raises
  `ValueError: Tap '...' names no module` when the engine is built. Keep the set
  small: each tap splits the graph.
- **A tap can be a `.source` operation**:
  `"model.layers.10.self_attn.source.qkv_split_0.output"`. One the forward does
  not have fails the engine build ([errors](vllm.md#errors-and-warnings)).
- **Only taps are served.** A read of any other module location fails its request
  with `... is not a tap on this engine`.
- **Edits land in place.** A replacement (`layer.output = t`) is copied back into
  the graph's memory and must keep the block's rows, as on the eager engine.
- **Clone what you keep.** A served value is the graph's memory, rewritten on the
  next step. Un-cloned decode entries all come back holding the last step's
  values, while the prefill entry, a separate buffer, survives: the list looks
  partly right and nothing warns.
- **Tapped graphs are piecewise.** A tapped engine sets
  `cudagraph_mode="PIECEWISE"`; a `compilation_config` that asks for full graphs
  fails the engine build.

## Tensor parallelism

`tensor_parallel_size=N` shards the model over N GPUs, and a block does not
change: nnsight gathers a sharded value into the whole tensor before the block
reads it and re-splits what the block writes. This example needs two GPUs.

```python
import nnsight
from nnsight.modeling.vllm import VLLM

if __name__ == "__main__":
    model = VLLM("Qwen/Qwen2.5-0.5B-Instruct", dispatch=True, tensor_parallel_size=2,     # 14 heads: splits in two
                 gpu_memory_utilization=0.2, max_model_len=1024)

    with model.trace("The capital of France is", temperature=0.0, max_tokens=1):
        o_proj = model.model.layers[10].self_attn.o_proj
        shapes = nnsight.save({"input": tuple(o_proj.input.shape),
                               "weight": tuple(o_proj._module.weight.shape)})

    assert shapes["input"][-1] == 896         # the whole tensor
    assert shapes["weight"][-1] == 448        # this rank's half
```

- **Activations are whole.** `qkv_proj.output[0]`, `gate_up_proj.output[0]`,
  `o_proj.input` and `down_proj.input` read at full width, LoRA-wrapped layers
  included, and the [logit lens](vllm.md#logit-lens) gathers the vocabulary
  shards.
- **Parameters are not gathered.** `weight` is this rank's slice, which is how you
  tell what is sharded. The client's `print(model)` shows `tp_size=1` whatever you
  asked for: check `._module.weight.shape` inside the block.
- **Every rank runs the block.** No rank-dependent control flow. A tensor
  referenced from outside travels to every rank, and an in-block `torch.randn`
  agrees across ranks.
- **Fused projections keep the one-GPU layout**, `[q | k | v]` and `[gate | up]`,
  so `chunk(2, -1)` and a slice by head mean what they do on one GPU. A vLLM
  subclass of those two layers, such as MiniMax-M3's `qkv_proj`, reads in rank
  order instead (`[q0 k0 v0 | q1 k1 v1]`).

## Mixture-of-experts

The router `mlp.gate` is full and identical on every rank, and its `.output` is
the pair `(logits, bias)` with logits `[tokens, num_experts]`. `mlp.experts`
returns the experts' combined output, whole under tensor parallelism too, except
on a sequence-parallel MoE layer, which reads as one rank's rows. Individual
experts are not submodules, because vLLM stacks them into one grouped kernel, and
neither is the top-k selection: recompute it from the logits with the
checkpoint's `num_experts_per_tok` and `norm_topk_prob`.

This example needs a GPU that holds Qwen1.5-MoE-A2.7B: about 14B parameters,
close to 29 GB in bfloat16.

```python
import nnsight
from nnsight.modeling.vllm import VLLM

if __name__ == "__main__":
    model = VLLM("Qwen/Qwen1.5-MoE-A2.7B", dispatch=True, gpu_memory_utilization=0.5)

    with model.trace("The theory of relativity was developed by",
                     temperature=0.0, max_tokens=6, ignore_eos=True) as tracer:
        tops = nnsight.save([])
        for _ in tracer.iter[:6]:
            logits, _bias = model.model.layers[5].mlp.gate.output
            tops.append(logits[-1].topk(4).indices)      # this checkpoint routes a token to 4 of 60 experts
        result = tracer.result.save()

    assert len(tops) == 6 and all(t.shape == (4,) for t in tops)
    print(result.outputs[0].text, [t.tolist() for t in tops])
```

To ablate an expert, mask its router logit:
`mlp.gate.output[0][:, e] = float("-inf")`.
**That recipe is right only on vLLM.** Under `TransformersModel` the router returns
a different tuple and the same line is a silent no-op:
[modules-and-architectures](modules-and-architectures.md) has that version.

## Hybrid linear-attention trunks

Qwen3-Next and Qwen3.5-style models interleave gated-delta-net layers with
full-attention layers. Both are ordinary decoder layers with the same
`(hidden, residual)` output; tell them apart by the child they carry,
`linear_attn` or `self_attn`:

```python
from nnsight.modeling.vllm import VLLM

if __name__ == "__main__":
    model = VLLM("Qwen/Qwen3.5-0.8B", dispatch=True, gpu_memory_utilization=0.4)
    layers = model.language_model.model.layers          # a checkpoint with a vision tower
    attention = [i for i in range(len(layers)) if hasattr(layers[i]._module, "self_attn")]
    assert attention == [3, 7, 11, 15, 19, 23]          # the other 18 layers are linear attention
```

The recurrent state lives in vLLM's state cache, not in any module output, so it
is not a location. `taps=` works on these models: a tapped engine matches the
eager one exactly.
