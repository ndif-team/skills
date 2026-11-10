# Mixture of experts

On the 39 MoE families the mixture is a `Moe` (an `Mlp`): `layers[i].mlp` on most,
and `mlp_output` is still what the block adds. Six values say how the mixture made
it, each read where the model consumes it and written back there:

| value | layout | what |
|---|---|---|
| `router_logits` | `RouterLogits`: `[batch, seq, experts]` | the router's logits before the scoring; a write changes the routing |
| `expert_weights` | `ExpertWeights`: `[batch, seq, top_k]` | the weight each slot's expert output is scaled by, after the router's normalization |
| `expert_indices` | `ExpertIndices`: `[batch, seq, top_k]`, int64 | the expert each slot sends the token to |
| `expert_outputs` | `ExpertOutputs`: `[batch, seq, top_k, hidden]` | each slot's *weighted* expert output; `.sum(2) == routed_output` |
| `routed_output` | `Residual` | the routed experts' sum, without the shared expert |
| `shared_expert_output` | `Residual` | the shared expert's contribution, where there is one |

`moe.num_experts` and `moe.top_k` are read off the module; `Moe.SCORING` names
what the logits mean: `"softmax"` (Mixtral, Qwen-MoE, OLMoE, DeepSeek-V2, Jamba,
DBRX, Gemma-4, ZAYA, ...), `"topk_softmax"` (GPT-OSS, the GraniteMoE line, JetMoE),
`"sigmoid"` (DeepSeek-V3/V3.2, GLM-4-MoE, GLM-5, Nemotron-H, MiniMax-M2, Llama 4,
...), `"sparsemixer"` (Phi-3.5-MoE), per block on DeepSeek-V4 (`"hash"` routes by
token id). `SCORING` alone does not recompute the weights: whether the top-k is
renormalized (`norm_topk_prob`, off on OLMoE, whose rows sum to 0.24-0.89), a
selection bias and a routed scaling factor are per-family config. So a token's
weights sum to 1 on most softmax and top-k-softmax routers (Mixtral, Qwen3-MoE, the
GraniteMoE line, GPT-OSS, JetMoE, DBRX, ...), under 1 where the top-k is not
renormalized (OLMoE, Qwen1.5-MoE, Jamba, ZAYA, DeepSeek-V2 before its factor), to
`routed_scaling_factor` on the renormalized sigmoid routers (2.5 on DeepSeek-V3,
GLM-4.5, GLM-5 and dots.llm1, 1.8 on GLM-4.7-Flash, 2.827 on Kimi-K2.5; AFMoE's
`route_scale` 2.826), between 1 and 2 on Phi-3.5-MoE, and to any value on Gemma-4
(each slot times a learned `per_expert_scale`): divide by the sum before comparing
routing across families. Children:
`router` (alias of `gate`), `experts`, `shared_experts` (alias of `shared_expert` /
`shared_mlp`).

A mixture has no single activation or down projection, so a `Moe`'s `mlp_activation`
and `mlp_neurons` are unavailable with `routed experts; read per-expert via the Moe
component`; a dense block beside the mixtures serves both. The block's `mlp_input` is
the mixture's input, as on a dense block.

Every value is a `TokenEProperty`: the model routes tensors flat over tokens,
`[batch * seq, ...]`, and each invoke is served its own `[batch, seq, ...]` rows
as a view, so in-place edits land on that invoke only.

No MoE checkpoint is in the executed set; every block below ran as written on
`hf-internal-testing/tiny-random-MixtralForCausalLM` in float32 (4 experts, top-2).

## Pick the mixture blocks outside the trace

Dense blocks sit beside mixture blocks on DeepSeek-V3, GLM-4-MoE, Llama 4, Jamba,
Gemma-4 and others; there `support()` reports `"no <value> value on this block's
mlp"`, and reading `num_experts` on a dense block raises a bare `AttributeError`.

<!-- test: skip -->
```python
import torch
from nnterp import StandardizedTransformer

model = StandardizedTransformer("mistralai/Mixtral-8x7B-v0.1", device="cpu", dispatch=True, dtype=torch.float32)
prompt = "The Eiffel Tower is in the city of"
ids = model.tokenizer("Paris", add_special_tokens=False).input_ids       # sentencepiece: " Paris" is ['▁', 'Paris'], "Paris" is ['▁Paris']
assert len(ids) == 1, ids
target = ids[0]

blocks = [i for i in range(model.num_layers) if model.support(layer=i).get("mlp.expert_weights", "absent") is None]
moe = model.layers[blocks[0]].mlp
print(moe.num_experts, moe.top_k, moe.SCORING)
```

## Usage and entropy: mask the padding

A batch of prompts is left-padded, and pad tokens are routed like any other
(25-31% of the slots in a typical batch): count only real tokens.

<!-- test: skip -->
```python
prompts = ["The Eiffel Tower is in the city of", "def f(x): return"]
with model.trace(prompts):
    mask = model.attention_mask.save()                  # read first: the model's input
    logits = moe.router_logits.save()                   # [batch, seq, experts]
    idx = moe.expert_indices.save()                     # the indices first: JetMoE computes them before the weights
    w = moe.expert_weights.save()

real = mask.bool()                                      # [batch, seq]
slots = real[..., None].expand_as(idx) & (w != 0)       # w != 0: ZAYA's skipped slots read as expert 0, weight 0
usage = torch.bincount(idx[slots], minlength=moe.num_experts)

probs = logits.float().softmax(-1)                      # a softmax router (SCORING == "softmax")
entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)[real].mean()
```

## Expert ablation and rerouting

Zero the weight of every slot that chose expert `e`: those slots add nothing and
the token is *not* renormalized onto its other experts. Compare log-probabilities
in float32.

<!-- test: skip -->
```python
with model.trace(prompt):
    clean = model.logits[0, -1].float().log_softmax(-1)[target].save()

effects = []
for e in range(moe.num_experts):
    with model.trace(prompt):
        idx = moe.expert_indices                # the indices first: JetMoE computes them before the weights
        moe.expert_weights = moe.expert_weights.masked_fill(idx == e, 0)
        effects.append((model.logits[0, -1].float().log_softmax(-1)[target] - clean).save())

with model.trace(prompt):                       # reroute the last token's first slot to an expert none of its slots uses
    rerouted = moe.expert_indices.clone()
    rerouted[:, -1, 0] = next(e for e in range(moe.num_experts) if e not in rerouted[0, -1].tolist())
    moe.expert_indices = rerouted               # its weight stays what the router gave the expert it chose
    rerouted_logits = model.logits.save()

with model.trace(prompt):                       # force the router: logits decide weights and indices
    moe.router_logits[:, -1] = -10.0
    moe.router_logits[:, -1, 2] = 10.0
    forced = moe.expert_indices.save()
```

The sweep fits one trace with one invoke per expert: an in-place edit
(`moe.expert_weights[:] = moe.expert_weights.masked_fill(...)`) and an assignment both
reach that invoke only. In float32 the batched sweep equals the per-trace one exactly.

Reroute to an expert the token's other slots do not use: on JetMoE a repeated one
raises `RuntimeError` (`split_with_sizes`), and the other families run that expert
once per slot.

**bf16 sweeps are noise.** On a bf16 GraniteMoE the one-invoke-per-expert sweep
correlated 0.32 with per-trace ablation and named a different "most important"
expert: ablating in one invoke moved another invoke's logits by ten times the
ablation's own effect. Load `dtype=torch.float32` for single-expert effects, or
keep a clean baseline invoke in the same batch and compare against it.

Unused experts read exactly zero (no slot to zero), which is not "unimportant";
the shared expert is not an expert here (`shared_expert_output[:] = 0`); zero
ablation overstates, and rerouting is the milder intervention.

## `experts_implementation=`

transformers runs the routed experts as `"grouped_mm"` (the default),
`"batched_mm"` or `"eager"` (a Python loop), chosen at load like
`attn_implementation`. `expert_outputs` exists only under the first two;
under `"eager"` it is unavailable with a reason naming the kwarg. Everything else
reads the same under every implementation. Unweighted per-slot outputs are
`expert_outputs / expert_weights[..., None]` where the weight is not zero.

## Read order

`router_logits`, then `expert_weights` / `expert_indices`, then `expert_outputs`,
then `routed_output`, then `mlp_output`. `shared_expert_output` comes first on
Hunyuan, ERNIE, Laguna and Gemma-4, between router and experts on AFMoE, after the
routing and before `routed_output` on Llama 4, and after `routed_output` on
DeepSeek, GLM, Qwen, Nemotron-H and the GraniteMoE line. Out of order raises
`OutOfOrderError` naming an internal op (`...router.source.F_linear_0.output.i0`).
`model.order(i)` is this order measured on the loaded checkpoint, the shared expert's
place included, and `model.rank(name, i)` sorts a list of reads by it
([availability-and-support.md](availability-and-support.md#forward-order-order-and-rank)).

## Per-family gaps

| family | what differs |
|---|---|
| GraniteMoE (-SWA, -Shared, -Hybrid) | `mlp_output` is the scaled term the block adds; `expert_outputs`, `routed_output` and `shared_expert_output` are unscaled (about 4.5x what reaches the stream on granite-3). -Shared / -Hybrid: `routed_output + shared_expert_output == mlp_output / residual_multiplier`; on -Hybrid `mlp` is the shared expert, handed the block's `router` and `experts`. -SWA with `shared_intermediate_size > 0` (granite-swash-3b-a600m): the block also adds its `shared_mlp` times the multiplier, which `mlp_output` leaves out and `shared_expert_output` does not serve, so the identity misses that term |
| Gemma-4 (26B-A4B) | `router` / `experts` are the block's children, handed to `layers[i].mlp` (the dense MLP, also `shared_expert_output`); every mixture value is unavailable on a dense checkpoint |
| Llama 4 | the router scales each expert's *input* by a dense score: `expert_weights` and `expert_outputs` unavailable |
| DBRX, JetMoE | `expert_outputs` unavailable (the experts loop in their own forward); JetMoE's `routed_output` is before `+ bias`, its indices come before its weights, a written index must not repeat another slot's expert (`RuntimeError`), and on a fresh model the first trace that reads `router_logits` then the routing pair raises `OutOfOrderError` (an earlier trace that read the routing pair, or `moe.router.source` touched outside a trace on a dispatched model, avoids it; an earlier trace that never reached the router's operations does not) |
| ZAYA | `router_logits` has `num_experts + 1` columns, the last is *skip*: a skipped slot reads index 0, weight 0 |
| DeepSeek-V4 | on a `hash_moe` block the token ids pick the experts: writing `router_logits` changes the weights, not the indices |
| Laguna | `expert_outputs.sum(2) * routed_scaling_factor == routed_output` |
| Qwen2-MoE, Qwen3-Next, Qwen3.5-MoE | `shared_expert_output` is the sigmoid-gated product; `shared_experts.output` is ungated |
| Nemotron-H with `moe_latent_size` | `routed_output` is the latent up-projection's output; `expert_outputs` unavailable |
| Doge | all six unavailable: `router_gate` gives two sets of product-key logits per token, not one per expert, and the experts are embedding rows (the mixture runs from transformers 5.18). The routing views `router_gate`'s output as `(2, batch * seq, -1)`, so on the `is_moe` checkpoints a prompt's outputs depend on its length and on the other prompts in the trace: trace clean and corrupt prompts separately |

The nnterp repo pages behind this file: `docs/usage/mixture-of-experts.md`,
`docs/patterns/expert-ablation.md`.
