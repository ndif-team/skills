# Cross-family sweeps, probing, and recurrent state patching

Three recipes whose family-specific part nnter removes: a loop over checkpoints
whose trace body never branches on architecture, a probe whose dataset comes from
one batched read of `layer_output`, and an experiment on the recurrent state of a
gated DeltaNet block. nnter pages: `docs/patterns/cross-family-sweep.md`,
`docs/patterns/probing.md`, `docs/patterns/delta-net-state.md`,
`docs/usage/availability.md`, `docs/usage/activations.md`, `docs/usage/delta-net.md`,
`docs/usage/selective-scan.md`, `docs/usage/state-space.md`.

## Cross-family sweep

The experiment: zero each block's mixer contribution and each block's MLP
contribution in turn and record the KL from the clean next-token distribution, one
batched forward per checkpoint; then residual norms and attention entropy per block.
The loop body differs from a single-model script in three lines: the `support()`
read, the `has_mlp` guard, and `mixers` decided outside the trace.

```python
import torch
import torch.nn.functional as F
from nnter import StandardizedTransformer

REPOS = {
    "gpt2": "openai-community/gpt2",
    "llama": "HuggingFaceTB/SmolLM2-135M-Instruct",
    # "qwen3_5": "Qwen/Qwen3.5-9B",          # a hybrid: three DeltaNet blocks in four
    # "mamba": "state-spaces/mamba-130m-hf", # no attention, no MLP: route_kernels(model.family, "torch") first
    # "opt": "facebook/opt-125m",            # no MLP module
}
prompt = "The Eiffel Tower is in the city of"


def mixer(layer):
    """The block's sequence mixer: self_attn, or linear_attn on a recurrent block."""
    attn = getattr(layer, "self_attn", None)
    return attn if attn is not None else layer.linear_attn      # `is not None`: `or` truth-tests the envoy


def kl(clean_logits, other_logits):
    """KL(clean || other) from last-position logits, float32, safe at exact zeros."""
    return float(F.kl_div(other_logits.float().log_softmax(-1), clean_logits.float().log_softmax(-1),
                          log_target=True, reduction="sum"))


table = {}
for name, repo in REPOS.items():
    model = StandardizedTransformer(repo, device="cpu", dispatch=True, attn_implementation="eager")
    support = model.support()                                     # before any trace: None means every block has it
    has_mlp = support.get("mlp.mlp_output", "absent") is None   # no key at all where no block has an mlp (OPT)
    mixers = [mixer(layer) for layer in model.layers]           # decided outside the trace
    attention_blocks = [i for i, layer in enumerate(model.layers) if getattr(layer, "self_attn", None) is not None]
    pattern_blocks = [i for i in attention_blocks if model.support(layer=i)["self_attn.attention_probabilities"] is None]

    outs = {}                                                   # containers outside; entries saved
    with model.trace() as tracer:
        with tracer.invoke(prompt):
            base = model.logits[:, -1].save()
        for i in range(model.num_layers):
            with tracer.invoke(prompt):
                mixers[i].attention_output[:] = 0
                outs["mixer", i] = model.logits[:, -1].save()
            if has_mlp:
                with tracer.invoke(prompt):
                    model.layers[i].mlp.mlp_output[:] = 0
                    outs["mlp", i] = model.logits[:, -1].save()

    norms, entropy = {}, {}
    with model.trace(prompt):
        for i, layer in enumerate(model.layers):
            if i in pattern_blocks:
                p = layer.self_attn.attention_probabilities
                entropy[i] = (-p * (p + 1e-12).log()).sum(-1).mean().save()
            norms[i] = layer.layer_output[0, -1].norm().save()

    table[name] = {
        "kind": ["linear" if getattr(layer, "linear_attn", None) is not None else "attn" for layer in model.layers],
        "mixer_kl": [kl(base, outs["mixer", i]) for i in range(model.num_layers)],
        "mlp_kl": [kl(base, outs["mlp", i]) for i in range(model.num_layers)] if has_mlp else None,
        "resid_norm": [float(norms[i]) for i in range(model.num_layers)],
        "entropy": {i: float(entropy[i]) for i in entropy},
    }
    del model

for name, row in table.items():
    print(name, "mixer_kl", [round(v, 2) for v in row["mixer_kl"]])
    print(name, "mlp_kl  ", [round(v, 2) for v in row["mlp_kl"]])

assert len(table["gpt2"]["mixer_kl"]) == 12 and len(table["llama"]["mixer_kl"]) == 30
assert max(table["gpt2"]["mlp_kl"]) == table["gpt2"]["mlp_kl"][0]         # block 0 dominates under zero ablation ...
assert max(table["llama"]["mlp_kl"]) == table["llama"]["mlp_kl"][0]       # ... on both families
```

```
gpt2  mixer_kl [3.04, 0.07, 0.05, 0.02, 0.02, 0.05, 0.03, 0.02, 0.03, 0.12, 0.02, 0.04]
gpt2  mlp_kl   [4.25, 0.18, 0.13, 0.14, 0.04, 0.07, 0.05, 0.06, 0.05, 0.08, 0.13, 0.24]
llama mixer_kl [6.09, 5.49, 0.02, 0.0, 0.03, 0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.02, 0.01, 0.08, 0.0, 0.01, 0.03, 0.0, 0.01, 0.01, 0.0, 0.24, 0.0, 0.87, 0.08, 0.0, 0.0, 0.02, 0.02, 0.02]
llama mlp_kl   [9.99, 0.26, 0.45, 0.05, 0.21, 0.09, 0.04, 0.02, 0.01, 0.06, 0.58, 1.55, 0.02, 0.03, 0.08, 0.02, 0.01, 0.06, 0.0, 0.01, 0.02, 0.02, 0.07, 0.01, 0.17, 0.04, 0.06, 0.04, 0.1, 0.17]
```

The two rows the executed loop leaves out behave like this (verified on the tiny
checkpoints of each family): on the **hybrid** `kind` reads
`['linear', 'linear', 'linear', 'attn', ...]`, `entropy` has one entry per attention
block, and `mixer_kl` has one per block because a DeltaNet mixer's contribution is
`attention_output` too; on **OPT** `mlp_kl` is `None`, because no block has an `mlp`
and `support()` then has no `mlp.mlp_output` key at all, which the `.get` above turns
into `has_mlp = False`.

The guards:

- **Availability.** `support()` returns `None` when every block has the value, else
  `{block: reason}`. Guard on it, not on `hasattr`, which raises. Common reasons in a
  sweep: `no self_attn module` (a hybrid's linear blocks; a module no block has,
  OPT's `mlp`, has no key at all, so `.get` it),
  `runs 'sdpa'; load with attn_implementation='eager'`, GPT-2's
  `reorder_and_upcast_attn`, and a DeltaNet interior with `flash-linear-attention`
  installed (no Python source; `attention_output` stays available).
- **Hybrids.** A block has either `self_attn` or `linear_attn`. `layer_output`,
  `mlp_output` and the mixer's `attention_output` exist on both kinds, so most
  experiments never branch. When one must, branch outside the trace.
- **Recurrent-only families.** Mamba and Mamba-2 have no `self_attn` and no `mlp`
  anywhere (`pattern_blocks` is empty, `has_mlp` is `False`), and with `mamba_ssm`
  installed need `nnter.route_kernels(model.family, "torch")` before the first trace
  even for `layer_output` on CPU.
- **Non-additive blocks.** Zeroing a contribution is still a valid ablation on
  Gemma-4, Doge, ZAYA and DeepSeek-V4; attributing by the plain sum is not (the
  `nnter` skill's `references/values.md`).
- **Memory.** One checkpoint at a time; `del model` (and `torch.cuda.empty_cache()`
  on a GPU) between; `device="cuda", dtype=torch.bfloat16` for 8B-class checkpoints
  (choose the device with `device=`: `device_map="cpu"` is ignored).

KL and norm values from different checkpoints are not directly comparable
(vocabularies, depths and residual scales differ): compare curve shapes, or
normalize per model. Sizes come off the model (`model.num_heads`,
`model.head_dim`, or a block's own `layers[i].self_attn.num_heads`), not off a
config key whose name differs by family. The KL is computed from logits in
float32: the textbook `p * (p.log() - q.log())` on `next_token_probs` is NaN on
Pythia (exact zeros) and slightly negative between near-equal bf16 rows.

## Probing

### The dataset: one batched forward

```python
from nnter.nnsight_utils import get_token_activations

model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True,
                                tokenizer_kwargs={"padding_side": "left", "pad_token": "<|endoftext|>"})

positive = ["wonderful", "fantastic", "delightful", "excellent", "brilliant", "joyful", "superb", "lovely"]
negative = ["terrible", "awful", "dreadful", "disgusting", "horrible", "miserable", "dismal", "boring"]
templates = ["The movie was {}.", "I found the book {}.", "That meal was {}.", "Their performance was {}."]

texts = [t.format(w) for w in positive for t in templates] + [t.format(w) for w in negative for t in templates]
labels = torch.tensor([1.0] * (len(positive) * len(templates)) + [0.0] * (len(negative) * len(templates)))

acts = get_token_activations(model, texts, idx=-1)              # [layers, prompts, hidden], on the CPU, no grad
assert acts.shape == (model.num_layers, len(texts), model.hidden_size)
```

`get_token_activations(model, prompts, layers=None, get_activations=None, idx=-1)`
reads every block's `layer_output` at one position (`get_activations=lambda m, i:
m.layers[i].mlp.mlp_output` reads another site), under `torch.no_grad`, and raises
`ValueError` before running when `idx` is negative and the tokenizer does not pad
left. nnsight's `TransformersModel` already gives GPT-2 left padding and
`<|endoftext|>` as pad, so the `tokenizer_kwargs` above are the portable form for a
tokenizer that lacks them; the pad string is the tokenizer's own (`<|endoftext|>` is
GPT-2's, not Llama's), so read `model.tokenizer.pad_token` before copying it. On
DeepSeek-V4 `layer_output` is `[batch, seq, streams, hidden]` and the activations
gain a streams axis. `collect_token_activations_batched(model, texts,
batch_size)` chunks a set too large for one batch.

### A closed-form ridge probe per layer

```python
generator = torch.Generator().manual_seed(0)
order = torch.randperm(len(texts), generator=generator)
split = int(0.7 * len(texts))
train, test = order[:split], order[split:]


def fit_probe(x, y, ridge=1e-2):
    """Ridge regression of the {-1, +1} label on [n, hidden]; standardized on the rows it is given."""
    mean, std = x.mean(0), x.std(0) + 1e-6
    design = lambda z: torch.cat([(z - mean) / std, torch.ones(len(z), 1)], dim=1)
    z = design(x)
    penalty = torch.eye(z.shape[1]); penalty[-1, -1] = 0         # no penalty on the bias
    w = torch.linalg.solve(z.T @ z + ridge * len(z) * penalty, z.T @ (2 * y - 1))
    return lambda x_new: design(x_new) @ w > 0


accuracy = []
for layer in range(model.num_layers):
    x = acts[layer].float()
    predict = fit_probe(x[train], labels[train])
    accuracy.append(float((predict(x[test]).float() == labels[test]).float().mean()))

print([round(a, 2) for a in accuracy])
assert min(accuracy) > 0.9                                       # perfect from layer 0: the probe reads token identity
```

```
[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

Read the curve's shape. High from layer 0, as here, means the probe reads token
identity ("wonderful" and "terrible" are different tokens, and the last position is
the period right after them); a rise through the middle means something the model
computes; high only at the end may be the prediction itself. This dataset says
nothing about sentiment *processing*, and the controls are what say so.

### Controls

```python
shuffled = labels[torch.randperm(len(labels), generator=generator)]     # anything above chance is memorization capacity
predict = fit_probe(acts[2][train].float(), shuffled[train])
control = float((predict(acts[2][test].float()).float() == shuffled[test]).float().mean())
print(f"shuffled-label accuracy {control:.2f}")
assert control < 0.75

x = acts[2].float()                                                     # mass-mean direction: no fitting
direction = x[train][labels[train] == 1].mean(0) - x[train][labels[train] == 0].mean(0)
direction = direction / direction.norm()
threshold = (x[train] @ direction).mean()
mass_mean = float((((x[test] @ direction) > threshold).float() == labels[test]).float().mean())
print(f"mass-mean accuracy {mass_mean:.2f}")
assert mass_mean > 0.6
```

```
shuffled-label accuracy 0.45
mass-mean accuracy 0.85
```

A held-out template (train on three, test on the fourth) is the third control: if
accuracy collapses, the probe learned the template. The mass-mean direction is the
one to hand to `model.steer` for the causal check (`references/steering-and-ablation.md`);
a direction that decodes and steers nothing is also a result.

Fit on training rows only, including the standardization statistics; keep the ridge
on and report it (with `hidden` far above the example count an unregularized probe
fits anything); accuracy is not comparable across `hidden_size` unless probe
capacity is held fixed. `idx=0` or a positive `idx` needs right padding.

## Recurrent state patching

A gated DeltaNet block (Qwen3-Next, Qwen3.5) has no pattern. Each head keeps a
recurrent state `[key_dim, value_dim]`; every token decays it, writes its key/value
pair in scaled by a beta, and the query reads against it. Mamba-1 and Mamba-2 blocks
keep a state too, with other layouts and fewer writable forms (the `nnter` skill's
`references/recurrent-mixers.md` has the "same name, different meaning" table).
The blocks below are not in the executed set; every one ran on the tiny Qwen3.5
checkpoint nnter's suite pins, with the assertions shown.

### The state entering and leaving a prompt

<!-- test: skip -->
```python
import nnter
from nnter import StandardizedTransformer, route_kernels

route_kernels(nnter.families.qwen3_5_text, "torch")              # before the first trace: per-token state, readable kernels
model = StandardizedTransformer("Qwen/Qwen3.5-9B", device="cpu", dispatch=True, attn_implementation="eager")
linear_blocks = [i for i, layer in enumerate(model.layers) if getattr(layer, "linear_attn", None) is not None]
mix = model.layers[linear_blocks[0]].linear_attn                  # decided outside the trace
prompt = "The Eiffel Tower is in the city of"
other = "My favourite food is pizza with"

entering = []                                                     # a None bound inside the trace would not survive it
with model.trace(prompt):
    state_in = mix.state_input
    entering.append(state_in.save() if state_in is not None else None)
    state_out = mix.state_output.save()                           # [batch, heads, key_dim, value_dim]
    clean = model.next_token_probs.save()

assert entering[0] is None and state_out.dim() == 4               # fresh prompt: nothing enters
```

`state_input` is a copy of the cache's buffer (the cache overwrites it in place);
`state_output` is what the next decode step starts from. `heads` here is the
mixer's `num_v_heads`, not `model.num_heads`.

### Patching the state on a decode step

<!-- test: skip -->
```python
with model.trace(other):
    other_state = mix.state_output.save()

N = 3
with model.generate(prompt, max_new_tokens=N, do_sample=False) as tracer:
    for step in tracer.iter[2]:
        entering_clean = mix.state_input.save()
    ids = tracer.result.save()

with model.generate(prompt, max_new_tokens=N, do_sample=False) as tracer:
    for step in tracer.iter[1]:
        mix.state_input = other_state                             # decode step 1 continues from the other prompt's memory
    for step in tracer.iter[2]:
        entering_patched = mix.state_input.save()
    ids_patched = tracer.result.save()

assert not torch.equal(entering_patched, entering_clean)          # step 2 starts from a different state
```

Step 0 is the prompt (its `state_input` is `None`); steps 1 and beyond are one token
each with the cached state entering. Assert on the state, not on the tokens: on
Qwen3.5-0.8B the write lands and the greedy tokens do not change (one mixer of
many, and the short conv window still carries the real prompt's last tokens).

### The state after every token

After `route_kernels` a prompt runs the token-by-token kernel, so every position is a
value (Mamba-2 needs `nnter.chunk_per_token(model)` instead, and has no per-token
`state` / `set_state_after`):

<!-- test: skip -->
```python
with model.trace(prompt):
    states = mix.states.save()                                    # [batch, seq, heads, key_dim, value_dim]
    final = mix.state_output.save()
    probs = model.next_token_probs.save()

assert torch.equal(states[:, -1], final)                          # the last token's state is the one the prompt leaves
norms = states.flatten(2).norm(dim=-1)[0]                         # [seq]: how the memory grows along the prompt
tokens = [model.tokenizer.decode(t) for t in model.tokenizer(prompt).input_ids]
for token, norm in zip(tokens, norms):
    print(f"{token!r:12} {float(norm):.3f}")
```

`mix.state` is the same value one token at a time under `tracer.iter[:n]`
(`walked[t] == states[:, t]`). Without the routing, `states` raises
`nnter.Unavailable` naming `route_kernels`, and `support()` reports it under
`linear_attn.states`. `route_kernels(model.family, "default")` restores the
default kernels for later loads.

### Patching the state inside a prompt

<!-- test: skip -->
```python
T = 3
with model.trace(other):
    donor = mix.state_after(T).save()

with model.trace(prompt):
    before = mix.state_after(T - 1).save()                        # positions before the write: read first
    mix.set_state_after(T, donor)                                 # tokens T+1 onward continue from the donor state
    after = mix.state_after(T + 1).save()                         # positions after it: read after
    patched = model.next_token_probs.save()

assert torch.equal(before, states[:, T - 1]) and not torch.equal(after, states[:, T + 1])

with model.trace(prompt) as tracer:                               # the same write as an assignment under tracer.iter
    for t in tracer.iter[T]:
        mix.state = donor                                         # mix.state * 0 here works too
    patched_iter = model.next_token_probs.save()

assert torch.allclose(patched, patched_iter, atol=1e-6)
```

The write also changes token `T`'s own output (it is read from the state it
writes), and the conv window means tokens just after `T` still see the real
prompt's inputs: a state write is not a clean "forget everything before `T`".

`mix.decays` (`[batch, seq, heads]`, log decay, `<= 0`) and `mix.betas` (in `(0, 1)`)
are the gate and the write strength; `decays.exp()` is the fraction of the state
each token keeps per head, and a head with it near one is the long memory whose
norm curve above keeps growing.

### Gotchas

- Reads follow the forward: positions before a write before it, positions after it
  afterwards; `states` reads every position, so it goes in a trace of its own or
  before any write.
- `route_kernels` is process-wide and must run before the layer's forward is
  instrumented; a forward already instrumented keeps the kernel it was compiled with.
- On a Mamba decode step read `state_output` before `attention_head_outputs`: the
  other order returns the next step's state on Mamba-1, with no warning.
- `states` is read-only, a stack of copies; write one position with
  `set_state_after` or `state` under `tracer.iter`. Write by assignment, not in place.
- Under `generate`, `states`, `state_after` and `set_state_after` count from the
  current call's own first token: the prompt's tokens on step 0, the single token on
  a decode step.
- With `flash-linear-attention` or `causal-conv1d` installed the kernel has no Python
  source, every interior value is unavailable, and `support()` says so;
  `attention_output` stays available.
