# Recurrent mixers: DeltaNet, Mamba-1, Mamba-2

A recurrent block mixes the sequence through a state instead of a pattern, and
its mixer is `layers[i].linear_attn` on every family. There are three kinds, each
a `RecurrentMixer` (`nnterp.components`), and they share the attention vocabulary
where the roles match: a query reads the state, a key says where a token writes,
a value is what it writes, `betas` how strongly, `decays` how much of the state
each token keeps, `state_input` / `state_output` the state entering and leaving
the call, `attention_output` what the mixer adds to the stream.

| mixer | class | families | blocks |
|---|---|---|---|
| gated DeltaNet | `LinearAttention` | `qwen3_next`, `qwen3_5_text`, `qwen3_5_moe_text`, `olmo_hybrid` | three in four (`config.layer_types`); `self_attn` on the rest |
| Mamba-1 selective scan | `SelectiveScan` | `mamba`, `falcon_mamba`, `jamba` | every block on Mamba / Falcon-Mamba (no attention, no MLP); all but one in `attn_layer_period` on Jamba |
| Mamba-2 (SSD) | `StateSpace` | `mamba2`, `nemotron_h`, `bamba`, `falcon_h1`, `granitemoehybrid` | every block on Mamba-2 (mixer alone) and Falcon-H1 (beside `self_attn`); per block type elsewhere |

No recurrent checkpoint is in the executed set, so every block in this file is
skipped by the suite. Each ran as written against the tiny checkpoint nnterp's
suite pins (`yujiepan/qwen3.5-tiny-random`, `hf-internal-testing/tiny-random-MambaForCausalLM`,
`yujiepan/mamba2-tiny-random`), on CPU with `mamba_ssm` installed.

## Before the first trace: kernels

transformers dispatches each mixer to a compiled kernel when one is installed
(`mamba_ssm`, `flash-linear-attention`, `causal-conv1d`); those have no Python
source, so nothing inside them is readable, and `mamba_ssm`'s do not run on CPU
at all: an unrouted Mamba-1 model on CPU crashes with `Expected u.is_cuda()` even
for a `layer_output` read. `route_kernels(family, "torch")` binds the family's
kernel names to transformers' pure-torch functions, **process-wide**; call it
before the layer's first trace (a forward already instrumented keeps its
kernel), with the family module before loading or `model.family` after.
`route_kernels(family, "default")` restores them; `route_delta_rule(family,
"recurrent" | "chunked")` is the DeltaNet spelling of the same call.

<!-- test: skip -->
```python
import torch
import nnterp
from nnterp import StandardizedTransformer, chunk_per_token, route_kernels

route_kernels(nnterp.families.qwen3_5_text, "torch")     # DeltaNet: also what makes per-token `state` exist
model = StandardizedTransformer("Qwen/Qwen3.5-0.8B", device="cpu", dispatch=True, attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"

linear = [i for i, layer in enumerate(model.layers) if getattr(layer, "linear_attn", None) is not None]
softmax = [i for i, layer in enumerate(model.layers) if getattr(layer, "self_attn", None) is not None]
mix = model.layers[linear[0]].linear_attn                # decided outside the trace
assert model.support(layer=linear[0])["linear_attn.state"] is None
```

On a DeltaNet hybrid the routing also switches a prompt to the token-by-token
kernel, the only one that materializes the state after every token (the two
compute the same rule; logits agree to float error). On Mamba-1 the pure-torch
scan already is a token loop. Mamba-2 keeps its chunked scan; set
`chunk_per_token(model)` (per model, not process-wide; `chunk_per_token(model,
False)` undoes it) to make every token a chunk boundary.

## The same name, different meanings

| value | DeltaNet (`LinearAttention`) | Mamba-1 (`SelectiveScan`) | Mamba-2 (`StateSpace`) |
|---|---|---|---|
| `attention_queries` | `q` after conv, activation, repeat to the value heads; **before** the kernel's l2norm and scale | `C`, `[batch, seq, 1, state_dim]` | `C` per group of heads |
| `attention_keys` | `k`, likewise before the l2norm | `B` | `B` per group |
| `attention_values` | `v`, `[batch, seq, heads, value_dim]` | `x` per channel, `[batch, seq, channels]` | `x` per head, `[batch, seq, heads, head_dim]` |
| `betas` | write strength in `(0, 1)`, per head; assignable | `softplus(dt + dt_bias)` per channel; read-only | `softplus(dt + dt_bias)` per head; assignable, written back into `dt` |
| `decays` | log gate per head, float32, `<= 0`; assignable | `dt * A` per channel and state dim; read-only | `A * dt` per head; assignable, but it *is* `dt`: writing it changes `betas` |
| `attention_head_outputs` | each head's read, before the gated norm and `out_proj` | `C . h + D x` before `silu(z)` | `h C + D x` before the gated norm |
| state layout | `State`: `[batch, heads, key_dim, value_dim]` | `ScanState`: `[batch, channels, state_dim]` (value side first) | `State`: `[batch, heads, state_dim, head_dim]` (key side first; the cache's transpose) |
| `state` (per token, `tracer.iter`) | after `route_kernels` | after `route_kernels` | unavailable |
| `states`, `state_after(t)` | after `route_kernels` | after `route_kernels` | after `chunk_per_token(model)`; a read-only copy |
| `set_state_after(t, v)` | yes | yes | unavailable: assign `state_input` |
| `state_input` on a prompt | `None` | `None`; assigning raises `ValueError` | `None` |
| in-place edit of q/k/v | lands | a view: assign, or edit under `torch.no_grad()` | a view: assign, or edit under `torch.no_grad()` |

A hand recurrence must apply what the kernel does after the read: DeltaNet's
kernel l2-normalizes `q` and `k` and scales `q` itself (skip it and the recurrence
is off by ~70%); Mamba-1's and Mamba-2's add the `D` skip.

## Reading the values

<!-- test: skip -->
```python
with model.trace(prompt):
    q = mix.attention_queries.save()        # [batch, seq, heads, key_dim]
    g = mix.decays.save()                   # [batch, seq, heads], float32, <= 0
    b = mix.betas.save()                    # [batch, seq, heads], in (0, 1)
    entering = mix.state_input              # None on a fresh prompt: nothing to save
    state = mix.state_output.save()         # [batch, heads, key_dim, value_dim]: after the last token
    out = mix.attention_output.save()       # [batch, seq, hidden]: what the mixer adds to the stream
    pattern = model.layers[softmax[0]].self_attn.attention_probabilities.save()   # the attention block still has one

assert g.dtype == state.dtype == torch.float32          # the gate and the state are float32 whatever the model dtype
```

Read a call's values in forward order: queries/keys/values and gates, then
`attention_head_outputs`, then `state_output`, on a prompt. `state_input` is a
clone (the cache overwrites its buffer). `model.support()` on a hybrid is a dict
per value, `"no self_attn module on this block"` on the mixer blocks and the
reverse on the attention blocks; on Mamba-130m it repeats one reason per block
for hundreds of lines, so read `support(layer=i)`.

## The state after every token

<!-- test: skip -->
```python
with model.trace(prompt):
    states = mix.states.save()                 # [batch, seq, heads, key_dim, value_dim]
    final = mix.state_output.save()
assert torch.equal(states[:, -1], final)

with model.trace(prompt) as tracer:
    for t in tracer.iter[3]:
        mix.state = mix.state * 0              # a write at token 3: tokens 4.. continue from zeros
    for t in tracer.iter[5]:
        s5 = mix.state.save()                  # the state after token 5, downstream of the write
    logits = model.logits.save()
assert not torch.equal(s5, states[:, 5])

with model.trace(prompt):
    before = mix.state_after(2).save()                       # positions before a write: read first
    mix.set_state_after(3, torch.zeros_like(final))          # the same write as a call
    after = mix.state_after(5).save()                        # positions after it: read after
```

`states`, `state_after` and `set_state_after` count from the current call's own
first token (on a decode step, the one token). `states` reads every position, so
it goes before any write or in a trace of its own; `state` takes assignment, not
an in-place edit. Two things the state is not:

- **All a block remembers.** A width-4 causal conv runs before the scan and
  carries the last few tokens' inputs, so zeroing the state at `t` is not a
  clean "forget everything before t", and patching the state at a subject token
  barely moves a Mamba-1 answer until a token later.
- **Only downstream.** `set_state_after(t, v)` (and `mix.state = v` at step `t`)
  also changes token `t`'s own output, which is read from the state it writes.

## Under `generate`

A prompt runs the chunk (or scan) kernel and each decode step the single-step
kernel; the values follow whichever fires, and the state hands off:

<!-- test: skip -->
```python
entering, leaving = [], []                       # outside the block: a name bound inside does not survive it
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[:3]:
        s = mix.state_input
        entering.append(s.save() if s is not None else None)
        leaving.append(mix.state_output.save())

assert entering[0] is None and torch.equal(leaving[0], entering[1])   # step 1 starts from what step 0 left
```

**The decode-step read-order trap (Mamba-1, Mamba-2).** In the prompt's scan
`attention_head_outputs` is computed before `state_output`; in a decode step the
state is updated first and the output read from it. On a decode step read
`state_output` before `attention_head_outputs`. The other order does not raise:
on Mamba-1 it silently returns the *next* step's state (verified: the saved
`state_output` equals step 2's when asked for step 1); on Mamba-2 the block is cut
short with a `was never reached` warning that blames the loop. Read the two in
separate runs when unsure.

Writing `state_input` at a decode step continues from another prompt's memory:

<!-- test: skip -->
```python
with model.trace("My favourite food is pizza with"):
    other_state = mix.state_output.save()

with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[1]:
        mix.state_input = other_state          # decode step 1 starts from the other prompt's state
    for step in tracer.iter[2]:
        patched_entering = mix.state_input.save()
```

Assert on the state, not on the tokens: on Qwen3.5-0.8B the write lands and the
greedy tokens do not change (one mixer of many, and the conv window still holds
the real prompt).

## Writing Mamba-2's `betas` and `decays`

Both are the kernel's `dt` seen through `dt_bias`, the softplus and `A`, written
back by inverting them, so they are one argument:

- `betas[t] = 0` is an exact "skip token t" (no write, no decay).
- `decays = 0` ("keep everything") writes `betas = decays / A = 0` too: it zeroes
  every write, not just the decay.
- The write-back goes through the softplus inverse in the model's dtype: on a
  bf16 checkpoint `mix.betas = mix.betas` moves logits by up to 1.0, so an
  "assign it back" control is not a control. Write a change in float32, or skip
  the unchanged write.
- On a prompt the scan clamps `dt` to `time_step_limit`; a decode step does not.

## Gotchas

- **Route before the layer is traced**, and remember it is process-wide; on a
  DeltaNet an unrouted `state` / `states` raises `Unavailable` naming
  `route_kernels`.
- **`state_input` is `None` on a fresh prompt**; save it only when it is not.
- **Remotely, the per-token state is unavailable**: routing rebinds kernels in
  *your* process. The call-level values are ordinary source values there.
- **`layer_output` is float32** on Mamba / Falcon-Mamba (`residual_in_fp32`),
  and Mamba-2 / Nemotron-H return float32 logits.
- **A Mamba-1 checkpoint with `use_mambapy`** (and `mambapy` installed) runs a
  parallel scan with no per-token binding: `state` / `states` say so.
- **A prompt run with `use_cache=False`** returns no final state on Mamba-2:
  `state_output` raises `Unavailable`.

The nnterp repo pages behind this file: `docs/usage/delta-net.md`,
`docs/usage/selective-scan.md`, `docs/usage/state-space.md`,
`docs/patterns/delta-net-state.md`.
