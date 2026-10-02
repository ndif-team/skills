# Availability: `support()`, `Unavailable`, `SourceNotAvailable`

Not every checkpoint has every standard value. OPT has no MLP module; a model
loaded with `sdpa` never builds the attention pattern; GPT-2 with
`reorder_and_upcast_attn` leaves the shared attention interface; a hybrid's linear
blocks have no softmax and their per-token state needs a kernel switch; a dense
block beside mixture blocks has no routing. Every nnterp value can say
when it is not there and why, and `support()` collects those reasons from the
config alone, so a script decides what to read before anything runs.

`None` means available. Anything else is the reason, a string.

<!-- test: setup -->
```python
import torch
import nnsight
import nnterp
from nnterp import StandardizedTransformer, Unavailable

plain = StandardizedTransformer("openai-community/gpt2", device="cpu")   # no dispatch: meta, no weights yet; transformers' default attention
prompt = "The Eiffel Tower is in the city of"
```

## The three forms

`model.support()` is the root's values plus every block value, by dotted name. A
block value is `None` when every block has it, else `{layer: reason}` for the
blocks that do not. `model.support(layer=i)` is one block, flat. `envoy.support()`
is one envoy's own values. Block values are walked off the tree: every block child
that carries standard values is listed under its standard name, so a value installed
through `envoys=` appears as `self_attn.<name>`, and a module no block has (OPT's
`mlp`) has no key at all; `.get` a key on a family you do not know. All three read
the config, so they work on a meta model:

```python
EAGER = "read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'"

assert plain.config._attn_implementation == "sdpa"
support = plain.support()
assert support["layer_output"] is None and support["self_attn.attention_output"] is None and support["mlp.mlp_output"] is None
assert support["self_attn.attention_probabilities"] == {i: EAGER for i in range(12)}
assert set(support) == {
    "logits", "token_embeddings", "next_token_probs", "input_ids", "attention_mask", "input_size",
    "layer_output", "self_attn.attention_output", "self_attn.attention_probabilities",
    "self_attn.attention_queries", "self_attn.attention_keys", "self_attn.attention_values",
    "self_attn.attention_scores", "self_attn.attention_head_outputs", "mlp.mlp_output",
}
assert plain.support(layer=0)["self_attn.attention_probabilities"] == EAGER          # one block, flat
assert plain.layers[0].self_attn.support()["attention_probabilities"] == EAGER       # one envoy
assert plain.layers[0].self_attn.support()["attention_output"] is None
```

Loaded with `attn_implementation="eager"`, every `self_attn.*` entry is `None`
on GPT-2 and Llama. The root values (`logits`, `token_embeddings`,
`next_token_probs`, `input_ids`, `attention_mask`, `input_size`) and the three
boundary values never depend on the implementation.

A hybrid (`Qwen/Qwen3.5-9B` with `attn_implementation="eager"`) reads as a dict
per value, because each block has `self_attn` or `linear_attn` and the per-token
state needs routing (verified on the tiny Qwen3.5 checkpoint):

<!-- test: skip -->
```python
hybrid = StandardizedTransformer("Qwen/Qwen3.5-9B", attn_implementation="eager")
support = hybrid.support()
# support["self_attn.attention_output"]   -> {0: 'no self_attn module on this block', 1: ..., 2: ..., ...}   the DeltaNet blocks
# support["linear_attn.decays"]           -> {3: 'no linear_attn module on this block', 7: ..., ...}          the attention blocks
# support["linear_attn.state"]            -> {0: "the state after each token is materialized only by the token-by-token kernel; ...", ..., 3: 'no linear_attn module on this block', ...}
# support["mlp.mlp_output"]               -> None
```

A pure state-space model (Mamba-130m) repeats one reason per block, hundreds of
lines; read `support(layer=i)` there.

## Reading an unavailable value

Reading or writing one raises `nnterp.Unavailable` (a `RuntimeError`) with the
same reason, at that line, before the forward runs:

<!-- test: expect-error Unavailable -->
```python
with plain.trace(prompt):
    pattern = plain.layers[0].self_attn.attention_probabilities.save()
    # Unavailable: model.transformer.h.0.attn.attention_probabilities is not available:
    #   read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'
```

The trace above dispatched the weights (`trace` loads a meta model) and then
raised; the model is usable afterwards, and the boundary values work under
`sdpa`:

```python
assert plain.dispatched

with plain.trace(prompt):
    attn = plain.layers[0].self_attn.attention_output.save()
    out = plain.layers[0].layer_output.save()
    logits = plain.logits.save()

assert attn.shape == out.shape == (1, 10, 768)
assert plain.tokenizer.decode(logits[0, -1].argmax()) == " Paris"
```

A module that does not exist on a block is an ordinary missing attribute:
`model.layers[0].mlp` on OPT raises `AttributeError`, and
`getattr(layer, "mlp", None)` is `None`. Use that form outside the trace to pick
blocks.

## `hasattr` raises

Python's `hasattr` treats only `AttributeError` as absence. An unavailable value
raises `Unavailable` through it; an *available* value read outside a trace raises
`ValueError` (`Cannot access ... outside of interleaving`), or, for one read inside
the attention call's own source (`attention_probabilities`, `attention_scores`),
`SourceNotAvailable` (`recursive .source is only available inside a trace`).
Never `False`:

<!-- test: expect-error Unavailable -->
```python
hasattr(plain.layers[0].self_attn, "attention_probabilities")
```

```python
from nnsight.intervention.source import SourceNotAvailable

eager = StandardizedTransformer("openai-community/gpt2", device="cpu", attn_implementation="eager")   # meta, eager

outcomes = {}
for name in ("attention_queries", "attention_probabilities"):
    try:
        outcomes[name] = hasattr(eager.layers[0].self_attn, name)
    except (ValueError, SourceNotAvailable) as error:
        outcomes[name] = type(error).__name__

assert outcomes == {"attention_queries": "ValueError", "attention_probabilities": "SourceNotAvailable"}
assert eager.support(layer=0)["self_attn.attention_probabilities"] is None               # the question hasattr cannot answer
```

## The reasons you will see

| reason (exact string, or its prefix) | value(s) | when |
|---|---|---|
| `read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'` | the attention interior on every interface family (and GPT-J, Falcon) | loaded with any `attn_implementation` other than `"eager"`; the quoted name is whatever the model runs |
| `this checkpoint sets reorder_and_upcast_attn, which takes GPT-2's own upcast attention path` | the GPT-2 interior | `config.reorder_and_upcast_attn` is set |
| `The attention does its own arithmetic rather than transformers' shared attention interface; not mapped for this family yet` | an interior value a family has not mapped | a family off the shared interface that marks it `unavailable(NOT_ON_INTERFACE)`; no shipped family does |
| `no self_attn module on this block` / `no linear_attn module on this block` | every value of that module | another block has the module and this one does not (a hybrid's blocks). A module no block has (OPT's `mlp`) has no key in `support()` at all |
| `the state after each token is materialized only by the token-by-token kernel; the chunked kernel a prompt runs through carries it between chunks. Call nnterp.route_kernels(model.family, 'torch') before tracing this layer ...` | a DeltaNet's `linear_attn.state`, `.states` | the family is not routed |
| `read inside transformers' pure-torch <kernel>, but this process dispatches it to an optimized kernel (<package>) with no Python source; uninstall it, or call nnterp.route_kernels(model.family, 'torch'), to read these` | every recurrent `linear_attn` value but `attention_output` | `mamba_ssm` / `flash-linear-attention` installed and the family not routed |
| `the chunk scan materializes the state only at chunk boundaries, every <n> tokens (chunk_size=<n>); call nnterp.chunk_per_token(model) ...` | a Mamba-2 `linear_attn.states` | `chunk_per_token` not set |
| `the chunk scan computes every token's state in one tensor per call, ...` / `the chunk scan computes every boundary state in one cumulative step ...` | a Mamba-2 `linear_attn.state` / `.set_state_after` | always |
| `no <value> value on this block's mlp` | a mixture's six values | a dense block beside mixture blocks |
| `read inside transformers' grouped_mm / batched_mm experts forward, but this model runs 'eager'; load with experts_implementation='grouped_mm' (the default) or 'batched_mm'` | `mlp.expert_outputs` | `experts_implementation="eager"` |
| `this mixture has no shared expert` | `mlp.shared_expert_output` | a mixture without one |
| `this checkpoint has no per-layer embeddings ...` | Gemma-4's `per_layer_output` | 26B-A4B, 31B |

BLOOM and MPT do their attention arithmetic themselves, so their pattern and
interior need no eager load and are `None` under any implementation.

The reasons are evaluated on the instance, so a change after the load changes the
answer: `route_kernels(model.family, "torch")` turns a DeltaNet's `state` and
`states` entries to `None` on the linear blocks. Reading a mixture's
`num_experts`, `top_k` or `SCORING` on a dense block raises a bare
`AttributeError`, not `Unavailable`; pick the blocks with `support()` first.

## `SourceNotAvailable`: the forward took another path

`support()` predicts from the config. A source-located value is then read at a
named operation inside the forward, and if this run's forward does not contain
that operation, the read raises nnsight's `SourceNotAvailable` naming the value,
the operation and what the forward does have:

```
SourceNotAvailable: model.transformer.h.0.attn.<value> reads operation '<op>' under .source, which
this run does not have: 'model.transformer.h.0.attn.source' has no operation '<op>'; available:
is_cross_attention_0, ..., attention_interface_0, attention_interface_1, .... The forward took a
path this family's toolkit does not expect.
```

`Unavailable` is the expected answer for a known configuration;
`SourceNotAvailable` means a forward the family has not been checked against
(another transformers release renaming an operation, a branch the config does
not announce, a remote server on another transformers). `print(envoy.source)`
lists the real operation names; the `extending` skill covers relocating a
value.

## Guarding a script

Check `support()` outside the trace and branch there; the trace body then reads
only what exists:

```python
model = plain                                            # any StandardizedTransformer
support = model.support()
has_pattern = support["self_attn.attention_probabilities"] is None
has_mlp = support.get("mlp.mlp_output", "absent") is None   # no key at all where no block has an mlp (OPT)
attn_blocks = [i for i, layer in enumerate(model.layers) if getattr(layer, "self_attn", None) is not None]

with model.trace(prompt):
    patterns = nnsight.save({})
    for i in attn_blocks[:3]:
        if has_pattern:
            patterns[i] = model.layers[i].self_attn.attention_probabilities
    mlp = model.layers[0].mlp.mlp_output.save() if has_mlp else None

assert patterns == {} and mlp.shape == (1, 10, 768)      # sdpa here: no patterns, but the MLP
```

For one block, `model.support(layer=i)["self_attn.attention_probabilities"]` is
the flat form of the same check. `scripts/inspect_family.py <repo_id>` prints
`support()` for a checkpoint without loading weights.

## Gotchas

- `hasattr(envoy, "attention_probabilities")` raises; use `support()`.
- `getattr(envoy, name, None)` inside a trace can trip a served value. Decide
  which blocks have `self_attn` or `mlp` before the trace.
- `support()` predicts; the forward decides. A `SourceNotAvailable` at read time
  means the forward differs from what the family expects, not that the value is
  unavailable by configuration.
- `attn_implementation` is transformers' default unless you pass it. The most
  common reason in `support()` is the `'sdpa'` one, and the fix is in the message.
- The repr lists a value whether or not this checkpoint has it. Only a value a
  family marks `unavailable("...")` in its class body prints as
  `Unavailable: <reason>`; a config-dependent reason (eager, `reorder_and_upcast_attn`) shows only in
  `support()`.
- A hybrid's `support()` is a long dict by design; read `support(layer=i)` for one
  block.

The nnterp repo page behind this file: `docs/usage/availability.md`.
