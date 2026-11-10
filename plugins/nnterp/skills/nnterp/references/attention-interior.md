# The attention interior

`attention_output` is what attention adds to the residual stream. The six values
on `model.layers[i].self_attn` below are what happens *inside* it: the queries,
keys and values the softmax attention receives, the scores entering the softmax,
the pattern the values are mixed with, and each head's output before the heads
are concatenated and projected. Head ablation, pattern surgery, key-side
steering and query/key patching are reads and writes of these. Two more,
`attention_queries_prerope` and `attention_keys_prerope`, are the queries and keys one
step earlier, before the rotary embedding, with `rotate` to apply a block's rotary
yourself: the values to write when a query or key moves to another position
([below](#before-the-rotary-embedding-attention_queries_prerope-attention_keys_prerope-rotate)).
And `interface` is the block's attention function itself, to call again on queries, keys
and values of your own ([below](#calling-the-attention-function-again-interface)); the
per-head weights, `value_weight`, `output_weight` and `attention_input_gain`, are read off
the modules ([below](#the-weights-per-head)).

They are source-located: nnterp reaches them through nnsight `.source` at
transformers' shared eager attention call (`attention_interface_1`), or at the
family's own operations where it does its own arithmetic (GPT-J, GPT-Neo,
CodeGen, GPT-NeoX-Japanese, XGLM, BLOOM, MPT, Falcon), and presents one layout on
every family.

<!-- test: setup -->
```python
import torch
import nnsight
import nnterp
from nnterp import Attention, StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"
attn = model.layers[5].self_attn
```

## The six values, in one trace

```python
with model.trace(prompt):
    q = attn.attention_queries.save()               # [batch, heads, seq, qk_head_dim]   after RoPE
    k = attn.attention_keys.save()                  # [batch, kv_heads, seq, qk_head_dim] before repeat_kv
    v = attn.attention_values.save()                # [batch, kv_heads, seq, head_dim]
    scores = attn.attention_scores.save()           # [batch, heads, query, key]          masked and scaled
    pattern = attn.attention_probabilities.save()   # [batch, heads, query, key]          what the values are mixed with
    heads = attn.attention_head_outputs.save()      # [batch, seq, heads, head_dim]       before concat and c_proj
    attn.attention_head_outputs[:, :, 7] = 0        # ablate head 7 of block 5
    ablated = model.logits.save()

with model.trace(prompt):
    clean = model.logits.save()

assert q.shape == k.shape == v.shape == (1, 12, 10, 64)
assert scores.shape == pattern.shape == (1, 12, 10, 10)
assert heads.shape == (1, 10, 12, 64)
assert torch.allclose(scores.softmax(-1), pattern, atol=1e-6)      # no sink on GPT-2
assert torch.allclose(pattern.sum(-1), torch.ones_like(pattern.sum(-1)), atol=1e-5)
assert pattern.triu(diagonal=1).abs().max() == 0                   # causal
assert not torch.allclose(ablated, clean)
```

The six reads are in forward order, so they fit in one trace; a write follows
the same rule (write the head outputs after reading the scores, not before). A
read out of that order raises `OutOfOrderError` naming the attention call
(`'...attn.source.attention_interface_1.fn.i0' was requested but the model already
ran past it`), not the value. `model.order(i)` gives block `i`'s order on any family,
Falcon's different one included, and `model.rank(name, i)` sorts a list of reads and
writes by it ([availability-and-support.md](availability-and-support.md#forward-order-order-and-rank)).

| value | what it is | layout (`nnterp.components`) |
|---|---|---|
| `attention_queries` | the queries entering the interface, after the query projection and, where the family has one, after RoPE | `Queries`: `batch num_heads seq qk_head_dim` |
| `attention_keys` | the keys entering the interface, after RoPE and before `repeat_kv`: `num_kv_heads` wide under grouped-query attention | `Keys`: `batch num_kv_heads seq qk_head_dim` |
| `attention_values` | the values entering the interface, before `repeat_kv` | `Values`: `batch num_kv_heads seq head_dim` |
| `attention_scores` | the scaled, masked scores that enter the softmax | `Pattern`: `batch num_heads query key` |
| `attention_probabilities` | the pattern the values are mixed with: the dropout's output after the softmax, in the model dtype, a sink column already dropped | `Pattern` |
| `attention_head_outputs` | what the interface returns: each head's mix of the values, before the reshape to `[batch, seq, hidden]` and the output projection | `HeadOutputs`: `batch seq num_heads head_dim` |

The pattern is read after the dropout, not at the softmax, because that is the
tensor the values are multiplied with on every family: cast back to the model
dtype, and on a sink model without the sink column. The sequence axis is 2 on
the queries, keys and values (transformers' layout for its interface) and 1 on
the head outputs. Sizes come off the root: `model.num_heads`,
`model.num_kv_heads`, `model.head_dim`, `model.qk_head_dim`.

## `attention_premix`: the heads at the output projection

`self_attn.attention_premix` is what the output projection `o_proj` receives,
`[batch, seq, num_heads, head_dim]` (`HeadOutputs`), split by head as a view of the
projection's flat input. It is a module boundary, not an operation inside the eager
forward, so it is served under any `attn_implementation`; on a plain attention it is the
same numbers as `attention_head_outputs`, and on one with a gate or a norm between the
heads and the projection (AFMoE, Laguna, BitNet, Qwen3-Next) it is what comes after
them. `o_proj` is a standard child name: a family that spells it `c_proj`, `dense`,
`out_proj` or `proj` aliases it, so `layers[i].self_attn.o_proj` is the module on every
family. In-place edits reach the model; assignment flattens the heads back.

```python
fast = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True)     # sdpa, transformers' default
assert fast.support(layer=5)["self_attn.attention_head_outputs"] is not None           # inside the eager forward
assert fast.layers[5].self_attn.o_proj.path.endswith("h.5.attn.c_proj")

with model.trace(prompt):
    eager_heads = attn.attention_head_outputs.save()
    premix = attn.attention_premix.save()

with fast.trace(prompt):
    fast_premix = fast.layers[5].self_attn.attention_premix.clone().save()   # clone: the next line edits it in place
    fast.layers[5].self_attn.attention_premix[:, :, 7] = 0    # head 7 never reaches o_proj, under sdpa
    fast_ablated = fast.logits.save()

assert torch.equal(premix, eager_heads)                       # eager: the head outputs, at the projection
torch.testing.assert_close(fast_premix, eager_heads)          # sdpa computes the same heads, to float rounding
torch.testing.assert_close(fast_ablated, ablated, rtol=1e-4, atol=1e-4)    # the eager head-7 ablation above
```

Use `attention_premix` for head ablation, head patching and per-head attribution
wherever the pattern itself is not needed; it needs no eager load.

## Before the rotary embedding: `attention_queries_prerope`, `attention_keys_prerope`, `rotate`

`attention_queries` and `attention_keys` are rotated for the positions they run at. Two
values on `model.layers[i].self_attn` are the same queries and keys one step earlier, as
the block's rotary embedding receives them: after the projection and any query or key
norm (Qwen3's `q_norm` / `k_norm`, OLMo-2/3, Gemma-3/4), before the rotation. And
`attn.rotate(x, positions)` applies that block's own rotary embedding to any tensor laid
out like them, at the absolute positions you give it.

| value | what it is | layout (`nnterp.components`) |
|---|---|---|
| `attention_queries_prerope` | the queries the rotary embedding receives | `Queries`: `batch num_heads seq qk_head_dim` |
| `attention_keys_prerope` | the keys the rotary embedding receives, before the cache and `repeat_kv`: the new tokens' only | `Keys`: `batch num_kv_heads seq qk_head_dim` (`SharedRopeKeys` under latent attention, below) |

`attn.rotate(attention_queries_prerope, positions)` is `attention_queries` bit for bit, and
the same for the keys, wherever the block does nothing more between its rotary and its
attention ([below](#where-a-familys-values-before-the-rotary-differ)): `rotate` calls the
rotary embedding module the model computes its cos and sin with (rope scaling included, and
the block's own layer type where sliding and full blocks rotate differently) and the apply
function the block's forward calls (partial and interleaved rotations included). It takes
`x` as `[..., seq, qk_head_dim]` and `positions` as `[seq]` (a tensor, a list or a
`range`), or `[batch, seq]` for an `x` of `[batch, heads, seq, qk_head_dim]` whose prompts
sit at different positions, and needs no trace. Both values are on the attention module's
own forward, not inside the eager interface, so they are served under any
`attn_implementation`, and they come before `attention_queries` in a trace
(`model.order(i)` ranks them first).

Use them instead of `attention_queries` / `attention_keys` for any transplant to another
position or prompt (a rotated query carries its position in its rotation; written before
the rotary, it is rotated for the position it lands at, so the move is one assignment), for
an edit that must happen before the rotation (a direction found on the unrotated queries or
keys), and to recompute attention with keys from elsewhere (`rotate` the source keys at the
destination positions, outside the trace, and run the block's attention on them with
[`interface`](#calling-the-attention-function-again-interface)). The `nnterp` skill's "Before the rotary
embedding" runs the transplant. Use them instead of `self_attn.q_proj.output` too: nnterp
promises nothing about a projection's layout (Phi-3 and GPT-NeoX fuse the queries, keys and
values into one projection, Phi-3 has no `q_proj`), and on Qwen3, OLMo-2/3 and Gemma a norm
sits between the projection and the rotary, which a hand-rotated `k_proj.output` misses.

GPT-2 has no rotary embedding, so the check runs on SmolLM2:

```python
llama = StandardizedTransformer("HuggingFaceTB/SmolLM2-135M-Instruct", device="cpu", dispatch=True, attn_implementation="eager")
lattn = llama.layers[5].self_attn

with llama.trace("The Eiffel Tower is in Paris"):
    q_src = lattn.attention_queries_prerope.save()
    k_src = lattn.attention_keys_prerope.save()
with llama.trace("The Eiffel Tower is in Paris"):
    k = lattn.attention_keys.save()

assert q_src.shape == (1, 9, 7, 64) and k_src.shape == (1, 3, 7, 64)
assert torch.equal(lattn.rotate(k_src, range(k_src.shape[2])), k)   # the block's own rotation, outside the trace
```

On a block without a rotary embedding both values are `Unavailable` and `rotate` raises
the same reason (`no rotary embedding: ...`, or `no rotary embedding on this block: ...`):
GPT-2, GPT-BigCode, GPT-Neo, OPT and XGLM add absolute positions to the stream, BLOOM, MPT
and Falcon with `alibi` bias the scores with ALiBi, and the attention blocks of Jamba,
Nemotron-H and Kimi-Linear use no position encoding. Some families skip it per block: the
full-attention blocks of Cohere 2, AFMoE and a hybrid EXAONE 4, the blocks SmolLM3's and
Llama 4's `no_rope_layers` mark, a Granite-SWA block whose `layer_rope_theta` is 0, and
every block of a GraniteMoE-Hybrid or OLMo-Hybrid checkpoint whose config gives no rotary
(`position_embedding_type`, `rope_theta`). A vision tower's blocks report both unavailable
too: they are served on the text model's blocks. `support()` names the reason per block.

### Where a family's values before the rotary differ

- **A partial rotary** (GPT-NeoX, Phi, Phi-3, StableLM, Persimmon, GPT-NeoX-Japanese, GLM,
  MiniMax-M2, MiMo-V2-Flash, GPT-J and CodeGen with a `rotary_dim`): the values are the
  whole heads, `qk_head_dim` wide, though the rotary rotates only each head's leading
  channels; `rotate` rotates those and passes the rest, as the forward does.
- **GPT-J and CodeGen** rotate interleaved channel pairs, tokens before heads, with sin and
  cos from the module's own table; the values are served heads first as views, as
  everywhere else.
- **Latent attention** (DeepSeek-V2/V3, Kimi-K2, DeepSeek-V3.2, GLM-5, GLM-4.7-Flash, Youtu):
  only the trailing `qk_rope_head_dim` channels of a query head rotate, and one rope key per
  position that every head shares; the rest of each key comes out of the latent and carries
  no position. So `attention_queries_prerope` is whole heads (`Queries`) and
  `attention_keys_prerope` that shared key, `SharedRopeKeys`: `batch 1 seq qk_rope_head_dim`.
  `rotate` rotates a query head's trailing channels, or all of a rope key; every head's
  `attention_keys[..., -qk_rope_head_dim:]` is the rotated shared key.
- **Gemma-4**: a KV-sharing block rotates no keys of its own, so its
  `attention_keys_prerope` is `Unavailable` (`a KV-sharing block: its keys are an earlier
  block's, rotated there; ...`; read the source block's). A source block stores its rotated
  keys for its borrowers, so a write to its `attention_keys_prerope` reaches them too, as
  an edit of its `k_proj` does.
- **DeepSeek-V4** rotates each head's trailing rope slice of one tensor that is both its keys
  and its values: a write to `attention_keys_prerope` rewrites the values too. Its sliding
  and compressed blocks take the rotary's `main` and `compress` frequencies.
- **Steps between the rotary and the attention**: HunYuan (dense and MoE) norms the queries
  and keys after the rotary (`query_layernorm`, `key_layernorm`), and Llama 4 too
  (`qk_norm`, an L2 norm), so `attention_queries` is the norm of `rotate(...)` there; JetMoE
  tiles the keys to every routing slot (`repeat(1, top_k, 1, 1)`) after the rotary, so its
  `attention_keys_prerope` is `num_kv_heads` wide where `attention_keys` is not; Ministral 3
  scales its queries after the rotary by a factor that is 1 below
  `original_max_position_embeddings` and grows past it. `rotate` applies the rotary alone.

Assignment reaches the model on every family. An in-place edit does too, except where the
value is a split view of one fused tensor and torch refuses it as it does GPT-2's queries
(`RuntimeError: Output 0 of Select is a view and is being modified inplace`): CodeGen's,
DBRX's and GPT-NeoX's queries and keys, JetMoE's keys, and latent attention's shared rope
key. Assign there, or edit under `torch.no_grad()`.

### `rotate` and the positions it is given

A rope scaling that adapts to the sequence (`dynamic`, Phi-3's `longrope`) picks its
frequencies from the largest position in the call: the model's from the whole prompt,
`rotate`'s from the positions you pass. Past `original_max_position_embeddings` the two
differ unless the positions you pass reach as far as the prompt did; rotate the whole
sequence and index the result to match a long prompt. Under `generate` the keys before the
rotary on decode step `k` of `tracer.iter` (step 0 is the prompt) are the new token's only,
at position `n + k - 1` for a prompt of `n` tokens: `attn.rotate(keys, [n + k - 1])` is the
last row of that step's `attention_keys`. On a model split tensor-parallel the two values
are gathered along their head axis as the queries and keys are, so every rank reads and
writes them whole, and `rotate` runs on the whole tensor on every rank.

### On the vLLM engine

`StandardizedVLLM` serves both values as the arguments of the attention module's own
`self.rotary_emb(positions, q, k)` call, `[1, heads, tokens, head_dim]`, in these layouts.
vLLM's rotary kernel rotates in place, so they are private copies taken before it runs; a
write is rotated for the position it lands at, as here. Three things differ:

- **`rotate` runs inside a trace only.** It calls the block's vLLM rotary embedding, which
  lives in the engine's worker, on a copy of `x`; there `rotate(attention_queries_prerope,
  range(tokens))` is the `attention_queries` vLLM's attention layer receives, bit for bit, and
  outside a trace it raises `RuntimeError`. `positions` is `[tokens]`; on the M-RoPE rotary
  of the Qwen-VL text families a position is a text token's, the same on all three rows.
- **Not served on vLLM's transformers backend** (OLMo, OLMo 2, SmolLM3, StarCoder2): the
  rotary is a call inside a forward the backend rewrites at load, which nnsight cannot
  instrument.
- **Not served under latent attention** (DeepSeek-V2/V3, Kimi-K2, GLM-4.7-Flash): vLLM
  rotates only each query head's rope slice and the shared rope key, inside its
  latent-attention layer, where `StandardizedTransformer` serves whole-head queries.

On those families read the values, and rotate, on a `StandardizedTransformer`. The blocks
without a rotary embedding are the ones above, with the same reasons. The transplant on the
engine (vLLM 0.27.1, a GPU):

<!-- test: skip -->
```python
from nnterp import StandardizedVLLM

vmodel = StandardizedVLLM("HuggingFaceTB/SmolLM2-135M-Instruct", dispatch=True,
                          gpu_memory_utilization=0.2, max_model_len=1024)
attention = vmodel.layers[12].self_attn

source = "The capital of France is Paris"
with vmodel.trace(source, temperature=0.0, max_tokens=1):
    q_src = attention.attention_queries_prerope.save()      # [1, num_heads, tokens, head_dim], not rotated yet

with vmodel.trace("The Eiffel Tower is in the city of", temperature=0.0, max_tokens=1):
    before = attention.attention_queries_prerope
    last = before.shape[2] - 1
    before[:, :, last] = q_src[:, :, -1]                     # the source's last query, rotated for this position
    expected = attention.rotate(q_src[:, :, -1:], [last]).save()   # inside the trace: the engine's rotary
    moved = attention.attention_queries.save()

print(tuple(q_src.shape), torch.equal(moved[:, :, -1:], expected))
# (1, 9, 6, 64) True
```

HunYuan's and Llama 4's attention norm the queries and keys after the rotary on this engine
too, so `attention_queries` is the norm of `rotate(...)` there; on Llama 4 both values, and
`rotate`'s input and output, are in transformers' channel order.

## Requirement: eager attention

All six live inside the eager attention forward. A model loaded with `sdpa` or
`flash_attention_2` never runs it, and each value reports
`read inside the eager attention forward, but this model runs 'sdpa'; load with
attn_implementation='eager'` in `support()`; reading one raises
`nnterp.Unavailable` with the same reason before the model runs.
`attention_output` does not depend on the implementation. A load with no
`attn_implementation` gets transformers' default, `sdpa` on every family that
supports it, so pass `attn_implementation="eager"` at load for any of the six.
BLOOM, MPT, CodeGen, XGLM and GPT-NeoX-Japanese have no other implementation and
carry no such check; GPT-J, GPT-Neo and Falcon need the flag.
See [availability-and-support.md](availability-and-support.md).

## Reading, editing in place, assigning

Each value is the tensor the forward holds, so an in-place edit reaches the
model, and an assignment replaces it. Assigning one argument of the interface
replaces just that argument:

```python
with model.trace(prompt):
    attn.attention_scores[:, :, :, 0] = float("-inf")      # no query may attend to key 0 ...
    masked = attn.attention_probabilities.save()           # ... so the pattern's first column is zero
    _ = model.logits.save()

assert masked[..., 0].abs().max() == 0

with model.trace(prompt):
    attn.attention_keys = attn.attention_keys * 0          # only the keys change
    v_after = attn.attention_values.save()                 # untouched
    keyless = model.logits.save()

assert torch.equal(v_after, v) and not torch.allclose(keyless, clean)
```

A written pattern must be `[batch, heads, query, key]` in the model dtype; a
written argument must match the shape the interface expects. The model, not
nnterp, reports a mismatch, from inside the forward. Keep a replacement pattern
causal unless you mean to let tokens read the future.

**Overwriting the scores lifts the causal mask.** `attention_scores` is read
*after* the mask is added, so the future positions hold the mask's large negative
value; replacing the tensor replaces the mask too. Add to the scores (as above),
or re-apply the mask yourself:

```python
with model.trace(prompt):
    attn.attention_scores[:] = 0                           # overwrites the mask as well
    leaked = attn.attention_probabilities.save()

assert leaked.triu(diagonal=1).sum() > 0                   # every query now attends to the future
```

GPT-Neo's masked scores are `-inf` rather than the dtype's minimum, so
`scores * 0` is NaN there.

**Two invokes of one trace each read and edit their own rows of every interior value.** At
one location the invoke written first is served first: an empty `tracer.invoke()` after an
invoke that edits the scores reads the edited pattern. BLOOM, XGLM and Falcon with alibi
flatten batch and heads together for their `bmm`, which nnsight cannot narrow to an invoke's
rows, so their families read each value where the batch still leads (BLOOM's and Falcon's
head outputs once `_merge_heads` has merged them).

**Padded rows.** In a left-padded batch a pad *query* row attends uniformly over
all keys, which pollutes per-row head metrics (entropy, previous-token mass);
mask query rows with `attention_mask` before averaging.

**Recomputing by hand.** To compute with other tensors, call
[`interface`](#calling-the-attention-function-again-interface); to check the served values
against the arithmetic, two facts matter. The scale is the module's own (`attn._module.scaling`;
0.0078 on Granite-SWA, not `1/sqrt(head_dim)`; GPT-Neo has none), and
grouped-query expansion is `repeat_interleave` over the key/value heads.

## Calling the attention function again: `interface`

`model.layers[i].self_attn.interface` is the attention function this block's forward called
(`eager_attention_forward`, or the family's own: Gemma-2's with its logit softcap, GPT-OSS's
with its sink column), bound to everything else the call received this time: the mask, the
scale, the dropout, the sliding window. Call it on queries, keys and values and it returns
`(head_outputs, pattern)`, computed the way the model computes its own. On GPT-2 (`attn` is
block 5, above):

```python
with model.trace(prompt):
    q, k, v = attn.attention_queries, attn.attention_keys, attn.attention_values
    heads, pattern = attn.interface(q, k, v)         # the function this call ran, run again on the same tensors
    probs = attn.attention_probabilities.save()      # read after the call: the pattern comes later in the forward
    served = attn.attention_head_outputs.save()
    heads, pattern = heads.save(), pattern.save()

assert torch.equal(heads, served) and torch.equal(pattern, probs)   # the same function on the same shapes
assert heads.shape == (1, 10, 12, 64) and pattern.shape == (1, 12, 10, 10)
```

The tensors you pass take the layouts of `attention_queries`, `attention_keys` and
`attention_values`, keys and values `num_kv_heads` wide (the function repeats them for each
group itself), with the live call's query and key lengths, which its mask has. The result is
in the layouts of `attention_head_outputs` and `attention_probabilities`, and on the served
tensors it is those two values bit for bit; the per-family suite checks that on every
family, on the prompt, on a decode step and on a padded batch. The call computes and
returns: nothing reaches the model, the cache, the other heads or the other rows until you
write a result. `interface` is a `DerivedEProperty`: assigning it raises `AttributeError`;
write its result into `attention_head_outputs`.

Use it for:

- **any edit of keys or values that must not reach the cache or the other heads of a
  group.** Writing `attention_keys` changes the keys every query head of the group reads, at
  every query row; under `generate` an in-place edit also stays in the cache for every later
  step ([Under `generate`](#under-generate)). Call `interface` on edited copies instead and
  write only the heads you chose.
- **any per-row patch**: compute with the edited tensors, then write the rows you want of
  `attention_head_outputs` (the last one, for the prediction).
- **a recompute that must match the model bit for bit.** A hand-written
  `softmax(q @ k.T * scale)` needs the module's scale, misses Gemma-2's softcap and a sink
  column, and rounds differently in bf16 when its matmuls have other shapes than the
  model's; the call runs the model's function on the model's shapes.

A key or query edited before the rotary goes in rotated for its position,
`interface(q, attn.rotate(k_prerope, positions), v)`, with `attention_keys_prerope` read
first, since the forward reaches it before the interface
([Before the rotary embedding](#before-the-rotary-embedding-attention_queries_prerope-attention_keys_prerope-rotate)).

Reading `attn.interface` takes the call's arguments, so it is read where the queries, keys
and values are, in any order with them; calling the result takes the function, which the
forward hands over just before it runs it, so call it before reading or writing anything
later in the attention (the scores, the pattern, the head outputs). `model.order(i)` ranks
`self_attn.interface` between the values and the scores. Read it once per forward and call
the result as often as you need; a second `attn.interface` after a call, or a call after
the scores, the pattern or the head outputs, asks for a point the model has passed and
raises `OutOfOrderError` (`'...attention_interface_1.input.i0'` or `'...fn.i0' was requested
but the model already ran past it`), under `tracer.iter` too. Under `tracer.iter` a first
read placed after the step's later values binds the next step's call without an error, as
any late read there does. On SmolLM2, three query heads per key/value head:

```python
llama = StandardizedTransformer("HuggingFaceTB/SmolLM2-135M-Instruct", device="cpu", dispatch=True, attn_implementation="eager")
lattn = llama.layers[14].self_attn
fact = "The capital of France is"

with llama.trace(fact):
    interface = lattn.interface                      # read once, with the queries, keys and values
    lq, lk, lv = lattn.attention_queries, lattn.attention_keys, lattn.attention_values
    k2 = lk.clone()                                  # a copy: lk is what the model attends with (and, under generate, the cache)
    k2[:, 1, 0] = 0                                  # key/value head 1, position 0
    unedited, _ = interface(lq, lk, lv)
    edited, _ = interface(lq, k2, lv)                # called again: same mask, same scale
    lattn.attention_head_outputs[:, -1, 4] = edited[:, -1, 4]   # head 4, last row: the only write
    moved = edited.ne(unedited).flatten(0, 1).any(0).any(-1).save()

assert moved.nonzero().flatten().tolist() == [3, 4, 5]   # the result moves for the whole group; only head 4 is written
```

Inside `tracer.iter` it is that step's call: on a decode step the queries have one row, the
keys and values are the whole cache, and the mask is the step's (none, on an unpadded
`generate`). Within one invoke, a padded batch included, it is bit for bit. Across invokes
the model runs every invoke's rows in one call and `interface` runs one invoke's rows, so
the two agree to rounding (about `1e-8` in float32 on the tiny checkpoints), not bit for
bit: put the prompts in one invoke (`model.trace([a, b])`) when that matters. On a model
split tensor-parallel it takes the gathered queries, keys and values and runs every head on
every rank. In training mode with a dropout the call draws its own mask.

Which call it makes follows from where the family reads its queries, keys, values and head
outputs: the one call that takes the first three and returns the last. That is
transformers' shared interface call on most families, and the `_attn` method on GPT-J,
GPT-Neo, CodeGen and GPT-NeoX-Japanese. `support()` reports `self_attn.interface`
unavailable where those values are (any `attn_implementation` but `"eager"`, with the same
reason) and, with `OWN_ARITHMETIC`, where no such call exists: BLOOM, MPT, Falcon and XGLM
do their own arithmetic, and DeepSeek-V4 rotates the interface's output back before its
projection. The Qwen ViT and RADIO make one call per image and report that reason; the
SigLIP, CLIP, Pixtral, Gemma 4 and Llama 4 towers serve it. `StandardizedVLLM`'s attention
is the engine's kernel, which writes the step's keys and values into the cache as it runs,
so `interface` is unavailable there.

```python
from nnterp.components import OWN_ARITHMETIC

assert OWN_ARITHMETIC.startswith("the queries, keys and values are not the arguments of one call")
```

The head-selective recipe (chosen heads reading another run's keys and values over a span,
on the prompt and on every `generate` step) is the `patterns` skill's
`references/patching-and-attention.md`, "Head-selective K/V patching".

## The weights per head

`self_attn.value_weight(kv)` (`[head_dim, hidden_size]`), `self_attn.output_weight(j)`
(`[hidden_size, head_dim]`) and `layers[i].attention_input_gain` (`[hidden_size]`) are read
off the modules, outside a trace or inside one. `value_weight(kv) @ x`, plus that head's
slice of the projection's bias, is `attention_values[:, kv]` for what the attention is called
with; `sum_j output_weight(j) @ attention_premix[:, t, j]`, plus the bias, is what `o_proj`
returns at `t`; and the attention reads `attention_input_gain * n(x)` (plus the norm's bias)
of a stream `x` normalized by its norm (`x / rms(x)`, centred first on a LayerNorm). nnterp's
suite checks all three on every family, to the rounding of a matmul over a slice of rows.
On GPT-2 (`model`, above) `c_attn` is a `Conv1D` holding every query, key and value as
columns; the accessors read it:

```python
with model.trace(prompt):
    x = attn.input.save()
    v = attn.attention_values.save()
    premix = attn.attention_premix.save()
    out = attn.o_proj.output.save()

bias = attn.c_attn.bias.unflatten(0, (3, 12, 64))[2]            # [num_heads, head_dim]: the values' third of the bias
torch.testing.assert_close(x @ attn.value_weight(3).T + bias[3], v[:, 3], rtol=1e-4, atol=1e-4)
heads = sum(premix[:, :, j] @ attn.output_weight(j).T for j in range(12))
torch.testing.assert_close(heads + attn.o_proj.bias, out, rtol=1e-4, atol=1e-4)
gain = model.layers[5].attention_input_gain                     # ln_1's weight: a LayerNorm, its bias left out
assert torch.allclose(gain, model.layers[5].input_layernorm.weight, atol=1e-6)
```

The layouts a family's `Attention` reads for you: every query, every key, then every value
in one projection (Phi-3's `qkv_proj`, DBRX's and MPT's `Wqkv`, GPT-BigCode's `c_attn` under
multi-query); each head's query, key and value, `[num_heads, 3, head_dim]` (GPT-NeoX,
GPT-NeoX-Japanese, BLOOM, Persimmon, GPT-BigCode and Falcon-RW without multi-query); per
key/value head its group's queries, key and value (Falcon's 40B layout); every query, then
the one key and value (Falcon's multi-query 7B); four parts of the heads' queries, values and
keys (CodeGen); `Conv1D` columns and rows (GPT-2). MiMo-V2-Flash's `value_weight` carries the
`attention_value_scale` its forward multiplies the values by; DBRX, and MPT and OLMo with
`clip_qkv`, clamp the projections' outputs after these weights.

On a BitNet load (`quant_method: bitnet`, transformers' `AutoBitLinear` or `BitLinear`) the
accessors slice the weight a projection applies, not the one it stores: the stored weight
ternarized as each forward does it (`online`, `-bf16` checkpoints), times `weight_scale`
(`offline`, the packed checkpoint), or the unpacked codes over `weight_scale` (`BitLinear`).
The layer rounds its input to 8 bits per token first, so `value_weight(kv) @ x` gives the
values to that rounding (relative error 0.01 on bitnet-b1.58-2B-4T; 0.002 with the input
rounded as the layer rounds it).

Under a PEFT adapter the weight is the one the LoRA layer applies, computed as `merge()`
computes it, in the base weight's dtype: the base plus each active adapter's delta; the base
alone once an adapter is merged (the forward then runs the base layer only); the base without
a merged adapter when the adapters are disabled (the forward unmerges first). That is a new
tensor; otherwise the result is a view of the parameter. A DoRA or aLoRA adapter, or another
tuner, raises `NotImplementedError`. The gain is measured by calling the norm on `+-256` in
alternating channels, which any RMS norm or LayerNorm normalizes to `+-1`: exact on a norm
without a bias, to rounding with one; ones on a post-norm block; with
`attention_norm_output`'s reason where that is unavailable.

`self_attn.no_head_weights()` is the one reason `value_weight` and `output_weight` raise:
latent attention, DeepSeek-V4, Gemma-4 (`v_norm`), JetMoE (an output projection per expert),
ZAYA (half of each value head from the previous token), the default family, a vision tower,
a model split tensor-parallel (`nnterp.components.SHARDED_WEIGHTS`), and `StandardizedVLLM`.

## Family caveats

### GPT-2 and MPT: queries, keys and values are split views

GPT-2's `c_attn` produces one tensor that `split` divides into q, k, v; MPT's
`Wqkv` one that `chunk` divides. torch refuses to edit such a view in place:

<!-- test: expect-error RuntimeError -->
```python
with model.trace(prompt):
    attn.attention_queries[:, 0] = 0
    # RuntimeError: Output 0 of Select is a view and is being modified inplace ...
```

Assign instead (`attn.attention_queries = attn.attention_queries * 0`), or edit
under `torch.no_grad()`. On GPT-2 and GPT-BigCode only the queries refuse: their keys
and values take an in-place edit, and it lands. On MPT the queries refuse, and the keys and
values on a call without a cache; with one they are the cache update's new tensors and take
in-place edits, which the cache keeps. The scores, the pattern and the head outputs accept in-place edits on both (GPT-2's
head outputs are a transposed view, which torch allows). A GPT-2 checkpoint
with `reorder_and_upcast_attn` set (the `stanford-crfm` GPT-2s) calls GPT-2's own
`_upcast_and_reordered_attn(query, key, value, attention_mask)` under eager in the
interface's place; the six values and `interface` are read at that call, and the
scores there are float32 whatever the model's dtype. `openai-community/gpt2` does
not set it.

### Grouped-query attention: two head widths

On a checkpoint with `num_key_value_heads < num_attention_heads` the keys and
values are read *before* `repeat_kv`, so their head axis is `num_kv_heads` wide
while the queries, scores, pattern and head outputs are `num_heads` wide. A
head-wise edit of the keys touches the group of query heads sharing that key
head (`llama` is SmolLM2, loaded above):

```python
n = len(llama.tokenizer.encode(prompt))

with llama.trace(prompt):
    lq = llama.layers[5].self_attn.attention_queries.save()
    lk = llama.layers[5].self_attn.attention_keys.save()
    lp = llama.layers[5].self_attn.attention_probabilities.save()
    lh = llama.layers[5].self_attn.attention_head_outputs.save()

assert (llama.num_heads, llama.num_kv_heads) == (9, 3)
assert lq.shape == (1, 9, n, 64) and lk.shape == (1, 3, n, 64)
assert lp.shape == (1, 9, n, n) and lh.shape == (1, n, 9, 64)
assert torch.allclose(lp.sum(-1), torch.ones_like(lp.sum(-1)), atol=5e-3)   # bf16: rows sum to 1 within ~4e-3
```

On Llama the queries, keys and values also take in-place edits.

### Falcon: two branches, picked by `config.alibi`

Falcon's attention does its own arithmetic, and `config.alibi` picks one of two
branches with different operations; the family names every interior value's
operation by that flag, so all six are available under eager on both. Without
alibi the queries are the rotary embedding's first return; with alibi there is no
rotary and they are the reshaped projection. Either branch then updates the cache
itself, and where the call has a cache (every call but `use_cache=False`) the keys
and values are what the update returns, every position so far, read after the
queries:

<!-- test: skip -->
```python
falcon = StandardizedTransformer("tiiuae/falcon-7b", device="cpu", dispatch=True, attn_implementation="eager")
fattn = falcon.layers[0].self_attn

with falcon.trace(prompt):
    q = fattn.attention_queries.save()
    k = fattn.attention_keys.save()         # the cache update's return
    v = fattn.attention_values.save()
```

Without a cache they are the new tokens' own: the rotary's second return and the
`value_layer` binding before it without alibi (read the values first there), the
reshaped projections with alibi. Whether the call has a cache is read off the
attention's arguments by its first standard value, so read its standard values before
its child modules' (`query_key_value.output`) in one trace. Without alibi the pattern
is the first softmax's output itself (no dropout follows); with alibi it is the dropout
after the second softmax, and the head outputs are read once `_merge_heads` has merged
them (the product before it is flattened over batch and heads), served as the same
`[batch, seq, heads, head_dim]` view as elsewhere, so in-place edits land. The 7B
layout is multi-query on both branches: keys and values have one head
(`num_kv_heads == 1`).

### Falcon-40B layout: keys and values already expanded

The `new_decoder_architecture` layout broadcasts its key/value heads to every
query head before the rotary embedding, so `attention_keys` and
`attention_values` are `num_heads` wide there, although `model.num_kv_heads`
reports the config's `num_kv_heads` (8).

### Own-arithmetic families: the same six on their own operations

GPT-J, GPT-Neo, CodeGen, GPT-NeoX-Japanese, XGLM, BLOOM, MPT and Falcon do their
own attention arithmetic, and their families map the same six values onto it
(GPT-J/GPT-Neo/CodeGen/GPT-NeoX-Japanese around a `self._attn(q, k, v, mask)`
call; BLOOM, XGLM and MPT on their projections, their cache update and the bindings
around their matmuls, where the batch still leads). BLOOM, MPT, Falcon and XGLM update
the cache in their own forwards, and their keys and values are the update's return where
the call has a cache, as on every other family. Heads-first head outputs are served as a
sequence-first view, so in-place edits land. Where the scale sits differs (GPT-Neo
has none, CodeGen divides after the mask, XGLM scales the queries as it projects
them). GPT-Neo's `self_attn` is the module inside the `attn` wrapper. Their output
projection is `out_proj` (GPT-Neo, CodeGen) or `dense`, not `o_proj` / `c_proj`:
search all four when you need it. BLOOM's q/k/v take in-place edits; MPT's are
split views (above); GPT-J's keys and values are `num_heads` wide. `self_attn.interface`
calls the `_attn` method on GPT-J, GPT-Neo, CodeGen and GPT-NeoX-Japanese, and is
`Unavailable` (`OWN_ARITHMETIC`) on XGLM, BLOOM, MPT and Falcon, whose values are not one
call's arguments.

### DeepSeek-V4: one tensor for keys and values, compressed keys, rotated-back heads

Plain sizes (one key/value head; not latent attention), GPT-OSS's sink on every block
(`SINK = True`, scores at `attn_weights_1`). `attention_keys` and `attention_values` are
one tensor object: an in-place edit of either edits both; an assignment to one separates
it. On a compressed block (`compressed_sparse_attention`, `heavily_compressed_attention`)
the compressor's entries follow the token keys once the prompt reaches the block's rate,
so the keys and the pattern's key axis are `seq + seq // rate` long.
`attention_head_outputs` is the interface's output with its rotary slice rotated back,
what the grouped output projection (`o_a_proj`) reads, so `self_attn.interface` is
`Unavailable` (`OWN_ARITHMETIC`) there.

### Gemma-4: borrowed keys and values, per-layer head sizes

The last `num_kv_shared_layers` blocks (20 of 35 on E2B, 18 of 42 on E4B) have no
`k_proj` / `v_proj`: each attends with the keys and values of the last earlier block
of its kind (sliding or full). `attention_keys` / `attention_values` are what the
attention receives, served as a copy private to the block on every Gemma-4 block: an edit,
in place or by assignment, changes what that block attends with and nothing else, and its
borrowers keep the source block's tensors as computed. To change the shared keys or values
for every borrower, edit each block that attends with them, or the source block's `k_proj` /
`v_proj` output, which the shared tensors are computed from. Skipping a source block
with `skip_layers` fails in transformers (`KeyError: 'sliding_attention'`). On
`attention_k_eq_v` checkpoints (26B-A4B, 31B, 12B) the full blocks' values are
`v_norm(k_proj(x))`. Sliding blocks have `head_dim` 256 and full blocks 512, often
with fewer key/value heads; `model.head_dim` / `model.num_kv_heads` are the sliding
blocks' (the config's top-level values), so read a full block's widths off its
tensors.

### Granite-SWA: a sink that scales the head outputs

`granite_swa` / `granitemoe_swa` carry a per-head sink outside the softmax: the
pattern's rows sum to one and `softmax(attention_scores)` reproduces it, but each
query's head output is then scaled by `sigmoid(logsumexp(scores) - sink)` (as low
as 0.001 on real weights). Forcing the pattern does not control the output there;
`attention_scores` is read before both the softmax and the scale, so a write to the
scores moves the pattern and the scale together. `SINK` is `False`. GPT-OSS, DeepSeek-V4 and MiMo-V2-Flash's sliding blocks put
the sink in the pattern instead (below).

### GPT-OSS: an attention sink

Each GPT-OSS head carries a learned sink logit that joins the softmax as one
extra key column and is dropped afterwards. So the pattern's rows sum to *less*
than one (`Attention.SINK` is `True` on the family), and `attention_scores` is
read at the masked scores just before the sink column joins them, the same
shape as the pattern. A plain `softmax(scores)` does not reproduce the pattern;
appending the sink does:

<!-- test: skip -->
```python
oss = StandardizedTransformer("openai/gpt-oss-20b", device="cuda", dispatch=True, attn_implementation="eager")
oattn = oss.layers[0].self_attn

with oss.trace(prompt):
    scores = oattn.attention_scores.save()
    pattern = oattn.attention_probabilities.save()

assert pattern.sum(-1).max() < 1
sinks = oattn._module.sinks.to(scores.dtype)                                    # [heads]
column = sinks.view(1, -1, 1, 1).expand(scores.shape[0], -1, scores.shape[2], 1)
combined = torch.cat([scores, column], dim=-1)
combined = combined - combined.max(dim=-1, keepdim=True).values
assert torch.allclose(combined.softmax(-1)[..., :-1], pattern, atol=1e-4)      # the pattern again
```

### DeepSeek: multi-head latent attention

DeepSeek-V2/V3 give queries and keys `qk_head_dim = qk_nope_head_dim +
qk_rope_head_dim` and values `v_head_dim`, so `attention_queries` and
`attention_keys` are `model.qk_head_dim` wide (192 on V3) and `attention_values`
and `attention_head_outputs` are `model.head_dim` wide (128). The interface sees
`num_heads` key/value heads whatever `num_key_value_heads` says: the latent
projection produces keys and values for every head, so `attention_keys.shape[1]
== num_heads` there, not `num_kv_heads`. Before the rotary only part of that is there:
`attention_keys_prerope` is the one rope key per position that every head shares,
`SharedRopeKeys`, `qk_rope_head_dim` wide
([above](#where-a-familys-values-before-the-rotary-differ)).

<!-- test: skip -->
```python
deepseek = StandardizedTransformer("deepseek-ai/DeepSeek-V3", attn_implementation="eager")
assert (deepseek.head_dim, deepseek.qk_head_dim) == (128, 192)

with deepseek.trace(prompt):
    q = deepseek.layers[0].self_attn.attention_queries.save()        # (1, heads, seq, 192)
    v = deepseek.layers[0].self_attn.attention_values.save()         # (1, heads, seq, 128)
```

## Under `generate`

The values are per forward call. On a decode step the queries, scores, pattern
and head outputs have one query position, and the keys, values and the
pattern's key axis are as long as the cache: `attention_probabilities` is
`[batch, heads, 1, prompt + step]`, and `attention_probabilities[0, h, 0]` is
what head `h` of the new token attends to across the whole context so far.
Read `input_ids` first within a step, then the block's values:

```python
with model.generate(prompt, max_new_tokens=2, do_sample=False) as tracer:
    for step in tracer.iter[1]:                                    # the first decode step
        step_ids = model.input_ids.save()
        step_k = model.layers[3].self_attn.attention_keys.save()
        step_p = model.layers[3].self_attn.attention_probabilities.save()

assert step_ids.shape == (1, 1)
assert step_k.shape == (1, 12, 11, 64) and step_p.shape == (1, 12, 1, 11)   # the cache: 10 prompt + 1
```

The keys and values a decode step's attention receives are the cache's own tensors on
most families, so an in-place edit of `attention_keys` or `attention_values` writes into the
cache and every later step attends with it, while an assignment replaces the call's
argument for that one forward. Under latent attention (DeepSeek-V2/V3 and the families on
their attention) the cache holds the compressed latent and the keys and values are expanded
from it on every call, and Gemma-4 serves copies, so there an edit lasts one forward either
way. To change what chosen heads read without touching the cache, call
[`interface`](#calling-the-attention-function-again-interface) on edited copies.

See [generation-and-helpers.md](generation-and-helpers.md).

## Gotchas

- `attn_implementation="eager"` or nothing: under `sdpa` every interior value is
  `Unavailable`, and `support()` says so per block before any trace.
- Reads follow the forward within one trace: the queries and keys before the
  rotary, then q, k, v and `interface`, then scores, then pattern, then head outputs; on Falcon,
  BLOOM and MPT the keys and values after the queries where the call has a cache (on
  Falcon without alibi and without one, the values first). Out of order raises `OutOfOrderError` naming the attention call. Under
  `generate` + `tracer.iter` it does not raise: the read binds the *next* step's
  value.
- Read `interface` once per forward, and call it before the scores. Reading it again
  after a call, or calling it after reading the scores, the pattern or the head outputs,
  asks for a point the forward has passed. Across invokes it matches the model to
  rounding only; one invoke is bit for bit.
- An in-place edit of `attention_keys` / `attention_values` on a decode step stays in the
  cache on most families; an assignment lasts one forward. To patch what chosen heads
  read without either, use `interface` on copies.
- Never overwrite `attention_scores` wholesale (it holds the mask).
- GPT-2's and GPT-BigCode's queries, MPT's queries, keys and values: assign, do not
  edit in place (or edit under `torch.no_grad()`); GPT-2's and GPT-BigCode's keys and
  values take an in-place edit. The same before the rotary
  for CodeGen's, DBRX's and GPT-NeoX's queries and keys, JetMoE's keys and latent
  attention's shared rope key.
- A query or key moved to another position or prompt is written to
  `attention_queries_prerope` / `attention_keys_prerope`, never to the rotated
  `attention_queries` / `attention_keys`, which carry the position they ran at.
- `rotate` is the rotary alone: on HunYuan and Llama 4 `attention_queries` is the
  norm of its result, JetMoE tiles the keys after it, Ministral 3 scales the
  queries after it. Under `dynamic` or `longrope` scaling pass positions reaching
  as far as the prompt did. On `StandardizedVLLM` it runs inside a trace only.
- `hasattr(attn, "attention_probabilities")` raises `Unavailable` when
  unavailable and `ValueError` outside a trace when available. Use
  `attn.support()` or `model.support(layer=i)`.
- A sink model's pattern rows sum to less than one, and its `attention_scores`
  are one step before the softmax's own input.
- On a hybrid, three blocks in four have no `self_attn`. Decide which blocks have
  one outside the trace; see [recurrent-mixers.md](recurrent-mixers.md).
- Remote runs re-resolve these on the server against its transformers; see
  [generation-and-helpers.md](generation-and-helpers.md).

Under eager attention the raw `self_attn.output[1]` is the same pattern, a
label-free second reading to sanity-check against; it is read-only (the weights
are returned, not consumed). Edit through `attention_probabilities`. The
`attention-analysis` skill has the per-head metrics.

The nnterp repo page behind this file: `docs/usage/attention-interior.md`.
