# The attention interior

`attention_output` is what attention adds to the residual stream. The six values
on `model.layers[i].self_attn` below are what happens *inside* it: the queries,
keys and values the softmax attention receives, the scores entering the softmax,
the pattern the values are mixed with, and each head's output before the heads
are concatenated and projected. Head ablation, pattern surgery, key-side
steering and query/key patching are reads and writes of these.

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
ran past it`), not the value.

| value | what it is | layout (`nnterp.components`) |
|---|---|---|
| `attention_queries` | the queries entering the interface, after the query projection and, where the family has one, after RoPE | `Queries`: `batch heads seq qk_head_dim` |
| `attention_keys` | the keys entering the interface, after RoPE and before `repeat_kv`: `num_kv_heads` wide under grouped-query attention | `Keys`: `batch kv_heads seq qk_head_dim` |
| `attention_values` | the values entering the interface, before `repeat_kv` | `Values`: `batch kv_heads seq head_dim` |
| `attention_scores` | the scaled, masked scores that enter the softmax | `Pattern`: `batch heads query key` |
| `attention_probabilities` | the pattern the values are mixed with: the dropout's output after the softmax, in the model dtype, a sink column already dropped | `Pattern` |
| `attention_head_outputs` | what the interface returns: each head's mix of the values, before the reshape to `[batch, seq, hidden]` and the output projection | `HeadOutputs`: `batch seq heads head_dim` |

The pattern is read after the dropout, not at the softmax, because that is the
tensor the values are multiplied with on every family: cast back to the model
dtype, and on a sink model without the sink column. The sequence axis is 2 on
the queries, keys and values (transformers' layout for its interface) and 1 on
the head outputs. Sizes come off the root: `model.num_heads`,
`model.num_kv_heads`, `model.head_dim`, `model.qk_head_dim`.

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

**Patterns in two invokes of one trace raise.** Touching `attention_probabilities`
or `attention_scores` (any value read inside the attention call's own source) in
two invokes of one trace fails with `TypeError: 'NoneType' object is not
subscriptable`, an nnsight bug. Put the prompts in one invoke (a batch), or use a
trace per prompt; the "every head in one forward" sweep goes through
`attention_head_outputs` instead.

**Padded rows.** In a left-padded batch a pad *query* row attends uniformly over
all keys, which pollutes per-row head metrics (entropy, previous-token mass);
mask query rows with `attention_mask` before averaging.

**Recomputing by hand.** The scale is the module's own (`attn._module.scaling`;
0.0078 on Granite-SWA, not `1/sqrt(head_dim)`; GPT-Neo has none), and
grouped-query expansion is `repeat_interleave` over the key/value heads.

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

Assign instead (`attn.attention_queries = attn.attention_queries * 0`). The
scores, the pattern and the head outputs accept in-place edits on both (GPT-2's
head outputs are a transposed view, which torch allows). A GPT-2 checkpoint
with `reorder_and_upcast_attn` set takes GPT-2's own upcast path, where the
interface never runs; all six then report
`this checkpoint sets reorder_and_upcast_attn, which takes GPT-2's own upcast
attention path`. `openai-community/gpt2` does not set it.

### Grouped-query attention: two head widths

On a checkpoint with `num_key_value_heads < num_attention_heads` the keys and
values are read *before* `repeat_kv`, so their head axis is `num_kv_heads` wide
while the queries, scores, pattern and head outputs are `num_heads` wide. A
head-wise edit of the keys touches the group of query heads sharing that key
head:

```python
llama = StandardizedTransformer("HuggingFaceTB/SmolLM2-135M-Instruct", device="cpu", dispatch=True, attn_implementation="eager")
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
alibi the queries and keys are the two returns of the rotary embedding, and the
values are bound *before* it, so in one trace the values must be read before
the queries or keys:

<!-- test: skip -->
```python
falcon = StandardizedTransformer("tiiuae/falcon-7b", device="cpu", dispatch=True, attn_implementation="eager")
fattn = falcon.layers[0].self_attn

with falcon.trace(prompt):
    v = fattn.attention_values.save()       # binds first
    q = fattn.attention_queries.save()
    k = fattn.attention_keys.save()
```

The other order raises `OutOfOrderError` naming `value_layer_0`. On that branch
the pattern is the first softmax's output itself (no dropout follows). With
alibi there is no rotary: queries, keys and values are the reshaped projections,
bound in that order, so read them as queries, keys, values (values first raises
`OutOfOrderError` naming `query_layer_0`); the pattern is the dropout after the
second softmax, and the head outputs are computed flattened over batch and
heads but served as the same `[batch, seq, heads, head_dim]` view as elsewhere,
so in-place edits land. The 7B layout is multi-query on both branches: keys and
values have one head (`num_kv_heads == 1`).

### Falcon-40B layout: keys and values already expanded

The `new_decoder_architecture` layout broadcasts its key/value heads to every
query head before the rotary embedding, so `attention_keys` and
`attention_values` are `num_heads` wide there, although `model.num_kv_heads`
reports the config's `num_kv_heads` (8).

### Own-arithmetic families: the same six on their own operations

GPT-J, GPT-Neo, CodeGen, GPT-NeoX-Japanese, XGLM, BLOOM, MPT and Falcon do their
own attention arithmetic, and their families map the same six values onto it
(GPT-J/GPT-Neo/CodeGen/GPT-NeoX-Japanese around a `self._attn(q, k, v, mask)`
call; BLOOM and XGLM as views of their flattened `[batch * heads, ...]` tensors;
MPT on its `*_states` bindings). Heads-first head outputs are served as a
sequence-first view, so in-place edits land. Where the scale sits differs (GPT-Neo
has none, CodeGen divides after the mask, XGLM scales the queries as it projects
them). GPT-Neo's `self_attn` is the module inside the `attn` wrapper. Their output
projection is `out_proj` (GPT-Neo, CodeGen) or `dense`, not `o_proj` / `c_proj`:
search all four when you need it. BLOOM's q/k/v take in-place edits; MPT's are
split views (above); GPT-J's keys and values are `num_heads` wide.

### DeepSeek-V4: one tensor for keys and values, compressed keys, rotated-back heads

Plain sizes (one key/value head; not latent attention), GPT-OSS's sink on every block
(`SINK = True`, scores at `attn_weights_1`). `attention_keys` and `attention_values` are
one tensor object: an in-place edit of either edits both; an assignment to one separates
it. On a compressed block (`compressed_sparse_attention`, `heavily_compressed_attention`)
the compressor's entries follow the token keys once the prompt reaches the block's rate,
so the keys and the pattern's key axis are `seq + seq // rate` long.
`attention_head_outputs` is the interface's output with its rotary slice rotated back,
what the grouped output projection (`o_a_proj`) reads.

### Gemma-4: borrowed keys and values, per-layer head sizes

The last `num_kv_shared_layers` blocks (20 of 35 on E2B, 18 of 42 on E4B) have no
`k_proj` / `v_proj`: each attends with the keys and values of the last earlier block
of its kind (sliding or full). `attention_keys` / `attention_values` there are that
block's tensors themselves, so an in-place edit on the source block reaches every
borrower; an assignment swaps only that block's argument. Skipping a source block
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
`SINK` is `False`. GPT-OSS, DeepSeek-V4 and MiMo-V2-Flash's sliding blocks put
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
== num_heads` there, not `num_kv_heads`.

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

See [generation-and-helpers.md](generation-and-helpers.md).

## Gotchas

- `attn_implementation="eager"` or nothing: under `sdpa` every interior value is
  `Unavailable`, and `support()` says so per block before any trace.
- Reads follow the forward within one trace: q, k, v, then scores, then pattern,
  then head outputs; on Falcon without alibi the values before the queries and
  keys, with alibi queries, keys, values. Out of order raises `OutOfOrderError`
  naming the attention call. Under `generate` + `tracer.iter` it does not raise:
  the read binds the *next* step's value.
- Never overwrite `attention_scores` wholesale (it holds the mask); never read the
  pattern in two invokes of one trace.
- GPT-2 and MPT q/k/v: assign, do not edit in place.
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
