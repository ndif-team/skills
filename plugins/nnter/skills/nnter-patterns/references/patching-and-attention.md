# Activation patching and attention patterns

Patching copies a value from a clean run into a corrupt one at the same site and
position; if the corrupt run now gives the clean answer, that value was sufficient.
The sites are `layer_output` (everything up to block `i` at that position),
`attention_output` / `mlp_output` (one sublayer's addition) and
`attention_head_outputs[..., h, :]` (one head). Attention-pattern analysis reads
`attention_probabilities`, `[batch, heads, query, key]`, on any family loaded eager.
nnter pages: `docs/patterns/activation-patching.md`,
`docs/patterns/attention-patterns.md`, `docs/usage/attention-interior.md`.

<!-- test: setup -->
```python
import torch
from nnter import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager")
clean = "The Eiffel Tower is in the city of"
corrupt = "The Colosseum is in the city of"
ids = model.tokenizer(" Paris", add_special_tokens=False).input_ids     # encode(" Paris")[0] is BOS on Llama-3 / Gemma
assert len(ids) == 1
paris = ids[0]
LAYER, POS = model.num_layers // 2, -1
attn_blocks = [i for i, layer in enumerate(model.layers) if getattr(layer, "self_attn", None) is not None]
ATTN = attn_blocks[len(attn_blocks) // 2]          # every block on GPT-2; one in four on a Qwen3.5 hybrid
```

## Activation patching

### Saved clean value, two invokes

```python
with model.trace(clean):
    clean_resid = model.layers[LAYER].layer_output.save()              # [1, seq, hidden]

with model.trace() as tracer:
    with tracer.invoke(corrupt):
        baseline = model.next_token_probs.save()
    with tracer.invoke(corrupt):
        model.layers[LAYER].layer_output[:, POS] = clean_resid[:, POS]
        patched = model.next_token_probs.save()

print(f"P(Paris) corrupt {baseline[0, paris]:.4f}  patched {patched[0, paris]:.4f}")
assert patched[0, paris] > baseline[0, paris]
```

`clean_resid` is an ordinary tensor by the time the second trace runs, so nothing
has to be synchronized. This form also runs remotely.

### The barrier form: clean and corrupt in one forward

Across invokes of one trace, a write that *uses* another invoke's value needs a
barrier, because the assignment evaluates its right-hand side before it parks the
worker. Without one the name is not bound yet:

<!-- test: expect-error NameError -->
```python
with model.trace() as tracer:
    with tracer.invoke(clean):
        source = model.layers[LAYER].layer_output[:, POS]
    with tracer.invoke(corrupt):
        model.layers[LAYER].layer_output[:, POS] = source              # NameError: name 'source' is not defined
        unsynchronized = model.next_token_probs.save()
```

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke(clean):
        source = model.layers[LAYER].layer_output[:, POS]
        barrier()                                                      # source is read
    with tracer.invoke(corrupt):
        barrier()                                                      # wait for it
        model.layers[LAYER].layer_output[:, POS] = source
        patched_barrier = model.next_token_probs.save()

assert torch.allclose(patched_barrier, patched, atol=1e-5)
```

### The session form: several traces, no saves between

```python
with model.session():
    with model.trace(clean):
        source = model.layers[LAYER].layer_output[:, POS]
    with model.trace(corrupt):
        model.layers[LAYER].layer_output[:, POS] = source
        patched_session = model.next_token_probs.save()

assert torch.allclose(patched_session, patched, atol=1e-5)
```

### Per-layer sweep

```python
cache = []                                                             # outside the trace
with model.trace(clean):
    for layer in model.layers:
        cache.append(layer.layer_output.save())

sweep = []
for i in range(model.num_layers):
    with model.trace(corrupt):
        model.layers[i].layer_output[:, POS] = cache[i][:, POS]
        sweep.append(model.next_token_probs[0, paris].save())

sweep = [float(p) for p in sweep]
print([round(p, 3) for p in sweep])
assert sweep[-1] > 0.06 and sweep[0] < 0.005                          # the last block at the last position is the clean run
```

```
[0.003, 0.003, 0.003, 0.003, 0.004, 0.004, 0.006, 0.007, 0.017, 0.036, 0.052, 0.07]
```

Two ends of a residual sweep are fixed by construction: patching the last block at
the last position reproduces the clean prediction, and patching an early block at
a non-final position overwrites everything computed there so far. Read the curve
as *reach*, and localize with the map.

### Layer × position map, on SmolLM2

One trace per layer, one invoke per position. The prompts must tokenize to the same
length; the tokenizer decides, so assert it. GPT-2 small barely moves under this
prompt pair at any single cell; SmolLM2 gives the textbook picture, so the map runs
there (`bf16`, 30 layers, 9 tokens: 30 traces of 9 rows, about a second):

```python
llama = StandardizedTransformer("HuggingFaceTB/SmolLM2-135M-Instruct", device="cpu", dispatch=True, attn_implementation="eager")
lparis = llama.tokenizer(" Paris", add_special_tokens=False).input_ids[0]
n = len(llama.tokenizer(corrupt).input_ids)
assert n == len(llama.tokenizer(clean).input_ids)                      # 9 and 9

lcache = []
with llama.trace(clean):
    for layer in llama.layers:
        lcache.append(layer.layer_output.save())

grid = []
for i in range(llama.num_layers):
    row = []                                                           # outside the trace
    with llama.trace() as tracer:
        for pos in range(n):
            with tracer.invoke(corrupt):
                llama.layers[i].layer_output[:, pos] = lcache[i][:, pos]
                row.append(llama.next_token_probs[0, lparis].save())
    grid.append([float(p) for p in row])
grid = torch.tensor(grid)                                              # [layers, positions]

tokens = [llama.tokenizer.decode(t) for t in llama.tokenizer(corrupt).input_ids]
print("     " + "".join(f"{t!r:>8}" for t in tokens))
for i in (0, 10, 22, 23, 29):
    print(f"L{i:2d}  " + "".join(f"{v:8.2f}" for v in grid[i]))

assert (grid[1:23, 2] > 0.5).all() and (grid[:23, -1] < 0.05).all()   # the fact sits at the subject through block 22 ...
assert (grid[23:, -1] > 0.5).all() and (grid[23:, 2] < 0.05).all()    # ... and moves to the last position at block 23
```

```
        'The'  ' Col' 'osse'    'um'   ' is'   ' in'  ' the' ' city'   ' of'
L 0     0.00    0.00    0.34    0.00    0.00    0.00    0.00    0.00    0.00
L10     0.00    0.00    0.96    0.00    0.00    0.00    0.00    0.00    0.00
L22     0.00    0.00    0.89    0.00    0.00    0.00    0.00    0.00    0.00
L23     0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.95
L29     0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.95
```

Position 2 is the subject's second token (`iffel` in the clean prompt, `osse` in the
corrupt one). Patching the clean stream there restores P(Paris) to 0.9 at every
block through 22 and does nothing from 23 on; the last position is the mirror
image. The attention that moves the fact sits between blocks 22 and 23, which is
where to look with a head-level patch or the attention-output ablation (block 23's
`attention_output` is the one whose zeroing drops P(Paris) to 0.37 on the clean
prompt). This is the same code that produced the flat GPT-2 map: the model, not
the recipe, decided.

### Patch a contribution instead of the stream

```python
with model.trace(clean):
    clean_attn = model.layers[LAYER].self_attn.attention_output.save()
    clean_mlp = model.layers[LAYER].mlp.mlp_output.save()

with model.trace(corrupt):
    model.layers[LAYER].self_attn.attention_output[:, POS] = clean_attn[:, POS]
    patched_attn = model.next_token_probs.save()

with model.trace(corrupt):
    model.layers[LAYER].mlp.mlp_output[:, POS] = clean_mlp[:, POS]
    patched_mlp = model.next_token_probs.save()

print(f"attn {patched_attn[0, paris]:.4f}  mlp {patched_mlp[0, paris]:.4f}")
```

Patching `layer_output` at block `i` replaces everything up to and including `i` at
that position; a contribution replaces one sublayer's addition. On a family whose
block returns a tuple (GPT-J, BLOOM, MPT, Falcon) `layer_output` is still the
tensor and an assignment puts it back in the tuple. One head is
`attention_head_outputs[..., HEAD, :]` patched the same way, under eager. Noising
swaps the roles; the code is symmetric.

### Interpretation

- Keep the unpatched baseline in the same trace: batch effects shift probabilities
  slightly.
- Position is the question. The subject's position localizes the *source*; the
  last position tests whether the *final prediction* is sensitive.
- Use a logit difference (Paris minus Rome) when the corrupt run is near-degenerate;
  a tiny shift can flip an argmax.
- Effects are a few percent on one pair; average over pairs.
- **Gemma-4**: an in-place patch of a KV-sharing source block's keys or values
  spreads to every block that borrows them; assign to keep it local.
- **Granite line in bf16**: a write at one position of a contribution shifts every
  position slightly (the scaled copy is divided back); patch in float32.
- **Gemma-4, Doge, ZAYA**: a contribution patched at block `i` reaches the last
  stream scaled by the later blocks' scalars or gates (the `nnter` skill's
  `references/values.md`); patching `layer_output` is unaffected.

## Attention patterns

### Canonical read and the anchors

```python
prompt = "The cat sat on the mat because the cat"
with model.trace(prompt):
    pattern = model.layers[ATTN].self_attn.attention_probabilities.save()   # [batch, heads, query, key]

assert pattern.shape[1] == model.layers[ATTN].self_attn.num_heads
assert torch.equal(pattern.tril(), pattern)                                    # causal
assert torch.allclose(pattern.sum(-1), torch.ones_like(pattern.sum(-1)), atol=1e-4)   # rows sum to one (sink caveat below)
tokens = [model.tokenizer.decode(t) for t in model.tokenizer(prompt).input_ids]  # labels for both axes
```

The pattern is read after the softmax *and* the dropout: in the model's dtype and
with a sink's column already dropped, so it is the tensor the values are mixed with.

### Eager is required, and `hasattr` is not the guard

<!-- test: expect-error Unavailable -->
```python
sdpa = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True)   # the checkpoint's default is sdpa
print(sdpa.support(layer=0)["self_attn.attention_probabilities"])
hasattr(sdpa.layers[0].self_attn, "attention_probabilities")                  # raises nnter.Unavailable, not False
```

```
read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'
```

Python's `hasattr` treats only `AttributeError` as absence. On an eager model the
same `hasattr` outside a trace raises `SourceNotAvailable` (`recursive .source is
only available inside a trace`). Guard on `model.support()`; `None` means available.

### Per-head metrics

```python
entropy = -(pattern * (pattern + 1e-12).log()).sum(-1).mean(-1)[0]     # [heads]; low: sharp heads (one unpadded prompt)
previous = pattern.diagonal(offset=-1, dim1=-2, dim2=-1).mean(-1)[0]    # mass on key i-1
first = pattern[0, :, 1:, 0].mean(-1)                                   # mass on key 0 from every later query
print([round(float(x), 2) for x in entropy], [round(float(x), 2) for x in previous], [round(float(x), 2) for x in first])
```

A head with `previous > 0.5` is a previous-token head; one with `first > 0.5` parks
its mass on the first token and is usually not engaged on this prompt. An induction
head attends from `i` to the token after the previous occurrence of the token at
`i`; build that target from the token ids and take the same mean.

### Every block in one trace: find the previous-token heads

```python
prev = {}
with model.trace(clean):
    for i in attn_blocks:                                                # block order, attention blocks only
        p = model.layers[i].self_attn.attention_probabilities
        prev[i] = p.diagonal(offset=-1, dim1=-2, dim2=-1).mean(-1)[0].save()

best = {i: (int(prev[i].argmax()), round(float(prev[i].max()), 2)) for i in attn_blocks}
print(best)
assert best[4] == (11, 1.0)                                              # GPT-2 L4H11: all of its mass on the previous token
```

`attn_blocks` is decided outside the trace; on a non-hybrid it is every block and the
code is unchanged, on a Qwen3.5 or Jamba hybrid it skips the recurrent blocks. The
same search finds induction heads unmodified on GPT-2 (5.1, 5.5, 6.9, 7.2, 7.10) and
on six other families' trained checkpoints. All the heads of all the prompts go in
**one invoke**: a pattern read in two invokes of one trace raises
`TypeError: 'NoneType' object is not subscriptable` (an nnsight bug).

### Editing the pattern

The value is the tensor the values are mixed with, so a write reaches the model:

```python
HEAD, KEY = 0, 0
with model.trace(clean):
    reference = model.logits.save()

with model.trace(clean):                                                 # zero one head: it reads nothing
    model.layers[LAYER].self_attn.attention_probabilities[:, HEAD] = 0
    zeroed = model.logits.save()

with model.trace(clean):                                                 # a uniform causal pattern, assigned
    p = model.layers[LAYER].self_attn.attention_probabilities
    uniform = torch.ones_like(p).tril()
    model.layers[LAYER].self_attn.attention_probabilities = uniform / uniform.sum(-1, keepdim=True)
    uniformed = model.logits.save()

with model.trace(clean):                                                 # knock out one key, renormalize
    p = model.layers[LAYER].self_attn.attention_probabilities
    p[..., KEY] = 0
    p /= p.sum(-1, keepdim=True).clamp_min(1e-12)
    knocked = model.logits.save()

assert not torch.equal(zeroed, reference) and not torch.equal(uniformed, reference) and not torch.equal(knocked, reference)
```

Leaving the knocked-out mass removed instead of renormalizing is a different
question (how much did that key carry, versus what do the other keys say). Zeroing
a head's pattern and zeroing its slice of `attention_head_outputs` give identical
logits (`references/steering-and-ablation.md`). `attention_scores` is the masked,
scaled softmax input in the same layout; add a constant to one key column there to
shift mass without breaking normalization. Never overwrite it wholesale
(`scores[:] = 0`): the mask is part of the tensor, and the queries then attend to
the future.

### Caveats

- **Sink.** On GPT-OSS, DeepSeek-V4 and MiMo-V2-Flash's sliding blocks a learned
  sink logit joins the softmax as an extra key column and is dropped afterwards, so
  rows sum to *less* than one (`Attention.SINK`); the row-sum anchor and the entropy
  have to account for it. On Granite-SWA the sink instead scales each head's output
  after the pattern (rows sum to one): forcing the pattern does not control the
  output there.
- **Order.** Read the pattern before `model.logits`; read `attention_keys` /
  `attention_queries` / `attention_values` before the pattern and
  `attention_head_outputs` after it. On Falcon read the values before the queries
  (without alibi; queries, keys, values in that order with it).
- **Batches are left-padded**: a padded row's pattern has zero columns at its pad
  positions, its real tokens start later, and its pad *query* rows attend uniformly
  over all keys; mask query rows with `attention_mask` before a per-row metric.
- **Size.** A pattern is `heads * seq * seq` per block; index inside the trace and
  wrap read-only capture in `torch.no_grad()`.
- **GPT-2 with `reorder_and_upcast_attn`**: `support()` reports the pattern
  unavailable with the reason. GPT-J, BLOOM and MPT map it onto their own softmax
  and need no eager flag; Falcon maps it onto whichever softmax `config.alibi`
  picks and needs eager.
