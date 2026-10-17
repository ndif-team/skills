---
name: patterns
description: Run an interpretability technique against nnterp's standard values so the recipe is written once and runs on every transformer family: logit lens, direct logit attribution, attention-pattern analysis, ablation (zero, mean, heads, skip_layers, experts), activation patching, steering, probing, cross-family sweeps and recurrent-state patching on a StandardizedTransformer. Load when an experiment touches layer_output, attention_output, mlp_output, attention_probabilities, attention_head_outputs, expert_weights, a recurrent mixer's state, next_token_probs, project_on_vocab, steer or skip_layers; when the same experiment has to run on GPT-2, a Llama, a Gemma or a Qwen3.5 / Mamba hybrid without per-architecture branches; or when a recipe from the nnsight technique skills should be ported to nnterp. Carries the gotchas that silently corrupt results on trained checkpoints (BOS target token, NaN KL, non-additive blocks, bf16 sweeps, pad tokens, read order under generate).
---

# nnterp Patterns

The recipes below are shorter than the ones in the nnsight technique skills for one
reason: the standard values do the normalization. A block's residual stream is
`layer_output` on every family, a sublayer's contribution is `attention_output` /
`mlp_output` (defined by `layers[i].input + attention_output + mlp_output ==
layer_output`), the pattern is `attention_probabilities`, the per-head view is
`attention_head_outputs`, and `project_on_vocab` is the model's own head. Nothing
here indexes `.output[0]` or names `ln_f` versus `norm`. Verified on nnterp
`103f697` (branch `encyclopedia`), nnsight 0.8 dev `3553b930`, transformers 5.17.0; the executed blocks run on
`openai-community/gpt2`, `HuggingFaceTB/SmolLM2-135M-Instruct` (30 layers, 576
hidden, 9 heads, 3 kv heads, bf16) and `trl-internal-testing/tiny-LlavaForConditionalGeneration`.

<!-- test: setup -->
```python
import torch
import torch.nn.functional as F
from nnterp import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"


def target_token(model, word):
    """The one token `word` is; raises on a word that is not one token."""
    ids = model.tokenizer(word, add_special_tokens=False).input_ids     # never encode(word)[0]: that is BOS on Llama-3 / Gemma
    if len(ids) != 1:
        raise ValueError(f"{word!r} is {len(ids)} tokens: {model.tokenizer.convert_ids_to_tokens(ids)}")
    return ids[0]


def kl(clean_logits, other_logits):
    """KL(clean || other) per row from logits, in float32; safe where a probability is exactly 0."""
    return F.kl_div(other_logits.float().log_softmax(-1), clean_logits.float().log_softmax(-1),
                    log_target=True, reduction="none").sum(-1)


paris = target_token(model, " Paris")
attn_blocks = [i for i, layer in enumerate(model.layers) if getattr(layer, "self_attn", None) is not None]
LAYER = model.num_layers // 2          # a depth, not a constant: the same line picks a layer on every family
ATTN = attn_blocks[len(attn_blocks) // 2]   # an attention block: on a hybrid, not every block has self_attn
```

`target_token` catches the two silent failures: `tokenizer.encode(" Paris")[0]` is
BOS on Llama-3 and Gemma, and a word that splits (`' Par', 'is'` on Granite;
`'▁', 'Paris'` on a Llama-2 / Mistral tokenizer, where `"Paris"` without the space
is the one token) is not one logit. `kl` works from logits: the textbook
`p * (p.log() - q.log())` is NaN wherever `next_token_probs` holds an exact zero
(Pythia). `attn_implementation="eager"` is only needed for the attention interior;
the boundary values and `project_on_vocab` work on an `sdpa` load.

## Logit lens

```python
lens = {}                                   # made outside the trace; entries saved
with model.trace(prompt):
    for i, layer in enumerate(model.layers):
        lens[i] = model.project_on_vocab(layer.layer_output)[0, -1].save()   # [vocab]
    logits = model.logits.save()

decoded = [model.tokenizer.decode(lens[i].argmax()) for i in range(model.num_layers)]
print(decoded)
assert decoded[10:] == [" Paris", " Paris"]
assert torch.equal(lens[model.num_layers - 1], logits[0, -1])   # the wiring check: the last block's lens is the model
```

```
[' the', ' the', ' the', ' the', ' the', ' the', ' East', ' Ing', ' Rome', ' London', ' Paris', ' Paris']
```

The last assertion is the check to run on any new checkpoint before reading a
curve: `project_on_vocab` on the last block *is* the final computation (softcap,
`logit_scale`, `/ logits_scaling`, DeepSeek-V4's `hc_head`). It holds on every
family, Gemma-4 and Granite included. Argmax hides a race: P(" Paris") peaks at
layer 9 (0.25) and is 0.07 at the output; the curve is in
`references/logit-lens-and-attribution.md`.

## Direct logit attribution

`model.lm_head(x)` inside a trace runs the unembedding on any `[..., hidden]`
tensor, so each contribution's logits come from the same forward and sum linearly:

```python
dla = {}
with model.trace(prompt):
    dla["base"] = model.lm_head(model.layers[0].input)[0, -1].save()
    for i, layer in enumerate(model.layers):                      # forward order: attn i, mlp i, attn i+1 ...
        dla["attn", i] = model.lm_head(layer.self_attn.attention_output)[0, -1].save()
        dla["mlp", i] = model.lm_head(layer.mlp.mlp_output)[0, -1].save()
    head_final = model.lm_head(model.layers[-1].layer_output)[0, -1].save()

assert torch.allclose(sum(dla.values()), head_final, atol=1e-3)     # linear: the attributions sum
print({i: round(float(dla["attn", i][paris]), 1) for i in range(model.num_layers)})
```

This is on the *unnormed* scale. **On Gemma-4, Doge, ZAYA and DeepSeek-V4 the
plain sum is wrong** (Gemma-4: by ~7,700%, naming block 0's attention the top
term): weight each term first, with the recipes in the `nnterp` skill's
`references/values.md`. Putting the terms on the model's scale needs the final
norm's family details (LayerNorm centering and bias, Gemma's `1 + w`, Granite's
`logits_scaling`, an `lm_head` bias); `references/logit-lens-and-attribution.md`
lists them, plus the per-head split.

## Attention patterns

```python
with model.trace(prompt):
    pattern = model.layers[ATTN].self_attn.attention_probabilities.save()   # [batch, heads, query, key]

assert pattern.shape[1] == model.layers[ATTN].self_attn.num_heads and torch.equal(pattern.tril(), pattern)
assert torch.allclose(pattern.sum(-1), torch.ones_like(pattern.sum(-1)), atol=1e-4)
previous = pattern.diagonal(offset=-1, dim1=-2, dim2=-1).mean(-1)[0]   # mass on key i-1, per head
first = pattern[0, :, 1:, 0].mean(-1)                                  # mass on key 0: sink heads
```

Index blocks through `attn_blocks` (decided outside the trace): on a Qwen3.5 or
Mamba hybrid most blocks have no `self_attn`. Never read the pattern in two invokes
of one trace (nnsight raises `TypeError: 'NoneType' object is not subscriptable`):
batch the prompts in one invoke. In a padded batch, pad *query* rows attend
uniformly: mask them with `attention_mask` before averaging a head metric. Edits,
the previous-token head search and the sink caveats:
`references/patching-and-attention.md`.

## Ablation

Clean and ablated as two invokes of one trace; the write touches only its own rows:

```python
with model.trace() as tracer:
    with tracer.invoke(prompt):
        clean = model.logits[:, -1].save()                           # [1, vocab]
    with tracer.invoke(prompt):
        model.layers[LAYER].self_attn.attention_output[:] = 0        # zero what block LAYER's attention adds
        ablated = model.logits[:, -1].save()

p_clean, p_ablated = clean.float().softmax(-1), ablated.float().softmax(-1)
print(f"P(Paris) clean {p_clean[0, paris]:.3f}  ablated {p_ablated[0, paris]:.3f}  KL {float(kl(clean, ablated)[0]):.3f}")
assert p_ablated[0, paris] < p_clean[0, paris]
```

```
P(Paris) clean 0.070  ablated 0.048  KL 0.030
```

Zero ablation can *raise* a target too: zeroing block 6's `mlp_output` moves
P(Paris) from 0.070 to 0.080 here. Report the KL alongside the target, and say
which ablation you ran. Mean ablation, `skip_layers`, one head via
`attention_head_outputs` or the pattern, a recurrent mixer:
`references/steering-and-ablation.md`.

### Expert ablation (mixtures of experts)

On a `Moe` (`layers[i].mlp` on the 36 MoE families), remove expert `e` by zeroing
the weights of the slots that chose it. Ran on the tiny Mixtral checkpoint nnterp
pins:

<!-- test: skip -->
```python
moe_model = StandardizedTransformer("mistralai/Mixtral-8x7B-v0.1", device="cpu", dispatch=True, dtype=torch.float32)
blocks = [i for i in range(moe_model.num_layers) if moe_model.support(layer=i).get("mlp.expert_weights", "absent") is None]
moe = moe_model.layers[blocks[0]].mlp
target = target_token(moe_model, "Paris")          # sentencepiece: "Paris" is the one token '▁Paris'

rows = []
with moe_model.trace() as tracer:
    with tracer.invoke(prompt):
        base = moe_model.logits[0, -1].float().log_softmax(-1)[target].save()
    for e in range(moe.num_experts):
        with tracer.invoke(prompt):
            moe.expert_weights[:] = moe.expert_weights.masked_fill(moe.expert_indices == e, 0)   # in place under several invokes
            rows.append(moe_model.logits[0, -1].float().log_softmax(-1)[target].save())
effects = torch.stack(rows) - base                  # [experts]: change in log p(target)
```

Load float32 for single-expert effects: on a bf16 GraniteMoE this batched sweep
correlated 0.32 with per-trace ablation and named a different top expert. Count
usage and entropy over real tokens only (pad tokens take 25-31% of the slots).
Usage, entropy, rerouting and per-family gaps: the `nnterp` skill's
`references/mixture-of-experts.md`.

## Activation patching

```python
corrupt = "The Colosseum is in the city of"

with model.trace(prompt):
    clean_resid = model.layers[LAYER].layer_output.save()                 # [1, seq, hidden]

with model.trace() as tracer:
    with tracer.invoke(corrupt):
        baseline = model.next_token_probs.save()
    with tracer.invoke(corrupt):
        model.layers[LAYER].layer_output[:, -1] = clean_resid[:, -1]     # a saved tensor: nothing to synchronize
        patched = model.next_token_probs.save()

print(f"P(Paris) corrupt {baseline[0, paris]:.4f}  patched {patched[0, paris]:.4f}")
assert patched[0, paris] > baseline[0, paris]
```

One layer at the last position is a weak probe (0.0032 to 0.0056 here); the
per-layer sweep and the layer×position map are where the result is. On SmolLM2
the map is textbook: the clean fact restores P(Paris) to 0.9 when patched at the
subject's second token through layer 22, and at the last position from layer 23
on. Sweep, map, barrier and session forms, and the Gemma-4 / Granite write
caveats: `references/patching-and-attention.md`.

## Steering

```python
with model.trace() as tracer:
    with tracer.invoke("I love this so much"):
        positive = model.layers[LAYER].layer_output[:, -1].save()
    with tracer.invoke("I hate this so much"):
        negative = model.layers[LAYER].layer_output[:, -1].save()
vector = (positive - negative)[0]
vector = vector / vector.norm()

steer_prompt = "I went to the bakery and"
with model.trace(steer_prompt):
    scale = model.layers[LAYER].layer_output[0, -1].norm().save()      # the stream's norm where you add
factor = 0.5 * float(scale)                                             # sweep the fraction, not a raw number

N = 5
with model.generate(steer_prompt, max_new_tokens=N, min_new_tokens=N, do_sample=False) as tracer:
    plain = tracer.result.save()
with model.generate(steer_prompt, max_new_tokens=N, min_new_tokens=N, do_sample=False) as tracer:
    for step in tracer.iter[:N]:                                        # the prefill and every decode step
        model.steer(LAYER, vector, factor=factor, token_positions=-1)
    steered = tracer.result.save()

print(repr(model.tokenizer.decode(plain[0])), repr(model.tokenizer.decode(steered[0])))
assert not torch.equal(plain, steered)
```

```
'I went to the bakery and bought a bag of cookies' 'I went to the bakery and I was very happy with'
```

A bare `steer` in a `generate` body fires on the prefill only. The useful band is
per model: a factor of 4 barely moves GPT-2 (stream norm ~92 mid-depth), while
Gemma-3 degenerates at half the stream norm; sweep the fraction and watch fluency.
On Gemma-4 `steer` adds after the block's `layer_scalar`, so it is not the same
as adding to a contribution. Hand-written form, `batch_index`, both poles:
`references/steering-and-ablation.md`.

## Probing

```python
from nnterp.nnsight_utils import get_token_activations

texts = [f"The movie was {w}." for w in ["wonderful", "fantastic", "excellent", "lovely",
                                         "terrible", "awful", "horrible", "boring"]]
labels = torch.tensor([1.0] * 4 + [0.0] * 4)

acts = get_token_activations(model, texts, idx=-1)          # [layers, prompts, hidden], one batched forward, CPU
assert acts.shape == (model.num_layers, len(texts), model.hidden_size)
```

`idx=-1` needs left padding, which nnsight's tokenizer sets for causal models;
pass `tokenizer_kwargs={"padding_side": "left", "pad_token": ...}` where it does
not (check `model.tokenizer.pad_token`: GPT-2's `<|endoftext|>` is not Llama's).
On DeepSeek-V4 `layer_output` is rank 4 (`Streams`): a probe written for
`[batch, seq, hidden]` runs and answers per stream. The ridge probe, the
shuffled-label control and the mass-mean direction: `references/sweeps-and-probing.md`.

## Recurrent state patching (hybrids, Mamba)

A DeltaNet or Mamba block keeps a recurrent state instead of a pattern.
`state_input` on a decode step is what the step starts from; assign it to continue
from another prompt's memory (ran on the tiny Qwen3.5 checkpoint):

<!-- test: skip -->
```python
import nnterp

nnterp.route_kernels(nnterp.families.qwen3_5_text, "torch")       # before the first trace: per-token state, readable kernels
hybrid = StandardizedTransformer("Qwen/Qwen3.5-0.8B", device="cpu", dispatch=True, attn_implementation="eager")
linear_blocks = [i for i, layer in enumerate(hybrid.layers) if getattr(layer, "linear_attn", None) is not None]
mix = hybrid.layers[linear_blocks[0]].linear_attn                      # decided outside the trace

with hybrid.trace("My favourite food is pizza with"):
    other_state = mix.state_output.save()                             # [batch, heads, key_dim, value_dim]

with hybrid.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    for step in tracer.iter[1]:
        mix.state_input = other_state                                 # decode step 1 starts from the other prompt
    for step in tracer.iter[2]:
        entering = mix.state_input.save()                             # assert on this, not on the tokens
```

Assert on the state: on Qwen3.5-0.8B the write lands and the greedy tokens do not
change. The state is not all a block remembers (a short conv window carries the
last few tokens). On Mamba decode steps read `state_output` before
`attention_head_outputs`. Per-token states (`states`, `set_state_after`):
`references/sweeps-and-probing.md` and the `nnterp` skill's
`references/recurrent-mixers.md`.

## The image pathway

On a vision-language model the image enters the text model once, at the image tokens:
`vision.image_features` (`[image_tokens, hidden]`) is what the wrapper scatters there, so
`layers[0].input[mask] == image_features` exactly, and `vision.image_token_mask`
(`[batch, seq]`) says which positions are the image. Ablate or patch the features to
change the image as the text model sees it (ran on the tiny Llava checkpoint, random
weights, so shapes and identities only):

```python
from PIL import Image

vlm = StandardizedTransformer("trl-internal-testing/tiny-LlavaForConditionalGeneration", task="image-text-to-text",
                              device="cpu", dispatch=True, attn_implementation="eager")
red, blue = Image.new("RGB", (64, 64), "red"), Image.new("RGB", (64, 64), "blue")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the square? Answer with one word."}]}]
vprompt = vlm.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)   # places the image token

with vlm.trace(vprompt, images=[red]):
    mask = vlm.vision.image_token_mask.save()          # first: it comes off the inputs
    red_features = vlm.vision.image_features.save()    # [576, hidden]
    first = vlm.layers[0].input.save()
    red_logits = vlm.logits.save()

with vlm.trace(vprompt, images=[red]):
    vlm.vision.image_features[:] = 0                   # the text model receives zeros at the image tokens
    zeroed = vlm.logits.save()

with vlm.trace(vprompt, images=[blue]):
    vlm.vision.image_features[:] = red_features        # a saved tensor: the blue run sees the red image
    patched = vlm.logits.save()

assert mask.shape[0] == 1 and int(mask.sum()) == 576 and red_features.shape == (576, vlm.hidden_size)
assert torch.equal(first[mask], red_features)          # the features are what enters the text model
assert not torch.equal(zeroed, red_logits)
assert torch.equal(patched, red_logits)                # same prompt, same features: the red run, bit for bit
```

Each text block's pattern has the image tokens as ordinary keys, so a head's mass on the
image is the pattern summed over the mask's columns:

```python
masses = []                                            # made outside the block: a name bound inside does not survive it
with vlm.trace(vprompt, images=[red]):
    image = vlm.vision.image_token_mask                # read first, before any text-block value
    for layer in vlm.layers:
        masses.append(layer.self_attn.attention_probabilities[0, :, -1, image[0]].sum(-1).save())   # [heads]

to_image = torch.stack(masses)                         # [layers, heads]: the last token's mass on the image
assert to_image.shape == (vlm.num_layers, vlm.layers[0].self_attn.num_heads)
```

On `llava-hf/llava-1.5-7b-hf`, asked the color of a red square on white (clean: `Red` at
0.990), zeroing `image_features` leaves `Red` at 0.010 with `''` on top, and the blue
image's run with the red features assigned answers `Red` at 0.990. Zeroing the image rows
of `layer_output` after block 4, 8, 16 and 24 leaves `Red` at 0.012, 0.114, 0.384 and
0.994: the text positions have read the image out by the middle of the stack. The last
token's mean mass on the image is 0.73 at block 0, under 0.1 from block 3, and 0.1 to 0.26
across blocks 10 to 24, where single heads put up to 0.95 on it. It peaked at 14.6 GB in
float16 under eager with `torch.no_grad()`.

1. **Read the mask first, `image_features` after the tower's values and before the text
   model's**; out of order raises `OutOfOrderError`. **One image-carrying invoke per
   trace**: clean and ablated runs are two traces. Several images go in one invoke as a
   list. `model.trace(prompt, images=[...])` refuses to batch two image-carrying invokes;
   chat-message inputs that embed the image do run as one batch, but there
   `image_features` is flat over the whole batch's image tokens in every invoke, and a
   write in one invoke reaches every row.
2. **A tower edit that changes nothing is at a block the wrapper does not read.** Llava's
   projector reads `vision.layers[-2].layer_output`, so zeroing `tower_output` leaves the
   logits unchanged; `model.projector.input` is what the projector receives.
3. **On Qwen3-VL `image_features` is not the only way in**: the text model re-adds the image
   after blocks 0 to 2 (`layers[k].deepstack_output`); zero those too.

Inside the tower, patching part of an image, and editing the image positions of a text
block: `references/image-pathway.md`.

## The same recipes on SmolLM2

Nothing in the trace bodies changes; only the numbers do:

```python
llama = StandardizedTransformer("HuggingFaceTB/SmolLM2-135M-Instruct", device="cpu", dispatch=True, attn_implementation="eager")
lparis = target_token(llama, " Paris")
LLAYER = llama.num_layers // 2

llens = {}
with llama.trace(prompt):
    for i, layer in enumerate(llama.layers):
        llens[i] = llama.project_on_vocab(layer.layer_output)[0, -1].save()
    llogits = llama.logits.save()
ldecoded = [llama.tokenizer.decode(llens[i].argmax()) for i in range(llama.num_layers)]
assert ldecoded[-3:] == [" Paris"] * 3 and " Paris" not in ldecoded[:24]
assert torch.equal(llens[llama.num_layers - 1], llogits[0, -1])

with llama.trace() as tracer:
    with tracer.invoke(prompt):
        lclean = llama.logits[:, -1].save()
    with tracer.invoke(prompt):
        llama.layers[LLAYER].self_attn.attention_output[:] = 0
        lablated = llama.logits[:, -1].save()
lp_clean, lp_ablated = lclean.float().softmax(-1), lablated.float().softmax(-1)
print(f"SmolLM2 P(Paris) clean {lp_clean[0, lparis]:.3f}  ablated {lp_ablated[0, lparis]:.3f}  KL {float(kl(lclean, lablated)[0]):.3f}")
assert lp_ablated[0, lparis] < lp_clean[0, lparis] and kl(lclean, lablated)[0] >= 0
```

SmolLM2 decodes punctuation for 27 layers and then " Paris" at 27, 28, 29; the
wiring check passes the same way. It loads in bf16, so `next_token_probs` is a bf16
softmax (take `logits.float().softmax(-1)`, as above) and the linear DLA sum is
off by a couple of percent of the logit range; load `dtype=torch.float32` for the
exact checks.

## Gotchas that bite in recipes

- **Target token**: `model.tokenizer(word, add_special_tokens=False).input_ids`, one
  element, checked (`target_token` above). A recipe on the BOS token runs and prints 0.
- **KL from logits in float32** (`kl` above); a bf16 KL between nearly equal rows can
  come out slightly negative, which the float32 form avoids.
- **Nothing bound inside a trace survives it.** Make containers outside (`lens = {}`)
  and `.save()` each entry.
- **A write in one invoke that consumes another invoke's value needs
  `tracer.barrier`**; two traces with a `.save()` between them, or a
  `model.session()`, need nothing.
- **`skip_layers` goes in a trace of its own, or in every invoke**: inside one
  invoke of several it raises `ValueError: A batched .skip() has to cover every row`.
- **Decide blocks outside the trace with `is not None`**: `getattr(a, ...) or
  getattr(b, ...)` over envoys raises `TypeError` (an envoy's truth is its module's
  `__len__`), and `hasattr(envoy, value)` raises instead of answering. Guard on
  `model.support()`.
- **Reads follow the forward within one invoke**: one loop over the layers is
  ordered, two comprehensions over them are not; queries/keys/values before the
  pattern, the pattern before `attention_head_outputs`, both before
  `attention_output`; `model.logits` after the loop. Out of order raises
  `OutOfOrderError`, except under `generate` + `tracer.iter`, where the read binds
  the *next* step and per-step lists come back shifted by one.
- **Overwriting `attention_scores` lifts the causal mask**; add to them instead.
- **Equal prompt lengths for a layer×position map**: assert
  `len(tokenizer(clean).input_ids) == len(tokenizer(corrupt).input_ids)`.
- **Pick layers as depths** (`model.num_layers // 2`), never as constants, and
  attention layers from `attn_blocks`.
- **On a vision-language model, read `vision.image_token_mask` first** (under `generate` on
  transformers 5.18 and later, after the tower's values) and keep one
  image-carrying invoke per trace; clean and ablated runs are two traces.
- **Compare against a baseline in the same trace**: the clean row of a batched trace
  is not bit-equal to the prompt run alone (5e-7 apart on GPT-2), and in bf16 an
  edit in one invoke can move another invoke's logits by more than a small effect.

## References

- `references/logit-lens-and-attribution.md`: the lens, top-k grid, target curve,
  softcap; direct logit attribution by block, sublayer and head; the final-norm and
  non-additive caveats.
- `references/steering-and-ablation.md`: the steering vector, `steer` and `+=`,
  under `generate`; zero/mean ablation, `skip_layers`, heads, a recurrent mixer.
- `references/patching-and-attention.md`: saved, barrier and session patching,
  sweeps, the layer×position map; pattern metrics and edits.
- `references/sweeps-and-probing.md`: the cross-family loop, ridge probing with
  controls, recurrent state patching.
- `references/image-pathway.md`: on a vision-language model, ablating the image at
  `image_features` and inside the tower, patching one image into another's run, attention
  onto the image, editing the image positions of a text block.
- nnterp docs: `docs/patterns/index.md` and one page per technique under
  `docs/patterns/`.

## Related skills

- `nnterp`: the values themselves, loading, `support()`, layouts, MoE and recurrent
  mixers; start there if a name in these recipes is unfamiliar.
- `extending`: a checkpoint nnterp does not know, or a value it gets wrong.
- `logit-lens`, `activation-patching`, `ablation`, `model-steering`,
  `attention-analysis`, `probing`: the same techniques against raw nnsight module
  paths, with the deeper treatment of each method's failure modes.
- `interp-experiment-design`: controls, metrics and prompt design before any of
  these numbers mean something.
