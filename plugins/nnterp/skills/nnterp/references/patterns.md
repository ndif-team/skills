# nnterp Patterns

The recipes below are shorter than the ones in the `nnsight` skill's technique guides because the
standard values ([SKILL.md](../SKILL.md)) do the normalization: nothing here indexes
`.output[0]` or names `ln_f` versus `norm`, so each trace body runs on every family. The
executed blocks run on `openai-community/gpt2`, `HuggingFaceTB/SmolLM2-135M-Instruct`
(30 layers, 576 hidden, 9 heads, 3 kv heads, bf16) and
`trl-internal-testing/tiny-LlavaForConditionalGeneration`.

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

`target_token` refuses a word that is not one token (never `encode(word)[0]`, the BOS
token on Llama-3 and Gemma); `kl` works from float32 logits, safe where a probability is
exactly zero. `attn_implementation="eager"` is only needed for the attention interior.

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
curve: `project_on_vocab` on the last block *is* the final computation, any softcap or
scale included, on every family. Argmax hides a race: P(" Paris") peaks at
layer 9 (0.25) and is 0.07 at the output; the curve is in
[patterns-logit-lens-and-attribution.md](patterns-logit-lens-and-attribution.md).

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

This is on the *unnormed* scale. **On Gemma-4, Doge, ZAYA and DeepSeek-V4 the plain sum is
wrong**: weight each term first (the [values.md](values.md), "Blocks that are
not a plain sum"). The terms on the model's scale, and the per-head split:
[patterns-logit-lens-and-attribution.md](patterns-logit-lens-and-attribution.md).

## Attention patterns

```python
with model.trace(prompt):
    pattern = model.layers[ATTN].self_attn.attention_probabilities.save()   # [batch, heads, query, key]

assert pattern.shape[1] == model.layers[ATTN].self_attn.num_heads and torch.equal(pattern.tril(), pattern)
assert torch.allclose(pattern.sum(-1), torch.ones_like(pattern.sum(-1)), atol=1e-4)
previous = pattern.diagonal(offset=-1, dim1=-2, dim2=-1).mean(-1)[0]   # mass on key i-1, per head
first = pattern[0, :, 1:, 0].mean(-1)                                  # mass on key 0: sink heads
```

Index blocks through `attn_blocks` (setup): a hybrid's recurrent blocks have no
`self_attn`. In a padded batch, pad *query* rows attend uniformly: mask them with
`attention_mask` before averaging a head metric. Edits, the previous-token head search and
the sink caveats: [patterns-patching-and-attention.md](patterns-patching-and-attention.md).

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
`attention_head_outputs`, `attention_premix` or the pattern, MLP neurons via
`mlp_neurons`, a recurrent mixer: [patterns-steering-and-ablation.md](patterns-steering-and-ablation.md).

### Expert ablation (mixtures of experts)

On a `Moe` (`layers[i].mlp` on a mixture-of-experts family), remove expert `e` by zeroing
the weights of the slots that chose it. Ran on the tiny Mixtral checkpoint nnterp
pins:

```python
moe_model = StandardizedTransformer("hf-internal-testing/tiny-random-MixtralForCausalLM", device="cpu", dispatch=True, dtype=torch.float32)
blocks = [i for i in range(moe_model.num_layers) if moe_model.support(layer=i).get("mlp.expert_weights", "absent") is None]
moe = moe_model.layers[blocks[0]].mlp
target = target_token(moe_model, "Paris")          # sentencepiece: "Paris" is the one token '▁Paris'

rows = []
with moe_model.trace() as tracer:
    with tracer.invoke(prompt):
        base = moe_model.logits[0, -1].float().log_softmax(-1)[target].save()
    for e in range(moe.num_experts):
        with tracer.invoke(prompt):
            idx = moe.expert_indices                                                      # first: JetMoE computes the indices before the weights
            moe.expert_weights[:] = moe.expert_weights.masked_fill(idx == e, 0)           # this invoke's rows only
            rows.append(moe_model.logits[0, -1].float().log_softmax(-1)[target].save())
effects = torch.stack(rows) - base                  # [experts]: change in log p(target)
```

Load float32 for single-expert effects: on a bf16 GraniteMoE this batched sweep
correlated 0.32 with per-trace ablation and named a different top expert. Usage, entropy
over real tokens, rerouting and per-family gaps: the [mixture-of-experts.md](mixture-of-experts.md).

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
per-layer sweep and the layer×position map are where the result is. Sweep, map, barrier
and session forms, and the per-family caveats: [patterns-patching-and-attention.md](patterns-patching-and-attention.md).

A query or key patched from another prompt or position goes in before the rotary, at
`self_attn.attention_queries_prerope` / `attention_keys_prerope`: the [attention-interior.md](attention-interior.md), "Before the rotary embedding".

### Head-selective K/V patching

To make chosen query heads read another run's keys and values over a span, without the
rest of their group, the other rows or the KV cache seeing the edit, do not write
`attention_keys` / `attention_values`: every query head of the group and every row reads
them, and under `generate` an in-place edit stays in the cache. Clone the keys and values,
replace the span, call `self_attn.interface` (the block's own attention function) on the
copies, and write only the chosen heads' last row of `attention_head_outputs`. Eager only;
`support()["self_attn.interface"]` says where it is unavailable. The recipe and its
`generate` form: [patterns-patching-and-attention.md](patterns-patching-and-attention.md), "Head-selective K/V patching".

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
Hand-written form, `batch_index`, both poles: [patterns-steering-and-ablation.md](patterns-steering-and-ablation.md).

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
The ridge probe, the shuffled-label control and the mass-mean direction:
[patterns-sweeps-and-probing.md](patterns-sweeps-and-probing.md).

## Recurrent state patching (hybrids, Mamba)

A DeltaNet or Mamba block keeps a recurrent state instead of a pattern.
`state_input` on a decode step is what the step starts from; assign it to continue
from another prompt's memory (ran on Qwen3.5-0.8B):

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
change. Per-token states (`states`, `set_state_after`) and the decode-step read order:
[patterns-sweeps-and-probing.md](patterns-sweeps-and-probing.md) and the [recurrent-mixers.md](recurrent-mixers.md).

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

1. **Read the mask first, `image_features` after the tower's values and before the text
   model's. One image-carrying invoke per trace**: clean and ablated runs are two traces.
   Chat-message inputs that embed the image do batch, but there `image_features` is flat
   over the whole batch's image tokens, and a write in one invoke reaches every row.
2. **A tower edit that changes nothing is at a block the wrapper does not read.** Llava 1.5's
   projector reads `vision.layers[-2].layer_output`, so zeroing `tower_output` leaves the
   logits unchanged; `model.projector.input` is what the projector receives.
3. **On Qwen3-VL `image_features` is not the only way in**: the text model re-adds the image
   after blocks 0 to 2 (`layers[k].deepstack_output`); zero those too.

Inside the tower, patching part of an image, and editing the image positions of a text
block: [patterns-image-pathway.md](patterns-image-pathway.md).

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
wiring check passes the same way. It loads in bf16, where the linear DLA sum is off by a
couple of percent of the logit range; load `dtype=torch.float32` for the exact checks.

## Gotchas that bite in recipes

[SKILL.md](../SKILL.md)'s traps (the target token, `[0]`, blocks that are not a plain sum, late
reads under `tracer.iter`, overwriting the scores, bf16 metrics) all apply. Five more are
about recipes:

- **A write in one invoke that consumes another invoke's value needs
  `tracer.barrier`**; two traces with a `.save()` between them, or a
  `model.session()`, need nothing.
- **One loop over the layers is ordered, two comprehensions over them are not**: one over
  every attention then one over every MLP raises `OutOfOrderError`. To read, patch or
  ablate several sites in one invoke, sort them by `model.rank(name, layer)` first.
- **Equal prompt lengths for a layer×position map**: assert
  `len(tokenizer(clean).input_ids) == len(tokenizer(corrupt).input_ids)`.
- **Pick layers as depths** (`model.num_layers // 2`), never as constants, and
  attention layers from `attn_blocks`.
- **Compare against a baseline in the same trace**: the clean row of a batched trace
  is not bit-equal to the prompt run alone (5e-7 apart on GPT-2), and in bf16 an
  edit in one invoke can move another invoke's logits by more than a small effect.

## References

- [patterns-logit-lens-and-attribution.md](patterns-logit-lens-and-attribution.md): the lens, top-k grid, target curve,
  softcap; direct logit attribution by block, sublayer and head; the final-norm and
  non-additive caveats.
- [patterns-steering-and-ablation.md](patterns-steering-and-ablation.md): the steering vector, `steer` and `+=`,
  under `generate`; zero/mean ablation, `skip_layers`, heads, a recurrent mixer.
- [patterns-patching-and-attention.md](patterns-patching-and-attention.md): saved, barrier and session patching,
  sweeps, the layer×position map; head-selective K/V patching through
  `self_attn.interface`; pattern metrics and edits.
- [patterns-sweeps-and-probing.md](patterns-sweeps-and-probing.md): the cross-family loop, ridge probing with
  controls, recurrent state patching.
- [patterns-image-pathway.md](patterns-image-pathway.md): on a vision-language model, ablating the image at
  `image_features` and inside the tower, patching one image into another's run, attention
  onto the image, editing the image positions of a text block.
- nnterp docs: `docs/patterns/index.md` and one page per technique under
  `docs/patterns/`, in https://github.com/ndif-team/nnterp; the documentation site serves
  every page as Markdown, listed at https://ndif-team.github.io/nnterp/llms.txt.

## Related guides

- [nnterp](../SKILL.md): the values themselves, loading, `support()`, layouts, MoE and recurrent
  mixers; start there if a name in these recipes is unfamiliar.
- [extending](extending.md): a checkpoint nnterp does not know, or a value it gets wrong.
- `logit-lens`, `activation-patching`, `ablation`, `model-steering`,
  `attention-analysis`, `probing`: the same techniques against raw nnsight module
  paths, with the deeper treatment of each method's failure modes.
- `interp-experiment-design`: controls, metrics and prompt design before any of
  these numbers mean something.
