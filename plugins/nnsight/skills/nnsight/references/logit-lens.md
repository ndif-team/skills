# Logit lens

The logit lens decodes an intermediate residual stream with the model's own final
norm and unembedding, `lm_head(final_norm(block.output))`, to show at which layer
an answer becomes readable. Every layer comes from one forward pass:

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
paris = model.tokenizer.encode(" Paris", add_special_tokens=False)[0]

with model.trace(prompt):
    top = nnsight.save([])
    p_paris = nnsight.save([])
    for block in model.transformer.h:
        probs = model.lm_head(model.transformer.ln_f(block.output))[0, -1].softmax(dim=-1)
        top.append(probs.argmax())
        p_paris.append(probs[paris])

for layer in range(len(top)):
    print(f"layer {layer:2d}  {model.tokenizer.decode(top[layer])!r:9}  P(' Paris') {p_paris[layer].item():.4f}")

assert max(range(12), key=lambda layer: p_paris[layer].item()) == 9
assert p_paris[11] < p_paris[9]
```

```
layer  0  ' the'     P(' Paris') 0.0000
layer  1  ' the'     P(' Paris') 0.0000
layer  2  ' the'     P(' Paris') 0.0000
layer  3  ' the'     P(' Paris') 0.0000
layer  4  ' the'     P(' Paris') 0.0001
layer  5  ' the'     P(' Paris') 0.0004
layer  6  ' East'    P(' Paris') 0.0014
layer  7  ' Ing'     P(' Paris') 0.0028
layer  8  ' Rome'    P(' Paris') 0.0255
layer  9  ' London'  P(' Paris') 0.2484
layer 10  ' Paris'   P(' Paris') 0.1834
layer 11  ' Paris'   P(' Paris') 0.0700
```

- **Lens probabilities locate a transition. They are not confidences to compare
  across layers.** `P(' Paris')` peaks at layer 9, where `' London'` is still the
  top token, and the model's real output (layer 11) is less confident than the
  peak.
- **Report a token's probability or rank, not a layer's top token.** The top two
  logits are 0.14 apart at layer 9 and 0.05 apart at layer 10. Loaded in
  bfloat16, layers 10 and 11 read `' London'` or `' Paris'` depending on the
  device and the batch shape.
- `[0]` in place of `[0, -1]` decodes every position, for a layers x positions
  grid. Inside `for step in tracer.iter[:n]` of a `model.generate` run the same
  line decodes each generated token.
- The lens shows where an answer is readable. Whether that layer carries it is a
  causal question: [activation-patching](activation-patching.md).

## Check the wiring first

Applied to the last block, the lens is the model's own final computation, so it
must reproduce the model's logits. Run this on every new model before reading
anything off a lens:

```python
cap = getattr(model.config, "final_logit_softcapping", None)       # None on GPT-2

with model.trace(prompt):
    last = model.transformer.h[-1].output
    lens = model.lm_head(model.transformer.ln_f(last))                 # the whole (batch, seq, hidden)
    lens_last = model.lm_head(model.transformer.ln_f(last[:, -1]))     # one position
    if cap is not None:
        lens = torch.tanh(lens / cap) * cap
        lens_last = torch.tanh(lens_last / cap) * cap
    lens = lens.save()
    lens_last = lens_last.save()
    real = model.output.logits.save()

assert torch.equal(lens, real)                                     # same shape: bit-equal
assert torch.allclose(lens_last, real[:, -1], atol=1e-3)           # sliced first: equal to rounding
print(f"{(lens_last - real[:, -1]).abs().max().item():.1e}")
```

```
1.8e-04
```

Use the comparison that matches the lens you wrote. Over the whole block output
the lens is bit-equal to `model.output.logits`. A lens applied to a slice of it
(`last[:, -1]`, which skips the unembedding of every other position) runs the
same arithmetic on another shape and differs in the last digits, so `torch.equal`
fails on a correct lens: compare with `torch.allclose(..., atol=1e-3)`, on a
model loaded with `dtype=torch.float32`. In bfloat16 a sliced lens can be off by
a whole rounding step (0.03 on `meta-llama/Llama-3.2-1B` on a GPU), so there
compare the whole sequence.

A mismatch is one of three things, and none of them raises:

- **The wrong norm, or none.** `lm_head(block.output)` with no final norm decodes
  `' the'` at probability above 0.999 at every GPT-2 layer: a confident lens that
  is wrong, and that looks like a lens that [does not transfer](#when-the-curve-is-flat).
- **The wrong head or block output**, such as a block output indexed like a
  tuple.
- **The model post-processes its logits.** `google/gemma-2-2b` softcaps them
  (`config.final_logit_softcapping` is 30.0). Without the `cap` lines the check
  fails by 51.0 and the lens gives its top token probability 0.9995 where the
  model gives 0.9257. Read the config field, not the family name:
  `google/gemma-3-1b-it` has it set to `None`.

## Other families

Only the two names change. Llama, Qwen, SmolLM2 and Gemma-2 use `model.model.norm`
and `model.lm_head`. For another family find the final norm and the unembedding
in [modules-and-architectures](modules-and-architectures.md); the wiring check
confirms the pair on your checkpoint.

```python
smol = TransformersModel("HuggingFaceTB/SmolLM2-135M-Instruct", task="text-generation",
                         dtype=torch.float32, dispatch=True)
smol_paris = smol.tokenizer.encode(" Paris", add_special_tokens=False)[0]

with smol.trace(prompt):
    curve = nnsight.save([])
    for block in smol.model.layers:
        probs = smol.lm_head(smol.model.norm(block.output))[0, -1].softmax(dim=-1)
        curve.append(probs[smol_paris])

print("  ".join(f"L{layer} {curve[layer].item():.3f}" for layer in range(22, 30)))
assert curve[24] < 0.01 and curve[-1] > 0.9
```

```
L22 0.000  L23 0.000  L24 0.002  L25 0.015  L26 0.160  L27 0.750  L28 0.789  L29 0.955
```

## When the curve is flat

The lens assumes an intermediate residual is written in the basis the unembedding
reads. A curve that is flat and then jumps, or the same frequent token at every
layer, may only report where the lens starts to work. On SmolLM2 above the lens
gives `' Paris'` 0.000 at layer 23, and restoring that one last-token state into
a run whose subject was noised brings the answer back in full
([causal-tracing](causal-tracing.md)): the answer was there, and the lens could
not read it.

Run the wiring check first, because a missing final norm gives the same picture.
If the wiring is right, the remedy is a tuned lens (Belrose et al., 2023): train
one affine map `A` per layer to match the final logits and decode
`lm_head(final_norm(A(h)))`, keeping the model's final norm.
