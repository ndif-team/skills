"""Intermediate code tasks — batching, generation, gradients, source tracing."""

import torch

from ..registry import Difficulty, Task, register_task
from ._common import GPT2_EAGER_SETUP, GPT2_SETUP, as_tensor, has_shape


def _register(id, name, prompt, verify, description, tags, reference, setup=GPT2_SETUP):
    register_task(
        Task(
            id=id,
            name=name,
            difficulty=Difficulty.INTERMEDIATE,
            prompt=prompt,
            setup_code=setup,
            verify=verify,
            expected_output_description=description,
            tags=tags,
            reference_solution=reference,
        )
    )


def verify_01(result: dict) -> bool:
    if "out_a" not in result or "out_b" not in result:
        return False
    a, b = as_tensor(result["out_a"]), as_tensor(result["out_b"])
    if not (has_shape(a, last_dim=50257) and has_shape(b, last_dim=50257)):
        return False
    return not torch.equal(a.float().cpu(), b.float().cpu())


_register(
    "intermediate_01_multiple_invokes",
    "Two prompts in one pass",
    """Write nnsight code that runs BOTH prompts "The Eiffel Tower is in" and
"The Colosseum is in" through a single forward pass, saving each prompt's final
logits as `out_a` and `out_b` respectively.""",
    verify_01,
    "out_a and out_b logits from one batched pass",
    ["batching", "invoke"],
    """
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        out_a = model.output.logits.save()
    with tracer.invoke("The Colosseum is in"):
        out_b = model.output.logits.save()
""",
)


def verify_02(result: dict) -> bool:
    if "patched" not in result or "baseline" not in result:
        return False
    patched, baseline = as_tensor(result["patched"]), as_tensor(result["baseline"])
    if not hasattr(patched, "shape") or not hasattr(baseline, "shape"):
        return False
    return not torch.equal(patched.float().cpu(), baseline.float().cpu())


_register(
    "intermediate_02_activation_patching",
    "Patch an activation between invokes",
    """Write nnsight code that, in ONE trace, does all of the following:

- runs "The Eiffel Tower is in the city of" and captures the output of
  transformer block 5
- runs "The Colosseum is in the city of" unmodified and saves its final-position
  logits as `baseline`
- runs "The Colosseum is in the city of" again, this time replacing block 5's
  output with the value captured from the first prompt, and saves its
  final-position logits as `patched`

Both prompts have the same token length.""",
    verify_02,
    "patched differs from baseline",
    ["patching", "barrier", "batching"],
    """
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in the city of"):
        donor = model.transformer.h[5].output
        barrier()
    with tracer.invoke("The Colosseum is in the city of"):
        baseline = model.output.logits[0, -1].save()
    with tracer.invoke("The Colosseum is in the city of"):
        barrier()
        model.transformer.h[5].output[:] = donor
        patched = model.output.logits[0, -1].save()
""",
)


def verify_03(result: dict) -> bool:
    ids = as_tensor(result.get("generated_ids"))
    return hasattr(ids, "shape") and ids.shape[-1] >= 5


_register(
    "intermediate_03_generation",
    "Generate tokens",
    """Write nnsight code that generates 5 new tokens from the prompt
"The Eiffel Tower is in" and saves the resulting token ids as `generated_ids`.""",
    verify_03,
    "generated_ids tensor including the new tokens",
    ["generation"],
    """
with model.generate("The Eiffel Tower is in", max_new_tokens=5) as tracer:
    generated_ids = tracer.result.save()
""",
)


def verify_04(result: dict) -> bool:
    steps = result.get("per_step")
    ids = result.get("generated_ids")
    if not isinstance(steps, list) or len(steps) != 4:
        return False
    return ids is not None and hasattr(as_tensor(ids), "shape")


_register(
    "intermediate_04_iter_generation",
    "Per-step capture plus final result",
    """Write nnsight code that generates 4 new tokens from "The Eiffel Tower is in"
and, for each generated step, appends the last-position output of the final
transformer block to a list called `per_step`.

After the loop, also save the generated token ids as `generated_ids`. Both
`per_step` (with 4 entries) and `generated_ids` must exist after the block.""",
    verify_04,
    "per_step list of 4 entries and generated_ids",
    ["generation", "iteration"],
    """
with model.generate("The Eiffel Tower is in", max_new_tokens=4) as tracer:
    per_step = nnsight.save([])
    for step in tracer.iter[:4]:
        per_step.append(model.transformer.h[-1].output[0, -1])
    generated_ids = tracer.result.save()
""",
)


def verify_05(result: dict) -> bool:
    grad = as_tensor(result.get("grad"))
    if not has_shape(grad, last_dim=768):
        return False
    return bool(torch.any(grad != 0))


_register(
    "intermediate_05_gradients",
    "Gradient of a logit w.r.t. an activation",
    """Write nnsight code that traces "The Eiffel Tower is in the city of", takes the
logit of the token " Paris" at the final position as the metric, and saves the
gradient of that metric with respect to the output of the final transformer
block into a variable called `grad`.""",
    verify_05,
    "grad tensor with hidden dim 768 and non-zero entries",
    ["gradients"],
    """
paris = model.tokenizer.encode(" Paris")[0]
with model.trace("The Eiffel Tower is in the city of"):
    hidden = model.transformer.h[-1].output
    metric = model.output.logits[0, -1, paris]
    with metric.backward():
        grad = hidden.grad.clone().save()
""",
)


def verify_06(result: dict) -> bool:
    whole = as_tensor(result.get("whole_batch"))
    return has_shape(whole) and whole.shape[0] == 3


_register(
    "intermediate_06_empty_invoke",
    "Empty invoke over the whole batch",
    """Write nnsight code with a single trace containing three invokes: one for
"First prompt", one for the two prompts ["Second one", "Third one"], and one
EMPTY invoke that saves the final logits for the entire combined batch into
`whole_batch`.

`whole_batch` must have 3 rows.""",
    verify_06,
    "whole_batch logits with batch dimension 3",
    ["batching", "invoke"],
    """
with model.trace() as tracer:
    with tracer.invoke("First prompt"):
        pass
    with tracer.invoke(["Second one", "Third one"]):
        pass
    with tracer.invoke():
        whole_batch = model.output.logits.save()
""",
)


def verify_07(result: dict) -> bool:
    ids = as_tensor(result.get("generated_ids"))
    flags = result.get("steps_modified")
    if not hasattr(ids, "shape"):
        return False
    return isinstance(flags, list) and [int(f) for f in flags] == [2]


_register(
    "intermediate_07_conditional_generation",
    "Intervene on one generation step only",
    """Write nnsight code that generates 4 tokens from "The Eiffel Tower is in" and
zeroes the last position of transformer block 6's output ONLY on generation step
index 2 (zero-based). Append the step index to a list `steps_modified` each time
the intervention actually fires, and save the generated ids as `generated_ids`.

`steps_modified` must end up equal to [2].""",
    verify_07,
    "steps_modified == [2] and generated_ids present",
    ["generation", "iteration", "conditionals"],
    """
with model.generate("The Eiffel Tower is in", max_new_tokens=4) as tracer:
    steps_modified = nnsight.save([])
    for step in tracer.iter[:4]:
        if step == 2:
            model.transformer.h[6].output[:, -1, :] = 0
            steps_modified.append(step)
    generated_ids = tracer.result.save()
""",
)


def verify_08(result: dict) -> bool:
    acts = result.get("layer_acts")
    if not isinstance(acts, list) or len(acts) != 12:
        return False
    return all(has_shape(a, last_dim=768) for a in acts)


_register(
    "intermediate_08_all_layers_one_pass",
    "Every layer's activation in one forward pass",
    """Write nnsight code that traces "Hello world" ONCE and collects the output of
every transformer block into a list called `layer_acts` (12 entries, in layer
order). Do not open more than one trace.""",
    verify_08,
    "layer_acts list of 12 tensors",
    ["trace", "save", "efficiency"],
    """
with model.trace("Hello world"):
    layer_acts = nnsight.save([])
    for block in model.transformer.h:
        layer_acts.append(block.output)
""",
)


def verify_09(result: dict) -> bool:
    out = result.get("forward_output")
    logits = getattr(out, "logits", None)
    if logits is None and isinstance(out, (tuple, list)) and out:
        logits = out[0]
    return has_shape(logits, last_dim=50257)


_register(
    "intermediate_09_tracer_result",
    "The forward pass's return value",
    """Write nnsight code that traces "Hello world" and saves the traced call's whole
return value (the model output object, not a single module's activation) into a
variable called `forward_output`.""",
    verify_09,
    "forward_output with a logits attribute",
    ["trace", "result"],
    """
with model.trace("Hello world") as tracer:
    forward_output = tracer.result.save()
""",
)


def verify_10(result: dict) -> bool:
    probs = as_tensor(result.get("attn_probs"))
    if not hasattr(probs, "shape") or len(probs.shape) != 4:
        return False
    row_sums = probs[0, 0].sum(-1).float()
    return bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2))


_register(
    "intermediate_10_source_attention_probs",
    "Attention probabilities via source tracing",
    """Write nnsight code that traces "The cat sat on the" and saves the attention
PROBABILITY matrix (the softmax output, shape [batch, heads, query, key]) of
transformer block 0 into a variable called `attn_probs`.

The attention module does not return this value, so you will need operation-level
access inside its forward. The model has already been loaded with
attn_implementation="eager".""",
    verify_10,
    "attn_probs [1, 12, seq, seq] whose rows sum to 1",
    ["source", "attention"],
    """
with model.trace("The cat sat on the"):
    _, attn_probs = model.transformer.h[0].attn.source.attention_interface_0.output
    attn_probs = attn_probs.save()
""",
    setup=GPT2_EAGER_SETUP,
)


def verify_11(result: dict) -> bool:
    scores = result.get("ablation_scores")
    if not isinstance(scores, list) or len(scores) != 12:
        return False
    values = [float(as_tensor(s)) for s in scores]
    return len({round(v, 4) for v in values}) > 1


_register(
    "intermediate_11_batched_sweep",
    "Ablation sweep in a single forward pass",
    """Write nnsight code that measures, for each of the 12 transformer blocks, the
logit of " Paris" at the final position of "The Eiffel Tower is in the city of"
when THAT block's final-position output is zeroed.

Collect the 12 values, in layer order, into a list called `ablation_scores`. Use
a single forward pass — one invoke per layer inside one trace, not 12 traces.""",
    verify_11,
    "ablation_scores list of 12 differing values",
    ["batching", "ablation", "efficiency"],
    """
paris = model.tokenizer.encode(" Paris")[0]
with model.trace() as tracer:
    ablation_scores = nnsight.save([])
    for layer in range(len(model.transformer.h)):
        with tracer.invoke("The Eiffel Tower is in the city of"):
            model.transformer.h[layer].output[:, -1, :] = 0
            ablation_scores.append(model.output.logits[0, -1, paris])
""",
)


def verify_12(result: dict) -> bool:
    picked = result.get("middle_steps")
    return isinstance(picked, list) and len(picked) == 2


_register(
    "intermediate_12_bounded_iter_slice",
    "Target a slice of generation steps",
    """Write nnsight code that generates 5 tokens from "The Eiffel Tower is in" and
captures the final block's last-position output ONLY on generation steps 1 and 2,
appending them to a list called `middle_steps` (which must have exactly 2
entries).""",
    verify_12,
    "middle_steps list with 2 entries",
    ["generation", "iteration"],
    """
with model.generate("The Eiffel Tower is in", max_new_tokens=5) as tracer:
    middle_steps = nnsight.save([])
    for step in tracer.iter[1:3]:
        middle_steps.append(model.transformer.h[-1].output[0, -1])
""",
)
