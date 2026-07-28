"""Debugging tasks — the agent is given broken code plus the symptom.

Every bug here is one I reproduced against nnsight 0.8 while writing the
`nnsight-debugging` skill, so the symptoms are the real messages, not invented
ones. A few are *silent* failures (wrong results, no exception), which are the
ones documentation most needs to prevent.
"""

import torch

from ..registry import Difficulty, register_debug
from ._common import GPT2_SETUP, as_tensor, has_shape

# Counts real forward passes so a task can require an efficient solution.
COUNTING_SETUP = GPT2_SETUP + """
forward_calls = []
model._module.register_forward_pre_hook(lambda *a, **k: forward_calls.append(1))
"""


def _debug(id, name, difficulty, symptom, buggy, verify, reference, tags, setup=GPT2_SETUP):
    register_debug(
        id=id,
        name=name,
        difficulty=difficulty,
        symptom=symptom,
        buggy_code=buggy,
        setup_code=setup,
        verify=verify,
        tags=tags,
    )
    from ..registry import get_task

    get_task(id).reference_solution = reference


# --- missing save -----------------------------------------------------------

_debug(
    "debug_01_missing_save",
    "Value does not survive the trace",
    Difficulty.BASIC,
    "UnboundLocalError: cannot access local variable 'hidden' where it is not associated with a value",
    """
def get_hidden():
    with model.trace("Hello world"):
        hidden = model.transformer.h[-1].output
    return hidden

hidden_states = get_hidden()
""",
    lambda r: has_shape(r.get("hidden_states"), last_dim=768),
    """
def get_hidden():
    with model.trace("Hello world"):
        hidden = model.transformer.h[-1].output.save()
    return hidden

hidden_states = get_hidden()
""",
    ["save"],
)


# --- out of order -----------------------------------------------------------

_debug(
    "debug_02_out_of_order",
    "Reads in the wrong order",
    Difficulty.BASIC,
    "OutOfOrderError: 'model.transformer.h.2.output.i0' was requested but the model already ran past it",
    """
with model.trace("Hello world"):
    late = model.transformer.h[8].output.save()
    early = model.transformer.h[2].output.save()
""",
    lambda r: has_shape(r.get("late"), last_dim=768) and has_shape(r.get("early"), last_dim=768),
    """
with model.trace("Hello world"):
    early = model.transformer.h[2].output.save()
    late = model.transformer.h[8].output.save()
""",
    ["ordering"],
)


# --- tuple indexing a tensor (silent) ---------------------------------------

def _verify_shape_full(result: dict) -> bool:
    hidden = as_tensor(result.get("hidden"))
    return has_shape(hidden, ndim=3, last_dim=768) and hidden.shape[0] == 1


_debug(
    "debug_03_tuple_index_tensor",
    "Indexing a tensor as if it were a tuple",
    Difficulty.INTERMEDIATE,
    "No exception, but `hidden` comes out with shape [seq, 768] instead of "
    "[1, seq, 768], and every downstream batch index is off by one dimension.",
    """
with model.trace("Hello world"):
    hidden = model.transformer.h[5].output[0].save()
""",
    _verify_shape_full,
    """
with model.trace("Hello world"):
    hidden = model.transformer.h[5].output.save()
""",
    ["types", "silent"],
)


# --- unbounded iteration (silent) -------------------------------------------

def _verify_iter(result: dict) -> bool:
    steps = result.get("per_step")
    ids = as_tensor(result.get("generated_ids"))
    return isinstance(steps, list) and len(steps) == 3 and hasattr(ids, "shape")


_debug(
    "debug_04_unbounded_iter",
    "Code after the loop never runs",
    Difficulty.INTERMEDIATE,
    "`per_step` comes back with 3 entries as expected, but `generated_ids` is "
    "never assigned — NameError when the next line tries to use it.",
    """
with model.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
    per_step = nnsight.save([])
    for step in tracer.all():
        per_step.append(model.transformer.h[-1].output[0, -1])
    generated_ids = tracer.result.save()
""",
    _verify_iter,
    """
with model.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
    per_step = nnsight.save([])
    for step in tracer.iter[:3]:
        per_step.append(model.transformer.h[-1].output[0, -1])
    generated_ids = tracer.result.save()
""",
    ["generation", "iteration", "silent"],
)


# --- tuple item assignment --------------------------------------------------

def _verify_attn_zeroed(result: dict) -> bool:
    logits = result.get("logits")
    zeroed = result.get("check_zero")
    return has_shape(logits, last_dim=50257) and bool(zeroed)


_debug(
    "debug_05_tuple_item_assignment",
    "Assigning into a tuple output",
    Difficulty.INTERMEDIATE,
    "TypeError: 'tuple' object does not support item assignment",
    """
with model.trace("Hello world"):
    attn_out = model.transformer.h[0].attn.output[0]
    model.transformer.h[0].attn.output[0] = torch.zeros_like(attn_out)
    check_zero = nnsight.save(bool(model.transformer.h[0].attn.output[0].abs().sum() == 0))
    logits = model.output.logits.save()
""",
    _verify_attn_zeroed,
    """
with model.trace("Hello world"):
    model.transformer.h[0].attn.output[0][:] = 0
    check_zero = nnsight.save(bool(model.transformer.h[0].attn.output[0].abs().sum() == 0))
    logits = model.output.logits.save()
""",
    ["modification", "types"],
)


# --- in-place write breaks autograd -----------------------------------------

def _verify_grad_flow(result: dict) -> bool:
    losses = result.get("losses")
    if not isinstance(losses, list) or len(losses) < 2:
        return False
    return float(losses[-1]) < float(losses[0])


_debug(
    "debug_06_inplace_breaks_autograd",
    "In-place intervention breaks the backward pass",
    Difficulty.ADVANCED,
    "RuntimeError: one of the variables needed for gradient computation has been "
    "modified by an inplace operation",
    """
direction = torch.zeros(768, device=model.device, requires_grad=True)
optimizer = torch.optim.Adam([direction], lr=0.5)
paris = model.tokenizer.encode(" Paris")[0]
losses = []

for _ in range(3):
    with model.trace("The Eiffel Tower is in the city of"):
        hidden = model.transformer.h[6].output
        model.transformer.h[6].output[:] = hidden + direction
        loss = -model.output.logits[0, -1, paris]
        with loss.backward():
            pass
        tracked = nnsight.save(loss.item())
    optimizer.step()
    optimizer.zero_grad()
    losses.append(tracked)
""",
    _verify_grad_flow,
    """
direction = torch.zeros(768, device=model.device, requires_grad=True)
optimizer = torch.optim.Adam([direction], lr=0.5)
paris = model.tokenizer.encode(" Paris")[0]
losses = []

for _ in range(3):
    with model.trace("The Eiffel Tower is in the city of"):
        hidden = model.transformer.h[6].output
        model.transformer.h[6].output = hidden + direction
        loss = -model.output.logits[0, -1, paris]
        with loss.backward():
            pass
        tracked = nnsight.save(loss.item())
    optimizer.step()
    optimizer.zero_grad()
    losses.append(tracked)
""",
    ["gradients", "modification"],
)


# --- save outside the trace -------------------------------------------------

_debug(
    "debug_07_save_outside_trace",
    "Accumulator created before the trace",
    Difficulty.BASIC,
    "ValueError: save() was called outside a trace.",
    """
collected = nnsight.save([])
with model.trace("Hello world"):
    for block in model.transformer.h:
        collected.append(block.output[0, -1])
""",
    lambda r: isinstance(r.get("collected"), list) and len(r["collected"]) == 12,
    """
with model.trace("Hello world"):
    collected = nnsight.save([])
    for block in model.transformer.h:
        collected.append(block.output[0, -1])
""",
    ["save"],
)


# --- wrong module path ------------------------------------------------------

_debug(
    "debug_08_wrong_module_path",
    "Module path from a different architecture",
    Difficulty.BASIC,
    "AttributeError: 'TransformersModel' object (nor its module) has attribute 'model'",
    """
with model.trace("Hello world"):
    hidden = model.model.layers[5].output[0].save()
""",
    lambda r: has_shape(r.get("hidden"), last_dim=768),
    """
with model.trace("Hello world"):
    hidden = model.transformer.h[5].output.save()
""",
    ["architecture"],
)


# --- cross-invoke without a barrier ----------------------------------------

def _verify_patch(result: dict) -> bool:
    patched, baseline = result.get("patched"), result.get("baseline")
    if patched is None or baseline is None:
        return False
    return not torch.equal(as_tensor(patched).float().cpu(), as_tensor(baseline).float().cpu())


_debug(
    "debug_09_cross_invoke_no_barrier",
    "Value read before the other invoke produced it",
    Difficulty.ADVANCED,
    "NameError: name 'donor' is not defined",
    """
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in the city of"):
        donor = model.transformer.h[5].output
    with tracer.invoke("The Colosseum is in the city of"):
        baseline = model.output.logits[0, -1].save()
    with tracer.invoke("The Colosseum is in the city of"):
        model.transformer.h[5].output[:] = donor
        patched = model.output.logits[0, -1].save()
""",
    _verify_patch,
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
    ["batching", "barrier"],
)


# --- branching on fake tensor data ------------------------------------------

_debug(
    "debug_10_scan_branches_on_data",
    "Branching on a value under scan",
    Difficulty.ADVANCED,
    "GuardOnDataDependentSymNode: Could not guard on data-dependent expression",
    """
with meta_model.scan("Hello world"):
    hidden = meta_model.transformer.h[0].output
    if hidden.mean() > 0:
        hidden_size = nnsight.save(hidden.shape[-1])
    else:
        hidden_size = nnsight.save(0)
""",
    lambda r: int(r.get("hidden_size", 0)) == 768,
    """
with meta_model.scan("Hello world"):
    hidden = meta_model.transformer.h[0].output
    hidden_size = nnsight.save(hidden.shape[-1])
""",
    ["scan"],
    setup=GPT2_SETUP + """
meta_model = TransformersModel("openai-community/gpt2")
""",
)


# --- legacy API port --------------------------------------------------------

def _verify_legacy(result: dict) -> bool:
    hidden = as_tensor(result.get("hidden"))
    ids = as_tensor(result.get("generated_ids"))
    if not has_shape(hidden, last_dim=768):
        return False
    return hasattr(ids, "shape") and ids.shape[-1] >= 3


_debug(
    "debug_11_legacy_api_port",
    "Code written for an older nnsight",
    Difficulty.ADVANCED,
    "This worked on an older nnsight. Now it fails — `.value` does not exist on a "
    "tensor, and the generated ids never arrive.",
    """
with model.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
    hidden_proxy = model.transformer.h[-1].output[0].save()
    with tracer.all():
        pass
    generated = model.generator.output.save()

hidden = hidden_proxy.value
generated_ids = generated.value
""",
    _verify_legacy,
    """
with model.generate("The Eiffel Tower is in", max_new_tokens=3) as tracer:
    hidden = model.transformer.h[-1].output.save()
    generated_ids = tracer.result.save()
""",
    ["porting", "legacy"],
)


# --- inefficiency, not an exception -----------------------------------------

def _verify_single_pass(result: dict) -> bool:
    acts = result.get("layer_acts")
    calls = result.get("forward_calls")
    if not isinstance(acts, list) or len(acts) != 12:
        return False
    if not all(has_shape(a, last_dim=768) for a in acts):
        return False
    return isinstance(calls, list) and len(calls) <= 2


_debug(
    "debug_12_one_trace_per_layer",
    "Twelve forward passes where one would do",
    Difficulty.INTERMEDIATE,
    "The results are correct but it is 12x slower than it should be, and against "
    "a remote model it issues 12 separate requests.",
    """
layer_acts = []
for layer in range(12):
    with model.trace("Hello world"):
        layer_acts.append(model.transformer.h[layer].output.save())
""",
    _verify_single_pass,
    """
with model.trace("Hello world"):
    layer_acts = nnsight.save([])
    for block in model.transformer.h:
        layer_acts.append(block.output)
""",
    ["efficiency", "silent"],
    setup=COUNTING_SETUP,
)


# --- barrier count ----------------------------------------------------------

_debug(
    "debug_13_barrier_count",
    "Barrier created for the wrong number of blocks",
    Difficulty.ADVANCED,
    "ValueError: A barrier was never reached by every block it waits for; check "
    "the count it was created with",
    """
with model.trace() as tracer:
    barrier = tracer.barrier(3)
    with tracer.invoke("The Eiffel Tower is in the city of"):
        donor = model.transformer.wte.output
        barrier()
        donor_token = model.output.logits[0, -1].argmax().save()
    with tracer.invoke("_ _ _ _ _ _ _ _ _"):
        barrier()
        model.transformer.wte.output = donor
        receiver_token = model.output.logits[0, -1].argmax().save()
""",
    lambda r: int(as_tensor(r.get("donor_token"))) == int(as_tensor(r.get("receiver_token"))),
    """
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in the city of"):
        donor = model.transformer.wte.output
        barrier()
        donor_token = model.output.logits[0, -1].argmax().save()
    with tracer.invoke("_ _ _ _ _ _ _ _ _"):
        barrier()
        model.transformer.wte.output = donor
        receiver_token = model.output.logits[0, -1].argmax().save()
""",
    ["barrier"],
)


# --- partial skip across invokes --------------------------------------------

def _verify_two_rows(result: dict) -> bool:
    logits = as_tensor(result.get("both_logits"))
    return has_shape(logits) and logits.shape[0] == 2


_debug(
    "debug_14_partial_skip",
    "Skip applied to only one invoke",
    Difficulty.ADVANCED,
    "ValueError: A batched `.skip()` has to cover every row: skip the module in "
    "every invoke, or none",
    """
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        model.transformer.h[3].skip(model.transformer.h[3].input)
    with tracer.invoke("The Colosseum is in"):
        pass
    with tracer.invoke():
        both_logits = model.output.logits.save()
""",
    _verify_two_rows,
    """
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        model.transformer.h[3].skip(model.transformer.h[3].input)
    with tracer.invoke("The Colosseum is in"):
        model.transformer.h[3].skip(model.transformer.h[3].input)
    with tracer.invoke():
        both_logits = model.output.logits.save()
""",
    ["skip", "batching"],
)


# --- access outside interleaving --------------------------------------------

_debug(
    "debug_15_access_outside_trace",
    "Reading an activation outside the trace",
    Difficulty.BASIC,
    "ValueError: Cannot access `model.transformer.h.0.output` outside of interleaving",
    """
with model.trace("Hello world"):
    pass

hidden = model.transformer.h[0].output
""",
    lambda r: has_shape(r.get("hidden"), last_dim=768),
    """
with model.trace("Hello world"):
    hidden = model.transformer.h[0].output.save()
""",
    ["trace"],
)
