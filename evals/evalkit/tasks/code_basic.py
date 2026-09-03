"""Basic code tasks — fundamentals of tracing, saving, and simple interventions.

Ported from nnsight's tests/agent-evals (dev) to the 0.8 API: TransformersModel
instead of LanguageModel, no `.value`, and verifiers that tolerate either a
tensor or a tuple where the model's return type is version-dependent.
"""

import torch

from ..registry import Difficulty, Task, register_task
from ._common import GPT2_SETUP, as_tensor, has_shape


def _register(id, name, prompt, verify, description, tags, setup=GPT2_SETUP):
    register_task(
        Task(
            id=id,
            name=name,
            difficulty=Difficulty.BASIC,
            prompt=prompt,
            setup_code=setup,
            verify=verify,
            expected_output_description=description,
            tags=tags,
        )
    )


# --- 01: trace and save ----------------------------------------------------

def verify_01(result: dict) -> bool:
    if "hidden_states" not in result:
        return False
    hidden = as_tensor(result["hidden_states"])
    return has_shape(hidden, last_dim=768, ndim=3) and hidden.shape[0] == 1


_register(
    "basic_01_trace_and_save",
    "Trace and save",
    """Write nnsight code that traces the model on the input "Hello world" and saves
the output of the final transformer block (model.transformer.h[-1]) into a
variable called `hidden_states`.

After the trace block exits, `hidden_states` must hold the actual tensor.""",
    verify_01,
    "hidden_states tensor of shape [1, seq, 768]",
    ["trace", "save"],
)


# --- 02: logits and prediction ---------------------------------------------

def verify_02(result: dict) -> bool:
    if "logits" not in result or "predicted_token" not in result:
        return False
    logits = as_tensor(result["logits"])
    if not has_shape(logits, last_dim=50257, ndim=3):
        return False
    token = result["predicted_token"]
    if hasattr(token, "shape") and token.numel() != 1:
        return False
    return True


_register(
    "basic_02_logits_and_prediction",
    "Logits and predicted token",
    """Write nnsight code that traces the model on "The capital of France is", saves
the language-model head logits into `logits`, and stores the argmax token id of
the final position into `predicted_token`.""",
    verify_02,
    "logits [1, seq, 50257] and a scalar predicted_token",
    ["trace", "logits"],
)


# --- 03: zero activations --------------------------------------------------

def verify_03(result: dict) -> bool:
    if "zeroed_output" not in result or "logits" not in result:
        return False
    zeroed = as_tensor(result["zeroed_output"])
    if not hasattr(zeroed, "shape"):
        return False
    return bool(torch.all(zeroed == 0))


_register(
    "basic_03_zero_activations",
    "Zero out activations",
    """Write nnsight code that traces the model on "Hello", zeroes the output of the
first transformer block (model.transformer.h[0]) using in-place slice
assignment, saves that zeroed output as `zeroed_output`, and saves the model's
final logits as `logits`.""",
    verify_03,
    "zeroed_output is all zeros",
    ["trace", "intervention", "in-place"],
)


# --- 04: access input ------------------------------------------------------

def verify_04(result: dict) -> bool:
    return "layer_input" in result and has_shape(result["layer_input"], last_dim=768)


_register(
    "basic_04_access_input",
    "Access a module's input",
    """Write nnsight code that traces the model on "Machine learning" and saves the
INPUT of transformer block 5 into a variable called `layer_input`.""",
    verify_04,
    "layer_input tensor with hidden dim 768",
    ["trace", "input"],
)


# --- 05: clone before modify -----------------------------------------------

def verify_05(result: dict) -> bool:
    if "before" not in result or "after" not in result:
        return False
    before = as_tensor(result["before"])
    after = as_tensor(result["after"])
    if not hasattr(before, "shape") or not hasattr(after, "shape"):
        return False
    return bool(not torch.all(before == 0) and torch.all(after == 0))


_register(
    "basic_05_clone_before_modify",
    "Clone before modifying",
    """Write nnsight code that traces the model on "Test", captures a copy of
transformer block 0's output as `before`, then zeroes that output in place and
saves the modified value as `after`.

After the trace, `before` must still hold the original (non-zero) values and
`after` must be all zeros.""",
    verify_05,
    "before is non-zero, after is all zeros",
    ["trace", "clone", "intervention"],
)


# --- 06: cache with an explicit module list --------------------------------

def verify_06(result: dict) -> bool:
    if "cached_h2" not in result or "cached_h7" not in result:
        return False
    return all(has_shape(result[name], last_dim=768) for name in ("cached_h2", "cached_h7"))


_register(
    "basic_06_cache_explicit_modules",
    "Cache specific modules",
    """Write nnsight code that traces the model on "Caching test" and uses
tracer.cache(...) restricted to exactly two modules: transformer block 2 and
transformer block 7.

After the trace, read the cached output for block 2 into `cached_h2` and the
cached output for block 7 into `cached_h7`.""",
    verify_06,
    "cached_h2 and cached_h7 tensors with hidden dim 768",
    ["cache"],
)


# --- 07: modify .input by assignment ---------------------------------------

def verify_07(result: dict) -> bool:
    return "layer3_output" in result and has_shape(result["layer3_output"], last_dim=768)


_register(
    "basic_07_modify_input",
    "Replace a module's input",
    """Write nnsight code that traces the model on "Hello world" and replaces the
INPUT of transformer block 3 with an all-zeros tensor of the same shape, dtype
and device, using direct assignment rather than slice assignment. Save the
resulting block 3 output as `layer3_output`.""",
    verify_07,
    "layer3_output tensor with hidden dim 768",
    ["trace", "input", "intervention"],
)


# --- reference solutions ----------------------------------------------------
# Kept together at the end so the task definitions above stay readable. audit.py
# runs each of these to prove the task is solvable and the verifier accepts it.

from ..registry import get_task  # noqa: E402

_REFERENCES = {
    "basic_01_trace_and_save": """
with model.trace("Hello world"):
    hidden_states = model.transformer.h[-1].output.save()
""",
    "basic_02_logits_and_prediction": """
with model.trace("The capital of France is"):
    logits = model.output.logits.save()
    predicted_token = model.output.logits[0, -1].argmax(dim=-1).save()
""",
    "basic_03_zero_activations": """
with model.trace("Hello"):
    model.transformer.h[0].output[:] = 0
    zeroed_output = model.transformer.h[0].output.save()
    logits = model.output.logits.save()
""",
    "basic_04_access_input": """
with model.trace("Machine learning"):
    layer_input = model.transformer.h[5].input.save()
""",
    "basic_05_clone_before_modify": """
with model.trace("Test"):
    before = model.transformer.h[0].output.clone().save()
    model.transformer.h[0].output[:] = 0
    after = model.transformer.h[0].output.save()
""",
    "basic_06_cache_explicit_modules": """
with model.trace("Caching test") as tracer:
    cache = tracer.cache(modules=[model.transformer.h[2], model.transformer.h[7]])

cached_h2 = cache["model.transformer.h.2"].output
cached_h7 = cache["model.transformer.h.7"].output
""",
    "basic_07_modify_input": """
with model.trace("Hello world"):
    original = model.transformer.h[3].input
    model.transformer.h[3].input = torch.zeros_like(original)
    layer3_output = model.transformer.h[3].output.save()
""",
}

for _task_id, _solution in _REFERENCES.items():
    get_task(_task_id).reference_solution = _solution
