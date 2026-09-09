"""Shared setup snippets for code tasks.

Every code task runs in a fresh subprocess, so each needs its own model load.
gpt2 loads from the HF cache in about a second.
"""

GPT2_SETUP = """
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
"""

GPT2_EAGER_SETUP = """
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel(
    "openai-community/gpt2", dispatch=True, attn_implementation="eager"
)
"""


def as_tensor(value):
    """Unwrap a one-element tuple; some modules return tuples, some tensors."""
    if isinstance(value, tuple) and value:
        return value[0]
    return value


def has_shape(value, *, last_dim=None, ndim=None) -> bool:
    value = as_tensor(value)
    if not hasattr(value, "shape"):
        return False
    if last_dim is not None and value.shape[-1] != last_dim:
        return False
    if ndim is not None and len(value.shape) != ndim:
        return False
    return True
