---
name: nnsight-basics
description: "Core nnsight patterns for neural network interpretability — load models, trace forward passes, hook into transformer layers, save and modify activations, and access gradients. Use when setting up nnsight models, extracting hidden states, patching activations, performing mechanistic interpretability experiments, or making interventions on transformer internals."
---

# NNsight Basics

## Loading Models

**For language models (recommended):**

```python
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

**For any PyTorch model:**

```python
from nnsight import NNsight

model = NNsight(your_pytorch_model)
```

## The Tracing Context

```python
with model.trace("The Eiffel Tower is in"):
    # Access any layer's output
    hidden_states = model.transformer.h[5].output[0].save()

    # Access final logits
    logits = model.lm_head.output.save()

# Values available after tracing
print(hidden_states.value.shape)
```

## Accessing Module Values

| Property | Returns |
| -------- | ------- |
| `.output` | Module's return value |
| `.input` | First positional argument |
| `.inputs` | All arguments (tuple + kwargs dict) |

```python
with model.trace(prompt):
    layer_output = model.transformer.h[0].output[0].save()
    layer_input = model.transformer.h[0].input[0].save()
    attn_output = model.transformer.h[0].attn.output[0].save()
```

## Saving Values

```python
output = model.transformer.h[0].output[0].save()
```

Access the actual tensor via `.value` after exiting the trace:

```python
print(output.value.shape)
```

## Basic Interventions

**Zero out activations:**

```python
with model.trace(prompt):
    model.transformer.h[5].output[0][:, :, :] = 0
    output = model.lm_head.output.save()
```

**Add noise:**

```python
with model.trace(prompt):
    noise = 0.1 * torch.randn_like(model.transformer.h[5].output[0])
    model.transformer.h[5].output[0] = model.transformer.h[5].output[0] + noise
```

**Replace with custom values:**

```python
with model.trace(prompt):
    model.transformer.h[5].output[0][:, -1, :] = my_custom_vector
```

## Batched Processing with Invokers

Process multiple inputs in one trace using `tracer.invoke()`:

```python
with model.trace() as tracer:
    with tracer.invoke("First prompt"):
        first_output = model.lm_head.output.save()

    with tracer.invoke("Second prompt"):
        second_output = model.lm_head.output.save()

# Access results
print(first_output.value.shape, second_output.value.shape)
```

## Cross-Prompt Interventions (Barriers)

To use values from one invoke in another invoke, use `barrier()` for synchronization:

```python
with model.trace() as tracer:
    # Create barrier for 2 participating invokes
    barrier = tracer.barrier(2)

    with tracer.invoke("The Eiffel Tower is in"):
        embeddings = model.transformer.wte.output
        barrier()  # Signal: value captured

    with tracer.invoke("_ _ _ _ _ _ _"):
        barrier()  # Wait for value from previous invoke
        model.transformer.wte.output = embeddings  # Use captured value
        output = model.lm_head.output.save()
```

**Note:** Barriers are only needed when **setting** values across invokes. Reading values independently in each invoke doesn't require barriers.

## Multi-Token Generation

For autoregressive generation with interventions:

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    # Get the full generated output
    output = model.generator.output.save()

# Decode the generated tokens
print(model.tokenizer.decode(output[0]))
```

**Apply interventions to all generation steps:**

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    hidden_states = list().save()

    with tracer.all():  # Apply to ALL generation iterations
        # Intervention applied at each step
        model.transformer.h[5].output[0][:, -1, :] *= 0.9
        hidden_states.append(model.transformer.h[-1].output)
```

**Apply interventions to specific iterations:**

```python
with model.generate("Hello", max_new_tokens=10) as tracer:
    with tracer.iter[2:5]:  # Only iterations 2, 3, 4
        model.transformer.h[5].output[0][:] = 0
```

## Gradient Access

Access gradients using the `.backward()` context manager. Gradients must be accessed in **reverse order** (following backpropagation order):

```python
with model.trace(prompt):
    # Register intermediate values in forward order
    hidden = model.transformer.h[-1].output[0]
    hidden.requires_grad = True  # Enable gradient flow

    logits = model.lm_head.output

    # Enter backward context
    with logits.sum().backward():
        # Access gradients in REVERSE order
        logits_grad = logits.grad.save()
        hidden_grad = hidden.grad.save()
```

**Alternative with retain_grad() (simpler for just accessing gradients):**

```python
with model.trace(prompt):
    hidden = model.transformer.h[-1].output[0].save()
    hidden.retain_grad()

    logits = model.lm_head.output.save()
    logits.sum().backward()

# Access gradient after trace
print(hidden.grad)
```

## Utility Features

**Early stopping (skip remaining layers):**

```python
with model.trace(prompt) as tracer:
    early_hidden = model.transformer.h[3].output[0].save()
    tracer.stop()  # Skip layers 4+
```

**Shape inference without execution:**

```python
with model.scan(prompt):
    shape = model.transformer.h[0].output[0].shape.save()
```

**Cache all module outputs:**

```python
with model.trace(prompt) as tracer:
    cache = tracer.cache()  # Stores all outputs
```

## Remote Execution (NDIF)

For large models, use NDIF's remote infrastructure:

```python
from nnsight import CONFIG
CONFIG.set_default_api_key("YOUR_NDIF_API_KEY")

model = LanguageModel("meta-llama/Llama-3.1-70B")

with model.trace("Hello", remote=True):
    hidden = model.model.layers[-1].output[0].save()
```

