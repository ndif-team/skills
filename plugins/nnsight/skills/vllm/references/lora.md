# LoRA adapters

Build with `enable_lora=True` and name a PEFT adapter directory per request with
vLLM's `LoRARequest`. Needs an adapter on disk, so the suite does not run it:

<!-- test: skip -->
```python
from nnsight.modeling.vllm import VLLM
from vllm.lora.request import LoRARequest

model = VLLM("meta-llama/Llama-3.2-1B", dispatch=True, enable_lora=True, max_lora_rank=8)
adapter = LoRARequest(lora_name="mine", lora_int_id=1, lora_path="/path/to/adapter")
model.vllm_entrypoint.llm_engine.add_lora(adapter)    # load it now; raises if it cannot

with model.trace(temperature=0.0, max_tokens=1, lora_request=adapter) as tracer:
    with tracer.invoke("The Eiffel Tower is in the city of"):                      # the trace's adapter
        adapted = model.logits.save()
    with tracer.invoke("The Eiffel Tower is in the city of", lora_request=None):   # opts out: base model
        base = model.logits.save()
```

- **A request naming an adapter the engine cannot load kills the engine**
  (`EngineDeadError`; a missing path, a rank over `max_lora_rank`), as in plain
  vLLM. `add_lora` first: it raises an ordinary exception with the real message
  and the engine keeps serving.
- `lora_request=` goes on `trace`, `invoke` or a plain `generate`. On `trace` it is
  the default for its invokes; an invoke that names its own, `None` included,
  keeps it. Without `enable_lora` it raises `VLLMValidationError: ... LoRA is not
  enabled!`.
- `max_lora_rank` (vLLM's default is 16) has to cover the adapter's rank;
  `max_loras` (default 1) is how many different adapters one step can hold.
- A LoRA-wrapped layer's `.output` is the adapted value. Adapters work with taps,
  `model.edit()`, the async engine and `serve=` (start the server with
  `--enable-lora True`; `lora_path` is read on the server).
