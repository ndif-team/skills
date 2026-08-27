# Serving: nnsight-serve

`nnsight-serve` holds one dispatched async engine and accepts traces and edits
from GPU-less clients. Every block in this file needs a running server, so none
is executed by the suite.

```bash
nnsight-serve Qwen/Qwen3-8B --port 8000 --enable-prefix-caching False \
    [--api-key SECRET] [--gpu-memory-utilization 0.5] [--max-model-len 4096]
```

- `--help` lists only `--host`, `--port`, `--api-key`. Every other `--flag value`
  is forwarded to vLLM's `EngineArgs` as `flag=value`. Booleans take a literal:
  `--enable-prefix-caching False`, **not** vLLM's `--no-enable-prefix-caching`
  (that crashes the engine with an unexpected keyword).
- Start with prefix caching off if any client will `edit(serve=url)`.
- Poll `GET /health` for `{"status": "ok"}`; the engine takes a minute or two.
- The routes are `/health`, `POST /v1/nnsight/generate`,
  `POST /v1/nnsight/register/{id}` and `.../clear`. It is **not** an
  OpenAI-compatible server: every request is an nnsight trace. A "plain" request
  is a trace whose body saves only `tracer.result`.
- The engine core is a child process; if you kill the server rather than
  interrupt it, kill the `EngineCore` pid from its log too.
- `--host` defaults to loopback; the server runs client-sent code.

## A client

<!-- test: skip -->
```python
from nnsight.modeling.vllm import VLLM

URL = "http://127.0.0.1:8000"
model = VLLM("Qwen/Qwen3-8B")                 # meta tree only, never dispatched

with model.trace("The capital of France is", serve=URL, temperature=0.0, max_tokens=1) as tracer:
    out = model.model.layers[16].output
    resid = (out[0] + out[1])[-1].clone().save()
    top = model.logits.topk(3).indices.save()
    result = tracer.result.save()               # last read; the finished RequestOutput

print(resid.shape, model.tokenizer.batch_decode(top[0]), result.outputs[0].text)
```

The server returns saved values only; save `tracer.result` to get the
`RequestOutput` back. `serve=` is accepted by `trace` and `edit`; a with-less
`model.generate(..., serve=url)` is not routed and would try to dispatch a local
engine. Build and runtime errors come back with their real type and traceback.

## An installed block, seen by every request

<!-- test: skip -->
```python
import nnsight

with model.edit(serve=URL) as (tracer, edit):
    norms = nnsight.save([])
    for step in tracer.all():
        norms.append(model.model.layers[10].output[1][-1].float().norm().item())

with model.trace("Water boils at", serve=URL, temperature=0.0, max_tokens=6) as tracer:
    result = tracer.result.save()
print(result.outputs[0].text, result.saves["norms"])     # ' 100°C.' and six floats

edit.clear()
# after clear, an output has no .saves attribute at all: getattr(result, "saves", {})
```

Other clients of the same server — anything that submits a trace — get the
edit's values on their outputs too, which is the way to instrument traffic you do
not control.
