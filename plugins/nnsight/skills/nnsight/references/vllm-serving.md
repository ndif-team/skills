# vLLM: serving and the async engine

Depth for the [vLLM guide](vllm.md). `nnsight-serve` holds one engine and accepts
traces and edits from clients with no GPU; `mode="async"` streams a trace's
outputs as they generate.

## nnsight-serve

The server needs a CUDA GPU and `pip install "nnsight[serve]"`.

```bash
nnsight-serve HuggingFaceTB/SmolLM2-135M-Instruct --port 8000 \
    --gpu-memory-utilization 0.2 --max-model-len 1024 [--api-key SECRET]
```

- **Flags.** `--host`, `--port` (default 6677) and `--api-key` are the server's.
  Every other `--flag value` becomes the engine keyword `flag=value`. A boolean
  takes a literal: `--enable-prefix-caching False`, not vLLM's
  `--no-enable-prefix-caching`, which stops the server with an unexpected-keyword
  `TypeError`.
- **Short flags are dropped.** `-tp 2` prints `Ignoring unknown argument` to
  stderr and the server comes up at `tensor_parallel_size=1`.
- **`taps=` has no flag**: `--taps model.layers.*.output` arrives as one string
  and is refused with `ValueError: Tap 'm' names no module`. Build a tapped server
  in Python, the way `nnsight.modeling.vllm.serve.cli.main` builds this one.
- **Readiness.** Poll `GET /health` until it answers `{"status": "ok"}`.
- **It is not an OpenAI-compatible server.** Every request is an nnsight trace
  (`POST /v1/nnsight/generate`); a plain request is a trace whose body saves only
  `tracer.result`.
- **It runs code that clients send.** `--host` defaults to loopback. Bind another
  address only on a network you trust, with `--api-key`.

## A client

A client needs `pip install "nnsight[vllm]"` to build the module tree, and no
GPU. The two blocks below need the server above to be running.

```python
import nnsight
from nnsight.modeling.vllm import VLLM

URL = "http://127.0.0.1:8000"
model = VLLM("HuggingFaceTB/SmolLM2-135M-Instruct")       # the module tree only, never dispatched

with model.trace("The capital of France is", serve=URL, temperature=0.0, max_tokens=1) as tracer:
    out = model.model.layers[16].output
    resid = (out[0] + out[1])[-1].save()
    top = model.logits.topk(3).indices.save()
    result = tracer.result.save()               # the finished RequestOutput

assert tuple(resid.shape) == (576,) and resid.device.type == "cpu"
print(model.tokenizer.batch_decode(top[0]), result.outputs[0].text)
```

- Saved names are bound after the block as on a local engine, a list for a name
  saved in several invokes or under `n > 1`, and tensors arrive on the CPU. Only
  saved values come back: save `tracer.result` for the text.
- **Save inside the invokes.** A `nnsight.save(...)` in the trace body above them
  fails the served trace with `RuntimeError: ValueError: save() was called
  outside a trace`.
- **`serve=` is for `trace` and `edit`.** `model.generate(..., serve=url)` outside
  a `with` is not routed and tries to build a local engine.
- An error in the block arrives as the `RuntimeError` of the vLLM guide, and the
  trace binds nothing. A transport failure is `ConnectionError: nnsight-serve
  returned <status>: <detail>`, such as 401 for a wrong key: with
  `--api-key SECRET` on the server, pass `api_key="SECRET"` beside `serve=`.

## An edit that every request runs

```python
with model.edit(serve=URL) as (tracer, edit):
    norms = nnsight.save([])
    for step in tracer.all():
        norms.append(model.model.layers[10].output[1][-1].float().norm().item())

with model.trace("Water boils at", serve=URL, temperature=0.0, max_tokens=6, ignore_eos=True) as tracer:
    result = tracer.result.save()

assert len(result.saves["norms"]) == 6        # one float per step, on the request's output
edit.clear()
```

Every other client of the server gets the edit's values on its outputs too, which
is how to instrument traffic you do not control. Named edits work as on a local
engine, except that an unknown name comes back as the request's error
(`RuntimeError: ValueError: edits=...`).

## The async engine

`VLLM(repo, dispatch=True, mode="async")` builds vLLM's `AsyncLLM`. The block is
written the same way; on exit the request is submitted and `tracer.backend` is
its stream. `async for output in tracer.backend` yields cumulative
`RequestOutput`s, and `last = await tracer.backend` drains the stream and returns
the finished one. This example needs a CUDA GPU and `nnsight[vllm]`.

```python
import asyncio
from nnsight.modeling.vllm import VLLM

async def main():
    model = VLLM("HuggingFaceTB/SmolLM2-135M-Instruct", dispatch=True, mode="async",
                 gpu_memory_utilization=0.2, max_model_len=1024)     # built on the loop that awaits it
    with model.trace("The capital of France is", temperature=0.0, max_tokens=8) as tracer:
        resid = sum(model.model.layers[10].output).save()
    last = await tracer.backend
    assert last.saves["resid"].shape[-1] == 576      # `resid` itself is unbound on this path
    print(last.outputs[0].text)

if __name__ == "__main__":
    asyncio.run(main())
```

- **Saved names are not bound after the block**, which has returned before the
  stream is consumed. The values are on the finished output,
  `last.saves["resid"]`; intermediate outputs carry an empty `.saves`. With
  `n > 1`, sequence *i*'s values are `last.nnsight_sequences[i]`.
- **The stream is consumed once.** A second `await tracer.backend` returns `None`
  with no error, and fails later as an `AttributeError` on `.outputs`.
- **One engine, one event loop.** `AsyncLLM` binds to the loop that built it: two
  `asyncio.run()` calls over one model hang on the second with no error. Build the
  model inside the coroutine, or in a server's startup hook.
- **One prompt per trace.** Several invokes raise `NotImplementedError`. Fire many
  traces and `asyncio.gather` their backends; vLLM batches them.
- An error in the block is raised when the stream is consumed.
- `model.generate(...)` outside a `with` returns a coroutine to await. Edits use
  `async with model.edit()`, `await edit.aclear()` and
  `await model.aclear_edits()`; the plain forms raise `NotImplementedError`.
