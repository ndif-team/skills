# vLLM

`VLLM` runs your block inside vLLM's worker process: the block is serialized onto
each request, runs against the engine's real modules, and its saved values are
shipped back. Use it for many concurrent prompts, streamed tokens or a served
engine. It has no gradients, `model.scan()` or image inputs
([Not supported](#not-supported)).

```bash
pip install "nnsight[vllm]"
```

nnsight targets vLLM 0.33: the extra pins `vllm>=0.33,<0.34`, and another vLLM
installed over it can be wrong without an error. That vLLM needs Python 3.11 or
later, and its wheel is built for CUDA 13, so on an NVIDIA driver too old for
CUDA 13 torch finds no GPU.

Every example on this page needs a CUDA GPU and that install. They use
`HuggingFaceTB/SmolLM2-135M-Instruct`, a Llama-style trunk, so what they show
about layer outputs holds for Llama, Qwen and Mistral.

- [serving](vllm-serving.md): open it to run `nnsight-serve`, to trace or edit
  from a client with no GPU (`serve=url`), or to stream tokens from the async
  engine.
- [throughput and architectures](vllm-throughput-and-architectures.md): open it to
  keep CUDA-graph replay with `taps=`, to shard with `tensor_parallel_size`, or to
  trace a mixture-of-experts or hybrid linear-attention checkpoint.

## What a block sees

```python
import torch
import nnsight
from nnsight.modeling.vllm import VLLM

if __name__ == "__main__":      # required in a script; later blocks continue inside it
    model = VLLM("HuggingFaceTB/SmolLM2-135M-Instruct", dispatch=True,
                 gpu_memory_utilization=0.2, max_model_len=1024)

    with model.trace("The Eiffel Tower is in the city of", temperature=0.0, max_tokens=1):
        out = model.model.layers[12].output
        kind = nnsight.save(type(out).__name__)
        mlp_out = out[0].clone().save()           # a served value: clone what you keep
        resid = (out[0] + out[1]).save()          # a new tensor: the stream after layer 12
        logits = model.logits.save()
        samples = model.samples.save()

    assert kind == "tuple"
    assert tuple(mlp_out.shape) == tuple(resid.shape) == (9, 576)     # 9 prompt tokens, no batch axis
    assert tuple(logits.shape) == (1, 49152) and tuple(samples.shape) == (1, 1)
    assert model.tokenizer.decode(samples.item()) == " Paris"
```

1. **No batch axis.** A served value is `[tokens, hidden]` for your request's
   rows: the prefill serves every prompt token, each decode step one row. The last
   position is `[-1]`, never `[:, -1, :]`. (A checkpoint vLLM runs through its
   Transformers backend serves `[1, tokens, hidden]`: check `.shape` once.)
2. **A decoder layer's `.output` is a pair `(hidden, residual)`.** vLLM fuses each
   residual add into the next norm, so `output[0]` is this layer's MLP output and
   `output[1]` the residual stream after its attention. The stream after the layer
   is `out[0] + out[1]`. To patch a layer write both elements; to steer, add to
   either.
3. **Clone what you keep.** A served value is a view of the engine's live buffer,
   which a fused kernel overwrites once your block moves on: an un-cloned
   `out[0].save()` comes back holding a later layer's data, with no error.
   `.clone()` it, or save something computed from it. (`NNSIGHT_VLLM_CLONE_READS=1`
   in the environment before `VLLM(...)` serves copies instead; in-place edits
   then never reach the model, so assign `layer.output = out`.)
4. **Where tensors live.** A tensor the block references from outside (a steering
   vector) travels with the block: move it onto the served value with
   `v.to(h.device, h.dtype)`. Saved tensors come back on the engine's device
   unless you `.cpu()` them in the block.
5. **Numbers depend on the batch a request was scheduled with.** In bfloat16 a
   nearly tied argmax can flip between two runs of one greedy prompt. Compare a
   clean and an edited invoke of the same trace, not two runs.

`model.logits` is `[1, vocab]` for the step and `model.samples` is `[1, 1]`.
Assigning `model.logits` changes what the sampler sees; assigning `model.samples`
forces the token the engine continues from.

**The module tree is vLLM's, not transformers'.** `self_attn.qkv_proj` and
`mlp.gate_up_proj` are fused projections, and every linear layer returns
`(value, bias)`, so its value is `.output[0]`. A decoder layer's `.input` is the
positions tensor: `args, _ = layer.inputs` gives `(positions, hidden, residual)`,
and `args[1] + args[2]` is the stream entering the layer. `print(model)` shows
the tree, without `dispatch=True` too;
`scripts/inspect_model.py` describes the transformers tree. Attention heads are
slices of `self_attn.o_proj.input`, as in
[access-and-modify](access-and-modify.md).

## Loading

`VLLM(repo_id, *, mode="sync", dispatch=False, taps=(), **engine_kwargs)`. Without
`dispatch=True` only a weightless module tree is built and the engine starts at
the first trace. Other keywords are vLLM engine arguments
(`tensor_parallel_size`, `gpu_memory_utilization`, `max_model_len`).
`mode="async"` builds the streaming engine ([serving](vllm-serving.md)).

- **A script needs the `if __name__ == "__main__":` guard**; a notebook does not.
  The engine's spawned process re-imports your file, and unguarded it dies with
  `An attempt has been made to start a new process before the current process has
  finished its bootstrapping phase`.
- **Leave the engine's mode to nnsight.** It runs eagerly and serves every
  location unless you declare `taps=`
  ([throughput](vllm-throughput-and-architectures.md)), and it builds with prefix
  caching and chunked prefill off. Leave `VLLM_USE_V2_MODEL_RUNNER` unset as
  well: `0` selects a runner nnsight does not instrument. The
  [error table](#errors-and-warnings) shows what overriding these gets you.
- **Prompts.** A string, a list of token ids, a tokenizer's output or a vLLM
  prompt dict. A chat messages list is read as several prompts (`ValueError:
  Multiple prompts per invoke are not supported`): pass
  `model.tokenizer.apply_chat_template(messages, add_generation_prompt=True)`.

## Sampling and generation

Sampling keywords are the fields of vLLM's `SamplingParams` and go on
`trace(...)` or `invoke(...)`, never on the constructor. Greedy is
`temperature=0.0` (the default is `1.0`). The length is `max_tokens` (default 16;
`max_new_tokens` is accepted for it), and `min_tokens=N` or `ignore_eos=True`
holds a run to its count. `do_sample` and `min_new_tokens` raise `TypeError`.
`with model.generate(...)` is `with model.trace(...)`. Under `tracer.iter`, step 0
is the prefill and step *k* serves generated token *k*:

```python
with model.trace("The capital of France is", temperature=0.0, max_tokens=6, ignore_eos=True) as tracer:
    rows, tokens = nnsight.save([]), nnsight.save([])
    for step in tracer.iter[:6]:
        rows.append(model.model.layers[20].output[0].shape[0])
        if step == 2:
            model.samples = torch.full_like(model.samples, 3_000)       # force this step's token
        tokens.append(model.samples.item())
    result = tracer.result.save()          # the finished RequestOutput: the last read

assert rows == [5, 1, 1, 1, 1, 1]          # the prefill, then one row per step
assert tokens == list(result.outputs[0].token_ids) and tokens[2] == 3_000
```

- **`max_tokens` is a cap.** A loop the run cannot supply is cut short as in
  [generation](generation.md), and its warning is printed in the engine's process,
  not yours: check `len()` of what you collected.
- **A block can see one step more than the output holds.** vLLM schedules the
  next step before the last token is known, so a request that ends on a stop
  token or `stop` string runs one more forward than
  `len(result.outputs[0].token_ids)`; ending at `max_tokens` is exact. Count
  against the output, or build with `VLLM(..., async_scheduling=False)`.
- A write to a layer below one already read this step lands on the next step
  ([generation](generation.md)): write upstream first.
- `tracer.cache()` works as in [caching-and-scan](caching-and-scan.md); it copies
  to the CPU as it records, so it needs no clone.

## Logit lens

`model.lm_head(h)` raises `LMHead's weights should be used in the sampler`.
Decode through the model's own logits path, `logits_processor`.
`norm(h)` returns a tensor; `norm(h, residual)` is the fused add and returns a
pair.

```python
with model.trace("The Eiffel Tower is in the city of", temperature=0.0, max_tokens=1):
    out = model.model.layers[20].output
    h = (out[0] + out[1])[-1:]                         # residual stream, last token: [1, hidden]
    lens = model.logits_processor(model.lm_head, model.model.norm(h)).argmax(-1).item().save()

print(model.tokenizer.decode(lens))
```

## One prompt per invoke

Each `tracer.invoke(prompt)` is one vLLM request, and the engine batches them. A
list of prompts in one invoke is refused. What comes back:

- a name saved in one invoke is that value;
- the same name saved in several invokes is a **list**, in invoke order;
- a container saved *above* the invokes is copied to every request and merged
  afterwards: same-length lists slot by slot (`None` is an unwritten slot), dicts
  by key, so the keyed dict of [SKILL.md](../SKILL.md) works. **`append` does not
  merge**: you get one element.

```python
prompts = ["The capital of France is", "The capital of Japan is", "Two plus two is"]

with model.trace(temperature=0.0, max_tokens=1) as tracer:
    slots = nnsight.save([None] * len(prompts))
    appended = nnsight.save([])
    for i, p in enumerate(prompts):
        with tracer.invoke(p):
            out = model.model.layers[20].output
            last = (out[0] + out[1])[-1].save()                # one name, every invoke -> list
            slots[i] = last.norm().item()
            appended.append(i)

assert len(last) == 3 and all(s is not None for s in slots) and len(appended) == 1
```

### A value from one prompt into another: two traces

A value one invoke reads is not visible to a sibling (`NameError`),
`tracer.barrier` raises `NotImplementedError`, and `model.session()` does not
bridge traces. Save in one trace and reference it in the next; the saved value
ships with the second block. Activation patching:

```python
CLEAN, CORRUPT, L, POS = "The Eiffel Tower is in the city of", "The Colosseum is in the city of", 12, slice(1, 4)
paris = model.tokenizer.encode(" Paris", add_special_tokens=False)[0]

with model.trace(CLEAN, temperature=0.0, max_tokens=1):
    donor = nnsight.save(tuple(o.clone() for o in model.model.layers[L].output))

with model.trace(temperature=0.0, max_tokens=1) as tracer:
    with tracer.invoke(CORRUPT):
        base = model.logits.softmax(-1)[0, paris].item().save()
    with tracer.invoke(CORRUPT):
        for served, saved in zip(model.model.layers[L].output, donor):     # both elements
            served[POS] = saved[POS].to(served.device)
        patched = model.logits.softmax(-1)[0, paris].item().save()

assert patched > base
```

Writing *into* the served value, as above, always fits. A whole-value replacement
(`layer.output = t`) must keep the rows the block owns, and nothing checks: a
donor from a prompt of another length can kill the engine and every request in
it. Slice the donor to the rows you write.

### Trace-level keywords and `n > 1`

Sampling keywords on `trace(...)` fill in what an invoke did not name; whatever an
invoke names wins. `n=k` fans one prompt into `k` sequences and the block runs
once per sequence, so every saved name comes back as a list of `k` values in
sequence order. That includes `tracer.result`: `result[i].outputs` holds sequence
`i` alone.

```python
with model.trace("My favourite colour is", temperature=0.9, seed=0, n=3, max_tokens=4, ignore_eos=True) as tracer:
    caps = nnsight.save([])
    for _ in tracer.iter[:4]:
        caps.append(model.model.layers[8].output[1][-1].clone())
    result = tracer.result.save()

assert len(caps) == 3 and len(caps[1]) == 4      # one list per sequence, four steps each
assert len(result) == 3 and [o.index for o in result[1].outputs] == [1]
```

## Sweeps: `model.edit()`

`model.edit()` installs a block on the engine. It is sent once and runs for every
request the engine serves afterwards, plain `model.generate(...)` calls and other
clients of a served engine included; each request's saved values come back on its
output as `.saves`. There is no `tracer.invoke` inside it, and
`for step in tracer.all():` makes it follow every generated step. Each request
gets its own copy of the block's scope, containers bound outside it included, so
an edit cannot accumulate across requests.

```python
sweep = [f"Fact number {i} about the ocean is that it" for i in range(200)]

with model.edit() as (tracer, edit):
    out = model.model.layers[20].output
    feat = (out[0] + out[1]).mean(0).cpu().save()

outputs = model.generate(sweep, max_tokens=1, temperature=0.0)      # plain requests; the block runs
feats = torch.stack([o.saves["feat"] for o in outputs])
texts = [o.outputs[0].text for o in outputs]
edit.clear()

assert feats.shape == (200, 576) and len(texts) == 200
```

The same sweep as one invoke per prompt serializes the block once per invoke, and
the model's whole envoy tree with it when the block names `model`
(`model.logits`, `model.samples`, `model.model.layers[i]`) or reads
`tracer.result`. Prefer the edit, and where you do trace a sweep bind the layer
outside the block (`layer = model.model.layers[20]`).

- `edit.clear()` removes one edit, `model.clear_edits()` all of them. Afterwards
  an output has no `.saves` attribute at all: read `getattr(output, "saves", {})`.
- **Named edits.** `model.edit(name="probe")` tags an edit, and `edits=[...]` on
  `trace`, `invoke` or `generate` runs the named edits it lists plus every unnamed
  one. No `edits=` runs them all, `edits=[]` only the unnamed, an unknown name
  raises `ValueError`. On a traced request the edit's values are on
  `tracer.result.saves`.
- **Leave prefix caching off.** A cached token gets no forward pass, so an edit
  would silently see fewer rows than the prompt has.

## LoRA adapters

Name a PEFT adapter directory per request with vLLM's `LoRARequest`. It needs an
engine built with `enable_lora=True` and an adapter on disk trained for the base
model, so this block builds its own engine.

```python
from vllm.lora.request import LoRARequest

model = VLLM("HuggingFaceTB/SmolLM2-135M-Instruct", dispatch=True, enable_lora=True, max_lora_rank=8,
             gpu_memory_utilization=0.2, max_model_len=1024)
adapter = LoRARequest(lora_name="mine", lora_int_id=1, lora_path="/path/to/adapter")
model.vllm_entrypoint.llm_engine.add_lora(adapter)    # load it now; raises if it cannot

with model.trace(temperature=0.0, max_tokens=1, lora_request=adapter) as tracer:
    with tracer.invoke("The Eiffel Tower is in the city of"):                      # the trace's adapter
        adapted = model.logits.save()
    with tracer.invoke("The Eiffel Tower is in the city of", lora_request=None):   # opts out: the base model
        base = model.logits.save()

assert not torch.equal(adapted, base)         # a trained adapter moves the logits
```

- **Load the adapter with `add_lora` first.** A request naming an adapter the
  engine cannot load (a missing path, a rank over `max_lora_rank`) kills the
  engine with `EngineDeadError`. `add_lora` raises an ordinary exception with the
  real message, and the engine keeps serving.
- `lora_request=` goes on `trace`, `invoke` or a plain `generate`. On `trace` it
  is the default for its invokes; an invoke that names its own, `None` included,
  keeps it.
- `max_lora_rank` (vLLM's default is 16) has to cover the adapter's rank;
  `max_loras` (default 1) is how many different adapters one step can hold.
- A LoRA-wrapped layer's `.output` is the adapted value. Over `serve=`, start the
  server with `--enable-lora True`; `lora_path` is read on the server.

## `.skip()` can kill the engine

A skip must replace every row of the step, and a step's rows are whatever the
scheduler batched: other invokes, other clients' requests. When the replacement
does not cover them, `ValueError: A batched .skip() has to cover every row` is
raised outside any one request and the engine dies with `EngineDeadError`.
`module.skip(module.input)` on a decoder layer is fatal whatever is scheduled:
its `.input` is the positions tensor. Treat `.skip()` as unsupported and write the
module's output instead (`model.model.layers[L].mlp.output[:] = 0`).

## Not supported

- **Gradients, `backward()`, `model.scan()`.** Use `TransformersModel`; on a
  model larger than one GPU, the [tensor-parallel guide](tensor-parallel.md);
  larger than your machine, [remote](remote.md). For shapes, trace a prompt and
  read `.shape`.
- **`.source` inside a fused kernel.** `.source` reaches the Python operations of
  a forward (`self_attn.source.qkv_split_0`), not a kernel's interior.
- **Image inputs.** A vision-language checkpoint loads and traces on text
  prompts. For images use
  [diffusion-and-multimodal](diffusion-and-multimodal.md).
- **Pipeline and data parallelism, speculative decoding.**
  `pipeline_parallel_size > 1` builds and then fails every trace with
  `RuntimeError: UnknownPersistentIdError: Module:...`; `data_parallel_size > 1`,
  `speculative_config=` and `enable_batch_sharded_sampling=True` raise
  `NotImplementedError: ... is not supported on an nnsight engine` at
  construction. Shard with `tensor_parallel_size`, or build one `VLLM` per GPU.

## Errors and warnings

An exception in a block ends its own request only. It is re-raised in your process
as a `RuntimeError` whose message starts with the original type and message and
carries an "Intervention traceback" at your line: `1 / 0` arrives as
`RuntimeError: ZeroDivisionError: division by zero`. Catch `RuntimeError` and
match on the text. A trace that raises binds none of its saved names, not even
another invoke's. **Warnings do not come back**: the engine's
`(EngineCore pid=...)` process prints them.

| Message contains | Cause and fix |
|---|---|
| `is not a tap on this engine` | A block read a module location that is not in `taps=`; its request's `RuntimeError`. Add the location, or build without `taps` |
| `prompt was split across steps by chunked prefill` | The engine was built with `enable_chunked_prefill=True`; that request's `RuntimeError`. Drop the flag, or raise `max_num_batched_tokens` |
| `enforce_eager=... contradicts taps=[...]` | `ValueError` when the engine is built: its mode follows from `taps`. Drop `enforce_eager` |
| `vLLM chose its V1 model runner for this engine` | In the `(EngineCore pid=...)` log of a build that failed: `VLLM_USE_V2_MODEL_RUNNER=0` is set, or an engine option only vLLM's older runner supports was passed. Unset the variable, or drop the option the log names |
| `records full CUDA graphs, which no tap can be registered in` | In the log of a build that failed: a `compilation_config` asking for full graphs on a `taps=` engine. Drop its `cudagraph_mode` |
| `EngineDeadError` | The engine process died, with every in-flight request: a `.skip()`, a replacement with the wrong rows, or a `lora_request` it could not load. The cause is in the `(EngineCore pid=...)` log. Build a new engine |
| `Can't pickle`, `cannot pickle 'generator' object`, `Can't get local object` | That request's `RuntimeError`: saved values come home through plain `pickle`, which cannot carry a lambda, a generator or an instance of a locally defined class. Save tensors, numbers and plain containers |
| `Engine core initialization failed` | The build failed in the worker; the real message is in the `(EngineCore pid=...)` lines above it. Common: a `.source` operation in `taps=` that the forward does not have, and on a shared GPU `Error in memory profiling` (build again) |

The engine core is a child process. If your script is killed rather than
interrupted, it can outlive the script and hold the GPU memory: kill the
`EngineCore` pid that `nvidia-smi` lists.
