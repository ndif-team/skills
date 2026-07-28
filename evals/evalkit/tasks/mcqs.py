"""Multiple-choice questions.

Ported from nnsight's tests/agent-evals (dev). Questions whose answer changed in
0.8 were rewritten rather than dropped — the wrong-on-0.8 version of each is a
good distractor.
"""

from ..registry import Difficulty, register_mcq


register_mcq(
    id='mcq_basic_01_save_required',
    name='01 save required',
    difficulty=Difficulty.BASIC,
    question='Inside `with model.trace(...)`, you assign `hs = model.layer.output`. After the trace exits you reference `hs` -- what happens?',
    choices=[
        '`hs` holds the tensor; nnsight always exposes assigned variables.',
        '`hs` is undefined / stale: only variables marked with `.save()` (or `nnsight.save(x)`) survive the root-trace exit filter.',
        '`hs` raises `OutOfOrderError` because the model is no longer running.',
        '`hs` is a deferred proxy; calling it executes the trace again.',
    ],
    correct_index=1,
    explanation='Root-trace exit filters locals against `Globals.saves` (`src/nnsight/intervention/tracing/base.py:537`); only `.save()`-d objects propagate out.',
    tags=['save', 'trace', 'basic'],
)

register_mcq(
    id='mcq_basic_03_output_vs_input',
    name='03 output vs input',
    difficulty=Difficulty.BASIC,
    question='What is the difference between `module.input` and `module.inputs` on an Envoy?',
    choices=[
        '`module.input` returns the first positional (or first kwarg) argument; `module.inputs` returns the full `(args_tuple, kwargs_dict)` pair.',
        '`module.input` is read-only; `module.inputs` is writable.',
        "`module.input` is the previous layer's output; `module.inputs` is the input embeddings.",
        'They are aliases; both return the same value.',
    ],
    correct_index=0,
    explanation="`docs/concepts/envoy-and-eproperty.md` -- both share key='input' but `input` is preprocessed to extract the first positional, while `inputs` returns the full `(args, kwargs)` tuple.",
    tags=['envoy', 'basic', 'input'],
)

register_mcq(
    id='mcq_basic_04_default_invoke',
    name='04 default invoke',
    difficulty=Difficulty.BASIC,
    question='`with model.trace("Hello"):` is equivalent to which form using explicit invokes?',
    choices=[
        "`with model.trace() as tracer: tracer.invoke('Hello')` -- only the call, no with-block.",
        "`with model.trace() as tracer:\\n    with tracer.invoke('Hello'):\\n        ...` (the body becomes the body of an implicit invoke).",
        "`with model.session('Hello'):` -- session and trace are interchangeable for single inputs.",
        'There is no equivalent; the implicit form is special-cased and cannot be expressed with explicit invokes.',
    ],
    correct_index=1,
    explanation='`docs/concepts/deferred-execution.md` -- positional args to `.trace(...)` create an implicit `Invoker` whose body is the with-block.',
    tags=['trace', 'invoke', 'basic'],
)

register_mcq(
    id='mcq_intermediate_02_cross_invoke_barrier',
    name='02 cross invoke barrier',
    difficulty=Difficulty.INTERMEDIATE,
    question='Two invokes on the same trace each access `model.transformer.h[5].output`. Invoke 1 captures `clean_hs = ...output[0][:, -1, :]`; invoke 2 writes `...output[0][:, -1, :] = clean_hs`. With no barrier, what happens?',
    choices=[
        'Cross-invoke variable propagation handles it transparently; `clean_hs` is always available.',
        "Invoke 2 sees `NameError` (or a missed value) because invoke 1 hasn't materialized `clean_hs` by the time invoke 2 reaches the same module.",
        'The two invokes run in true parallel and a race condition produces nondeterministic results.',
        'nnsight automatically inserts a barrier whenever it detects same-module access.',
    ],
    correct_index=1,
    explanation='`docs/gotchas/cross-invoke.md` -- when both invokes touch the same module path you must call `tracer.barrier(n)` to synchronize at the materialization point.',
    tags=['barrier', 'cross-invoke', 'intermediate'],
)

register_mcq(
    id='mcq_intermediate_03_inplace_vs_replace',
    name='03 inplace vs replace',
    difficulty=Difficulty.INTERMEDIATE,
    question="What's the difference between `module.output[0][:] = 0` and `module.output = (torch.zeros_like(...), ...)`?",
    choices=[
        'They are interchangeable; both mutate the tensor the model sees.',
        '`[:] = 0` mutates the existing storage; bare `=` rebinds and triggers a SWAP event so the batcher substitutes the new value for the rest of the forward pass.',
        '`[:] = 0` is illegal inside a trace; only bare `=` is supported.',
        'Bare `=` is silently ignored unless wrapped in `nnsight.swap(...)`.',
    ],
    correct_index=1,
    explanation='`docs/gotchas/modification.md` -- in-place edits storage; bare assignment goes through `eproperty.__set__` which sends a SWAP event.',
    tags=['modify', 'intermediate', 'swap'],
)

register_mcq(
    id='mcq_intermediate_05_clone_before_save',
    name='05 clone before save',
    difficulty=Difficulty.INTERMEDIATE,
    question="""Inside a trace you write:
    before = model.h[0].output[0].save()
    model.h[0].output[0][:] = 0
    after = model.h[0].output[0].save()
What does `before` contain after the trace exits?""",
    choices=[
        'The original (pre-zero) tensor.',
        'The zeroed tensor -- `before` aliases the same storage that the in-place edit modified.',
        'A `RuntimeError` because `.save()` was called on the same tensor twice.',
        '`None` -- only the most recent save survives.',
    ],
    correct_index=1,
    explanation='`docs/gotchas/modification.md` -- `.save()` records id, not a snapshot. Use `.clone().save()` to capture the pre-mutation state.',
    tags=['save', 'modify', 'intermediate'],
)

register_mcq(
    id='mcq_intermediate_07_all_is_iter',
    name='07 all is iter',
    difficulty=Difficulty.INTERMEDIATE,
    question='What is the relationship between `tracer.all()` and `tracer.iter[:]`?',
    choices=[
        '`tracer.all()` runs every iteration in parallel; `tracer.iter[:]` is sequential.',
        '`tracer.all()` is a deprecated alias of `tracer.iter[0]`.',
        "`tracer.all()` literally returns `self.iter[:]` -- it's the same unbounded iterator with the same trailing-code footgun.",
        '`tracer.all()` includes the prefill pass; `tracer.iter[:]` does not.',
    ],
    correct_index=2,
    explanation='`docs/gotchas/iteration.md` -- `InterleavingTracer.all` returns `self.iter[:]` (`tracing/tracer.py:457`).',
    tags=['iter', 'all', 'intermediate'],
)

register_mcq(
    id='mcq_intermediate_10_grad_reverse_order',
    name='10 grad reverse order',
    difficulty=Difficulty.INTERMEDIATE,
    question='You captured `h3 = model.h[3].output[0]` and `h10 = model.h[10].output[0]` (both with `requires_grad_(True)`). Inside `with logits.sum().backward():`, in what order should you access `.grad`?',
    choices=[
        '`h3.grad.save()` then `h10.grad.save()` -- mirror forward order.',
        '`h10.grad.save()` then `h3.grad.save()` -- gradient hooks fire in reverse of the forward order.',
        'Either order works; gradients are buffered so order is irrelevant.',
        "Always wrap in `tracer.barrier(2)` -- there's no inherent order.",
    ],
    correct_index=1,
    explanation="`docs/gotchas/backward.md` -- backward fires hooks in reverse; deeper layers' grads arrive first.",
    tags=['backward', 'grad', 'intermediate'],
)

register_mcq(
    id='mcq_advanced_02_invoke_during_execution',
    name='02 invoke during execution',
    difficulty=Difficulty.ADVANCED,
    question='Which pattern triggers `ValueError: Cannot invoke during an active model execution / interleaving.`?',
    choices=[
        'Calling `.trace(...)` twice on the same model with no overlap.',
        'Opening a `tracer.invoke(...)` block inside another `tracer.invoke(...)` body, OR opening one inside a `for step in tracer.iter[:]:` loop.',
        'Calling `tracer.barrier(2)` after the model has started.',
        'Saving the same tensor with `.save()` twice in one trace.',
    ],
    correct_index=1,
    explanation='`docs/errors/invoke-during-execution.md` -- `Invoker.__init__` rejects construction when `tracer.model.interleaving` is true.',
    tags=['error', 'invoke', 'advanced'],
)

register_mcq(
    id='mcq_advanced_05_envoy_call_hook_default',
    name='05 envoy call hook default',
    difficulty=Difficulty.ADVANCED,
    question='Inside a trace, calling `model.sae(hidden)` on an auxiliary module routes through which path by default, and how do you get `.input`/`.output` hooks to fire?',
    choices=[
        'Routes through `module.__call__` by default; hooks always fire.',
        'Routes through `module.forward(...)` by default (bypassing hooks); pass `hook=True` to route through `__call__` so hooks fire.',
        'Always routes through `__call__`; pass `hook=False` to bypass.',
        'Auxiliary modules can never have hooks; you must register a custom `eproperty`.',
    ],
    correct_index=1,
    explanation='`docs/gotchas/integrations.md` and `docs/concepts/envoy-and-eproperty.md` -- `Envoy.__call__` defaults to `hook=False` inside a trace; pass `hook=True` to enable hook dispatch.',
    tags=['envoy', 'hook', 'advanced', 'sae'],
)

register_mcq(
    id='mcq_advanced_06_scan_save_required',
    name='06 scan save required',
    difficulty=Difficulty.ADVANCED,
    question="Inside `with model.scan('Hi'):`, you write `dim = model.transformer.h[0].output[0].shape[-1]` (a plain int). Outside, `print(dim)` -- what happens?",
    choices=[
        "Prints the int; scan blocks don't filter local variables.",
        'Raises `NameError` / undefined: scan is a tracing context; non-saved locals are filtered at exit. Use `nnsight.save(...)` for non-tensor values.',
        'Always prints `0` because FakeTensor shapes are zero.',
        'Prints a `FakeTensor` symbol; scan never produces ints.',
    ],
    correct_index=1,
    explanation='`docs/usage/scan.md` and `docs/gotchas/save.md` -- scan is a tracing context that goes through the same exit filter; use `nnsight.save(...)` for non-tensor values like ints.',
    tags=['scan', 'save', 'advanced'],
)

register_mcq(
    id='mcq_advanced_08_session_bundling',
    name='08 session bundling',
    difficulty=Difficulty.ADVANCED,
    question='When running multiple traces on a remote model, where should `remote=True` go?',
    choices=[
        'On every `model.trace(...)` call so each one queues independently.',
        'On the outer `model.session(remote=True)`; inner traces inherit the remote backend, the whole session is one request, and variables flow between traces without `.save()`.',
        'On `model.dispatch(remote=True)` once at the top of the script.',
        'Both on the session AND every inner trace -- doubling up reduces flakiness.',
    ],
    correct_index=1,
    explanation='`docs/gotchas/remote.md` and `docs/usage/session.md` -- one outer `remote=True` bundles all inner traces into a single NDIF request and a single queue wait.',
    tags=['remote', 'session', 'advanced'],
)

register_mcq(
    id='mcq_advanced_09_remote_save_transmission',
    name='09 remote save transmission',
    difficulty=Difficulty.ADVANCED,
    question='On a remote trace (`remote=True`), why does `local_list = []; with model.trace(..., remote=True): local_list.append(x)` end up empty?',
    choices=[
        "Remote traces don't support `.append`.",
        'The `local_list` lives in the client process; the `.append` runs on the server and is discarded when the request returns. Build the list inside the trace and `.save()` it.',
        '`.save()` was forgotten on `x`; once saved, the local list would populate.',
        'vLLM strips list mutations; use a `dict` instead.',
    ],
    correct_index=1,
    explanation="`docs/gotchas/remote.md` -- `.save()` is the only mechanism that ships values back; client-side containers don't travel to the server.",
    tags=['remote', 'save', 'advanced'],
)

register_mcq(
    id='mcq_advanced_10_blocking_false',
    name='10 blocking false',
    difficulty=Difficulty.ADVANCED,
    question="Using `with model.trace('Hi', remote=True, blocking=False) as tracer:`, how do you retrieve the result?",
    choices=[
        'The `tracer` object becomes the result tensor automatically once the job finishes.',
        'Poll `tracer.backend()`; it returns `None` while pending and the dict of saved values once `COMPLETED`. `tracer.backend.job_id` and `tracer.backend.job_status` track the request.',
        'Re-enter the same `with` block to fetch results.',
        'Call `model.fetch(tracer.id)` -- backend objects are not exposed.',
    ],
    correct_index=1,
    explanation='`docs/remote/non-blocking-jobs.md` -- the trace exits immediately after submission; poll `backend()` for the result dict.',
    tags=['remote', 'non-blocking', 'advanced'],
)

register_mcq(
    id='mcq_advanced_11_vllm_logits_samples',
    name='11 vllm logits samples',
    difficulty=Difficulty.ADVANCED,
    question='On `nnsight.modeling.vllm.VLLM`, what are `model.logits` and `model.samples`?',
    choices=[
        'Methods you call to fetch tensors: `model.logits()`.',
        "VLLM-specific eproperties on the `VLLM` instance: `model.logits` is the pre-sampling logit tensor (per step), `model.samples` is the sampled token ids (per step). They iterate via `tracer.iter` and don't exist on standard `LanguageModel`.",
        'Aliases for `model.lm_head.output` and `model.generator.output` respectively.',
        'Internal vLLM debug flags; not user-accessible.',
    ],
    correct_index=1,
    explanation="`docs/models/vllm.md` -- `vllm.py:102/112` defines `logits` and `samples` as iterating eproperties; they're VLLM-specific.",
    tags=['vllm', 'advanced', 'eproperty'],
)

register_mcq(
    id='mcq_meta_02_barrier_n',
    name='02 barrier n',
    difficulty=Difficulty.INTERMEDIATE,
    question='What is `n` in `tracer.barrier(n)`?',
    choices=[
        'The number of generation steps the barrier blocks for.',
        'The number of mediators (worker threads / invokes) that must hit `barrier()` before any are released.',
        'The number of attention heads to synchronize.',
        'The maximum number of seconds to wait before timing out.',
    ],
    correct_index=1,
    explanation='`docs/concepts/threading-and-mediators.md` -- BARRIER event releases all participants once `n` mediators have reached it (`interleaver.py:1123`).',
    tags=['barrier', 'meta', 'concept'],
)

register_mcq(
    id='mcq_meta_04_envoy_call_logitlens',
    name='04 envoy call logitlens',
    difficulty=Difficulty.ADVANCED,
    question='In a logit-lens snippet, `logits = model.lm_head(model.transformer.ln_f(hs))` runs inside a trace WITHOUT triggering `.input`/`.output` hooks on `lm_head` and `ln_f`. Why?',
    choices=[
        'Hooks are disabled for any module called via `__call__` inside a trace.',
        "`Envoy.__call__` defaults to `hook=False` inside an active trace, routing through `module.forward(...)` and bypassing the wrapped `__call__` (so the sentinel hook isn't taken).",
        'Hooks fire but their results are discarded silently.',
        '`lm_head` and `ln_f` are special-cased in `LanguageModel`.',
    ],
    correct_index=1,
    explanation='`docs/concepts/envoy-and-eproperty.md` -- `Envoy.__call__` (`envoy.py:239`) routes through `.forward(...)` when hook=False inside a trace.',
    tags=['envoy', 'logit-lens', 'meta'],
)

register_mcq(
    id='mcq_meta_05_debug_mode',
    name='05 debug mode',
    difficulty=Difficulty.INTERMEDIATE,
    question='When you set `CONFIG.APP.DEBUG = True`, what changes?',
    choices=[
        'The model runs in `torch.no_grad` mode.',
        'Exceptions inside a trace include the full nnsight internal stack frames; without it, tracebacks are reconstructed to point at user code only.',
        'All `.save()` calls also print their values to stderr.',
        'Remote traces are forced to run locally.',
    ],
    correct_index=1,
    explanation='`docs/reference/config.md` and `docs/errors/debug-mode.md` -- DEBUG controls traceback rewriting; default hides internals.',
    tags=['config', 'debug', 'meta'],
)


# ---------------------------------------------------------------------------
# Rewritten for 0.8. Each of these had an answer that was correct on the older
# API; the superseded answer is kept as a distractor, since it is exactly what
# an agent working from stale material will pick.
# ---------------------------------------------------------------------------

register_mcq(
    id="mcq_basic_02_trace_no_input",
    name="trace with no input",
    difficulty=Difficulty.BASIC,
    question="What happens with `with model.trace():` — no positional input and no inner `tracer.invoke(...)` block?",
    choices=[
        "It runs the model once on an empty string.",
        "It raises `ValueError: trace() needs an input, or at least one `with tracer.invoke(...)` block`.",
        "It raises a `MissedProviderError` when the dangling mediator is collected.",
        "It silently does nothing and returns None.",
    ],
    correct_index=1,
    explanation="0.8 validates up front. The MissedProviderError path no longer exists.",
    tags=["trace", "errors"],
)

register_mcq(
    id="mcq_basic_05_generate_multi_token",
    name="capture across generated tokens",
    difficulty=Difficulty.BASIC,
    question="You want a hidden state from each of 5 generated tokens, and you also need the final generated ids afterwards. Which is correct?",
    choices=[
        "`for step in tracer.iter[:]:` — the unbounded loop covers every step and trailing code still runs.",
        "`for step in tracer.iter[:5]:` — bound the loop to the number of steps, so code after it still runs.",
        "`with tracer.all():` — the block form applies to all steps.",
        "`tracer.next()` between accesses to advance to the next step.",
    ],
    correct_index=1,
    explanation="Unbounded iter/all unwinds at the over-run step, dropping everything after the loop. `tracer.next()` was removed.",
    tags=["generation", "iteration"],
)

register_mcq(
    id="mcq_intermediate_01_out_of_order",
    name="out of order access",
    difficulty=Difficulty.INTERMEDIATE,
    question="In one trace you read `model.transformer.h[8].output` and then `model.transformer.h[2].output`. What happens?",
    choices=[
        "Both succeed; nnsight reorders requests automatically.",
        "It raises `OutOfOrderError` — the model already ran past layer 2 when you asked for it.",
        "It raises `OutOfOrderError`, which is a subclass of `MissedProviderError`.",
        "It deadlocks and hangs forever.",
    ],
    correct_index=1,
    explanation="0.8 collapsed the two error classes into `OutOfOrderError` alone, and it is raised at the end of the run rather than hanging.",
    tags=["ordering", "errors"],
)

register_mcq(
    id="mcq_intermediate_04_tuple_output",
    name="writing into a tuple output",
    difficulty=Difficulty.INTERMEDIATE,
    question="`model.transformer.h[0].attn.output` is a tuple `(attn_out, ...)`. You want to zero the first element. Which works?",
    choices=[
        "`model.transformer.h[0].attn.output[0] = torch.zeros_like(attn_out)`",
        "`model.transformer.h[0].attn.output[0][:] = 0`",
        "`model.transformer.h[0].attn.output = 0`",
        "`model.transformer.h[0].attn.output.zero_()`",
    ],
    correct_index=1,
    explanation="Item assignment on a tuple raises TypeError. Mutate through the element in place, or rebuild the tuple and assign the whole thing.",
    tags=["modification", "tuples"],
)

register_mcq(
    id="mcq_intermediate_06_unbounded_iter",
    name="code after an unbounded iter",
    difficulty=Difficulty.INTERMEDIATE,
    question="Inside `model.generate(..., max_new_tokens=3)` you loop `for step in tracer.all():` and then, after the loop, call `tracer.result.save()`. What happens to `result`?",
    choices=[
        "It is saved normally once generation ends.",
        "It is never assigned — the unbounded loop unwinds at the over-run step and drops every line after it.",
        "It raises `MissedProviderError` immediately.",
        "It contains only the last generated token.",
    ],
    correct_index=1,
    explanation="Bound the loop (`tracer.iter[:3]`) if you need trailing code to run. Old nnsight bounded `all()` internally; 0.8 does not.",
    tags=["generation", "iteration", "gotcha"],
)

register_mcq(
    id="mcq_intermediate_08_generated_ids",
    name="reading generated ids",
    difficulty=Difficulty.INTERMEDIATE,
    question="Inside `with model.generate('Hi', max_new_tokens=5) as tracer:`, what is the supported way to get the generated token ids?",
    choices=[
        "`model.generator.output.save()` — the canonical accessor.",
        "`tracer.result.save()` — `model.generator.output` still works but is deprecated.",
        "`model.output.logits.save()` — generate returns logits.",
        "`tracer.iter[-1].output.save()`.",
    ],
    correct_index=1,
    explanation="0.8 splits generate (token ids on tracer.result) from pipe (decoded records). generator.output remains only for per-step streamer access.",
    tags=["generation"],
)

register_mcq(
    id="mcq_intermediate_09_backward_access",
    name="module access inside backward",
    difficulty=Difficulty.INTERMEDIATE,
    question="Inside `with metric.backward():` you try to read `model.transformer.h[-1].output`. What happens?",
    choices=[
        "It returns the forward activation, cached from earlier in the trace.",
        "It raises `OutOfOrderError` — the forward pass is over, so capture activations before the backward block.",
        "It returns the gradient of that module.",
        "It silently returns None.",
    ],
    correct_index=1,
    explanation="Only `.grad` on tensors captured during the forward is meaningful inside a backward context.",
    tags=["gradients", "ordering"],
)

register_mcq(
    id="mcq_advanced_01_error_classes",
    name="error class for missed values",
    difficulty=Difficulty.ADVANCED,
    question="A worker is still waiting on a location when the run ends (for example after `tracer.stop()` skipped it). Which exception does 0.8 raise?",
    choices=[
        "`MissedProviderError`, the parent class of `OutOfOrderError`.",
        "`OutOfOrderError` — 0.8 uses one class for both the asked-too-late and never-delivered cases.",
        "`ValueError: value was not provided`.",
        "No exception; the value is silently None.",
    ],
    correct_index=1,
    explanation="`MissedProviderError` and the class split were removed in the 0.8 rewrite.",
    tags=["errors"],
)

register_mcq(
    id="mcq_advanced_03_source_submodule",
    name="recursive source into a submodule",
    difficulty=Difficulty.ADVANCED,
    question="You call `.source` on a source operation whose callee is a `torch.nn.Module` submodule. What happens?",
    choices=[
        "It transparently drills into the submodule's forward.",
        "It raises `SourceNotAvailable` — call `.source` on that submodule's own envoy instead.",
        "It raises `ValueError: Don't call .source on a module`.",
        "It works, but only outside a trace.",
    ],
    correct_index=1,
    explanation="Recursive .source handles plain Python functions; submodules have their own envoy and their own .source.",
    tags=["source"],
)

register_mcq(
    id="mcq_meta_03_mediator_events",
    name="mediator event protocol",
    difficulty=Difficulty.ADVANCED,
    question="Which events make up the worker/model protocol in 0.8's interleaver?",
    choices=[
        "REQUEST, RESPONSE, ACK, FIN.",
        "VALUE, SWAP, SKIP, BARRIER.",
        "VALUE, SWAP, SKIP, BARRIER, END, EXCEPTION.",
        "FORWARD, BACKWARD, GENERATE, CACHE.",
    ],
    correct_index=1,
    explanation="END and EXCEPTION were removed in the greenlet rewrite; workers are greenlets, not threads.",
    tags=["internals"],
)

register_mcq(
    id="mcq_advanced_12_vllm_unsupported",
    name="what vLLM cannot do",
    difficulty=Difficulty.ADVANCED,
    question="Which of these is NOT supported on nnsight's vLLM path?",
    choices=[
        "Reading `model.logits` during generation.",
        "Gradients via `with tensor.backward():`.",
        "Tensor parallelism across several GPUs.",
        "Per-request sampling parameters on `tracer.invoke(...)`.",
    ],
    correct_index=1,
    explanation="No backward through the engine, and no scan or source-tracing on fused kernels. Tensor parallelism is supported and transparent.",
    tags=["vllm"],
)


# ---------------------------------------------------------------------------
# Corrected against the 0.8 source. These three were ported from the 0.7-era
# suite with their old answers intact — the sort of drift this testbed exists to
# catch, found by a reader rather than by the harness. Each keeps the superseded
# description as a distractor.
# ---------------------------------------------------------------------------

register_mcq(
    id="mcq_advanced_04_eproperty",
    name="eproperty stub semantics",
    difficulty=Difficulty.ADVANCED,
    question="What does the body of a method decorated with `@eproperty` do at runtime in nnsight 0.8?",
    choices=[
        "Nothing — the body is never executed for its return value; it exists to carry "
        "pre-setup decorators and donate `__name__`/`__doc__` to the descriptor.",
        "It is the preprocess step: `__get__` parks on the interleaver for the served "
        "value, then calls the body with it and returns whatever the body returns.",
        "It runs once and the result is memoized on the class for the lifetime of the model.",
        "It is scheduled as a coroutine on the worker's event loop.",
    ],
    correct_index=1,
    explanation=(
        "src/nnsight/intervention/eproperty.py __get__: value = Mediator.value(location); "
        "if self._preprocess is not None: value = self._preprocess(obj, value). The module "
        "docstring states 'The decorated stub *is* the preprocess'. `.postprocess` handles "
        "writes and `.transform` maps an edited view back to the model's layout. Option A "
        "describes the pre-0.8 descriptor."
    ),
    tags=["internals", "eproperty"],
)

register_mcq(
    id="mcq_meta_01_pymount_config",
    name="PYMOUNT",
    difficulty=Difficulty.ADVANCED,
    question="What does `CONFIG.APP.PYMOUNT` (default True) control?",
    choices=[
        "Whether the C extension mounts `.save()` onto every Python object, so `value.save()` "
        "works on builtins; with it off, use `nnsight.save(value)`.",
        "Whether the C extension injects both `.save()` and `.stop()` onto every Python object.",
        "Whether model weights are memory-mapped rather than copied into RAM.",
        "Whether the interleaver mounts its hooks eagerly at model load.",
    ],
    correct_index=0,
    explanation=(
        "src/nnsight/__init__.py: `if CONFIG.APP.PYMOUNT: from ._c import mount; "
        "mount(save, 'save')` — one name, `save`. The mount is optional and wrapped in a "
        "try/except, so `nnsight.save(value)` always works regardless."
    ),
    tags=["internals", "config"],
)

register_mcq(
    id="mcq_advanced_07_edit_inplace",
    name="persistent edits",
    difficulty=Difficulty.ADVANCED,
    question="Where does `with model.edit(inplace=True):` put the intervention, and how is it undone?",
    choices=[
        "It rewrites the module's forward; undo by reloading the model.",
        "It appends a compiled Mediator to `envoy._edits`, which is prepended to the "
        "mediators of every later run; `model.clear_edits()` empties that list.",
        "It appends to `_default_mediators`; `model.reset()` clears it.",
        "It stores a hook handle on the interleaver; removing it requires the handle.",
    ],
    correct_index=1,
    explanation=(
        "src/nnsight/intervention/envoy.py: `self._edits: list[Mediator] = []`, and "
        "`clear_edits()` sets `self._edits = []`. Non-inplace `edit()` stores on a shallow "
        "copy instead, so the original envoy stays clean."
    ),
    tags=["internals", "edit"],
)
