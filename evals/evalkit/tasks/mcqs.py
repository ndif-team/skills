"""Multiple-choice questions about nnsight 0.8.

Rewritten from the ported 0.7-era set after measurement showed the questions were
answerable without reading them:

- the keyed answer was the **longest** choice in 29 of 32 questions
- the keyed answer was **B** in 29 of 32 questions

Either heuristic alone scored 91%, which is where the no-resources baseline
landed — so the MCQ half was measuring string length and letter position, not
knowledge. Every question here is written with choices of comparable length and
detail, and the correct position is rotated. `evalkit.audit --mcq-bias` enforces
both, so the property cannot quietly regress.

Four keyed answers were also factually wrong for 0.8 (eproperty's stub body,
PYMOUNT's mounted names, where edits are stored, and what an unsaved local does
after a scan). Each was checked against the source or by running it; the
superseded claim is kept as a distractor, since that is what an agent working
from stale material will pick.
"""

from ..registry import Difficulty, register_mcq

# --- fundamentals ----------------------------------------------------------

register_mcq(
    id="mcq_basic_01_save_required",
    name="save required",
    difficulty=Difficulty.BASIC,
    question=(
        "Inside a function, `with model.trace(...):` assigns `hs = model.transformer.h[0].output` "
        "with no `.save()`. The next line, after the block, returns `hs`. What happens?"
    ),
    choices=[
        "`hs` holds the tensor; assigned names always survive the block.",
        "`hs` raises `OutOfOrderError`, since the model is no longer running.",
        "`hs` is unbound — only marked values are pushed back to the caller.",
        "`hs` is a proxy that re-runs the trace when it is first read.",
    ],
    correct_index=2,
    explanation="The body runs in another frame; push_result returns only saved locals.",
    tags=["save"],
)

register_mcq(
    id="mcq_basic_02_trace_no_input",
    name="trace with no input",
    difficulty=Difficulty.BASIC,
    question="What happens with `with model.trace():` — no positional input and no inner invoke?",
    choices=[
        "A `ValueError` saying trace needs an input or at least one invoke block.",
        "The model runs once on an empty string and returns its logits.",
        "A `MissedProviderError`, raised once the dangling mediator is finally collected.",
        "Nothing at all — the block is skipped and `None` is returned.",
    ],
    correct_index=0,
    explanation="0.8 validates up front; the MissedProviderError path no longer exists.",
    tags=["trace", "errors"],
)

register_mcq(
    id="mcq_basic_03_output_vs_input",
    name="input vs inputs",
    difficulty=Difficulty.BASIC,
    question="How do `module.input` and `module.inputs` differ on an Envoy?",
    choices=[
        "`input` is read-only; `inputs` is the writable form of the same value.",
        "`input` is the previous layer's output; `inputs` is the embedding matrix.",
        "They are aliases kept for backwards compatibility, and both return the same object.",
        "`input` is the first positional argument; `inputs` is the `(args, kwargs)` pair.",
    ],
    correct_index=3,
    explanation="`.input` is a convenience view over `.inputs`, which holds everything.",
    tags=["access"],
)

register_mcq(
    id="mcq_basic_04_default_invoke",
    name="implicit invoke",
    difficulty=Difficulty.BASIC,
    question="`with model.trace('Hello'):` is equivalent to which explicit form?",
    choices=[
        "`with model.trace() as t:` then `with t.invoke('Hello'):` around the same body.",
        "`with model.trace() as t:` then a bare `t.invoke('Hello')` call, no block.",
        "`with model.session('Hello'):` — a session and a trace are interchangeable here.",
        "Nothing — the implicit form is special-cased and has no explicit equivalent.",
    ],
    correct_index=0,
    explanation="A positional input creates one implicit invoke wrapping the whole body.",
    tags=["trace", "invoke"],
)

register_mcq(
    id="mcq_basic_05_generate_multi_token",
    name="per-step capture plus result",
    difficulty=Difficulty.BASIC,
    question=(
        "You need a hidden state from each of 5 generated tokens AND the final ids "
        "afterwards. Which loop form works?"
    ),
    choices=[
        "`with tracer.all():` — the block form covers every step and keeps trailing code.",
        "`tracer.next()` between reads, advancing one step at a time.",
        "`for step in tracer.iter[:]:` — unbounded, and trailing code still runs.",
        "`for step in tracer.iter[:5]:` — bounded, so the line after the loop runs.",
    ],
    correct_index=3,
    explanation="Unbounded iter/all unwinds at the over-run step and drops trailing code.",
    tags=["generation", "iteration"],
)

# --- ordering, batching, modification ---------------------------------------

register_mcq(
    id="mcq_intermediate_01_out_of_order",
    name="out of order access",
    difficulty=Difficulty.INTERMEDIATE,
    question="One trace reads `h[8].output` and then `h[2].output`. What happens?",
    choices=[
        "Both succeed — nnsight quietly reorders the requests to match the forward order.",
        "`OutOfOrderError`: the model had already run past layer 2 when it was asked for.",
        "`OutOfOrderError`, which in 0.8 is a subclass of `MissedProviderError`.",
        "The trace deadlocks and hangs until the process is killed.",
    ],
    correct_index=1,
    explanation="One class covers both cases in 0.8, and it raises rather than hanging.",
    tags=["ordering", "errors"],
)

register_mcq(
    id="mcq_intermediate_02_cross_invoke_barrier",
    name="cross-invoke without a barrier",
    difficulty=Difficulty.INTERMEDIATE,
    question=(
        "Invoke 1 captures `donor = model.transformer.h[5].output`; invoke 2 writes that "
        "same location from `donor`. With no barrier, what happens?"
    ),
    choices=[
        "nnsight inserts a barrier automatically when it sees same-module access.",
        "The invokes run in parallel and the result is nondeterministic.",
        "Invoke 2 raises `NameError` — invoke 1 has not bound `donor` yet.",
        "It works: values propagate across invokes transparently.",
    ],
    correct_index=2,
    explanation="All invoke workers start together; ordering needs tracer.barrier(n).",
    tags=["batching", "barrier"],
)

register_mcq(
    id="mcq_intermediate_03_inplace_vs_replace",
    name="in-place vs replacement",
    difficulty=Difficulty.INTERMEDIATE,
    question="How does `module.output[:] = 0` differ from `module.output = new_tensor`?",
    choices=[
        "Slice assignment is illegal inside a trace, so only whole-value assignment works.",
        "They are identical; both mutate the tensor the model is holding.",
        "Whole-value assignment is ignored unless wrapped in a swap helper.",
        "Slice assignment mutates the existing storage; assignment swaps in a new object.",
    ],
    correct_index=3,
    explanation="Both take effect. Only replacement preserves the autograd graph.",
    tags=["modification"],
)

register_mcq(
    id="mcq_intermediate_04_tuple_output",
    name="writing into a tuple output",
    difficulty=Difficulty.INTERMEDIATE,
    question=(
        "`model.transformer.h[0].attn.output` is a tuple. Which statement zeroes its "
        "first element?"
    ),
    choices=[
        "`attn.output[0] = torch.zeros_like(x)`",
        "`attn.output[0][:] = 0.0`",
        "`attn.output = torch.zeros_like(x)`",
        "`attn.output.zero_(x.shape)`",
    ],
    correct_index=1,
    explanation="Item assignment on a tuple raises TypeError; mutate through the element.",
    tags=["modification", "types"],
)

register_mcq(
    id="mcq_intermediate_05_clone_before_save",
    name="saving before an in-place edit",
    difficulty=Difficulty.INTERMEDIATE,
    question=(
        "You save `before = h[0].output.save()`, then zero that output in place, then save "
        "`after`. What does `before` hold after the trace?"
    ),
    choices=[
        "The original values, because `.save()` snapshots at the moment it is called.",
        "`None`, because only the most recent save of a location survives.",
        "A `RuntimeError`, because the same location was saved twice.",
        "The zeroed values, because it aliases the storage the edit mutated.",
    ],
    correct_index=3,
    explanation="save() marks an object, it does not copy. Use .clone() for a snapshot.",
    tags=["save", "modification"],
)

register_mcq(
    id="mcq_intermediate_06_unbounded_iter",
    name="code after an unbounded loop",
    difficulty=Difficulty.INTERMEDIATE,
    question=(
        "Under `generate(max_new_tokens=3)` you loop `for step in tracer.all():` and then, "
        "after the loop, call `tracer.result.save()`. What happens?"
    ),
    choices=[
        "`result` is saved normally once generation finishes.",
        "`result` is never assigned; the loop unwound and dropped the trailing line.",
        "A `MissedProviderError` is raised as soon as the loop starts.",
        "`result` holds only the very last generated token rather than the whole sequence.",
    ],
    correct_index=1,
    explanation="Bound the loop if you need trailing code. Old nnsight bounded all() itself.",
    tags=["generation", "iteration", "gotcha"],
)

register_mcq(
    id="mcq_intermediate_07_all_is_iter",
    name="all versus iter",
    difficulty=Difficulty.INTERMEDIATE,
    question="What is the relationship between `tracer.all()` and `tracer.iter[:]`?",
    choices=[
        "`all()` runs the iterations in parallel, whereas `iter[:]` runs them in sequence.",
        "`all()` is a deprecated alias that now resolves to `tracer.iter[0]`.",
        "`all()` returns `iter[:]` — the same unbounded iterator, same trailing-code trap.",
        "`all()` includes the prefill pass, whereas `iter[:]` starts after it.",
    ],
    correct_index=2,
    explanation="They are the same object, and share the same failure mode.",
    tags=["iteration"],
)

register_mcq(
    id="mcq_intermediate_08_generated_ids",
    name="reading generated ids",
    difficulty=Difficulty.INTERMEDIATE,
    question="Inside `model.generate('Hi', max_new_tokens=5)`, how do you get the generated ids?",
    choices=[
        "`model.output.logits.save()` — generate hands back logits rather than token ids.",
        "`model.generator.output.save()` — the supported accessor in 0.8.",
        "`tracer.iter[-1].output.save()` — index the final step.",
        "`tracer.result.save()` — `model.generator.output` works but is deprecated.",
    ],
    correct_index=3,
    explanation="0.8 splits generate (ids on tracer.result) from pipe (decoded records).",
    tags=["generation"],
)

register_mcq(
    id="mcq_intermediate_09_backward_access",
    name="module access inside backward",
    difficulty=Difficulty.INTERMEDIATE,
    question="Inside `with metric.backward():` you read `model.transformer.h[-1].output`. What happens?",
    choices=[
        "It returns the forward activation, which was cached earlier in the same trace.",
        "It returns that module's gradient, the same as reading `.grad`.",
        "It raises `OutOfOrderError` — the forward is over; capture before the block.",
        "It returns `None` silently, and the backward pass continues.",
    ],
    correct_index=2,
    explanation="Only .grad on tensors captured in the forward is meaningful there.",
    tags=["gradients", "ordering"],
)

register_mcq(
    id="mcq_intermediate_10_grad_reverse_order",
    name="gradient access order",
    difficulty=Difficulty.INTERMEDIATE,
    question=(
        "You captured `h3` from layer 3 and `h10` from layer 10 during the forward. In what "
        "order must you read their `.grad` inside the backward block?"
    ),
    choices=[
        "`h3` then `h10`, mirroring the order they were produced in.",
        "`h10` then `h3`, because gradients arrive in reverse of the forward.",
        "Either order — gradients are buffered until the block exits.",
        "Neither; wrap both in `tracer.barrier(2)` since there is no inherent order.",
    ],
    correct_index=1,
    explanation="Reads park on the autograd hook, so they follow the backward order.",
    tags=["gradients"],
)

# --- advanced --------------------------------------------------------------

register_mcq(
    id="mcq_advanced_01_error_classes",
    name="error class for missed values",
    difficulty=Difficulty.ADVANCED,
    question=(
        "A worker still waits on a location when the run ends — for instance after "
        "`tracer.stop()` skipped it. What does 0.8 raise?"
    ),
    choices=[
        "`MissedProviderError`, the parent class from which `OutOfOrderError` derives.",
        "`ValueError` reporting that the value was never provided.",
        "`OutOfOrderError` — one class covers asked-too-late and never-delivered.",
        "Nothing; the value is left as `None` and the run completes.",
    ],
    correct_index=2,
    explanation="The class split was removed in the 0.8 rewrite.",
    tags=["errors"],
)

register_mcq(
    id="mcq_advanced_02_invoke_during_execution",
    name="invoking while running",
    difficulty=Difficulty.ADVANCED,
    question="Which pattern raises `Cannot invoke while the model is already running.`?",
    choices=[
        "Calling `model.trace(...)` twice in a row, with no overlap between the two runs.",
        "Opening a `tracer.invoke(...)` block nested inside another invoke or an iter loop.",
        "Calling `tracer.barrier(2)` after the model has started executing.",
        "Marking the same tensor with `.save()` more than once in a single trace.",
    ],
    correct_index=1,
    explanation="Invokes define the batch up front; they cannot be opened mid-run.",
    tags=["batching", "errors"],
)

register_mcq(
    id="mcq_advanced_03_source_submodule",
    name="recursive source into a submodule",
    difficulty=Difficulty.ADVANCED,
    question="You call `.source` on a source operation whose callee is a `torch.nn.Module`. What happens?",
    choices=[
        "`SourceNotAvailable` — use that submodule's own envoy and its `.source`.",
        "It drills into the submodule's forward and exposes each of its operations.",
        "`ValueError` telling you not to call `.source` on a module.",
        "It succeeds, but only when called outside an active trace.",
    ],
    correct_index=0,
    explanation="Recursive .source handles plain functions; submodules have their own envoy.",
    tags=["source"],
)

register_mcq(
    id="mcq_advanced_04_eproperty",
    name="eproperty stub semantics",
    difficulty=Difficulty.ADVANCED,
    question="At runtime, what does the body of a method decorated with `@eproperty` do?",
    choices=[
        "Nothing — it is never run; it only carries decorators and donates `__name__`.",
        "It runs once and the result is memoized on the class for the model's lifetime.",
        "It is scheduled as a coroutine on the interleaver's event loop.",
        "It preprocesses: `__get__` parks for the value, calls the body, returns its result.",
    ],
    correct_index=3,
    explanation=(
        "eproperty.py __get__: value = Mediator.value(location); "
        "if self._preprocess is not None: value = self._preprocess(obj, value). "
        "Option A describes the pre-0.8 descriptor."
    ),
    tags=["internals", "eproperty"],
)

register_mcq(
    id="mcq_advanced_05_envoy_call_hook_default",
    name="calling a module inside a trace",
    difficulty=Difficulty.ADVANCED,
    question=(
        "Inside a trace, `model.sae(hidden)` on an attached module runs which path, and how "
        "do its own hooks fire?"
    ),
    choices=[
        "Through `__call__`, with hooks always firing; there is no supported way to bypass.",
        "Through `forward(...)`, bypassing hooks; pass `hook=True` to route through `__call__`.",
        "Through `__call__` by default; pass `hook=False` to bypass the hooks.",
        "Attached modules cannot fire hooks at all without a custom eproperty.",
    ],
    correct_index=1,
    explanation="Envoy.__call__ signature: `def __call__(self, *args, hook: bool = False, **kwargs)`.",
    tags=["modules"],
)

register_mcq(
    id="mcq_advanced_06_scan_save_required",
    name="unsaved values after scan",
    difficulty=Difficulty.ADVANCED,
    question=(
        "Inside a function, `with model.scan('Hi'):` assigns `dim = h[0].output.shape[-1]` "
        "with no save. The function then returns `dim`. What happens?"
    ),
    choices=[
        "It returns the int; scan is exempt from the save filter since nothing executes.",
        "It returns `0`, because fake tensors report zero-length shapes.",
        "It raises `UnboundLocalError`; scan filters unsaved locals like any trace.",
        "It returns a symbolic shape object rather than a Python int.",
    ],
    correct_index=2,
    explanation=(
        "scan is a tracing context and gates saves the same way. Note the scope: at module "
        "level the name may survive, which is why this asks about a function."
    ),
    tags=["scan", "save"],
)

register_mcq(
    id="mcq_advanced_07_edit_inplace",
    name="persistent edits",
    difficulty=Difficulty.ADVANCED,
    question="Where does `model.edit(inplace=True)` store the intervention, and how is it undone?",
    choices=[
        "It rewrites the module's forward in place; only reloading the model undoes it.",
        "It appends to `_default_mediators`, which `model.reset()` clears.",
        "It registers a hook handle that you must keep in order to remove it.",
        "It appends a Mediator to `envoy._edits`, which `model.clear_edits()` empties.",
    ],
    correct_index=3,
    explanation="envoy.py: `self._edits: list[Mediator] = []`; clear_edits() resets it.",
    tags=["internals", "edit"],
)

register_mcq(
    id="mcq_advanced_08_session_bundling",
    name="where remote=True goes",
    difficulty=Difficulty.ADVANCED,
    question="Running several traces against a remote model, where does `remote=True` belong?",
    choices=[
        "On `model.session(...)`; the inner traces inherit it as one request.",
        "On each `model.trace(...)` so every one queues as its own request.",
        "On a `model.dispatch(remote=True)` call once at the top of the script.",
        "On both the session and each inner trace, which makes the job more reliable.",
    ],
    correct_index=0,
    explanation="One session is one job; values also flow between traces without downloads.",
    tags=["remote", "session"],
)

register_mcq(
    id="mcq_advanced_09_remote_save_transmission",
    name="client lists on a remote trace",
    difficulty=Difficulty.ADVANCED,
    question=(
        "On a remote trace, a list created in client code and appended to inside the trace "
        "comes back empty. Why?"
    ),
    choices=[
        "Remote traces do not implement `.append` on transmitted objects.",
        "The list lives client-side; the appends happen on the server and are discarded.",
        "The appended value was not `.save()`d, so the list received nothing.",
        "List mutations are stripped during serialization; a dict would have worked here.",
    ],
    correct_index=1,
    explanation="Build the container inside the trace and save the container itself.",
    tags=["remote", "save"],
)

register_mcq(
    id="mcq_advanced_10_blocking_false",
    name="non-blocking remote jobs",
    difficulty=Difficulty.ADVANCED,
    question="With `model.trace('Hi', remote=True, blocking=False)`, how do you get the result?",
    choices=[
        "Re-enter the same `with` block later; it resumes and yields the values.",
        "The tracer object turns into the result tensor once the job completes.",
        "Call `model.fetch(tracer.id)`; the backend object is not exposed to calling code.",
        "Poll `tracer.backend()`: `None` while pending, then the dict of saved values.",
    ],
    correct_index=3,
    explanation="Saves are not pushed into your locals here — read them from the dict.",
    tags=["remote"],
)

register_mcq(
    id="mcq_advanced_11_vllm_logits_samples",
    name="vLLM logits and samples",
    difficulty=Difficulty.ADVANCED,
    question="On `nnsight.modeling.vllm.VLLM`, what are `model.logits` and `model.samples`?",
    choices=[
        "Methods that you call to fetch the current tensors, as in `model.logits()`.",
        "Aliases for `model.lm_head.output` and `model.generator.output`.",
        "Hookable values: this step's pre-sampling logits, and the ids the sampler drew.",
        "Internal engine flags used for debugging, not intended for user code.",
    ],
    correct_index=2,
    explanation="Both are readable and assignable; tracer.result is not served on vLLM.",
    tags=["vllm"],
)

register_mcq(
    id="mcq_advanced_12_vllm_unsupported",
    name="what vLLM cannot do",
    difficulty=Difficulty.ADVANCED,
    question="Which of these is NOT supported on nnsight's vLLM path?",
    choices=[
        "Reading `model.logits` at each generated decode step.",
        "Gradients through a `with tensor.backward():` block.",
        "Tensor parallelism sharded across several GPUs.",
        "Per-request sampling parameters on `tracer.invoke(...)`.",
    ],
    correct_index=1,
    explanation="No backward, no scan, no source on fused kernels. TP is transparent.",
    tags=["vllm"],
)

# --- internals and configuration -------------------------------------------

register_mcq(
    id="mcq_meta_01_pymount_config",
    name="PYMOUNT",
    difficulty=Difficulty.ADVANCED,
    question="What does `CONFIG.APP.PYMOUNT` (default `True`) control?",
    choices=[
        "Whether the C extension mounts `.save()` onto every object, builtins included.",
        "Whether the extension injects both `.save()` and `.stop()` onto every object type.",
        "Whether model weights are memory-mapped instead of copied into host RAM.",
        "Whether the interleaver installs its forward hooks eagerly at model load.",
    ],
    correct_index=0,
    explanation="__init__.py: `mount(save, 'save')` — one name. nnsight.save() always works.",
    tags=["internals", "config"],
)

register_mcq(
    id="mcq_meta_02_barrier_n",
    name="barrier count",
    difficulty=Difficulty.ADVANCED,
    question="What does `n` mean in `tracer.barrier(n)`?",
    choices=[
        "The number of generation steps that the barrier holds execution open for.",
        "The number of attention heads whose values are synchronized.",
        "The number of seconds to wait before the barrier times out.",
        "The number of blocks that must call it before any of them proceed.",
    ],
    correct_index=3,
    explanation="Set it wrong and the run ends with 'a barrier was never reached'.",
    tags=["barrier"],
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
    explanation="END and EXCEPTION went away with the greenlet rewrite.",
    tags=["internals"],
)

register_mcq(
    id="mcq_meta_04_envoy_call_logitlens",
    name="why logit lens skips hooks",
    difficulty=Difficulty.ADVANCED,
    question=(
        "In `logits = model.lm_head(model.transformer.ln_f(hs))` inside a trace, why do "
        "`lm_head` and `ln_f` not trigger their own hooks?"
    ),
    choices=[
        "Hooks are disabled for every module that is invoked from inside an active trace.",
        "The hooks do fire, but their results are discarded without being served.",
        "`Envoy.__call__` defaults to `hook=False`, so it runs `forward` directly.",
        "Those two modules are special-cased by the transformers model wrapper.",
    ],
    correct_index=2,
    explanation="That is what lets you apply a module out of order without deadlocking.",
    tags=["modules"],
)

register_mcq(
    id="mcq_meta_05_debug_mode",
    name="debug mode",
    difficulty=Difficulty.ADVANCED,
    question="What changes when you set `CONFIG.APP.DEBUG = True`?",
    choices=[
        "The model runs under `torch.no_grad`, so any backward context is disabled.",
        "Tracebacks keep nnsight's internal frames instead of being filtered to your code.",
        "Every `.save()` also prints the value it marked to stderr.",
        "Remote traces are redirected to run locally for easier stepping.",
    ],
    correct_index=1,
    explanation="It also makes remote runs log payload sizes and every status transition.",
    tags=["config", "debugging"],
)
