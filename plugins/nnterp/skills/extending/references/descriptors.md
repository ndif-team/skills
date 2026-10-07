# The Descriptors

Every standard value is an `EProperty` (`nnterp/components/eproperty.py`), nnsight's
`eproperty` (a `property` over a location) whose key is a *path* from the host envoy and
which can say, before anything runs, whether this checkpoint has the value and why not;
or one of its kin: `DerivedEProperty` (computed from other values), `TokenEProperty`
(`nnterp/components/tokens.py`: a tensor the model holds flat over tokens) and
`unavailable("...")` (a marker carrying only a reason). The root's sizes and
`project_on_vocab` are a different, plain descriptor (`StandardizedProperty` /
`StandardizedCapability`). This file is each one's arguments, what it serves, how a
write reaches the model, and its trap. Blocks run on subclasses of GPT-2's family
classes installed through `envoys=`; the `TokenEProperty` block, which needs a mixture,
was verified on `hf-internal-testing/tiny-random-MixtralForCausalLM`. The nnterp repo's
pages: `docs/extending/overriding-values.md`, `docs/extending/custom-values.md`,
`docs/developing/eproperty-internals.md`.

<!-- test: setup -->
```python
import torch
from jaxtyping import Float
from torch import Tensor

import nnterp
from nnterp import DerivedEProperty, EProperty, StandardizedTransformer, Unavailable, unavailable
from nnterp.components import Pattern, Residual, interface_reason, needs_eager
from nnterp.families import gpt2
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2MLP   # after import nnterp

prompt = "The Eiffel Tower is in the city of"
```

## Which one

| The value is | Write | Location it serves | Writes |
|---|---|---|---|
| a view over the host's own boundary (`layer_output`, `mlp_output`, `logits`) | `EProperty("output")` | the same location as `.output` | `postprocess` then swap; `transform` for a reshaping preprocess |
| an operation inside the host's forward (`attention_probabilities`) | `EProperty("source.<op>.output")`, `select=` for one element | `{path}.source.<op>.output` after a drill | `postprocess`, then the operation's own descriptor; `transform` available |
| produced by another module (Gemma-2's `attention_output`, the root's `token_embeddings`) | `EProperty("../post_attention_layernorm.output")`, `EProperty("embed_tokens.output")` | that module's location | as `"output"` |
| an operation inside the *parent's* forward (Llama 4's `mlp_output`) | `EProperty("../source.<op>.output")`, plus `sourced = True` on the parent's envoy class | `{parent}.source.<op>.output`; the parent's forward is instrumented at build | as above |
| computed from other values (`states`) | `DerivedEProperty(compute)` | none | refused |
| held flat over tokens, `[batch*seq, ...]` (a mixture's routing) | `TokenEProperty(...)`, same arguments as `EProperty` | as `EProperty`, served as this invoke's `[batch, seq, ...]` view | spliced back into the flat tensor |
| something the family does not have | `unavailable("reason")` | none | refused with the reason |
| a root size (`num_kv_heads`) or the root's `project_on_vocab` | a module-level function in the family (`StandardizedProperty` / `StandardizedCapability`, below) | none: plain Python | no setter |
| one block's own size (`self_attn.head_dim`, `mlp.intermediate_size`) | a plain `@property` on the family's envoy subclass | none | no setter |

## `EProperty`

`EProperty(key=None, description=None, unavailable=None, select=None)` decorates a
*preprocess*: a function of `(self, value)` mapping the served value to what the user
reads. `key` is where the value lives, as a path from the host envoy; it defaults to the
stub's name. `description` is what the repr prints. `unavailable` is a reason string or
a function of the envoy returning one or `None`, checked on every read and write.
`select` picks one element of the served value.

### The path

Dotted segments ending in `output`, `input` or `inputs`:

- **`"output"`, `"input"`, `"inputs"`**: the host's own. `output` is what the module
  returned; `inputs` the `(args, kwargs)` it was called with; `input` its first argument.
- **`"<child>.output"`**: a child module, by native name or alias
  (`"embed_tokens.output"` on the root is `token_embeddings`, reaching `transformer.wte`
  on GPT-2).
- **`"../"`**, repeatable: the parent module. A `../` path is not walked through envoys:
  the location is the host's native path minus one segment per `../`, plus the rest as
  written, so what follows must be **native names** (`"../ln_2.output"` on a GPT-2
  attention). An alias there (`"../post_attention_layernorm.output"`) builds a location
  the model never serves, and the read fails as `OutOfOrderError` (below). Gemma-2's
  override works because its native name is the standard one.
- **A leading `/`**: the model's root. The rest is walked down from `envoy.root` like a
  path below the host, aliases and `source` included, for a value that lives far from its
  host: `Vision.image_token_mask` is keyed `"/inputs"` (the model's inputs, read from the
  tower), `"/norm.output"` on a GPT-2 block is the final norm's output, and
  `"/projector.output"` from a tower is the projector's. A `../` right after it is refused.
- **`"source.<op>.output"`**: an operation inside the current module's forward,
  instrumenting it for this run. A `source` after an operation drills into that call's
  callee: `"source.attention_interface_1.source.nn_functional_softmax_0.output"`. After
  `../` it is the parent's forward: `"../source.hidden_states_view_0.output"`.
  [finding-source-ops.md](finding-source-ops.md) is how to read an op's name.
- **A function of the host** returning such a path, run once per access inside the
  trace, for a forward that branches (Falcon's `by_alibi`, a `RecurrentMixer`'s
  `kernel("inputs")`; below).

A value's code reaches beyond its host through nnsight's `Envoy.parent` and `Envoy.root`:
`parent` is the envoy of the native parent module (`model.layers[3].parent` is
`model.transformer.h` on GPT-2, `model.vision.parent` is the wrapper's inner `model` on
Llava), and `root` is the top of the parent links, the `StandardizedTransformer`. The
vision values read the wrapper's config, processor and `projector` through `self.root`.

`select` applies to the last segment: with `inputs` an int is a positional argument and a
str a keyword; with `output` an int indexes the returned tuple; `input` needs none. A
write repacks the element into the current value, so assigning one argument of a call
replaces just that argument.

```python
class Mlp(gpt2.Mlp):
    @EProperty("output", description="The MLP output at the last position, [batch, hidden]")
    def last_position(self, value) -> Float[Tensor, "batch hidden"]:    # no named layout has this shape
        return value[:, -1]


print(Mlp.last_position.key, Mlp.last_position.name, Mlp.last_position.dims)

model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager", envoys={GPT2MLP: Mlp})
mlp = model.layers[0].mlp

with model.trace(prompt):
    last = mlp.last_position.save()
    whole = mlp.mlp_output.save()

assert torch.equal(last, whole[:, -1])
assert isinstance(last, Mlp.last_position.layout)      # the annotation checks rank and dtype
print(mlp.support())
```

```
output last_position ('batch', 'hidden')
{'mlp_output': None, 'last_position': None}
```

- **`layout` and `dims`** are read off the stub's return annotation, and `layout` is
  that annotation itself. Every shipped value is annotated with a named `jaxtyping`
  alias defined beside the envoy that serves it: 28 exported by `nnterp.components`
  (`Residual`, `Streams`, `Queries`, `Keys`, `Values`, `Pattern`, `HeadOutputs`,
  `RouterLogits`, `ExpertWeights`, `ExpertIndices`, `ExpertOutputs`, `State`, `States`,
  `Gates`, the `LinearQK`/`Scan*`/`SSD*` recurrent-mixer families, ...) plus `Logits`,
  `NextTokenProbs`, `Tokens` in `nnterp.standardized`. So
  `Attention.attention_keys.layout is Keys`, and a family's override that writes
  `-> Keys` cannot drift from the base. The repr prints each value as
  `(name) -> Layout [axes]: description`. A new shape takes an inline type, as
  `last_position` does: `Float[Tensor, "batch hidden"]` gives
  `dims == ("batch", "hidden")` and `isinstance(tensor, value.layout)`. The family suite
  checks every value's tensor against its annotation and axis sizes; without one
  `layout` is `None` and the suite skips it. A `State | None` annotation is accepted;
  the non-`None` member is the layout.
- **A slice or a copy is a view the model does not see.** `last_position` above reads
  fine; an in-place edit to it needs a `@last_position.transform` to land, and an
  assignment needs a `@last_position.postprocess` that rebuilds the served shape. The
  base boundary values pair `first_tensor` (preprocess) with `rewrap` (postprocess) so a
  tuple module reads as a tensor and an assignment puts it back in its tuple.
- **`"inputs"` serves the raw `(args, kwargs)` pair; `"input"` the first argument.** The
  root's `input_ids` is `EProperty(key="inputs")` reading `value[1]["input_ids"]` and its
  postprocess repacks `(args, {**kwargs, "input_ids": value})`. Destructure on read,
  repack on write, or name the element with `select=` and let the descriptor repack.
- **`__set_name__`** fills `name` and `key` from the class body when the descriptor is
  never called on a stub, which is how a bare `unavailable("...")` marker learns its
  name. A descriptor subclass that overrides `__set_name__` must keep that.
- **`transform(self, value, raw)`** is the write-back of a reshaping preprocess: it fires
  once on the model side after the read and splices its result in like a swap; `raw` is
  the value as served, so a module returning a tuple can be rebuilt around the edited
  element. It works at every path, a `source` one included. Falcon's `mlp_output` is
  the shipped use ([family-module-recipe.md](family-module-recipe.md)).

## `unavailable(...)` and `unavailable=`

A marker in the class body replaces the inherited descriptor, keeps the name in the tree
and the repr, makes `support()` report the reason, and makes any access raise
`nnterp.Unavailable` before the model runs. A predicate is evaluated on the instance, so
the checkpoint's config decides:

```python
def scaled_reason(self):
    if self._module.config.scale_attn_by_inverse_layer_idx:
        return "this checkpoint sets scale_attn_by_inverse_layer_idx"
    return needs_eager(self)                                  # None under eager, the sdpa reason otherwise


class Attention(gpt2.Attention):
    sink_column = unavailable("GPT-2 has no attention sink")  # a marker: never called on a stub

    @EProperty("source.attention_interface_1.source.attn_weights_0.output",
               description="q @ k^T * scale before the mask, [batch, heads, query, key]",
               unavailable=scaled_reason)
    def raw_scores(self, value) -> Pattern:
        return value


assert Attention.sink_column.name == Attention.sink_column.key == "sink_column"   # from __set_name__

model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager", envoys={GPT2Attention: Attention})
attn = model.layers[0].self_attn
print(attn.support()["sink_column"], "|", attn.support()["raw_scores"])
print("(sink_column): Unavailable: GPT-2 has no attention sink" in repr(attn))

config = attn._module.config
config.scale_attn_by_inverse_layer_idx = True
try:
    print(attn.support()["raw_scores"])                    # the predicate reads the live config
finally:
    config.scale_attn_by_inverse_layer_idx = False

with model.trace(prompt):
    raw = attn.raw_scores.save()
    scores = attn.attention_scores.save()

assert torch.equal(raw.tril(), scores.tril()) and (raw.triu(1) != scores.triu(1)).any()   # the mask is the difference
```

```
GPT-2 has no attention sink | None
True
this checkpoint sets scale_attn_by_inverse_layer_idx
```

`hasattr` does not answer `False` for an unavailable value; it raises:

<!-- test: expect-error Unavailable -->
```python
hasattr(attn, "sink_column")
# Unavailable: model.transformer.h.0.attn.sink_column is not available: GPT-2 has no attention sink
```

Only `AttributeError` counts as absence in Python, and `Unavailable` is a `RuntimeError`
on purpose: an `AttributeError` from a descriptor is rewritten by `Envoy.__getattr__`
into "no attribute" and the reason is lost. Outside a trace, `hasattr` on an *available*
value raises too, never answers `False`:

```python
for name in ("attention_output", "attention_queries", "attention_probabilities"):
    try:
        hasattr(attn, name)
    except Exception as error:
        print(name, type(error).__name__)
```

```
attention_output ValueError
attention_queries ValueError
attention_probabilities SourceNotAvailable
```

A boundary value or one `source` level deep is nnsight's `ValueError: Cannot access ...
outside of interleaving`; a value inside a call (`attention_probabilities`) needs the
live callee and raises `SourceNotAvailable: recursive .source is only available inside
a trace`. Ask `support()`.

A predicate that itself raises `AttributeError` (a typo in a config attribute) is
re-raised on a read as a `RuntimeError` naming the check, so the bug does not hide
behind "no attribute `<value>`". `support()` calls the predicate directly and lets the
`AttributeError` through:

```python
class Typo(gpt2.Attention):
    @EProperty("output", description="a predicate with a typo", unavailable=lambda self: self._module.config.no_such_flag)
    def typo(self, value):
        return value


typo_model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager", envoys={GPT2Attention: Typo})
try:
    typo_model.layers[0].self_attn.typo
except RuntimeError as error:
    assert not isinstance(error, Unavailable)
    print(error)
```

```
the availability check of model.transformer.h.0.attn.typo failed: 'GPT2Config' object has no attribute 'no_such_flag'
```

The shipped predicates: `needs_eager` (the module's `config._attn_implementation` is not
`"eager"`), `interface_reason` (calls `self.off_interface()`, which the base `Attention`
defines as `needs_eager` and a family may override once for every interface value), and
`NOT_ON_INTERFACE`, the reason string for an interface value a family with its own
arithmetic has not mapped. On GPT-2 `self._module.config` exists on the attention module
and not on `GPT2MLP`; a predicate on another host reads the config from where that
module keeps it.

## Inside the forward: a `source` segment

A key with a `source` segment locates the value at an operation inside a forward. The
op's name is its path under the module's `.source`, with `.source.` between a call and
an operation inside it; the last segment says which of the operation's served values
this is: `output` (what the call returned, or what a binding bound), `input` (the call's
first argument; assigning replaces it and keeps the rest), `inputs` (the full
`(args, kwargs)`). `select` picks one element: with `inputs` an int is a positional
argument and a str a keyword (`select="g"` for a kernel's `g=` argument); with `output`
an int indexes the returned tuple. A write repacks that one element into the current
value, so assigning `attention_keys` (`"source.attention_interface_1.inputs"`,
`select=2`) replaces just the keys.

```python
class Attention(gpt2.Attention):
    @EProperty("source.attention_interface_1.source.nn_functional_softmax_0.output",
               description="The softmax output before the dropout, [batch, heads, query, key]",
               unavailable=interface_reason)
    def attention_softmax(self, value) -> Pattern:
        return value

    @EProperty("source.attention_interface_1.output", select=1,
               description="The weights the interface returns beside the head outputs",
               unavailable=interface_reason)
    def returned_weights(self, value) -> Pattern:
        return value

    @EProperty("source.attention_interface_1.inputs", select="scaling",
               description="The scale the interface multiplies q @ k^T by; a keyword argument",
               unavailable=interface_reason)
    def scaling(self, value) -> float:
        return value

    @EProperty("source.no_such_op_0.output", description="an operation this forward does not have")
    def broken(self, value):
        return value


print(Attention.attention_softmax.key)
assert Attention.attention_softmax.layout is Attention.attention_probabilities.layout is Pattern

model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager", envoys={GPT2Attention: Attention})
attn = model.layers[0].self_attn

with model.trace(prompt):                       # forward order: the call's inputs, inside the call, the call's output
    scale = attn.scaling.save()
    soft = attn.attention_softmax.save()
    probs = attn.attention_probabilities.save()
    weights = attn.returned_weights.save()
    clean = model.logits.save()

assert torch.equal(soft, probs) and torch.equal(weights, probs)    # eval, dropout p=0: one tensor, three names
assert scale == model.head_dim ** -0.5

with model.trace(prompt):
    soft = attn.attention_softmax
    uniform = torch.ones_like(soft).tril()      # built from the served tensor: same device, same dtype
    attn.attention_softmax = uniform / uniform.sum(-1, keepdim=True)
    edited = model.logits.save()

assert not torch.equal(clean, edited)          # the write reached the model

with model.trace(prompt):
    attn.scaling = 0.0                          # a keyword write: the call is repacked with scaling=0.0
    flat = attn.attention_softmax.save()

n = flat.shape[-1]
assert torch.allclose(flat, torch.ones_like(flat).tril() / torch.arange(1, n + 1, device=flat.device).view(-1, 1))   # every row uniform over its causal keys
```

```
source.attention_interface_1.source.nn_functional_softmax_0.output
```

An operation the run does not have raises `SourceNotAvailable` at the read, naming the
value, its path and every operation the module has:

<!-- test: expect-error SourceNotAvailable -->
```python
with model.trace(prompt):
    attn.broken.save()
# SourceNotAvailable: model.transformer.h.0.attn.broken reads 'source.no_such_op_0.output', which this run does
# not have: 'model.transformer.h.0.attn.source' has no operation 'no_such_op_0'; available: is_cross_attention_0, ...
# The forward took a path this family's toolkit does not expect.
```

What the descriptor does on every read and write: walk the path, instrumenting the
module's forward at the first `source` (once; the module stays instrumented), drilling
into each call at a later one (the callee is resolved from the live call, so this needs
a running trace, and nnsight clears what it built at the start of the next run), then
read or write the operation's own `.output` / `.input` / `.inputs` descriptor.
Consequences:

- **The element is the object the call holds.** In-place edits land with nothing more;
  `preprocess`, `postprocess` and `transform` all fire as at any other path. A
  transposed view (`seq_first`) keeps in-place edits landing because a transpose shares
  storage; a preprocess that copies needs a `transform` to hand edits back, as Falcon's
  `mlp_output` does at `"output"`.
- **`unavailable=` is yours to state.** A `source` value on an interface op without
  `unavailable=interface_reason` raises `SourceNotAvailable` at read time under `sdpa`
  instead of reporting in `support()`.
- **Read order.** Values in one trace follow the forward: the call's `inputs`
  (`attention_queries`) before an operation inside it (`attention_probabilities`), and
  a module's interior before its boundary. Out of order inside a call raises
  `OutOfOrderError` naming the call's `.fn`, the location the drill waits on, not the
  value (`layer_output` then `attention_probabilities` is
  `'...attn.source.attention_interface_1.fn.i0' was requested but the model already ran
  past it`). `support()` cannot know an op moved; only a trace can.
- **A branching forward.** The key may be a function of the envoy returning the path,
  evaluated once per read or write (a second evaluation after a pinned read would run
  with the `tracer.iter` pin relaxed). A `RecurrentMixer` names whichever kernel fires
  on this call with `kernel(attribute)`; `KERNEL` reads the forward's own branch
  bindings and caches the choice once per module call with `per_call`:

<!-- test: skip -->
```python
# nnterp/components/recurrent.py
def kernel(attribute: str) -> Callable[[Envoy], str]:
    """A key at whichever kernel fires on this call: ``source.<KERNEL>.<attribute>``."""

    def locate(envoy: Envoy) -> str:
        return f"source.{type(envoy).KERNEL(envoy)}.{attribute}"

    locate.__name__ = f"kernel.{attribute}"
    return locate


# nnterp/components/linear_attention.py
class LinearAttention(RecurrentMixer):
    #: The call a prompt runs through: ``torch_chunk_gated_delta_rule(query, key, value, g=, beta=, initial_state=, ...)``.
    CHUNK_KERNEL = "torch_chunk_gated_delta_rule_0"
    #: The call each decode step runs through, with the same arguments.
    RECURRENT_KERNEL = "torch_recurrent_gated_delta_rule_0"
    #: Inside the token-by-token kernel, the binding of the state after each token's update.
    STATE_OP = "last_recurrent_state_3"

    @EProperty(kernel("inputs"), select=0, description="The queries entering the delta rule", unavailable=needs_torch_kernels)
    def attention_queries(self, value: torch.Tensor) -> LinearQK:
        return value
```

A new recurrent mixer subclasses `RecurrentMixer` (or `LinearAttention`, `SelectiveScan`,
`StateSpace`), sets the three constants and declares its values at `kernel(...)`;
`KERNEL`, the per-token state and kernel routing come with the base
(`docs/developing/recurrent-mixer-internals.md`). `per_call(envoy, key, compute)` is the
general form: `compute()` once per module call, cached on the worker. `pinned(n)` pins
the worker's next read to occurrence `n` as `tracer.iter[n]` does (`None` relaxes it).
Both are in `nnterp.components`. A function key's `.key` is `<name>` of the function
(`<kernel.inputs>`); the path it resolves to for an envoy is `path(envoy)`.

## Another module: a relative key

A key that names a module serves a value that module produces. Below the host, the
first segment resolves the way attribute access does, aliases included; a leading `../`
steps to the parent by name arithmetic, so what follows it is native.

```python
class Layer(gpt2.Layer):
    @EProperty("post_attention_layernorm.input", description="The residual stream after the attention sublayer")
    def mid_stream(self, value) -> Residual:       # below the host: the alias resolves
        return value                               # .input is the call's first argument


class Attention(gpt2.Attention):
    @EProperty("../ln_2.output", description="The sibling norm's output: what the MLP reads")
    def next_norm(self, value) -> Residual:        # ../ steps to the block; ln_2 is the native name
        return value

    @EProperty("../post_attention_layernorm.output", description="The same norm, through the alias")
    def next_norm_alias(self, value) -> Residual:  # a location nothing serves: see below
        return value


model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager",
                                envoys={GPT2Block: Layer, GPT2Attention: Attention})
block = model.layers[0]

with model.trace(prompt):                          # forward order: block input, attn, ln_2's input, ln_2's output, mlp's input
    x = block.input.save()
    attn_out = block.self_attn.attention_output.save()
    mid = block.mid_stream.save()
    normed = block.self_attn.next_norm.save()
    mlp_in = block.mlp.input.save()

assert torch.equal(mid, x + attn_out) and torch.equal(normed, mlp_in)
print(Attention.next_norm_alias.path(block.self_attn))
print("mid_stream" in model.support(), "self_attn.next_norm" in model.support(layer=0))
```

```
../post_attention_layernorm.output
True True
```

The alias after `../` is joined into the location string as written,
`model.transformer.h.0.post_attention_layernorm.output`, which the model never serves
(the native module is `ln_2`), so the read waits until the run ends and fails as an
ordering error:

<!-- test: expect-error OutOfOrderError -->
```python
with model.trace(prompt):
    alias = block.self_attn.next_norm_alias.save()
# OutOfOrderError: 'model.transformer.h.0.post_attention_layernorm.output.i0' was requested but the model already ran past it
```

The value is served at that location, so in-place edits reach the model with nothing
more, and `transform` is available as at `"output"`. `model.support()` walks the tree, so
both custom values are listed, the block's under its own name and the attention's under
`self_attn.`.

## Inside the parent's forward: `../source.`

A child's value may read an operation in the *block's* forward: Llama 4's `mlp_output`
is `"../source.hidden_states_view_0.output"`, the block's view of the feed-forward's
output in the residual's shape. Such a read can legitimately come after the block's
call began (a block's `mlp_output` read after its `attention_output`), when
instrumenting the block on first read is too late: the worker is already parked
inside the block, and the plain forward is what is running. Nothing detects this.
The family opts in: it sets `sourced = True` on the *parent's* envoy class, and
`Standard` then reads `self.source` when the envoy is built and again when real
weights replace meta ones (`_update`), which reinstalls the plain forward. Llama 4's
`Layer` is exactly that, `class Layer(Layer): sourced = True`, beside the `Mlp` whose
value reads the block. On GPT-2, the block's `hidden_states_1` binding is
`attn_output + residual`, and the same pair does it:

```python
class Mlp(gpt2.Mlp):
    @EProperty("../source.hidden_states_1.output", description="The residual stream after the attention, from the block's own forward")
    def stream_in(self, value) -> Residual:
        return value


class Layer(gpt2.Layer):
    sourced = True                                  # the opt-in; gpt2.Layer, like Standard, has sourced = False


assert Mlp.stream_in.inside_forward() and not gpt2.Layer.sourced

model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager",
                                envoys={GPT2Block: Layer, GPT2MLP: Mlp})
block = model.layers[0]
assert block.sourced and block._module.__nnsight__.sourced                  # the block's forward is instrumented at build...
untouched = getattr(typo_model.layers[0]._module, "__nnsight__", None)     # ...where a load whose blocks are not flagged leaves it alone
assert untouched is None or not untouched.sourced

with model.trace(prompt):
    x = block.input.save()
    attn_out = block.self_attn.attention_output.save()
    stream = block.mlp.stream_in.save()                     # the block's binding, read after the attention ran
    mlp_out = block.mlp.mlp_output.save()
    out = block.layer_output.save()

assert torch.equal(stream, x + attn_out) and torch.equal(out, stream + mlp_out)
```

The flag is what makes that trace work. With the same `Mlp` and no `Layer`, the block's
forward is instrumented at the `stream_in` read, after the block has run past the
binding, and the read is an ordering error:

<!-- test: expect-error OutOfOrderError -->
```python
unflagged = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager", envoys={GPT2MLP: Mlp})
unflagged_block = unflagged.layers[0]
assert not unflagged_block.sourced and not unflagged_block._module.__nnsight__.sourced   # nothing instrumented it

with unflagged.trace(prompt):
    _ = unflagged_block.self_attn.attention_output.save()
    late = unflagged_block.mlp.stream_in.save()     # OutOfOrderError: '...h.0.source.hidden_states_1.output' was requested but the model already ran past it
```

Forward order still applies with the flag: `hidden_states_1` binds before the MLP runs,
so reading `stream_in` after `mlp_output` in one trace is `OutOfOrderError` too.
`hidden_states_3` is the same binding on the cross-attention branch this config never
takes; asking for it is the ordering error as well
([finding-source-ops.md](finding-source-ops.md), dead branches).

## Introspection

Two methods on the descriptor answer where a value reads, without a trace, and one
class attribute on the envoy says whether its forward is instrumented at build:

```python
inside = {name: value.inside_forward() for name, value in gpt2.Attention.values().items()}
print(inside)
print(gpt2.Attention.attention_output.path(attn), gpt2.Attention.attention_probabilities.path(attn))
assert Mlp.stream_in.inside_forward() and Layer.sourced and not gpt2.Layer.sourced   # a `../source.` key is inside a forward too; the parent opts in
```

```
{'attention_queries': True, 'attention_keys': True, 'attention_values': True, 'attention_scores': True, 'attention_output': False, 'attention_probabilities': True, 'attention_head_outputs': True}
output source.attention_interface_1.source.nn_functional_dropout_0.output
```

- **`path(envoy)`** is the key, or what a key function returns for that envoy (inside a
  trace, when the function reads the forward's own branch variable).
- **`inside_forward()`** is whether the key has a `source` segment, an operation in
  the host's forward, the parent's, or deeper; a function key counts as inside, since
  that is what a key function is for. The family suite uses it to pick the values
  `test_every_source_value_resolves_on_every_layer` reads.
- **`sourced`** is a class attribute of `Standard`, `False` by default: `True` on an
  envoy class makes `Standard.__init__` and `_update` read `self.source`, so the forward
  is instrumented at build and after real weights replace meta ones. A family sets it on
  the envoy whose forward holds a value read after the call starts (Llama 4's `Layer`,
  above); nothing infers it from the children's keys, so a `"../source."` value on an
  unflagged parent resolves only when it is read before the parent runs.

## `DerivedEProperty`

`DerivedEProperty(compute, description=, unavailable=)` has no location: `compute(envoy)`
runs at read time inside the trace and may read any served values; its return annotation
gives the value its `layout`. The result is read-only.

```python
def attention_entropy(self) -> Float[Tensor, "batch heads query"]:
    probs = self.attention_probabilities
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)


class Attention(gpt2.Attention):
    attention_entropy = DerivedEProperty(attention_entropy, description="Per-row entropy of the pattern, [batch, heads, query]", unavailable=interface_reason)


print(Attention.attention_entropy.key, Attention.attention_entropy.dims)

model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager", envoys={GPT2Attention: Attention})
attn = model.layers[0].self_attn

with model.trace(prompt):
    entropy = attn.attention_entropy.save()

n_tokens = entropy.shape[-1]
assert (entropy >= 0).all() and (entropy <= torch.log(torch.tensor(float(n_tokens))) + 1e-4).all()
print(entropy.shape)
```

```
<attention_entropy> ('batch', 'heads', 'query')
torch.Size([1, 12, 10])
```

<!-- test: expect-error AttributeError -->
```python
with model.trace(prompt):
    attn.attention_entropy = entropy
# AttributeError: attention_entropy is derived and read-only
```

Reading through other values means forward order applies to the values it reads:
`attention_entropy` reads the pattern, so a trace that reads it and then
`attention_queries` (the call's inputs, served earlier) is out of order. `compute` takes
the envoy only, not a served value; it is stored as `_preprocess` so `layout` can read its
annotation, and never called as a preprocess.

## `TokenEProperty`: a value held flat over tokens

A mixture of experts routes `[batch * seq, ...]` tensors: the router's logits, the
weights and indices the experts receive, the experts' output. nnsight narrows a served
tensor to an invoke's rows only when its leading dim is the batch, so a plain `EProperty`
there serves the flat tensor, the whole run's tokens under every invoke, and fails the
suite's layout check. `TokenEProperty` takes the same arguments and serves this invoke's
`[batch, seq, ...]` rows as a view (in-place edits land on the model's tensor), and
splices an assignment back into the flat tensor; a tensor already at the layout's rank
passes through. Every routing value on `nnterp.components.Moe` is one, and a family that
relocates one keeps it one, with the base's layout and description:

<!-- test: skip -->
```python
# nnterp/families/hunyuan_v1_moe.py: the router's projection is its own module
class Mlp(Moe):
    @TokenEProperty("router.wg.output", description=Moe.router_logits.description)
    def router_logits(self, value) -> RouterLogits:
        return value
```

A value of your own on a mixture, verified on `hf-internal-testing/tiny-random-MixtralForCausalLM`
(`hidden_size` 64):

<!-- test: skip -->
```python
from nnterp.components import TokenEProperty, mixture_reason
from nnterp.families import mixtral
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock


class Mlp(mixtral.Mlp):
    @TokenEProperty("experts.input", description="The hidden states the routed experts receive", unavailable=mixture_reason)
    def experts_input(self, value) -> Residual:
        return value


moe_model = StandardizedTransformer("hf-internal-testing/tiny-random-MixtralForCausalLM", device="cpu", dispatch=True,
                                    envoys={MixtralSparseMoeBlock: Mlp})
mlp = moe_model.layers[0].mlp
with moe_model.trace("Hello world there"):
    entering = mlp.input.save()                 # the mixture's input comes first in the forward
    routed_in = mlp.experts_input.save()        # [1, 4, 64]; a plain EProperty here serves [4, 64]

assert torch.equal(routed_in, entering)
```

Under two invokes each reads its own `[1, seq, 64]` rows; assigning
`torch.zeros_like(mlp.experts_input)` moves the logits.

## A block hands its children what their modules lack

An envoy knows its own path, not its parent, and a child module often lacks what its
value needs: Gemma-4's mixture is hosted on `mlp` but its `router` and `experts` are the
block's children; GraniteMoE's mixture module keeps no config, so no
`residual_multiplier`. The family's `Layer.__init__` hands them down once the children
exist (`self.mlp.router = self.router`, `hand_residual_multiplier(self)`). Never reach a
parent through `interleaver.envoys`. On GPT-2 the MLP module has no config, so a
per-checkpoint predicate on an `Mlp` reads what its block handed it:

```python
def not_gelu_new(self):
    return None if self.activation == "gelu_new" else f"this MLP runs {self.activation!r}"


class Layer(gpt2.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp.activation = self.self_attn._module.config.activation_function   # the block hands it down


class Mlp(gpt2.Mlp):
    @EProperty("act.output", description="The activation's output, [batch, seq, intermediate]", unavailable=not_gelu_new)
    def activations(self, value):
        return value


handed = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, envoys={GPT2Block: Layer, GPT2MLP: Mlp})
assert handed.layers[0].mlp.activation == "gelu_new" and handed.layers[0].mlp.support()["activations"] is None

with handed.trace(prompt):
    acts = handed.layers[0].mlp.activations.save()

assert acts.shape[-1] == handed.layers[0].mlp.intermediate_size
```

What is handed is set on the envoy, which survives the weight swap of a lazy load.

## Sizes: per module, and on the root

Each block's own sizes are plain properties on its envoys, read off the module:
`self_attn.num_heads`, `.num_kv_heads`, `.head_dim`, `.qk_head_dim` and
`mlp.intermediate_size` (one routed expert's on a mixture). They are what that block
runs with, which differs from the root's on Gemma-4, MiMo-V2-Flash and Laguna. A module
that keeps a size under another name overrides the property on the family's subclass
(JetMoE's and GraniteMoE-Hybrid's `Mlp` return the module's `hidden_size`):

```python
attn0 = handed.layers[0].self_attn
print(attn0.num_heads, attn0.num_kv_heads, attn0.head_dim, attn0.qk_head_dim, handed.layers[0].mlp.intermediate_size)
assert (attn0.num_heads, attn0.head_dim, handed.layers[0].mlp.intermediate_size) == (handed.num_heads, handed.head_dim, handed.intermediate_size)
```

```
12 12 64 64 3072
```

The root's sizes (`num_layers`, `hidden_size`, `vocab_size`, `num_heads`,
`num_kv_heads`, `head_dim`, `qk_head_dim`, `intermediate_size`) are
`StandardizedProperty` descriptors on `StandardizedTransformer` (`nnterp.standardized`,
not exported from `nnterp`), and not eproperties: no location, no trace, no `layout`,
nothing in `support()` or the repr. Each wraps the plain rule over the text config and
on every read does `getattr(model.family, name, None)`: a function of that name in the
family module (or on a registered object) is called with the model instead.
`project_on_vocab` is a `StandardizedCapability`, the same for a method: a family's
`def project_on_vocab(model, hidden)` is bound in the root's place. Both are read-only;
an assignment raises `AttributeError` naming the function to write:

```python
from nnterp.standardized import StandardizedCapability, StandardizedProperty

assert isinstance(StandardizedTransformer.intermediate_size, StandardizedProperty)
assert isinstance(StandardizedTransformer.project_on_vocab, StandardizedCapability)
try:
    handed.hidden_size = 5
except AttributeError as error:
    print(error)
```

```
hidden_size is read off the config; a family defines `def hidden_size(model)` to say it otherwise
```

An `AttributeError` out of either path (the plain `config.intermediate_size` on a
config that has none) is rewritten by `Envoy.__getattr__` into "no attribute", so a
registered variant that spreads a shipped module's `RENAME` and `ENVOYS` carries its
size and `project_on_vocab` functions too. Which shipped families define which:
`docs/reference/families.md`, "Logits, scales and sizes";
[family-module-recipe.md](family-module-recipe.md) quotes gpt2's, falcon's and cohere's.

## The helpers

| name | from | what |
|---|---|---|
| `needs_eager(envoy)` | `nnterp.components` | the sdpa reason, or `None` under `attn_implementation="eager"` |
| `interface_reason(envoy)` | `nnterp.components` | `envoy.off_interface()`; the `unavailable=` every base interface value takes |
| `Attention.off_interface(self)` | method | one decision for every interface value; override it (GPT-2) rather than each value |
| `NOT_ON_INTERFACE` | `nnterp.components` | the reason for `attention_scores = unavailable(NOT_ON_INTERFACE)` on a family with its own arithmetic |
| `seq_first(t)` | `nnterp.components` | `t.transpose(1, 2)`: `[batch, heads, seq, d]` to `[batch, seq, heads, d]` as a view, its own inverse |
| `first_tensor(v)` / `rewrap(envoy, t)` | `nnterp.components` | a tuple module's first element on read; the tensor back in its tuple on write |
| `per_call(envoy, key, compute)` / `pinned(n)` | `nnterp.components` | one computation per module call; pin the next read to occurrence `n` |
| `kernel(attribute)` | `nnterp.components.recurrent` | a `RecurrentMixer` key at whichever kernel fires on this call |
| `mixture_reason` / `needs_grouped_experts` / `no_shared_expert` | `nnterp.components` | a `Moe` value's `unavailable=` predicates |
| `Standard.values()` / `envoy.support()` | class / method | the descriptors by name, base classes first; each one's reason or `None` |

```python
from nnterp.components import seq_first, first_tensor, NOT_ON_INTERFACE

heads_first = torch.zeros(1, 12, 10, 64)
assert seq_first(heads_first).shape == (1, 10, 12, 64)
assert seq_first(seq_first(heads_first)).shape == heads_first.shape
assert first_tensor((heads_first, None)) is heads_first
print(NOT_ON_INTERFACE)
```

## Gotchas

- **Keep the name when overriding.** An override under another name adds a value; the
  inherited descriptor stays and `support()` keeps reporting it.
- **Keep the return annotation, by name.** `layout` and `dims` are read off it and the
  suite checks every value's tensor against it; an override of a standard value writes
  the base's name (`-> Residual`, `-> Pattern`), not a retyped `Float[...]`.
- **The key ends in `output`, `input` or `inputs`.** `"source.F_softmax_0"` without one
  is not a path; the last segment says which served value of the op this is.
- **Tensors you build in a block must be on the model's device.** `dispatch=True` puts
  gpt2 on `cuda:0` when there is one; `torch.arange(...)` is on the CPU. Build from the
  served tensor (`torch.ones_like(value)`) or `.to(value)`.
- **A preprocess that raises `AttributeError` is swallowed** by `Envoy.__getattr__` and
  resurfaces as "no attribute `<name>`"; raise anything else from one that can fail.
- **Subclass the family's class** (`gpt2.Attention`), not `nnterp.Attention`, so the
  family's own overrides stay.
- **Remote traces carry envoy classes by reference.** A class defined in a script's
  `__main__` is not importable on the server; a value that will run on NDIF lives in an
  installed module. An `EProperty` is not cloudpicklable by value.

## Related

- [family-module-recipe.md](family-module-recipe.md): the shipped modules these descriptors are used in.
- [finding-source-ops.md](finding-source-ops.md): the op names a `source` segment takes.
- The `nnsight` skill's `modules-and-architectures.md` reference: `eproperty` and `envoys=` without nnterp.
