# Finding Source Operations

A value read inside a module's forward names an operation on its `EProperty` key, after
a `source` segment: `"source.attention_interface_1.source.nn_functional_dropout_0.output"`
for the pattern, `"source.dropout_add_0.input"` for BLOOM's contribution,
`"source.value_layer_0.output"` for Falcon's values. The names come from
transformers' own source as nnsight labels it, and the only correct list is the one
printed from the version you run. This file is how to read that list, what the labels
mean, which one to pick, and what a release does to them. The `nnsight` skill's
`source-tracing.md` reference is the general account of `.source`; the nnter repo's
page is `docs/extending/finding-source-ops.md`.

<!-- test: setup -->
```python
import warnings

import torch
import nnter
from nnter import StandardizedTransformer
from nnsight.intervention.source import SourceNotAvailable

model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"
attn = model.layers[0].self_attn
```

## Print outside a trace

`print(envoy.source)` needs no trace. It instruments the module's forward (once, and the
module stays instrumented) and renders it with every call and binding labelled at its
line; the labels on the left are the operation names. Trimmed from `openai-community/gpt2`
on transformers 5.17:

```python
print(attn.source)
```

```
                                             * def forward(
 ...
 is_cross_attention_0                    ->  9     is_cross_attention = encoder_hidden_states is not None
 ...
                                            21     if is_cross_attention:
 ...
 self_q_attn_0                           -> 27         query_states = self.q_attn(hidden_states)
 query_states_0                          ->  +         ...
 ...
 self_c_attn_0                           -> 35             key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
 split_0                                 ->  +             ...
 ...
                                            39     else:
 self_c_attn_1                           -> 40         query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
 split_1                                 ->  +         ...
 shape_kv_1                              -> 41         shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
 key_states_view_1                       -> 42         key_states = key_states.view(shape_kv).transpose(1, 2)
 transpose_2                             ->  +         ...
 key_states_2                            ->  +         ...
 ...
 query_states_view_0                     -> 46     query_states = query_states.view(shape_q).transpose(1, 2)
 transpose_4                             ->  +     ...
 query_states_1                          ->  +     ...
 ...
 using_eager_0                           -> 56     using_eager = self.config._attn_implementation == "eager"
 ALL_ATTENTION_FUNCTIONS_get_interface_0 -> 57     attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
 attention_interface_0                   ->  +     ...
                                            61     if using_eager and self.reorder_and_upcast_attn:
 self__upcast_and_reordered_attn_0       -> 62         attn_output, attn_weights = self._upcast_and_reordered_attn(
                                            65     else:
 attention_interface_1                   -> 66         attn_output, attn_weights = attention_interface(
                                            67             self,
                                            68             query_states,
                                            69             key_states,
                                            70             value_states,
                                            71             attention_mask,
 ...
 attn_output_reshape_0                   -> 77     attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
 contiguous_0                            ->  +     ...
 attn_output_0                           ->  +     ...
 self_c_proj_0                           -> 78     attn_output = self.c_proj(attn_output)
 attn_output_1                           ->  +     ...
 self_resid_dropout_0                    -> 79     attn_output = self.resid_dropout(attn_output)
 attn_output_2                           ->  +     ...
                                            81     return attn_output, attn_weights
```

The interface call is `attention_interface_1` on every family that uses it, which is why
`nnter.components.INTERFACE` is that string. Its positional arguments are `(self, query,
key, value, attention_mask)`, so the base `Attention` reads the queries at
`"source.attention_interface_1.inputs"` with `select=1`. A wrong name raises `AttributeError` listing every
operation the module has:

<!-- test: expect-error AttributeError -->
```python
attn.source.nope_0
# AttributeError: 'model.transformer.h.0.attn.source' has no operation 'nope_0'; available: is_cross_attention_0, isinstance_0, ...
```

## How operations are named

- **A call** is `<callee>_<n>`, the whole attribute chain joined with `_`: `self.c_proj(...)`
  is `self_c_proj_0`, `nn.functional.softmax(...)` is `nn_functional_softmax_0`,
  `self._attn(...)` is `self__attn_0` (the underscore in `_attn` stays), `torch.bmm(...)`
  is `torch_bmm_0`, `F.softmax(...)` is `F_softmax_0` when the module imports it as `F`.
  The counter is per callee, in source order, counting every call in the source whether
  or not this config executes it.
- **A binding** is an operation too, `<name>_<n>`: the n-th assignment to that name in
  the forward. `query_states_0` is the first binding of `query_states`; `attn_weights_1`
  the second binding of `attn_weights`. Calls and bindings share one counter per name,
  so `attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(...)` is
  `attention_interface_0` and the call `attention_interface(...)` is
  `attention_interface_1`. The `+ ...` lines under a call are the bindings of its result.
- **`.output`** is what the call returned (or what the binding bound); **`.input`** the
  call's first argument; **`.inputs`** the full `(args, kwargs)`. For a binding, `.input`
  and `.output` are the same value. An assignment's `.output` need not be a tensor:
  `attention_interface_0` binds the implementation function, `shape_q_0` a tuple. The
  key ends in the same three words: `"source.<op>.output"`, `"source.<op>.input"`,
  `"source.<op>.inputs"`.

Families use both kinds: MPT reads its queries at the `query_states_0` binding, Falcon
its values at `value_layer_0` and its head outputs at the `attn_output_1` binding
(`flatten_0` on its alibi branch); BLOOM
reads its `dropout_add_0` call's `input`.

## Drill inside a trace

`attention_interface_1` calls a function, transformers' `eager_attention_forward`. Its
operations exist only under `attention_interface_1.source`, and that drill works only
inside a trace: the callee is a local variable bound at run time, so nnsight resolves it
from the live call (a served read of the call's `.fn`) the first time someone drills into
it in that run, and clears what it built at the start of the next run.

<!-- test: expect-error SourceNotAvailable -->
```python
attn.source.attention_interface_1.source
# SourceNotAvailable: recursive `.source` is only available inside a trace
```

The Llama family, on `HuggingFaceTB/SmolLM2-135M-Instruct` (`model_type` `llama`):

<!-- test: setup -->
```python
llama = StandardizedTransformer("HuggingFaceTB/SmolLM2-135M-Instruct", device="cpu", dispatch=True, attn_implementation="eager")
assert llama.family.__name__ == "nnter.families.llama"
llama_attn = llama.layers[0].self_attn
print([op.name for op in llama_attn.source])
```

```
['input_shape_0', 'hidden_shape_0', 'self_q_proj_0', 'view_0', 'transpose_0', 'query_states_0', 'self_k_proj_0', 'view_1', 'transpose_1', 'key_states_0', 'self_v_proj_0', 'view_2', 'transpose_2', 'value_states_0', 'apply_rotary_pos_emb_0', 'past_key_values_update_0', 'ALL_ATTENTION_FUNCTIONS_get_interface_0', 'attention_interface_0', 'attention_interface_1', 'attn_output_reshape_0', 'contiguous_0', 'attn_output_0', 'self_o_proj_0', 'attn_output_1']
```

Inside a trace, in forward order: the call's arguments, the drill, an operation inside,
the call's own return:

```python
with llama.trace(prompt):
    args = llama_attn.source.attention_interface_1.inputs.save()      # 1. the call's (args, kwargs)
    inner = llama_attn.source.attention_interface_1.source            # 2. the drill, from the live callee
    print(inner)
    probs = inner.nn_functional_dropout_0.output.save()               # 3. an operation inside the callee
    returned = llama_attn.source.attention_interface_1.output.save()  # 4. the call's return

assert len(args[0]) == 5 and args[0][1].shape[1] == llama.num_heads    # (module, query, key, value, mask)
assert torch.equal(returned[1], probs)
print(probs.shape)
```

```
                             * def eager_attention_forward(
 ...
 repeat_kv_0             ->  9     key_states = repeat_kv(key, module.num_key_value_groups)
 key_states_0            ->  +     ...
 repeat_kv_1             -> 10     value_states = repeat_kv(value, module.num_key_value_groups)
 value_states_0          ->  +     ...
 key_states_transpose_0  -> 12     attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
 torch_matmul_0          ->  +     ...
 attn_weights_0          ->  +     ...
                            13     if attention_mask is not None:
 attn_weights_1          -> 14         attn_weights = attn_weights + attention_mask
 nn_functional_softmax_0 -> 16     attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
 to_0                    ->  +     ...
 attn_weights_2          ->  +     ...
 nn_functional_dropout_0 -> 17     attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
 attn_weights_3          ->  +     ...
 torch_matmul_1          -> 18     attn_output = torch.matmul(attn_weights, value_states)
 attn_output_0           ->  +     ...
 attn_output_transpose_0 -> 19     attn_output = attn_output.transpose(1, 2).contiguous()
 contiguous_0            ->  +     ...
 attn_output_1           ->  +     ...
                            21     return attn_output, attn_weights
torch.Size([1, 9, 9, 9])
```

This listing is where the base `Attention`'s values come from:

```python
print({name: attr.key for name, attr in type(llama_attn).values().items() if attr.inside_forward()})
```

```
{'attention_queries': 'source.attention_interface_1.inputs', 'attention_keys': 'source.attention_interface_1.inputs', 'attention_values': 'source.attention_interface_1.inputs', 'attention_scores': 'source.attention_interface_1.source.nn_functional_softmax_0.input', 'attention_probabilities': 'source.attention_interface_1.source.nn_functional_dropout_0.output', 'attention_head_outputs': 'source.attention_interface_1.output'}
```

An `EProperty` whose key has a `source` segment does this drill for you at every read:
the first `source` instruments the module's forward, each later `.source.` drills from
the call before it into the callee. `inside_forward()` says whether a value's key has
one. GPT-OSS's interface concatenates a sink column between `attn_weights_1` and the
softmax, so its family reads `attention_scores` at
`"source.attention_interface_1.source.attn_weights_1.output"`.

## Picking the operation

- **The pattern is the dropout after the softmax**, where one exists:
  `nn_functional_dropout_0` on the interface, `self_attention_dropout_0` on BLOOM,
  `self__attn_0.source.self_attn_dropout_0` on GPT-J. That is the tensor the values are
  mixed with, in the model's dtype (the softmax runs in float32 on the interface and
  casts back) and, on a sink model, with the sink column already dropped. In eval on a
  float32 model the softmax output and the dropout output are equal tensors; the rule is
  about what the location means. Falcon without alibi has no dropout after `F_softmax_0`,
  so its pattern is the softmax itself; its alibi branch runs a second softmax,
  `F_softmax_1`, with `self_attention_dropout_0` after it, and the pattern is that
  dropout's output.
- **A contribution added inside the module** is the tensor entering the add: BLOOM's
  `dropout_add_0` input, MPT's `F_dropout_0` output.
- **Queries, keys and values** are what the attention arithmetic receives: the interface
  call's arguments, GPT-J's `self__attn_0` arguments, BLOOM's `self__reshape_0` returns,
  Falcon's `apply_rotary_pos_emb_0` returns without alibi and its `query_layer_0` /
  `key_layer_0` bindings with it (that branch has no rotary). Take them after the rotary embedding where
  there is one, before `repeat_kv` so the head axis is `num_kv_heads` wide.
- **Head outputs** are the last matmul: the interface call's output element 0,
  `torch_bmm_0` on BLOOM, `torch_matmul_1` on MPT.
- **Prefer a real submodule** when one exposes the value: `mlp.output` beats
  `source.self_c_proj_0.output`; it is cheaper and stable across releases.

## Order within a trace

Requests are served in the forward's order. For a call and its inside: the call's
`.inputs`, then the drill (served from the call's `.fn` just before it runs), then the
operations inside, then the call's `.output`. Reading the call's inputs after an
operation inside it is out of order:

<!-- test: expect-error OutOfOrderError -->
```python
with llama.trace(prompt):
    late_probs = llama_attn.source.attention_interface_1.source.nn_functional_dropout_0.output.save()
    late_args = llama_attn.source.attention_interface_1.inputs.save()
# OutOfOrderError: 'model.model.layers.0.self_attn.source.attention_interface_1.input.i0' was requested
#                  but the model already ran past it
```

The same rule governs two standard values read in one trace, since each read drills,
and a module-boundary read before a value inside that module's call: the module has
returned, so the drill waits on a `.fn` handoff that already happened. Either way the
error names the call's `.fn`, the location the drill waits on, not the value you asked
for:

<!-- test: expect-error OutOfOrderError -->
```python
with llama.trace(prompt):
    probs_first = llama_attn.attention_probabilities.save()   # inside the call
    queries_after = llama_attn.attention_queries.save()       # the call's inputs: already served
```

<!-- test: expect-error OutOfOrderError -->
```python
with model.trace(prompt):
    resid = model.layers[0].layer_output.save()                # the block has returned...
    late_probs = attn.attention_probabilities.save()           # ...so the drill into the call is too late
# OutOfOrderError: 'model.transformer.h.0.attn.source.attention_interface_1.fn.i0' was requested
#                  but the model already ran past it
```

Read a module's interior values before its boundary values. The family suite reads
interior values one per trace for this reason, and on Falcon the order between values of
one module matters too (`attention_values` before `attention_queries` without alibi;
queries, keys, values in forward order with it).

A read pinned by `tracer.iter` to a step or position the run never serves does not
raise: nnsight cuts the block short with a `UserWarning` ("was never reached ... cut
short"), keeps what was saved before, and skips every statement after, so later names
are unbound. If a saved name is missing after a trace, look for that warning:

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    with model.generate(prompt, max_new_tokens=2) as tracer:
        first_logits = model.logits.save()
        for step in tracer.iter[5]:                            # a step this run never makes
            never = model.layers[0].layer_output.save()
        after_loop = model.logits.save()

assert any("was never reached" in str(w.message) for w in caught)
assert "first_logits" in globals()
assert "never" not in globals() and "after_loop" not in globals()   # the block stopped at the loop
```

## Dead branches

The listing includes every operation in the forward, including the ones under an `if`
this config makes false. GPT-2's attention lists 50 operations and runs 28: the
cross-attention path (`self_q_attn_0`, `self_c_attn_0`, `transpose_0`, `transpose_1`) and
the cache-hit path sit beside the live ones, and the upcast path
(`self__upcast_and_reordered_attn_0`) runs only with `reorder_and_upcast_attn`. Asking for
a dead label parks a request at a location the model never reaches, and the trace fails
with the *ordering* error:

<!-- test: expect-error OutOfOrderError -->
```python
with model.trace(prompt):
    dead = attn.source.transpose_0.output.save()      # the cross-attention key transpose
```

```python
with model.trace(prompt):
    key = attn.source.transpose_2.output.save()       # the live twin: the next occurrence of the same callee

assert key.shape == (1, model.num_heads, len(model.tokenizer(prompt)["input_ids"]), model.head_dim)
```

When a label you can see reports as run past, read the listing around it before
suspecting your request order: an operation under a false condition is dead, and the
live twin has the next occurrence index. Requesting one operation per trace tells you
which labels answer.

## What cannot be drilled

- **A `torch.nn.functional` entry point.** Every one refers to its own name in its body
  to reach the dispatcher, so `nn_functional_softmax_0.source` raises `KeyError:
  'softmax'`. Its `.output` and `.input` are what a value wants anyway.
- **A binding.** `attn_weights_1.source` raises `SourceNotAvailable: 'attn_weights_1' is
  an assignment, not a call; there is no function to drill into`.
- **A submodule call.** `self_c_proj_0.source` is refused; read `attn.c_proj.output` or
  `.source` on that submodule.
- **A builtin or C function.** No Python source exists.

<!-- test: expect-error KeyError -->
```python
with llama.trace(prompt):
    llama_attn.source.attention_interface_1.source.nn_functional_softmax_0.source
```

## What a release changes and how the suite catches it

An op name is `{callable}_{occurrence}`, so three things move one: the call is renamed,
a binding of the same name is added or removed before it, or the arithmetic moves onto
or off transformers' shared interface (DBRX is on the interface in 5.17 and its family
is the base class; an earlier layout had its own arithmetic). `RENAME` keys and `ENVOYS`
classes are module names and types, which move rarely; a renamed module is a silently
skipped alias, caught by `test_standard_names_alias_native_envoys`.

The family suite guards the op names on every pinned checkpoint:

- `test_every_source_value_resolves_on_every_layer`: every available value on the
  family's `Attention` whose key reads inside the forward (`inside_forward()`) reads a
  tensor on every attention block. A renamed op fails
  here with `SourceNotAvailable` naming the missing op and the ops that exist.
- `test_written_pattern_moves_the_logits`: an op that still resolves but is no longer
  what the values are mixed with (a copy, a tensor the forward returns and never uses)
  reads fine and fails here.
- `test_interior_shapes` checks `softmax(scores) == pattern`, which catches a `_0`/`_1`
  slip that lands on a different tensor of the same shape.
- `test_contribution_identity` catches a moved contribution op on BLOOM, MPT and Falcon.

The procedure when upgrading transformers: run `HF_HUB_OFFLINE=1 pytest -q` in the nnter
repo and read each `SourceNotAvailable`; for each failing family print
`model.layers[0].self_attn.source` on the old and new versions (and, inside a trace,
`attn.source.attention_interface_1.source`) and diff the labelled listings; decide
whether the value moved or the family moved onto or off the interface; update the op
string in the family module, or delete the override when the family now uses the shared
interface, keeping the standard name's meaning fixed; re-run the family's file, then the
whole suite. `support()` cannot know an op moved, since it does not run the forward; only
a trace can. Do not add version conditionals to a family module.

## Gotchas

- **Print outside a trace, drill inside one.** `print(envoy.source)` needs no trace;
  `call.source` needs the live callee.
- **First `.source` access on a module must come before that module's forward runs** in
  a trace, since it rewrites the forward. A `source`-keyed value read as the first
  request on its module is fine; a bare `_ = envoy.source` outside the trace instruments
  it up front, and a `Standard` envoy class with `sourced = True` (Llama 4's `Layer`,
  whose `Mlp` reads `"../source.<op>"` after the block's attention ran) is instrumented
  when it is built; the flag is set by the family, never inferred from a child's key.
- **A dead label is an `OutOfOrderError`, not an `AttributeError`.** The listing does not
  say which branch runs.
- **Drill into a call on step 0 if you will read inside it with `tracer.iter`.** Occurrences
  of an operation inside a called function are counted from the first `op.source` drill of
  the run, not from the run's start. Under `generate`, a call first drilled on step 1 has
  that step as its occurrence 0, so a later `tracer.iter[1]` read inside it returns step
  2's value, without an error. Touch `op.source` before the first step, or read it on
  every step from 0.
- **Sourcing a module costs a little on every forward afterwards**, trace or not (about
  6% for all of GPT-2's blocks); a family value instruments only the modules it is read
  on.
- **`.source` snapshots module globals** at first instrumentation: a module-level name
  rebound afterwards (a kernel switch, a monkeypatch) is not what the instrumented copy
  sees. This is why `nnter.route_kernels(model.family, "torch")` must run before the
  first trace of a recurrent mixer.

## Related

- [descriptors.md](descriptors.md): the `EProperty` key these names go into.
- [family-module-recipe.md](family-module-recipe.md): the suite test that guards every name.
- The `nnsight` skill's `source-tracing.md` reference: iteration, dispatchers, the full limits table.
- The `debugging` skill: `OutOfOrderError` and a trace that ends early.
