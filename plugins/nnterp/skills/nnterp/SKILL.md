---
name: nnterp
description: "Write interpretability code once against nnterp's StandardizedTransformer and run it on 98 transformer families and on vision-language models: one module vocabulary (model.layers[i].self_attn, .mlp, model.norm, model.lm_head) and standard values (layer_output, attention_output, mlp_output, attention_probabilities, logits), plus support(), steer and project_on_vocab. Load before writing Python that reads or edits a transformer's activations (the residual stream after each layer, a sublayer's contribution, attention, expert routing, recurrent state, an image's path) when the code must run on more than one model family, or when an experiment needs logic that an nnspec document cannot express. Bundles the recipes (logit lens, direct logit attribution, ablation, patching, steering, probing, cross-family sweeps) and a guide to adding or overriding a family. Prefer the nnspec skill when the experiment fits a JSON document; use the nnsight skill when nnterp does not support the family or needs lower-level access."
---

# nnterp

nnterp is a layer over nnsight. `StandardizedTransformer` is a `TransformersModel` whose
module tree answers to one set of names on every family nnterp ships (98, from GPT-2 to
DeepSeek and the Qwen3.5 hybrids), and whose blocks carry *standard values*
(`layer_output`, `attention_output`, `attention_probabilities`, `logits`, ...) that mean the
same thing on each, so one script runs on every family without a branch.

This skill covers what nnterp adds. The API underneath is nnsight's and works unchanged; its
home is the `nnsight` skill: its `SKILL.md` for `trace`, `generate`, `.save()`, invokes,
`tracer.iter` and `.source`, and its remote, vllm, tensor-parallel and debugging guides for
NDIF, the other engines, and an error you cannot place.

## Which library

nnterp sits between nnspec and nnsight. When the user names a library, use it. Otherwise take the highest row that fits,
and take a row only when its skill is available in this session:

| Use | When |
|---|---|
| `nnspec` skill | A standard experiment on a transformer language model (lens, patching, ablation, attention, steering, probes, SAEs, DAS) that fits a JSON document. No intervention code to write. |
| nnterp (this skill) | The experiment needs Python that a document cannot express, on a family nnterp knows. One module vocabulary on every family. |
| `nnsight` skill | nnterp does not know the family, or the work needs access below nnterp's standard values: arbitrary modules, `.source`, your own or non-transformer models, diffusion, vLLM, tensor parallelism. Also existing nnsight code, and any error an nnsight trace raises. |

## Guides

Two guides go with this file. Read the one that matches before writing code:

- [patterns](references/patterns.md): the techniques (logit lens, direct logit
  attribution, ablation, patching, steering, probing, cross-family sweeps) written
  once against the standard values.
- [extending](references/extending.md): adding a family nnterp does not know,
  overriding a value, or defining a value of your own.

The depth files for single topics are listed under [When you need more](#when-you-need-more).

## Load, read, and the contribution identity

```python
import torch
import nnsight
import nnterp
from nnterp import StandardizedTransformer, Unavailable

# device= places the model; device_map="cpu" is ignored by the pipeline nnsight loads through
model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"

ids = model.tokenizer(" Paris", add_special_tokens=False).input_ids
assert len(ids) == 1, ids                  # one token, or it is not one logit
paris = ids[0]
```

```python
with model.trace(prompt):
    x = model.layers[5].input.save()                                     # the stream entering block 5
    pattern = model.layers[5].self_attn.attention_probabilities.save()   # inside the attention: read before its output
    attn = model.layers[5].self_attn.attention_output.save()             # what the attention adds to the stream
    mlp = model.layers[5].mlp.mlp_output.save()                          # what the MLP adds
    resid = model.layers[5].layer_output.save()                          # the stream leaving the block
    logits = model.logits.save()

assert model.family is nnterp.families.gpt2
assert (model.num_layers, model.hidden_size, model.num_heads) == (12, 768, 12)
assert x.shape == attn.shape == mlp.shape == resid.shape == (1, 10, 768) and pattern.shape == (1, 12, 10, 10)
assert torch.equal(x + attn + mlp, resid)                       # the contribution identity, exact in fp32
assert logits[0, -1].argmax() == paris
```

`layers[i].input + attention_output + mlp_output == layer_output` defines the two
contributions, on sequential, parallel and post-norm blocks alike. `model.layers[5]` *is*
`model.transformer.h[5]`, an alias, so native paths keep working. `attn_implementation="eager"`
is what makes the attention interior readable (queries, keys, values, scores, the pattern,
head outputs); everything else works under transformers' default `sdpa`. The dtype is the
checkpoint's (often bf16).

## The vocabulary

Llama's names, with the containers lifted out of the inner `.model`:

| standard name | GPT-2 | Llama | GPT-NeoX |
|---|---|---|---|
| `model.embed_tokens` | `transformer.wte` | `model.embed_tokens` | `gpt_neox.embed_in` |
| `model.layers[i]` | `transformer.h[i]` | `model.layers[i]` | `gpt_neox.layers[i]` |
| `model.layers[i].self_attn` | `...h[i].attn` | same | `...layers[i].attention` |
| `model.layers[i].self_attn.o_proj` | `...attn.c_proj` | same | `...attention.dense` |
| `model.layers[i].mlp` | same | same | same |
| `model.layers[i].mlp.act_fn` | `...mlp.act` | same | `...mlp.act` |
| `model.layers[i].mlp.down_proj` | `...mlp.c_proj` | same | `...mlp.dense_4h_to_h` |
| `model.norm` | `transformer.ln_f` | `model.norm` | `gpt_neox.final_layer_norm` |
| `model.lm_head` | same | same | same |

`self_attn.input` is what enters the attention and `mlp.input` what enters the MLP, on every
family. The norm aliases (`input_layernorm`, `post_attention_layernorm`) mean different
things per family (on Gemma-2/3/4 `post_attention_layernorm` follows the attention): read
the boundary values `attention_norm_output`, `layer_mid` and `mlp_norm_output` instead,
which each family points at its own modules.

A hybrid block has `linear_attn`, a recurrent mixer, instead of `self_attn` (three blocks in
four on Qwen3.5). Decide which blocks have which outside the trace; `getattr` inside one can
trip a served value:

```python
attn_blocks = [i for i, layer in enumerate(model.layers) if getattr(layer, "self_attn", None) is not None]
assert attn_blocks == list(range(12))       # every block, on GPT-2
```

## The values most experiments touch

| value | on | layout | what it is |
|---|---|---|---|
| `layer_input`, `layer_output` | `layers[i]` | `Residual` `[batch, seq, hidden]` | the stream entering the block (`layers[i].input`) and leaving it |
| `attention_output`, `mlp_output` | `self_attn` / `linear_attn`, `mlp` | `Residual` | what each sublayer adds to the stream |
| `attention_norm_output` | `layers[i]` | `Residual` | the normed stream the attention is called with |
| `layer_mid` | `layers[i]` | `Residual` | the stream between the sublayers, `layer_input + attention_output`; none on a parallel block; an assignment writes in place |
| `mlp_norm_output`, `mlp_input` | `layers[i]` | `Residual` | the MLP's normed input; what the MLP is called with |
| `attention_premix` | `self_attn` | `HeadOutputs` `[batch, seq, heads, head_dim]` | `o_proj.input` split by head, after any gate; any `attn_implementation` |
| `mlp_activation`, `mlp_neurons` | `mlp` | `Neurons` `[batch, seq, intermediate]` | `act_fn.output`; `down_proj.input` (`act(gate) * up` on a gated MLP) |
| `attention_queries`, `_keys`, `_values` | `self_attn` | `Queries`, `Keys`, `Values` `[batch, heads, seq, head_dim]` | after the rotary; keys and values `num_kv_heads` wide; eager |
| `attention_scores`, `attention_probabilities` | `self_attn` | `Pattern` `[batch, heads, query, key]` | masked, scaled, before the softmax; the pattern the values are mixed with; eager |
| `attention_head_outputs` | `self_attn` | `HeadOutputs` | each head's output, before the output projection; eager |
| `token_embeddings`, `final_norm_output` | root | `Residual` | `embed_tokens.output`; `norm.output`, what the head projects |
| `lm_head_output`, `logits` | root | `Logits` | the head's raw output; the model's logits, after any softcap or scale |
| `next_token_probs` | root | `NextTokenProbs` `[batch, vocab]` | `logits[:, -1].softmax(-1)` in the model dtype; read-only |
| `input_ids`, `attention_mask` | root | `Tokens` | the model's inputs: read first; assignable |

A mixture of experts adds its routing values, a recurrent mixer its state, a
vision-language wrapper its tower's values ([below](#when-you-need-more)). Batch is axis 0;
the sequence axis is 1, except on the queries, keys and values, where it is 2: the last token
is `value[:, -1]` for a residual value and `value[:, :, -1]` for those three.

```python
with model.trace(prompt):
    block = model.layers[5]
    x = block.layer_input.save()
    premix = block.self_attn.attention_premix.save()     # o_proj's input, by head
    attn = block.self_attn.attention_output.save()
    mid = block.layer_mid.save()                         # the stream between the sublayers
    neurons = block.mlp.mlp_neurons.save()               # down_proj's input
    head = model.lm_head_output.save()

assert torch.equal(mid, x + attn)
assert premix.shape == (1, 10, 12, 64) and neurons.shape == (1, 10, 3072)
assert torch.equal(head, logits)                         # GPT-2's head applies nothing after lm_head
```

To ablate or attribute one head without an eager load, use `attention_premix`: on a plain
attention it is the same tensor as `attention_head_outputs`, which needs eager. For one
neuron, `mlp_neurons` is the coefficient the down projection sums into the stream.

**Sizes.** `model.num_layers`, `hidden_size`, `vocab_size`, `num_heads`, `num_kv_heads`,
`head_dim` and `intermediate_size` are the config's, whatever the family calls them. Where
blocks differ (Gemma-4's full blocks have other head sizes than its sliding ones) read the
block's own: `layers[i].self_attn.num_heads` / `.num_kv_heads` / `.head_dim`,
`layers[i].mlp.intermediate_size`.

## Writing a value

```python
with model.trace(prompt):
    model.layers[5].self_attn.attention_output[:, -1] = 0       # ablate attention at the last position
    model.layers[5].mlp.mlp_output[:] = 0                        # ablate the MLP everywhere
    model.layers[8].layer_output[:, -1, :] *= 2                  # scale the stream leaving block 8
    changed = model.logits.save()

with model.trace(prompt):
    model.layers[5].layer_output = model.layers[5].layer_output * 0     # assign a new tensor
    zeroed = model.logits.save()

assert not torch.allclose(changed, logits) and not torch.allclose(zeroed, logits)
```

An in-place edit reaches the model because the value is the live tensor; an assignment
replaces it, and where the block or attention module returns a tuple the other elements are
put back for you. A read after an in-place edit returns the edited tensor: `.clone()` first
when you need the value before it.

## Traps that corrupt a result silently

**The target token is BOS.** `tokenizer.encode(" Paris")[0]` is the BOS token on Llama-3 and
Gemma tokenizers, and every metric then runs and prints 0.000. Take
`tokenizer(word, add_special_tokens=False).input_ids` and assert one element, as the setup
block does. A Llama-2 or Mistral sentencepiece tokenizer splits `" Paris"` into
`['▁', 'Paris']`: tokenize `"Paris"` there. A word that splits otherwise is not one logit:
pick another, or use `nnterp.prompt_utils.get_first_tokens`.

**`[0]` takes a batch row.** The values are the tensor on every family, including attention
modules and blocks whose raw `.output` is a tuple. Indexing `[0]` to unwrap, as code written
against `.output` does, silently takes batch row 0:

```python
with model.trace(prompt):
    raw = model.layers[3].self_attn.output.save()
    wrong = model.layers[3].layer_output[0].save()

assert isinstance(raw, tuple) and wrong.shape == (10, 768)       # batch row 0, not "the tensor"
```

**Gemma-4's blocks are not a plain sum.** A Gemma-4 block multiplies its sum by a per-block
`layer_scalar` (0.005 to 0.99): `(input + attention_output + mlp_output [+ per_layer_output])
* layer_scalar == layer_output`. A term added in block `i` reaches the last stream times every
scalar from `i` on, and naive direct logit attribution ranks the wrong terms by orders of
magnitude. Doge, ZAYA and DeepSeek-V4 (whose `layer_output` is
`[batch, seq, streams, hidden]`) break the plain sum too. The weighted sums, exact:
[references/values.md](references/values.md#blocks-that-are-not-a-plain-sum). The logit lens
through `project_on_vocab` is unaffected.

**A late read under `tracer.iter` binds the next step.** In a plain trace a read placed after
the model ran past it raises `OutOfOrderError`. Inside `tracer.iter` it raises on step 0, but
on a later step it binds the same location on the *next* step, so a per-step list comes back
shifted by one; the scores and the pattern, read inside the attention call, shift from step 0.
Put a step's reads in forward order (`model.rank`, below) and check each step's shapes
(`[batch, 1, ...]` on a decode step).

**Overwriting `attention_scores` lifts the causal mask.** The scores are read after masking,
so `scores[:] = 0` lets every query attend to the future. Add to them
(`scores[..., k] += c`, or `-inf` to block key `k`).

**bf16 metrics.** On a bf16 checkpoint `next_token_probs` is a bf16 softmax, off by up to
~2e-3, and a pattern's rows sum to one within ~4e-3. Take `logits[:, -1].float().softmax(-1)`
for a metric, compare with a tolerance, and compute a KL with `F.kl_div` on float32
log-softmaxes: `next_token_probs` can hold exact zeros, where `p * (p.log() - q.log())` is NaN.

## Finding out what a checkpoint has

`model.support()` says which values a checkpoint has, from the config, without running
anything: every root and block value by dotted name, `None` where every block has it, else
`{block: reason}`. `model.support(layer=i)` is one block, flat. Reading an unavailable value
raises `nnterp.Unavailable` with the same reason, before the model runs. A model built
without `dispatch=True` sits on the meta device, so asking costs no weights:

```python
plain = StandardizedTransformer("openai-community/gpt2")        # meta device, transformers' default attention
reason = plain.support(layer=0)["self_attn.attention_probabilities"]
assert reason == "read inside the eager attention forward, but this model runs 'sdpa'; load with attn_implementation='eager'"
assert plain.support(layer=0)["self_attn.attention_output"] is None    # boundary values never need eager
assert plain.support().get("mlp.mlp_output", "absent") is None         # .get: a module no block has has no key
```

Branch on `support()` outside the trace. `hasattr(envoy, value)` cannot answer: it raises
`Unavailable` for an unavailable value, and `ValueError` (or `SourceNotAvailable`) outside a
trace for an available one.

`print(envoy)` lists the values an envoy carries, one line each, `(name) -> Layout [axes]:
description`, whether or not this checkpoint has them; the descriptor answers too:

```python
print(type(model.layers[0].self_attn).attention_probabilities)
# (attention_probabilities) -> Pattern [batch num_heads query key]: The attention pattern the values are mixed with
assert type(model.layers[0].self_attn).attention_probabilities.dims == ("batch", "num_heads", "query", "key")
```

Reads and edits follow the forward within one invoke: in a block, the attention's interior,
then `attention_output`, the MLP's values, `layer_output`; block 3 before block 5; `logits`
last. A read placed after the forward passed it raises `OutOfOrderError`, naming an internal
location (`...attention_interface_1.fn.i0`), not the value:

```python
with model.trace(prompt):
    late_attn = model.layers[3].self_attn.attention_output.save()
    late_pattern = model.layers[3].self_attn.attention_probabilities.save()   # inside the attention: too late
```

`model.order(i)` lists block `i`'s available values in forward order, measured on the loaded
checkpoint, and `model.rank(name, layer)` is a sortable key, both by `support()`'s names.
When code decides what to read or edit (a sweep over several values per block, a port of
code that hooked modules in its own order), sort the sites by `rank`; the order differs per
family:

```python
sites = [("layer_output", 3), ("self_attn.attention_probabilities", 3), ("mlp.mlp_output", 1)]
sites.sort(key=lambda site: model.rank(*site))                 # forward order, on any family
assert sites == [("mlp.mlp_output", 1), ("self_attn.attention_probabilities", 3), ("layer_output", 3)]

saved = []
with model.trace(prompt):
    for name, layer in sites:
        host = model.layers[layer]
        for part in name.split("."):                           # "self_attn.attention_probabilities" -> the value
            host = getattr(host, part)
        saved.append(host.save())

assert [tuple(s.shape) for s in saved] == [(1, 10, 768), (1, 12, 10, 10), (1, 10, 768)]
```

Before writing against an unfamiliar checkpoint, run the script (meta device, no weights): it
prints the family, the standard-name to native-path table, the sizes, `support()` and a
block's repr (`--eager`: as an eager load reports them; `--task image-text-to-text`: a
vision-language wrapper's tower too).

```
python scripts/inspect_family.py <repo_id> [--layer N] [--eager] [--task image-text-to-text]
```

## Methods over the values

```python
torch.manual_seed(0)
direction = torch.randn(model.hidden_size)

with model.trace(prompt):
    before = model.layers[1].layer_output.save()
    model.skip_layers(2, 3)                                      # blocks 2 and 3 do not run (inclusive)
    after = model.layers[3].layer_output.save()
    model.steer(6, direction, factor=3.0, token_positions=-1)    # add to the stream leaving block 6, in place
    mid = model.layers[9].layer_output.save()
    lens = model.project_on_vocab(mid).save()                    # logit lens: the model's own head
    last = model.layers[-1].layer_output.save()
    out = model.logits.save()

assert torch.equal(before, after)
assert torch.equal(model.project_on_vocab(last), out)            # exact at the last block, on every family
print(model.get_topk_closest_tokens(mid[0, -1], k=3))            # [{token: probability}]
```

`project_on_vocab` is the model's own path from the stream to its logits: the final norm,
`lm_head`, then any softcap or scale (Gemma-2's and Gemma-4's softcap included), so the lens
on the last block equals `logits` on every family. `steer` adds `factor * vector` in place to
`layer_output`; scale the factor to the stream's norm (`layer_output[0, -1].norm()`, which
grows with depth). `skip_layers` consumes `layers[start].input`: read it first or not at all.
`get_topk_closest_tokens` takes a residual-stream tensor, not logits.

## When you need more

| when you need to | read |
|---|---|
| move a query or key to another position or prompt, or edit it before the rotation: write `attention_queries_prerope` / `attention_keys_prerope` (any `attn_implementation`); `attention_queries` / `_keys` are rotated for where they ran | [attention-interior.md](references/attention-interior.md#before-the-rotary-embedding-attention_queries_prerope-attention_keys_prerope-rotate) |
| recompute a block's attention with other keys or values, or patch what some heads read without the KV cache or the rest of their group seeing it: `self_attn.interface` | [attention-interior.md](references/attention-interior.md#calling-the-attention-function-again-interface) |
| multiply a head's weights (an OV circuit, attribution through the output projection): `self_attn.value_weight(kv)`, `output_weight(j)`, `layers[i].attention_input_gain` / `attention_output_gain`, never slices of `.weight` (fused and transposed layouts; under a PEFT adapter `.weight` is the base alone) | [attention-interior.md](references/attention-interior.md#the-weights-per-head) |
| go deeper into the attention interior: sinks (GPT-OSS), latent attention (DeepSeek), grouped-query widths, edits under `generate` | [attention-interior.md](references/attention-interior.md) |
| read or edit expert routing (Qwen-MoE, DeepSeek-V3, GPT-OSS): router logits, expert weights and indices, ablating or rerouting an expert | [mixture-of-experts.md](references/mixture-of-experts.md) |
| read or edit a recurrent mixer's values or state (Qwen3.5's DeltaNet blocks, Nemotron-H's Mamba-2 blocks, Mamba); `nnterp.route_kernels` goes before the first trace where `support()` asks for it | [recurrent-mixers.md](references/recurrent-mixers.md) |
| follow an image into a vision-language model (Qwen3.5, Gemma 3, Llava): load with `task="image-text-to-text"`, read `vision.image_token_mask` first, edit `vision.image_features` | [vision.md](references/vision.md) |
| read values under `generate` and `tracer.iter`: per-step shapes, edits that stay in the KV cache | [generation-and-helpers.md](references/generation-and-helpers.md#the-values-under-generate) |
| collect activations at one position over many prompts, or a target word's probability mass | [generation-and-helpers.md](references/generation-and-helpers.md#nnterpnnsight_utils-activations-over-prompts) |
| run for throughput or as a served engine (`StandardizedVLLM`), or shard a model too big for one GPU | [generation-and-helpers.md](references/generation-and-helpers.md#the-vllm-engine-standardizedvllm) |
| run on NDIF with `remote=True` (the server needs nnterp installed) | [generation-and-helpers.md](references/generation-and-helpers.md#remotetrue-the-contract) |
| know where a family puts a contribution, tuple blocks, the root values and layouts; size a tensor without a family's config key (`nnterp.axis_size`) | [values.md](references/values.md) |
| read a `support()` reason or a `SourceNotAvailable`; guard a script | [availability-and-support.md](references/availability-and-support.md) |
| look up an exported name, or the 98 families grouped by quirk | [api-reference.md](references/api-reference.md) |
| load a checkpoint with no family (it loads with a best-effort default and a `UserWarning`; `support()` says what it serves), or add a family or a value of your own | the [extending guide](references/extending.md) |

nnterp's repo (https://github.com/ndif-team/nnterp) has a page per feature under
`docs/usage/` and a row per family in `docs/reference/families.md`; the site serves every
page as Markdown (https://ndif-team.github.io/nnterp/llms.txt), and its encyclopedia
(https://ndif-team.github.io/nnterp/encyclopedia/) draws each family's block.
