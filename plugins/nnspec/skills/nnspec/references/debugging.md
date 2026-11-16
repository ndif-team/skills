# Debugging

A run either refuses, printing one `nnspec:` line and writing nothing, or finishes. This guide reads a refusal,
finds what is wrong when nothing was refused, and makes two runs agree.

## How a refusal reads

```text
nnspec: <where>: <what>
```

`<where>` says which stage refused, and so what is known at that point:

| `<where>` | refused | for example |
|---|---|---|
| a document path: `steps.…`, `sites.…`, `data.…`, `saves[i]`, `model` | at compile, before any model loads; `nnspec compile` catches it | `steps.patched_ld.of: 'patched.logit' names no value produced before this step; known: clean.logits, …` |
| none: a flag or an engine option | before the model loads | `device 'gpu': not a torch device ('cpu', 'cuda:0') or a device map transformers computes (…)` |
| `<step>.<entry>` or `<step>`, without `steps.` | after the model loads, before any forward | `clean.v_self: layer 12; the model has 12 layers` |
| `<step>: generate rejected …` | when the step runs; nothing is written | an option the model's `generate` does not take |

Most refusals name the fix: they list what is valid (`known: …`, `valid: …`), the row and the sweep point
(`row 1: …`, `at layers=12`), and the form to write instead. Copy the spelling from the message and compile
again. These need more than the message says:

| the refusal | what it does not say |
|---|---|
| `model.revision: Field required` | The revision is the checkpoint's commit hash. For a cached checkpoint it is the directory name under `~/.cache/huggingface/hub/models--<org>--<name>/snapshots/` |
| `… names no value produced before this step; known: none` | The name may be right and its step written later: steps run in the order their keys are written, and generated JSON can reorder them |
| `'Rome' is 2 tokens; the read has 1 position on that row; …` | The answer lacks its leading space: ` Rome` is one token. Where the bare word is also one token nothing is refused and the score is wrong ([answer columns](data.md#answer-columns)) |
| `' Colosseum' is 4 tokens; the read has 1 position on that row; …` | An answer of several tokens is scored by teacher forcing: append it to the input and read the positions that predict it ([teacher forcing](generation-teacher-forcing.md)) |
| `row 1: text ' Tower' is not in its input` | A `text` scope is one literal that every row must hold; for a word that differs by row use a `column` scope. A refused row cannot be skipped: change the data |
| `… does not exist under the corpus root; run the document that writes it first`; `no file data/… under the corpus root` | `file_path`, `from` and `path` resolve against the folder above `documents/`, not the working directory ([where output goes](running-and-engines.md#where-output-goes)) |
| `over: 'pos' is not list of str or null` | A reduction's `over` is a list, `["pos"]`; `argmax`'s is one name, `top_k`'s one name or a list |
| `operand '…' holds no value on row 2 (alignment_missing): …`, at run | A write's operand was read at a decode call that row never made: read at `{"frame": -1, "index": -1}`, or hold EOS off with `min_new_tokens` |
| `'<name>' was not saved; add it to saves`, from `nnspec table`, for a name that is in `saves` | A later run into the same `--out` replaced the manifest: give each run its own directory |
| a refusal of an attention component on the hooks engine | Hooks serves module boundaries only; run that document on nnterp and cross-check at `attention_premix` |

A document that is refused leaves the previous output in place, so after a fix check that `nnspec show`'s first
line has a new `started` time.

## Wrong numbers, nothing refused

Open `nnspec show OUT` first. Each part answers one question:

- **The first line and `served:`**: is this the run you mean (document, engine, batch size, padding, readout, `started`), and what dtype did the model run in?
- **The entry table**: where each read and write landed on each row; `—` is a row it did not reach.
- **The result table**: each value's `axes` and `shape`; `2 points` where you meant a 2 by 2 grid is two sweeps zipped under one label.
- **The step table**: `tokens` against `real` is the padding; a forward whose own write is swept, listed once with `all N points` under `serves`, ran one point for all.

| what you see | what it is | the fix |
|---|---|---|
| a patch's scores equal the clean run's to the bit | the entry's `ops` hold no edit, so it only reads; or a freeze listed after the patch at one site put the clean value back | restore the edit (`swap`, `add`, …); list the freeze first |
| a patch meant for one layer at a time changed every layer at once | a `layers` slice or list on a write writes at every layer it picks | `"layers": {"sweep": [...]}` |
| a grid of two sweeps came back as its diagonal | two unlabelled sweeps of one field share a label | its own `label` on each |
| a position sweep shows effects where the counterfactual was never read | the write's `pos` sweeps alone; the read stays put | sweep the read's `pos` over the same values, one label |
| a `mean` or `sum` came back with one number per layer | a reduction keeps a `layers` axis unless `over` names it | `"over": ["layers", "rows", "pos"]` |
| a metric near zero on every row, or scoring another token than the model says | the answer column lacks the leading space the model continues with | a `top_k` at the read, decoded, shows the token; write ` Paris` |
| `token_prob` higher than the probability of the token you meant | `strip` or `lower` makes each answer a class of tokens, and `token_prob` sums the class | drop the flag where one spelling is meant |
| a read or write on the first mention of a word, not the one you meant | `occurrence` defaults to 0, the first | `"occurrence": -1` for the last, `1` for the second |
| a teacher-forced score read at the prompt's own mention of the answer | the default occurrence finds the prompt's copy first | `"occurrence": -1`, the appended copy |
| a window swap between two inputs moved other tokens than you meant | one unscoped `pos` pairs index with index; rows of equal length are never checked | a scope on each side naming its own text |
| `pos: -1` on a chat reads a template marker; a read over every token includes a BOS | every input token is a position | a `turn` scope for a message's own tokens; `{"index": "1:"}` past a BOS |
| `argmax` or `top_k` over heads gives 0 and 1 where you selected heads 3 and 1 | along a selected axis, indices count within the `select` list | map each index back through the list |
| an accuracy or mean covers fewer rows than the dataset, or rows are NaN | a read missed those rows, which every reduction skips | count them ([rows a scope never reached](positions.md#rows-a-scope-never-reached)); in a generation `min_new_tokens`, or `{"frame": -1, "index": -1}` |
| a mean over each row's entity favours long entities | a reduction over a ragged read weighs positions | reduce `"over": ["pos"]` first, then over `rows` |
| a steered run equals the unsteered one at every dose | the direction is exactly zero: `normalize` leaves a zero vector zero | save the direction's `norm` before normalizing |
| a write changed the prompt and not the generated tokens | in a `generate` step a `pos` without a `frame` is the prefill only | a second entry at `{"frame": "1:"}`, or `"frame": ":"` |
| two runs of one `generate` step said different things | the checkpoint's generation config samples, unseeded | `"do_sample": false` |
| a step after a fit scores like an untrained featurizer | the bare name is the declared draw; the trained one is `<fit>.<name>` | name `<fit>.<name>` ([featurizers](featurizers.md#what-breaks)) |
| splicing an SAE in moved the output with no edit | `apply` replaces the value with the reconstruction, dropping the error term | an `encode` … `decode` bracket, which keeps it |
| a published SAE's latents or L0 look wrong | `centered`, `activation` and the layer in `site` are the document's word, unchecked against the file | state them from the release ([a published dictionary](featurizers-sae.md#splice-in-a-published-dictionary)) |
| a gradient map is plausible and does not predict patching | `of` names another entry than the metric you report; the `mul 0` null cannot tell | a central difference at one point ([check the gradient](patching-attribution.md#check-the-gradient)) |
| small effects at every head of a bfloat16 sweep | bfloat16 rounding of the model's own forward | recompute the points you report in float32 ([below](#the-float32-rule)) |
| the last bits of a value differ between two runs | another batch size, padding, readout, thread count or device | compare at one setting ([below](#reproducing-a-number)) |
| a fixed document still shows the old numbers | the fixed document was refused, and a refusal writes nothing | read the refusal; check `started` in `nnspec show` |

## The nulls

Each null changes nothing, so its value is known before the run:

| null | written as | must equal | a miss means |
|---|---|---|---|
| self-swap | `swap` with the base's own read, same site and `pos` | the clean run, to the bit | the read and the write are not one selection |
| dose 0 | `add` with `"scale": 0.0`, the first point of a dose sweep | the clean run, to the bit | something else writes at the site |
| scale 1 | `scale` with `"factor": 1.0` | the clean run, to the bit | something else writes at the site |
| identity permutation | a `permute` step with `"seed": null`, swapped in | the clean run, to the bit | the permuted read is not the one the write replaces |
| wiring check | `kl` of a `readout` step at the last layer against `logits` | 0, to float32 rounding | the lens is not the model's computation |
| gradient null | the metric's ops with `{"op": "mul", "with": 0}` appended, and a `grad` of that entry | exactly 0 | the gradient is not of the zeroed entry alone |
| `epochs: 0` | the same fit, training nothing | the draw, which the trained featurizer must beat | the fit learned nothing |

`documents/null_patch.json` is the self-swap beside a patch of block 6's output at the last token, on rows of
10, 10, 7 and 9 tokens:

```json
{
  "description": "Patch layer 6's residual stream at the last token from a counterfactual city prompt into the base prompt, beside the null control: the base's own value written back, which must equal the clean run.",
  "model": {"key": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e", "dtype": "float32"},
  "data": {"pairs": {"rows": [
    {"input": "The Eiffel Tower is in the city of", "cf_input": "The Colosseum is in the city of", "base_answer": " Paris", "cf_answer": " Rome"},
    {"input": "The Colosseum is in the city of", "cf_input": "The Eiffel Tower is in the city of", "base_answer": " Rome", "cf_answer": " Paris"},
    {"input": "Big Ben is in the city of", "cf_input": "The Brandenburg Gate is in the city of", "base_answer": " London", "cf_answer": " Berlin"},
    {"input": "The Brandenburg Gate is in the city of", "cf_input": "Big Ben is in the city of", "base_answer": " Berlin", "cf_answer": " London"}
  ]}},
  "sites": {"target": {"component": "layer_output", "layers": 6}, "head": {"component": "lm_head_output"}},
  "interventions": {"scored": {"logits": {"site": "head", "pos": -1}}},
  "steps": {
    "clean": {"kind": "forward", "input": "pairs.input", "interventions": ["scored", {"v_self": {"site": "target", "pos": -1}}]},
    "counterfactual": {"kind": "forward", "input": "pairs.cf_input", "interventions": {"v_cf": {"site": "target", "pos": -1}}},
    "null": {"kind": "forward", "input": "pairs.input",
             "interventions": ["scored", {"patch": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "clean.v_self"}]}}]},
    "patched": {"kind": "forward", "input": "pairs.input",
                "interventions": ["scored", {"patch": {"site": "target", "pos": -1, "ops": [{"op": "swap", "with": "counterfactual.v_cf"}]}}]},
    "clean_ld": {"kind": "op", "op": "logit_diff", "of": "clean.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "null_ld": {"kind": "op", "op": "logit_diff", "of": "null.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"},
    "patched_ld": {"kind": "op", "op": "logit_diff", "of": "patched.logits", "a": "pairs.cf_answer", "b": "pairs.base_answer"}
  },
  "saves": ["clean_ld", "null_ld", "patched_ld"]
}
```

```text
$ nnspec table out/null_patch clean_ld null_ld patched_ld
row  pos  pos_token  input                                   cf_answer  base_answer  clean_ld             null_ld              patched_ld
0    9    Ġof        The Eiffel Tower is in the city of       Rome       Paris       -2.4247055053710938  -2.4247055053710938  -2.0760040283203125
1    9    Ġof        The Colosseum is in the city of          Paris      Rome        -1.4871978759765625  -1.4871978759765625  -1.4206924438476562
2    6    Ġof        Big Ben is in the city of                Berlin     London      -1.9296493530273438  -1.9296493530273438  -0.9876708984375
3    8    Ġof        The Brandenburg Gate is in the city of   London     Berlin      -5.286407470703125   -5.286407470703125   -4.356742858886719
```

`nnspec table` prints each float in full, so equal printed values are equal bits, and the effect is `patched_ld`
less `null_ld`. With the patch's `ops` deleted, `patched_ld` prints the clean column on every row: the mistake
looks like a null that passed. On the same rows and site, a dose-0 `add`, a `scale` by 1.0 and a `swap` of the
identity permutation each print the clean column too. Value files of different names always differ, since a
file's header holds the value's name: compare values of different names in a table or in torch, and files byte
for byte only between two runs of one document.

## Reproducing a number

One document run at one batch size, padding, readout, device and thread count writes byte-identical value
files. Change any of them and a row can move in its last bits:

| setting | why it moves bits |
|---|---|
| `--batch-size` | another batch is a matmul of another shape, which sums in another order |
| `--padding` (`dataset`, `batch`, `sorted`) | a row padded to another width is another matmul; under `dataset` the input's longest row sets every row's width, so `take`, `skip` or a split can move a row too |
| `--readout at-reads` | the head runs at the read positions alone, a matmul of another shape |
| an op in an entry, not an op step | an entry's op runs per batch and a step's on the whole value, so a matmul (`readout`, `encode`) or a long reduction can round otherwise |
| a fit's `batch_size` | a fit's body and evals run at the fit's own, whatever `--batch-size` says: run an apply document at that size to reproduce an eval |
| thread count (`OMP_NUM_THREADS`) | the CPU splits a matmul's sum by thread |
| device, machine, `attn_implementation` | other kernels; a bfloat16 `model.dtype` rounds more ([below](#the-float32-rule)) |

The run record holds each setting; `nnspec show` prints all but the threads, `run.threads` in `manifest.json`.
To see a difference, change a row's padded width: under `--padding batch`, batch size 1 pads no row of
`null_patch.json` and 4 pads rows 2 and 3 to 10 tokens.

```text
$ nnspec run documents/null_patch.json --padding batch --batch-size 1 --out out/bs1
$ nnspec run documents/null_patch.json --padding batch --batch-size 4 --out out/bs4
$ for f in out/bs1/*; do [ "${f##*/}" = manifest.json ] || cmp -s "$f" "out/bs4/${f##*/}" || echo "$f differs"; done
out/bs1/clean_ld.safetensors differs
out/bs1/null_ld.safetensors differs
out/bs1/patched_ld.safetensors differs
```

Row 3's `clean_ld` is `-5.2863922119140625` at batch size 1 and `-5.286407470703125` at 4, and its `null_ld`
moves with it; a second run at 4 is byte-identical to the first. Under the default padding the same document
writes identical files at batch sizes 8 and 2 here, since no row's width changes; whether the batch size alone
moves bits depends on the thread count, so compare two runs, never infer it.

**To the bit** holds for a rerun at one setting on one machine and for a null against its clean run. Across
settings, machines, `tp` or vLLM expect rounding, and a fit's trained values can move further, since Adam
scales a rounding-sized gradient up to its step size.

### The float32 rule

Metrics, reductions and arithmetic compute in float32; a read keeps the model's dtype; an edit computes in
float32 and rounds once to the value's dtype. No op undoes the model's own rounding. GPT-2's logits are near
-100, where bfloat16's step is 0.5, so `null_patch.json` at `"dtype": "bfloat16"` prints:

```text
row  pos  pos_token  input                                   cf_answer  base_answer  clean_ld  null_ld  patched_ld
1    9    Ġof        The Colosseum is in the city of          Paris      Rome        -1.5      -1.5     -1.5
2    6    Ġof        Big Ben is in the city of                Berlin     London      -1.5      -1.5     -1.0
```

The null still equals the clean run, and row 1's float32 effect (-1.487 to -1.421) is gone. A null that passes
in bfloat16 certifies the wiring, not small effects: recompute every effect you report near the dtype's step in
float32.

## Cross-checking engines

The hooks engine reaches the same modules through plain forward hooks, and `--engine ndif -o remote=local` makes
the ndif engine's trip in this process with no server. Where they agree with nnterp, the result does not depend
on how the model was reached:

```text
$ nnspec run documents/null_patch.json --out out/x_nnterp
$ nnspec run documents/null_patch.json --engine hooks --out out/x_hooks
$ nnspec run documents/null_patch.json --engine ndif -o remote=local --out out/x_ndif
$ for e in hooks ndif; do for f in out/x_nnterp/*; do [ "${f##*/}" = manifest.json ] || cmp "$f" "out/x_$e/${f##*/}"; done; done
$
```

`cmp` prints nothing: every value file and `tokens.json` match, and only the manifests differ, since they
record the engine and the time. That every read and write landed alike is
`nnspec.output.manifest("out/x_nnterp")[0]["steps"] == nnspec.output.manifest("out/x_hooks")[0]["steps"]`, `True`
here. Hooks serves module boundaries only, so cross-check an attention result at `attention_premix`, and
compare engines under one `attn_implementation`.

## Out of memory

A step that runs out of CUDA memory stops the run, and is the one refusal that writes output: the steps before
it are saved, the refusal names the step, the batch size and every value the run holds, `nnspec show` prints a
`stopped:` line, and `nnspec table` refuses a save the run never reached. Rerun with a smaller `--batch-size`
(a fit's refusal names the fit's own `batch_size`), or name and save fewer values, since every value a step
binds stays on the device until the run ends ([memory](running-and-engines.md#memory)).
