# Refusals by first clause

Find the first clause of the `<what>` you saw, after its `clab: <where>: `. Every message below was printed by
clab on `openai-community/gpt2`, provoked by a small broken document; `<x>`-style names in your message are
your document's. The **refused** column says when:

- **before load**: before any weight loads. A document path in `<where>` (`steps.…`, `sites.…`), and
  `Plan(...).compile()` catches it without a model, unless the row says "as the engine loads".
- **after load**: the model and tokenizer are loaded, no forward has run; `<where>` is `<step>.<entry>` or
  `<step>`, and the message names the row or the sweep point.
- **at run**: the step was running; the steps before it ran, and nothing is written.

A refusal writes nothing, so the output directory keeps the last run that finished. A run that finishes with
wrong numbers is [symptoms.md](symptoms.md).

## The document's own fields

| first clause | refused | cause | fix |
|---|---|---|---|
| `unknown field 'position'; valid: site, pos, ops` | before load | a misspelled key, or a key on the wrong record | take the spelling from the `valid:` list |
| `Field required` (at `model.revision`) | before load | a required key left out: `model`'s `key` and `revision`, `steps` | add it |
| `unknown step kind 'forwards'; valid: forward, generate, op, fit, load` | before load | `kind` names no step kind | one of the listed kinds |
| `unknown op 'logitdiff'; valid: swap, add, …` | before load | `op` names no op | the spelling from the list; `clab describe ops` says what each takes |
| `'logit_diff' takes no 'token'; its arguments: a, b` | before load | an argument of another op | the listed arguments |
| `'logit_diff' needs 'b'` | before load | a required argument left out | add it |
| `'batch_size' is a derived field, which compile writes; a step sets none` | before load | a field clab fills (a forward's `batch_size`, an entry's `positions`) | delete it; `--batch-size` sets the batch size |
| `'positions' is no argument a document gives: the run supplies it` | before load | an op argument the run supplies | delete it |
| `'reads': a step's reads and writes are one 'interventions' value` | before load | reads and writes in separate keys | one `interventions` dict; an entry with an edit writes |
| `num_beams 2; it stays 1: beams change the rows each forward call sees` | before load | a generate option clab holds at one value | leave it out |

Several field errors print one per line, the first prefixed `clab:`: `clab: model.revision: Field required`,
then `steps.clean: unknown step kind 'forwards'; …`.

## Sites, positions and sweeps as written

| first clause | refused | cause | fix |
|---|---|---|---|
| `unknown component 'residual'; the components clab addresses …` | before load | a component nnterp has no name for | one from the list; `clab describe components` says what each holds |
| `layer_output is a block's value and needs 'layers'` | before load | a block's component without `layers` | give `layers` |
| `lm_head_output is the model's own value and takes no layers` | before load | `layers` on `lm_head_output` or `logits` | delete `layers` |
| `'-1' is a string, which is not read as a number` | before load | a `pos` written as a string | an int (`-1`), or a slice inside an object (`{"index": "-2:"}`) |
| `a scope is one of {'column': …}, {'text': …}, {'turn': …}` | before load | a scope with none of the three, or two | exactly one |
| `unknown field 'word'; valid: column, text, turn, occurrence, strip, lower, until, before` | before load | a misspelled scope key | the listed keys |
| `a column's strip goes on its reference` | before load | `strip` or `lower` beside a `column` scope | `{"column": {"column": "<dataset>.<column>", "strip": true}}` |
| `frame 1: a forward makes one call, the prefill` | before load | a decode frame on a `forward` | frame 0, or a `generate` step |
| `label 'layers' is also the sweep at sites.target.layers, over [3, 6]; one label is one axis` | before load | two sweeps sharing a label (a field's name is the default) over other values | a `label` on one to cross them, or the same values to pair them |

At a sweep point the path gains the point: `steps.patched.interventions.logits.pos.frame at frame=1: frame 1: …`.

## Names and data

| first clause | refused | cause | fix |
|---|---|---|---|
| `'patched.logit' names no value produced before this step; known: …` | before load | a misspelled name, or a producer later in `steps` (`known: none`) | a name from `known:`; steps run in the order written |
| `no dataset 'pair'; datasets: pairs` | before load | an `input` or reference to an undeclared dataset | a listed dataset |
| `dataset 'pairs' has no column 'prompt' on every row; columns: …` | before load | a column some or all rows lack | a listed column |
| `'cf_answer' is not <dataset>.<column> over a dataset` | before load | a column named without its dataset | `pairs.cf_answer`; text the same on every row is `{"text": …}` |
| `no site 'heads'; declared: target, head` | before load | a site name `sites` does not declare | a declared site, or a site written inline |
| `'patched.logit' names no step's value; a save is a step or a step's member` | before load | a `saves` entry no step makes | a step's name or `<step>.<entry>` |
| `'ld' is saved twice` | before load | a name listed twice in `saves` | list it once |
| `no file data/cities/train.json under the corpus root` | before load | a `path` dataset with no `data/` directory above the document | inline `rows`, or a `data/` directory above the document |

## Operands

An operand's axes are among the value's, in the value's order; `pos` never broadcasts; a `select` list pairs
by position; it holds every layer the write acts at; it is one dataset's rows. The why is in
[SKILL.md](../SKILL.md#an-operand-does-not-fit).

| first clause | refused | cause | fix |
|---|---|---|---|
| `'counterfactual.v_cf' is laid out ['rows', 'pos', 'vocab_size']; 'swap' acts on a value laid out ['rows', 'pos', 'hidden_size']` | before load | an operand with an axis the value lacks: here read at the head, written into the stream | read it at a site of the value's layout, or reduce the extra axis first |
| `'counterfactual.v_cf' holds 6 position(s) per row and the value 'swap' acts on holds 1` | before load | two static widths that differ (`{"index": "0:6"}` against `-1`) | read both at as many positions |
| `'counterfactual.v_cf' has no width compile can fix` | before load | an open-ended index (`"5:"`), an `until` scope or several occurrences on one side | a static index, one occurrence of a scope, or a `mean` over `pos` first |
| `'counterfactual.v_cf' holds num_heads [1, 3], and this write's selection holds [3, 1]; a select list is positional` | before load | the same heads in another order at the read and the write | one list at both, best one named site |
| `'counterfactual.v_cf' holds layers [6, 7], and this op acts at layers [5, 6]` | before load | the operand lacks a layer the write acts at | read it at the write's layers |
| `'pairs.input' has a row per example of 'pairs' and 'counterfactual.v_cf' of 'other'; an op takes one dataset's rows` | before load | values of two datasets in one op, even two over the same rows | read both inputs from one dataset |
| `'patched.logits' is this entry or one this forward makes after it; an operand names an earlier entry` | before load | a same-forward operand listed later | list it earlier, or read it in an earlier forward |
| `operand 'patched.later' is read at layer_output layer 8, which the nnterp engine's forward reaches no earlier than this write's layer_output layer 6` | after load | a same-forward operand at a later site | read it in a forward before this one |
| `operand 'counterfactual.v_cf' holds 4 positions on rows 0-1, 3 positions on row 2 and 2 positions on row 3; a scoped read pairs with another forward's value when every row's scope holds the same number of tokens` | after load | a scope whose rows hold different numbers of tokens (entities of 4, 3, 2 tokens) | `"index": -1` inside the scope on both sides, or reduce over `pos` |
| `operand 'counterfactual.v_cf' was read on pairs.cf_input, row 2 of which is 9 tokens; pairs.input has 7, so pairs.cf_input's position 5 (' in') pairs with pairs.input's position 5 (' city')` | after load | one unscoped `pos` on two inputs of different lengths | a scope on each side naming the token in its own row |

## Entries and ops

| first clause | refused | cause | fix |
|---|---|---|---|
| `'scale' is an edit, which writes back, and it follows 'readout'` | before load | an edit after a transform, metric or reduction in one entry | the edit first, or the transform in a later entry at the site |
| `'mean' aggregates over rows, and inside an entry it sees one batch; make it a step after the forward` | before load | a reduction over `rows` inside an entry | an op step after the forward |
| `'mean' over 'layers': inside an entry an op acts at each layer` | before load | a reduction over `layers` inside an entry | an op step with `"over": ["layers"]` |
| `a reduction cannot remove 'layers'; the value is ['rows', 'pos']` | before load | `over` names an axis the value lacks | an axis of the printed layout |
| `readout projects a residual-stream value; attention_premix is not one` | before load | `readout` on a value that is not a stream | one of the listed stream components |
| `'top' is a top_k's value, several tensors (its ids, their probs and their tokens)` | before load | an op over a whole `top_k` | name a member: `top.ids`, `top.probs` |

## Gradients

| first clause | refused | cause | fix |
|---|---|---|---|
| `'patched.metric' names no entry of this forward; grad's of is an entry of its own forward, by its bare name` | before load | `of` with the step's name, or an entry of another step | the bare name (`"metric"`), an entry of the same forward |
| `grad follows 'swap', an edit, and an entry that takes a gradient writes nothing` | before load | an edit before `grad` in one entry | the write in its own entry, listed before the `grad` at the site |
| `grad is taken in a forward, not a generate` | before load | `grad` in a `generate` step | a forward over the prompt with the generated text appended (a list input) |
| `grad's of 'early' is read at layer_output layer 2, which the nnterp engine's forward reaches no later than this entry's layer_output layer 6` | after load | `of` at a site the forward reaches before the gradient's | an `of` the site feeds, at a later site such as the head |

## Positions in a row

The prompt alone decides where a prefill `pos` lands, so every row is checked after the tokenizer loads. A
refused row cannot be skipped: change the data (`take`, `skip`, other rows).

| first clause | refused | cause | fix |
|---|---|---|---|
| `row 1: text ' John gave' is not in its input` | after load | a `text` scope, one literal, absent from a row | a `column` scope for a value that differs by row; a list of texts for alternatives |
| `row 0: column 'ioi.subject' (' John') occurs 2 times in the row's input; it has no occurrence 2` | after load | an `occurrence` past the row's count (counted from 0) | `1` for the second, `-1` for the last |
| `row 0: index 40 does not fit the row's 10 positions at this frame` | after load | a static index wider than the shortest row | an index every row holds |
| `row 2: the write's positions hold no token of this row; a write needs one to write` | after load | a write at an open-ended index (`"9:"`) past a short row | an index every row holds; a read there is `alignment_missing` instead |
| `row 0: text 'When' begins the row, at position 0, which no position comes before` | after load | `before` on a row's first token; GPT-2 adds no BOS | scope a later token |

## Answers

A metric tokenizes the answer as written, without special tokens, and scores its token i at the read's
position i; every row is checked after the tokenizer loads.

| first clause | refused | cause | fix |
|---|---|---|---|
| `pairs.cf_answer row 0: ' Colosseum' is 4 tokens; the read has 1 position on that row` | after load | an answer of several tokens against one position | teacher-force it (append it to the input, read `before` its scope with `"occurrence": -1`), or a one-token answer |
| `facts.bare row 1: 'Rome' is 2 tokens; the read has 1 position on that row` | after load | the leading space left out, where the bare word splits | ` Rome`, as the model continues the prompt |
| `pairs.cf_answer row 0: [' Rome', ' Paris'] stands for 2 ids (a set, or a class under strip or lower), which have no one logit for logit_diff` | after load | a set where the metric reads one logit (`logit_diff`, `token_logit`) | `token_prob`, which sums the set, or one spelling per row |
| `pairs.cf_answer row 0: its members are ' Rome' 1, ' Colosseum' 4 tokens; a set's members are one width` | after load | set members of different token counts | members of one width, or one column per alternative |
| `days.answer row 0: position 3 predicts ' 3' in the input, which is not token 0 of '3' as it tokenizes alone` | after load | a teacher-forced prompt ending in a space, which GPT-2 merges into the next token | the space on the answer (` 3`), not on the prompt |

## Rows a read did not reach

A read is never refused for a row it misses; a write whose operand is missing on a row it lands on is.

| first clause | refused | cause | fix |
|---|---|---|---|
| `operand 'g.r' holds no value on row 2 (alignment_missing): 'g' made 2 calls on row 2 (0 to 1), and 'g.r' reads call 3` | at run | a write's operand read at a decode call a row never made | read at `{"frame": -1}` (each row's last), or `min_new_tokens` that reaches the call |
| `'g.r' is read at decode call 6, and 'g' makes calls 0 to 5 (max_new_tokens 6), so it holds no value on any row` | before load | a decode call at or past `max_new_tokens` | a call below it, or a larger `max_new_tokens` |

## The model and the engine

| first clause | refused | cause | fix |
|---|---|---|---|
| `model.device: the nnterp engine cannot honor it: where the model runs is a run option: --device` | before load, as the engine loads | a placement key in `model` | `--device` on the command line |
| ``model.torch_dtype: the nnterp engine cannot honor it: the dtype is spelled `dtype` `` | before load, as the engine loads | the transformers spelling | `"dtype"` |
| `model.key 'openai-community/gpt2' at revision '…': the nnterp engine cannot load it: HF_HUB_OFFLINE is set and the Hugging Face cache lacks it` | before load, as the engine loads | offline, a revision the cache does not hold | the cached revision, or one run online first |
| `layer 12; the model has 12 layers` | after load | a layer past the model (layers count from 0) | at most `num_layers - 1` |
| `select num_heads 12 outside the model's 12` | after load | a `select` past the model's size | an index below the size |
| `the nnterp engine does not serve attention_probabilities: read inside the eager attention forward, but this model runs 'sdpa'` | after load | the attention interior under `sdpa` | `"attn_implementation": "eager"` in `model` |
| `the hooks engine does not serve attention_probabilities: nnterp reads it inside a module's forward` | after load | a value inside a module, which hooks cannot reach | run that document on nnterp; cross-check at module boundaries |

## Flags, running and reading back

| first clause | refused | cause | fix |
|---|---|---|---|
| `no document at 'documents/nope.json'` | before load | the path given to `clab run` | the document's path |
| `batch_size 0: a batch holds one row or more` | before load | `--batch-size` below 1 | 1 or more |
| `padding 'left': one of 'dataset', 'batch', 'sorted'` | before load | a `--padding` clab does not have | a listed value |
| `device 'gpu': not a torch device ('cpu', 'cuda:0') or a device map transformers computes` | before load | an unknown device | `cpu`, `cuda:0` or `auto` |
| `device 'cuda:0': no CUDA device is visible` | before load | a GPU where none is visible | `--device cpu`, or a visible GPU |
| `unknown engine 'hook'; valid: nnterp, hooks, ndif, vllm` | before load | `--engine` names no engine | a listed engine |
| `the nnterp engine takes no 'remote'; its options: tp` | before load | an `-o` option of another engine | the engine's options, in `clab describe engines` |
| `-o 'tp': an option is KEY=VALUE` | before load | an `-o` without `=` | `-o tp=2` |
| `tp 2 shards the model over 2 processes, one per rank, that torchrun starts` | before load | `-o tp` outside torchrun | the `torchrun` command the message prints |
| ``answer: generate rejected 'max_new_tokns': The following `model_kwargs` are not used by the model`` | at run | an option the model's `generate` does not take | the option's spelling in transformers |
| `'patched.logits' was not saved; add it to saves` | `clab table` | a name not in `saves`, or a run whose `--out` a later run reused | add it to `saves` and rerun, each run to its own `--out` |
| `no run at 'out/nowhere' (no manifest.json); run the document first` | `clab show`, `clab table` | a directory no run wrote | the run's `--out` |

A step that runs out of CUDA memory is the one refusal that writes output ([SKILL.md](../SKILL.md#out-of-memory)).
