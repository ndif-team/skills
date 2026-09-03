"""Advanced code tasks — sessions, edits, skip, scan, barriers, and patterns."""

import torch

from ..registry import Difficulty, Task, register_task
from ._common import GPT2_SETUP, as_tensor, has_shape


def _register(id, name, prompt, verify, description, tags, reference, setup=GPT2_SETUP):
    register_task(
        Task(
            id=id,
            name=name,
            difficulty=Difficulty.ADVANCED,
            prompt=prompt,
            setup_code=setup,
            verify=verify,
            expected_output_description=description,
            tags=tags,
            reference_solution=reference,
        )
    )


def verify_sessions(result: dict) -> bool:
    value = result.get("similarity")
    if value is None:
        return False
    return -1.01 <= float(as_tensor(value)) <= 1.01


_register(
    "advanced_01_session",
    "Share a value between traces",
    """Write nnsight code that uses a session to run two traces —
"The Eiffel Tower is in" and "The Colosseum is in" — captures the final block's
last-position activation from each, and saves their cosine similarity as
`similarity`.

The activation from the first trace must be used inside the second trace without
being returned to your process in between.""",
    verify_sessions,
    "similarity scalar in [-1, 1]",
    ["session"],
    """
with model.session():
    with model.trace("The Eiffel Tower is in"):
        a = model.transformer.h[-1].output[0, -1]
    with model.trace("The Colosseum is in"):
        b = model.transformer.h[-1].output[0, -1]
        similarity = torch.nn.functional.cosine_similarity(a, b, dim=0).save()
""",
)


def verify_edit(result: dict) -> bool:
    edited, original = result.get("edited_logits"), result.get("original_logits")
    if edited is None or original is None:
        return False
    edited, original = as_tensor(edited), as_tensor(original)
    return not torch.equal(edited.float().cpu(), original.float().cpu())


_register(
    "advanced_02_model_editing",
    "Edit a copy, keep the original clean",
    """Write nnsight code that creates an EDITED COPY of the model in which block 5's
output is always zeroed, leaving the original model unmodified. Then trace
"Hello world" through both and save the final logits as `edited_logits` (from the
edited copy) and `original_logits` (from the untouched model).""",
    verify_edit,
    "edited_logits differs from original_logits",
    ["edit"],
    """
with model.edit() as (tracer, edited):
    edited.transformer.h[5].output[:] = 0

with edited.trace("Hello world"):
    edited_logits = edited.output.logits.save()
with model.trace("Hello world"):
    original_logits = model.output.logits.save()
""",
)


def verify_cache(result: dict) -> bool:
    first, last = result.get("cached_first"), result.get("cached_last")
    return has_shape(first, last_dim=768) and has_shape(last, last_dim=768)


_register(
    "advanced_03_caching",
    "Cache every module in one pass",
    """Write nnsight code that traces "Caching everything" while recording the
activations of ALL modules at once (without listing them individually), then
reads the cached output of transformer block 0 into `cached_first` and of
transformer block 11 into `cached_last`.""",
    verify_cache,
    "cached_first and cached_last tensors",
    ["cache"],
    """
with model.trace("Caching everything") as tracer:
    cache = tracer.cache()

cached_first = cache["model.transformer.h.0"].output
cached_last = cache["model.transformer.h.11"].output
""",
)


def verify_skip(result: dict) -> bool:
    skipped, normal = result.get("skipped_logits"), result.get("normal_logits")
    if skipped is None or normal is None:
        return False
    return not torch.equal(as_tensor(skipped).float().cpu(), as_tensor(normal).float().cpu())


_register(
    "advanced_04_skip_module",
    "Bypass a module's forward",
    """Write nnsight code that traces "Hello world" twice: once normally, saving the
final logits as `normal_logits`; and once where transformer block 6 does not run
at all — its output is replaced by block 5's output — saving the final logits as
`skipped_logits`.

Bypass the module rather than overwriting its output after it runs.""",
    verify_skip,
    "skipped_logits differs from normal_logits",
    ["skip"],
    """
with model.trace("Hello world"):
    normal_logits = model.output.logits.save()

with model.trace("Hello world"):
    fifth = model.transformer.h[5].output
    model.transformer.h[6].skip(fifth)
    skipped_logits = model.output.logits.save()
""",
)


def verify_scan(result: dict) -> bool:
    hidden = result.get("hidden_size")
    layers = result.get("n_layers")
    dispatched = result.get("was_dispatched")
    if hidden is None or layers is None:
        return False
    return int(hidden) == 768 and int(layers) == 12 and dispatched is False


_register(
    "advanced_05_scan_mode",
    "Shapes without running the model",
    """A second model object is available as `meta_model`, loaded WITHOUT dispatch
(no weights in memory). Write nnsight code that discovers, without running the
real model or loading its weights:

- the hidden size of transformer block 0's output, into `hidden_size`
- the number of transformer blocks, into `n_layers`

Then record `meta_model.dispatched` into `was_dispatched`, which must still be
False afterwards.""",
    verify_scan,
    "hidden_size 768, n_layers 12, was_dispatched False",
    ["scan"],
    """
with meta_model.scan("Hello world"):
    hidden_size = nnsight.save(meta_model.transformer.h[0].output.shape[-1])
    n_layers = nnsight.save(len(meta_model.transformer.h))

was_dispatched = meta_model.dispatched
""",
    setup=GPT2_SETUP + """
meta_model = TransformersModel("openai-community/gpt2")
""",
)


def verify_barrier(result: dict) -> bool:
    donor, receiver = result.get("donor_token"), result.get("receiver_token")
    if donor is None or receiver is None:
        return False
    return int(as_tensor(donor)) == int(as_tensor(receiver))


_register(
    "advanced_06_barrier_sync",
    "Transfer embeddings between invokes",
    """Write nnsight code with a single trace containing two invokes:

- "The Eiffel Tower is in the city of" — capture its token embeddings and save
  its final-position argmax token id as `donor_token`
- "_ _ _ _ _ _ _ _ _" — overwrite its token embeddings with the ones captured from
  the first invoke, and save its final-position argmax token id as
  `receiver_token`

Because the second prompt runs on the first one's embeddings, the two saved token
ids must be equal.""",
    verify_barrier,
    "donor_token == receiver_token",
    ["barrier", "batching"],
    """
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke("The Eiffel Tower is in the city of"):
        embeddings = model.transformer.wte.output
        barrier()
        donor_token = model.output.logits[0, -1].argmax().save()
    with tracer.invoke("_ _ _ _ _ _ _ _ _"):
        barrier()
        model.transformer.wte.output = embeddings
        receiver_token = model.output.logits[0, -1].argmax().save()
""",
)


def verify_logit_lens(result: dict) -> bool:
    tokens = result.get("layer_tokens")
    if not isinstance(tokens, list) or len(tokens) != 12:
        return False
    return all(as_tensor(t) is not None for t in tokens)


_register(
    "advanced_07_logit_lens",
    "Logit lens across every layer",
    """Write nnsight code that applies the logit lens to
"The Eiffel Tower is in the city of": for each transformer block, decode that
block's residual output through the model's final layer norm and unembedding, and
append the argmax token id at the final position to a list called `layer_tokens`.

`layer_tokens` must have one entry per layer (12), collected in a single forward
pass.""",
    verify_logit_lens,
    "layer_tokens list of 12 token ids",
    ["logit-lens"],
    """
with model.trace("The Eiffel Tower is in the city of"):
    layer_tokens = nnsight.save([])
    for block in model.transformer.h:
        decoded = model.lm_head(model.transformer.ln_f(block.output))
        layer_tokens.append(decoded[0, -1].argmax(dim=-1))
""",
)


def verify_steering(result: dict) -> bool:
    steered, base = result.get("steered_logits"), result.get("base_logits")
    if steered is None or base is None:
        return False
    return not torch.equal(as_tensor(steered).float().cpu(), as_tensor(base).float().cpu())


_register(
    "advanced_08_steering_vector",
    "Add a direction to the residual stream",
    """Write nnsight code that builds a steering direction as the difference between
the final-position activations of "I love this" and "I hate this" at transformer
block 6, then applies it (scaled by 2.0) to the final position of block 6 while
running "The movie was".

Save the resulting final logits as `steered_logits` and the unsteered logits for
the same prompt as `base_logits`.""",
    verify_steering,
    "steered_logits differs from base_logits",
    ["steering"],
    """
with model.trace() as tracer:
    with tracer.invoke("I love this"):
        positive = model.transformer.h[6].output[0, -1].detach().save()
    with tracer.invoke("I hate this"):
        negative = model.transformer.h[6].output[0, -1].detach().save()

direction = positive - negative

with model.trace("The movie was"):
    base_logits = model.output.logits.save()

with model.trace("The movie was"):
    model.transformer.h[6].output[:, -1, :] += 2.0 * direction
    steered_logits = model.output.logits.save()
""",
)


def verify_early_stop(result: dict) -> bool:
    early = result.get("early_hidden")
    return has_shape(early, last_dim=768)


_register(
    "advanced_09_early_stop",
    "Stop the forward pass early",
    """Write nnsight code that traces "Hello world", saves transformer block 3's
output as `early_hidden`, and then aborts the forward pass so that blocks 4-11
never run.""",
    verify_early_stop,
    "early_hidden tensor with hidden dim 768",
    ["stop"],
    """
with model.trace("Hello world") as tracer:
    early_hidden = model.transformer.h[3].output.save()
    tracer.stop()
""",
)


def verify_lens_subset(result: dict) -> bool:
    probs = result.get("paris_probs")
    if not isinstance(probs, list) or len(probs) != 3:
        return False
    values = [float(as_tensor(p)) for p in probs]
    return all(0.0 <= v <= 1.0 for v in values)


_register(
    "advanced_10_logit_lens_subset",
    "Track one token's probability across layers",
    """Write nnsight code that, for transformer blocks 6, 9 and 11 only, decodes the
residual stream through the final layer norm and unembedding on
"The Eiffel Tower is in the city of", converts the final-position logits to
probabilities, and appends the probability of the token " Paris" to a list called
`paris_probs` (3 entries, in that layer order).""",
    verify_lens_subset,
    "paris_probs list of 3 probabilities",
    ["logit-lens"],
    """
paris = model.tokenizer.encode(" Paris")[0]
with model.trace("The Eiffel Tower is in the city of"):
    paris_probs = nnsight.save([])
    for layer in (6, 9, 11):
        decoded = model.lm_head(model.transformer.ln_f(model.transformer.h[layer].output))
        paris_probs.append(decoded[0, -1].softmax(dim=-1)[paris])
""",
)


def verify_steer_generation(result: dict) -> bool:
    steered, base = result.get("steered_ids"), result.get("base_ids")
    if steered is None or base is None:
        return False
    steered, base = as_tensor(steered), as_tensor(base)
    if not hasattr(steered, "shape") or not hasattr(base, "shape"):
        return False
    return steered.shape[-1] >= 5 and base.shape[-1] >= 5


_register(
    "advanced_11_steering_during_generation",
    "Steer every generation step",
    """Write nnsight code that generates 5 tokens from "The movie was" twice: once
unmodified, saving the ids as `base_ids`; and once where a fixed direction
(`torch.ones(768, device=model.device) * 0.5`) is added to the final position of
transformer block 6 at EVERY generated step, saving the ids as `steered_ids`.""",
    verify_steer_generation,
    "base_ids and steered_ids, both with the generated tokens",
    ["steering", "generation", "iteration"],
    """
with model.generate("The movie was", max_new_tokens=5) as tracer:
    base_ids = tracer.result.save()

direction = torch.ones(768, device=model.device) * 0.5

with model.generate("The movie was", max_new_tokens=5) as tracer:
    for step in tracer.iter[:5]:
        model.transformer.h[6].output[:, -1, :] += direction
    steered_ids = tracer.result.save()
""",
)


def verify_head_ablation(result: dict) -> bool:
    scores = result.get("head_scores")
    if not isinstance(scores, list) or len(scores) != 12:
        return False
    values = [float(as_tensor(s)) for s in scores]
    return len({round(v, 4) for v in values}) > 1


_register(
    "advanced_12_head_ablation",
    "Ablate each attention head",
    """Write nnsight code that zero-ablates each of the 12 attention heads of
transformer block 5 one at a time on "The Eiffel Tower is in the city of", and
records the resulting final-position logit of " Paris" for each head into a list
`head_scores` (12 entries, in head order).

Heads occupy contiguous slices of the attention output projection's input. Use a
single forward pass.""",
    verify_head_ablation,
    "head_scores list of 12 differing values",
    ["attention", "ablation", "batching"],
    """
paris = model.tokenizer.encode(" Paris")[0]
n_heads = model.config.n_head
head_dim = model.config.n_embd // n_heads

with model.trace() as tracer:
    head_scores = nnsight.save([])
    for head in range(n_heads):
        with tracer.invoke("The Eiffel Tower is in the city of"):
            lo, hi = head * head_dim, (head + 1) * head_dim
            model.transformer.h[5].attn.c_proj.input[:, :, lo:hi] = 0
            head_scores.append(model.output.logits[0, -1, paris])
""",
)


def verify_edit_clear(result: dict) -> bool:
    during, after = result.get("during_edit"), result.get("after_clear")
    baseline = result.get("baseline_logits")
    if during is None or after is None or baseline is None:
        return False
    during, after, baseline = (as_tensor(x).float().cpu() for x in (during, after, baseline))
    return (not torch.equal(during, baseline)) and torch.equal(after, baseline)


_register(
    "advanced_13_inplace_edit_clear",
    "Install a persistent edit, then remove it",
    """Write nnsight code that:

1. traces "Hello world" and saves the final logits as `baseline_logits`
2. installs a PERSISTENT in-place edit on the model that zeroes block 4's output
3. traces "Hello world" again, saving the logits as `during_edit`
4. removes all edits from the model
5. traces "Hello world" once more, saving the logits as `after_clear`

`during_edit` must differ from `baseline_logits`, and `after_clear` must match it
exactly.""",
    verify_edit_clear,
    "during_edit differs from baseline, after_clear equals baseline",
    ["edit"],
    """
with model.trace("Hello world"):
    baseline_logits = model.output.logits.save()

with model.edit(inplace=True):
    model.transformer.h[4].output[:] = 0

with model.trace("Hello world"):
    during_edit = model.output.logits.save()

model.clear_edits()

with model.trace("Hello world"):
    after_clear = model.output.logits.save()
""",
)


def verify_conditional(result: dict) -> bool:
    fired = result.get("branch_taken")
    logits = result.get("logits")
    return isinstance(fired, bool) and has_shape(logits, last_dim=50257)


_register(
    "advanced_14_python_conditional",
    "Branch on a tensor value inside a trace",
    """Write nnsight code that traces "Hello world" and, inside the trace, checks
whether the mean of transformer block 6's output is greater than zero. If it is,
halve that block's final-position output. Record whether the branch was taken
into a boolean `branch_taken`, and save the final logits as `logits`.""",
    verify_conditional,
    "branch_taken boolean and logits tensor",
    ["conditionals"],
    """
with model.trace("Hello world"):
    hidden = model.transformer.h[6].output
    branch_taken = nnsight.save(bool(hidden.mean() > 0))
    if hidden.mean() > 0:
        model.transformer.h[6].output[:, -1, :] *= 0.5
    logits = model.output.logits.save()
""",
)
