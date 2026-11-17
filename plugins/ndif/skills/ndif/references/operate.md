# Running models on a server

`SKILL.md` has the everyday commands, the three levels, and the rules that
deploy is additive and that a model key is the identity. This page is the rest
of the operator's surface. `ndif <command> --help` lists every flag.

## The model key

```text
nnsight.modeling.transformers.TransformersModel:{"repo_id": "openai-community/gpt2", "revision": null, "task": "text-generation"}
```

The nnsight wrapper class, then JSON with the Hub's canonical repo id
(`gpt2` becomes `openai-community/gpt2`), the revision, the pipeline task, and
`"trust_remote_code": true` when set. Each part makes a different deployment.

- **Every model command resolves a checkpoint to a key through the Hub**, so it
  needs network where it runs, and `HF_TOKEN` for a gated repo that is not in
  the cache yet. Left unset, the task is inferred exactly as an nnsight client
  with no `task=` infers it, so the two agree.
- **`dtype`, GPU count and padding are not in the key.** Two replicas of one
  key deployed differently answer under the same name.
- **"Model not deployed" on the client is almost always a key mismatch.**
  Compare the key the client computed (`model.to_model_key()`) with
  `ndif status --json-output | jq -r '.deployments[] | select(.model_key) | .model_key'`.

## Deploy, scale, restart, evict, export

```bash
ndif deploy meta-llama/Llama-3.1-8B --pinned --gpus 1    # one more replica, placed as you say
ndif scale meta-llama/Llama-3.1-8B -n 2                  # two more, copied from a running replica
ndif restart openai-community/gpt2 --replica a1b2c       # kill the actor, wait for it to reload in place
ndif evict openai-community/gpt2 --replica a1b2c         # one replica: demoted to WARM
ndif evict --all                                         # every HOT model, pinned included
ndif export --stdout > models.yaml                       # the HOT set as a file for `deploy -f`
```

- **`deploy`** waits for each new replica with no deadline, since a download
  has no sensible upper bound. It prints `✓ <model_key> [a1b2c]: ready` (the
  bracket is the replica id) or `✗ ... initialization failed - <the load
  error>`, then any replicas the controller evicted to make room. Placement
  flags (`--dtype`, `--gpus`, `--size-bytes`, `--padding-factor`,
  `--padding-bias`, `--max-tp`) are in
  [sizing and placement](operate-sizing-and-placement.md).
- **`scale`** differs from `deploy --replicas` only in where unset options
  come from: `deploy` takes the controller's defaults, `scale` copies the actor
  class, dtype, GPU count and timeout of a replica already running. Use it to
  grow a model that was deployed with non-default settings. A WARM replica of
  the key is promoted under its old id instead of loading a new one.
- **`restart`** reloads the weights on the same GPUs. A request dispatched to
  the replica meanwhile is re-queued or fails with
  `Error submitting request to model deployment.`
- **`evict`** without `--replica` removes every HOT and WARM replica of the
  key. With `--replica` a HOT replica is demoted to WARM when the node's
  host-RAM budget has room, and removed otherwise. `--all` reads the HOT set,
  so a model that is only WARM has to be named.
- **`status`** lists COLD models by name with `--show-cold`, and `--watch` on
  `status` or `queue` refreshes every two seconds.
- **`export` is not a full snapshot.** It writes one entry per key with
  `model_key`, `dtype` and `actor_class`, plus `pinned`, `replicas`, `gpus` and
  the timeout where they are not the default. Padding, `size_bytes` and
  `max_tp` are not stored on a deployment and are lost, and replicas of one key
  that differ are merged. Keep `ndif status --json-output` beside the file.

## models.yaml

```yaml
models:
  - openai-community/gpt2
  - checkpoint: Qwen/Qwen2.5-0.5B
    pinned: true
    replicas: 1
    dtype: bfloat16
    padding_factor: 0.3
    execution_timeout_seconds: 600
```

Other fields: `revision`, `task`, `trust_remote_code`, `gpus`, `size_bytes`,
`padding_bias`, `max_tp`, `actor_class`, `envoy_class` (the nnsight wrapper
class, `TransformersModel` by default), `model_key`. An unknown field is
dropped without a word. `ndif deploy -f models.yaml` adds what the
file lists. `--sync` makes the cluster match it: every HOT key not in the file
is evicted, pinned or not, surplus replicas are trimmed, and only the shortfall
is deployed.

## Loading at boot

`NDIF_DEPLOYMENTS` on the `ray` service is a `|`-separated list of model keys
that the controller deploys, pinned, when it starts:

```bash
-e 'NDIF_DEPLOYMENTS=nnsight.modeling.transformers.TransformersModel:{"repo_id": "openai-community/gpt2", "revision": null, "task": "text-generation"}'
```

Each entry is used verbatim. A bare repo id deploys nothing and leaves
`Model openai-community/gpt2 failed to evaluate` in the controller log; a key
without the task deploys a model no client's key matches. Copy keys from the
`jq` line above.

## Pinning, eviction and autoscaling

Who is pinned: `ndif deploy --pinned`, every `NDIF_DEPLOYMENTS` key and every
dashboard schedule entry. A model loaded by a first request or by the
autoscaler is not.

- **The controller evicts only to place something else**, and only replicas
  that are unpinned and older than `NDIF_MINIMUM_DEPLOYMENT_TIME_SECONDS`
  (3600). When the incoming model is pinned the age rule is waived. Among
  candidates it takes the smallest first. So a deploy on a full node fails
  `CANT_ACCOMMODATE` because the occupant arrived 40 minutes ago: deploy
  `--pinned`, or evict the occupant yourself.
- **An evicted replica goes WARM** when host RAM allows, keeping its replica id,
  and comes back from RAM on the next deploy, scale or request for its key.
- **To pin or unpin a replica that is already serving**: `ndif evict
  <checkpoint> --replica <id>`, then `ndif scale <checkpoint> --pinned` to pin
  or `ndif scale <checkpoint>` to unpin. The same replica comes back from RAM
  with the new flag, and that is how a model pinned at boot is unpinned without
  a restart. Nothing changes the flag in place: `deploy` and `scale` only add
  replicas, and `--sync` compares replica counts, not flags.
- **Pinning does not survive a restart of `ray`**, stop `ndif evict`, or
  reserve room for more replicas.
- **Autoscaling** adds a replica to a model whose oldest queued request has
  waited more than `NDIF_AUTOSCALING_WAIT_THRESHOLD_S` (30 s), up to
  `NDIF_AUTOSCALING_MAX_REPLICAS` (3), then waits `NDIF_AUTOSCALING_BACKOFF_S`
  (120 s). It never removes one: evict the extras yourself.

## The queue, and killing a request

`ndif queue` asks the dispatcher. A model appears once a request has arrived
for it and stays listed when idle. Its `Replicas:` count can lag an eviction,
because a replica that is gone is dropped on the next dispatch to it.

```text
  openai-community/gpt2 (text-generation)
    Status: READY
    Replicas: 1
    Queue depth: 2
    ⚙ [05052] executing e41f2efa7200498190e2026fcbf63027 (for 0:00:06)
    Queued: f49be56aea54492ea43442ad50efd406, c4f3f5eb0ff545be8620bdbb973a2782
```

`ndif kill <request_id>` takes one id.

- **Queued**: removed. The client gets `ERROR` `Request cancelled by operator.`
- **Running**: the client gets the same error at once and the actor is told to
  stop the block. The replica stays in service and the next request runs
  normally. A non-blocking request's stored status ends as `ERROR` `Your job
  was cancelled or preempted by the server.`
- **The stop lands at the next Python instruction.** A block inside one long
  native call (a single huge CUDA kernel) finishes that call first, on a thread
  the replica has already abandoned. On a sandbox actor the runner process is
  terminated, which is immediate.
- **An unknown id** exits non-zero. `No response from the dispatcher` means the
  API is down or its dispatcher has died.

A request holds its replica for as long as its block runs: there is no
execution timeout unless `NDIF_DEFAULT_EXECUTION_TIMEOUT_SECONDS` or a
per-model `execution_timeout_seconds` is set. With one, the client gets
`Your job exceeded the execution timeout of 20.0s.` and the replica moves on.

## dtype and quantization

`--dtype` takes a torch dtype (`bfloat16`, `float32`) or a quantization name
(`nf4`, `int4`, `4bit`, `fp4`, `int8`, `8bit`, `fp8`). It decides both how the
weights load and how the model is sized. Client code does not change and cannot
ask for it.

- **The estimate for a quantized model runs low.** It is parameters times the
  nominal width, but embeddings, norms and the head stay 16-bit.
  Llama-3.2-1B at `nf4` reserves 1,178 MiB and takes 1,385 MiB on the card.
  Give it a measured `--size-bytes`.

## Tensor parallelism

Not run for this guide: it needs a model across two or more GPUs.

Off unless `NDIF_TP_MODEL_ACTOR_CLASS` is set on the `ray` service, to
`ndif.services.ray.tp.model.TPModelActor`, or
`ndif.services.ray.tp.model.SandboxedTPModelActor` on a server that takes
untrusted requests. Unset, a multi-GPU model is spread layer by layer with
accelerate and `max_tp` does nothing.

- **A replica is placed tensor-parallel** when it gets more than one GPU and
  the model shards evenly into exactly that many. A model needing three cards
  that shards eight ways is given four.
- **`--gpus N` must divide the model's largest degree**, or the deploy is
  refused: `asked for 3 GPUs, but this model shards at most 8 ways...`.
  `--max-tp 0` keeps one model off tensor parallelism.
- **A tensor-parallel replica cannot go WARM.** Eviction removes it outright,
  so pin one you want to keep.
- **It replaces the actor class**, sandbox included: the plain `TPModelActor`
  runs every block in-process.
- **`scale`, not `deploy`,** to add replicas, so they shard like the first.

`docs/operating/models-and-deployment.md#tensor-parallelism`.

## PEFT adapters

Adapters are per request, not per deployment. A client that loads
`TransformersModel("openai-community/gpt2", task="text-generation",
peft="peft-internal-testing/gpt2-lora-random")` computes the same model key as
the base model, and one deployment serves both; the actor swaps the adapter
before the run. The server fetches the adapter from the Hub by id, so a local
adapter directory on the client is invisible to it. The client needs `peft`
installed too.

## Packages a block may import

A block that imports a package the server lacks fails with a traceback ending
`ModuleNotFoundError: No module named '<package>'` at the user's import line,
or with `Your request payload could not be read (ModuleNotFoundError: ...)`
when an object the client shipped refers to it. The image installs `datasets`,
`einops`, `pillow`, `diffusers`, `scipy` and `nnterp` for blocks to use (the
package's `ext` extra). Add another with `docker exec ndif pip
install <package>`, which serves until the container is removed, or keep it
with a two-line Dockerfile (`FROM ndif/ndif`, `RUN pip install <package>`);
replicas loaded afterwards see it.

## trust_remote_code

It is a field of the model key, so it is decided where the key is made.

- **Deploying a repository that ships its own modelling code needs
  `--trust-remote-code`** (or `trust_remote_code: true` in the file, or the
  checkbox in the dashboard). Without it the deploy fails before any GPU is
  touched, with Hugging Face's "requires you to execute the code" error.
- **A client that loads with `trust_remote_code=True` sends a key that carries
  it.** If nothing is running under that key, the first request deploys it only
  when its API key has the `trust_remote_code` or `trusted` tag; with auth off
  every request is trusted. Otherwise every queued request for the key gets
  `Could not deploy this model. This model runs code from its own repository
  (trust_remote_code=True)...`
- **A running model serves every caller.** Evict it if untagged keys should not
  use it.

## By exact key

`evict`, `restart` and `scale` act on the key their flags produce. For any
other key, take `model_key` and `replica_id` from `ndif status --json-output`
and call the library the CLI wraps, where Ray is reachable:

```bash
docker exec -e MK="$MK" -e RID="$RID" ndif python -c "
import os
from ndif.cli.lib.restart import restart
from ndif.cli.lib.evict import evict
restart(model_key=os.environ['MK'], replica=os.environ['RID'], on_message=print)
evict(model_keys=[os.environ['MK']], replica=os.environ['RID'], on_message=print)"
```

Both print the same `✗` line and return normally when nothing matches.

## The dashboard

An optional admin web app on 8081: uptime history, deploy, evict and restart
buttons per replica, and a calendar that keeps models pinned over a time
window. It is in the compose stack, `NDIF_SERVICE="all dashboard"` in the
image, and `ndif start dashboard` from source. With
`NDIF_DASHBOARD_DEV_MODE=true`, the compose default, it has no login:
[production and auth](operate-production-and-auth.md#the-dashboard-login).

Every button is a JSON call that takes the model key verbatim:

```bash
D=http://localhost:8081
curl -s $D/api/status                                    # deployments grouped by model
curl -s -H 'content-type: application/json' $D/api/deployments/deploy \
     -d '{"checkpoint": "HuggingFaceTB/SmolLM2-135M-Instruct", "pinned": true}'
curl -s -H 'content-type: application/json' $D/api/deployments/evict \
     -d '{"model_key": "<model_key>", "replica": "<replica_id>"}'
```

`/api/deployments/restart` takes the same body as evict. With a login on, post
`{"username", "password"}` to `/api/auth/login` first and send the cookie.
**Check the body, not the status code**: a deploy that could not be placed, or
an evict that matched nothing, still answers 200, with `"status": "not_found"`
and the `✗` line inside `logs`.

Three crons run in the dashboard container. **reconcile**, every 2 minutes,
deploys what the calendar says should be loaded, always pinned, and evicts what
it pinned earlier; it does nothing when it cannot reach the controller.
**monitor**, every 10 minutes, probes `/connected` and `/status` and, every two
hours, runs a real remote trace against each HOT model, for which it needs
`NDIF_API_KEY` when auth is on. **report** writes a daily digest. Its data
directory (`NDIF_DASHBOARD_DATA_DIR`) holds the calendar, so it must be a
volume. `docs/operating/dashboard.md`.
