---
name: operate
description: Run models on an NDIF you host — deploy, evict, pin, scale, restart, inspect the queue, kill a request, export a models.yaml, and size a model so it actually fits. Use for "deploy a model on NDIF", "ndif deploy/evict/scale/status/queue/kill", "pin a model", "models.yaml", "NDIF_DEPLOYMENTS", "how much GPU will this model take", "padding factor", "HOT/WARM/COLD", "NDIF dashboard", "schedule a model", "add a GPU node", "turn on NDIF auth", "API keys", "NDIF in production", or NDIF Grafana/Loki/InfluxDB telemetry. Assumes a server is already up — standing one up is `selfhost`.
---

# Operating an NDIF

Two facts shape everything an operator does here.

1. **You usually do not have to deploy anything.** The queue is lazy: the first
   request for a model with no HOT replica provisions one on the spot. Explicit
   deploys exist so a model is *already warm* when the first user arrives, and so
   it can be pinned.
2. **Deploy is additive.** Every call places `--replicas` **new** replicas
   regardless of what is running, so `ndif deploy gpt2` twice gives you two. The
   only ways to shrink are `ndif evict` and `ndif deploy -f models.yaml --sync`.

Where you run these commands: anywhere that can reach `NDIF_RAY_ADDRESS`. Under
compose that is `docker compose -f docker/docker-compose.yml exec ray ndif ...`;
with the published image, `docker exec <container> ndif ...`; from source, just
your shell.

## The model key is the identity

A model key is `"<nnsight wrapper class path>:<JSON>"`, the JSON carrying the
canonical repo id and revision:

```text
nnsight.modeling.transformers.TransformersModel:{"repo_id": "openai-community/gpt2", "revision": null}
```

It is built by constructing the wrapper on the meta device and asking nnsight for
`to_model_key()`. Three consequences:

- The repo id is **canonicalized through the Hub**, so `gpt2` and
  `openai-community/gpt2` land on the same key — and deploying needs network
  access to huggingface.co, plus `HF_TOKEN` for a gated repo.
- The **revision is part of the identity**: `--revision` does not set a stored
  field, it changes the key, and two revisions are two independent deployments.
- The client sends the key it computed and the server never guesses. "Model not
  deployed" is almost always a key mismatch — compare
  `ndif status --json-output` against what nnsight computed locally.

## The commands you will actually use

```bash
ndif status                      # HOT/WARM/COLD by level, plus cluster GPU totals
ndif status --json-output        # the raw controller payload — model keys live here
ndif status --watch              # re-render every 2s
ndif deploy openai-community/gpt2 --pinned
ndif deploy meta-llama/Llama-3.1-8B --dtype nf4 --gpus 1
ndif deploy -f models.yaml --sync   # the only reconciling form
ndif scale meta-llama/Llama-3.1-8B -n 2
ndif evict openai-community/gpt2               # every HOT and WARM replica
ndif evict openai-community/gpt2 --replica a1b2c
ndif evict --all                               # every HOT deployment, pinned included
ndif restart openai-community/gpt2 --replica a1b2c
ndif export -f models.yaml       # snapshot the current HOT set
ndif queue                       # per-model processors, depth, in-flight requests — a model gets a processor on its FIRST REQUEST, so a freshly pre-deployed one is absent here; `ndif status` is the readiness surface
ndif kill <request_id>           # cancel a queued or executing request
ndif env                         # the cluster's python + packages (vs `--local`)
```

Every flag, with real output, is in [references/cli-commands.md](references/cli-commands.md)
and exhaustively in `docs/operating/cli.md`.

**`deploy` vs `scale`.** Both are additive; the difference is where the
*unspecified* settings come from. `deploy` uses the controller's defaults;
`scale` copies a live replica. That only matters for a model served differently
from how the controller would choose on its own — a tensor-parallel replica, or a
non-default dtype — where `deploy` would add a replica that answers differently
under the same model key. The queue's autoscaler provisions through `scale`. At
cold start there is nothing to copy and `scale` behaves as a plain deploy; a
durable answer belongs in `models.yaml`.

**`ndif queue` and `ndif kill` do not use HTTP.** They write to a Redis stream and
block 5 s for the dispatcher's reply, so `No response from the dispatcher` means
Redis is up and the **API** (or its dispatcher) is not.

## Getting models deployed, five ways

| Surface | `pinned` | `trusted` |
|---|---|---|
| `ndif deploy <checkpoint>` | `False`; `--pinned` sets it | `False`; `--trusted` sets it |
| `ndif deploy -f models.yaml` (`--sync`) | per-entry | per-entry, default `False` |
| `NDIF_DEPLOYMENTS` at controller boot | **`True`**, hard-coded | `False` |
| Dashboard buttons and schedule reconcile | per-request; always `True` for schedule entries | **`True`**, hard-coded (an admin action) |
| Implicit, on the first request for an undeployed model | `False` | **the requesting client's `trusted` flag** |

`NDIF_DEPLOYMENTS` is a pipe-separated list of **model keys**, not checkpoints —
entries are used verbatim, so a bare repo id there becomes a bogus key that fails
to evaluate. Get real ones from
`ndif status --json-output | jq -r '.deployments[] | select(.model_key) | .model_key'`
— `select` because COLD stubs carry no key, and on the host, because `jq` is
not in the image. `ndif deploy <repo-id>` after start is the simpler way to
pre-load one checkpoint by name.

### models.yaml

```yaml
models:
  - openai-community/gpt2
  - checkpoint: meta-llama/Llama-3.1-8B
    revision: main
    pinned: true
    replicas: 2
    trusted: false
    dtype: bfloat16
    padding_factor: 0.15
    padding_bias: 2000000000
    size_bytes: 6425499648
    gpus: 4
    max_tp: 8
    execution_timeout_seconds: 3600
    actor_class: ndif.services.ray.deployments.modeling.base.ModelActor
```

Every field the deploy path understands is passed through; any other key is
silently dropped. `--sync` evicts every HOT model key not in the file, trims
replicas above the requested count, and deploys only the shortfall. Pair it with
`ndif export` for a snapshot/restore loop — but note `padding_factor` does **not**
round-trip: it is a deploy-time sizing input, never stored on the deployment, so
a restored model falls back to `NDIF_DEFAULT_PADDING_FACTOR`. `gpus` round-trips
from 0.1.2; on 0.1.1 an export drops it, and restoring a two-card model from that
file lets the placer put it on one card that cannot hold it — check the YAML
before `--sync`.

## Sizing: how the controller decides a model fits

**NDIF's GPU accounting is a model, not a measurement.** The controller keeps a
per-GPU ledger of reserved bytes, sized from a meta-device estimate made *before*
anything loads, and never reads the card's real free memory. A deploy is refused
when the *ledger* says no; a deploy OOMs when the ledger said yes and the card
disagreed.

```text
padded = ceil(base + base * padding_factor + padding_bias)
```

`ndif status`'s "GPU Memory ... free" line is this ledger (0.1.2 labels it
"unreserved by NDIF"); COLD on 0.1.1 also lists datasets and adapters found in
the HF cache, none of them deployable.

`base` is parameters + buffers at the target dtype. Defaults: `padding_factor`
0.15, `padding_bias` 500 MiB. That padding is the **entire** budget for
activations, KV cache, CUDA workspaces and the CUDA context — and the actor
enforces it twice, with accelerate's `max_memory` at load and a per-process
allocator cap at run time. This is why a block can die with
`CUDA out of memory ... N MiB allowed` on a nearly empty card.

Placement charges each card the replica's **share**, `ceil(size / gpus_needed)`,
not the whole card — so a model 1% over one card's capacity takes two cards at
about half each and the rest stays usable. On the default actor, `--gpus N`
loads the model across N cards with an accelerate device map; it is not tensor
parallelism and needs no `NDIF_TP_MODEL_ACTOR_CLASS`.

Worked example, a 27B bf16 model on cards with 56 GB actually free of 80 GB:
base = 27.4e9 × 2 B = 54.8 GB, padded = 54.8 × 1.15 + 0.5 = 63.5 GB. The ledger
sees 80 GB free per card and would place it on one; the card cannot hold it.
`ndif deploy google/gemma-3-27b-it --gpus 2` charges 31.8 GB to each card and
loads 26.8 + 26.4 GB. An 8B model (16.06 GB base → 18.99 GB padded) fits one.

What the ledger cannot see, in order of how often it bites: **anything NDIF did
not place** (a stray training job, an actor left behind by a killed controller),
the CUDA context (~400 MiB per process per device), controller restarts (the
ledger is in-memory and rebuilds with every GPU marked free while detached actors
still hold weights), and fragmentation. Cross-check with `nvidia-smi` whenever a
number looks impossible.

Overrides, all per-model: `size_bytes` (a measurement, replacing the estimate),
`padding_factor`, `padding_bias`, `gpus`, `max_tp`, `dtype`. These are overrides,
not requirements — supply any part and the rest is still derived. Detail and the
OOM procedure: [references/sizing-and-placement.md](references/sizing-and-placement.md),
`docs/runbooks/model-oom-on-deploy.md`, `docs/gotchas/gpu-and-memory.md`.

## dtype, quantization, tensor parallelism

**Quantization is a dtype name.** `--dtype nf4` (or `int4`, `4bit`, `fp4`,
`int8`, `8bit`, `fp8`) deploys the weights that narrow. Nothing client-side
changes — module paths and activations are identical, so a trace written against
one works against the other. Two things to know: the **size estimate runs low by
more than padding covers** (measured on Llama-3.2-1B, `nf4` estimates 0.62 GB
against 1.07 GB really allocated), so give a quantized deploy a measured
`size_bytes` or a padding factor you worked out on the hardware; and **a client
cannot ask for it** — quantization is not part of the model key, so the
deployment decides.

**Tensor parallelism is off unless `NDIF_TP_MODEL_ACTOR_CLASS` is set.** Unset is
not a fallback to the built-in actor: no degree is worked out, no GPU count is
rounded up to a shardable one, and per-model `max_tp` is inert. Set it to
`ndif.services.ray.tp.model.TPModelActor`, or to
`ndif.services.ray.tp.model.SandboxedTPModelActor` on a cluster that takes
untrusted traffic — a TP placement replaces the actor class, so the plain
`TPModelActor` runs untrusted code in-process. It needs transformers >= 5.15
(below it a tied LM head returns logits `tp_size` times too wide, with a
plausible argmax), and **a TP replica can never be cached** — its ranks' devices
are fixed at process start, so HOT→WARM is impossible and the controller evicts
it outright. Pin one you want to keep.

**PEFT adapters are per request, not per deployment.** The client instantiates
`TransformersModel(repo, peft="<adapter repo id>")` and the actor applies it
before each run, so one deployment of `gpt2` serves every LoRA over `gpt2`. Both
sides need `peft` installed; the adapter is fetched from the Hub by id, so a
local adapter directory on the client is invisible to the server.

## Levels, pinning, and eviction

**HOT** = weights on GPU, serving. **WARM** = offloaded to host RAM, GPU
released, the actor alive but raising `CachedActorError` if asked to run.
**COLD** = present only in that node's Hugging Face cache.

Only HOT replicas are returned to the queue, so a model with nothing but WARM
replicas looks undeployed and the user watches a `PROVISIONING` → `DEPLOYING`
cycle for a model that never left the node (usually satisfied by promoting the
WARM copy rather than reloading from disk).

The controller may evict a deployment only if:

- it is **not pinned** — pinning blocks *automatic* eviction only; `ndif evict`,
  including `--all`, removes pinned replicas like anything else; and
- it is older than `NDIF_MINIMUM_DEPLOYMENT_TIME_SECONDS` (3600), **unless** the
  incoming model is itself pinned, which waives the age rule.

That age rule is behind the most surprising failure in the system: on a full
cluster a deploy can come back `CANT_ACCOMMODATE` purely because the current
occupant landed 40 minutes ago, and the same command succeeds an hour later with
no configuration change. Workarounds, in order of preference: deploy the incoming
model `--pinned`, evict the occupant explicitly, or lower the variable and
restart the controller.

Autoscaling adds up to `NDIF_AUTOSCALING_MAX_REPLICAS` (3) replicas per model
when the *head of the queue* has waited past
`NDIF_AUTOSCALING_WAIT_THRESHOLD_S` (30 s) — it keys off head-of-line wait, not
depth — and **never scales back down**. Those extra replicas hold GPU memory
until something evicts them or you trim them by hand.

## The dashboard

An optional admin web app on 8081: a browsable uptime/latency history, per-replica
deploy/evict/restart buttons, and a calendar that keeps a set of models pinned
over a time window. It has no privileged channel of its own — every action goes
through the same `cli/lib` functions the CLI uses.

- **Every dashboard deploy is `trusted: True`**, hard-coded as an admin action.
  That becomes `trust_remote_code=True` at load, so deploy from the UI only what
  you would deploy with that flag from a shell.
- **Dev mode is not a login shortcut, it is no auth.** `NDIF_DASHBOARD_DEV_MODE=true`
  (the compose default) exposes deploy and evict to anyone who can reach 8081.
- Three crons run in the container (never on a laptop — `start.sh` wires cron only
  where `/etc/cron.d` is writable): **monitor** every 10 min probes `/connected`
  and `/status` and, every 2 h, runs a real remote nnsight trace against every HOT
  model; **reconcile** every 2 min enforces the schedule; **report** posts a daily
  digest. The monitor needs `NDIF_API_KEY` for its traces and skips them without one.
- A schedule event is *one model, pinned, over a `[start, end)` window*; `end:
  null` means open-ended. If the controller is unreachable a reconcile pass does
  **nothing** rather than acting blind.
- The dashboard's data dir must be a volume, or a rebuild discards every schedule
  event and all monitor history.

Detail: `docs/operating/dashboard.md`.

## Observability at a glance

| Surface | Holds | Dev URL |
|---|---|---|
| **Loki** | every `ndif` log record as a structured JSON line | http://localhost:3100 |
| **InfluxDB** | NDIF's own metrics: `request_size`, `response_size`, `execution_time`, `status_time`, `gpu_mem`, `model_load_time` | http://localhost:8086 |
| **Prometheus** | Ray's own metrics, scraped from `ray:8080` | http://localhost:9090 |
| **Grafana** | queries the other three, plus Postgres | http://localhost:3000 |

Both telemetry providers are **fail-open**: unconfigured, uninstalled, or down,
nothing errors and "no data" is the only symptom. (Before ndif 0.1.1 the
metrics provider defaulted to `localhost:8086` and a down server's
connection-refused traceback was streamed to the client as a LOG event
mid-trace; set `NDIF_INFLUX_ENABLED=false` on 0.1.0 to silence it.) The
console handler is always installed, so `just logs` never goes quiet.

An unpinned deployment's status record carries `schedule.end_time`: the end of
its minimum-deployment window (`NDIF_MINIMUM_DEPLOYMENT_TIME_SECONDS`, one
hour). It is not a teardown time. After it the model is *evictable* when the
placer needs the space; pin it (`--pinned`, `NDIF_DEPLOYMENTS`, or a schedule
entry) to keep it regardless.

The one label that catches everyone: **model actor logs are `service="model"`,
not `service="ray"`** — the controller overrides `NDIF_SERVICE` in each actor's
runtime env. `{service="ray"}` carries only the Ray node's own output. Filter on
`logger` (`ndif.modeling`, `ndif.controller`, `ndif.queue.replica`) when in
doubt. `docs/operating/observability.md`.

## Before anyone else can reach it

Turning on auth is one variable, and it changes more than who may submit:
`NDIF_POSTGRES_URL` decides whether a request runs in a sandboxed runner process
or in-process next to the weights, and whether models load with
`trust_remote_code`. Grant the `trusted` user_tag only to keys you would hand
your own shell to.

The rest of the list — dashboard credentials, real object-store URLs and
credentials, `NDIF_MIN_NNSIGHT_VERSION`, an execution timeout, durable volumes,
which ports must never be public — is in
[references/production-and-auth.md](references/production-and-auth.md),
`docs/runbooks/enable-auth.md` and `docs/operating/production.md`.

NDIF itself provides **no TLS, no rate limiting or quotas, no multi-tenancy, and
no secret management**, and its sandbox is process separation rather than a
hardened jail. Terminate TLS yourself and never expose 8001 directly.

## References

- [references/cli-commands.md](references/cli-commands.md) — every `ndif` verb,
  its flags, and what it talks to.
- [references/sizing-and-placement.md](references/sizing-and-placement.md) — the
  estimate, placement scoring, eviction order, and right-sizing from the
  `gpu_mem` metric.
- [references/production-and-auth.md](references/production-and-auth.md) —
  enabling API-key auth end to end, the two user_tags, and the pre-flight list.

## Related skills

- `selfhost` — getting a server up in the first place.
- `troubleshoot` — when a deploy fails or a request hangs.
- `develop` — changing placement, the queue, or the actor.
- nnsight plugin `remote` — the client side of what these deployments serve.
