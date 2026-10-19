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

Run these anywhere that can reach `NDIF_RAY_ADDRESS`: under compose,
`docker compose -f docker/docker-compose.yml exec ray ndif ...`.

## The model key is the identity

A model key is `"<nnsight wrapper class path>:<JSON>"`, the JSON carrying the
canonical repo id, the revision, the pipeline task and, when set,
`trust_remote_code`:

```text
nnsight.modeling.transformers.TransformersModel:{"repo_id": "openai-community/gpt2", "revision": null, "task": "text-generation"}
nnsight.modeling.transformers.TransformersModel:{"repo_id": "x/custom", "revision": null, "task": "text-generation", "trust_remote_code": true}
```

Every part is identity:

- The repo id is **canonicalized through the Hub** (`gpt2` →
  `openai-community/gpt2`), so deploying needs Hub access, plus `HF_TOKEN` for a
  gated repo.
- The **revision**, the **task** and **`trust_remote_code`** change the key: two
  revisions, two tasks, or the same checkpoint with and without its own code are
  independent deployments. Left unset, the CLI infers the task exactly as an
  nnsight client with no `task=` does, so the two agree. A model deployed with a
  non-default `--task` or with `--trust-remote-code` needs the same flags on
  every later `scale`, `evict` and `restart`.
- The client sends the key it computed and the server never guesses. "Model not
  deployed" is almost always a key mismatch — compare
  `ndif status --json-output` against what nnsight computed locally.

## The commands you will actually use

```bash
ndif status                      # HOT/WARM/COLD by level, plus cluster GPU totals
ndif status --json-output        # the raw controller payload — model keys live here
ndif deploy openai-community/gpt2 --pinned
ndif deploy meta-llama/Llama-3.1-8B --dtype nf4 --gpus 1
ndif deploy -f models.yaml --sync   # the only reconciling form
ndif scale meta-llama/Llama-3.1-8B -n 2
ndif evict openai-community/gpt2               # every HOT and WARM replica
ndif evict openai-community/gpt2 --replica a1b2c
ndif evict --all                               # every HOT deployment, pinned included
ndif restart openai-community/gpt2 --replica a1b2c
ndif export --stdout > models.yaml   # snapshot the current HOT set
ndif queue                       # per-model processors, depth, in-flight requests
ndif kill <request_id>           # see "Killing a request" below
ndif env                         # the cluster's python + packages (vs `--local`)
```

`ndif queue` lists a model once a request has created its processor, and the
processor stays listed when idle; its `Replicas: N` can lag an eviction until
the next dispatch. `ndif status` is the readiness surface. A deploy prints one
line per model, `✓ <model_key> [b7f60]: ready`. Every flag:
[references/cli-commands.md](references/cli-commands.md), `docs/operating/cli.md`.

**`deploy` vs `scale`.** Both are additive; `deploy` takes unspecified settings
from the controller's defaults, `scale` copies a live replica — so `scale` a
tensor-parallel or non-default-dtype model, or `deploy` adds a replica that
answers differently under the same key.

**Check output, not exit codes.** `✗ <model_key>: nothing to evict` and
`✗ <model_key>: no HOT replicas to restart` exit 0.

## Getting models deployed

| Surface | `pinned` | `trust_remote_code` |
|---|---|---|
| `ndif deploy <checkpoint>` | `False`; `--pinned` sets it | `False`; `--trust-remote-code` puts it in the minted key |
| `ndif deploy -f models.yaml` (`--sync`) | per entry | per entry, default `False`; goes into the key |
| `NDIF_DEPLOYMENTS` at controller boot | **`True`**, hard-coded | whatever each key says (entries are raw keys) |
| Dashboard Deploy / Add Replica | per request | per request, default `False` — a checkbox in the Deploy modal |
| Dashboard schedule reconcile cron | **`True`**, hard-coded | per event, default `False`; written into the event's key |
| The queue's autoscaler | `False` | whatever the key says |
| Implicit, on the first request for an undeployed model | `False` | whatever the key says; **gated by the API key's tag** |

`trust_remote_code` is a field of the model key, not of the deployment: a
client's own `trust_remote_code=True` mints it too, keys that differ in it are
two deployments, and the implicit path deploys a key carrying it only when a
queued request's API key has the `trust_remote_code` or `trusted` tag
(`docs/concepts/auth-and-limits.md`). A running model serves every caller.

`NDIF_DEPLOYMENTS` is a pipe-separated list of **model keys**, used verbatim.
Each entry must be the exact current key, task included: a bare repo id fails to
evaluate, and a key without the task deploys a model no client's key will ever
match. Copy them from
`ndif status --json-output | jq -r '.deployments[] | select(.model_key) | .model_key'`
(on the host — `jq` is not in the image).

### models.yaml

```yaml
models:
  - openai-community/gpt2
  - checkpoint: meta-llama/Llama-3.1-8B
    task: text-generation    # part of the key; inferred when unset
    pinned: true
    replicas: 2
    dtype: bfloat16
    gpus: 4
    size_bytes: 6425499648
    execution_timeout_seconds: 3600
    # actor_class: omit for the cluster default; never the base ModelActor
    # on a cluster that takes untrusted traffic
```

Other fields: `revision`, `trust_remote_code` (default `false`; part of the
key), `padding_factor`, `padding_bias`, `max_tp`, `envoy_class`, `model_key`;
unknown keys are silently dropped. `--sync` evicts
every HOT key not in the file, trims surplus replicas, and deploys the shortfall.

**`ndif export` is not a full snapshot.** It always writes `dtype` and
`actor_class` but other fields only where non-default, drops `padding_factor`,
`padding_bias`, `size_bytes` and `max_tp` (never stored on the deployment, so a
restore is re-sized from cluster defaults), and merges replicas of one key that
differ — one pinned replica makes the whole entry `pinned: true`. Save
`ndif status --json-output` first.

## Sizing in one paragraph

**NDIF's GPU accounting is a ledger, not a measurement.** The controller reserves
`ceil(base + base × padding_factor + padding_bias)` per replica (defaults 0.15,
500 MiB) from a pre-load estimate and never reads real free memory: a deploy is
refused when the ledger says no and OOMs when the card disagrees. The padding is
the whole budget for activations, KV cache and the CUDA context, enforced per card
on the replica's share, `ceil(padded / gpus)`. Overrides: `size_bytes`,
`padding_factor`, `padding_bias`, `gpus`, `max_tp`, `dtype`. Everything else —
[references/sizing-and-placement.md](references/sizing-and-placement.md),
`docs/runbooks/model-oom-on-deploy.md`, `docs/concepts/deployments-and-eviction.md`.

## dtype, quantization, tensor parallelism

**Quantization is a dtype name** (`--dtype nf4`, `int4`, `fp4`, `int8`, `fp8`,
...); client code is unaffected and cannot ask for it — it is not in the model
key. The **size estimate runs low by more than padding covers** (Llama-3.2-1B
`nf4`: 0.62 GB estimated, 1.07 GB allocated), so give it a measured `size_bytes`.

**Tensor parallelism is off unless `NDIF_TP_MODEL_ACTOR_CLASS` is set** on the
`ray` service; unset, `max_tp` is inert and `--gpus N` spreads whole layers over
N cards with accelerate. Use `ndif.services.ray.tp.model.SandboxedTPModelActor`
on a cluster that takes untrusted traffic — a TP placement replaces the actor
class, and the plain `TPModelActor` runs untrusted code in-process. It needs
transformers >= 5.15, and **a TP replica can never be cached** (HOT→WARM is
impossible, so the controller evicts it outright). Pin one you want to keep.

**PEFT adapters are per request** (`TransformersModel(repo, peft="<hub id>")`):
one deployment serves every LoRA over its base; a local adapter directory is
invisible to the server.

## Levels, pinning, and eviction

**HOT** = on GPU, serving. **WARM** = in host RAM, actor alive, raising
`CachedActorError` if run. **COLD** = only in the node's HF cache. Only HOT
replicas reach the queue, so a WARM-only model looks undeployed.

The controller evicts on its own only what is **not pinned** and older than
`NDIF_MINIMUM_DEPLOYMENT_TIME_SECONDS` (3600; waived when the incoming model is
pinned). `ndif evict`, `--all` included, removes pinned replicas too. The age rule
makes a deploy on a full cluster come back `CANT_ACCOMMODATE` because the occupant
landed 40 minutes ago — deploy the incoming model `--pinned`, evict the occupant,
or lower the variable.

**Pinning a model already deployed.** `pinned` is set only when a replica is
created, and deploy is additive: `ndif deploy <ckpt> --pinned` adds a second,
pinned replica beside the old one. To pin in place, deploy `--pinned`, then
`ndif evict <ckpt> --replica <old id>`. `ndif scale` will not do it — it
re-promotes an existing WARM replica under its old id and settings, and
`pinned` is not copied (it defaults to false on that path).

**Changing a controller variable** (that one, the padding defaults, the actor
class) means editing the `ray` service's `environment:` and `just up ray` — not
`just restart ray` (old environment), not `just ta ray` (whole stack down). It is
an outage, and the first request afterwards errors `Error starting model.`; the
sizing reference has the rest.

Autoscaling adds up to `NDIF_AUTOSCALING_MAX_REPLICAS` (3) replicas per model
when the *head of the queue* has waited past `NDIF_AUTOSCALING_WAIT_THRESHOLD_S`
(30 s), and **never scales back down**.

## Killing a request

`ndif kill <id>` goes to the dispatcher over a Redis stream (5 s timeout, no
HTTP). What it does depends on where the request is:

- **Queued** — removed; the client gets `ERROR` "Request cancelled by operator."
- **Executing** — only the **dispatch** is cancelled. A blocking client gets one
  `ERROR`, "Request cancelled by operator." with an empty description line; the
  log shows one `ndif.request` ERROR plus a `ndif.queue.replica` WARNING
  `request errored: cancelled mid-dispatch`. A non-blocking client raises
  `Replica was evicted while processing your request.`, and its stored response
  shows that until the actor's own final response overwrites it (often
  `COMPLETED`). The actor is not interrupted: it runs the block to completion,
  holding its GPU memory.

**Restart the replica only while the killed block is still running**:
`ndif restart <checkpoint> --replica <id>` (the id is on the `⚙` line of
`ndif queue`). Work dispatched to it meanwhile fails with
`ValueError: Cannot switch to a different thread` or
`ValueError: parent cannot be on a different thread`. Once the block finishes
on its own — Loki `{service="model"} |= "<request id>"` shows
`Your job has been completed.` — the next request is clean with no restart.
Requests sent to a replica during its restart are re-queued or errored.
`No response from the dispatcher` means the API is down — or restarted under
five seconds ago.

**By exact key.** `evict`, `restart` and `scale` resolve a checkpoint to the key
their flags mint (`--task`, `--trust-remote-code`). For any other key, pass the `model_key` from
`ndif status --json-output` to `ndif.cli.lib.restart.restart` /
`ndif.cli.lib.evict.evict`, or POST the dashboard's `/api/deployments/{evict,
restart}` (`{model_key, replica}`) or `/api/deployments/deploy`
(`{checkpoint, model_key?, trust_remote_code}`) — recipes in
[references/cli-commands.md](references/cli-commands.md). An unknown replica
still returns HTTP 200, with the `✗` line inside the response's `logs[]`: check
the body, not the status code.

## The dashboard

An optional admin web app on 8081: uptime/latency history, per-replica
deploy/evict/restart buttons, and a calendar that keeps a set of models pinned
over a time window. Every action goes through the same `cli/lib` functions the
CLI uses.

- **A dashboard deploy can mint a `trust_remote_code` key** (a checkbox in the
  Deploy and Schedule modals, default off; the dashboard does not check who
  asks), and **dev mode is no auth** — `NDIF_DASHBOARD_DEV_MODE=true`, the
  compose default, exposes deploy and evict to anyone who can reach 8081.
- Crons in the container: **monitor** every 10 min (a real remote trace per HOT
  model every 2 h, needing `NDIF_API_KEY`), **reconcile** every 2 min (does
  nothing if the controller is unreachable), **report** daily. The data dir must
  be a volume, or a rebuild discards every schedule event.

Detail: `docs/operating/dashboard.md`.

## Observability at a glance

Loki (:3100) holds every log record as JSON; InfluxDB (:8086) the request
metrics (`execution_time`, `status_time`, `gpu_mem`, ...); Prometheus (:9090)
Ray's own metrics; Grafana (:3000) queries them all. Both telemetry providers are
**fail-open** and off until given `NDIF_LOKI_URL` / `NDIF_INFLUX_URL`. **Model
actor logs are `service="model"`, not `service="ray"`** — filter on `logger` when
in doubt. `docs/operating/observability.md`; LogQL in the `troubleshoot` skill.

## Adding a GPU node

Same image, `NDIF_RAY_HEAD_ADDRESS=<head>:6385` (or `ndif start
--ray-head-address`), with 6385, 8076 and 52366 open to the head. It shows in
`ndif status` within 30 s; `Nodes` up but not `Total GPUs` means it reported no
GPU. `docs/runbooks/add-a-gpu-node.md`.

## Before anyone else can reach it

Auth (`NDIF_POSTGRES_URL`) decides which tags a request carries — with it unset,
every request that does not say otherwise is `trusted`; **the actor class
decides whether that changes anything.** An untrusted request is isolated in a
runner process only on `SandboxModelActor` — set by `NDIF_MODEL_IMPORT_PATH`
(legacy fallback `NDIF_DEFAULT_MODEL_ACTOR_CLASS`), plus `SandboxedTPModelActor`
if TP is on. Only the compose file sets it; the published image and a source
install run untrusted code in-process even with auth on. Once it is set, untagged
keys lose weight reads, weight edits, gradients and training, and cannot be the
first to deploy a model whose key carries `trust_remote_code`. Grant the `trusted`
tag only to keys you would hand your own shell to; `trust_remote_code` is the
narrower grant — deploy such a model, nothing more.

Keys live in Postgres's anonymous volume: they survive a container recreate and
are **lost on `just down` / `just ta`** (existing keys then 403 until
re-inserted). Environment edits take `just up <svc>`, never `just restart`. NDIF
provides **no TLS, no rate limiting, no multi-tenancy, no secret management**;
never expose 8001 directly. The untagged-key limits, how to verify which path a
request took, and the pre-flight list:
[references/production-and-auth.md](references/production-and-auth.md),
`docs/runbooks/enable-auth.md`, `docs/operating/production.md`.

## Related skills

- `selfhost` — getting a server up in the first place.
- `troubleshoot` — when a deploy fails or a request hangs.
- `develop` — changing placement, the queue, or the actor.
- nnsight plugin `remote` — the client side of what these deployments serve.
