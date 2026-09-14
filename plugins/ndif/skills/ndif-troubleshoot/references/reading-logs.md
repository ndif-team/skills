# Reading the logs, and the two end-to-end procedures

Source of record: `docs/operating/observability.md`,
`docs/runbooks/debug-a-stuck-request.md`,
`docs/runbooks/trace-a-users-failed-job.md`.

## Where the logs actually are

| Route | Command |
|---|---|
| Compose | `just logs api` / `ray` / `dashboard`, or `docker compose -f docker/docker-compose.yml logs --tail=50 api` |
| Published image | `docker logs ndif` (one container runs everything) |
| From source | `ndif logs api -f` — tails `$NDIF_HOME/logs/<service>.log`; only **detached** services have a log file, a `--foreground` one writes to the terminal |
| Model actors | the Ray dashboard (`http://<head>:8265` → Actors), or Loki `{service="model"}` |
| The controller | **not** in `just logs ray` — it is a Ray actor, so its output goes to Ray's own session logs |

The controller one catches everyone:

```bash
docker compose -f docker/docker-compose.yml exec ray \
  bash -c "grep -hE 'Analyzing deployment|cannot be deployed|Deploying .* on ' \
           /tmp/ray/session_latest/logs/worker-*.out"
```

That is where the padded byte count for a `CANT_ACCOMMODATE` lives.

## The logger tree

Every module logs under `ndif.<component>`, and the sub-name becomes a Loki
stream label — a query dimension, not decoration.

| Logger | Process | Tells you |
|---|---|---|
| `ndif.api` | API workers | rejections (`api request rejected: 403`), unhandled 500s |
| `ndif.request` | everywhere the request travels | one record per status transition, with `stage`, `prev_stage`, `prev_stage_ms` |
| `ndif.queue.dispatcher` | API master | Ray connect/reconnect, dispatch failures |
| `ndif.queue.processor` | API master | `request enqueued` (with `queue_size`), autoscaling, provisioning errors |
| `ndif.queue.replica` | API master | `DISPATCHED`, eviction and re-queue, `request errored during execution` |
| `ndif.controller` | Ray head | node updates, placement decisions, `CANT_ACCOMMODATE`, evictions |
| `ndif.modeling` | model actor | model load, `model execution timed out` / `cancelled` / `errored` |

**Loki stream labels are deliberately few**: `service`, `environment`, `logger`,
`severity`, and `model_key`. Everything else — `request_id`, `session_id`,
`replica_id`, `email`, `api_key`, `stage`, `exec_ms`, `host`, `pid` — lives in
the JSON line and needs `| json`. They are unbounded, and label cardinality is
what kills Loki.

**There are four `service` values but only three appear in any config file.**
`api`, `ray` and `dashboard` are in the compose file; **`model` is not** — the
controller injects `NDIF_SERVICE=model` into each model actor's Ray runtime env
so a replica's logs attribute to the model rather than to the controller that
spawned it. `{service="ray"}` carries only the Ray node's own output.

```logql
{environment=~".+"} | json | request_id=`3f9c1e2b...`   # one job across every service
{service="model"} | json | model_key=`<key>`           # one model's replicas
{logger="ndif.controller"} |= "Evicting"               # eviction history
{environment=~".+", severity=~"error|warning"} | json
{service="api"} | json | status_code >= 500
```

Without Loki, the identical records go to each process's console in one-line
form with the fields appended as `key=value`, so `just logs api` and the Ray
dashboard's per-actor log view carry the same information.

Grafana's Loki datasource adds derived fields: `request_id`, `email` and `host`
in any log line render as links, `request_id` to every log for that request
across all services.

## The metrics

Six InfluxDB measurements, all tagged `service` / `environment` / `model_key` /
`api_key` / `email` plus a metric-specific dimension, with `request_id` as a
**field** rather than a tag:

| Measurement | Records | Key fields |
|---|---|---|
| `request_size` | payload size at ingress | `payload_bytes`, `ip_address`, `user_agent` |
| `response_size` | result blob staged | `response_bytes`, `compressed` |
| `execution_time` | time on the model actor, by phase | `deserialize_ms`, `exec_ms`, `upload_ms` |
| `status_time` | time in each lifecycle status | `duration_ms`, tagged by the status it *left* |
| `gpu_mem` | extra GPU a request drove past the weights | `baseline_bytes`, `peak_bytes`, `extra_bytes` |
| `model_load_time` | weights onto GPU | `duration_ms`, `num_gpus`, tagged `load_type` |

`status_time` is the one people miss: every transition emits a point for the
phase just left, so end-to-end latency is attributable with no extra
instrumentation. A big `status_time{queued}` with a successful run means the user
experienced "hung", not "failed"; a big `execution_time.deserialize_ms` means a
large serialized block; a big `upload_ms` means a large result.

Absence of metrics is **not** evidence of absence of requests — Influx is
optional and fail-open.

## Procedure: a stuck request

**Step 0 — get the id.** The client's status line prints it dimmed in brackets:
`[3f9c...] QUEUED  Added to Queue at position 2.` Without one, `ndif queue` lists
every in-flight and queued id.

**Step 1 — is the backend up?** `curl /ping`, then `/connected`. If
`ray:connected` is missing, the dispatcher is looping in `connect()` and has
already **purged every Processor** — every queued request was answered with
`ERROR`. A user still waiting at that point has a client that is not receiving.

**Step 2 — stuck at ingress?** `redis-cli llen queue` should be 0 or a small
transient number. Persistently growing means the dispatcher is not popping.
Nothing else in Redis tracks an individual request — there is no `request:<id>`
key to `GET`. Status is either published to a pub/sub channel named after the
client's `session_id` (blocking; nothing is persisted) or written to
`responses/{request_id}.json` in the object store (non-blocking; latest only).

**Step 3 — ask the dispatcher.** `ndif queue`.

- *No processor for the model* — the request never got there: still on the
  ingress list, rejected at the API, or you are inspecting a different NDIF.
- *`uninitialized` with requests queued* — provisioning failed and `purge` reset
  it. Look for `Error provisioning model` / `Error starting model` in the API log.
- *`provisioning` / `deploying` for a long time* — the controller is placing, or
  the Processor is waiting on `__ray_ready__` **with no timeout**. Go to step 4.
- *`ready`, id in `Queued:`* — behind other work; check depth and `Replicas:`, and
  look for `autoscale_trigger` / `Autoscaling ... failed to add replica`.
- *`ready`, id on an `executing` line* — note the duration, go to step 5.

**Step 4 — is there a live HOT replica?**

```bash
ndif status --json-output | jq '.deployments[] |
  select(.repo_id=="openai-community/gpt2") |
  {level: .deployment_level, state: .application_state, replica: .replica_id, pinned}'
```

| level | state | Meaning |
|---|---|---|
| HOT | RUNNING | Healthy — the request is stuck *inside* the actor |
| HOT | DEPLOYING | Weights still loading; big models legitimately take minutes |
| HOT | UNHEALTHY | The actor died; Ray restarts it. Repeatedly ⇒ a load-time OOM |
| WARM | — | The weights are on CPU. `run()` raises `CachedActorError`, the queue treats it as an eviction and re-queues. A model flapping HOT↔WARM makes requests loop without ever finishing |
| absent | — | Only HOT replicas are returned to the queue, so a WARM-only model looks undeployed |

Deployments are plain detached Ray actors named
`{replica_id}:ModelActor:{model_key}` in the `NDIF` namespace — there is no Ray
Serve here and no `serve status` to run. Find them in the Ray dashboard's Actors
view, which also gives you the node, the pid, and the logs.

**Step 5 — where is it executing?** Depends on `request.trusted`. Trusted: the
block runs in-process in the model actor, one thread, no socket, and a hang is
user code or the forward pass itself. Untrusted: a fresh runner subprocess runs
the block and the actor drives the forward pass over a Unix socket, so a hang can
be on either side. With auth off every request is trusted by default, so on a
default local stack you are always in the in-process case.

Two signals: user `print()` output arrives as `LOG` responses, so a console still
printing means the block is running, not blocked; and `ndif.modeling` shows
whether `run` ever got past `Your job has started running.`

**Step 6 — the timeouts.**

| Boundary | Limit | Default |
|---|---|---|
| Ingress HTTP POST | `NDIF_API_TIMEOUT` (gunicorn worker) | 120 s |
| Ingress queue wait | none | — |
| Per-model queue wait | none | — (autoscaling is the only relief) |
| Waiting for a replica to be ready | none — `Replica.wait` polls forever | — |
| **Execution** | `NDIF_DEFAULT_EXECUTION_TIMEOUT_SECONDS`, or a per-deployment `execution_timeout_seconds` | **unset — no cap at all** |
| `ndif queue` / `ndif kill` round trip | event round-trip | 5 s |

The execution timeout is enforced by injecting an exception into the execution
thread, which CPython delivers only at a bytecode boundary — it **cannot
interrupt a native call already in flight**, so a single enormous CUDA kernel
runs to completion first. On the sandboxed path the interrupt also stops the
runner process, which is a real stop.

**Step 7 — unstick it.** `ndif kill <id>`, or
`ndif restart <checkpoint> --replica <id>` (the in-flight request is re-queued,
not dropped).

## Procedure: reconstructing a failed job

One asymmetry governs what evidence still exists:

| | Blocking (default) | Non-blocking |
|---|---|---|
| How statuses reach the client | Redis pub/sub → the `/subscribe` websocket | written to `responses/{id}.json` in the object store |
| What survives | **nothing** — pub/sub is fire-and-forget | the **latest** response only |
| The user's `print()` | `LOG` responses, streamed live | dropped |

So for a blocking job — what almost everyone runs — the only durable record is
the logs and metrics. Get the user's console output if you can.

**Find the id.** Best case they have it. Otherwise search by identity and time:

```logql
{environment=~".+"} | json | email=`researcher@example.edu` | stage=`error`
{environment=~".+", model_key=~".*Llama-3.1-8B.*"} | json | stage=`error`
```

(With auth off there is no `email` — per-user attribution is a reason to turn
auth on.)

**Read the lifecycle timeline.** `ndif.request` logs exactly one record per
transition:

```logql
{environment=~".+", logger="ndif.request"} | json | request_id=`3f9c1e2b...`
```

```text
stage=received                        prev_stage=null
stage=queued      prev_stage=received     prev_stage_ms=4.1
stage=dispatched  prev_stage=queued       prev_stage_ms=812.6
stage=running     prev_stage=dispatched   prev_stage_ms=6.2
stage=error       prev_stage=running      prev_stage_ms=48213.9
```

The **last stage reached** localizes the failure:

| Last stage | Owner |
|---|---|
| nothing at all | rejected at ingress before a request object existed — look for `api request rejected` under `ndif.api` |
| `received` | never enqueued — the API errored between accepting and `lpush` |
| `queued` / `provisioning` / `deploying` | provisioning failed; the user got a canned "Error starting/provisioning model" |
| `dispatched` | the handoff to the actor failed, or the replica was evicted mid-flight |
| `running` | execution failed — the interesting case |

**Decide whose fault it is.** *A traceback means their code; one of a handful of
fixed English sentences means ours.* That holds by construction: the runner (or
the actor) formats the traceback where it is live — tracebacks do not survive
cloudpickle — and ships the text, after stripping nnsight plumbing and the
actor's own frames, so the user reads a traceback of *their own* source lines,
produced on the server.

**Where the result would have gone.** Two objects in the same bucket:
`{request_id}.pt` written **only on success**, and `responses/{request_id}.json`
written for every non-LOG status of a non-blocking job. So a failed job has no
`.pt`, but a failed *non-blocking* job still has its last response:

```bash
curl -s localhost:8001/response/3f9c1e2b... | jq .
```

For a blocking job that is always 404 — nothing was ever written.
