# The request lifecycle in detail

Source of record: `docs/concepts/request-lifecycle.md` (the keystone),
`docs/developing/queue-internals.md`, `docs/developing/model-actor.md`,
`docs/developing/sandbox-internals.md`.

## Where it can go wrong, hop by hop

| Hop | Failure | What the user sees |
|---|---|---|
| 2 — subscribe | websocket never connects | the client hangs, or errors before any status |
| 3 — ingress | Ray unreachable | `503` "compute backend is reconnecting" |
| 3 | bad or missing API key | `401` / `400` / `403` |
| 3 | client too old | `400` naming the minimum version |
| 4 — dispatch | dispatcher process dead | the request sits in Redis; no `QUEUED` ever arrives |
| 5 — provision | model never deploys | stuck on `PROVISIONING`/`DEPLOYING`, then `ERROR` |
| 6 — placement | won't fit on any node | `ERROR` mentioning `CANT_ACCOMMODATE` |
| 7 — replica | evicted mid-flight | silently re-queued at the front (no error) |
| 8 — execution | the block raises | `ERROR` with the user's own traceback |
| 8 | the block exceeds the timeout | `ERROR` "exceeded the execution timeout of Ns" |
| 8 | a module the payload references is missing | a `PayloadError` sentence at deserialize |
| 8a | the runner dies mid-run | `ERROR` carrying the runner's formatted traceback |
| 9 — upload | object store unreachable | `ERROR` after a successful run (object-store route only) |
| 10 — download | presigned URL signed for the wrong host | `COMPLETED`, then a client-side download failure |
| 10 | `NDIF_MAX_SOCKET_RESULT_BYTES` raised or set to `0` | the result exceeds Redis's pubsub output-buffer limit, the subscriber is dropped, and `COMPLETED` never arrives |

## The queue objects

One process, one asyncio loop, so a blocking call anywhere here stalls **every**
model.

- **`Dispatcher`** — `BRPOP`s the shared Redis list with a 10 s timeout, drains up
  to 31 more with non-blocking `RPOP`, and routes each request by `model_key` to a
  lazily created `Processor`. It also owns the `ray:connected` flag: deleted on
  entering `connect()`, set once Ray and the `Controller` actor answer, **with no
  expiry** — which is why a dead dispatcher leaves the API reporting healthy.
- **`Processor`** — one per model. Holds that model's in-memory `asyncio.Queue`,
  provisions replicas, and runs the autoscaling loop. `RequestQueue.put` stamps
  `enqueued_at` and orders by `(group, prepend, enqueued_at)`. On a failure before
  `READY` it calls `purge()`, which errors **every** queued request for that model,
  so several users see the same canned message at once.
- **`Replica`** — one per deployed actor. `wait()` polls `__ray_ready__` forever,
  treating a lookup `ValueError` as "not registered yet"; then a worker task pulls
  from the model's queue, publishes `DISPATCHED`, and makes
  `handle.run.remote(request)`.

Autoscaling checks the queue **head** every `NDIF_AUTOSCALING_INTERVAL_S` (5 s)
and adds a replica if it has waited past `NDIF_AUTOSCALING_WAIT_THRESHOLD_S`
(30 s), up to `NDIF_AUTOSCALING_MAX_REPLICAS` (3), then backs off
`NDIF_AUTOSCALING_BACKOFF_S` (120 s). It never scales down.

The three failure classes, distinguished **by type, not by message**:

<!-- test: skip -->
```python
EVICTED_ERRORS = (ValueError, ActorDiedError, CachedActorError)
```

On any of them the replica ends its own worker loop and re-queues the in-flight
request at the **front**. `asyncio.CancelledError` gets its own branch (it
inherits from `BaseException`, so a plain `except Exception` would miss it) and
answers the user before re-raising. Everything else is errored to the user.

Reading an actor's exception across the Ray boundary needs `.cause`: over Ray
Client the `RayTaskError` wrapper arrives plain, so a bare `isinstance` matches
nothing — silently.

## The `run()` template

`BaseModelDeployment.run` publishes `RUNNING`, snapshots GPU counters and the
linecache/`SOURCES`/`BLOCKS` globals, applies `request.env`, races `execute()` on
a worker thread against `execution_timeout` and the kill switch, emits metrics and
event logs, sends the result back, publishes exactly one `COMPLETED` or `ERROR`,
and restores state in `finally`.

The contract it upholds, which any custom actor must keep:

1. **Exactly one terminal response per request.** A request with no terminal
   response hangs the client's websocket until it times out.
2. **`COMPLETED` carries either the bytes or a presigned URL** — the route is
   decided by `max_socket_result_bytes` and whether there is a live socket.
   `prepare_result` compresses iff `request.compress` and meters the size on both
   routes, so the blob is byte-identical either way.
3. **Saved values are `nnsight.save()`-marked locals**, matched by identity in the
   block's frame. The sandbox runner reuses the same `cpu_pickle_module` so the
   blob format is identical.
4. **`execute` returns `(blob, deserialize_ms)`**; `deserialize_ms` may be `None`.
5. **Be interruptible.** `run()` calls `interrupt()` and returns *without waiting
   for the thread*, so an executor that can block outside Python must be
   unblockable or the thread leaks.
6. **Raise `CachedActorError` when WARM**, so the queue re-queues instead of
   erroring the user.
7. **Do not let request state outlive the request.**

`cancel()`'s `reason` matters: the default means "genuinely cancelled" and the
user is told. `KILL_REASON_PREEMPTED` means the request is blameless and still
runnable — it raises `CachedActorError` so the queue re-queues it *at the front*,
so mislabelling a deliberate cancellation re-runs it forever.

## Where state lives

| State | Lives in | Survives a restart? |
|---|---|---|
| Queued requests | Redis list (`NDIF_QUEUE_KEY`) | Yes — until popped |
| In-flight requests | dispatcher process memory | **No** |
| Status / env caches | Redis, TTL'd | Rebuilt on demand |
| Live status updates | Redis pub/sub → websocket | Not stored at all |
| Non-blocking responses | object store, latest only | Yes |
| Result blobs | object store, `{request_id}.pt` | Yes — nothing deletes them |
| Deployment state | controller actor memory + Ray | Rebuilt from the cluster |
| Model weights (WARM) | node CPU RAM | No |
| Dashboard schedule/logs | `dashboard_data` volume | Yes |

Redis keys are **unprefixed** — `queue`, `status`, `env`, `ray:connected` — so two
NDIF deployments sharing one Redis instance steal each other's requests. Give
each its own Redis, or its own logical database.

## Model actors, concretely

A deployment is a set of detached Ray actors, one per replica, created as:

<!-- test: skip -->
```python
actor_class.options(
    name=self.name,                        # "{replica_id}:ModelActor:{model_key}"
    resources={f"node:{node_name}": 0.01}, # pin to the chosen node
    namespace="NDIF",
    lifetime="detached",                   # survives the controller
    runtime_env={"env_vars": env_vars},    # provider config + CUDA flags
).remote(**deployment_args.model_dump())
```

Note `resources={f"node:{node_name}": 0.01}` and **no `num_gpus`**: placement is
by node, and GPU targeting happens inside the actor via accelerate's `max_memory`
plus a per-process allocator cap.

The `runtime_env` is how config crosses the one boundary it crosses by itself:
the controller exports its own Redis, object-store, Loki and Influx settings into
every actor, plus `NDIF_SERVICE=model` so the actor's telemetry attributes to the
model rather than to the controller. **You configure model actors by configuring
the `ray` service.**

Actors are declared `max_restarts=-1`, which is what makes
`BaseModelDeployment.restart()` — used after a CUDA-context-corrupting failure —
work at all.

## Known TODOs in the controller

Two are marked in the source and are worth knowing before you debug around them:

- **Detached model actors from a previous controller are not re-adopted.** A
  rebuilt cluster starts empty and can re-place onto GPUs those orphans still
  hold, over-committing memory.
- **`check_nodes` reconciles *nodes* against reality but never *actors*.**
