# The request lifecycle in detail

Source of record: `docs/concepts/request-lifecycle.md` (the keystone),
`docs/developing/queue-internals.md`, `docs/developing/model-actor.md`,
`docs/developing/controller-internals.md`, `docs/developing/sandbox-internals.md`.

## Where it can go wrong, hop by hop

| Hop | Failure | What the user sees |
|---|---|---|
| 2 — subscribe | websocket never connects | the client hangs, or errors before any status |
| 3 — ingress | Ray unreachable | `503` "compute backend is reconnecting" |
| 3 | bad or missing API key | `401` / `400` / `403` |
| 3 | client too old | `400` naming the minimum version |
| 4 — dispatch | dispatcher process dead | the request sits in Redis; no `QUEUED` ever arrives |
| 5 — provision | model never deploys | stuck on `PROVISIONING`/`DEPLOYING`, then `ERROR` |
| 6 — placement | won't fit on any node | `ERROR` "Could not deploy this model." with a node's stated reason, `CANT_ACCOMMODATE` when none gave one |
| 7 — replica | evicted mid-flight | silently re-queued at the front (no error) |
| 8 — execution | the block raises | `ERROR` with a traceback ending at the user's line. Only nnsight's frames are stripped, so asyncio/Ray/NDIF wrapper and torch/transformers frames precede it |
| 8 | the block exceeds the timeout | `ERROR` "exceeded the execution timeout of Ns" |
| 8 | a module the payload references is missing | `ModuleNotFoundError` at deserialize |
| 8a | the runner dies mid-run | `ERROR` carrying the runner's formatted traceback |
| 9 — upload | object store unreachable | `ERROR` after a successful run (object-store route only) |
| 10 — download | presigned URL signed for the wrong host | `COMPLETED`, then a client-side download failure |
| 10 | near-cap inline results published to a subscriber that is not draining | about three reach Redis's hard pubsub limit, the subscriber is dropped, and `COMPLETED` never arrives |

On a cold start the processor is already provisioning when it replies, so the
first status is `PROVISIONING`, not `QUEUED`.

## The queue objects

One process, one asyncio loop, so a blocking call anywhere here stalls **every**
model.

- **`Dispatcher`** — `BRPOP`s the shared Redis list with a timeout, drains more
  with non-blocking `RPOP`, and routes by `model_key` to a lazily created
  `Processor`. It owns the `ray:connected` flag, set **with no expiry** — so a
  dead dispatcher leaves the API reporting healthy.
- **`Processor`** — one per model: its in-memory queue, replica provisioning,
  autoscaling. `RequestQueue.put` orders by `(group, prepend, enqueued_at)`. A
  failure before `READY` calls `purge()`, which errors **every** queued request
  for that model with the same message.
- **`Replica`** — one per deployed actor. `wait()` polls `__ray_ready__` every
  second with **no timeout**, treating `ValueError` (not created yet) and
  `ActorUnavailableError` (restarting) as "not yet"; then a worker pulls from the
  queue, publishes `DISPATCHED`, and calls `handle.run.remote(request)`.

Autoscaling checks the queue **head** every `NDIF_AUTOSCALING_INTERVAL_S` and adds
a replica once it has waited past `NDIF_AUTOSCALING_WAIT_THRESHOLD_S`, up to
`NDIF_AUTOSCALING_MAX_REPLICAS`, then backs off. It never scales down.

`Replica.dispatch` splits failures **by type, not by message**:

| Caught | Result |
|---|---|
| `asyncio.CancelledError` | error the user, then re-raise (it is a `BaseException`, so a plain `except Exception` would leave them on `DISPATCHED`) |
| `ValueError` / `ActorDiedError` / `CachedActorError` | drop the replica, re-queue the request at the **front** |
| `ActorUnavailableError` | error that request, then `wait()` until the same replica serves again — nothing re-queued, replica kept |
| anything else | error the user, keep serving |

Over Ray Client the `RayTaskError` wrapper arrives plain, so the check reads
`.cause`; a bare `isinstance` would match nothing and turn every HOT→WARM
demotion into a user-facing error.

## The `run()` contract

`BaseModelDeployment.run` publishes `RUNNING`, applies `request.env`, races
`execute()` on a worker thread against the timeout and the kill switch, sends the
result back, publishes exactly one `COMPLETED` or `ERROR`, and restores state in
`finally`. The result route: a blocking request's blob at or under
`RedisProvider.max_publish_bytes` (`min(soft, hard // 2)` of Redis's pubsub
limits; 8 MiB on stock Redis; raise Redis's limits to raise it) rides on
`COMPLETED`, otherwise it is uploaded and presigned. The published frame is up
to ~1.5x the compressed blob. The seven obligations a custom actor keeps are in
[extending.md](extending.md#the-contract).

`cancel()`'s `reason` matters: unset means "genuinely cancelled" and the user is
told. `KILL_REASON_PREEMPTED` raises `CachedActorError` so the queue re-queues it
*at the front* — mislabel a deliberate cancellation and it re-runs forever.

## Where state lives

| State | Lives in | Survives a restart? |
|---|---|---|
| Queued requests | Redis list (`NDIF_QUEUE_KEY`) | Yes — until popped |
| Popped and in-flight requests | dispatcher process memory | **No** |
| Live status updates | Redis pub/sub → websocket | Not stored at all |
| Result blobs, non-blocking responses | object store | Yes — nothing deletes them |
| Deployment state | controller actor memory + Ray | Rebuilt from the cluster |
| Model weights (WARM) | node CPU RAM | No |

Redis keys are **unprefixed** — `queue`, `status`, `env`, `ray:connected` — so two
NDIF deployments sharing one Redis steal each other's requests.

## Model actors and the controller

A replica is a detached Ray actor named `{replica_id}:ModelActor:{model_key}` in
namespace `NDIF`, pinned with `resources={f"node:{node_name}": 0.01}` and **no
`num_gpus`**: placement is by node, and GPU targeting happens inside the actor
via accelerate's `max_memory` plus a per-process allocator cap. A tensor-parallel
placement swaps the actor class for `NDIF_TP_MODEL_ACTOR_CLASS`. The
controller exports its Redis, object-store, Loki and Influx settings into each
actor's `runtime_env`, so **you configure model actors by configuring the `ray`
service.**

The controller does not re-adopt or reconcile actors
(`docs/developing/controller-internals.md`):

- **A restarted controller starts from a blank cluster** while the detached
  actors survive, still holding GPU memory it believes is free; the next deploy
  can over-commit. Kill the orphans directly — `ndif evict` won't find them.
- **`check_nodes` checks nodes, not actors**, so a permanently dead actor's
  record and GPU reservation outlive it until `ndif evict` clears them.
- **A node that drops out of the `ALIVE` list even briefly** has every actor on it
  purged on the next pass.
