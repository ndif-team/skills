# Error messages, both sides

Source of record: `docs/reference/error-messages.md` (exact-text lookup); the
whole-stack entry point is `docs/operating/troubleshooting.md`.

## Rules that do most of the triage

**An HTTP status code means the request was rejected at ingress and never entered
the queue. An `ERROR` status means it got past ingress.** The user sees an
ingress rejection as `RemoteError: Failed to send request: <server sentence>`.
`RECEIVED` is the only status that travels over HTTP; everything after it arrives
on the `/subscribe` websocket or via `GET /response/{id}`.

**A traceback ending in the user's own lines means their code ran and failed; a
sentence means the server refused or failed the request.** Deserialization
failures are deliberately sentences: every frame at that point is server-side.

**Read a traceback from the bottom.** Only nnsight's frames (and those of the
actor's own `format_error` file) are stripped, so asyncio, Ray, NDIF and
torch/transformers frames still come *before* the user's lines. A traceback whose
last frame is the user's own line is a user error — their code raised there, and
the fix is in that line. A traceback ending in NDIF paths is a server fault. **The
cross-thread failure is identified by its message, not its frames**:
`ValueError: Cannot switch to a different thread` or
`ValueError: parent cannot be on a different thread` is a server fault even though
the last frames are ordinary torch/transformers lines.

Nearly everything reaches the user as `RemoteError`, which carries no structure —
the *text* is the diagnostic.

## Rejected at ingress

| Code | Server detail | Cause | Fix |
|---|---|---|---|
| 401 | `Missing or invalid API key...` | Auth is on and no `ndif-api-key` header arrived | Set the key client-side |
| 400 | `Invalid API key format: '<key>'...` | Not parseable as a UUID — usually a truncated paste | Re-copy the key |
| 403 | `Invalid API key.` | A well-formed UUID with **no row in `keys`** — including every key after Postgres lost its volume on `just down` / `just ta` | Issue or re-insert the key |
| 422 | a pydantic error list, or `Invalid JSON: …` | Client/server schema mismatch on the `data` form field | Upgrade nnsight |
| 400 | `Client nnsight version X is below the minimum supported Y.` and friends | The version gate (below) | Tell them the minimum |
| 503 | `Service temporarily unavailable: compute backend is reconnecting.` | The `ray:connected` flag is absent | Operator problem — check the ray container |
| 503 | `Auth backend unavailable.` | Postgres errored, or `NDIF_POSTGRES_URL` is set with `asyncpg` missing. Auth **fails closed** | Bring Postgres back / install the `postgres` extra |
| 503 | `Could not retrieve cluster status.` / `Could not retrieve cluster env.` | On `GET /status` or `/env`, the dispatcher's refresh from the controller failed | Retry; check the dispatcher |
| 504 | `Timed out waiting for cluster status.` / `… cluster env.` | No refresh within `NDIF_STATUS_TIMEOUT_S` / `NDIF_ENV_TIMEOUT_S` — the dispatcher is busy, wedged or gone | See the dead-dispatcher section of the skill |
| 500 | `Internal server error.` | Anything unhandled in ingress | The traceback is only in the API log, `api unhandled error` under `ndif.api` |

**401 or 403 proves auth is configured.** With `NDIF_POSTGRES_URL` unset,
`verify_api_key` returns immediately and no key check happens. `/whoami` answers a
malformed key with 400 and an unknown one with `{"email": null, "tags": []}`.

The version gate (only when `NDIF_MIN_NNSIGHT_VERSION` / `NDIF_MIN_PYTHON_VERSION`
is set) is always a 400: `Client nnsight version was not provided.` (an old client,
or an nnsight run from a source checkout, which sends an empty version),
`Malformed nnsight version '<v>'.`, or below the minimum. The python check
compares major.minor only.

## Sentences, not tracebacks

| Message begins | Class | What to tell the user |
|---|---|---|
| `Your request payload could not be read (` | `PayloadError` | Re-send it once. If it repeats: a `ModuleNotFoundError` in the detail is a package the server lacks; an `AttributeError`/`UnpicklingError` on a class is an nnsight version mismatch. Compare `ndif env` with `ndif env --local` |
| `The model architecture on this server doesn't match the one your code was traced against` | `ArchitectureMismatchError` | Run `ndif.compare()` and align whatever it flags — almost always `transformers`. The named path is the *first* divergence, not the layer they reached for |

Neither is fatal to the replica.

**Every remote `.save()` failing with `AttributeError: 'Tensor' object has no attribute 'save'`** is the server's nnsight
lacking its compiled `nnsight._c` extension — the published wheels ship it; an
sdist install with no compiler, or an unbuilt checkout bind-mounted into the
container, does not.

## ERROR responses after ingress

| Message | Means |
|---|---|
| A traceback through the user's own lines | User code raised. Not fatal (except the two CUDA errors below). Retry is pointless until the code changes |
| `OutOfOrderError: '<location>' was requested but the model already ran past it` | User: modules accessed out of forward-pass order |
| `ValueError: A barrier was never reached by every block it waits for…` | User: a `tracer.barrier(n)` whose `n` exceeds the blocks that call it |
| `Your job exceeded the execution timeout of Ns.` | **There is no default** — only emitted when a timeout is configured |
| `Request cancelled by operator.` | `ndif kill`. Queued: removed. Executing: its dispatch cancelled while the actor runs the block to completion. A **blocking** client gets this as its one `ERROR`, with an **empty description** line; the log shows one `ndif.request` ERROR and a `ndif.queue.replica` WARNING `request errored: cancelled mid-dispatch`. Restart the replica only while the block still runs (`operate` skill, "Killing a request") |
| `Replica was evicted while processing your request.` | The worker task was cancelled mid-dispatch. After a kill, this is what a **non-blocking** client raises; its stored response shows it until the actor's own final response overwrites it (often `COMPLETED`). It is not a second status for a blocking client. An ordinary eviction re-queues instead |
| `ValueError: Cannot switch to a different thread` or `ValueError: parent cannot be on a different thread` — the last frames are ordinary torch/transformers lines | **Server**: new work was dispatched to a replica whose killed block is still running. Restart the replica, then retry; once the killed block has finished, no restart is needed |
| `Error starting model.` / `Error provisioning model.` | The controller could not place or ready a replica **and did not say why**; the traceback is in the API log under `ndif.queue.processor`. Every queued request for the model gets it at once. Right after `ray` is recreated it is instead the API's stale compute connection (`ConnectionError` in the log), and the next request may 503 |
| `Could not deploy this model. <reason>` | The controller refused and explained: a mistyped `repo_id`, a gated repo, or `CANT_ACCOMMODATE`. "Try again later" is wrong advice for all three |
| `Error submitting request to model deployment.` | The Ray call to the actor raised something that is not an eviction. If the logged class is `ActorUnavailableError`, the actor is restarting after a fatal CUDA error and the worker waits for it — retry |
| `Critical server error occurred.` | A Ray connection error purged every Processor; everything queued at that moment was errored |
| `Your job was cancelled or preempted by the server.` | **No current path sends it.** The actor's only cancel caller is the HOT→WARM demotion, which passes the preempt reason and re-queues instead |

## No error text at all

| What happens | Meaning |
|---|---|
| The status shows `QUEUED` a second time | An eviction — `CachedActorError` (demoted to WARM before or during the run), `ActorDiedError`, or an actor-lookup `ValueError`. The request goes back to the **front** of its queue and re-runs from the start. A demotion mid-run is a preemption: it logs `model execution preempted; requeued` and records `execution_time` with status tag `preempted` |
| Sits on `RECEIVED`, `QUEUED`, `PROVISIONING` or `DEPLOYING` | The stuck-request procedure in [reading-logs.md](reading-logs.md) |
| `COMPLETED` with no saves, no error | Early stop (`tracer.stop()`) is normal; otherwise a client-side `.save()` question |
| A warning that a `tracer.iter` loop outran the model | Not an error: the iterations reached are kept |

**An eviction mid-flight loses the work done so far**, and a model flapping
HOT↔WARM makes requests loop without finishing. And **you cannot catch an actor's
exception by type across the Ray boundary**: it arrives wrapped in a
`RayTaskError`, and over Ray Client a bare `isinstance` matches nothing — read
`.cause`.

## Server exceptions, and whether they are fatal

**A user-caused exception is never fatal to a replica** — the actor catches it,
formats it, answers `ERROR`, keeps its weights and serves the next request.

| Exception | User's view | Whose fault |
|---|---|---|
| `RunnerError` | `ERROR` with the block's own traceback | **user** — always user code; the untrusted path's carrier |
| `CachedActorError` | **nothing** — re-queued at the front | server: demoted to WARM |
| `ActorDiedError`, actor-lookup `ValueError` | nothing — re-queued | server |
| `ActorUnavailableError` | `Error submitting request to model deployment.` | server: actor restarting after a fatal CUDA error |
| `asyncio.CancelledError` | `Replica was evicted while processing your request.` | operator/server: `ndif kill` or a purge |
| `ConnectionError: connection closed before message complete` / `runner process exited before its socket was ready` / `timed out waiting for the runner socket` | `ERROR` with a host-side traceback | server — the runner died (expected during a timeout), crashed at import, or took >30 s to bind |
| `torch.cuda.OutOfMemoryError` **at run** | a CUDA OOM traceback | user (or too little padding) |
| `torch.cuda.OutOfMemoryError` **at load** | `Error starting model.` | server — the estimate was wrong, or the GPU was not actually free; the controller logs `failed during create` |
| `RuntimeError` from `verify_device_placement` | `Error starting model.` | server — a weight landed on `meta`, `cpu`, or an unassigned GPU |
| `Cannot copy out of meta tensor` at result upload | `ERROR` | user on the untrusted path: they saved something derived from a weight, which is a meta tensor in the runner |
| `NDIFConnectivityError` | CLI/dashboard only: `Cannot connect to Ray at <url>` | server — Ray is down, the address is not listening, or the `Controller` actor is not resolvable |
| `No response from the dispatcher for <event>. Is the API running?` | CLI only | transient for ~5 s after an API restart; persistent means the API is down |

### The two fatal CUDA errors

<!-- test: skip -->
```python
_UNRECOVERABLE_CUDA_ERRORS = (
    "device-side assert triggered",
    "an illegal memory access was encountered",
)
```

A common innocent trigger for the first: a prompt longer than the model's
context dies as `device-side assert triggered` inside `masking_utils`, not as an
index error naming the limit (gpt2: 1024 positions).

These poison the process's CUDA context permanently, so the actor restarts
itself and Ray brings it back with a reload, costing every other user of that
model a load cycle. The triggering request is still a user error and still gets
its traceback. No operator action is needed: `ndif status` keeps showing the
replica HOT through the restart, and the next request works.

## After `COMPLETED`: download failures

`data` is either the result itself (a blocking result whose **compressed blob** is
at or under the inline cap, 8 MiB on stock redis:7 — a 7.5 MiB blob goes inline,
a 9 MiB one by URL) or a presigned URL the client fetches — for anything over the
cap and for **every non-blocking job**.

| Text the user sees | Whose fault | Fix |
|---|---|---|
| `httpx.ConnectError: [Errno -2] Name or service not known` (or a hang) | Operator | `NDIF_OBJECT_STORE_PUBLIC_URL` must be the address *the client* resolves; empty, it falls back to `NDIF_OBJECT_STORE_URL` |
| `403 SignatureDoesNotMatch` | Operator | Signed with different credentials than the store accepts, or a proxy rewrote the Host header |
| A `404` with body `NoSuchBucket` | Operator | A different S3 server answered at the public URL: the URL is wrong, or the client's tunnel lands on another store |
| `403 Request has expired` | User | Presigned URLs live one hour; a non-blocking job polled later gets a stale URL. Re-run |
| Download works, `torch.load` fails | Client | A compression mismatch |

## A websocket that closes mid-run

| Cause | What happened to the job |
|---|---|
| The API restarted | **The job is gone.** Every queued and in-flight request was a Python object in the dispatcher. The client gets no further status at all |
| An idle proxy timed out | The job is still running; the client just is not listening. Nothing replays it |
| The client was interrupted | Same — the server keeps going |

Nothing is stored for a blocking job. For a job that must survive a disconnect,
submit non-blocking: each non-`LOG` status is written to `responses/{id}.json`
and `GET /response/{id}` reads the latest. A 404 there means "nothing recorded
yet", which the client treats as still running.

## CLI output

`ndif evict` and `ndif restart` print `✗ <model_key>: nothing to evict` and
`✗ <model_key>: no HOT replicas to restart` as failures but **exit 0**. Check the
output, not the exit code, in a script.
