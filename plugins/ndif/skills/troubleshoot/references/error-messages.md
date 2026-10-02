# Error messages, both sides

Source of record: `docs/errors/client-side-failures.md`,
`docs/errors/server-exceptions.md`, `docs/errors/index.md`.

## Two rules that do most of the triage

**An HTTP status code means the request was rejected at ingress and never entered
the queue. An `ERROR` status means it got past ingress.** `RECEIVED` is the only
status that travels over HTTP; everything after it arrives on the `/subscribe`
websocket or via `GET /response/{id}`.

**A traceback means their code ran and failed; a sentence means the server
refused the request before their code existed.** Deserialization failures are
deliberately not tracebacks: every frame at that point is server-side, so a
traceback would leak module layout while telling the caller nothing.

Nearly everything reaches the user as `RemoteError`, which carries no structure —
the *text* is the diagnostic.

## Rejected at ingress

| Code | Server detail | Cause | Fix |
|---|---|---|---|
| 401 | `Missing or invalid API key...` | Auth is on and no `ndif-api-key` header arrived | Set the key client-side |
| 400 | `Invalid API key format: '<key>'...` | Not parseable as a UUID — usually a truncated paste | Re-copy the key |
| 403 | `Invalid API key.` | A well-formed UUID with **no row in `keys`**. Validity is exactly "the row exists" | Issue a key, or check they are pointed at the right NDIF |
| 422 | a pydantic error list | The `data` form field is not JSON, or does not validate as a `BackendRequestModel` — almost always a schema mismatch | Upgrade nnsight |
| 400 | `Client nnsight version X is below the minimum supported Y.` | The version gate | Tell them the minimum |
| 503 | `Service temporarily unavailable: compute backend is reconnecting.` | The `ray:connected` flag is absent | Operator problem — check the ray container |
| 503 | `Auth backend unavailable.` | Postgres errored. Auth **fails closed**: a DB outage rejects everything rather than admitting unverified requests | Bring Postgres back |
| 500 | `Internal server error.` | Anything unhandled in ingress | The real traceback is only in the API log, logger `ndif.api` |

**401 or 403 proves auth is configured.** With `NDIF_POSTGRES_URL` unset,
`verify_api_key` returns immediately and no key check happens at all.

Every version-gate rejection is a 400: a *missing* header (read as "outdated
nnsight" — old clients do not send it), an *unparseable* version (a source
checkout with no installed distribution reports `""`), or one *below* the
minimum. The python check compares major.minor only.

## Sentences, not tracebacks

| Message begins | Class | What to tell the user |
|---|---|---|
| `Your request payload could not be read (` | `PayloadError` | Re-send it — most are transit damage. If it repeats it is a client/server nnsight mismatch, or a package the block imports that the server lacks. Compare `ndif env` with `ndif env --local` |
| `The model architecture on this server doesn't match the one your code was traced against` | `ArchitectureMismatchError` | The message carries the fix: run `nnsight.compare()` and align whatever it flags — almost always `transformers` |

Neither is fatal to the replica. The architecture case names the **first** path
where the two module trees diverge, not the layer the user was reaching for,
because the block references the whole envoy tree — do not read the named module
as the one they got wrong.

Underlying failures, all classified in one place (`BackendRequestModel.deserialize`),
so the trusted and sandboxed paths cannot drift:

| Underlying | Reported as | Cause |
|---|---|---|
| `ModuleNotFoundError` | `PayloadError` | The block references a package installed on the client, not the server. nnsight auto-registers *local, non-installed* modules, so this is normally an installed third-party package |
| `AttributeError` / `UnpicklingError` on a class | `PayloadError` | Client and server disagree on a type's shape — an nnsight version mismatch |
| `UnknownPersistentIdError` | `ArchitectureMismatchError` | A `Module:<path>` id the server's tree has no entry for |
| `ModuleNotFoundError: nnsight._c` | `PayloadError` | The optional C extension did not compile — `value.save()` on a non-tensor breaks server-side while working locally. `nnsight.save(x)` is the portable form |

## Canned server-side messages

| Message | Means |
|---|---|
| `Your job exceeded the execution timeout of Ns.` | The execution race lost. **There is no default** — an unconfigured deployment never emits this |
| `Your job was cancelled or preempted by the server.` | The kill switch fired for a reason other than parking — somebody deliberately cancelled it. A HOT→WARM demotion does *not* land here |
| `Replica was evicted while processing your request.` | The worker task was cancelled mid-dispatch — `ndif kill`, or a purge. An ordinary eviction re-queues instead |
| `Error starting model.` / `Error provisioning model.` | The controller could not place or ready a replica **and did not say why**. The real traceback is in the API log. On a failure before `READY` the Processor purges, so several users see this at once |
| `Could not deploy this model. <reason>` | The controller refused and explained: a mistyped `repo_id`, a gated repo, or `CANT_ACCOMMODATE`. These are the actionable cases — "try again later" would be wrong advice for all three |
| `Request cancelled by operator.` | Someone ran `ndif kill` on a still-queued request |
| `Error submitting request to model deployment.` | The Ray call to the actor raised something unclassified |
| `Critical server error occurred.` | A Ray connection error purged every Processor; everything queued at that moment was errored |

## Server exceptions, and whether they are fatal

**A user-caused exception is never fatal to a replica** — the actor catches it,
formats it, answers `ERROR`, keeps its weights and serves the next request. Only
two CUDA messages break that rule.

| Exception | User's view | Whose fault |
|---|---|---|
| `RunnerError` | `ERROR` with the block's own traceback | **user** — it is *always* user code; it is the untrusted path's carrier |
| Any block exception on the trusted path | same | **user** |
| `OutOfOrderError` | `ERROR`: `'<location>' was requested but the model already ran past it` | **user** — a worker parked on a location the forward pass never reached. An open-ended `tracer.iter` that outran the model **warns** instead of failing |
| `EarlyStopException` | nothing — the job completes normally | neither; `tracer.stop()`. Seeing it in a stack trace is normal |
| `PayloadError` / `ArchitectureMismatchError` | a sentence | **user** (environment drift) |
| timeout | `Your job exceeded the execution timeout` | usually user |
| `CachedActorError` | **nothing** — silently re-queued at the front | server: the actor was demoted to WARM before or during the run |
| `ActorDiedError`, actor-lookup `ValueError` | nothing — silently re-queued | server |
| `asyncio.CancelledError` | `Replica was evicted while processing your request.` | operator/server |
| `ConnectionError` / `runner process exited before its socket was ready` / `timed out waiting for the runner socket` | `ERROR` with a host-side traceback | server — the runner died, crashed at import, or took >30 s to bind |
| `torch.cuda.OutOfMemoryError` **at run** | a CUDA OOM traceback | user |
| `torch.cuda.OutOfMemoryError` / `RuntimeError` **at load** | `Error starting model...` | server — the size estimate was wrong, or the GPU was not actually free |
| `RuntimeError` from `verify_device_placement` | `Error starting model...` | server — a weight landed on `meta`, `cpu`, or an unassigned GPU |
| `NDIFConnectivityError` | CLI only: `Cannot connect to Ray at <url>` | server — Ray is uninitialized, the address is not listening, or the `Controller` actor is not resolvable |

### The two fatal CUDA errors

<!-- test: skip -->
```python
_UNRECOVERABLE_CUDA_ERRORS = (
    "device-side assert triggered",
    "an illegal memory access was encountered",
)
```

These poison the process's CUDA context permanently — every subsequent CUDA op
raises — so the actor kills itself and Ray restarts it, costing every other user
of that model a load cycle. The triggering request is still a user error and
still gets its traceback; the collateral damage is what makes this the one
user-caused failure that is fatal.

### The eviction bucket

<!-- test: skip -->
```python
EVICTED_ERRORS = (ValueError, ActorDiedError, CachedActorError)
```

On any of these the replica ends its own worker loop and hands the in-flight
request back to the **front** of the Processor's queue. The user is told nothing
and sees `QUEUED` a second time.

Two consequences. **An eviction mid-flight loses the work done so far** — there
is no checkpoint, the request re-runs from the start, and a model flapping
HOT↔WARM makes requests loop without ever finishing. And **you cannot catch an
actor's exception by type across the Ray boundary**: it arrives wrapped in a
`RayTaskError`, and the dual class that would satisfy `isinstance` is only built
when Ray applies `as_instanceof_cause()`, which it does not over Ray Client. The
cause has to be read off `.cause`. A bare `isinstance` matches nothing, silently,
which is how a whole retry path can be dead while every log line looks correct.

## A websocket that closes mid-run

| Cause | What happened to the job |
|---|---|
| The API restarted | **The job is gone.** Restarting the API restarts the dispatcher, and every per-model queue and in-flight request is a plain Python object in that process. The client gets no further status at all — not even an `ERROR` |
| An idle proxy timed out | The job is still running; the client just is not listening. Nothing replays it |
| The client was interrupted | Same — the server keeps going |

Nothing is stored for a blocking job, so there is no id to poll afterwards.
Re-submit. For a job that must survive a disconnect, submit non-blocking: each
non-`LOG` response is written to `responses/{id}.json` and `GET /response/{id}`
reads the latest back. A 404 there means "no status recorded yet", which the
client treats as still-running — it does **not** distinguish an unknown id.
