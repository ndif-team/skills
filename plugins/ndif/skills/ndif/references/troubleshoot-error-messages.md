# Error messages, by their text

Three rules do most of the triage.

- **An HTTP status code means the request was refused at the door and never
  queued.** The client shows it as `RemoteError: Failed to send request: <the
  server's sentence>`. Everything after `RECEIVED` arrives as a status, and a
  failure there is an `ERROR` status.
- **A traceback that ends in the user's own line means their code ran and
  raised.** A sentence with no traceback means the server refused or failed the
  request.
- **Read a traceback from the bottom.** Only nnsight's frames are stripped, so
  Ray, asyncio and NDIF frames come before the user's lines.

`docs/reference/error-messages.md` is the exhaustive list.

## Refused at the door

| Code | Detail | Cause |
|---|---|---|
| 401 | `Missing or invalid API key. ...` | Auth is on and no `ndif-api-key` header arrived |
| 400 | `Invalid API key format: '<key>'. ...` | Not a UUID. `'local'` is an nnsight client with no key set |
| 403 | `Invalid API key.` | A well-formed UUID with no row in `keys`, which is every key after Postgres lost its data |
| 400 | `Client nnsight version X is below the minimum supported Y.`, `Client nnsight version was not provided.`, `Malformed nnsight version` | `NDIF_MIN_NNSIGHT_VERSION` is set and the client is older or does not report one. The Python gate compares major.minor only |
| 422 | a pydantic error list, or `Invalid JSON: ...` | The request envelope does not match the server's schema: a client and server too far apart |
| 503 | `Service temporarily unavailable: compute backend is reconnecting. ...` | The dispatcher has no Ray connection: [troubleshoot](troubleshoot.md#ray-is-booting-or-is-not) |
| 503 | `Auth backend unavailable.` | Postgres is down or erroring, or `NDIF_POSTGRES_URL` is set without `asyncpg` installed. Auth fails closed |
| 503, 504 | `Could not retrieve cluster status.`, `Timed out waiting for cluster status.` (and the same for `cluster env`) | On `/status` or `/env`: the dispatcher did not refresh the cache. Busy, or [dead](troubleshoot.md#a-dead-dispatcher) |
| 500 | `Internal server error.` | Unhandled in the API. The traceback is in the api log as `api unhandled error` |

Every refusal is also a line in the api log: `[ndif.api] api request rejected:
403 path=/request ...`. A 401 or 403 proves auth is on.

## Sentences from the model actor

| Message begins | Means |
|---|---|
| `Your request payload could not be read (` | The block could not be deserialized. `ModuleNotFoundError` in the parentheses is a package the server lacks; an unpickling error on a class is an nnsight difference; otherwise re-send once |
| `The model architecture on this server doesn't match the one your code was traced against` | Client and server built different module trees, nearly always a transformers difference. The path named is the first divergence |
| `Your job exceeded the execution timeout of Ns.` | A configured timeout. There is none by default |
| `Your job was cancelled or preempted by the server.` | The actor stopped the block on request: `ndif kill`. A blocking client has already been told `Request cancelled by operator.` |

None of these hurts the replica.
[Version mismatch](troubleshoot.md#version-mismatch) has the comparison.

## ERROR statuses from the queue

| Message | Means |
|---|---|
| `Request cancelled by operator.` | `ndif kill`, of a queued or a running request |
| `Could not deploy this model. <reason>` | The controller refused and said why: `CANT_ACCOMMODATE`, a mistyped or gated repository (Hugging Face's own text), or the `trust_remote_code` rule below. Retrying does not help |
| `Could not deploy this model. This model runs code from its own repository (trust_remote_code=True), and deploying such a model needs an API key with the trust_remote_code or trusted tag...` | Nothing is running under that key and no queued request may deploy it. An admin deploys it (`ndif deploy --trust-remote-code`) or grants the tag |
| `Error starting model.` / `Error provisioning model.` | Placing or loading a replica failed for a reason the server does not pass on. Every request queued for the model gets it. The traceback is in the api log under `ndif.queue.processor`, and a failed load in the controller log as `failed during create` |
| `Error submitting request to model deployment.` | The call to the actor failed. Right after a fatal CUDA error or a restart of `ray` it is the actor or the connection coming back: retry |
| `Critical server error occurred.` | The dispatcher lost Ray and purged every queue; everything waiting got this |
| `Replica was evicted while processing your request.` | The dispatcher gave up on a dispatch: a kill the actor did not answer within five seconds, or a purge. An ordinary eviction re-queues instead |

## A traceback

| Ends in | Means |
|---|---|
| the user's own line | Their code raised. The replica keeps serving; a retry needs a code change |
| `OutOfOrderError: '<location>' was requested but the model already ran past it` | Modules accessed out of forward-pass order. With `.grad` in the location, a gradient asked for by an untrusted request on a sandbox actor |
| `NotImplementedError: Cannot copy out of meta tensor; no data!` | An untrusted request saved something derived from a weight; the runner holds no weights |
| `torch.OutOfMemoryError: CUDA out of memory ... N MiB allowed` | The block passed the replica's memory cap: [sizing and placement](operate-sizing-and-placement.md#reading-a-refusal) |
| `ConnectionError: connection closed before message complete`, `runner process exited before its socket was ready`, `timed out waiting for the runner socket` | The sandbox runner died or did not start. Expected when a timeout or a kill stopped it; otherwise a server fault |
| NDIF or Ray frames only | A server fault. The full traceback is in the model actor's log as `model execution errored` |

### Three CUDA errors restart the replica

`device-side assert triggered`, `an illegal memory access was encountered` and
`unspecified launch failure` leave the process's CUDA context unusable, so the
actor kills itself and Ray brings it back with a reload. The request that
caused it still gets its traceback. No action is needed: `ndif status` keeps
showing the replica HOT, requests that arrive during the reload get `Error
submitting request to model deployment.` or wait, and the next one works. The
usual innocent cause of the first is a prompt longer than the model's context.

## No error at all

| What happens | Means |
|---|---|
| `QUEUED` a second time | The replica was evicted or demoted to WARM under the request, which went back to the front of the queue and runs again from the start. The model actor logs `model execution preempted; requeued` |
| The status stops changing | [A stuck request](troubleshoot.md#a-stuck-request) |
| `COMPLETED` with nothing saved | `tracer.stop()` or a missing `.save()`: a client question, for the `nnsight` skill |
| The client raises a pydantic `ValidationError ... Invalid JSON: EOF while parsing` | Its websocket closed: the API restarted. The job is lost to that client even if the block finishes |

A blocking job keeps nothing on the server. For work that must survive a
dropped connection, submit non-blocking (`blocking=False`): each status is
stored and `GET /response/{id}` returns the latest.

## After COMPLETED: the download

`data` on the `COMPLETED` response is the result itself, or a presigned URL.

| The fetch answers | Cause |
|---|---|
| a connection error, or it hangs | The host and port in `NDIF_OBJECT_STORE_PUBLIC_URL` are not reachable from the client |
| 403 `SignatureDoesNotMatch` | The client reached the store under a different host or port than the URL was signed for (`127.0.0.1` for `localhost` is enough, or a proxy that rewrites `Host`), or the store's credentials differ from the server's |
| 404 `NoSuchBucket` | A different S3 server answered at that address: the variable, or a tunnel, points at another store |
| 403 on a URL that used to work | It is over an hour old. Run the job again |

## CLI output

- `ndif deploy` exits non-zero when any model is not ready, after a line `✗
  <model_key>: <reason>`.
- `ndif evict` and `ndif restart` print `✗ <model_key>: nothing to evict` and
  `✗ <model_key>: no HOT replicas to restart` and exit 0.
- `Cannot connect to Ray at <url>` from any model command: Ray is down, or
  `NDIF_RAY_ADDRESS` is wrong where the command runs.
- `No response from the dispatcher for <event>. Is the API running?` from
  `ndif queue` or `ndif kill`: the API is down or its dispatcher is dead.
