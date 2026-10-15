---
name: troubleshoot
description: Diagnose a self-hosted NDIF that is broken — container won't start, no GPU, 503 "compute backend is reconnecting", requests that hang in QUEUED or DEPLOYING, CANT_ACCOMMODATE, CUDA out of memory on deploy or inside a block, "COMPLETED but the client can't download", "The model architecture on this server doesn't match", "Your request payload could not be read", "Cannot switch to a different thread" / "parent cannot be on a different thread", "Starting Ray client server failed", code changes with no effect, a blank dashboard, empty Grafana panels. Use when someone pastes an NDIF server log, an nnsight RemoteError from their own deployment, or asks where NDIF's logs are, how to read `ndif queue`, or how to reconstruct a failed request. For the public ndif.us service use the nnsight plugin's `remote` skill instead.
---

# Troubleshooting a self-hosted NDIF

Triage first: decide which component to blame, then open the section. Anything
about the *client's* code rather than the server belongs to the nnsight plugin's
`debugging` and `remote` skills. The docs' entry point is
`docs/operating/troubleshooting.md`; exact error text is in
`docs/reference/error-messages.md`.

## The first five commands

```bash
just ps                      # container status + health (compose), or `docker ps`
just logs api                # follow one service: api | ray | dashboard | redis | rustfs
curl -s localhost:8001/ping        # "pong" — only proves the web process is alive
curl -i localhost:8001/connected   # 200 = Ray reachable, 503 = dispatcher reconnecting
ndif status                        # what the controller thinks is deployed, and where
```

Then `ndif queue` (asks the dispatcher directly, 5 s timeout), `ndif doctor`
(host prerequisites — only meaningful on a host install) and `ndif version` /
`ndif env` versus `ndif env --local` (version drift).

**Run `ndif version` early.** A large share of the confusing failures on this
page are a client and a server disagreeing about nnsight, transformers or torch.

## Symptom index

| What you see | Section |
|---|---|
| `docker run` exits immediately, `could not select device driver ... [[gpu]]` | [No GPU](#no-gpu) |
| A compose container is `Restarting`, or `api` never starts with no logs | [A service will not start](#a-service-will-not-start) |
| `cuda_memory_bytes: 0` in the ray log; `ndif status` shows 0 nodes; `No GPU nodes available.` | [No GPU](#no-gpu) |
| `/connected` says `reconnecting`, api logs `Error connecting to Ray` once a second | [Ray is still booting — or is not](#ray-is-still-booting--or-is-not) |
| `/ping` 200 but every request hangs; `redis-cli llen queue` growing | [Healthy API, dead dispatcher](#healthy-api-dead-dispatcher) |
| A request sits in `QUEUED` / `PROVISIONING` / `DEPLOYING` for ages | [Stuck requests](#stuck-requests) |
| Requests fail right after an `ndif kill` with `ValueError: Cannot switch to a different thread` or `ValueError: parent cannot be on a different thread` (last frames ordinary torch/transformers lines) | [Stuck requests](#stuck-requests) — restart the replica while the killed block still runs |
| `RemoteError` whose traceback ends in **your own** line | User code raised there — reading a traceback from the bottom, in [references/error-messages.md](references/error-messages.md#rules-that-do-most-of-the-triage) |
| `device-side assert triggered` / `an illegal memory access was encountered` | [The two fatal CUDA errors](references/error-messages.md#the-two-fatal-cuda-errors) — the actor restarts itself, status stays HOT, the next request works |
| `CANT_ACCOMMODATE`, `CUDA out of memory`, a replica cycling `RUNNING` → `UNHEALTHY` | [OOM on deploy vs OOM in a block](#oom-on-deploy-vs-oom-in-a-block) |
| The first request after `just up ray` errors `Error starting model.` | [OOM on deploy vs OOM in a block](#oom-on-deploy-vs-oom-in-a-block) — not an OOM |
| Job reaches `COMPLETED`, then the client fails to download | [COMPLETED but no result](#completed-but-no-result) |
| `The model architecture on this server doesn't match...` / `Your request payload could not be read (...)` | [Version mismatch](#version-mismatch) |
| Every remote `.save()` fails with `AttributeError: 'Tensor' object has no attribute 'save'` | [Version mismatch](#version-mismatch) — the `nnsight._c` extension |
| `Starting Ray client server failed` after ~40 s, from `ndif status` or the dashboard | [Ray client server failed](#ray-client-server-failed) |
| A code change or an env edit has no effect | [The stale image](#the-stale-image) |
| Every API key suddenly 403s | Postgres lost its anonymous volume on `just down` / `just ta`; re-insert the keys (`operate` skill, production reference) |
| The dashboard returns JSON instead of a UI, or 503s on `/api/status` | [Dashboard problems](#dashboard-problems) |
| Grafana panels are empty | [Telemetry missing](#telemetry-missing) |
| `ndif queue` prints `No response from the dispatcher` | the **API** is down, not Redis — or it restarted under five seconds ago |
| `ndif evict gpt2` prints `✗ ... nothing to evict` | model-key mismatch (revision and task are part of the key), or the replica is only WARM. It exits 0 |

## A service will not start

Every NDIF container is the **same image** with a different `NDIF_SERVICE`, and
its default command is `ndif start --foreground`. A startup crash is almost always
in that service's `start.sh` or its first import.

| What the log shows | Cause |
|---|---|
| Nothing at all, the container never runs | A `depends_on: service_healthy` gate — `api` waits on four health checks (`redis`, `rustfs`, `postgres`, `influxdb`) even though it can run without the last two. Fix the unhealthy one or drop its `depends_on` entry |
| `ERROR: Cannot write to Ray temp directory` | `NDIF_RAY_TEMP_DIR` is not writable |
| `validate_socket_filename failed: AF_UNIX path length cannot exceed 107 bytes` in the ray log | `NDIF_RAY_TEMP_DIR` is too long for Ray's socket paths; use a short one such as `/tmp/ndif-ray` |
| `ndif start` said ✓ but `ndif info` says `stopped` | The service died after the 2 s liveness check; read `$NDIF_HOME/logs/<service>.log` |
| `Waiting for Ray head at ...` forever | A worker node's `NDIF_RAY_HEAD_ADDRESS` cannot be reached. Check the port — 6385, not 6379. `docs/runbooks/add-a-gpu-node.md` |
| A Python `ImportError` for an extra | The image installs its extras with `--no-deps`; a new dependency needs a `requirements.txt` entry |
| `queue/config.py` raising at import | A non-integer or non-positive `NDIF_QUEUE_*` / `NDIF_AUTOSCALING_*` value — deliberate: a typo fails the process rather than silently defaulting |

The compose project is named `dev`, so the containers are `dev-api-1`,
`dev-ray-1`, `dev-dashboard-1` if you reach past `just` for `docker logs`.

## No GPU

The ray node reports `cuda_memory_bytes` from torch as a custom Ray resource.
With no visible GPU that number is 0, Ray advertises no `GPU` resource, and the
controller **skips every node without one** — the cluster has zero nodes and
every deploy fails with `No GPU nodes available.` (a different failure from
`CANT_ACCOMMODATE`, which means nodes exist but none has room).

Check `nvidia-smi` on the host and in the `ray` container, and
`just logs ray | grep "resources:"`. Causes, in order: the NVIDIA Container Toolkit is not installed (or Docker was
not restarted after installing it) — the container then fails to create outright;
or the image's CUDA line is newer than the driver supports, which does **not**
fail loudly: `nvidia-smi` works inside the container but `torch.cuda` sees
nothing. Compare `docker run --rm ndif/ndif version` against `nvidia-smi`'s
header; tags are one per CUDA line — `<version>-cu126` for any 12.x driver,
`<version>-cu130` for CUDA 13 drivers and Blackwell (bare `<version>` and
`latest` are the cu126 build).

`shm_size: 4gb` is not optional either: Ray's plasma store lives in `/dev/shm`
and Docker's 64 MB default makes Ray spill to disk or fail, with an error that
never mentions shm.

## Ray is still booting — or is not

Ray takes **15–90 seconds**. Until then `/connected` answers `reconnecting`,
`/request` 503s `compute backend is reconnecting.`, and the api log prints `Error
connecting to Ray` about once a second — **expected during boot**. `just logs ray`
should show `Starting Ray head node with resources: {...}` then `Starting NDIF
controller...`; if not, read the first error after the first line.

## Healthy API, dead dispatcher

`GET /ping` has no dependencies, and the dispatcher's Redis flag `ray:connected`
**has no TTL**: if the dispatcher dies outright, `/connected` and `/request` keep
reporting healthy and `/status` / `/env` serve their cached blob until it expires
(then 503 `Could not retrieve cluster status.` or 504 `Timed out waiting...`).
Clients sit on `RECEIVED`. The signature is a growing `redis-cli llen queue` plus
`No response from the dispatcher` from `ndif queue`. `just restart api` brings the
dispatcher back — **and drops every in-flight request** (once popped from Redis a
request exists only in that process; blocking clients get no further status). A
health check that detects this must watch the *queue*, not the API.

## Stuck requests

| Observation | Cause |
|---|---|
| Never even reaches `QUEUED` | The dispatcher is not popping the Redis list |
| `QUEUED` with a rising position | Genuinely behind other work — `ndif queue` for depth and replica count |
| `QUEUED` at position 1, forever | Autoscaling cannot add a replica, or the only replica is wedged |
| `PROVISIONING` / `DEPLOYING` for many minutes | `Replica.wait` polls `__ray_ready__` **with no timeout**. Weights may just be downloading — check `ndif status` for `DEPLOYING` |
| `QUEUED` twice with no error in between | A replica was evicted or demoted mid-flight and the request went back to the **front** of the queue. Expected |

There is **no execution timeout by default**; the timeouts table is in
[references/reading-logs.md](references/reading-logs.md), with the full walk from
request id to root cause (`docs/runbooks/debug-a-stuck-request.md`).

**Unsticking one is the `operate` skill's "Killing a request".** What bites here:
`ndif kill` on an *executing* request cancels only the dispatch — the actor runs
the block to completion. **While that block still runs**, work dispatched to the
replica fails with `ValueError: Cannot switch to a different thread` or
`ValueError: parent cannot be on a different thread` (the message is the tell; the
last frames are ordinary torch/transformers lines), so restart it:
`ndif restart <checkpoint> --replica <id>`. Once the block finishes on its own
(Loki `{service="model"} |= "<request id>"` shows `Your job has been completed.`)
the next request is clean with no restart. A wedged replica gets the same restart.

## OOM on deploy vs OOM in a block

Same exception class, completely different handling.

**At load** (the actor's constructor): every user waiting on the model gets
`Error starting model.`, the controller log says `Deployment <model_key> failed
during create`, and the replica never becomes ready — a sizing problem.
`RuntimeError: '<weight>' is on 'cpu', expected one of CUDA devices [...]` reads
as "the estimate was too low". **Inside a block**: one user gets a CUDA traceback
and the replica keeps serving. `CUDA out of memory ... N MiB allowed` on a nearly
empty card is the per-process cap — the replica's padded share of each card —
doing its job; fix the batch, sequence length or saves, or raise padding.
`NDIF_MODEL_CACHE_PERCENTAGE` does not help a GPU shortage (it sizes host RAM).

**`Error starting model.` right after `just up ray` is not an OOM**: the API still
holds its old compute connection (its log shows a `ConnectionError`), and the next
request may 503 until it reconnects. Retry.

The ledger, every refusal message, what to evict and how to right-size are the
`operate` skill's sizing reference and `docs/runbooks/model-oom-on-deploy.md`.

## COMPLETED but no result

The job succeeded and the client cannot fetch it.

A *blocking* result at or under the inline cap rides back on the `COMPLETED`
response over Redis pub/sub; a larger one — and **every non-blocking result** — is
uploaded, and the client downloads a presigned URL. The cap is
`RedisProvider.max_publish_bytes`, `min(soft, hard // 2)` of Redis's
`client-output-buffer-limit pubsub`: **8 MiB on stock redis:7**, no env var (raise
the Redis limits to raise it). The cap is checked against the **compressed blob**
(a 7.5 MiB blob goes inline, a 9 MiB one by URL). The published frame is ~1.5×
that blob, which sets how fast undrained results fill the Redis buffer toward its
hard limit — not where the inline/URL boundary sits.
A presigned URL signs the host, so it must carry the address the client hits:

| Variable | Compose value | Role |
|---|---|---|
| `NDIF_OBJECT_STORE_URL` | `http://rustfs:9000` | what the **server** uploads through |
| `NDIF_OBJECT_STORE_PUBLIC_URL` | `http://localhost:9000` | what the client's GET is **signed with** — must be reachable **from the client** |

Swap them and every large or non-blocking job completes and then fails to
download. Leave the public one empty and it falls back to the internal URL —
correct for a single host, wrong for compose, and wrong for anyone not on the
machine. Set both on `api` and `ray`, and publish port 9000.

The client-side shapes: `ConnectError` or a hang (host unreachable from the
client), `403 SignatureDoesNotMatch` (other credentials, or a proxy rewrote Host),
**`404 NoSuchBucket`** (a *different* S3 answered at the public URL — wrong URL or
a tunnel to another store), `403 Request has expired` (URLs last one hour). The
full table is in [references/error-messages.md](references/error-messages.md).
Reproduce it by `curl -I`-ing the `data` field of a completed non-blocking job's
`GET /response/{id}` **from the client machine**.

## Version mismatch

Both messages are a *sentence with no traceback* — the server refused the request
before the user's code existed. `Your request payload could not be read (...)` is
transit damage (re-send once), a package the server lacks, or an nnsight
mismatch. `The model architecture on this server doesn't match...` is nearly
always a **transformers** difference; the path it names is the *first*
divergence, not the layer the user reached for. Confirm with `ndif env` against `ndif env --local`, or
`from nnsight import ndif; print(ndif.compare())` on the client. On compose, also
`just nnsight` — an editable nnsight checkout may be bind-mounted over the image's.

Setting `NDIF_MIN_NNSIGHT_VERSION` converts the whole class into a clear 400 at
submit. Both version minimums are read **once at import** and an empty string
counts as unset, so changing either needs `just up api`, not `just restart api`.
An nnsight installed from a source tree sends an **empty** version, which with a
minimum configured is a 400 `Client nnsight version was not provided.`

**Every remote `.save()` failing with `AttributeError: 'Tensor' object has no attribute 'save'`** is the server's nnsight
missing its compiled `nnsight._c` extension. The published wheels ship it; an
sdist install with no compiler silently skips it, and a fresh nnsight checkout
bind-mounted over the image has none until it is built in place
(`setup.py build_ext --inplace` with the image's toolchain).

## Ray client server failed

`ndif status` or the dashboard waits ~40 s, then fails `Starting Ray client server
failed` naming an empty `.err` file; `ray_client_server.err` shows `skipping
fork() handlers` before `SpecificServer startup failed`. `ray/start.sh` exports
`GRPC_ENABLE_FORK_SUPPORT=0` to prevent it — something overrode that variable.

## The stale image

A code change with no effect is almost always this: the source is **baked into
the image** with no bind mount of `src/`, while `just up` only builds when the
image is missing. Use `just ta` (down → build → up) — but `just ta <svc>` takes
the **whole** stack down, losing HOT deployments and, on compose, every API key in
Postgres. `just build` then `just up` replaces only the changed containers.

nnsight is the exception: `just up`/`just ta` bind-mount an **editable** nnsight
over the image's copy. `just nnsight` prints which applies.

**Env edits** need the container recreated: `just up <svc>`. `just restart <svc>`
restarts the process with its old environment and fails silently.

## Dashboard problems

| Symptom | Cause |
|---|---|
| `/api/status` returns 503 | It reaches the controller directly: `NDIF_RAY_ADDRESS` is wrong, Ray is down, or the `Controller` actor is not resolvable in the `NDIF` namespace |
| `localhost:8081/` returns JSON with a `hint` field | No built frontend: `NDIF_DASHBOARD_FRONTEND_DIST` points elsewhere or `dist/` was deleted. `npm ci && npm run build`, then rebuild the image |
| No login prompt / nobody can log in | `NDIF_DASHBOARD_DEV_MODE=true` (compose default) / dev mode off with an empty `NDIF_DASHBOARD_PASSWORD_HASH` |
| Crons never run; schedule events vanished after a rebuild | cron is wired only in the container; the data dir was not a volume |

## Telemetry missing

Both providers are **fail-open** and off until given a URL, so "no data" is the
only symptom. `just logs api | grep "telemetry enabled"`: no Loki line means
`NDIF_LOKI_URL` is unset (compose sets it for `api` and `ray`, not `dashboard`);
no InfluxDB line means `NDIF_INFLUX_URL` is unset, `NDIF_INFLUX_ENABLED=false`, or
the client library is missing (`metrics` extra). Prometheus scrapes only
`ray:8080`. Model-actor logs are **`service="model"`**, not `service="ray"`.

## References

- [references/reading-logs.md](references/reading-logs.md) — where every log lives
  (the controller's included), LogQL, timeouts, the stuck/failed procedures.
- [references/error-messages.md](references/error-messages.md) — every message and
  server exception mapped to a cause, and which are fatal to a replica.

## Related skills

- `selfhost` — prerequisites, tags, ports, and the configuration model.
- `operate` — deploys, sizing, eviction, killing a request, auth.
- `develop` — when the fix is a code change.
- nnsight plugin `debugging` — when the failure is in the user's own trace.
