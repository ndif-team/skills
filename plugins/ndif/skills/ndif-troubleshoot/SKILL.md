---
name: ndif-troubleshoot
description: Diagnose a self-hosted NDIF that is broken — container won't start, no GPU, 503 "compute backend is reconnecting", requests that hang in QUEUED or DEPLOYING, CANT_ACCOMMODATE, CUDA out of memory on deploy or inside a block, "COMPLETED but the client can't download", "The model architecture on this server doesn't match", "Your request payload could not be read", "Starting Ray client server failed", code changes with no effect, a blank dashboard, empty Grafana panels. Use when someone pastes an NDIF server log, an nnsight RemoteError from their own deployment, or asks where NDIF's logs are, how to read `ndif queue`, or how to kill or reconstruct a request. For the public ndif.us service use the nnsight plugin's `remote` skill instead.
---

# Troubleshooting a self-hosted NDIF

Triage first: decide which component to blame, then open the section. Anything
about the *client's* code rather than the server belongs to the nnsight plugin's
`debugging` and `remote` skills.

## The first five commands

```bash
just ps                      # container status + health (compose), or `docker ps`
just logs api                # follow one service: api | ray | dashboard | redis | minio
curl -s localhost:8001/ping        # "pong" — only proves the web process is alive
curl -i localhost:8001/connected   # 200 = Ray reachable, 503 = dispatcher reconnecting
ndif status                        # what the controller thinks is deployed, and where
```

Then `ndif queue` (asks the dispatcher directly, 5 s timeout), `ndif doctor`
(local prerequisites — only meaningful on a host install) and `ndif version` /
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
| `CANT_ACCOMMODATE: placed 0 of 1 new replicas...` | [OOM on deploy vs OOM in a block](#oom-on-deploy-vs-oom-in-a-block) |
| `CUDA out of memory ... N MiB allowed` inside a user's block | [OOM on deploy vs OOM in a block](#oom-on-deploy-vs-oom-in-a-block) |
| `ndif status` cycles a replica `RUNNING` → `UNHEALTHY` | [OOM on deploy vs OOM in a block](#oom-on-deploy-vs-oom-in-a-block) |
| Job reaches `COMPLETED`, then the client fails to download | [COMPLETED but no result](#completed-but-no-result) |
| `The model architecture on this server doesn't match...` / `Your request payload could not be read (...)` | [Version mismatch](#version-mismatch) |
| `Starting Ray client server failed` after ~40 s, from `ndif status` or the dashboard | [Ray client server failed](#ray-client-server-failed) |
| A code change has no effect | [The stale image](#the-stale-image) |
| The dashboard returns JSON instead of a UI, or 503s on `/api/status` | [Dashboard problems](#dashboard-problems) |
| Grafana panels are empty | [Telemetry missing](#telemetry-missing) |
| `ndif queue` prints `No response from the dispatcher` | the **API** is down, not Redis — restart `api` |
| `ndif evict gpt2` prints nothing to evict | model-key mismatch (revision is part of the key), or the replica is only WARM |

## A service will not start

Every NDIF container is the **same image** with a different `NDIF_SERVICE`, and
its default command is `ndif start --foreground`. A startup crash is almost always
in that service's `start.sh` or its first import.

| What the log shows | Cause |
|---|---|
| Nothing at all, the container never runs | A `depends_on: service_healthy` gate — `api` waits on `postgres` **and** `influxdb` even though it can run without either |
| `ERROR: Cannot write to Ray temp directory` | `NDIF_RAY_TEMP_DIR` is not writable |
| `Waiting for Ray head at ...` forever | A worker node's `NDIF_RAY_HEAD_ADDRESS` cannot be reached. Check the port — 6385, not 6379 |
| A Python `ImportError` for an extra | The image installs its extras with `--no-deps`; a new dependency needs a `requirements.txt` entry |
| `queue/config.py` raising at import | A non-integer or non-positive `NDIF_QUEUE_*` / `NDIF_AUTOSCALING_*` value — deliberate: a typo fails the process rather than silently defaulting |

The compose project is named `dev`, so the containers are `dev-api-1`,
`dev-ray-1`, `dev-dashboard-1` if you reach past `just` for `docker logs`.

## No GPU

The ray node reports `cuda_memory_bytes` from torch and passes it as a custom Ray
resource. With no visible GPU that number is 0, Ray advertises no `GPU` resource,
and the controller **skips every node without one** — the cluster then has zero
nodes and every deploy fails with `No GPU nodes available.` (a different failure
from `CANT_ACCOMMODATE`, which means nodes exist but none has room).

```bash
nvidia-smi                                              # on the host
docker compose -f docker/docker-compose.yml exec ray nvidia-smi
just logs ray | head -20                                # the "resources:" line
docker run --rm --gpus all ndif/ndif doctor
```

Causes, in order: the NVIDIA Container Toolkit is not installed (or Docker was
not restarted after installing it) — the container then fails to create outright;
or the image's CUDA line is newer than the driver supports, which does **not**
fail loudly, `torch.cuda.is_available()` simply returns `False`. Check with
`docker run --rm ndif/ndif version` against `nvidia-smi`'s header and take
`0.1.0-cu126` for any 12.x driver, `0.1.0-cu130` for CUDA 13 drivers and Blackwell.

`shm_size: 4gb` is not optional either: Ray's plasma store lives in `/dev/shm`
and Docker's 64 MB default makes Ray spill to disk or fail, with an error that
never mentions shm.

## Ray is still booting — or is not

Ray takes about **60–90 seconds**. Until it is up, `/connected` answers
`reconnecting`, `/request` 503s with `Service temporarily unavailable: compute
backend is reconnecting.`, and the api log prints `Error connecting to Ray`
tracebacks roughly once a second. **That is expected during boot.**

In `just logs ray` you want:

```text
Starting Ray head node with resources: {"head": 10, "cuda_memory_bytes": ..., "cpu_memory_bytes": ...}
Starting NDIF controller...
```

If it never appears, look at the first error after `Starting Ray head node`.

## Healthy API, dead dispatcher

`GET /ping` has no dependencies, so it is useless for telling you whether work is
being done. The trap is the Redis flag `ray:connected`: the dispatcher sets it on
connect and deletes it while reconnecting, and **it has no TTL**. If the
dispatcher process dies outright the flag survives forever, so `/connected`,
`/request`, `/status` and `/env` all keep reporting healthy while nothing is
dispatched. Requests pile up unserved and clients sit on `RECEIVED`.

```bash
redis-cli get ray:connected      # "1" — proves nothing on its own
redis-cli llen queue             # should be 0 or a small transient number
ndif queue                       # asks the dispatcher directly; 5s timeout
```

A growing `llen queue` plus `No response from the dispatcher` is the signature.
The dispatcher is a child of the API's gunicorn master, so `just restart api`
brings it back — **and drops every queued and in-flight request.** Redis's
`queue` list is the only durable point in the path; once popped, a request exists
only as a Python object in that process. Clients on a blocking websocket get no
further status at all, not even an `ERROR`.

A health check that actually detects this has to observe the *queue*, not the API.

## Stuck requests

| Observation | Cause |
|---|---|
| Never even reaches `QUEUED` | The dispatcher is not popping the Redis list |
| `QUEUED` with a rising position | Genuinely behind other work — `ndif queue` for depth and replica count |
| `QUEUED` at position 1, forever | Autoscaling cannot add a replica, or the only replica is wedged |
| `PROVISIONING` / `DEPLOYING` for many minutes | `Replica.wait` polls `__ray_ready__` **with no timeout**. Weights may just be downloading — check `ndif status` for `DEPLOYING` |
| `QUEUED` twice with no error in between | A replica was evicted mid-flight and the request was silently pushed back to the **front** of the queue. Expected, not a bug |

Autoscaling keys off **head-of-line wait**, not depth: one request that has waited
past `NDIF_AUTOSCALING_WAIT_THRESHOLD_S` (30 s) triggers a scale-up; a hundred
that all arrived a second ago do not. The cap is `NDIF_AUTOSCALING_MAX_REPLICAS`
(3), and it never scales down.

There is **no execution timeout by default** — `NDIF_DEFAULT_EXECUTION_TIMEOUT_SECONDS`
is unset, so a block runs until it finishes and holds its replica for the
duration. When a timeout *is* set and a request blows past it with no `ERROR`,
that is usually because the timeout is delivered by injecting an exception into
the execution thread, which CPython only delivers at a bytecode boundary — it
cannot interrupt a CUDA kernel already running.

Unstick one with `ndif kill <request_id>`; recover a wedged replica with
`ndif restart <checkpoint> --replica <id>` (Ray respawns it and the in-flight
request is re-queued rather than dropped). The full walk from request id to root
cause is [references/reading-logs.md](references/reading-logs.md) and
`docs/runbooks/debug-a-stuck-request.md`.

## OOM on deploy vs OOM in a block

Same exception class, completely different handling.

| | At load | Inside a block |
|---|---|---|
| Where | the actor's `__init__` | the user's traced code |
| Who sees it | the **queue** — every user waiting on that model gets `Error starting model...` | one user, with a CUDA traceback |
| Replica after | never became ready; Ray restarts it and it OOMs again (`RUNNING`→`UNHEALTHY` cycling) | still healthy, still serving |
| Fix | sizing and placement | the user's batch size, sequence length, or saved activations — or raise padding |

`CUDA out of memory ... N MiB allowed` **on a nearly empty card** is not a
mystery: the actor caps per-process GPU memory at the model's size ×
`NDIF_DEFAULT_PADDING_FACTOR` (+ `NDIF_DEFAULT_PADDING_BIAS`) precisely so a
runaway request hits its own limit instead of trampling a co-tenant.

`RuntimeError: '<weight>' is on 'cpu', expected one of CUDA devices [...]` is
more common than a raw OOM at load and more informative: the budget was too small,
accelerate quietly offloaded the overflow to CPU, and the post-load check refused
to serve a half-CPU model. Read it as "the estimate was too low".

**`NDIF_MODEL_CACHE_PERCENTAGE` does not help a GPU shortage** — it scales host
RAM for the WARM cache. For GPU pressure: evict explicitly, deploy the incoming
model `--pinned` (which waives the age check), lower
`NDIF_MINIMUM_DEPLOYMENT_TIME_SECONDS`, or tune the padding knobs. The whole
procedure, including what the ledger cannot see, is in the `ndif-operate` skill
and `docs/runbooks/model-oom-on-deploy.md`.

## COMPLETED but no result

The job succeeded and the client cannot fetch it. A result **under**
`NDIF_MAX_SOCKET_RESULT_BYTES` (4 MiB) rides back on the response itself and
never touches the object store, so this only bites above that — and always for a
non-blocking job.

A presigned URL is an HMAC **over the request including the host**, so it must be
signed with the address the downloader will actually hit:

| Variable | Compose value | Role |
|---|---|---|
| `NDIF_OBJECT_STORE_URL` | `http://minio:9000` | what the **server** uploads through |
| `NDIF_OBJECT_STORE_PUBLIC_URL` | `http://localhost:9000` | what the client's GET is **signed with** |

Swap them and every job completes and then fails to download. Leave the public
one empty and it falls back to the internal URL — correct for a single host,
wrong for compose, and wrong for anyone not on the machine. Also publish port
9000. Other shapes: `403 SignatureDoesNotMatch` (different credentials, or a
proxy rewrote the Host header) and `403 Request has expired` (**presigned URLs
last one hour** — a real constraint for a non-blocking job polled later).

Reproduce it end to end by `curl -I`-ing the `data` field of a completed job's
`GET /response/{id}` **from the client machine**.

## Version mismatch

Two client-visible messages, both a *sentence with no traceback* — which is the
triage rule: **a traceback means the caller's code ran and failed; a sentence
means the server refused the request before their code existed.**

| Message | Cause |
|---|---|
| `Your request payload could not be read (...)` | The blob did not deserialize — truncated or corrupted in transit, a compress-flag mismatch, a package the block imports that the server lacks, or a client/server nnsight mismatch. Re-sending resolves most of these |
| `The model architecture on this server doesn't match the one your code was traced against: it has no module at '<path>'.` | The server's live model tree has no module at some `Module:<path>` — nearly always a **transformers** difference. It names the *first* diverging path, not the layer the user was reaching for |

Confirm with `ndif env` against `ndif env --local`, or have the user run
`nnsight.compare()`. On a compose stack, also run `just nnsight` — an editable
nnsight checkout is bind-mounted over the image's copy, so the server may not be
running the nnsight you think it is.

Setting `NDIF_MIN_NNSIGHT_VERSION` converts the whole class of failure into a
clear 400 at submit (`Client nnsight version X is below the minimum supported
Y.`). Both version minimums are read **once at import** and an empty string
counts as unset, so changing either needs a fresh API process — and under compose
`just up api` (which recreates the container), not `just restart api`.

Two related traps: an nnsight installed from a source tree rather than a
distribution reports an **empty** version string, which with a minimum configured
is a 400; and `value.save()` on a non-tensor depends on nnsight's compiled
`nnsight._c` extension **on the server** — the 0.8 wheels ship it, an sdist
install compiles it and silently skips it with no compiler present.

## Ray client server failed

`ndif status` or the dashboard waits ~40 s and then fails with
`Starting Ray client server failed`, naming an `.err` file that is empty. The Ray
client proxier's forked per-client server died at birth, with
`skipping fork() handlers` in `ray_client_server.err` just before
`SpecificServer startup failed`.

`ray/start.sh` exports `GRPC_ENABLE_FORK_SUPPORT=0` before `ray start`, which
removes it. If someone still sees it, that variable was overridden. It is a
retry-able failure — the API's dispatcher loops — so it only slows boot.

## The stale image

A code change with no effect is almost always this: the source is **baked into
the image** and there is no bind mount of `src/`, while `just up` only builds when
the image is missing. Use `just ta` (down → build → up). `just restart` is worse
— it bounces the container without rebuilding anything.

nnsight is the exception: `just up`/`just ta` bind-mount an **editable** nnsight
over the image's copy, so client-side nnsight changes are picked up without a
rebuild. A non-editable one is deliberately not mounted. `just nnsight` prints
which applies.

Env-var changes are the other half of this. Values are read at import, so a
changed variable needs the process restarted; a changed compose `environment:`
block needs the container **recreated** (`just up api`), not restarted.

## Dashboard problems

| Symptom | Cause |
|---|---|
| `/api/status` returns 503 | The dashboard reaches the Ray controller directly (deliberately bypassing the API's cached `/status`). `NDIF_RAY_ADDRESS` is wrong, Ray is down, or the `Controller` actor is not resolvable in the `NDIF` namespace — the third is the one people miss, since Ray can be perfectly healthy while the controller is gone |
| The UI opens with no login prompt | `NDIF_DASHBOARD_DEV_MODE=true` — the compose default — makes `require_auth` return the configured username unchecked |
| Nobody can log in | Dev mode off with an empty `NDIF_DASHBOARD_PASSWORD_HASH`: `verify_password` returns `False` unconditionally |
| The reconcile/monitor crons never run | `start.sh` wires cron only where `cron` is on `PATH` and `/etc/cron.d` is writable — true in the container, false on a laptop |
| Every schedule event vanished after a rebuild | The data dir was not a volume |
| `{"skipped": "controller_status_unavailable"}` from reconcile | Ray was down; the pass did nothing deliberately, rather than stacking a second pinned replica on one already serving |

## Telemetry missing

Both providers are **fail-open**: nothing errors when they are unconfigured, so
"no data" is the only symptom.

```bash
just logs api | grep "Loki telemetry enabled"      # Loki is opt-in on NDIF_LOKI_URL
just logs api | grep "InfluxDB telemetry enabled"  # on by default, silently no-ops if misconfigured
```

`NDIF_LOKI_URL is set but python-logging-loki is not installed` means the
`metrics` extra was not installed. Prometheus scrapes exactly one target,
`ray:8080` — Ray's metrics-export port; NDIF uses no Ray Serve.

And the label that hides model-actor logs: they are **`service="model"`**, not
`service="ray"`. Filter on `logger` instead when in doubt.

## References

- [references/reading-logs.md](references/reading-logs.md) — where every log
  actually lives (including the controller's, which `just logs ray` never shows),
  the LogQL queries, the lifecycle timeline, and the end-to-end procedures for a
  stuck request and a failed one.
- [references/error-messages.md](references/error-messages.md) — every
  client-visible message and server exception mapped to a cause, with which ones
  are fatal to a replica.

## Related skills

- `ndif-selfhost` — prerequisites, tags, ports, and the configuration model.
- `ndif-operate` — deploys, sizing, eviction, auth.
- `ndif-develop` — when the fix is a code change.
- nnsight plugin `debugging` — when the failure is in the user's own trace.
