# Troubleshooting a server

`SKILL.md` says where the logs are and has the commonest symptoms. This page is
the full index and the procedure for each. A failure in the client's own code,
rather than the server, belongs to the `nnsight` skill's debugging and remote
guides.

```bash
curl -s localhost:8001/ping         # the web process
curl -si localhost:8001/connected   # 200, or 503 while the dispatcher has no Ray
ndif status                         # what the controller has placed, and where
ndif queue                          # what the dispatcher is doing; fails if the dispatcher is gone
ndif doctor                         # GPU as torch sees it, binaries, which endpoints answer
ndif env                            # the server's package versions, to compare with the client's
```

On compose, `just ps` first: it shows which container is unhealthy or
restarting.

## Symptom index

| What you see | Go to |
|---|---|
| A container exits at once or keeps restarting; `api` never starts and has no log | [A service will not start](#a-service-will-not-start) |
| `could not select device driver ... [[gpu]]`; `cuda_memory_bytes: 0`; `Nodes: 0`; `No GPU nodes available.` | [No GPU](#no-gpu) |
| 503 `compute backend is reconnecting`; `Error connecting to Ray` every second | [Ray is booting, or is not](#ray-is-booting-or-is-not) |
| `/ping` and `/connected` are fine, every request sits on `RECEIVED` | [A dead dispatcher](#a-dead-dispatcher) |
| A request stays in `PROVISIONING`, `DEPLOYING`, `QUEUED` or `RUNNING` | [A stuck request](#a-stuck-request) |
| `CANT_ACCOMMODATE`; `CUDA out of memory`; `Error starting model.` | [Out of memory at load, or in a block](#out-of-memory-at-load-or-in-a-block) |
| `COMPLETED`, then the client fails to download | [COMPLETED but no result](#completed-but-no-result) |
| `The model architecture on this server doesn't match...`; `Your request payload could not be read`; every `.save()` fails with `'Tensor' object has no attribute 'save'` | [Version mismatch](#version-mismatch) |
| A code change or an environment edit has no effect | [A change has no effect](#a-change-has-no-effect) |
| Requests fail for some seconds after `ray` was restarted | Expected while the dispatcher reconnects: [sizing and placement](operate-sizing-and-placement.md#changing-a-controller-variable) |
| Every API key answers 403 | Postgres lost its data (on compose, `just down` or `just ta`): re-insert the keys, [production and auth](operate-production-and-auth.md#turning-both-on) |
| `ndif evict` prints `✗ ... nothing to evict` | The key does not match (task and revision are part of it) or nothing is deployed. It exits 0 |
| `ndif status` or the dashboard waits 40 s, then `Starting Ray client server failed` | Something set `GRPC_ENABLE_FORK_SUPPORT`; the ray start script needs it `0` |
| The dashboard shows JSON, returns 503, or will not log in | [The dashboard](#the-dashboard) |
| Grafana panels are empty | [Telemetry missing](#telemetry-missing) |
| Any other message | [error messages](troubleshoot-error-messages.md) |

## A service will not start

Every NDIF container is the same image with a different `NDIF_SERVICE`, so a
start-up crash is in that service's start script or its first import.

| The log shows | Cause |
|---|---|
| Nothing, and the container never runs (compose) | A health gate: `api` waits for redis, rustfs, postgres and influxdb. Fix the unhealthy one in `just ps` |
| `ERROR: Cannot write to Ray temp directory` | `NDIF_RAY_TEMP_DIR` is not writable |
| `OSError: validate_socket_filename failed: AF_UNIX path length cannot exceed 107 bytes` | `NDIF_RAY_TEMP_DIR` is too long for Ray's socket paths |
| `Waiting for Ray head at ...`, then `ERROR: Ray head ... not reachable after 60 attempts` | A worker cannot reach `NDIF_RAY_HEAD_ADDRESS`. The port is the head's Ray port, 6385, not Redis's 6379 |
| `ValueError: NDIF_AUTOSCALING_MAX_REPLICAS: 'x' is not a valid integer` in the api | A queue or autoscaling variable is not a positive integer. The API refuses to start |
| `ndif start` printed `✓`, `ndif info` says stopped | The process died after the two-second check: `ndif logs <service>` |
| `ndif start` printed `✗ <service>: exited immediately` | Usually its port is taken: the log path is on that line |

## No GPU

The ray node reports its GPU memory to the controller as `cuda_memory_bytes`,
read through torch. At 0 the node is ignored: the server boots, `/connected`
answers 200, `ndif status` shows `Nodes: 0`, and every deploy fails `No GPU
nodes available.`

```bash
docker logs ndif 2>&1 | grep "Starting Ray head node"    # ... {"head": 10, "cuda_memory_bytes": 0, ...}
docker exec ndif ndif doctor                             # the Compute section says which half is missing
```

- `nvidia-smi not found`: the container has no GPU. `--gpus all` is missing, or
  the NVIDIA Container Toolkit is not registered with Docker (then `docker run
  --gpus` fails outright with `could not select device driver`).
- `torch cannot use the GPU (driver CUDA 12.5, torch built for CUDA 13.0)`: the
  image's CUDA line is newer than the driver. Take the `-cu126` tag.

## Ray is booting, or is not

For about half a minute after start, `/connected` and `/request` answer 503
`Service temporarily unavailable: compute backend is reconnecting.` and the api
log prints `Error connecting to Ray` with `ConnectionError: Ray is not
listening on localhost:10001` once a second. The ray log should reach:

```text
Starting Ray head node with resources: {"head": 10, "cuda_memory_bytes": 50936283136, "cpu_memory_bytes": 134985248768}
Starting NDIF controller...
```

and the api log `Connected to Ray`. Still 503 after two minutes means Ray died
at start, which looks the same from outside: read the ray log from the first
error after the first line. The same 503 later on means the dispatcher lost
its Ray connection, purged every queue (each waiting request got `Critical
server error occurred.`) and is retrying; look at the ray service.

## A dead dispatcher

`/ping` depends on nothing, and the flag `/connected` reads has no expiry, so
when the dispatcher process dies the API still looks healthy.

```bash
ndif queue                               # Error: No response from the dispatcher for queue_state_request. Is the API running?
docker exec ndif redis-cli llen queue    # grows by one with every request
```

Clients sit on `RECEIVED`. `/status` keeps answering from its cache for a
minute, then returns 504 `Timed out waiting for cluster status.`; `ndif status`
still works, because the CLI asks the controller directly. Restart the api
(`just restart api`, `ndif start api --restart`, or `docker restart ndif` for
the single container). Requests still in the Redis list are served afterwards;
the ones the old dispatcher had taken are lost. A health check that catches
this watches the queue length, not `/ping`.

## A stuck request

The client prints the request id in brackets on every status line. `ndif
queue` lists the ids each model has queued and running.

| Stays on | Cause | Do |
|---|---|---|
| `RECEIVED` | The dispatcher is not taking requests | [A dead dispatcher](#a-dead-dispatcher) |
| `PROVISIONING`, `DEPLOYING` | The controller is placing the model, or the replica is downloading and loading. Nothing times it out | `ndif status`: `DEPLOYING` beside the model is a load in progress. The controller log shows `failed during create` if it is failing |
| `QUEUED`, position falling | Behind other work | `ndif queue` for depth and replicas. The autoscaler adds a replica after 30 s if the ledger has room |
| `QUEUED` at position 1 | The model's replicas are all busy | The `⚙` line of `ndif queue` shows which request and for how long; `ndif kill` it |
| `QUEUED` again after `DISPATCHED` or `RUNNING` | Its replica was evicted or demoted to WARM under it. The request went back to the front and runs again from the start | Nothing. A model flapping between HOT and WARM makes this repeat |
| `RUNNING` | The block itself. `print()` output arrives as status lines, so a console still printing is still running | `ndif kill`, or wait |

What bounds a request:

| Stage | Limit |
|---|---|
| Submitting over HTTP | `NDIF_API_TIMEOUT`, 120 s |
| Waiting in a queue, or for a replica to load | none |
| Running | none, unless `NDIF_DEFAULT_EXECUTION_TIMEOUT_SECONDS` or a per-model `execution_timeout_seconds` is set |

`ndif status --json-output` tells a healthy replica from a broken one:

```bash
ndif status --json-output | jq -c '.deployments[] | select(.model_key) | {repo_id, replica_id, deployment_level, application_state}'
```

| Level | State | Means |
|---|---|---|
| HOT | RUNNING | Healthy. A stuck request is stuck inside the block |
| HOT | DEPLOYING | Loading, or Ray is restarting the actor |
| HOT | UNHEALTHY | The controller tracks a replica whose actor is gone. `ndif evict <checkpoint> --replica <id>` clears the record |
| WARM | | Parked in RAM. It does not serve, and a request for it brings it back |

A replica that is HOT and RUNNING yet serves nothing gets `ndif restart
<checkpoint> --replica <id>`. Finding one request in the logs:
[reading logs](troubleshoot-reading-logs.md#one-request-end-to-end).

## Out of memory at load, or in a block

Same exception, opposite handling.

- **At load**: the replica never becomes ready. `ndif deploy` prints `✗ ...
  initialization failed - ... CUDA out of memory`, the controller logs
  `Deployment <key> failed during create`, and a client waiting on the model
  gets `Error starting model.` It is a sizing problem.
- **In a block**: one request gets a CUDA traceback ending in the user's line,
  and the replica keeps serving. `CUDA out of memory ... N MiB allowed` on an
  almost empty card is the replica's own cap doing its job.
- **`Could not deploy this model. CANT_ACCOMMODATE...`** is a refusal by the
  ledger before anything loaded.

Each case, and what to change:
[sizing and placement](operate-sizing-and-placement.md#reading-a-refusal).

## COMPLETED but no result

The job succeeded and the client cannot fetch it. A result over the inline cap,
and every non-blocking result, is downloaded by the client from a presigned
URL, and that URL is built from two variables that must be the same on `api`
and `ray`:

| Variable | Role |
|---|---|
| `NDIF_OBJECT_STORE_URL` | what the server uploads through (`http://rustfs:9000` on compose) |
| `NDIF_OBJECT_STORE_PUBLIC_URL` | what the client's URL is signed for. It must be an address the client reaches, with that exact host and port. Empty, it falls back to the URL above |

Reproduce it from the client machine:

```bash
curl -s localhost:8001/response/<request id> | jq -r .data    # the URL of a completed non-blocking job
curl -sI "<that URL>" | head -1                               # 200, or the failure
```

A connection error means the host or port is not reachable from there: publish
9000, tunnel it, or fix the variable. The other answers are in
[error messages](troubleshoot-error-messages.md#after-completed-the-download).

## Version mismatch

Both messages are a sentence with no traceback: the server refused the request
before the user's code existed.

- **`The model architecture on this server doesn't match the one your code was
  traced against: it has no module at '<path>'`** is nearly always a
  transformers difference. The path is the first place the two model trees
  diverge, not the layer the user reached for.
- **`Your request payload could not be read (<Error>: ...)`** is a package the
  server lacks (`ModuleNotFoundError`), an nnsight difference (an unpickling
  error on a class), or damage in transit. Re-send once before digging.

`ndif env` prints the server's nnsight, transformers and torch; compare with
the client's, or run `from nnsight import ndif; print(ndif.compare())` there.
On compose also `just nnsight`: an editable nnsight on the host may be mounted
over the image's. Setting `NDIF_MIN_NNSIGHT_VERSION` on the api turns the whole
class into a 400 at submit that names the minimum.

**Every remote `.save()` failing with `AttributeError: 'Tensor' object has no
attribute 'save'`** is a server nnsight without its compiled extension: an
unbuilt nnsight checkout mounted into the containers
([develop](develop.md#against-a-local-nnsight)).

## A change has no effect

- **Code**: the source is baked into the image. `just up` and `just restart`
  run the old build; `just ta` rebuilds. `just build` then `just up` replaces
  only the containers whose image changed and keeps Postgres.
- **Environment**: the container has to be recreated. `just up <svc>` does it;
  `just restart <svc>` does not. Confirm with `docker compose -f
  docker/docker-compose.yml exec <svc> env | grep NDIF_`.
- **A controller variable**: also needs the controller replaced:
  [sizing and placement](operate-sizing-and-placement.md#changing-a-controller-variable).

## The dashboard

| Symptom | Cause |
|---|---|
| `/api/status` answers 503 | It asks the controller directly: `NDIF_RAY_ADDRESS` is wrong in the dashboard container, or Ray is down |
| `/` returns JSON with a `hint` field | The built frontend is missing where `NDIF_DASHBOARD_FRONTEND_DIST` points |
| No login prompt | `NDIF_DASHBOARD_DEV_MODE=true` |
| Nobody can log in | Dev mode is off and `NDIF_DASHBOARD_PASSWORD_HASH` is empty |
| Schedules never run; they vanished after a rebuild | The crons run only where `cron` is installed (the container); the data directory was not a volume |

## Telemetry missing

Logs go to Loki only with `NDIF_LOKI_URL` set and metrics to InfluxDB only with
`NDIF_INFLUX_URL`, and both fail open, so "no data" is the only symptom.

- `just logs api | grep "Loki telemetry enabled"` proves the api is shipping.
- Compose sets the Loki variable on `api` and `ray` and not on `dashboard`.
- A from-source install needs the `metrics` extra for either.
- Model-actor records are `service="model"`, not `service="ray"`.
- Prometheus scrapes only Ray's own metrics, `ray:8080`.
