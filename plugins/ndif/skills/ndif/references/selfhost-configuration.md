# Configuration, ports, routes, and what persists

## How a setting reaches a process

There is no config file. Every setting is an environment variable, read once
when its process starts, and a variable missing from a process falls back to
that process's own default (usually `localhost`), never to another service's
value. So a change is made on the service that reads it, and that service is
started again:

| Route | Change a variable by |
|---|---|
| Image | removing the container and running it again with `-e NAME=value` |
| Compose | editing the service's `environment:` block, then `just up <svc>`, which recreates the container. `just restart` keeps the old environment and changes nothing |
| Source | the shell, a `./.env` in the working directory, `ndif --env-file PATH`, or `ndif start -e NAME=value`, then `ndif start <svc> --restart` |

Later wins: code default, the CLI's single-host defaults (this is where Ray's
head port 6385 comes from), `./.env` (fills gaps only; the working directory in
a container is `/app`), the process environment, `--env-file` (overrides the
environment), `-e`.

- **Model actors are configured through the `ray` service.** The controller
  hands its Redis, object-store, Loki and InfluxDB settings to every actor it
  creates. A sandbox runner gets only an allowlist (paths, locale, cache
  locations, `CUDA_VISIBLE_DEVICES`), so `HF_TOKEN` and credentials are not in a
  sandboxed block's `os.environ`.
- **The controller reads its variables once.** Changing one means recreating
  `ray`, which is an outage:
  [sizing and placement](operate-sizing-and-placement.md#changing-a-controller-variable).
- **See what is in effect**: `docker exec ndif env | grep NDIF_` for a process,
  `ndif info` for the addresses the CLI uses and whether each answers, and
  `ndif status --verbose | jq '{default_padding_factor, default_padding_bias,
  minimum_deployment_time_seconds, model_cache_percentage,
  default_model_actor_class, tp_model_actor_class}'` for the controller.

## The variables an operator sets

| Variable | Default | Set it when |
|---|---|---|
| **every service** | | |
| `NDIF_REDIS_URL` | `redis://localhost:6379` | Redis is another container or host. Every service must name the same one; left at the default in a separate `ray` container, actors cannot publish and requests look hung |
| `NDIF_ENVIRONMENT` | `dev` | You run more than one deployment: it labels every log record and metric |
| `NDIF_LOKI_URL` | unset, console only | You ship logs to Loki (`http://loki:3100/loki/api/v1/push`) |
| `NDIF_INFLUX_URL`, `NDIF_INFLUX_TOKEN` | unset, no metrics | You record request metrics in InfluxDB 2 (`NDIF_INFLUX_ORG` `ndif`, `NDIF_INFLUX_BUCKET` `metrics`) |
| `NDIF_SERVICE` | `all` in the image | A container runs one role, or `all dashboard`. It is also the `service` label in telemetry |
| **api** | | |
| `NDIF_POSTGRES_URL` | unset | The auth switch: unset means no API keys and every request trusted |
| `NDIF_RAY_ADDRESS` | `ray://localhost:10001` | The Ray head is elsewhere. Also read by the dashboard and by the CLI |
| `NDIF_MIN_NNSIGHT_VERSION`, `NDIF_MIN_PYTHON_VERSION` | unset | Old clients should get a clear 400 at submit instead of failing in deserialization |
| `NDIF_AUTOSCALING_MAX_REPLICAS` | `3` | A busy model may or may not grow by itself (`_WAIT_THRESHOLD_S` 30, `_BACKOFF_S` 120). A non-integer or non-positive value stops the API at import |
| `NDIF_API_PORT`, `NDIF_API_WORKERS`, `NDIF_API_TIMEOUT` | `8001`, `1`, `120` | Workers scale ingress only; there is always one dispatcher |
| **ray** | | |
| `NDIF_DEPLOYMENTS` | unset | Models must be loaded, pinned, at start. `\|`-separated model keys, not repo ids ([operate](operate.md#loading-at-boot)) |
| `NDIF_MODEL_IMPORT_PATH` | the in-process `ModelActor` | Anyone else can submit: the sandbox actor class. Compose sets the older name `NDIF_DEFAULT_MODEL_ACTOR_CLASS`, read as a fallback |
| `NDIF_TP_MODEL_ACTOR_CLASS` | unset, tensor parallelism off | Multi-GPU models should shard within layers ([operate](operate.md#tensor-parallelism)) |
| `NDIF_DEFAULT_EXECUTION_TIMEOUT_SECONDS` | unset, no cap | The server is shared: a runaway block otherwise holds its replica for ever |
| `NDIF_DEFAULT_PADDING_FACTOR`, `NDIF_DEFAULT_PADDING_BIAS`, `NDIF_DEFAULT_DTYPE` | `0.15`, `524288000`, `bfloat16` | The sizing defaults do not suit your models |
| `NDIF_MINIMUM_DEPLOYMENT_TIME_SECONDS` | `3600` | A newly deployed model should be evictable sooner or later than an hour |
| `NDIF_MODEL_CACHE_PERCENTAGE` | `0.7` | Share of host RAM that WARM models may hold. Not a GPU setting |
| `NDIF_SANDBOX_POOL_SIZE` | `7` | Warm runner processes per sandboxed replica, about 0.5 GB of RAM each. Fewer saves RAM and makes a busy queue pay a ~4 s spawn |
| `NDIF_RAY_HEAD_ADDRESS` | unset, this node is the head | The node joins another head as a worker (`HEAD:6385`) |
| `NDIF_RAY_TEMP_DIR` | `/tmp/ray` | That filesystem can pass 95 % full. Keep the path short |
| `HF_TOKEN`, `HF_HOME` | unset, `/root/.cache/huggingface` in the image | Gated checkpoints; a cache somewhere else |
| **api and ray, identical on both** | | |
| `NDIF_OBJECT_STORE_URL` | `http://localhost:9000` | What the server uploads through. Empty means AWS S3 |
| `NDIF_OBJECT_STORE_PUBLIC_URL` | empty, falls back to the URL above | What result URLs are signed for: the address clients fetch from |
| `NDIF_OBJECT_STORE_ACCESS_KEY`, `NDIF_OBJECT_STORE_SECRET_KEY` | `rustfsadmin` | Real credentials, or both empty so boto3 uses its own chain (an instance role) |
| `NDIF_OBJECT_STORE_BUCKET`, `_REGION`, `_VERIFY` | `ndif-results`, `us-east-1`, `true` | Your own store |

The dashboard's `NDIF_DASHBOARD_*` variables are in
[production and auth](operate-production-and-auth.md#the-dashboard-login). The
rest (Ray's internal ports, queue batch sizes, cache lifetimes, the Postgres
pool, InfluxDB batching, the Loki buffer, cron schedules) is
`docs/reference/env-vars.md`, which lists every variable with the code that
reads it.

## Ports

| Port | Listener | Who should reach it |
|---|---|---|
| 8001 | api: HTTP and the status websocket | Clients, behind your TLS proxy |
| 9000 | object store, S3 API | The server, and clients downloading results |
| 8081 | admin dashboard | Operators only |
| 8265 | Ray dashboard: actors, nodes, per-actor logs | Operators only |
| 10001 | Ray client server (`ray://`) | The api, the dashboard and the CLI. It has no authentication: reaching it is running code on the cluster |
| 6385, 8076, 52366 | Ray head, object manager, dashboard agent | Other Ray nodes only. NDIF moves Ray's head port off 6379 so Ray and Redis can share a host |
| 8080 | Ray's Prometheus metrics | Prometheus |
| 6379 | Redis | Every service, nothing else. No authentication |
| 9001, 5432, 3100, 8086, 9090, 3000 | RustFS console, Postgres, Loki, InfluxDB, Prometheus, Grafana (compose) | Operators only |

`docs/reference/ports.md` has the variable that moves each one.

## HTTP routes of the API

| Route | Does |
|---|---|
| `POST /request` | Submit. Multipart: `data`, the JSON envelope with the `model_key`, and `blob`, the serialized block. The key goes in the `ndif-api-key` header. Answers `RECEIVED` with the request id; 503 while the dispatcher has no Ray connection |
| `WS /subscribe` | A blocking client opens it first. The server sends `{"session_id": ...}`, the client puts that id in its request, and every later status and an inline result arrive here |
| `GET /response/{id}` | The latest stored status of a non-blocking request; its `data` is the result URL once `COMPLETED`. 404 until the first status is stored. Needs no key |
| `GET /status` | Deployments and cluster, the payload of `ndif status --json-output`, cached 60 s |
| `GET /env` | The server's Python and package versions, what `ndif env` prints, cached 300 s |
| `GET /ping` | `"pong"`. Depends on nothing |
| `GET /connected` | 200 `{"status":"connected"}`, or 503 while the dispatcher has no Ray connection |
| `GET /whoami` | `{"email", "tags"}` of the key in `ndif-api-key`; nulls for no key or an unknown one |

Request and response schemas: `docs/reference/http-api.md`.

## How a result comes back

A blocking request whose compressed result is at or under the inline cap gets
it on the `COMPLETED` message. Anything larger, and every non-blocking result,
is uploaded as `<request id>.pt` and the client receives a presigned URL that
lasts one hour.

- **The cap comes from Redis**: `min(soft, hard // 2)` of its
  `client-output-buffer-limit pubsub`, which is 8 MiB on stock limits
  (`redis-cli config get client-output-buffer-limit`). NDIF has no variable for
  it; raise Redis's limits to raise it.
- **A non-blocking request also stores each status** as
  `responses/<request id>.json`, which is what `GET /response/{id}` reads.
- **Nothing deletes result blobs or stored responses.** Retention is yours.

## What survives a restart

| State | Lives in | Survives |
|---|---|---|
| Downloaded weights | the Hugging Face cache | Yes, when the cache is a bind mount or a volume |
| The deployed set: what is HOT, WARM, pinned | the controller's memory | Not a restart of `ray`. `NDIF_DEPLOYMENTS` and the dashboard's schedule are loaded again; nothing else is |
| Requests still in the Redis list | Redis | An API restart, yes |
| Requests the dispatcher has taken (queued for a model, or running) | the dispatcher's memory | No. An API restart drops them: a blocking client's websocket closes with no `ERROR`, and a block already running still finishes on the actor with nowhere to send its result |
| Status updates of a blocking request | Redis pub/sub | Never stored |
| Result blobs, non-blocking responses | the object store | Yes. Compose keeps them in the `rustfs_data` volume; the single container keeps them until `docker rm` |
| API keys | Postgres | On compose, only until `just down`: its volume is anonymous |
| Dashboard schedule and monitor history | the dashboard's data directory | Yes on compose, in the `dashboard_data` volume |
| Logs and metrics (compose) | Loki, InfluxDB, Prometheus | No volumes: gone on `just down` |
