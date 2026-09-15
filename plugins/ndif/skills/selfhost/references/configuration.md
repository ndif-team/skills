# Configuration, ports, and volumes

The exhaustive tables are `docs/reference/env-vars.md` (every variable with the
line that reads it), `docs/reference/ports.md`, and `docs/operating/configuration.md`
(the layering model). This page is what a self-hoster needs in front of them.

## The model

**There is no config file** — no YAML, no TOML, no `settings.py`. Every knob is
an `NDIF_*` environment variable, nearly all read once at process start. Three
consequences:

1. Changing a value means restarting the process that reads it; for anything
   baked into a compose image, `just ta`.
2. A missing variable never falls back to another service's value — it falls back
   to that process's hardcoded default, which is almost always `localhost`. Most
   cross-service breakage in this stack is a `localhost` default nobody overrode.
3. Every process is self-describing: whatever is in its environment is its entire
   configuration.

Precedence, later winning: code defaults → the CLI's `DEFAULTS` (overlaid
*beneath* the real environment) → `.env` files (a CWD `./.env` fills gaps only;
`ndif --env-file path` is loaded with `override=True` and beats the shell) → the
process environment → `ndif start -e KEY=VALUE` and the typed shortcuts
(`--redis-url`, `--ray-address`, `--ray-head-address`, `--api-port`).

Containers run `ndif start --foreground`, so the CLI layer applies inside compose
too; a compose `environment:` block is the process environment as far as the CLI
is concerned. Note that the auto-discovered `.env` is relative to the working
directory, which inside every NDIF container is `/app` — configuring a container
through `.env` means bind-mounting it to `/app/.env`.

Config crosses exactly one boundary by itself: when the controller creates a
model actor it exports its own Redis, object-store, Loki and Influx settings into
the actor's Ray `runtime_env` (plus `NDIF_SERVICE=model`). **You configure model
actors by configuring the `ray` service.**

## What must change to leave single-host

| Variable | Dev default | Why it must change |
|---|---|---|
| `NDIF_REDIS_URL` | `redis://localhost:6379` | Must name the real Redis from *every* service. Inside the ray container, `localhost:6379` is Ray's own GCS. |
| `NDIF_RAY_ADDRESS` | `ray://localhost:10001` | The `ray://` client address of the head, used by the API, dashboard and CLI. Unrelated to head-vs-worker. |
| `NDIF_RAY_HEAD_ADDRESS` | unset | Empty ⇒ this node starts a head. Every worker sets it to `HEAD:6385`. |
| `NDIF_OBJECT_STORE_URL` | `http://localhost:9000` | The endpoint the *server* uploads through. Empty means real AWS S3 for `NDIF_OBJECT_STORE_REGION`. |
| `NDIF_OBJECT_STORE_PUBLIC_URL` | empty → falls back to the above | The endpoint presigned URLs are **signed with**. Must be reachable by clients. |
| `NDIF_OBJECT_STORE_ACCESS_KEY` / `_SECRET_KEY` | `minioadmin` / `minioadmin` | Real credentials — or set **both** empty so boto3 uses its own chain (on AWS, the instance/task role). |
| `NDIF_POSTGRES_URL` | empty | Empty ⇒ no auth ⇒ every request trusted. |
| `NDIF_API_URL` | `http://localhost:8001` | How other components reach the API (compose sets `http://api:8001`). |
| `NDIF_DASHBOARD_DEV_MODE` | `false` in code, **`true` in compose** | `true` disables the dashboard login entirely. |
| `NDIF_DASHBOARD_SESSION_SECRET` | `change-me-please-this-is-not-secure` | Anyone with the default can forge a session cookie. |
| `NDIF_ENVIRONMENT` | `dev` | The prod/staging/dev label on every log line and metric. |
| `NDIF_DEFAULT_MODEL_ACTOR_CLASS` | in-process `ModelActor` in code, `SandboxModelActor` in compose | Whether untrusted code gets a separate runner process at all. Resolution order is `NDIF_MODEL_IMPORT_PATH` → `NDIF_DEFAULT_MODEL_ACTOR_CLASS` → the base actor. |

That last row is worth dwelling on: a bare `pip install` plus `ndif start ray`
does **not** run the same execution path as `just up`.

## Sizing and limits

| Variable | Default | Effect |
|---|---|---|
| `NDIF_DEFAULT_PADDING_FACTOR` | `0.15` | Proportional headroom over the weight estimate. Also caps per-process GPU memory inside the actor, so it is the lever for `CUDA out of memory ... N MiB allowed` inside a block. |
| `NDIF_DEFAULT_PADDING_BIAS` | `524288000` (500 MiB) | Flat headroom on every estimate — covers roughly one CUDA context. |
| `NDIF_DEFAULT_DTYPE` | `bfloat16` | Dtype a model loads in when the deployment does not name one. Pinned before sizing so the estimate and the load agree. |
| `NDIF_MODEL_CACHE_PERCENTAGE` | `0.9` | Fraction of the node's **host RAM** usable as WARM cache. Not a GPU knob. |
| `NDIF_MINIMUM_DEPLOYMENT_TIME_SECONDS` | `3600` | How long a fresh replica is protected from automatic eviction. |
| `NDIF_DEFAULT_EXECUTION_TIMEOUT_SECONDS` | **unset — no cap** | A block runs until it finishes and holds its replica. Set it before other people can submit. |
| `NDIF_AUTOSCALING_MAX_REPLICAS` | `3` | Per-model replica ceiling for autoscaling, which never scales back down. |
| `NDIF_MAX_SOCKET_RESULT_BYTES` | `20971520` (20 MiB) | Largest result (after compression) handed back on the COMPLETED response instead of through the object store. `0` removes the cap — don't, unless results are known to be small: past Redis's pubsub output-buffer limit the subscriber is disconnected and the response is silently lost. |
| `NDIF_SANDBOX_POOL_SIZE` | `7` | Runners pre-warmed per **sandboxed** model actor, each holding a few hundred MB whether or not anything is running. Turn it down on a node hosting several models. |
| `NDIF_TP_MODEL_ACTOR_CLASS` | **unset — tensor parallelism off** | Not a fallback to a built-in: unset means no replica is ever placed tensor-parallel and per-model `max_tp` is inert. |
| `NDIF_RAY_TEMP_DIR` | `/tmp/ray` | Ray refuses to schedule when the filesystem holding it is >95% full. |

## Fail-open switches

An empty value is never an error; it silently disables the feature.

| Empty | Effect |
|---|---|
| `NDIF_LOKI_URL` | No log shipping. The console handler is always installed, so `docker logs` / `just logs` still work and the package is never imported. |
| `NDIF_INFLUX_*` | Metrics are dropped. Nothing blocks, nothing raises. |
| `NDIF_MIN_NNSIGHT_VERSION` / `NDIF_MIN_PYTHON_VERSION` | No client version gating at all — an ancient client gets through and fails later, deep in deserialization, with a message about pickles rather than versions. An empty string counts as unset. |

`NDIF_POSTGRES_URL` is deliberately shaped the opposite way: set it without
`asyncpg` installed and `connect` raises loudly rather than quietly running
without auth.

## Ports

| Port | What | Public in a real deployment? |
|---|---|---|
| 8001 | API — the only port a client posts to | **yes**, behind TLS |
| 9000 | MinIO S3 API — clients fetch presigned result URLs here | **yes** (whatever `NDIF_OBJECT_STORE_PUBLIC_URL` names) |
| 9001 | MinIO console | no |
| 8081 | Admin dashboard — a control plane, not a viewer | no |
| 8265 | Ray dashboard | no |
| 10001 | Ray client (`ray://`) | **no** — `ray://` has no authentication; anyone who can route to it runs code on the cluster |
| 6385 | Ray GCS — the worker join address | cluster-internal, open between nodes |
| 8076 | Ray object manager (plasma transfer between nodes) | cluster-internal |
| 52366 | Ray dashboard agent gRPC | cluster-internal |
| 8080 | Ray metrics export (Prometheus scrape target) | no |
| 6379 / 5432 / 3100 / 8086 / 9090 / 3000 | redis / postgres / loki / influx / prometheus / grafana | no |

The image `EXPOSE`s `8001 9000 9001 8081 8265 10001 6379` — the union of every
`NDIF_SERVICE` mode. `EXPOSE` is documentation; `docker run` still needs `-p`.

The sandbox has no port: a model actor talks to its runner over a Unix domain
socket at `/tmp/sbx-<hex>.sock`.

## What persists

| State | Survives a restart? |
|---|---|
| Downloaded model weights | Yes, if the HF cache is a bind mount / named volume |
| Queued requests (still in the Redis list) | Yes — until popped |
| In-flight requests | **No.** Once the dispatcher pops a request it is a Python object in that process; an API restart drops every queued and in-flight job silently, with no ERROR to the client |
| Result blobs (`{request_id}.pt`) | Yes, in the object store — nothing deletes them |
| Non-blocking responses (`responses/{id}.json`) | Yes, latest only |
| Live status updates | Not stored at all — Redis pub/sub is fire-and-forget |
| Deployment state | Controller actor memory; rebuilt from the cluster, **not** from what was pinned before |
| Dashboard schedule and monitor history | Yes, on the `dashboard_data` volume |
| Metrics, logs, Postgres data (dev compose) | **No** — none of them declares a volume |

Presigned result URLs expire after **one hour**, which is a real constraint for a
non-blocking job polled much later.
