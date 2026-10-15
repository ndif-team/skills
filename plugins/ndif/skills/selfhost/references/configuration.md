# Configuration and what persists

The exhaustive tables live in the ndif repo: `docs/reference/env-vars.md` (every
variable and the code that reads it), `docs/reference/ports.md` (every port and
whether it may be public) and `docs/operating/configuration.md` (the layering
model). This page is what a self-hoster needs in front of them.

## The model

**There is no config file.** Every knob is an `NDIF_*` environment variable,
nearly all read once at process start. A missing variable falls back to that
process's hardcoded default — almost always `localhost` — never to another
service's value; most cross-service breakage is a `localhost` default nobody
overrode.

Precedence, lowest to highest (`docs/operating/configuration.md#precedence`):

1. code defaults;
2. the CLI's `DEFAULTS`, merged *underneath* the real environment (this is where
   `NDIF_RAY_HEAD_PORT=6385` comes from);
3. `./.env` in the working directory, which only fills gaps;
4. the shell environment (under compose, the service's `environment:` block);
5. `ndif --env-file PATH`, loaded with override, so it beats the shell;
6. `ndif start -e KEY=VALUE` and the typed shortcuts (`--redis-url`,
   `--ray-address`, `--ray-head-address`, `--api-port`).

Containers run `ndif start --foreground`, so these layers apply inside compose
too. The working directory in every NDIF container is `/app`, so configuring one
through `.env` means bind-mounting it at `/app/.env`. Under compose an
environment edit takes `just up <svc>`; `just restart` keeps the old environment.

**You configure model actors by configuring the `ray` service.** The controller
exports its Redis, object-store, Loki and Influx settings into each actor's Ray
`runtime_env`. The sandbox runner is the opposite: it starts from an allowlist,
so `HF_TOKEN`, `NDIF_POSTGRES_URL` and the object-store credentials are not
inherited by a sandboxed block (an environment allowlist, not a jail: the runner
still shares the actor's user, filesystem, network and GPUs).

## What must change to leave single-host

| Variable | Dev default | Why it must change |
|---|---|---|
| `NDIF_REDIS_URL` | `redis://localhost:6379` | Must name the real Redis from *every* service. Inside the `ray` container nothing listens on `localhost:6379` — Redis is another container, and Ray's head port is 6385. |
| `NDIF_RAY_ADDRESS` | `ray://localhost:10001` | The `ray://` client address of the head, used by the API, dashboard and CLI. |
| `NDIF_RAY_HEAD_ADDRESS` | unset | Empty ⇒ this node starts a head. Every worker sets it to `HEAD:6385`. |
| `NDIF_OBJECT_STORE_URL` | `http://localhost:9000` | The endpoint the *server* uploads through. Empty means real AWS S3. |
| `NDIF_OBJECT_STORE_PUBLIC_URL` | empty → falls back to the above | The endpoint presigned URLs are **signed with**; must be the address clients reach, host and port exactly. |
| `NDIF_OBJECT_STORE_ACCESS_KEY` / `_SECRET_KEY` | `minioadmin` / `minioadmin` | Real credentials — or both empty so boto3 uses its own chain. |
| `NDIF_POSTGRES_URL` | empty | Empty ⇒ no auth ⇒ every request trusted. |
| `NDIF_MODEL_IMPORT_PATH` | in-process `ModelActor` in code; compose sets the legacy `NDIF_DEFAULT_MODEL_ACTOR_CLASS` to `SandboxModelActor` | Only a sandbox actor class runs untrusted code in a separate runner; auth alone does not. The image and a from-source install set neither. |
| `NDIF_TP_MODEL_ACTOR_CLASS` | unset (tensor parallelism off) | If you turn TP on for untrusted traffic, use `ndif.services.ray.tp.model.SandboxedTPModelActor` — a TP placement swaps the actor class, and `TPModelActor` has no runner. |
| `NDIF_API_URL` | `http://localhost:8001` | How other components reach the API (compose sets `http://api:8001`). |
| `NDIF_DASHBOARD_DEV_MODE` | `false` in code, **`true` in compose** | `true` disables the dashboard login entirely. |
| `NDIF_DASHBOARD_SESSION_SECRET` | `change-me-please-this-is-not-secure` | Anyone with the default can forge a session cookie. |
| `NDIF_DEFAULT_EXECUTION_TIMEOUT_SECONDS` | unset — no cap | A runaway block otherwise holds its replica indefinitely. |

The actor class is the row that bites: a bare `pip install` plus `ndif start`
does **not** run the same execution path as `just up`. The sandboxed path is
process separation, not a hardened jail — the runner shares the actor's user,
filesystem, network and GPUs — and what it guarantees is that untrusted
requests cannot read or edit weights or train.
`docs/operating/production.md#the-sandboxed-actor-class`.

**If 9000 is remapped** on the host (a shared box), set
`NDIF_OBJECT_STORE_PUBLIC_URL` to the new host port on **both** `api` and `ray`;
each signs presigned URLs.

## What persists

| State | Survives a restart? |
|---|---|
| Downloaded model weights | Yes, if the HF cache is a bind mount or named volume |
| Queued requests still in the Redis list | Yes |
| Requests the dispatcher has popped (queued in its memory, or in flight) | **No.** An API restart drops them silently — no `ERROR` reaches the client. Requests still in the Redis list are unaffected |
| Result blobs (`{request_id}.pt`) | Yes, in the object store — nothing deletes them |
| Non-blocking responses (`responses/{id}.json`) | Yes, latest only |
| Live status updates | Not stored — Redis pub/sub is fire-and-forget |
| Deployment state | Controller actor memory; rebuilt from the cluster, **not** from what was pinned before |
| Dashboard schedule and monitor history | Yes, on the `dashboard_data` volume |
| Metrics, logs, Postgres data (dev compose) | **No** — none of them declares a volume |

Presigned result URLs expire after **one hour**, a real constraint for a
non-blocking job polled much later.
