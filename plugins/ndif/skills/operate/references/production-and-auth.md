# Auth, and what production needs

Source of record: `docs/runbooks/enable-auth.md`, `docs/operating/production.md`,
`docs/concepts/auth-and-limits.md`, `docs/concepts/sandbox-execution.md`.

## Two switches, not one

Where a request's code runs is decided by **two** things, and both must be set
before anyone else can reach the API:

1. **Auth decides whether the request is trusted.** `NDIF_POSTGRES_URL` on the
   `api` service turns API keys on; a key is trusted iff it carries the `trusted`
   user_tag.
2. **The actor class decides whether "untrusted" changes anything.** Only a
   sandbox actor class forks an untrusted request into a runner process. The
   base `ModelActor` runs every request in-process, next to the weights,
   regardless of `trusted`.

| Actor class | An untrusted request runs |
|---|---|
| `ModelActor` (the default in code) | in-process — no fork |
| `SandboxModelActor` | in a runner process |
| `TPModelActor` | in-process on every rank — no fork |
| `SandboxedTPModelActor` | in one runner the whole TP group hosts |

The controller reads `NDIF_MODEL_IMPORT_PATH`, falls back to the legacy
`NDIF_DEFAULT_MODEL_ACTOR_CLASS`, then to the base `ModelActor`. **Only the
compose file sets it** (to `ndif.services.ray.sandbox.model.SandboxModelActor`,
on `ray`). The published image and a from-source install run untrusted code
in-process **even with auth on** until you set, on the `ray` service:

| Variable | Value |
|---|---|
| `NDIF_MODEL_IMPORT_PATH` | `ndif.services.ray.sandbox.model.SandboxModelActor` |
| `NDIF_TP_MODEL_ACTOR_CLASS` (only if TP is on) | `ndif.services.ray.tp.model.SandboxedTPModelActor` |

A per-deployment `actor_class` (`--actor-class`, a `models.yaml` entry, the
dashboard's deploy modal) overrides this; never point one at the base
`ModelActor` on a cluster that takes untrusted traffic. Check every replica's
`actor_class` in `ndif status --json-output`.

## Why `NDIF_POSTGRES_URL` matters

<!-- test: skip -->
```python
client_set_trusted = "trusted" in request.model_fields_set
...
identity = await verify_api_key(request.api_key)
if identity is not None:
    request.email = identity.email
    request.trusted = identity.trusted
    request.priority = identity.priority
elif not client_set_trusted:
    # Auth is off: a trusted-network / dev mode. Default to trusted.
    request.trusted = True
```

With no Postgres, `verify_api_key` returns `None`, so a request that did not tag
itself defaults to `trusted=True`:

1. **User code runs in-process, in the model actor** — whatever the actor class.
2. **A model it auto-deploys loads with `trust_remote_code=True`**, executing
   whatever Python its Hugging Face repo ships.

An explicit `trusted` in the envelope is honored when auth is off, so a client
can send `trusted: false` to exercise the runner path (on a sandbox actor class)
with no Postgres at all — see the `develop` skill. With auth on, a
client-supplied value is overwritten from the key.

## Turning it on

1. **Bring up Postgres.** `docker/postgres/init.sql` is mounted into
   `/docker-entrypoint-initdb.d/` and runs **once, on an empty data directory**:
   nine tables, the read-only `ndifapi` role the API connects as, and the
   `login_page` role the account portal uses. It seeds no users, keys or tags.

2. **Know where the keys live.** The compose `postgres` service mounts no volume,
   so its data is in an **anonymous** Docker volume:

   | Command | Keys |
   |---|---|
   | `just restart`, `just up` (including a recreate after a config change) | kept — the anonymous volume reattaches |
   | `just down` | **lost** — the next `up` gets a fresh volume and `init.sql` re-runs |
   | `just ta` | **lost** — it runs `just down` first |

   After a loss every existing key gets `403 Invalid API key.` until its rows are
   re-inserted. Before issuing keys you care about, give Postgres a named volume
   (`pgdata:/var/lib/postgresql/data`), dump/restore around every teardown
   (`pg_dump -U admin -d ndif --data-only`), or rebuild with `just build` +
   `just up` instead of `just ta`.

3. **The one query.** The API runs exactly one:

   ```sql
   SELECT k.user_id, u.email, ut.name AS user_tag
   FROM keys k
   LEFT JOIN users u ON u.user_id = k.user_id
   LEFT JOIN key_user_tag_assignments kuta ON kuta.key_id = k.key_id
   LEFT JOIN user_tags ut ON ut.user_tag_id = kuta.user_tag_id
   WHERE k.key_id = $1
   ```

   **Validity is "a row exists in `keys`"** — issuance is the account portal's
   job, a separate repo. The API key *is* the `key_id` UUID. The `LEFT JOIN`s mean
   a key with no tags still returns one row.

4. **Create the two tags the server acts on.** Every other tag name is inert.

   | Tag | Effect |
   |---|---|
   | `trusted` | the request runs in-process instead of in a runner, and any model it triggers a deploy for loads with `trust_remote_code` |
   | `priority` | the request sorts ahead of all normal traffic for that model, FIFO against other priority requests |

   `priority` has **no aging**: sustained priority load starves normal traffic,
   with autoscaling (max 3 replicas) the only relief.

   ```sql
   INSERT INTO user_tags (name, description)
   VALUES ('trusted',  'code runs in the model process'),
          ('priority', 'served ahead of untagged traffic');
   ```

5. **Point the API at it** — uncomment `NDIF_POSTGRES_URL` on the `api` service,
   then **recreate** with `just up api`. `just restart api` keeps the old
   environment, so auth silently stays off; `just ta api` tears the whole stack
   down (and the keys with it). Confirm with
   `docker compose -f docker/docker-compose.yml exec api env | grep NDIF_POSTGRES_URL`.
   The pool is created lazily, so `Postgres connected` appears on the first keyed
   request. With the URL set and `asyncpg` missing, every keyed request gets
   **503** `Auth backend unavailable.` (`API-key verification failed: Postgres
   unavailable` in the log) — the image installs the `postgres` extra, a
   hand-rolled install needs `pip install '.[postgres]'`.

6. **Create a key.**

   ```sql
   INSERT INTO users (email) VALUES ('researcher@example.edu') RETURNING user_id;
   INSERT INTO keys (user_id) VALUES ('<user_id>') RETURNING key_id;  -- the API key
   INSERT INTO key_user_tag_assignments (key_id, user_tag_id)
   SELECT '<key_id>', user_tag_id FROM user_tags WHERE name = 'trusted';
   ```

7. **Verify.**

   | Condition | Status on `/request` |
   |---|---|
   | no `ndif-api-key` header | 401 |
   | header present but not a UUID | 400 |
   | well-formed UUID, no row in `keys` | 403 |
   | Postgres unreachable or erroring | 503 — auth fails **closed** |

   ```bash
   curl -s -H "ndif-api-key: $NDIF_API_KEY" localhost:8001/whoami
   # {"email": "researcher@example.edu", "tags": ["priority"]}
   ```

   `/whoami` returns `{"email": null, "tags": []}` for a missing or unknown key,
   **400** for a malformed one, and 503 on a Postgres failure. A 503 from every
   `/request` probe means `require_ray_connection` tripped first and auth was
   never reached; a 200 from every probe means auth is off.

## What an untagged key can do

On a sandbox actor class with auth on, an untagged key's request runs in a runner
that holds a **weightless (meta) copy** of the model and drives the real one over
a Unix socket. Reading and editing activations, `.save()`, `.source`, module
calls, caches and generation all work. These do not:

- **Reading weights** — `module.weight` is a meta tensor with a shape and no
  data. Saving anything derived from one fails at result upload with
  `Cannot copy out of meta tensor`.
- **Editing weights** — a write lands on the meta copy and is a silent no-op.
- **Gradients and `backward()`** — `.grad` fails with `OutOfOrderError`.
- **Training** — an optimizer stepping `model.parameters()` updates meta tensors
  and changes nothing.

Tell users before you flip it; anyone who needs those needs a `trusted` key.

## Verifying which path a request took

**No log record carries `trusted`**, and there is no log field that shows the
fork. The only check is the runner processes. Runners are children of the model
actor (`ndif.services.ray.sandbox.runner` on their command line), **kept warm in
a pool**: each untrusted request takes one, which is stopped afterwards and
replaced with a new pid. So a runner's presence proves nothing — compare pids
**before and after** a request:

```bash
docker compose -f docker/docker-compose.yml exec ray sh -c \
  'for p in /proc/[0-9]*; do echo "$(cut -d" " -f4 $p/stat) ${p#/proc/} $(tr "\0" " " < $p/cmdline)"; done' \
  | grep -i -e ModelActor -e runner
```

(The image has no `ps`.) The first column is the parent pid. New runner pids after
the request mean it took the untrusted path; a trusted request leaves the runners
untouched. On a base `ModelActor` there are no runners at all.

## Gotchas around auth

- **Auth is checked at ingress only.** Anyone who can reach Redis, the Ray client
  port (10001) or the Ray dashboard (8265) bypasses it completely.
- **Models deployed while auth was off keep their `trusted` deployment.** The flag
  is fixed on the `Deployment` at creation. Evict and redeploy after turning auth
  on if a model should stop loading with `trust_remote_code`.
- **The dashboard's auth is entirely separate.** Remove `NDIF_DASHBOARD_DEV_MODE`,
  then set `NDIF_DASHBOARD_USERNAME`, a bcrypt `NDIF_DASHBOARD_PASSWORD_HASH`
  (`python -m ndif.services.dashboard.backend.auth hash '<password>'`, each `$`
  written `$$` in compose YAML) and a random `NDIF_DASHBOARD_SESSION_SECRET` — the
  default is the public string `change-me-please-this-is-not-secure`. With dev mode
  off and no hash, **nobody can log in**. Recreate with `just up dashboard`. Its
  deploy endpoint defaults to `trusted: true`, so an open dashboard is remote code
  execution.
- **Enabling auth is also what makes per-user attribution work** — `email` rides
  with the request into every log record and metric.

## The sandbox, stated honestly

Where it applies (untrusted request, sandbox actor class), user code runs in a
**separate process** from the model actor, fresh per request, started with an
allowlisted environment (`RUNNER_ENV`) — so a block cannot read `HF_TOKEN`,
`NDIF_INFLUX_TOKEN`, `NDIF_POSTGRES_URL` or object-store credentials out of
`os.environ`.

That is one hole closed, not a boundary. The runner is an ordinary child process:
same user, filesystem, network, and visible GPUs, with no namespaces, seccomp,
rlimits, or filesystem jail, and outside the actor's per-process GPU cap. Size
your threat model accordingly.

## What NDIF does not provide

- **No TLS.** Gunicorn binds plain HTTP on `0.0.0.0:8001`; API keys travel in the
  `ndif-api-key` header. Terminate TLS yourself and never expose 8001 directly.
- **No key on result retrieval.** `GET /response/{id}` needs no API key; the
  request id is the only protection.
- **No rate limiting or quotas.** One key can submit unlimited requests.
- **No multi-tenancy.** All users share the same deployments and GPUs.
- **No secret management.** Credentials are environment variables.
- **No authentication on the backing services.** The dev compose publishes Redis,
  Postgres, MinIO, InfluxDB, Loki, Prometheus, Grafana, the Ray dashboard and the
  Ray client port on every interface.
- **No test CI and no compatibility guarantees.** The workflows build and publish;
  none of them runs tests.
- **No horizontal API tier.** The dispatcher is started exactly once by the
  gunicorn master; two API containers against one Redis would run two
  dispatchers.

## Pre-flight checklist

| # | Check | Done when |
|---|---|---|
| 1 | `NDIF_POSTGRES_URL` set on `api` (recreated with `just up api`), keys created, `trusted` granted deliberately | `/whoami` with a key returns the email; without one, `/request` 401s |
| 2 | Actor class is a sandbox class: `NDIF_MODEL_IMPORT_PATH` = `ndif.services.ray.sandbox.model.SandboxModelActor`, and `NDIF_TP_MODEL_ACTOR_CLASS` = `ndif.services.ray.tp.model.SandboxedTPModelActor` if TP is on | `ndif status --json-output` shows a sandbox `actor_class` on every replica; runner pids change across an untagged request |
| 3 | Users told that untagged keys lose weight reads, weight edits, gradients and training | — |
| 4 | Postgres on a durable volume, password not `admin`, not published | keys survive `just down` |
| 5 | `NDIF_DASHBOARD_DEV_MODE` removed; username, bcrypt hash, random session secret set | the dashboard prompts for a login |
| 6 | Object-store credentials replaced; bucket durable | — |
| 7 | `NDIF_OBJECT_STORE_PUBLIC_URL` resolves and fetches **from a client machine** | a non-blocking result downloads from off-host |
| 8 | `NDIF_REDIS_URL` explicitly set on **every** service, never left at localhost | `ndif info` in each container shows the right host |
| 9 | Head/worker split correct: `NDIF_RAY_HEAD_ADDRESS` on workers only | `ndif status` lists every GPU node |
| 10 | 6385 / 8076 / 52366 open between nodes only; 6379, 5432, 10001, 8265, 9001 and the telemetry ports never public | a scan from outside shows only your TLS endpoint (and the object store, if public) |
| 11 | TLS-terminating proxy in front of 8001; 8001 not directly exposed | `https://` endpoint answers `/ping` |
| 12 | Persistent volumes for the HF cache, MinIO, Postgres, Loki, Influx, dashboard data | survives `just down` |
| 13 | `HF_TOKEN` set on `ray` for gated checkpoints | the gated model deploys |
| 14 | `NDIF_ENVIRONMENT` set to something other than `dev` | Grafana can separate this deployment |
| 15 | `NDIF_DEFAULT_EXECUTION_TIMEOUT_SECONDS` set — **unset means no cap at all** | a runaway block stops holding its replica forever |
| 16 | `NDIF_MODEL_CACHE_PERCENTAGE`, `NDIF_AUTOSCALING_MAX_REPLICAS` and padding factors reviewed | — |
| 17 | `NDIF_MIN_NNSIGHT_VERSION` set to the oldest client you will support | an older client gets a clear 400 |
| 18 | A retention policy for result blobs and `responses/*.json` — NDIF deletes neither | disk does not fill unbounded |

Every environment change on compose takes `just up <svc>`; `just restart` keeps
the old environment. Actual deployment specifics are out of scope for NDIF: bring
your own TLS termination, orchestration, secret store and infrastructure. There
is no production compose file.
