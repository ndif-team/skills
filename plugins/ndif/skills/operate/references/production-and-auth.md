# Auth, and what production needs

Source of record: `docs/runbooks/enable-auth.md`, `docs/operating/production.md`,
`docs/concepts/auth-and-limits.md`.

## Why `NDIF_POSTGRES_URL` is the variable that matters

It does not only gate *who* may submit. It decides *how their code runs* and *how
models are loaded*, through one boolean:

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
itself defaults to `trusted=True`. Follow that flag:

1. **User code runs in-process, in the model actor** — the same process that
   holds the weights. No runner subprocess, no socket, none of the isolation the
   sandbox path provides.
2. **The model loads with `trust_remote_code=True`.** A model auto-deployed by the
   first request on an unauthenticated server executes whatever Python its Hugging
   Face repo ships.

An explicit `trusted` in the envelope is honored either way, which is the one
escape hatch: a client can send `trusted: false` to force the sandbox path with
no Postgres at all. That is how you exercise the untrusted path in dev — see the
`develop` skill.

## Turning it on

1. **Bring up Postgres.** `docker/postgres/init.sql` is bind-mounted into
   `/docker-entrypoint-initdb.d/` and runs **once, on an empty data directory**,
   creating nine tables plus the read-only `ndifapi` role the API connects as and
   the `login_page` role the account portal uses. It seeds no users and no keys.
   (The compose `postgres` service declares no named volume, so a plain
   `just down` keeps the anonymous one and `init.sql` will not re-run; editing it
   means dropping the volume.)

2. **Understand the query.** The API runs exactly one:

   ```sql
   SELECT k.user_id, u.email, ut.name AS user_tag
   FROM keys k
   LEFT JOIN users u ON u.user_id = k.user_id
   LEFT JOIN key_user_tag_assignments kuta ON kuta.key_id = k.key_id
   LEFT JOIN user_tags ut ON ut.user_tag_id = kuta.user_tag_id
   WHERE k.key_id = $1
   ```

   **Validity is "a row exists in `keys`"** — key issuance is the account portal's
   job, a separate repo. The API key *is* the `key_id` UUID. The `LEFT JOIN`s mean
   a key with no tags still returns one row, which is how "known key, no tags" is
   distinguished from "unknown key".

3. **Create the two tags the server acts on.** Every other tag name is inert.

   | Tag | Effect |
   |---|---|
   | `trusted` | the request runs in-process instead of in a runner, and any model it triggers a deploy for loads with `trust_remote_code` |
   | `priority` | the request sorts ahead of all normal traffic for that model, FIFO against other priority requests |

   `priority` is a strict group, not a queue jump, and there is **no aging**:
   sustained priority load starves normal traffic indefinitely, with autoscaling
   (max 3 replicas) the only relief.

   ```sql
   INSERT INTO user_tags (name, description)
   VALUES ('trusted',  'may run without sandbox isolation'),
          ('priority', 'jumps the model queue');
   ```

4. **Point the API at it** — `NDIF_POSTGRES_URL` on the `api` service (the line is
   commented out in the compose file). The pool is created lazily, so you see
   `Postgres connected` on the first authenticated request, not at boot. If the
   URL is set and `asyncpg` is not installed, `connect()` **raises** rather than
   silently disabling auth — the opposite of the fail-open telemetry providers.

5. **Create a key.**

   ```sql
   INSERT INTO users (email) VALUES ('researcher@example.edu') RETURNING user_id;
   INSERT INTO keys (user_id) VALUES ('<user_id>') RETURNING key_id;  -- the API key
   INSERT INTO key_user_tag_assignments (key_id, user_tag_id)
   SELECT '<key_id>', user_tag_id FROM user_tags WHERE name = 'trusted';
   ```

   Grant `trusted` only to keys you would hand a shell on the model node.

6. **Verify.** Each failure mode is a distinct status code:

   | Condition | Status |
   |---|---|
   | no `ndif-api-key` header | 401 |
   | header present but not a UUID | 400 |
   | well-formed UUID, no row in `keys` | 403 |
   | Postgres unreachable or erroring | 503 — auth fails **closed**, deliberately |

   ```bash
   curl -s -H "ndif-api-key: $NDIF_API_KEY" localhost:8001/whoami
   # {"email": "researcher@example.edu", "tags": ["priority"]}
   ```

   `/whoami` returns `{"email": null, "tags": []}` for a missing or unknown key
   rather than erroring. An empty `tags` with a real email is what you want for a
   normal user: not trusted, not priority.

A 503 from every probe means the route's `require_ray_connection` dependency
tripped first and auth was never reached.

## Gotchas around auth

- **Auth is checked at ingress only.** Nothing downstream re-verifies. Anyone who
  can reach Redis, the Ray client port (10001) or the Ray dashboard (8265)
  bypasses it completely.
- **Models deployed while auth was off keep their `trusted` deployment.** The flag
  is fixed on the `Deployment` at creation. Evict and redeploy after turning auth
  on if you care that a model no longer loads with `trust_remote_code`.
- **The dashboard's auth is entirely separate** and knows nothing about
  `NDIF_POSTGRES_URL`. Remove `NDIF_DASHBOARD_DEV_MODE`, then set
  `NDIF_DASHBOARD_USERNAME`, a bcrypt `NDIF_DASHBOARD_PASSWORD_HASH`
  (`python -m ndif.services.dashboard.backend.auth hash '<password>'`) and a
  random `NDIF_DASHBOARD_SESSION_SECRET` — the default is the literal string
  `change-me-please-this-is-not-secure`, so anyone who reads the repo can forge a
  session cookie. With dev mode off and no hash set, **nobody can log in at all**.
- **Enabling auth is also what makes per-user attribution work** — `email` and
  `api_key` ride with the request into every log record and metric.

## The sandbox, stated honestly

With auth on and a key that lacks `trusted`, user code runs in a **separate
process** from the model actor, interleaved with the forward pass over a Unix
socket, fresh per request and stopped afterwards, started with an allowlisted
environment (`RUNNER_ENV`) rather than a copy of the actor's — so a block cannot
read your `HF_TOKEN`, `NDIF_INFLUX_TOKEN`, `NDIF_POSTGRES_URL` or object-store
credentials out of `os.environ`.

That is one hole closed, not a boundary. The runner is an ordinary child process:
same user, filesystem, network, and visible GPUs, with no namespaces, seccomp,
rlimits, or filesystem jail. Size your threat model accordingly.

## What NDIF does not provide

State this plainly before deploying:

- **No TLS.** Gunicorn binds plain HTTP on `0.0.0.0:8001` and there are no
  certificate options anywhere; API keys travel in the `ndif-api-key` header.
  Terminate TLS yourself and never expose 8001 directly.
- **No rate limiting or quotas.** The only per-key behaviours implemented are the
  `trusted` and `priority` tags. One key can submit unlimited requests.
- **No multi-tenancy.** All users share the same deployments and GPUs.
- **No secret management.** Credentials are environment variables.
- **No test CI and no compatibility guarantees.** The workflows build and publish;
  none of them runs tests. This is v0.1.0.
- **No horizontal API tier.** `NDIF_API_WORKERS` scales gunicorn workers in one
  process group, and the dispatcher is started exactly once by the gunicorn
  master. Two API *containers* against one Redis would run two dispatchers — not
  a configuration the code anticipates.

## Pre-flight checklist

| # | Check | Done when |
|---|---|---|
| 1 | `NDIF_POSTGRES_URL` set, keys created, `trusted` granted deliberately | `/whoami` with a key returns the email; without one, `/request` 401s |
| 2 | Postgres password is not `admin`, and the DB is not published to the world | — |
| 3 | `NDIF_DASHBOARD_DEV_MODE` removed; username, bcrypt hash, random session secret set | the dashboard prompts for a login |
| 4 | Object-store credentials replaced; bucket durable | — |
| 5 | `NDIF_OBJECT_STORE_PUBLIC_URL` resolves and fetches **from a client machine** | a remote trace's result downloads from off-host |
| 6 | `NDIF_REDIS_URL` explicitly set on **every** service, never left at localhost | `ndif info` in each container shows the right host |
| 7 | Head/worker split correct: `NDIF_RAY_HEAD_ADDRESS` on workers only | `ndif status` lists every GPU node |
| 8 | 6385 / 8076 / 52366 open between nodes; 10001 and 8265 **not** public | — |
| 9 | TLS-terminating proxy in front of 8001; 8001 not directly exposed | `https://` endpoint answers `/ping` |
| 10 | Persistent volumes for the HF cache, MinIO, Postgres, Loki, Influx, dashboard data | survives a container recreate |
| 11 | `HF_TOKEN` set on `ray` for gated checkpoints | the gated model deploys |
| 12 | `NDIF_ENVIRONMENT` set to something other than `dev` | Grafana can separate this deployment |
| 13 | `NDIF_DEFAULT_EXECUTION_TIMEOUT_SECONDS` set — **unset means no cap at all** | a runaway block stops holding its replica forever |
| 14 | `NDIF_MODEL_CACHE_PERCENTAGE`, `NDIF_AUTOSCALING_MAX_REPLICAS` and padding factors reviewed | — |
| 15 | `NDIF_MIN_NNSIGHT_VERSION` set to the oldest client you will support | an older client gets a clear 400 |
| 16 | A retention policy for anything you must keep | disk does not fill unbounded |

Actual deployment specifics are out of scope for NDIF: bring your own TLS
termination, orchestration, secret store and infrastructure. There is no
production compose file — a real deployment is your own orchestration wrapping
the same images and the same `NDIF_*` environment.
