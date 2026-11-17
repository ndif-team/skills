# Letting other people use the server: auth, the sandbox, production

## Two switches

Where a request's block runs is decided by two things, and both must be set
before anyone else can reach the API.

1. **`NDIF_POSTGRES_URL` on the api decides the request's tags.** Set, the
   `ndif-api-key` header is looked up in Postgres and the request carries that
   key's tags; it is trusted when they include `trusted`. Unset, there are no
   keys and a request that sends no tags is stamped `trusted`.
2. **The model actor class decides whether "untrusted" changes anything.**

| Actor class | An untrusted request's block runs |
|---|---|
| `ndif.services.ray.deployments.modeling.base.ModelActor`, the default | in the model process |
| `ndif.services.ray.sandbox.model.SandboxModelActor` | in a runner process |
| `ndif.services.ray.tp.model.TPModelActor` | in the model process, on every rank |
| `ndif.services.ray.tp.model.SandboxedTPModelActor` | in one runner the whole group shares |

The controller reads `NDIF_MODEL_IMPORT_PATH`, then the older
`NDIF_DEFAULT_MODEL_ACTOR_CLASS` (what compose sets), then falls back to the
base `ModelActor`. `NDIF_TP_MODEL_ACTOR_CLASS` is separate and replaces the
class for a tensor-parallel replica. A per-deployment `--actor-class` overrides
both, so check every replica: `ndif status --json-output | jq -r
'.deployments[] | select(.model_key) | .actor_class'`.

## Turning both on

**The published image** has no Postgres. Run one beside it, initialized with
the repository's `docker/postgres/init.sql` (nine tables and the read-only
`ndifapi` role the API connects as, no users or keys):

```bash
docker network create ndif-net
docker run -d --name ndif-pg --network ndif-net \
    -e POSTGRES_USER=admin -e POSTGRES_PASSWORD=admin -e POSTGRES_DB=ndif \
    -v ndif-pgdata:/var/lib/postgresql/data \
    -v "$PWD/docker/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql:ro" \
    postgres:16-alpine
docker run -d --name ndif --network ndif-net --gpus all --shm-size 4g \
    -p 127.0.0.1:8001:8001 -p 127.0.0.1:9000:9000 \
    -e NDIF_POSTGRES_URL=postgresql://ndifapi:admin@ndif-pg:5432/ndif \
    -e NDIF_MODEL_IMPORT_PATH=ndif.services.ray.sandbox.model.SandboxModelActor \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    ndif/ndif
```

`init.sql` runs once, on an empty data directory, and gives `ndifapi` and
`admin` the password `admin`: change both before the database is reachable.

**Compose** already runs Postgres and the sandbox actor. Uncomment
`NDIF_POSTGRES_URL` on the `api` service and run `just up api`. `just restart
api` keeps the old environment, so auth silently stays off. Its Postgres
volume is anonymous: keys survive `just up` and `just restart` and are gone
after `just down` or `just ta`, when every key answers 403 until re-inserted.
Give the service a named volume first.

**From source**: `pip install "ndif[postgres]"`, both variables in the
environment, `ndif start --restart`. With the URL set and `asyncpg` missing,
every keyed request gets 503 `Auth backend unavailable.`

## Tags and keys

The server acts on three tag names. Any other name is inert.

| Tag | A request with it |
|---|---|
| `trusted` | runs in the model process, and may deploy a `trust_remote_code` model |
| `trust_remote_code` | may deploy a model whose key carries `trust_remote_code`, and nothing more |
| `priority` | is served before untagged requests for the same model, first come first served among priority requests |

An API key is the `key_id` UUID of a row in `keys`; the API runs one read-only
query and issues nothing. Create the tags once, then a user and a key:

```bash
psql() { docker exec ndif-pg psql -U admin -d ndif -At -c "$1"; }
psql "INSERT INTO user_tags (name, description) VALUES
        ('trusted', 'code runs in the model process'),
        ('trust_remote_code', 'may deploy checkpoints that run their own code'),
        ('priority', 'served ahead of untagged traffic');"
U=$(psql "INSERT INTO users (email) VALUES ('researcher@example.edu') RETURNING user_id;" | head -1)
K=$(psql "INSERT INTO keys (user_id) VALUES ('$U') RETURNING key_id;" | head -1)     # the API key
psql "INSERT INTO key_user_tag_assignments (key_id, user_tag_id)
        SELECT '$K', user_tag_id FROM user_tags WHERE name = 'trusted';"              # only for a key you trust
```

On compose, `psql` is `docker compose -f docker/docker-compose.yml exec -T
postgres psql -U admin -d ndif -At -c`. The client sets the key with
`nnsight.CONFIG.API.APIKEY = "<key>"`.

## Verify

```bash
curl -s -H "ndif-api-key: $K" localhost:8001/whoami
# {"email":"researcher@example.edu","tags":["trusted"]}
probe() { curl -s -w ' %{http_code}\n' -X POST localhost:8001/request "$@" -F 'data={"model_key":"x"}' -F blob=@/dev/null; }
probe                                    # 401  Missing or invalid API key. ...
probe -H 'ndif-api-key: abc'             # 400  Invalid API key format: 'abc'. ...
probe -H "ndif-api-key: $(uuidgen)"      # 403  Invalid API key.
```

- **An nnsight client with no key set sends the placeholder `local`**, so with
  auth on it gets the 400, `Invalid API key format: 'local'`, not the 401.
- **A 401 or 403 proves auth is on.** With it off, every probe gets past the key
  check.
- **Postgres down or erroring is a 503**, `Auth backend unavailable.` Auth
  fails closed. A 503 `compute backend is reconnecting` is Ray, and comes
  before the key is looked at.

## What an untagged key can do

On a sandbox actor an untagged key's block runs in a runner that holds a
weightless copy of the model and drives the real one over a Unix socket.
Reading and editing activations, `.save()`, `.source`, module calls, caches and
generation all work. These do not:

- **Reading weights.** `module.weight` there has a shape and no data, and
  saving anything derived from one fails with `NotImplementedError: Cannot copy
  out of meta tensor; no data!`
- **Gradients.** `.grad` fails with `OutOfOrderError: '...grad.i0' was
  requested but the model already ran past it`.
- **Editing weights and training.** Writes land on the weightless copy and
  change nothing.

Tell users before you switch; whoever needs those needs a `trusted` key.

**Which path did a request take?** No log record says. Runners are children of
the model actor, kept warm in a pool (`NDIF_SANDBOX_POOL_SIZE`, 7), and each
untrusted request uses one up, so compare runner pids before and after:

```bash
docker exec ndif sh -c 'for p in /proc/[0-9]*; do echo "${p#/proc/} $(tr "\0" " " < $p/cmdline)"; done' | grep sandbox.runner | cut -c1-70
```

New pids after the request mean it ran in a runner. A trusted request leaves
them unchanged, and a base `ModelActor` has none. (The image has no `ps`.)

## What the sandbox is, and is not

It is a separate process started with an allowlisted environment, so a block
cannot read `HF_TOKEN`, `NDIF_POSTGRES_URL` or the object-store credentials
from `os.environ`, and cannot read or edit the weights or train. It is not a
jail: the runner has the actor's user, filesystem, network and GPUs, no
namespaces, seccomp or resource limits, and sits outside the replica's GPU
memory cap. `docs/concepts/sandbox-execution.md`.

## The dashboard login

The dashboard's login is separate from API keys, and it can deploy and evict.
`NDIF_DASHBOARD_DEV_MODE=true`, the compose default, turns the login off. For a
real one remove that and set:

```bash
docker exec ndif python -m ndif.services.dashboard.backend.auth hash '<password>'   # prints the bcrypt hash
# NDIF_DASHBOARD_USERNAME=admin
# NDIF_DASHBOARD_PASSWORD_HASH=<the hash>      in compose YAML write each $ as $$
# NDIF_DASHBOARD_SESSION_SECRET=<32 random bytes>   the default is a public string
```

With dev mode off and no hash, nobody can log in. Its deploy endpoint takes
`trust_remote_code` with no tag check, so an open dashboard can load any
checkpoint's code onto the cluster.

## What NDIF does not provide

- **No TLS.** The API serves plain HTTP on `0.0.0.0:8001` and keys travel in a
  header. Terminate TLS in a proxy and never publish 8001 itself.
- **No key on result retrieval.** `GET /response/{id}` and a presigned result
  URL need only the id or the URL.
- **No rate limits, quotas or multi-tenancy.** Every key shares the same
  deployments and GPUs.
- **No authentication on Redis, the Ray client port or the Ray dashboard.**
  Auth is checked at the API only; anyone who reaches 6379, 10001 or 8265
  bypasses it.
- **No second API instance.** Each API container starts its own dispatcher, and
  the queue assumes exactly one.
- **No retention.** Result blobs and stored responses are never deleted.
- **No production compose file.** TLS, orchestration and secrets are yours.

## Before you open the port

| Check | Done when |
|---|---|
| Auth on, keys created, `trusted` granted deliberately | `/whoami` with a key returns its email; `/request` without one is refused |
| A sandbox actor class on every replica, the tensor-parallel class too if used | `actor_class` in `ndif status --json-output`; runner pids change across an untagged request |
| Users told what untagged keys cannot do | |
| Postgres on a durable volume, passwords changed, port not published | keys survive a teardown |
| Dashboard dev mode off; username, hash and secret set | it asks for a login |
| Object-store credentials changed; `NDIF_OBJECT_STORE_PUBLIC_URL` fetches from a client machine | a non-blocking result downloads from off the host |
| `NDIF_REDIS_URL` set on every service; 6379, 5432, 10001, 8265 and the telemetry ports unreachable from outside | a scan shows only the TLS endpoint and the object store |
| `NDIF_DEFAULT_EXECUTION_TIMEOUT_SECONDS` set | a runaway block stops holding its replica |
| `NDIF_MIN_NNSIGHT_VERSION` set to the oldest client you support | an older client gets a clear 400 |
| Volumes for the weights cache, object store, Postgres and dashboard data | they survive `just down` |

`docs/runbooks/enable-auth.md` and `docs/operating/production.md` are the long
forms.
