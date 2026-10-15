---
name: develop
description: Change NDIF's own server code. Use when working inside the ndif repo — "how does an NDIF request actually run", "where do I add an endpoint / a model actor / a provider / a CLI command", "what is the dispatcher", "trusted vs untrusted execution", "the sandbox runner", "how do I test NDIF", "how do I force the untrusted path", "how is nnsight wired in", "how do I develop against a local nnsight", "how do I cut an NDIF release" — or when reading a traceback from NDIF's internals rather than from a user's block. Covers the repo layout, the process map, the request lifecycle, the six model-actor hooks, the test suite (CI runs none of it), and the tag-driven publish workflows. For running a server rather than changing one, use `selfhost`, `operate` and `troubleshoot`.
---

# Developing NDIF

Read `docs/concepts/request-lifecycle.md` once. It is the keystone: every other
page in the repo's docs is a zoom into one of its hops, and this skill assumes
it. `docs/developing/architecture-overview.md` is the same system cut the other
way, by process.

## Four facts that frame the codebase

1. **One image, one service per container.** Every service is the same Docker
   image, whose `ENTRYPOINT` is the `ndif` CLI and whose default command is
   `start --foreground`. `NDIF_SERVICE` — one name, a space/comma list, or the
   image default `all` (redis, minio, ray, api) — selects what that container
   runs. There is no per-service build.
2. **The API process cannot talk to Ray.** Only the **dispatcher** — a separate
   process spawned by gunicorn's `on_starting` hook — holds a Ray client. Every
   endpoint that appears to know about the cluster is reading a Redis-backed
   cache the dispatcher refreshes.
3. **Execution forks on two things: `request.trusted` and the actor class.** A
   sandbox actor runs an untrusted request's block in a separate runner process
   driven over a Unix socket; the base `ModelActor` runs every request in-process
   whatever `trusted` says. With auth off a client-supplied `trusted` is honored
   and an unspecified one defaults to trusted.
4. **Redis is the only durable handoff.** Once a request leaves the Redis list it
   is a Python object in the dispatcher's memory until it reaches a terminal
   response. Nothing in between survives a restart.

## The process map

```text
api container (NDIF_SERVICE=api)
  gunicorn master ── on_starting ─> dispatcher process (one asyncio loop, owns THE Ray client)
                  └─ uvicorn workers × NDIF_API_WORKERS  (FastAPI routes only, never import Ray)

ray container (NDIF_SERVICE=ray, GPU)
  ray head ── Controller actor      (pinned to the head by a head=10 custom resource)
           └─ ModelActor replicas   (detached, namespace NDIF, "{replica_id}:ModelActor:{model_key}")
                 └─ runner subprocess, fresh per untrusted request (sandbox actors only)

dashboard container (NDIF_SERVICE=dashboard)
  uvicorn + Vue SPA, plus monitor/reconcile/report crons — all going through cli/lib

Redis   queue · caches · pub/sub · streams
Object store   result blobs
```

Several concurrency mechanisms are in play at once, which is the single most
confusing thing about the codebase:

| Code | Runs in | Mechanism |
|---|---|---|
| FastAPI routes | uvicorn worker processes | asyncio |
| Dispatcher, Processor, Replica | one spawned child process | one asyncio loop |
| Controller | a Ray actor on the head | Ray actor calls |
| `run()` orchestration | model actor, main thread | thread + timeout race |
| `execute()` — the forward pass | model actor, worker thread | blocking Python/CUDA |
| A trusted user block | model actor, same worker thread | greenlets, interleaved with the forward |
| An untrusted user block | a separate runner process | greenlets, driven over a Unix socket |

The greenlet model is nnsight's, not NDIF's — NDIF reuses it verbatim and, for
untrusted requests, splits it across a process boundary.

## Repo layout

```text
src/ndif/
├── cli/            the `ndif` command; commands/ is click decoration, lib/ is the logic
├── common/         shared by every service — may NOT import from services/
│   ├── providers/  one module per external system (Redis, Ray, S3, Postgres, Loki, Influx)
│   ├── redis/      key/channel/stream names only
│   ├── schema/     BackendRequestModel / BackendResponseModel — subclasses of nnsight's
│   ├── metrics.py  one class per InfluxDB measurement
│   └── telemetry.py  event(logger, msg, **fields)
└── services/
    ├── api/        start.sh, gunicorn_conf.py, app.py, auth.py, versioning.py, queue/
    ├── ray/        start.sh, resources.py, deployments/{controller,modeling}, sandbox/, tp/
    └── dashboard/  backend/, jobs/, frontend/ (dist/ is committed)
```

`docs/developing/repo-layout.md` carries the full "I want to change X, open Y"
table. The two entries people reach for most: queue behaviour and autoscaling are
`services/api/queue/{dispatcher,processor,replica,config}.py`; placement,
eviction and GPU accounting are
`services/ray/deployments/controller/cluster/{cluster,node,evaluator}.py`.

## The request, in ten hops

Each hop in full, with its code: `docs/concepts/request-lifecycle.md`.

1. **Client serializes** the block — source plus referenced globals,
   cloudpickled and zstd-compressed — under a small JSON envelope.
2. **Subscribe, then POST**, so no update is published before anyone listens.
3. **API ingress** runs `require_ray_connection`, `validate_request` and
   `validate_client_versions`, `LPUSH`es the request, returns `RECEIVED`.
4. **Dispatcher pops** (`BRPOP` plus a non-blocking drain) and routes by
   `model_key` to a lazily created `Processor`.
5. **Processor queues and provisions.** With a replica serving the reply is
   `QUEUED`; on a cold start it is `PROVISIONING`, then `DEPLOYING`, then
   `QUEUED` ("Moved to position N") once the replica is ready. A cold request's
   full sequence is `RECEIVED` → `PROVISIONING` → `DEPLOYING` → `QUEUED` →
   `DISPATCHED` → `RUNNING` → `COMPLETED`.
6. **Controller places a replica** — size on meta, pick node and GPUs, evict if
   it must, create a detached Ray actor.
7. **Replica dispatches**: wait on `__ray_ready__`, publish `DISPATCHED`, make
   the Ray call with the whole pickled request.
8. **The actor runs the block**: `run()` publishes `RUNNING`, applies
   `request.env`, and races `execute()` against the timeout and a cancel event.
   `print()` becomes `LOG` responses.
9. **Result back.** A blocking request's blob at or under
   `RedisProvider.max_publish_bytes` — `min(soft, hard // 2)` of Redis's pubsub
   output-buffer limits, 8 MiB on stock Redis — rides on the `COMPLETED`
   response; anything larger, and every non-blocking result, is PUT at
   `{request.id}.pt` and presigned for an hour.
10. **Client collects** off the frame or the URL and pushes the values into the
    caller's frame.

To watch the statuses from a client, set `nnsight.CONFIG.APP.DEBUG = True`
(each status on its own line instead of one overwritten line); for a
non-blocking job, poll `curl localhost:8001/response/<id>`, which returns only
the latest status.

What breaks at each hop, the queue objects and the `run()` contract:
[references/request-lifecycle.md](references/request-lifecycle.md).

## The trusted/untrusted fork

In a sandbox actor the fork is three lines of `SandboxModelDeployment.execute`:

<!-- test: skip -->
```python
if request.trusted:
    return super().execute(request)
return self.run_in_runner(request, None)
```

Untrusted, each `tracer.invoke(...)` becomes a **greenlet worker** in the runner
that **parks** whenever it needs a value from the model; a proxy inside the real
interleaver sends the value across when the forward reaches it. It is nnsight's
local behaviour with a socket round trip where a greenlet switch would be. The
runner holds only a *meta* model, so latency is per-*interaction* and everything
a block touches must pickle both ways. The invariant: **both paths produce
identical bytes for the same request** (`tests/test_sandbox_conformance.py`).
`docs/developing/sandbox-internals.md`.

**The runner's environment is an allowlist** (`RUNNER_ENV` in
`sandbox/host.py`): `HF_TOKEN`, `NDIF_POSTGRES_URL` and object-store credentials
are not inherited, and the torch.distributed rank variables are dropped. **That
is process separation, not a security boundary** — same user, filesystem,
network and GPUs; no namespaces, seccomp or rlimits.

## Boundaries not to cross

- **Never import Ray in an API worker.** The dispatcher owns the only client.
- **Never import `sandbox/nns.py` on the host.** It installs process-wide nnsight
  patches that would redirect the *real* model's `interleave` to a socket.
- **Connect telemetry providers after forking.** They connect at import and own
  threads that do not survive `fork()`.
- **Do not assume more than one dispatcher.** The queue design assumes exactly one
  consumer of the Redis list.
- **`common/` never imports from `services/`.**
- **NDIF does not use Ray Serve.** Deployments are detached Ray actors; ignore
  every Serve page a search hands you.

## Extending it

The shape every recipe shares: **prefer subclassing an existing seam to adding a
branch.**

- **A model actor** — `run()` is a template; a subclass overrides its six hooks
  (`execute`, `execution_scope`, `interrupt`, `format_error`, `error_name`,
  `cleanup`), plus `commit()` and `restart()` where needed, and inherits
  everything else. `docs/developing/adding-a-model-actor.md`.
- **A provider** — a classmethod singleton over an external system, with a
  `CONFIG` spec of `attr: (ENV_VAR, default, cast)` tuples and deliberate
  fail-open behaviour. `docs/developing/providers.md#writing-a-new-provider`.
- **A service** — `NDIF_SERVICE` + a `start.sh` + a `Service` in
  `cli/service.py`, plus a `[tool.setuptools.package-data]` entry, which the
  compose image needs too. `docs/developing/adding-a-service.md`.

The contract a custom actor must uphold, actor selection, and the CLI-command
recipe: [references/extending.md](references/extending.md).

## Testing

**No workflow runs the tests.** `docs/developing/testing.md`.

```bash
pip install -e ".[dev]"
just up && just logs api ray      # let Ray settle — it needs a GPU and is slow
pytest tests/                     # NOT a bare `pytest` — there is no testpaths
```

Three groups: **3 live** files (the real nnsight client against a running NDIF;
they skip themselves if nothing answers), **5 model-only** (gpt2 on CPU and a
spawned runner, no server) and **9 offline**. The suite uses gpt2 and the queue
provisions it on demand. **A skipped suite is not a passing suite** — with no
server, `pytest tests/` exits 0. The server address is `HOST` in
`tests/conftest.py` (no env var; a tunnel's local port goes there), and the
non-blocking tests need `NDIF_OBJECT_STORE_PUBLIC_URL` reachable from pytest.

### Forcing the untrusted path

**A normal `just up` never runs a single line of sandbox code.** Two things must
hold:

1. **The request must be untrusted.** Auth-off defaults an *unspecified*
   `trusted` to `True`, but `validate_request` honors an explicit one, so
   **sending `trusted: false` forces the sandbox path with no Postgres**.
   nnsight's request envelope has no `trusted` field;
   `tests/conftest_untrusted.py` injects it:

   ```bash
   PYTHONPATH=tests pytest tests/test_nnsight_remote.py -p conftest_untrusted
   ```

2. **The deployed actor class must be a sandbox one.** Compose sets it; a
   hand-run `ndif start ray` needs `NDIF_MODEL_IMPORT_PATH` (or the legacy
   `NDIF_DEFAULT_MODEL_ACTOR_CLASS`) set to
   `ndif.services.ray.sandbox.model.SandboxModelActor`.

Confirm the path with `pgrep -af 'ndif.services.ray.sandbox.runner'` inside the
`ray` container. The faithful alternative — a real key without the `trusted` tag —
also exercises the 401/400/403/503 ladder, which nothing else does.

## How nnsight is wired in

nnsight is an **ordinary pinned dependency** in `requirements.txt`. The coupling
is deep: `BackendRequestModel` subclasses nnsight's `RequestModel`,
`BackendResponseModel` its `ResponseModel`, and `Status` **is** nnsight's
`Status` — so **you cannot add a status server-side.** A new member has to land
in nnsight, ship, and be installed by the user first. To convey something new
without a client release, put it in `description` on an existing status, or emit
a `LOG`.

**Against a local nnsight, compose first:**

`NNSIGHT_PATH=<checkout>/src/nnsight` (the **package** dir) on **every** `just`
call bind-mounts it over the image's copy. Auto-detection runs
`python -c "import nnsight"` and needs an editable nnsight, so a host with only
`python3` always detects nothing, silently; `just nnsight` is the only reliable
pre-check (`just --dry-run up` always shows the overlay). A fresh checkout lacks
the compiled `_c` extension, so every remote `.save()` fails with
`AttributeError: 'Tensor' object has no attribute 'save'` until you run
`docker run --rm --entrypoint python -v <checkout>:/src -w /src dev-ray setup.py build_ext --inplace`
(as root — it leaves root-owned build artifacts in the checkout). Under the mount
`nnsight.__version__` still reports the image's version — check
`grep nnsight /proc/self/mountinfo` instead, where a mounted line reads
`... 259:2 /u/me/nnsight/src/nnsight /usr/local/lib/python3.12/site-packages/nnsight ...`
and the source column is relative to its filesystem's root, not the host path.
Outside Docker, `pip install -e` nnsight **before** ndif. Procedure:
`docs/developing/nnsight-integration.md#developing-against-a-local-nnsight`; what
to check when the pin moves:
`docs/developing/nnsight-integration.md#bumping-the-client`.

## Releasing

Three workflows, **all of which publish and none of which test**:

| Workflow | Trigger | Publishes |
|---|---|---|
| `build_images.yml` | push to `main` | the hosted image |
| `publish_docker.yml` | a `v*` tag | Docker Hub images (one per CUDA line) |
| `publish.yml` | a published GitHub release | the PyPI wheel |

**The version is the git tag** (setuptools-scm); there is nothing to bump in
`pyproject.toml`. Creating a GitHub release pushes its tag, so **one release
fires both publish workflows** — a `v*` tag is a release, not a checkpoint, and a
green `publish_docker.yml` means the image built, not that the server works.
`docs/developing/contributing.md`.

Before a PR: bring a stack up and run `pytest tests/`; if you touched execution,
run it again with `trusted=False` forced; `ruff check src/`; and if you added an
env var, add its row to `docs/reference/env-vars.md` — the most common drift in
the repo. House style lives in `docs/developing/contributing.md`.

## References

- [references/request-lifecycle.md](references/request-lifecycle.md) — what
  breaks at each hop, the queue objects, the `run()` contract, and where state
  lives.
- [references/extending.md](references/extending.md) — the model-actor contract
  and selection, the provider pattern, adding a service, and adding a CLI
  command.

## Related skills

- `selfhost` — getting a dev stack up.
- `troubleshoot` — reading the logs a change produces.
- `operate` — the operator surface your change has to keep working.
