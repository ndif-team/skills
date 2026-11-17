# Changing the server's code

You have the repository open, and `docs/developing/` in it is the reference.
This page is a map of it and the rules that are easy to get wrong. Read
`docs/concepts/request-lifecycle.md` once: every other page zooms into one of
its hops. The repository's `CLAUDE.md` has its own rules: branch off `dev`,
change the docs in the same PR as the code, cite code by path and symbol.

## Where a request goes

```text
client ── websocket /subscribe, then POST /request
api worker        services/api/app.py create_request: checks the Ray flag, the API key
                  (auth.py validate_request stamps email and tags), client versions;
                  LPUSH to the Redis list `queue`; answers RECEIVED
dispatcher        services/api/queue/dispatcher.py: one process, the only Ray client;
                  BRPOP, route by model_key to a Processor
Processor         queue/processor.py: the model's RequestQueue, provisioning
                  (PROVISIONING, DEPLOYING), autoscaling
controller        services/ray/deployments/controller/: sizes the model (evaluator.py),
                  picks node and GPUs and evictions (cluster.py, node.py), creates a
                  detached actor "<replica id>:ModelActor:<model key>" (deployment.py)
Replica           queue/replica.py dispatch: DISPATCHED, then the Ray call run(request)
model actor       services/ray/deployments/modeling/base.py run: RUNNING, runs the block
                  on a worker thread against a timeout and a cancel, sends the result
                  inline through Redis or uploads it; exactly one COMPLETED or ERROR
client            collects from the websocket, or from the presigned URL
```

Statuses after `RECEIVED` are published on a Redis channel named by the
client's session id, which `/subscribe` forwards; a non-blocking request has no
session and its statuses are stored in the object store instead.

**A new HTTP endpoint** lives in `services/api/app.py`. An API worker cannot
ask Ray anything: what an endpoint needs from the cluster it gets through
Redis, the way `/status` and `/env` do (a cached key, a trigger stream, a
worker in the dispatcher that refreshes it) or the way `ndif queue` does (the
events stream). `docs/developing/api-service.md`,
`docs/developing/redis-layer.md`.

## Where things live

| To change | Open | Docs page |
|---|---|---|
| ingress, auth, version gating | `services/api/{app,auth,versioning}.py` | `api-service.md` |
| queueing, priorities, autoscaling, kill | `services/api/queue/{dispatcher,processor,replica,request_queue,config}.py` | `queue-internals.md` |
| placement, eviction, GPU accounting | `services/ray/deployments/controller/cluster/{cluster,node,evaluator}.py` | `controller-internals.md` |
| loading, running a block, results | `services/ray/deployments/modeling/{base,util,nns}.py` | `model-actor.md` |
| the runner process and its protocol | `services/ray/sandbox/` | `sandbox-internals.md` |
| tensor-parallel replicas | `services/ray/tp/` | `src/ndif/services/ray/tp/ARCHITECTURE.md` |
| a connection to Redis, Ray, S3, Postgres, Loki, InfluxDB | `common/providers/` | `providers.md` |
| the request and response models | `common/schema/` | `docs/reference/schemas.md` |
| a CLI verb | `cli/commands/` (click only) and `cli/lib/` (the logic) | `cli-internals.md` |
| the dashboard | `services/dashboard/{backend,jobs,frontend}` | `dashboard-internals.md` |

Paths are under `src/ndif/`, pages under `docs/developing/`.
`docs/developing/repo-layout.md` has the full table.

## Rules that are easy to break

- **Never import Ray in an API worker.** The dispatcher, spawned once by
  gunicorn's `on_starting`, owns the only client.
- **Exactly one dispatcher.** The queue assumes one consumer of the Redis
  list, and one event loop serves every model, so a blocking call in the queue
  code stalls all of them.
- **Redis is the only durable handoff.** A request the dispatcher has popped
  is a Python object in its memory until a terminal response.
- **An actor's exception arrives wrapped in `RayTaskError`.** Over Ray Client a
  bare `isinstance` against the original type matches nothing: read `.cause`.
- **`common/` never imports from `services/`**, and CLI logic goes in
  `cli/lib/`, which the dashboard backend imports directly. Import Ray, nnsight
  and redis inside the functions there, so `ndif --help` stays fast.
- **Telemetry providers connect at import and own threads**, which do not
  survive a fork. Connect them after forking: `post_fork` for gunicorn workers,
  a spawned process for the dispatcher, `__init__` for a Ray actor.
- **Never import `services/ray/sandbox/nns.py` on the host.** It patches
  nnsight process-wide for the runner's side of the socket.
- **`Status` is nnsight's enum.** A new status has to ship in nnsight and reach
  users first. Say something new in `description`, or in a `LOG`.
- **NDIF does not use Ray Serve.** Deployments are detached actors; ignore
  every Serve page a search returns.
- **The image installs the package with `--no-deps`.** A new dependency goes in
  `requirements.txt` as well as `pyproject.toml`, and a new non-Python file
  (a `start.sh`) needs a `[tool.setuptools.package-data]` entry or it is missing
  from every installed copy, the compose image included.
- **A new environment variable gets a row in `docs/reference/env-vars.md`.**
  `pytest tests/test_docs.py` checks that the paths, symbols and links the docs
  cite exist.

## Trusted and untrusted execution

Two things decide where a block runs: `request.trusted`, read from the tags
stamped at ingress, and the actor class. `SandboxModelDeployment.execute` runs a
trusted request through the base class, in-process, and anything else in a
runner. In the runner each `tracer.invoke` is a greenlet that parks when it
needs a value, and the host drives the real forward pass and sends values over
a Unix socket. The runner holds a weightless model, so everything a block
touches must pickle both ways. The invariant is that both paths return the same
bytes for the same request (`tests/test_sandbox_conformance.py`).

A normal compose stack never runs a line of this: with auth off a request with
no tags is trusted. Client-supplied tags are honoured while auth is off, so
sending `"tags": []` reaches the runner without Postgres; the test plugin below
does that.

## Extending

- **A model actor**: subclass `BaseModelDeployment` and wrap it in
  `@ray.remote(num_cpus=1, max_restarts=-1)`. `run()` is a template; override
  its hooks (`execute`, `execution_scope`, `interrupt`, `format_error`,
  `error_name`, `cleanup`, and `commit`, `restart` where needed) and inherit the
  rest. What a replacement `execute` must keep: return `(blob,
  deserialize_ms)`; set `self.execution_ident` on the worker thread before
  anything can block, or a cancel is silently dropped; be interruptible; raise
  `CachedActorError` when WARM so the queue re-queues; leave no request state
  behind. `cancel(KILL_REASON_PREEMPTED)` re-queues the request at the front;
  use it for a deliberate cancel and the request runs again for ever. Select it with `--actor-class`, `NDIF_MODEL_IMPORT_PATH`, or
  `NDIF_TP_MODEL_ACTOR_CLASS` for tensor-parallel replicas; the path is imported
  inside the Ray worker. A constructor argument of your own has no route from
  the controller: read it from the environment.
  `docs/developing/adding-a-model-actor.md`; copy `SandboxModelDeployment`.
- **A provider**: a class-level singleton with a `CONFIG` of `attr: (ENV_VAR,
  default, cast)`. An optional system defaults to an empty URL and is silently
  off. Postgres is the deliberate exception: a URL without `asyncpg` raises,
  because auth that is silently off is a hole. Add a pyproject extra.
  `docs/developing/providers.md#writing-a-new-provider`.
- **A service**: a `start.sh` in its package, a `Service` entry in
  `cli/service.py` (`SERVICES` start by default, `OPTIONAL_SERVICES` by name),
  the package-data entry, and a compose block.
  `docs/developing/adding-a-service.md`.
- **A CLI command**: `cli/commands/<verb>.py` with one click command,
  registered in `cli/main.py`, logic in `cli/lib/` taking an `on_message`
  callback. `docs/developing/cli-internals.md#adding-a-command`.

## Run what you changed

```bash
just ta                       # down, build, up: the source is baked into the image
just logs api                 # wait for "Connected to Ray"
pip install -e ".[dev]"       # ruff, httpx, pytest, pytest-asyncio
pytest tests/test_docs.py     # the docs gate; needs no server
pytest tests/                 # NOT a bare `pytest`: there is no testpaths
```

- **The live files talk to `http://localhost:8001`** (`HOST` in
  `tests/conftest.py`; there is no variable for it) with gpt2, which the queue
  loads on demand. With no server they skip: `pytest tests/test_nnsight_remote.py`
  prints `65 skipped` and exits 0. A skipped suite is not a passing one.
- **The non-blocking tests download from `NDIF_OBJECT_STORE_PUBLIC_URL`**, which
  must be reachable from where pytest runs.
- **Force the untrusted path** when you touched execution:

  ```bash
  PYTHONPATH=tests pytest tests/test_nnsight_remote.py -p anyio -p conftest_untrusted
  ```

  `conftest_untrusted` adds `"tags": []` to every request. `-p anyio` goes
  first, because with the plugin loading nnsight before anyio, collection
  fails with `TypeError: Attribute 'save' is missing its type annotation`. The
  gradient tests fail on this path by design. Confirm the path by runner pids:
  [production and auth](operate-production-and-auth.md#what-an-untagged-key-can-do).
- **No workflow runs any of this**, or `ruff check src/`. They are yours to run.

## Against a local nnsight

Not run for this guide: it needs an nnsight checkout to mount.

nnsight is an ordinary dependency of the image. On compose, `just` bind-mounts
an nnsight from the host over the image's copy in `api`, `ray` and
`dashboard`, when `NNSIGHT_PATH` names its package directory
(`<checkout>/src/nnsight`) or when `python -c "import nnsight"` in your shell
finds an editable install.

- **`just nnsight` prints which applies.** `just --dry-run up` does not
  evaluate the detection. `NNSIGHT_PATH= just up` forces the image's own copy.
- **Carry `NNSIGHT_PATH` on every `just` call**, or the next `just up`
  recreates the containers without the mount.
- **A fresh checkout has no compiled extension**, and then every remote
  `.save()` fails with `AttributeError: 'Tensor' object has no attribute
  'save'`. Build it with the image's toolchain: `docker run --rm --entrypoint
  python -v <checkout>:/src -w /src dev-ray setup.py build_ext --inplace`
  (as root, so the build products are root-owned).
- **`nnsight.__version__` in the container reports the image's version under
  the mount.** `grep nnsight /proc/self/mountinfo` there shows whether it is
  mounted.
- **Outside Docker**, `pip install -e` nnsight before ndif.

`docs/developing/nnsight-integration.md#developing-against-a-local-nnsight`,
and `#bumping-the-client` for what to check when the nnsight version moves.

## Releasing

The version is the git tag (setuptools-scm); there is nothing to bump. One
published GitHub release fires `publish.yml` (the wheel to PyPI) and
`publish_docker.yml` (the `ndif/ndif` images to Docker Hub, one per CUDA line).
`build_images.yml` is manual. None of the three runs a test, so a green
workflow means it built. `docs/developing/contributing.md#ci-and-releases`.
