---
name: ndif-develop
description: Change NDIF's own server code. Use when working inside the ndif repo — "how does an NDIF request actually run", "where do I add an endpoint / a model actor / a provider / a CLI command", "what is the dispatcher", "trusted vs untrusted execution", "the sandbox runner", "how do I test NDIF", "how do I force the untrusted path", "how is nnsight pinned", "how do I cut an NDIF release" — or when reading a traceback from NDIF's internals rather than from a user's block. Covers the repo layout, the process map, the request lifecycle, the five model-actor hooks, the live-server test suite (CI runs none of it), and the tag-driven publish workflows. For running a server rather than changing one, use `ndif-selfhost`, `ndif-operate` and `ndif-troubleshoot`.
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
3. **Execution forks on one boolean.** `request.trusted` decides whether the
   user's traced block runs inside the model actor process or in a separate
   runner process driven over a Unix socket. With auth off a client-supplied
   `trusted` is honored and an unspecified one defaults to trusted.
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
                 └─ runner subprocess, fresh per untrusted request, over a Unix socket

dashboard container (NDIF_SERVICE=dashboard)
  uvicorn + Vue SPA, plus monitor/reconcile/report crons — all going through cli/lib

Redis   queue · caches · pub/sub · streams
Object store   result blobs
```

Four concurrency mechanisms are in play at once, which is the single most
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
table, reproduced in [references/extending.md](references/extending.md). The two
entries people reach for most: queue behaviour and autoscaling are
`services/api/queue/{dispatcher,processor,replica,config}.py`; placement,
eviction and GPU accounting are
`services/ray/deployments/controller/cluster/{cluster,node,evaluator}.py`.

## The request, in ten hops

1. **Client serializes** the traced block to source text plus the globals and
   locals it references, cloudpickled and zstd-compressed, with a small JSON
   envelope (`model_key`, `session_id`, `compress`, a per-request `env`).
2. **Subscribe, then POST.** The client opens `/subscribe` *first* and takes the
   server-minted `session_id`, so no update can be published before anyone is
   listening.
3. **API ingress** runs three gates — `require_ray_connection`,
   `validate_request` (parse, verify the key, stamp `email`/`trusted`/`priority`)
   and `validate_client_versions` — then `LPUSH`es the pickled request onto the
   Redis list and returns `RECEIVED`. Everything after this is asynchronous.
4. **Dispatcher pops** with `BRPOP` and drains up to 31 more, routing by
   `model_key` to a lazily created `Processor`.
5. **Processor queues and provisions.** `RequestQueue` orders by
   `(group, prepend, enqueued_at)`, so a `priority` request sorts ahead of normal
   traffic and stays FIFO against its peers. No HOT replica ⇒ ask the controller.
6. **Controller places a replica** — size on meta, pick node and GPUs, evict if it
   must, create a detached Ray actor.
7. **Replica dispatches**: poll `__ray_ready__`, publish `DISPATCHED`, make the
   Ray call. The pickled request, payload and all, crosses the Ray boundary here.
8. **The actor runs the block.** `run()` refuses immediately if WARM, publishes
   `RUNNING`, applies `request.env`, and races `execute()` on a worker thread
   against the execution timeout and a cancel event. `print()` becomes `LOG`
   responses, one per line.
9. **Result back, by one of two routes.** At or under
   `NDIF_MAX_SOCKET_RESULT_BYTES` (4 MiB) with a live socket it rides on the
   `COMPLETED` response with `pickled=True`; otherwise — and for every
   non-blocking request — it is PUT at `{request.id}.pt` and presigned for an hour.
10. **Client collects** off the binary frame or the URL, decompresses,
    `torch.load`s, and pushes the values into the caller's frame.

Hop-by-hop failure modes: [references/request-lifecycle.md](references/request-lifecycle.md).

## The trusted/untrusted fork

The whole fork is three lines in `SandboxModelDeployment.execute`:

<!-- test: skip -->
```python
if request.trusted:
    return super().execute(request)
return self.run_in_runner(request, None)
```

Untrusted, each `tracer.invoke(...)` block becomes a **greenlet worker** in the
runner process. A worker runs until it needs something from the model, then
**parks** on that location. On the host, one proxy per worker sits inside the
model's real interleaver; when the forward pass reaches a parked location the
proxy sends the value across, the worker wakes, runs to its next park, and hands
control back. It is exactly nnsight's local behaviour with a socket round trip
where a greenlet switch would be — which is why the sandbox reuses nnsight's
`Mediator`, `Batcher` and `Cache` rather than reimplementing them.

The invariant that shapes the subsystem: **both paths must produce identical
bytes for the same request.** `tests/test_sandbox_conformance.py` asserts exactly
that.

The model, its weights and its module tree never move. The runner holds a *meta*
model built from the model key, so structural work (tokenizing, walking the tree)
happens there and anything needing a real activation is a message — which makes
latency per-*interaction*, not per-request, and makes everything a block touches
have to be picklable in both directions.

**The runner does not inherit the actor's environment.** It starts with an
allowlist (`RUNNER_ENV` in `sandbox/host.py`): `PATH`, `LD_LIBRARY_PATH`, `HOME`,
`LANG`, `LC_ALL`, `TZ`, `CUDA_VISIBLE_DEVICES`, `HF_HOME`,
`HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `XDG_CACHE_HOME`,
`OMP_NUM_THREADS`, `TOKENIZERS_PARALLELISM` — and nothing else. A block therefore
cannot read the operator's `HF_TOKEN`, `NDIF_INFLUX_TOKEN`, `NDIF_POSTGRES_URL`
or object-store credentials out of `os.environ`. It also deliberately drops
`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR` and `MASTER_PORT`, which a
runner spawned from a TP rank-0 would otherwise inherit and start life claiming
to be rank 0 of a group it is not in.

**That is one hole closed, not a boundary.** The runner is an ordinary child
process — same user, filesystem, network and visible GPUs, no namespaces, no
seccomp, no rlimits. Do not describe it as a security boundary.

## Boundaries not to cross

- **Never import Ray in an API worker.** The dispatcher owns the only client.
- **Never import `sandbox/nns.py` on the host.** Importing it installs
  process-wide nnsight patches that would redirect the *real* model's
  `interleave` to a socket. It is imported only inside the runner.
- **Connect telemetry providers after forking.** They connect at import and own
  threads; a new entry point that imports them pre-fork gets neither console
  formatting nor telemetry, silently.
- **Do not assume more than one dispatcher.** The queue design assumes exactly one
  consumer of the Redis list.
- **`common/` never imports from `services/`.**
- **NDIF does not use Ray Serve.** Deployments are detached Ray actors; nothing in
  `src/` imports `ray.serve`. Ignore every Serve page a search hands you.

## Extending it

Three recipes, each with its own page. The shape they share: **prefer subclassing
an existing seam to adding a branch.**

- **A model actor** — `run()` is a template; a subclass overrides five hooks
  (`execute`, `execution_scope`, `interrupt`, `format_error`, `cleanup`) plus the
  `commit()` seam, and inherits everything else. The sandbox actor is the worked
  example. `docs/developing/adding-a-model-actor.md`.
- **A provider** — a classmethod singleton over an external system, with a
  `CONFIG` spec of `attr: (ENV_VAR, default, cast)` tuples and deliberate
  fail-open behaviour. `docs/developing/adding-a-provider.md`.
- **A service** — `NDIF_SERVICE` + a `start.sh` + a `Service` in
  `cli/service.py`, plus a `[tool.setuptools.package-data]` entry or it works from
  a checkout and breaks from a wheel. `docs/developing/adding-a-service.md`.

The hooks, the contract a custom actor must uphold, and the CLI-command recipe
are in [references/extending.md](references/extending.md).

## Testing

**No workflow runs the tests, and the suite that matters needs a live server.**

```bash
pip install -e ".[dev]"
just up && just logs api ray      # let Ray settle — it needs a GPU and is slow
pytest tests/                     # NOT a bare `pytest` — there is no testpaths
```

Three files drive the real nnsight client against a running NDIF and **skip
themselves** if nothing answers at `http://localhost:8001`; four are pure Python
(placement arithmetic, the replica wait, the fan-out barrier, the node registry)
and run anywhere. You do not need to deploy anything first — the suite uses gpt2
and the queue provisions on demand. That split is deliberate: everything else
NDIF does is a cross-process, cross-container interaction, and mocking it proves
nothing.

**A skipped suite is not a passing suite.** With no server, `pytest tests/`
reports all-skipped and exits 0.

### Forcing the untrusted path

**A normal `just up` never runs a single line of sandbox code.** Two independent
things have to be true:

1. **The request must be untrusted.** Auth-off defaults an *unspecified* `trusted`
   to `True`, but `validate_request` honors an explicit one (it checks
   `model_fields_set`), so **sending `trusted: false` forces the sandbox path with
   no Postgres and no code patch**. nnsight's `RequestModel` has no `trusted`
   field, so it has to be injected into the envelope —
   `tests/conftest_untrusted.py` does exactly that in about ten lines:

   ```bash
   PYTHONPATH=tests pytest tests/test_nnsight_remote.py -p conftest_untrusted
   ```

2. **The deployed actor class must be the sandbox one.** The code default is the
   in-process `ModelActor`; only compose sets `SandboxModelActor`. Resolution
   order is `NDIF_MODEL_IMPORT_PATH` → `NDIF_DEFAULT_MODEL_ACTOR_CLASS` → the base
   actor.

Confirm which path a request took by watching for a runner subprocess:

```bash
docker compose -f docker/docker-compose.yml exec ray \
    pgrep -af 'ndif.services.ray.sandbox.runner'
```

The faithful alternative, if you want the real trust plumbing: set
`NDIF_POSTGRES_URL`, insert a user and a key, and do *not* grant that key the
`trusted` tag. That also exercises the 401/400/403/503 ladder, which nothing else
does.

The suite's convention for a server limitation is a class-level
`@pytest.mark.xfail(reason=..., strict=False)` — `strict=False` is load-bearing,
because the day the server gains support pytest reports an **XPASS**, and the
marker should be deleted in the same change that made it pass. There are
currently no xfail classes.

## How nnsight is wired in

nnsight is an **ordinary pinned dependency** — `nnsight>=0.8.0rc1,<0.9` from
PyPI, in `requirements.txt`. The specifier names the pre-release explicitly
because pip only considers pre-releases when the specifier does; loosening it to
`nnsight<0.9` resolves to the newest *stable* release instead, i.e. an 0.7 server
against 0.8 clients.

The coupling is deep, not incidental: `BackendRequestModel` subclasses nnsight's
`RequestModel`, `BackendResponseModel` subclasses its `ResponseModel`, and
`Status` **is** nnsight's `Status` — so **you cannot add a status server-side.**
A new member has to land in nnsight, ship in a release, and be installed by the
user before the server may emit it. To convey something new without a client
release, put it in `description` on an existing status, or emit it as a `LOG`.

For local work, `pip install -e /path/to/nnsight` **before** installing ndif (the
ndif install resolves its nnsight requirement against whatever is present).
`just up` / `just ta` then bind-mount that editable checkout over the image's
copy in `api`, `ray` and `dashboard`, so client-side changes are picked up
without a rebuild; a non-editable nnsight is deliberately not mounted.
`just nnsight` prints the decision. To run against an unreleased fix, replace the
`requirements.txt` line with a git ref — the image carries `git` for that.

When you bump it, check in order: does a request still deserialize; does `Status`
still have every member the server emits; do `_saves()`, `inc()` and `dec()` still
exist with these semantics (they are underscore-private upstream); does the
sandbox still interleave; do model keys still round-trip (a change to
`to_model_key`'s format orphans every `models.yaml` and every pinned
`NDIF_DEPLOYMENTS` entry). Then set `NDIF_MIN_NNSIGHT_VERSION` to the oldest
client that still works.

## Releasing

Three workflows, **all of which publish and none of which test**:

| Workflow | Trigger | Publishes |
|---|---|---|
| `build_images.yml` | push to `main` | ECR images |
| `publish_docker.yml` | a `v*` tag | Docker Hub images (one per CUDA line) |
| `publish.yml` | a GitHub release | the PyPI wheel |

So **a `v*` tag is a release, not a checkpoint**, and a green
`publish_docker.yml` means the image built, not that the server works. The
version lives in `pyproject.toml`; bump it, tag `v<version>`, then cut the GitHub
release for PyPI.

House style, since there is no lint gate and no PR template: the **module
docstring is where the concept gets taught** (the constraint first, then the
mechanism); comment the *why*, naming the failure mode; **present tense only** —
no "this used to", no `TODO`/`FIXME`, no comment about the change you are making;
commits are `area: lowercase imperative summary` with a body that names the
counterfactual. `docs/developing/contributing.md`.

Before a PR: bring a stack up and run `pytest tests/`; if you touched execution,
run it again with `trusted=False` forced; `ruff check src/`; and if you added an
env var, add the row to the README table **and** `docs/reference/env-vars.md` —
that is the most common drift in the repo.

## References

- [references/request-lifecycle.md](references/request-lifecycle.md) — the ten
  hops with what breaks at each, the queue objects, and where state lives.
- [references/extending.md](references/extending.md) — the five model-actor
  hooks and the contract around them, the provider pattern, adding a service,
  and adding a CLI command.

## Related skills

- `ndif-selfhost` — getting a dev stack up.
- `ndif-troubleshoot` — reading the logs a change produces.
- `ndif-operate` — the operator surface your change has to keep working.
