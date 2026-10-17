# Extending NDIF

Source of record, each a page in the ndif repo:
`docs/developing/adding-a-model-actor.md`,
`docs/developing/providers.md#writing-a-new-provider`,
`docs/developing/adding-a-service.md`, `docs/developing/cli-internals.md`, and
`docs/developing/repo-layout.md` for the full "I want to change X, open Y" table.

Where new code belongs: used by two or more services → `common/`; a connection to
an external system → `common/providers/` **plus a pyproject extra**; a wire field
→ `common/schema/`; specific to one service → `services/<name>/`; a user-facing
verb → `cli/commands/` (flags) plus `cli/lib/` (logic). Two hard rules:
**`common/` never imports from `services/`**, and **CLI logic lives in `lib/`** —
the dashboard backend imports `lib/` directly.

## Adding a model actor

Subclass `BaseModelDeployment` for the logic and wrap it in a one-line
`@ray.remote(num_cpus=1, max_restarts=-1)` actor; keep `max_restarts=-1`, which
`restart()` relies on. Skeleton, constructor arguments and a worked example:
`docs/developing/adding-a-model-actor.md`. **A constructor parameter of your own
has no path from the controller** — read it from the environment, or add a field
to `BaseModelDeploymentArgs` and teach `DeploymentConfig` and the controller to
populate it.

`run()` is a template method. A different actor changes **six hooks** it calls —
`execute`, `execution_scope`, `interrupt`, `format_error`, `error_name`,
`cleanup` — plus `commit()` (between "block built" and "block runs"; the TP actor
releases its other ranks there) and `restart()` (release anything that would
outlive the actor, then `super()`). `SandboxHost` adds `before_payload` and
`after_payload` around sending the block to a runner. Everything else, `run`
included, should be inherited; the seam table is in
`docs/developing/adding-a-model-actor.md#the-seams`, and
`SandboxModelDeployment` is the reference implementation to copy.

### The contract

Replace `execute` only and you get this for free; go further and keep it
(`docs/developing/adding-a-model-actor.md#the-contract-you-must-uphold`):

1. **Exactly one terminal response per request**, or the client's websocket hangs.
2. **`execute` returns `(blob, deserialize_ms)`**; `run` picks the inline or
   object-store route.
3. **Saved values are `nnsight.save()`-marked locals**; reuse the shared
   `cpu_pickle_module` so the blob format is identical.
4. **Be interruptible.** `run` calls `interrupt()` and returns *without waiting*;
   an executor that blocks outside Python must be unblockable or the thread leaks.
5. **Set `self.execution_ident` on the worker thread before anything can block.**
   `cancel()` is gated on it — without it, a cancel or a preempt is **silently
   dropped**.
6. **Raise `CachedActorError` when WARM**, so the queue re-queues.
7. **Do not let request state outlive the request.**

### Selecting it

First match wins, chosen at placement:

| Source | Scope |
|---|---|
| `DeploymentConfig.actor_class` (`ndif deploy --actor-class`, `models.yaml`, the dashboard) | one deploy call |
| `NDIF_TP_MODEL_ACTOR_CLASS` | a replica placed tensor-parallel — the class is **swapped**, so a cluster taking untrusted traffic wants `SandboxedTPModelActor` |
| `NDIF_MODEL_IMPORT_PATH` | the whole cluster |
| `NDIF_DEFAULT_MODEL_ACTOR_CLASS` | legacy fallback for the above (what compose sets) |
| `ndif.services.ray.deployments.modeling.base.ModelActor` | built-in, in-process |

The path is imported inside the Ray worker, so it must be importable there. A
changed default affects only **new** deployments; evict and re-deploy to move one.

## Adding a provider

A classmethod singleton over an external system, driven by a `CONFIG` spec of
`attr: (ENV_VAR, typed_default, cast)` tuples that the base turns into
`from_env()` / `to_env()`. The discipline is **fail-open**: an optional
subsystem's URL defaults to `""`, and empty disables it silently. The deliberate
exception is Postgres — set the URL without `asyncpg` and `connect` raises,
because "auth silently off" is a security hole.

- **Add a pyproject extra** for the dependency, and keep the code working without it.
- **Telemetry providers connect at import and own threads**, which do not survive
  `fork()`. Connect them after forking — `post_fork` for gunicorn workers, `spawn`
  for the dispatcher, `__init__` for Ray actors.

## Adding a service

A `start.sh` in the service package, a `Service` entry in `cli/service.py`
(which makes `ndif start`, `stop` and `logs` pick it up), and a
`[tool.setuptools.package-data]` entry in `pyproject.toml`. Skip the last and it
works from an editable checkout but fails from any installed copy — **including
the compose image, which installs non-editable** — with
`bash: .../services/myservice/start.sh: No such file or directory` (exit 127).
Then a compose block if it belongs in the dev stack. `SERVICES` is what a bare
`ndif start` brings up; `OPTIONAL_SERVICES` (just `dashboard`) must be named.

## Adding a CLI command

`src/ndif/cli/commands/<verb>.py` with one `@click.command()`, logic in `lib/`
with an `on_message` callback, Ray/nnsight/redis imported **inside** the function
so `ndif --help` stays fast, registered in `main.py` (nothing else registers
commands), the house option and error conventions, and `docs/operating/cli.md`
updated. Steps: `docs/developing/cli-internals.md#adding-a-command`. A verb that
manages a *process* is a `Service`, not a command.

Traps: `lib/events.py` passes `socket_timeout=None` on purpose (redis-py 8+
otherwise times out the blocking wait); `notify_reconcile` swallows every
exception, so a missed reconcile leaves a stale replica pool until the next
refresh; `ndif info` iterates `SERVICES`, so the dashboard never appears there.
Only the model-key plumbing has an offline test (`tests/test_task_key_offline.py`);
the rest of the CLI is exercised through the live suite.
