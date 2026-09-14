# Extending NDIF

Source of record: `docs/developing/adding-a-model-actor.md`,
`adding-a-provider.md`, `adding-a-service.md`, `cli-internals.md`,
`repo-layout.md`.

## I want to change X, open Y

| I want to change... | Open |
|---|---|
| An HTTP endpoint, or what a request looks like at ingress | `src/ndif/services/api/app.py` |
| Who is allowed in, or how `trusted` is decided | `src/ndif/services/api/auth.py` |
| Queue behaviour, batching, autoscaling triggers | `src/ndif/services/api/queue/{dispatcher,processor,replica,config}.py` |
| Where replicas get placed, or eviction policy | `.../controller/cluster/{cluster,node}.py` |
| GPU footprint estimation | `.../controller/cluster/evaluator.py` |
| Ray actor lifecycle (create / delete / HOT↔WARM) | `.../controller/cluster/deployment.py` |
| How a model loads, or the per-request run template | `.../deployments/modeling/base.py` |
| The trusted/untrusted execution fork | `src/ndif/services/ray/sandbox/model.py` |
| The sandbox wire protocol | `src/ndif/services/ray/sandbox/protocol.py` (+ `ARCHITECTURE.md`) |
| An `ndif` subcommand's flags | `src/ndif/cli/commands/<verb>.py` |
| What an `ndif` subcommand *does* | `src/ndif/cli/lib/<verb>.py` |
| Which services `ndif start` knows about | `src/ndif/cli/service.py` |
| A default port or URL for single-host runs | `src/ndif/cli/config.py` (`DEFAULTS`) |
| A connection to an external system | `src/ndif/common/providers/<name>.py` + a pyproject extra |
| A Redis key, channel, or stream name | `src/ndif/common/redis/{env,events,status}.py` |
| What a log line carries | `src/ndif/common/telemetry.py`, `logging_setup.py` |
| A metric's tags or fields | `src/ndif/common/metrics.py` |
| The dev stack's topology, ports, or env | `docker/docker-compose.yml` |
| What is installed in the image | `requirements.txt` (pins) + `pyproject.toml` (extras) |

Where new code belongs: used by two or more services → `common/`; a connection to
an external system → `common/providers/` **plus a pyproject extra**; a wire field
→ `common/schema/`; specific to one service → `services/<name>/`; a user-facing
verb → `cli/commands/` (flags) plus `cli/lib/` (logic).

Two hard rules: **`common/` never imports from `services/`**, and **CLI logic
lives in `lib/`** — `commands/` holds click decoration only, because the
dashboard backend imports `lib/` directly and would otherwise have no door in.

## Adding a model actor

Keep the split the tree already uses: the *deployment* class holds the logic, the
`@ray.remote` subclass is a one-line wrapper.

<!-- test: skip -->
```python
# src/ndif/services/ray/mystack/model.py
import ray
from ..deployments.modeling.base import BaseModelDeployment


class MyModelDeployment(BaseModelDeployment):
    """Plain-Python behaviour; unit-testable without Ray."""


@ray.remote(num_cpus=1, max_restarts=-1)
class MyModelActor(MyModelDeployment):
    """The deployable actor."""
```

Keep `max_restarts=-1` unless you want a replica that never comes back:
`BaseModelDeployment.restart()` kills the actor with `no_restart=False` after a
CUDA-context-corrupting failure and relies on Ray to bring it back.

`__init__` must accept what the controller passes — `model_key`,
`execution_timeout`, `gpu_mem_bytes_by_id`, `dtype`, plus `**kwargs` (which is
how `trust_remote_code` arrives) — and call `super().__init__()`. **An extra
constructor parameter of your own has no path from the controller**: read it from
the environment (the actor's `runtime_env` already carries the provider config),
or add a field to `BaseModelDeploymentArgs`, which also means teaching
`DeploymentConfig` and the controller to populate it.

### The five hooks

`run()` is a template method. Everything a different actor needs to change is in
five methods it calls:

| Hook | Base behaviour | Override when |
|---|---|---|
| `execute(request) -> (bytes, float\|None)` | Deserialize against `self.model`'s persistent objects, run the block under autocast, collect `nnsight.save()` values, `torch.save` them | The block should run anywhere other than this thread/process |
| `execution_scope(request)` | Context manager around the raced wait; redirects `sys.stdout` **and** `sys.stderr` into a `LogStream` each, so prints and `warnings.warn` both become `LOG` responses | Your executor reports output another way |
| `interrupt()` | `kill_thread(self.execution_ident)` | There is something else to stop — a subprocess, a socket, an engine request |
| `format_error(exc) -> (str, bool)` | Clean nnsight and actor frames out of the traceback; flag unrecoverable CUDA errors as fatal | Your errors arrive pre-formatted, or a different class is fatal |
| `cleanup()` | Clear the kill switch and execution ident, cancel the interleaver, `synchronize`/`gc`/`empty_cache` | You hold per-request resources — call `super().cleanup()` |

A sixth seam sits one level down: `commit()` is called by the base's `execute`
after the block is built and immediately before it runs — the point at which a
request has stopped being able to fail on a bad payload. It does nothing in the
base; `TPModelDeployment.commit` is the one that uses it, releasing the other
ranks into the forward and arming the abort checkpoint.

Everything else — `load_from_disk`, `to_cache`/`from_cache`, `cancel`, `restart`,
`report`, `prepare_result`, `upload_bytes`, and `run` itself — should be
inherited unchanged. Overriding `run` means re-implementing its contract, and
every consumer (the queue, the dashboard, Grafana) depends on it.

Select your actor with `NDIF_MODEL_IMPORT_PATH` cluster-wide (falling back to
`NDIF_DEFAULT_MODEL_ACTOR_CLASS`), or per deployment with `actor_class` in
`models.yaml` / `ndif deploy --actor-class`.

## Adding a provider

A provider is a classmethod singleton over an external system, driven by a
`CONFIG` spec of `attr: (ENV_VAR, typed_default, cast)` tuples that the base
turns into `from_env()` / `to_env()` automatically.

The discipline that matters is **fail-open**: an optional subsystem's URL
defaults to `""`, and an empty URL disables the whole thing silently rather than
erroring. Don't add a separate `_ENABLED` boolean unless you need to disable
something that is otherwise configured (`NDIF_INFLUX_ENABLED` is the one case).

The deliberate exception is Postgres: set the URL without `asyncpg` installed and
`connect` **raises**, because "auth silently off" would be a security hole
whereas "no metrics" is harmless.

Two more constraints:

- **Add a pyproject extra** for the new dependency, and keep the code working
  without it.
- **Telemetry providers connect at import and own threads**, and threads do not
  survive `fork()`. Connect them *after* forking — `post_fork` for gunicorn
  workers, `spawn` for the dispatcher, `__init__` for Ray actors.

## Adding a service

The contract is three things plus packaging: a `start.sh` in the service package,
a `Service` entry in `cli/service.py` (which is what makes `ndif start`, `stop`
and `logs` pick it up), and a `[tool.setuptools.package-data]` entry in
`pyproject.toml` — a `start.sh` setuptools leaves out of the wheel means
`ndif start <service>` works from a checkout and fails with "cannot run bash"
from a wheel. Then a compose block if it belongs in the dev stack.

Decide up front whether it is core or opt-in: `SERVICES` is what a bare
`ndif start` brings up; `OPTIONAL_SERVICES` (currently just `dashboard`) must be
named explicitly.

## Adding a CLI command

1. `src/ndif/cli/commands/<verb>.py` with a single `@click.command()` function
   named after the verb, opening with `"""``ndif <verb>`` — purpose."""`.
2. Put the logic in `lib/` if it touches Ray, Redis or the controller, and give it
   an `on_message` callback.
3. **Import Ray / nnsight / redis inside the function**, not at module scope, so
   `ndif --help` stays fast.
4. Register it in `main.py` — the import and the `add_command` tuple. Nothing else
   registers commands.
5. Follow the option conventions: `--ray-address` defaults to `NDIF_RAY_ADDRESS`,
   `--redis-url` to `NDIF_REDIS_URL`, `--api-url` to `NDIF_API_URL`, all
   documented as `(default: NDIF_*)` in the help text. Read-only commands take
   `--json-output`; anything worth polling takes `--watch` (2 s loop,
   `click.clear()`, `KeyboardInterrupt` → clean return).
6. Give the function a docstring with an `\b`-escaped `Examples:` block — click
   renders it verbatim, and it is the only usage documentation most people read.
7. Handle errors the house way: catch `NDIFConnectivityError` and `Exception`,
   `click.echo(f"✗ Error: {e}", err=True)`, then `raise click.Abort()`.
8. Update `docs/operating/cli.md`.

If the verb manages a *process* rather than a model, add a `Service` instead — a
command is not the right shape.

## Small traps in the CLI

- `lib/events.py` constructs its Redis client with `socket_timeout=None` on
  purpose: redis-py 8.0+ otherwise applies a 5 s socket timeout against Redis 8,
  which would abort the blocking `brpop` before the dispatcher replies.
- `notify_reconcile` swallows every exception — a deploy that succeeded on the
  controller must not fail because Redis hiccuped, but a silently-missed reconcile
  leaves a live Processor with a stale replica pool until its next refresh.
- `ndif info` iterates `SERVICES`, not `SERVICE_MAP`, so the dashboard never
  appears in its output even when a PID file is tracking it.
- There are **no unit tests for the CLI**. The only suite is the live-server one.
