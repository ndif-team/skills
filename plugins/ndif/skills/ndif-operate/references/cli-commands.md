# The `ndif` CLI, command by command

`ndif` is the only binary the repo ships. It does three unrelated jobs, and
knowing which one a command belongs to is most of the battle:

1. **Process lifecycle on one host** — `start`, `stop`, `logs`, `info`, `doctor`,
   `version`. With compose you use `just` instead, except *inside* a container
   where `ndif start` is already the entrypoint.
2. **Model control-plane ops** — `deploy`, `scale`, `evict`, `restart`, `status`,
   `export`. They talk to the Ray controller actor from anywhere that can reach
   `NDIF_RAY_ADDRESS`.
3. **Queue introspection** — `queue`, `kill`. These reach the API's dispatcher
   over a Redis stream, not HTTP.

Source of record: `docs/operating/cli.md`.

## Where to run them

| Situation | Use |
|---|---|
| Dev on one machine, no docker | `ndif start` / `stop` / `logs` |
| Dev with the compose stack | `just up` / `just logs api` / `just down` — **not** `ndif start` |
| Inside a container | `ndif start --foreground` is already the default command; `NDIF_SERVICE` picks the role |
| Asking an image what it carries | `docker run --rm ndif/ndif version` |
| Adding a GPU worker | `ndif start --ray-head-address HEAD:6385` (brings up only `ray`) |
| Any cluster, managing models | `ndif deploy` / `scale` / `evict` / `restart` / `status` / `export` |
| Any cluster, debugging traffic | `ndif queue` / `ndif kill` / `ndif env` |

`just` and `ndif start` track different things and cannot see each other: one
manages containers, the other local processes.

## `ndif deploy`

```text
ndif deploy [CHECKPOINTS...] [-f FILE] [--sync] [--revision REV] [--pinned]
            [--replicas N] [--actor-class PATH] [--trusted] [--dtype DTYPE]
            [--gpus N] [--size-bytes N] [--padding-factor F] [--padding-bias N]
            [--max-tp N] [--ray-address ADDR] [--redis-url URL]
```

| Option | Default | Effect |
|---|---|---|
| `CHECKPOINTS...` | — | HF repo ids, one spec each; mutually exclusive with `--sync` |
| `-f/--file` | — | a YAML `models:` list |
| `--sync` | off | reconcile the cluster to the file exactly; requires `-f` |
| `--revision` | unset | HF branch/revision — part of the model key, so it changes identity |
| `--pinned` | off | exempt from automatic eviction |
| `--replicas` | `1` | how many **new** replicas to add |
| `--actor-class` | `NDIF_DEFAULT_MODEL_ACTOR_CLASS` | the Ray actor class serving the deployment |
| `--trusted` | off | load with `trust_remote_code=True` |
| `--dtype` | `NDIF_DEFAULT_DTYPE` (`bfloat16`) | a torch dtype or a quantization name (`nf4`/`int4`/`4bit`, `fp4`, `int8`/`8bit`, `fp8`); loads **and** sizes |
| `--gpus` | derived | place on exactly this many cards |
| `--size-bytes` | estimated | the model's weights, measured — skips the Hub estimate |
| `--padding-factor` | `NDIF_DEFAULT_PADDING_FACTOR` | proportional headroom for this model |
| `--padding-bias` | `NDIF_DEFAULT_PADDING_BIAS` | flat headroom for this model |
| `--max-tp` | from the checkpoint's config | largest TP degree; `0` never places it tensor-parallel |

One call: resolves a model key per spec (a Hub round-trip), connects to Ray,
optionally reconciles for `--sync`, asks the controller to deploy in two batches
(non-pinned then pinned), blocks per replica on `__ray_ready__` **with no
deadline**, then fires a best-effort Redis reconcile nudge so live dispatcher
Processors refresh their replica pools.

There is no deadline on purpose: a model that must download weights sits there a
while, and only two conditions mean "not yet" (the actor is unregistered, or
restarting). Everything else — including a constructor that raised — propagates
as itself rather than as a timeout.

Per-model status ends `READY`, `PARTIAL` (some replicas failed) or `ERROR`;
evictions the controller performed to make room are listed at the end.

### `trusted` on a deploy

It is not a label. It becomes `trust_remote_code=` for both the size evaluator
and the actor's load, so **an untrusted deploy of a model whose HF repo ships
custom modelling code fails** — in the evaluator, before any GPU is touched, with
`✗ <model_key>: <traceback>` and nothing placed. Architectures built into
transformers (GPT-2, Llama, Qwen) load fine untrusted; anything with `"auto_map"`
in its `config.json` does not.

The same field on a *request* decides where user code runs. A deployment
inherits `trusted` from whoever created it, and a later trusted request against
that model does **not** reload it.

## `ndif scale`

```text
ndif scale CHECKPOINT [-n N] [--revision REV] [--actor-class PATH] [--dtype DTYPE]
           [--gpus N] [--execution-timeout SECONDS] [--trusted] [--pinned]
```

Adds replicas that look like the ones already serving that model. Anything you
pass is used as-is *and* decides which live replica counts as a match to copy the
rest from, so `ndif scale gpt2 --gpus 1` copies the single-GPU replica rather than
a sharded one. With nothing running there is nothing to copy and it behaves as a
plain deploy.

## `ndif evict`

```text
ndif evict [CHECKPOINTS...] [--revision REV] [--replica ID] [--all]
```

Without `--replica` the controller removes **every HOT and WARM replica** of the
model key. `--replica` requires exactly one checkpoint. `--all` targets every
currently-HOT model key and cannot be combined with the others.

Eviction is unconditional: it respects neither `pinned` nor the age rule, which
govern only what the controller does on its own initiative. Because `--all` reads
the HOT set, a model that is only WARM is not a target — name it explicitly.

Evicting a HOT replica frees its GPU bytes and demotes it to WARM if the node's
CPU cache budget has room (greedily dropping smaller WARM entries to make it);
with no room the replica is removed outright and the next request pays a full
disk load.

## `ndif restart`

```text
ndif restart CHECKPOINT [--revision REV] [--replica ID]
```

Kills each replica's actor with `no_restart=False` and waits for Ray to respawn
it (`max_restarts=-1`). Use it to drop cached state, reload weights, or recover a
wedged replica without giving up its GPU placement. It sends **no** reconcile
event. Unrelated to `ndif start --restart`, which restarts local *services*.

## `ndif status`

```text
ndif status [--json-output] [--verbose] [--show-cold] [--watch]
```

`--json-output` prints the controller payload; each entry carries `model_key`,
`replica_id`, `deployment_level`, `application_state`, `pinned`, `trusted`,
`n_params`, `size_bytes`, and — for non-pinned deployments —
`schedule.end_time`, the moment age protection lapses. `--verbose` fetches
`get_state()` instead: per-node GPU inventory, per-replica placement, the
evaluator's size cache.

Read the two state columns separately. `deployment_level` (HOT/WARM/COLD) is the
controller's **bookkeeping**; `application_state` is the **Ray actor's** state —
`ALIVE`→`RUNNING`, `PENDING_CREATION`/`RESTARTING`/`DEPENDENCIES_UNREADY`→`DEPLOYING`,
`DEAD`→`UNHEALTHY`. HOT + DEPLOYING means weights are still loading; HOT +
UNHEALTHY repeatedly means the actor dies during load and Ray keeps restarting it.

COLD is synthesized for reporting by scanning the node's local HF cache.

## `ndif export`

```text
ndif export (-f FILE | --stdout)
```

Collapses the current HOT per-replica list into one entry per model key and
writes a `models.yaml` you can feed back to `ndif deploy -f`. `trusted`, `dtype`,
`execution_timeout_seconds`, `pinned`, `replicas`, `actor_class` and `model_key`
survive the round trip; **`padding_factor` does not**.

## `ndif queue` and `ndif kill`

```text
ndif queue [--json-output] [--watch]
ndif kill REQUEST_ID
```

`queue` snapshots every live Processor: lifecycle status (`uninitialized`,
`provisioning`, `deploying`, `ready`, `cancelled`), the replicas in its pool and
whether each is busy, queue depth, and the first few queued request ids. **A
model with no in-flight work has no Processor**, so an idle cluster prints
`No active processors.` even with HOT deployments.

`kill` removes the request from its Processor's queue if it is still waiting
(the client gets `ERROR` / "Request cancelled by operator."), otherwise cancels
the replica's worker task ("Replica was evicted while processing your request.").

Both are Redis-stream round trips with a 5 s timeout;
`No response from the dispatcher` means the API is down, not Redis.

## `ndif env`, `ndif version`, `ndif doctor`, `ndif info`

- `ndif env` fetches `GET {NDIF_API_URL}/env` — the cluster's Python version and
  installed packages, from a TTL'd cache (300 s), filtered to a fixed key list
  unless `--all`. `ndif env --local` reports this machine instead. The pair is the
  fastest way to spot client/server nnsight drift.
- `ndif version` prints the resolved stack of *this* install: python, ndif,
  nnsight, torch, transformers, ray, peft, accelerate, and `cuda` (the CUDA line
  the installed torch wheel was built for). `--write PATH` dumps it as JSON; the
  image runs exactly that at build time into `/etc/ndif/build.json`.
- `ndif doctor` probes local prerequisites: Python >= 3.12, `ndif`/`nnsight`
  installed, the `ray`/`redis-server`/`minio` binaries, `nvidia-smi`, and
  connectivity (informational — never sets the exit code).
- `ndif info` prints `NDIF_HOME`, the tracked PID of each core service, and a
  reachability probe per endpoint — the fastest way to find out that a variable
  you thought you set never reached the process. The dashboard is not listed even
  when tracked.

## Gotchas

- **`--replicas N` means "add N", not "have N"**, even under `--sync` (which
  computes the shortfall first).
- **`ndif deploy` needs Hub access** to canonicalize the checkpoint, and
  `HF_TOKEN` for a gated repo, wherever the deploy runs.
- **`pinned` does not stop you.** `ndif evict --all` sweeps pinned deployments.
- **`ndif stop` only knows about processes it started.** It cannot stop a compose
  stack or reap a Ray head you launched by hand.
- **`NDIF_SERVICE` is dual-purpose**: it selects which services `ndif start`
  launches *and* becomes the `service` label on every log line and metric, so a
  multi-service value appears verbatim in your logs.
