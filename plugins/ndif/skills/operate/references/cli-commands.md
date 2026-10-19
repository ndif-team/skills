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

## The model key, and `--task`

Every model command resolves a checkpoint to a model key — repo id, revision,
task and `trust_remote_code`: `--task` when given, else the task inferred from
the checkpoint exactly as an nnsight client with no `task=` infers it;
`trust_remote_code` only with `--trust-remote-code`. A deployment made with a
non-default task or with `--trust-remote-code` needs the same flag on every
later `scale`, `evict` and `restart` — without it they mint the key of a
different model. For a key no flag combination reproduces, use the exact-key recipes
at the end of this page.

## `ndif deploy`

```text
ndif deploy [CHECKPOINTS...] [-f FILE] [--sync] [--revision REV] [--task TASK]
            [--trust-remote-code] [--pinned] [--replicas N] [--actor-class PATH]
            [--dtype DTYPE] [--gpus N] [--size-bytes N] [--padding-factor F]
            [--padding-bias N] [--max-tp N] [--ray-address ADDR] [--redis-url URL]
```

| Option | Default | Effect |
|---|---|---|
| `CHECKPOINTS...` | — | HF repo ids, one spec each; mutually exclusive with `--sync` |
| `-f/--file` | — | a YAML `models:` list |
| `--sync` | off | reconcile the cluster to the file exactly; requires `-f` |
| `--revision` | unset | HF branch/revision — part of the model key |
| `--task` | inferred | pipeline task (`text-generation`, `fill-mask`, ...) — part of the model key, so each task is its own deployment |
| `--trust-remote-code` | off | let the checkpoint's own modelling code run at load — part of the model key |
| `--pinned` | off | exempt from automatic eviction |
| `--replicas` | `1` | how many **new** replicas to add |
| `--actor-class` | the controller's (`NDIF_MODEL_IMPORT_PATH`, then `NDIF_DEFAULT_MODEL_ACTOR_CLASS`) | the Ray actor class serving the deployment |
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
restarting). A constructor that raised propagates as itself rather than as a
timeout — normally as `initialization failed - <error>`. If the CLI is still
waiting minutes after the controller log shows `failed during create`, the actor
is already gone; interrupt it and read the log.

Each model gets one status line, e.g. `✓ <model_key> [b7f60]: ready` (the
bracket is the replica id), or a `✗` line with the error; evictions the
controller performed to make room are listed at the end.

### `trust_remote_code`

A field of the model key, written as `"trust_remote_code": true` in the key's
JSON and omitted otherwise. nnsight's `from_model_key` reads it for both the
size evaluator and the actor's load, so **a deploy without it of a model whose
HF repo ships custom modelling code fails** — in the evaluator, before any GPU is
touched, with `✗ <model_key>: <traceback>` and nothing placed. Architectures
built into transformers (GPT-2, Llama, Qwen) load without it; anything with
`"auto_map"` in its `config.json` does not. A client that built the model with
`trust_remote_code=True` sends a key carrying the field, and whether that
client may deploy it is the API's decision (`docs/concepts/auth-and-limits.md`).

The per-request `trusted` flag is unrelated: it decides where user code runs
(on a sandbox actor class), not how the model loads.

## `ndif scale`

```text
ndif scale CHECKPOINT [-n N] [--revision REV] [--task TASK] [--actor-class PATH]
           [--dtype DTYPE] [--gpus N] [--execution-timeout SECONDS]
           [--trust-remote-code] [--pinned] [--ray-address ADDR]
```

Adds replicas that look like the ones already serving that model. Anything you
pass is used as-is *and* decides which live replica counts as a match to copy the
rest from, so `ndif scale gpt2 --gpus 1` copies the single-GPU replica rather than
a sharded one. With nothing running there is nothing to copy and it behaves as a
plain deploy.

When a WARM replica of the key exists, `scale` re-promotes it under its old
replica id and settings rather than placing a fresh one. `pinned` is not copied
on this path — it defaults to false — so `scale` is not a way to pin.

### Pinning a model already deployed

`pinned` is set only when a replica is created, and `deploy --pinned` is
additive: it places a second, pinned replica beside the unpinned one. To pin in
place:

```bash
ndif deploy <checkpoint> --pinned            # new pinned replica
ndif evict <checkpoint> --replica <old_id>   # drop the unpinned one
```

## `ndif evict`

```text
ndif evict [CHECKPOINTS...] [--revision REV] [--task TASK] [--trust-remote-code]
           [--replica ID] [--all] [--ray-address ADDR] [--redis-url URL]
```

Without `--replica` the controller removes **every HOT and WARM replica** of the
model key. `--revision` and `--task` apply to every checkpoint given; `--replica`
requires exactly one checkpoint. `--all` targets every currently-HOT model key and
cannot be combined with the others.

Eviction is unconditional: it respects neither `pinned` nor the age rule, which
govern only what the controller does on its own initiative. Because `--all` reads
the HOT set, a model that is only WARM is not a target — name it explicitly.

Evicting a HOT replica frees its GPU bytes and demotes it to WARM if the node's
CPU cache budget has room (greedily dropping smaller WARM entries to make it);
with no room the replica is removed outright and the next request pays a full
disk load.

A key that matches nothing prints `✗ <model_key>: nothing to evict` and **exits
0** — usually a revision or task mismatch, or a WARM-only model.

## `ndif restart`

```text
ndif restart CHECKPOINT [--revision REV] [--task TASK] [--trust-remote-code]
             [--replica ID] [--ray-address ADDR]
```

Kills each replica's actor with `no_restart=False` and waits for Ray to respawn
it (`max_restarts=-1`) and reload the weights. Use it to drop cached state,
reload weights, recover a wedged replica, or clean up after `ndif kill` of an
executing request whose block is still running, without giving up its GPU placement. Requests sent to that
replica during the restart are re-queued or errored. It sends **no** reconcile
event. `✗ <model_key>: no HOT replicas to restart` also exits 0. Unrelated to
`ndif start --restart`, which restarts local *services*.

## `ndif status`

```text
ndif status [--json-output] [--verbose] [--show-cold] [--watch] [--ray-address ADDR]
```

The plain form prints cluster totals (`GPU Memory: X / Y GB free` — the ledger,
not `nvidia-smi`) and the deployments by level. `--json-output` prints the
controller payload; each entry carries `model_key`, `replica_id`,
`deployment_level`, `application_state`, `task`, `pinned`, `trust_remote_code`
(read off the key), `dtype`,
`execution_timeout_seconds`, `actor_class`, `gpus`, `n_params`, `size_bytes`,
and — for non-pinned deployments — `schedule.end_time`, the moment age protection
lapses. `--verbose` fetches `get_state()` instead: per-node GPU inventory,
per-replica placement, the evaluator's size cache.

Read the two state columns separately. `deployment_level` (HOT/WARM/COLD) is the
controller's **bookkeeping**; `application_state` is the **Ray actor's** state —
`RUNNING`, `DEPLOYING` or `UNHEALTHY`. HOT + DEPLOYING means weights are still
loading; HOT + UNHEALTHY repeatedly means the actor dies during load and Ray
keeps restarting it.

COLD is synthesized for reporting by scanning the node's local HF cache.

## `ndif export`

```text
ndif export (-f FILE | --stdout) [--ray-address ADDR]
```

Collapses the current HOT per-replica list into one entry per model key and
writes a `models.yaml` you can feed back to `ndif deploy -f`. Each entry carries
the full `model_key` (task and envoy class included, used verbatim on redeploy)
plus `dtype` and `actor_class`, **always written**, and `revision`, `pinned`,
`replicas`, `trust_remote_code`, `execution_timeout_seconds` and `gpus` (when
more than one) **only where they differ from the defaults**.

What it loses:

- **`padding_factor`, `padding_bias`, `size_bytes` and `max_tp`** — deploy-time
  sizing inputs never stored on the deployment. A restored model is re-sized from
  the cluster defaults unless you add them back.
- **Per-replica differences** — replicas of one key with different settings are
  merged into one entry; if any replica was pinned, the entry is `pinned: true`.

So run `ndif status --json-output` first and keep it beside the file. `-f`
writes where the CLI runs — inside the `ray` container on compose — so use
`docker exec dev-ray-1 ndif export --stdout > models.yaml` to land it on the host.

## `ndif queue` and `ndif kill`

```text
ndif queue [--json-output] [--watch] [--redis-url URL]
ndif kill REQUEST_ID [--redis-url URL]
```

`queue` snapshots every live Processor: lifecycle status (`uninitialized`,
`provisioning`, `deploying`, `ready`, `cancelled`), the replicas in its pool and
whether each is busy (`⚙ [replica] executing <request> (for 0:00:07)`), queue
depth, and the first few queued request ids. A Processor is created by the
first request for a model and **persists when idle**, so a model with nothing in
flight still shows a `ready` processor with an empty queue; a model never
requested since the API started has none. After an eviction a processor can
show a stale `Replicas: N` for a while — the pool drops evicted replicas lazily,
on its next dispatch.

`kill` first scans every Processor's queue:

- **Still queued** — removed; the client gets `ERROR` "Request cancelled by
  operator." The CLI prints `✓ Request <id> removed from the queue.`
- **Executing** — the CLI prints `✓ Request <id> cancelled while executing.`
  Only the **dispatch** is cancelled: the model actor is not interrupted and runs
  the block to completion.
  - A **blocking** client gets one `ERROR` status, "Request cancelled by
    operator.", with an empty description line. The log shows one
    `ndif.request` ERROR and a `ndif.queue.replica` WARNING
    `request errored: cancelled mid-dispatch`.
  - A **non-blocking** client raises
    `Replica was evicted while processing your request.`; the stored response
    shows that until the actor's own final response overwrites it (often
    `COMPLETED`).
- **Unknown id** — exits non-zero with a pointer to `ndif queue`.

While the killed block is still running, the processor can hand the same
replica new work, and that work fails with
`ValueError: Cannot switch to a different thread` or
`ValueError: parent cannot be on a different thread` — so **restart the replica
while the block runs** (`ndif restart <checkpoint> --replica <id>`, the id from
the `⚙` line). Once the block finishes on its own — Loki
`{service="model"} |= "<request id>"` shows `Your job has been completed.` —
the next request is clean with no restart.

Both are Redis-stream round trips with a 5 s timeout.
`No response from the dispatcher` means the API is down, not Redis — or that the
API restarted under five seconds ago and the dispatcher is still coming up.

## Exact-key operations

`evict`, `restart` and `scale` act on the inferred task's key. For a key with no
task, another task, or anything else the flags cannot reproduce, take the exact
`model_key` from `ndif status --json-output` and go through the library the CLI
wraps:

<!-- test: skip -->
```python
from ndif.cli.lib.restart import restart
restart(model_key="<model_key>", replica="<replica_id>", on_message=print)

from ndif.cli.lib.evict import evict
evict(model_keys=["<model_key>"], replica="<replica_id>", on_message=print)
```

Run it where Ray is reachable (`docker compose -f docker/docker-compose.yml exec
ray python -c "..."` on compose). Both print the same `✗` lines and exit 0 when
the key matches nothing.

Or use the dashboard's JSON API, which takes the key verbatim:

```bash
D=http://localhost:8081
curl -s -c jar -H 'content-type: application/json' \
  -d '{"username":"admin","password":"<password>"}' $D/api/auth/login   # skip in dev mode
curl -s -b jar -H 'content-type: application/json' $D/api/deployments/evict \
  -d '{"model_key":"<model_key>","replica":"<replica_id>"}'
curl -s -b jar -H 'content-type: application/json' $D/api/deployments/restart \
  -d '{"model_key":"<model_key>","replica":"<replica_id>"}'
curl -s -b jar -H 'content-type: application/json' $D/api/deployments/deploy \
  -d '{"checkpoint":"<repo_id>","model_key":"<model_key>"}'
```

`replica` is optional on evict and restart. An unknown replica still returns
HTTP 200: the `✗` line is inside the response body's `logs[]`, so check the
body, not the status code. A `model_key` is used verbatim; without one the
dashboard mints the key from `checkpoint` and the optional `trust_remote_code`
(default `false`), with no check on who asks. Endpoint bodies:
`docs/operating/dashboard.md`.

## `ndif env`, `ndif version`, `ndif doctor`, `ndif info`

- `ndif env` fetches `GET {NDIF_API_URL}/env` — the cluster's Python version and
  installed packages, from a TTL'd cache (300 s), filtered to a fixed key list
  unless `--all`. `ndif env --local` reports this machine instead. The pair is the
  fastest way to spot client/server nnsight drift.
- `ndif version` prints the resolved stack of *this* install: python, ndif,
  nnsight, torch, transformers, ray, peft, accelerate, and `cuda` (the CUDA line
  the installed torch wheel was built for). `--write PATH` dumps it as JSON; the
  image runs exactly that at build time into `/etc/ndif/build.json`.
- `ndif doctor` probes the *host*: Python, the `ray`/`redis-server`/`rustfs`
  binaries, a GPU visible to `nvidia-smi` and torch, and connectivity. Its
  Binaries section fails by design on a compose-only machine; the version and
  connectivity sections never set the exit code.
- `ndif info` prints `NDIF_HOME`, the tracked PID of each core service, and a
  reachability probe per endpoint — the fastest way to find out that a variable
  you thought you set never reached the process.

## Gotchas

- **`--replicas N` means "add N", not "have N"**, even under `--sync` (which
  computes the shortfall first).
- **`ndif deploy` needs Hub access** to canonicalize the checkpoint, and
  `HF_TOKEN` for a gated repo, wherever the deploy runs.
- **`pinned` does not stop you.** `ndif evict --all` sweeps pinned deployments.
- **`✗` is not a failure exit.** Check the printed output in scripts.
- **`ndif stop` only knows about processes it started.** It cannot stop a compose
  stack or reap a Ray head you launched by hand.
- **`NDIF_SERVICE` is dual-purpose**: it selects which services `ndif start`
  launches *and* becomes the `service` label on every log line and metric, so a
  multi-service value appears verbatim in your logs.
