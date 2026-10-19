---
name: selfhost
description: Stand up your own NDIF server — the backend that nnsight's remote=True talks to — on your own GPUs. Use for "run my own NDIF", "self-host NDIF", "docker run ndif/ndif", "local NDIF server", "point nnsight at my server", "ndif start", "install NDIF from source", "NDIF docker compose", or any question about NDIF prerequisites, ports, volumes, tags, or configuration. Covers all three routes (the published ndif/ndif image, the compose dev stack from a checkout, and a from-source install with the ndif CLI), how to prove it works, reaching it from another machine, and the two defaults that matter — an NDIF with no Postgres trusts every caller, and only a sandbox actor class (which only compose sets) runs untrusted code outside the model process. NOT for using the public ndif.us service; that is the nnsight plugin's `remote` skill.
---

# Self-hosting NDIF

NDIF is the server behind nnsight. It holds the weights on your GPUs, receives a
serialized `with model.trace(...)` block from a client, runs it against the real
forward pass, and sends the saved values back. Self-hosting it means your own
`remote=True` runs on hardware you control.

Using the public **ndif.us** service instead — API keys, model availability,
writing efficient remote code — is the nnsight plugin's `remote` skill. Nothing
below duplicates it.

## Read this before you start

Two separate switches decide where a caller's code runs, and the defaults of
both are open.

**No `NDIF_POSTGRES_URL` means no authentication, and every request is
trusted.** All three routes default to that. A trusted request runs the caller's
traced block **in-process inside the model actor**, next to the weights, and
may deploy a model whose key carries **`trust_remote_code`**, so a request
naming any Hugging Face repo can execute that repo's code on your GPU node.

**Auth alone does not sandbox anything.** Auth decides *whether* a request is
trusted; the **model actor class** decides whether that makes a difference. Only
the sandbox actor classes run an untrusted request in a separate runner process —
the base `ModelActor` runs every request in-process regardless of `trusted`.
That runner is process separation, not a hardened jail (same user, filesystem,
network and GPUs); what it guarantees is that an untrusted request cannot read
or edit the weights or train.
Only the compose stack selects a sandbox class. The published image and a
from-source install run untrusted code in-process **even with auth on**, until
you set, on the `ray` service:

| Variable | Value |
|---|---|
| `NDIF_MODEL_IMPORT_PATH` (legacy fallback: `NDIF_DEFAULT_MODEL_ACTOR_CLASS`) | `ndif.services.ray.sandbox.model.SandboxModelActor` |
| `NDIF_TP_MODEL_ACTOR_CLASS`, if tensor parallelism is on | `ndif.services.ray.tp.model.SandboxedTPModelActor` — the plain `TPModelActor` has no runner |

That is the right default for an NDIF you run for yourself, and the wrong one the
moment a second person can reach port 8001. Before anyone else can:
`docs/runbooks/enable-auth.md`, then
`docs/operating/production.md#the-sandboxed-actor-class`. The `operate` skill
covers what turning them on changes.

## Pick a route

| Route | Command | Take it when |
|---|---|---|
| **1. Published image** | `docker run --gpus all ndif/ndif` | You want an NDIF, not a checkout. One container runs redis, rustfs, ray and the API. |
| **2. Compose dev stack** | `just up` from a clone | You are changing NDIF, or want Postgres and the telemetry tier (Loki/Influx/Prometheus/Grafana) alongside. |
| **3. From source** | `pip install "ndif[api,ray]"`, then `ndif start` | No Docker on the box, or you want the services as ordinary host processes. |

All three come up with **no configuration at all** — every service has a working
single-host default. Full per-route detail: `docs/operating/quickstart.md` and
[references/routes.md](references/routes.md).

## Prerequisites

| Requirement | Why |
|---|---|
| An NVIDIA GPU and a CUDA driver | The controller only manages Ray nodes that report a `GPU` resource. A CPU-only node joins Ray and is then ignored; `/ping` answers but `/request` cannot be served. |
| NVIDIA Container Toolkit, registered with Docker (routes 1 and 2) | If containers see no GPU: `sudo nvidia-ctk runtime configure --runtime=docker`, then restart Docker. |
| `--shm-size 4g` (route 1; compose sets it) | Ray's plasma object store lives in `/dev/shm`; Docker's 64 MB default makes Ray spill to disk or fail, and the error never mentions shm. |
| [`just`](https://github.com/casey/just) (route 2) | Every compose recipe goes through it. `curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \| bash -s -- --to ~/.local/bin`, with `~/.local/bin` on `PATH`. |
| ~8 GB for the image, plus ~8 GB of build cache for a compose build | Both land under Docker's data root (typically `/var/lib/docker`) — check free space there, not only on the weights disk. |
| Room on the filesystem under `NDIF_RAY_TEMP_DIR` (default `/tmp/ray`) | Ray's raylet stops scheduling once that filesystem passes 95 % full, and the only symptom is a server that comes up and never runs anything. `ndif doctor` does not check this. On a full `/`, point it elsewhere (route 1: `-e NDIF_RAY_TEMP_DIR=/ndifray -v /big/disk/ray:/ndifray`). |
| Disk for weights, plus host RAM | Checkpoints download at deploy time (gpt2 ~0.5 GB, a 70B ~140 GB). Evicted models are held in host RAM as WARM. |
| Python 3.12 or 3.13 (route 3) | `requires-python = ">=3.12,<3.14"`; `ndif doctor` fails below 3.12. |

**Match the tag to your driver.** Images are published per CUDA line, `-cu126`
and `-cu130`; the bare version and `latest` point at cu126. cu126 runs on any
CUDA 12.x driver >= 525; cu130 is for CUDA 13 drivers (580+) and Blackwell
(RTX 50xx, B200). There is no cu128 tag. A wheel built for a CUDA line your
driver predates does not fail loudly — `torch.cuda.is_available()` just returns
`False` and Ray starts with `cuda_memory_bytes: 0`. Tags: `docker/README.md`.

Ask an image what it carries, with no GPU or volume needed:

```bash
docker run --rm ndif/ndif version
docker run --rm --gpus all ndif/ndif doctor
```

## On a GPU box you share

Every example here assumes the machine is yours. On a shared host:

- **Pin the cards.** `docker run --gpus '"device=0,1"'` (that exact quoting),
  `CUDA_VISIBLE_DEVICES=0,1` for route 3. `ndif doctor` should then report torch
  seeing exactly that many GPUs.
- **The controller sizes from each card's *total* memory and never reads what
  other people hold.** `ndif status` will say an 80 GB card is 80 GB free while a
  colleague's job holds 25 GB of it, and the placer will put a model there.
  Subtract other tenants' `nvidia-smi` usage yourself and force the placement
  with `ndif deploy --gpus N` (or `--size-bytes`); see the `operate` skill.
- **The container runs as root**, so anything it writes into a bind mount — the
  Ray temp dir, new files in the HF cache — is root-owned afterwards. Clean up
  through a container (`docker run --rm -v /path:/v alpine rm -rf /v/...`).
- **A port already taken: remap only the host side.** Route 1:
  `-p 127.0.0.1:9100:9000 -e NDIF_OBJECT_STORE_PUBLIC_URL=http://localhost:9100`
  — moving 9000 always needs the public URL to follow it. Route 2 is below.
- **Bind published ports to `127.0.0.1`.** The compose stack publishes Redis,
  Postgres, RustFS, the Ray client port and the telemetry tier on every
  interface, unauthenticated; anyone who reaches Redis or 10001 can run code on
  the cluster.

## Route 1 — the published image

```bash
docker run -d --name ndif --gpus all --shm-size 4g \
    -p 127.0.0.1:8001:8001 -p 127.0.0.1:9000:9000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    ndif/ndif
```

- **8001** is the API — the only port a client posts to.
- **9000** is RustFS. Publish it: a result over the inline cap, and every
  non-blocking result, comes back as a presigned URL the *client* fetches.
  Unpublished, those results complete server-side and then fail to download.
- The **HF cache mount** is what makes weights survive `docker rm`; they land
  under `<cache>/hub/models--<org>--<name>`.
- Serving gated checkpoints (Llama, Gemma)? Add `-e HF_TOKEN`.

`NDIF_SERVICE` defaults to `all` — redis + rustfs + ray + api in one container.
`NDIF_SERVICE="all dashboard"` adds the admin UI on 8081. Manage it through
`docker exec ndif ndif <verb>` or over HTTP (`GET /status` is the same payload as
`ndif status --json-output`). `docker stop ndif` is a clean shutdown; `docker rm`
discards which models were HOT but not the weights.

## Route 2 — the compose dev stack

```bash
git clone https://github.com/ndif-team/ndif.git && cd ndif
just up            # builds the image on first run (~10 min), then starts everything
just ps            # container status and health
just logs ray      # follow one service; Ctrl-C detaches without stopping it
```

Ten containers: redis, rustfs, api, ray (required), postgres and influxdb
(health-gated — the API waits on them but uses neither unless configured), loki,
prometheus, grafana, dashboard. `docs/operating/compose-stack.md` reads them
service by service. What a fresh install runs into:

- **Every port is published on `0.0.0.0`.** Change each `ports:` entry in
  `docker/docker-compose.yml` to a `127.0.0.1:` bind. A
  `docker-compose.override.yml` is **not read** — the justfile passes `-f`
  explicitly.
- **The project name is fixed at `dev`.** Containers are `dev-api-1`,
  `dev-ray-1`; a second `just up` on the same host takes over an existing `dev`
  stack, and existing `dev-*` images are retagged by the next build.
- **The HF cache is `${HOME}/.cache/huggingface`**, written as root; weights
  land under `<cache>/hub/models--<org>--<name>`. To move it, edit the `ray`
  volume line in `docker/docker-compose.yml` or symlink the host path. A cache
  shared with other work makes every model in it show up as COLD.
- **A host port already taken:** remap only the host side,
  `"127.0.0.1:<newport>:<port>"`; if 9000 moves, set
  `NDIF_OBJECT_STORE_PUBLIC_URL` to the new port on **both** `api` and `ray`.
- **Code changes need `just ta`** (down → build → up): the source is baked into
  the image, so `just up` or `just restart` runs the old build. `just ta <svc>`
  still tears the **whole** stack down and starts only that service and its
  dependencies — follow it with `just up`.
- **Environment edits need `just up <svc>`**, which recreates the container.
  `just restart` restarts processes with their *existing* environment.

### Developing against a local nnsight

`docker/docker-compose.nnsight.yml` bind-mounts a checkout over the image's
nnsight in `api`, `ray` and `dashboard`:

```bash
export NNSIGHT_PATH=/path/to/nnsight/src/nnsight   # the PACKAGE dir, not the repo root
just nnsight                                        # prints what would be mounted
just up
```

- **Every `just` call must carry `NNSIGHT_PATH`** — a `just up` or `just ta`
  without it recreates the containers without the mount.
- Auto-detection without it runs `python -c "import nnsight"` for an editable
  install; a host with only `python3` always detects nothing, silently. **`just
  nnsight` is the only reliable pre-check** — `just --dry-run up` always shows
  the overlay.
- **A fresh checkout has no compiled `_c` extension**: every remote `.save()`
  fails with `AttributeError: 'Tensor' object has no attribute 'save'`. Build it
  with the image's toolchain (as root, leaving root-owned build artifacts):
  `docker run --rm --entrypoint python -v <checkout>:/src -w /src dev-ray setup.py build_ext --inplace`.
- **`nnsight.__version__` reports the image's version under the mount.** Verify
  with `grep nnsight /proc/self/mountinfo` in the container; its source column is
  filesystem-relative, not the host path (example line in
  [references/routes.md](references/routes.md)).

Full procedure: `docs/developing/nnsight-integration.md#developing-against-a-local-nnsight`.

## Route 3 — from source, no Docker

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # FIRST — match your driver
pip install "ndif[api,ray]"                       # add metrics,postgres,dashboard as needed
ndif doctor        # versions, binaries, GPU, connectivity — read this before starting
ndif start         # redis, rustfs, ray, api — detached, PID files under ~/.ndif
ndif info
ndif stop
```

torch first: with no wheel present pip takes PyPI's default, the CUDA 13 build.
Route-specific traps — the `rustfs` binary (a GitHub release, or lifted from the
pinned image),
a temp dir too long for AF_UNIX sockets, and what a ✓ from `ndif start` does not
prove — are in [references/routes.md](references/routes.md).

## Prove it works

```bash
curl localhost:8001/ping         # "pong" — only proves the web process is alive
curl localhost:8001/connected    # {"status":"connected"} — the real readiness signal
ndif status                      # what the controller thinks is deployed, and on which GPUs
```

**Ray takes 15–90 seconds to boot.** Until it does, `/connected` says
`reconnecting`, `/request` 503s, and the api log prints `Error connecting to Ray`
about once a second. If it is still `reconnecting` after two minutes, read the
ray log; a Ray that died at start looks exactly like one that is slow. You want
`Starting Ray head node with resources: {"head": 10, "cuda_memory_bytes": ...}`
then `Starting NDIF controller...` — a `cuda_memory_bytes` of `0` means torch
cannot see a GPU.

Then the first remote trace. Nothing needs to be deployed first: the queue is
lazy, so the first request for a checkpoint replies `PROVISIONING` and then
`DEPLOYING` while the weights download, then `QUEUED`, `DISPATCHED`, `RUNNING`.

<!-- test: skip -->
```python
import nnsight
nnsight.CONFIG.API.HOST = "http://localhost:8001"

from nnsight.modeling.transformers import TransformersModel
model = TransformersModel("openai-community/gpt2", task="text-generation")

with model.trace("The Eiffel Tower is in the city of", remote=True):
    hidden = model.transformer.h[-1].output.save()

print(hidden.shape)   # torch.Size([1, 10, 768])
```

No API key is needed, because auth is off.

**`ndif status` on a fresh server may list hundreds of COLD models.** COLD is the
Hugging Face cache directory listing: mount a cache other projects have filled
and every checkpoint in it shows up. HOT is what is loaded; COLD is what could be.

### From another machine

A remote client needs **two** addresses: the API, and
`NDIF_OBJECT_STORE_PUBLIC_URL` — every non-blocking result, and any blocking one
over the inline cap, is a presigned URL the client downloads. With ports bound to
loopback:

```bash
ssh -fN -L 8001:localhost:8001 user@server     # the API
ssh -fN -L 9000:localhost:9000 user@server     # the object store
```

A presigned URL's signature covers its host, so the client must reach the store
under exactly the host and port `NDIF_OBJECT_STORE_PUBLIC_URL` names.

## Configuration

**There is no config file.** Every knob is an `NDIF_*` environment variable, read
once at process start, each with a working single-host default. Model actors are
configured by configuring the `ray` service.

| Variable | Default | Set it when |
|---|---|---|
| `HF_TOKEN` | unset | You serve gated checkpoints. |
| `NDIF_OBJECT_STORE_PUBLIC_URL` | falls back to `NDIF_OBJECT_STORE_URL` (`http://localhost:9000`) | Clients are on another machine — it must be the address they resolve. |
| `NDIF_MODEL_IMPORT_PATH` | in-process `ModelActor` (compose: the sandbox actor) | Anyone else can submit — see the top of this page. |
| `NDIF_DEFAULT_PADDING_FACTOR` | `0.15` | Blocks die with `CUDA out of memory ... N MiB allowed` on a nearly empty card. |
| `NDIF_DEPLOYMENTS` | unset | You want models loaded at controller start. Takes **model keys**, not repo ids — the `operate` skill covers getting them. |
| `NDIF_MIN_NNSIGHT_VERSION` | unset | You want an old client to get a clear 400 instead of failing deep in deserialization. |
| `NDIF_RAY_TEMP_DIR` | `/tmp/ray` | The filesystem holding it can pass 95 % full. |
| `NDIF_SERVICE` | `all` in the image | You want one role per container, or `all dashboard`. |

What must change off single-host and what survives a restart:
[references/configuration.md](references/configuration.md). Every variable:
`docs/reference/env-vars.md`.

## Gotchas

- **The inline result cap comes from Redis, not from NDIF.** A blocking result at
  or under `RedisProvider.max_publish_bytes` rides on the `COMPLETED` response;
  anything larger goes through the object store. The cap is `min(soft, hard // 2)`
  of Redis's `client-output-buffer-limit pubsub` — **8 MiB on stock Redis** — and
  there is no env var for it: raise Redis's pubsub limits to raise it. The cap is
  checked against the **compressed blob** (a 7.5 MiB blob goes inline, a 9 MiB
  one by URL). The published frame is up to ~1.5x that blob, which affects only
  buffer pressure, not the boundary: about three near-cap results to a
  subscriber that is not draining reach the hard limit, Redis drops the
  subscriber, and `COMPLETED` never arrives.
  `docs/concepts/status-and-results.md`.
- **A block that allocates a lot of GPU memory dies with `CUDA out of memory ...
  N MiB allowed` even on an almost-empty card.** The actor caps per-process GPU
  memory at `base + base × NDIF_DEFAULT_PADDING_FACTOR + NDIF_DEFAULT_PADDING_BIAS`,
  so a runaway request hits its own limit. Raise the factor or save less.
- **`NDIF_MODEL_CACHE_PERCENTAGE` is host RAM, not GPU memory.** It is the WARM
  cache budget, and it does nothing for a GPU OOM.
- **Nothing listens on `localhost:6379` inside the ray container.** Redis is
  another container, and NDIF moves Ray's head port to 6385. A provider that falls
  back to the `redis://localhost:6379` default there reaches nothing — compose
  sets `NDIF_REDIS_URL` explicitly on `ray`.
- **`ray/start.sh` exports `GRPC_ENABLE_FORK_SUPPORT=0`** before `ray start` so
  the Ray client server's per-client fork does not die. An explicit value wins;
  if `ray://` connections fail with `Starting Ray client server failed` after
  ~40 s, something set it.

## References

- [references/routes.md](references/routes.md) — the three routes in full:
  per-route commands, what each one installs, and each route's traps.
- [references/configuration.md](references/configuration.md) — the env-var model,
  the variables that must change off single-host, and what survives a restart.

## Related skills

- `operate` — deploying and sizing models, the dashboard, auth, production.
- `troubleshoot` — when it does not come up, or a request hangs.
- `develop` — changing the server's code.
- nnsight plugin `remote` — writing the client-side code that talks to it.
