---
name: selfhost
description: Stand up your own NDIF server — the backend that nnsight's remote=True talks to — on your own GPUs. Use for "run my own NDIF", "self-host NDIF", "docker run ndif/ndif", "local NDIF server", "point nnsight at my server", "ndif start", "install NDIF from source", "NDIF docker compose", or any question about NDIF prerequisites, ports, volumes, tags, or configuration. Covers all three routes (the published ndif/ndif image, the compose dev stack from a checkout, and a from-source install with the ndif CLI), how to prove it works, and the one default that matters — an NDIF with no Postgres runs every caller's Python inside the model process. NOT for using the public ndif.us service; that is the nnsight plugin's `remote` skill.
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

**An NDIF with no `NDIF_POSTGRES_URL` has no authentication, and no
authentication means every request is *trusted*.** All three routes default to
that. A trusted request:

- runs the caller's traced block **in-process inside the model actor**, in the
  same process as the weights — no runner subprocess, no isolation;
- loads models with **`trust_remote_code=True`**, so a request naming any
  Hugging Face repo can execute that repo's code on your GPU node.

That is the right default for an NDIF you run for yourself, and the wrong one the
moment a second person can reach port 8001. Before anyone else can:
`docs/runbooks/enable-auth.md`, then `docs/operating/production.md`. The
`operate` skill covers what turning it on changes.

## Pick a route

| Route | Command | Take it when |
|---|---|---|
| **1. Published image** | `docker run --gpus all ndif/ndif` | You want an NDIF, not a checkout. One container runs redis, minio, ray and the API. |
| **2. Compose dev stack** | `just up` from a clone | You are changing NDIF, or want Postgres and the telemetry tier (Loki/Influx/Prometheus/Grafana) alongside. |
| **3. From source** | `pip install "ndif[api,ray]"`, then `ndif start` | No Docker on the box, or you want the services as ordinary host processes. |

All three come up with **no configuration at all** — every service has a working
single-host default. Full per-route detail: `docs/operating/quickstart.md`.

## Prerequisites

| Requirement | Why |
|---|---|
| An NVIDIA GPU and a CUDA driver | The controller only manages Ray nodes that report a `GPU` resource. A CPU-only node joins Ray and is then ignored; `/ping` answers but `/request` cannot be served. |
| NVIDIA Container Toolkit (routes 1 and 2) | `docker run --gpus all` must work, or the container fails to create. |
| `--shm-size 4g` (routes 1 and 2) | Ray's plasma object store lives in `/dev/shm`; Docker's 64 MB default makes Ray spill to disk or fail, and the error never mentions shm. |
| Room on the filesystem under `NDIF_RAY_TEMP_DIR` (default `/tmp/ray`) | Ray's raylet stops scheduling once that filesystem passes 95 % full, and the only symptom is a server that comes up and never runs anything. On a full `/`, point it elsewhere (route 1: `-e NDIF_RAY_TEMP_DIR=/ndifray -v /big/disk/ray:/ndifray`). `ndif doctor` checks this from 0.1.2. |
| Disk for weights, plus host RAM | Checkpoints download at deploy time (gpt2 ~0.5 GB, a 70B ~140 GB). Evicted models are held in host RAM as WARM. |
| Python 3.12 or 3.13 (route 3) | `requires-python = ">=3.12,<3.14"`; `ndif doctor` fails below 3.12. |

**Match the tag to your driver.** Every release is published as
`ndif/ndif:<version>-cu126` and `<version>-cu130`; the bare `<version>` and
`latest` point at the newest cu126. cu126 runs on any CUDA 12.x driver >= 525;
cu130 is for CUDA 13 drivers (580+) and Blackwell (RTX 50xx, B200). There is no
cu128 tag — PyTorch's cu128 index stopped at torch 2.11. A wheel built for a
CUDA line your driver predates does not fail loudly — `torch.cuda.is_available()`
just returns `False` and Ray starts with `cuda_memory_bytes: 0`.

The 0.1 line carries nnsight 0.8.0rc1, torch 2.14, transformers 5.17, ray 2.55,
Python 3.12. Ask an image directly, with no GPU or volume needed:

```bash
docker run --rm ndif/ndif version
docker run --rm --gpus all ndif/ndif doctor
```

## On a GPU box you share

Every example here assumes the machine is yours. On a shared host three things
change:

- **Pin the cards.** `docker run --gpus '"device=0,1"'` (that exact quoting) for
  routes 1 and 2, `CUDA_VISIBLE_DEVICES=0,1` for route 3. `ndif doctor` should
  then report torch seeing exactly that many GPUs.
- **The controller sizes from each card's *total* memory and never reads what
  other people hold.** `ndif status` will say a 80 GB card is 80 GB free while a
  colleague's job holds 25 GB of it, and the placer will put a model there.
  Subtract other tenants' `nvidia-smi` usage yourself and force the placement
  with `ndif deploy --gpus N` (or `--size-bytes`); see the `operate` skill's
  sizing section for the arithmetic.
- **The container runs as root**, so anything it writes into a bind mount — the
  Ray temp dir, new files in the HF cache — is root-owned afterwards. Clean up
  through a container (`docker run --rm -v /path:/v alpine rm -rf /v/...`) or
  keep those mounts on directories you don't need to delete as yourself.

## Route 1 — the published image

```bash
docker run -d --name ndif --gpus all --shm-size 4g \
    -p 8001:8001 -p 9000:9000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    ndif/ndif
```

- **8001** is the API — the only port a client posts to.
- **9000** is MinIO. Publish it: a result over `NDIF_MAX_SOCKET_RESULT_BYTES`
  (4 MiB) comes back as a presigned URL the *client* fetches. Unpublished, large
  results complete server-side and then fail to download.
- The **HF cache mount** is what makes weights survive `docker rm`.
- Serving gated checkpoints (Llama, Gemma)? Add `-e HF_TOKEN` to pass your token
  through from the shell. Nothing else needs it.

The image's `ENTRYPOINT` is the `ndif` CLI and `NDIF_SERVICE` defaults to `all`,
which is redis + minio + ray + api in one container. `NDIF_SERVICE="all dashboard"`
adds the admin UI on 8081 (`-e NDIF_DASHBOARD_DEV_MODE=true` to skip its login on
a machine only you can reach). Everything else about the image is in
`docker/README.md`.

Manage it through `docker exec`, since the CLI is already installed there, or
over HTTP — `GET /status` returns the same payload as `ndif status
--json-output`, `GET /env` the server's package versions, and `/docs` is the
FastAPI page:

```bash
docker exec ndif ndif status
docker exec ndif ndif deploy openai-community/gpt2
curl -s localhost:8001/status | python3 -m json.tool | head
```

Shutting it down: `docker stop ndif` is enough — nothing needs draining, and
`ndif evict --all` first is tidy but not required. `docker rm` throws away the
deployment set (what was HOT) but not the weights, which live in the mounted
cache. Bring it back with the same `docker run` and models load again on demand.

## Route 2 — the compose dev stack

```bash
git clone https://github.com/ndif-team/ndif.git && cd ndif
just up            # builds the image on first run (~10 min), then starts everything
just ps            # container status and health
just logs ray      # follow one service; Ctrl-C detaches without stopping it
```

Ten containers: redis, minio, api, ray (required), postgres and influxdb
(health-gated — the API waits on them but uses neither unless configured), loki,
prometheus, grafana, dashboard (optional). `docs/operating/compose-stack.md` has
the service-by-service reading.

Two things that are only true here:

- **`just up` bind-mounts an *editable* nnsight over the image's copy.** It
  resolves `NNSIGHT_PATH` from your shell's Python; a non-editable nnsight under
  site-packages is deliberately **not** mounted, and no nnsight at all skips the
  mount and uses the image's pinned copy. `just nnsight` prints which applies. A
  mismatched client/server nnsight or transformers shows up as
  `The model architecture on this server doesn't match...` or
  `Your request payload could not be read (AttributeError: Can't get attribute ...)`.
- **`just up` after a code change runs the stale image.** The source is baked in
  and compose only builds when the image is missing. Use `just ta`
  (down → build → up). nnsight is the exception — its bind mount picks up changes
  without a rebuild.

## Route 3 — from source, no Docker

No checkout is needed; the package is on PyPI and pins the 0.8 nnsight itself.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # FIRST — match your driver
pip install "ndif[api,ray]"                       # add metrics,postgres,dashboard as needed
ndif doctor        # versions, binaries, GPU, connectivity — read this before starting
ndif start         # redis, minio, ray, api — detached, PID files under ~/.ndif
ndif info
ndif logs api -f
ndif stop          # also runs `ray stop` for the daemons ray start left behind
```

torch first: nnsight pulls torch in, and with no wheel present pip takes PyPI's
default, which is the CUDA 13 build. From a checkout, `pip install -r
requirements.txt` before `pip install ".[api,ray]"` gives you the exact pinned
set the image is built from; the sdist on PyPI also ships `requirements.txt`
and `docs/`.

Two traps on this route that nothing else warns about:

- **Keep `NDIF_RAY_TEMP_DIR` short.** Ray puts unix sockets under
  `<dir>/session_<ts>_<pid>/sockets/` and AF_UNIX paths cap at 107 bytes, so
  a long temp dir makes `ray start` die at once with `validate_socket_filename
  failed` in the ray log — which from outside looks like a Ray that never
  boots. `/tmp/ndif-ray` is fine; a deep project path is not.
- **A ✓ from `ndif start` means the process was alive two seconds later, no
  more.** If `/connected` is still `reconnecting` after a minute or two,
  `ndif info` (is `ray` `stopped`?) and `$NDIF_HOME/logs/ray.log` — do not
  keep waiting.

**The MinIO binary is the awkward part.** `ndif doctor` checks for `redis-server`
and `minio` on `PATH`, and MinIO publishes no standalone server binaries any more
(`dl.min.io` returns 410, the GitHub releases carry no assets). Doctor's hint
names the conda-forge package; the two options that work:

```bash
conda install --override-channels -c conda-forge redis-server minio-server   # verified; --override-channels skips the anaconda ToS prompt a stock miniconda raises
```

```bash
cid=$(docker create quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z)
docker cp "$cid:/usr/bin/minio" /usr/local/bin/minio && docker rm "$cid"
```

quay.io, not Docker Hub — `minio/minio` on the Hub no longer resolves at all.
Already have an S3-compatible store? Point `NDIF_OBJECT_STORE_URL` at it and run
`ndif start redis ray api`; you need no binary.

## Prove it works

The checks are the same whichever route you took; only how you list the processes
differs (`docker ps`, `just ps`, `ndif info`).

```bash
curl localhost:8001/ping         # "pong" — only proves the web process is alive
curl localhost:8001/connected    # {"status":"connected"} — the real readiness signal
ndif status                      # what the controller thinks is deployed, and on which GPUs
```

**Ray takes 15–90 seconds to boot, depending on the host.** Until it does,
`/connected` says `reconnecting`, `/request` 503s, and the api log prints `Error
connecting to Ray` tracebacks about once a second. That is normal during boot,
not a fault — but if it is still `reconnecting` after two minutes, stop waiting
and read the ray log; a Ray that died at start looks exactly like one that is
slow. In the
ray log you want `Starting Ray head node with resources: {"head": 10,
"cuda_memory_bytes": ..., ...}` followed by `Starting NDIF controller...` — a
`cuda_memory_bytes` of `0` means torch cannot see a GPU.

Then the first remote trace. Nothing needs to be deployed first: the queue is
lazy, so the first request for a checkpoint downloads and loads it while the
client sits in `QUEUED`/`DEPLOYING`.

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

No API key is needed, because auth is off. The model is never dispatched
locally — the client builds it on the meta device only to resolve module paths
and mint the model key. The client shows `PROVISIONING` even when the model is
already HOT; only the timing (no `DEPLOYING` wait) tells you a pre-deploy took.

**`ndif status` on a fresh server may list hundreds of COLD models.** COLD is
nothing more than the Hugging Face cache directory listing: bind-mount a cache
that other projects have filled and every checkpoint in it shows up as COLD,
whether NDIF has ever run it or not. HOT is what is loaded; COLD is what could
be.

## Configuration

**There is no config file.** Every knob is an `NDIF_*` environment variable, read
once at process start, and every one has a working single-host default. Changing
a value means restarting the process that reads it; for anything baked into a
compose image, a `just ta`. Config is additive — you set variables to move *away*
from single-host, not to get started.

The handful you are most likely to need:

| Variable | Default | Set it when |
|---|---|---|
| `HF_TOKEN` | unset | You serve gated checkpoints. |
| `NDIF_OBJECT_STORE_PUBLIC_URL` | falls back to `NDIF_OBJECT_STORE_URL` (`http://localhost:9000`) | Clients are on another machine. This is the address presigned result URLs are **signed with**, and it must be the one the client resolves. |
| `NDIF_DEFAULT_PADDING_FACTOR` | `0.15` | Blocks die with `CUDA out of memory ... N MiB allowed` on a nearly empty card. |
| `NDIF_DEPLOYMENTS` | unset | You want models loaded at controller start. pipe-separated **model keys**, not checkpoints — see the gotcha below. |
| `NDIF_MIN_NNSIGHT_VERSION` | unset | You want an old client to get a clear 400 instead of failing deep in deserialization. The cheapest operational improvement available to a self-hoster. |
| `NDIF_RAY_TEMP_DIR` | `/tmp/ray` | The filesystem holding it can exceed 95% full — Ray's file-system monitor then refuses to schedule any work. |
| `NDIF_SERVICE` | `all` in the image | You want one role per container, or `all dashboard`. |

Full tables — every variable with the line that reads it, every port, and the
volumes worth keeping — in [references/configuration.md](references/configuration.md),
and exhaustively in `docs/reference/env-vars.md` and `docs/reference/ports.md`.

## Gotchas

- **Results over 4 MiB *after compression* come back as a presigned MinIO
  URL.** Under `NDIF_MAX_SOCKET_RESULT_BYTES` they ride on the response itself
  and port 9000 is never touched; above it — and for *every* non-blocking
  request — the client fetches the URL directly, so it needs to reach 9000 and
  the signature has to name a host it can resolve. The threshold is on the
  serialized, compressed payload, not on the tensor bytes you count: 4.7 MiB of
  bf16 activations can still ride the socket. The client prints a
  `Downloading result` bar when MinIO was used.
- **A block that allocates a lot of GPU memory dies with `CUDA out of memory ...
  N MiB allowed` even on an almost-empty card.** The actor caps per-process GPU
  memory at the model's size × `NDIF_DEFAULT_PADDING_FACTOR` (plus
  `NDIF_DEFAULT_PADDING_BIAS`), so a runaway request hits its own limit instead
  of trampling a co-tenant. Raise the factor or save less.
- **`NDIF_MODEL_CACHE_PERCENTAGE` is host RAM, not GPU memory.** It is the WARM
  cache budget. It is the variable people reach for during a GPU OOM and it does
  nothing for one.
- **`NDIF_DEPLOYMENTS` takes model keys, not repo ids.** Entries are used
  verbatim as keys, so `openai-community/gpt2` there becomes a bogus key that
  fails to evaluate. Get real ones from
  `ndif status --json-output | jq -r '.deployments[] | select(.model_key) | .model_key'`
  (the `select` skips COLD stubs, which carry no key). `jq` is not in the
  image: pipe on the host, or use `python3 -c 'import json,sys; [print(d["model_key"]) for d in json.load(sys.stdin)["deployments"].values() if "model_key" in d]'`.
  For one checkpoint by name, `ndif deploy <repo-id>` after start is simpler.
- **Inside the ray container, `localhost:6379` is Ray's GCS, not Redis.** NDIF
  moves Ray's head port to 6385 for exactly this reason, but a provider that
  falls back to the `redis://localhost:6379` default in that container reaches
  the wrong server — compose sets `NDIF_REDIS_URL` explicitly there.
- **Fresh `ray://` connections used to fail about half the time** with
  `Starting Ray client server failed` after ~40 s. `ray/start.sh` exports
  `GRPC_ENABLE_FORK_SUPPORT=0` before `ray start`, which removes it; if you see
  it, something overrode that variable.
- **A prompt longer than the model's context dies as `CUDA error: device-side
  assert triggered` in `masking_utils`,** not as an index error naming the
  limit (gpt2: 1024 positions). The actor survives it; the next request runs.
- **`.save()` goes on the object you assign, not on what you put inside it.**
  `acts = [h[i].output.save() for i in ...]` and `acts = []` + `.append(...save())`
  leave `acts` unbound after the block — the request reports `COMPLETED`, nothing
  downloads, and the client hits `NameError`. Any of these work: save the
  container (`acts = [h[i].output for i in ...].save()`, `{i: ... }.save()`),
  save an empty one and append to it (`acts = list().save()` then
  `acts.append(h[i].output)`), or create the list *before* the block and append
  `.save()`d values into it. That is nnsight behaviour, not a server fault; the
  nnsight plugin's `debugging` skill covers it.
- **Multimodal checkpoints reshape the module tree.** A `*ForConditionalGeneration`
  model such as `google/gemma-3-27b-it` keeps its decoder under
  `model.model.language_model.layers[i]`, not `model.model.layers[i]`; print the
  model once before naming a layer. Decoder layers on transformers 5.x return a
  plain tensor, so `.output[0]` selects batch element 0, not a tuple slot.

## References

- [references/routes.md](references/routes.md) — the three routes in full:
  per-route commands, what each one installs, adding a GPU node, the MinIO
  binary, and what `ndif doctor` does and does not check.
- [references/configuration.md](references/configuration.md) — the env-var model,
  the variables that must change off single-host, ports, volumes, and what
  survives a restart.

## Related skills

- `operate` — deploying and sizing models, the dashboard, auth, production.
- `troubleshoot` — when it does not come up, or a request hangs.
- `develop` — changing the server's code.
- nnsight plugin `remote` — writing the client-side code that talks to it.
