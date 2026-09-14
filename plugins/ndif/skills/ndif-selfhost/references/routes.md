# The three routes, in full

Each section is self-contained: prerequisites, commands, what you get, and the
traps specific to that route. The source of record is
`docs/operating/quickstart.md` (all three) and `docker/README.md` (route 1).

## Route 1 — `docker run ndif/ndif`

One container runs the whole request path. `NDIF_SERVICE` defaults to `all` in
the image, and `all` resolves to **redis, minio, ray, api** — the image carries
its own `redis-server` (apt) and `minio` binary (copied out of
`quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z`), which is what makes a
single-container run possible.

```bash
docker run -d --name ndif --gpus all --shm-size 4g \
    -p 8001:8001 -p 9000:9000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN \
    ndif/ndif:0.1.0
```

### Tags

| Tag | Torch wheel | Use when |
|---|---|---|
| `latest`, `0.1.0`, `0.1.0-cu126` | CUDA 12.6 | Default. Any CUDA 12.x driver >= 525. |
| `0.1.0-cu130` | CUDA 13.0 | Drivers at 580 or newer, and Blackwell GPUs (RTX 50xx, B200). No cu128 tag: that torch index stopped at 2.11. |

Pick by the CUDA version in `nvidia-smi`'s header. The versions inside `0.1.0`:
ndif 0.1.0, nnsight 0.8.0rc1, torch 2.14.0, transformers 5.17.0, ray 2.55.1,
Python 3.12. The build also records what it resolved to at
`/etc/ndif/build.json`.

### Other commands from the same image

`ENTRYPOINT` is `ndif` and `CMD` is `start --foreground`, so replacing the
command runs any other CLI verb:

```bash
docker run --rm ndif/ndif:0.1.0 version              # no GPU, no volume, no port needed
docker run --rm --gpus all ndif/ndif:0.1.0 doctor
docker run --rm -e NDIF_SERVICE=api ndif/ndif:0.1.0  # one service (what compose does)
```

Against a running container, `docker exec ndif ndif <verb>`.

### Route-1 traps

- **Without a bind mount for `/root/.cache/huggingface`** the image's own
  `VOLUME` gives you an anonymous one: weights survive a restart, not a
  `docker rm`.
- **`--shm-size 4g` is a floor**, not a recommendation. Ray's plasma store lives
  in `/dev/shm` and Docker's default is 64 MB.
- **`--gpus all` hands over every GPU.** `--gpus '"device=0,1"'` restricts it.
  NDIF sizes placement from each visible GPU's *total* memory and never reads
  actual free memory, so give it cards nothing else is using.
- **The dashboard is opt-in:** `-e NDIF_SERVICE="all dashboard"` plus `-p 8081:8081`.

## Route 2 — the compose dev stack

```bash
git clone https://github.com/ndif-team/ndif.git && cd ndif
just up
```

The first `just up` builds one image and runs it three times — as `api`, `ray`
and `dashboard` — selected per container by `NDIF_SERVICE`. Expect ten minutes or
more the first time (torch is its own layer, then `requirements.txt`).

| Recipe | Does |
|---|---|
| `just up [services...]` | `compose up -d`, all or a subset |
| `just down [-v]` | stop and remove; `-v` also drops the dashboard volume |
| `just build [services...]` | rebuild the image |
| `just ta [services...]` | down → build → up — **the full refresh after any `src/` change** |
| `just restart [services...]` | bounce a container (does *not* pick up code changes) |
| `just logs [services...]` | follow; Ctrl-C detaches |
| `just ps` | container status and health |
| `just nnsight` | print which nnsight the stack will use |

Every recipe is a thin wrapper over
`docker compose -f docker/docker-compose.yml ...`; the compose project is named
`dev`, so the containers are `dev-api-1`, `dev-ray-1`, `dev-dashboard-1` if you
reach past `just`.

### What compose publishes to the host

| URL | What |
|---|---|
| http://localhost:8001 | the API |
| http://localhost:8081 | admin dashboard (no login — dev mode is on) |
| http://localhost:8265 | Ray dashboard: actors, nodes, logs |
| http://localhost:3000 | Grafana (anonymous admin), lands on NDIF — Overview |
| http://localhost:9000 / :9001 | MinIO S3 API / console (`minioadmin`/`minioadmin`) |

It also publishes Redis 6379, Postgres 5432, Loki 3100, Influx 8086, Prometheus
9090 and the Ray client port 10001, none of them authenticated. On a laptop that
is fine; on a routable address it is a full compromise — anyone who can reach
10001 can run arbitrary code on the cluster.

### The nnsight bind mount

`just up` / `just ta` add `docker/docker-compose.nnsight.yml` when an **editable**
nnsight is importable from the shell's Python, mounting that checkout over the
image's copy in `api`, `ray` and `dashboard`. A non-editable nnsight under
site-packages is deliberately not mounted; no nnsight at all skips the override.
`just nnsight` prints the decision. This is what lets client-side nnsight changes
be picked up without a rebuild — and what makes a mismatched client/server
nnsight or transformers produce `The model architecture on this server doesn't
match...` or `Your request payload could not be read (AttributeError: Can't get
attribute ...)`.

### Route-2 traps

- **`just up` after editing `src/` runs the stale image.** Compose only builds
  when the image is missing. Use `just ta`; `just ta ray` narrows the rebuild but
  still brings the whole stack down first.
- **`api` waits on `postgres` and `influxdb` health checks** even though it can
  run without either. If `api` never starts and has no logs, check those two in
  `just ps`.
- **Only `dashboard_data` is a named volume.** Result blobs, metrics, logs and
  the Postgres database do not survive `just down`. Downloaded weights do — the
  `ray` service bind-mounts the host HF cache.

## Route 3 — from source with the `ndif` CLI

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install ".[api,ray,metrics,postgres,dashboard]"
```

The extras gate optional subsystems: `api` (fastapi/gunicorn/the dispatcher),
`ray` (ray, transformers, accelerate, peft, zstandard — model actors),
`metrics` (influxdb-client, python-logging-loki), `postgres` (asyncpg — API-key
auth), `dashboard`, `dev` (pytest, ruff, httpx).

```bash
ndif doctor
ndif start                 # redis, minio, ray, api in dependency order, detached
ndif start dashboard       # opt-in; a bare `ndif start` never pulls it in
ndif info                  # tracked PIDs + reachability of each endpoint
ndif logs ray -f
ndif stop
```

State lives under `NDIF_HOME` (`~/.ndif`): `run/<service>.pid`,
`logs/<service>.log`, and `minio/` when the CLI spawns MinIO. `ndif stop` only
knows about processes it started — it cannot stop a compose stack.

### The two binaries

`ndif doctor` requires `redis-server` and `minio` on `PATH`.

- `redis-server` — your package manager, or conda-forge.
- `minio` — MinIO publishes no standalone server binaries any more, so doctor's
  hint has no download behind it. Verified alternative:

  ```bash
  conda install -c conda-forge minio-server
  ```

  Or lift it out of the image the Dockerfile uses:

  ```bash
  cid=$(docker create quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z)
  docker cp "$cid:/usr/bin/minio" /usr/local/bin/minio && docker rm "$cid"
  ```

  quay.io, not Docker Hub: `minio/minio` on the Hub no longer resolves. With an
  existing S3-compatible store you need neither — point
  `NDIF_OBJECT_STORE_URL` at it and run `ndif start redis ray api`.

### What `ndif doctor` checks

| Section | Checks | Counts as a failure? |
|---|---|---|
| Environment | Python >= 3.12; `ndif` and `nnsight` installed; then torch (with its CUDA build), transformers and ray | Python/ndif/nnsight **yes**; the other three are reported only |
| Binaries | `ray`, `redis-server`, `minio` on `PATH` | yes |
| Compute | `nvidia-smi` returns >= 1 GPU | yes |
| Connectivity | redis / minio / api / ray at their `NDIF_*` URLs | **no** — a stopped service is a normal answer |

Two caveats: it checks *this host's* `PATH`, which tells you nothing if you only
ever run compose; and a missing `nvidia-smi` is a hard failure even though the
non-GPU half of the stack runs fine without one.

## Adding a second GPU machine

One variable decides a node's role. `NDIF_RAY_HEAD_ADDRESS` unset ⇒ this node
runs `ray start --head` and launches the NDIF controller. Set to the head's
`HOST:PORT` ⇒ it waits for that port to accept TCP, then joins as a worker and
runs nothing else.

```bash
# on the new GPU machine
NDIF_RAY_HEAD_ADDRESS=10.0.0.5:6385 ndif start ray
```

```bash
docker run -d --name ndif-worker \
  --gpus all --shm-size 4g --network host \
  -e NDIF_SERVICE=ray \
  -e NDIF_RAY_HEAD_ADDRESS=10.0.0.5:6385 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ndif/ndif:0.1.0
```

`NDIF_SERVICE=ray` is required in the container — the image default `all` would
start redis, minio and an API on the worker too.

Ports that must be reachable from the worker to the head: **6385** (Ray GCS —
the join address, deliberately offset from Redis's 6379) and **8076** (object
manager), plus Ray's un-pinned raylet/worker range, which connects both
directions. Put both machines on a trusted private network rather than trying to
enumerate ports. Note the dev compose publishes only 8265 and 10001 from `ray`,
so a worker cannot reach a head started by a stock `just up` — give that service
`network_mode: host` or publish 6385/8076.

Confirm with `ndif status`: `Nodes` and `Total GPUs` should both rise, within one
`NDIF_CONTROLLER_SYNC_INTERVAL_S` (30 s). If `Nodes` went up and the GPU count
did not, the new node reported no GPU resource and the controller is ignoring it.
Procedure and drain steps: `docs/runbooks/add-a-gpu-node.md`.
