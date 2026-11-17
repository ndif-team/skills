# Standing a server up: the three routes in full

`SKILL.md` has the published-image command, the readiness check and the first
remote trace. This page is what each route adds, a GPU box you share, and a
second GPU machine. Both open defaults (no auth, no sandbox) apply to every
route: [production and auth](operate-production-and-auth.md).

## Before any route

| Need | Why |
|---|---|
| An NVIDIA GPU and driver | The controller manages only Ray nodes that report a GPU. Without one the server boots, `/connected` answers 200, and every deploy fails `No GPU nodes available.` |
| NVIDIA Container Toolkit registered with Docker (image, compose) | `sudo nvidia-ctk runtime configure --runtime=docker`, then restart Docker |
| ~8 GB under Docker's data root for the image, about as much again for a compose build | Check free space there, not only on the weights disk |
| Disk for weights, and host RAM | Checkpoints download at first deploy. Evicted models wait in host RAM as WARM, up to `NDIF_MODEL_CACHE_PERCENTAGE` (0.7) of it |
| Free space on the filesystem holding `NDIF_RAY_TEMP_DIR` (default `/tmp/ray`) | Ray stops scheduling when that filesystem is over 95 % full, and the only symptom is a server that never runs anything. `ndif doctor` does not check it |
| `--shm-size` on the container that runs ray | Ray's object store lives in `/dev/shm` and wants the smaller of 10 GB and 30 % of RAM. With less (Docker's 64 MB default, and on a host with plenty of RAM also the `4g` the compose file sets) Ray puts it under its temp dir and logs `The object store is using /tmp/ray instead of /dev/shm ... This will harm performance!`; the server still works. `--shm-size 10g` is always enough, and it is a ceiling, not a reservation |
| Python 3.12 or 3.13 (from source) | The package requires `>=3.12,<3.14` |

**Match the image to the driver.** Tags come one per CUDA line: `-cu126` runs on
any CUDA 12.x driver, `-cu130` is for CUDA 13 drivers and Blackwell cards, and
the bare version and `latest` are the cu126 build. A torch built for a newer
CUDA line than the driver does not fail: `nvidia-smi` works in the container,
torch sees no GPU, and the ray log shows `cuda_memory_bytes: 0`. Ask before you
deploy:

```bash
docker run --rm ndif/ndif version              # ndif, nnsight, torch, transformers, ray, cuda
docker run --rm --gpus all ndif/ndif doctor    # "torch sees 1 GPU(s) (driver CUDA 12.5, torch CUDA 12.6)"
```

## The published image

`ENTRYPOINT` is `ndif` and the default command is `start --foreground`, so any
other command replaces it (`docker run --rm ndif/ndif version`), and
`docker exec ndif ndif <command>` reaches a running server.

- **`NDIF_SERVICE` picks what the container runs.** The default `all` is redis,
  rustfs, ray and api. `-e NDIF_SERVICE="all dashboard" -e
  NDIF_DASHBOARD_DEV_MODE=true -p 127.0.0.1:8081:8081` adds the admin dashboard
  with its login off. One name (`api`, `ray`, ...) runs one service, which is
  how compose and a GPU worker use the same image.
- **State in the container.** The deployed set is in the controller's memory,
  and result blobs and the Redis queue are on the container's filesystem.
  `docker restart ndif` keeps the blobs and queued requests and forgets what was
  HOT. `docker rm` discards all of it; weights survive in the mounted cache.
- **Without a bind mount for `/root/.cache/huggingface`** the image's `VOLUME`
  creates an anonymous one. `docker rm -v ndif` removes it with the container;
  plain `docker rm` leaves it behind.
- **`ndif info` inside the container lists every service as stopped.** It
  tracks processes started by a detached `ndif start`. Read its Connectivity
  lines.
- **Your own build**: from the repository root, `docker build -f
  docker/Dockerfile -t my-ndif .` gives the same image from that checkout.
  `--build-arg TORCH_CUDA=cu130` picks the CUDA line and `--build-arg
  NDIF_VERSION=<version>` the version it reports (`docker/README.md`).
- **LAN clients**, once auth and the sandbox are on: publish 8001 and 9000 on
  the interface they use and set
  `NDIF_OBJECT_STORE_PUBLIC_URL=http://<server>:9000`.

## A GPU box you share

- **Hand over specific cards**: `--gpus '"device=0,1"'`, or
  `CUDA_VISIBLE_DEVICES=0,1` from source. `ndif doctor` then reports that many.
- **The controller sizes from each card's total memory and never reads what
  other jobs hold.** `ndif status` says a card is free while someone else's
  process fills half of it. Subtract `nvidia-smi` usage yourself and place with
  `ndif deploy --gpus N` or a lower-memory `--dtype`.
- **The container is root**, so what it writes into a bind mount (new files in
  the cache, a mounted Ray temp dir) is root-owned. Remove it through a
  container: `docker run --rm -v /path:/v --entrypoint rm ndif/ndif -rf /v/<dir>`.
- **Only published ports touch the host.** 8001 and 9000 are the two a client
  needs. Redis on 6379, Ray on 6385, 8265 and 10001 and the rest listen inside
  the container's own network, so they cannot collide with host services (or
  with a second NDIF container) unless you run with `--network host`.
- **A host port already taken: change only the host side** of `-p`, and when it
  is 9000, `NDIF_OBJECT_STORE_PUBLIC_URL` as well (`SKILL.md`).
- **The cache may be mounted read-only** (`-v ...:/root/.cache/huggingface:ro`).
  Checkpoints already in it deploy and serve, nothing root-owned is written,
  and anything not cached fails `[Errno 30] Read-only file system`.
- **`HF_HUB_OFFLINE=1` breaks every deploy, cached or not.** Sizing and key
  resolution ask the Hub, so a boot deployment, `ndif deploy <repo>` and a
  first request all fail with `offline mode is enabled`. With no network, a
  `models.yaml` entry that gives both `model_key` and `size_bytes` loads from
  the cache and serves.
- **A full `/tmp`**: `-e NDIF_RAY_TEMP_DIR=/ndifray -v /big/disk/ray:/ndifray`.
  Keep the path short. Ray puts Unix sockets under it, and past 107 bytes ray
  dies at start with `OSError: validate_socket_filename failed: AF_UNIX path
  length cannot exceed 107 bytes`.

## The compose stack

```bash
git clone https://github.com/ndif-team/ndif.git && cd ndif
just up            # builds the image on the first run, then starts ten containers
just ps            # status and health
just logs api      # follow one service
```

`just` is a one-line wrapper over `docker compose -f docker/docker-compose.yml`
(`just up` is that plus `up -d`), so the stack runs without it;
`just --dry-run <recipe>` prints the command. The containers are redis, rustfs,
api, ray, dashboard, postgres, influxdb, loki, prometheus and grafana, with the
same image run three times as `api`, `ray` and `dashboard`. Beside the API it
serves the dashboard on 8081 (no login), the Ray dashboard on 8265, Grafana on
3000 (anonymous admin) and the RustFS console on 9001.

| Recipe | Use it for |
|---|---|
| `just up [svc]` | start, and apply an `environment:` edit (it recreates the container) |
| `just ta [svc]` | after a code change: down, build, up. The source is baked into the image, so nothing else picks it up. It always takes the whole stack down, and `just ta ray` then starts only `ray` and what it depends on |
| `just restart [svc]` | restart processes with their old environment. Applies neither code nor configuration |
| `just down` | remove the containers. `just down -v` also removes the named volumes |

What a first run meets:

- **Every port is published on every interface, unauthenticated**: the API,
  both dashboards, Grafana, and also Redis, Postgres, RustFS, Loki, InfluxDB,
  Prometheus and the Ray client port 10001. Anyone who reaches Redis or 10001
  runs code on the cluster. Edit each `ports:` entry in
  `docker/docker-compose.yml` to the form `"127.0.0.1:6379:6379"`. A
  `docker-compose.override.yml` is not read, because the justfile passes `-f`.
- **The project name is fixed at `dev`.** Containers are `dev-api-1`,
  `dev-ray-1`, and a second `just up` on the same host takes over an existing
  `dev` stack.
- **The weights cache is `${HOME}/.cache/huggingface`**, written as root. To
  move it, edit the `ray` service's volume line.
- **A host port already taken**: change only the host side of that `ports:`
  entry. When it is 9000, set `NDIF_OBJECT_STORE_PUBLIC_URL` to the new port on
  both `api` and `ray`, since each signs result URLs.
- **`api` waits for four health checks** (redis, rustfs, influxdb, postgres),
  though it uses the last two only when configured. An `api` that never starts
  and has no log means one of them is unhealthy in `just ps`.
- **Only the dashboard's data and the object store are named volumes.**
  Postgres (with every API key), logs and metrics do not outlive `just down`,
  and so not `just ta` either.
- **An editable nnsight on the host is bind-mounted over the image's** by
  `just up`. `just nnsight` says whether one was found; see
  [develop](develop.md#against-a-local-nnsight).

`docs/operating/compose-stack.md` reads the file service by service.

## From source

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # first, and for your driver's CUDA line
pip install "ndif[api,ray]"           # extras: api, ray, metrics, postgres, dashboard, ext, dev
ndif doctor                           # Python, binaries, GPU, connectivity; exits non-zero on a missing requirement
ndif start                            # redis, rustfs, ray, api, detached
ndif info                             # tracked PIDs and whether each endpoint answers
ndif logs ray -f                      # $NDIF_HOME/logs/<service>.log
ndif stop
```

- **torch first.** Without a wheel already installed pip takes PyPI's default
  build, which is for CUDA 13. From a checkout, `pip install -r
  requirements.txt` before `pip install ".[api,ray]"` gives the image's pins.
- **`redis-server` and `rustfs` must be on `PATH`**; `ndif doctor` names both.
  `redis-server` comes from a package manager or conda-forge. `rustfs` is a
  release binary, or copy it out of the image the Dockerfile pins:

  ```bash
  docker create --name rustfs-tmp rustfs/rustfs:1.0.1
  docker cp rustfs-tmp:/usr/bin/rustfs ~/.local/bin/rustfs && docker rm rustfs-tmp
  ```

  With an S3-compatible store of your own, set `NDIF_OBJECT_STORE_URL` and run
  `ndif start redis ray api`.
- **A `✓` from `ndif start` means the process was alive two seconds later.**
  If `/connected` is not 200 after two minutes, read `ndif logs ray`.
- **State is under `NDIF_HOME`** (`~/.ndif`): `run/<service>.pid`,
  `logs/<service>.log`, and RustFS's data. `ndif stop` stops what `ndif start`
  started and nothing else.
- **The dashboard is named explicitly**: `ndif start dashboard`, and
  `ndif stop dashboard`, since a bare `ndif stop` leaves it running and
  `ndif info` does not list it.
- **Options**: `ndif start -e KEY=VALUE` sets a variable for the services it
  starts, `ndif start api --restart` replaces a running one, and `--foreground`
  attaches for a supervisor.

## A second GPU machine

Not run for this guide: it needs two GPU hosts. The procedure, the ports and
the drain are `docs/runbooks/add-a-gpu-node.md`.

A worker is the same software with one variable set. `NDIF_RAY_HEAD_ADDRESS`
names the head's Ray port, 6385, and the node then starts only Ray and joins:

```bash
NDIF_RAY_HEAD_ADDRESS=10.0.0.5:6385 ndif start ray        # from source
docker run -d --name ndif-worker --gpus all --shm-size 10g --network host \
    -e NDIF_SERVICE=ray -e NDIF_RAY_HEAD_ADDRESS=10.0.0.5:6385 -e HF_TOKEN \
    -v ~/.cache/huggingface:/root/.cache/huggingface ndif/ndif
```

- **Same image and tag as the head.** Requests are unpickled on whichever node
  holds the replica.
- **`NDIF_SERVICE=ray` is required in a container**, or the worker also starts
  its own redis, object store and API.
- **Weights and `HF_TOKEN` are per node.** The controller passes its Redis,
  object-store and telemetry settings to actors, not the Hugging Face ones.
- **Open the nodes to each other** (6385, 8076 and Ray's unpinned ports) on a
  private network. A head started by compose publishes neither 6385 nor 8076.
- **It shows in `ndif status` within 30 seconds.** `Nodes` up but `Total GPUs`
  not means the worker reported no GPU. A worker stuck on `Waiting for Ray head
  at ...` cannot reach the head's port.
- **A node Ray reports dead, even briefly, loses its replicas** on the
  controller's next pass.
