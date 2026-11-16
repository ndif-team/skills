# The three routes, in full

Each section is self-contained: commands, what you get, and the traps specific
to that route. The source of record is `docs/operating/quickstart.md` (all
three), `docker/README.md` (route 1, including the tag table),
`docs/operating/compose-stack.md` (route 2) and `docs/operating/cli.md`
(route 3, including what `ndif doctor` checks).

## Route 1 — `docker run ndif/ndif`

One container runs the whole request path. `NDIF_SERVICE` defaults to `all`,
which resolves to **redis, rustfs, ray, api** — the image carries its own
`redis-server` and `rustfs` binaries.

```bash
docker run -d --name ndif --gpus all --shm-size 4g \
    -p 127.0.0.1:8001:8001 -p 127.0.0.1:9000:9000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN \
    ndif/ndif
```

`ENTRYPOINT` is `ndif` and `CMD` is `start --foreground`, so replacing the
command runs any other CLI verb (`docker run --rm ndif/ndif version`,
`docker run --rm -e NDIF_SERVICE=api ndif/ndif` for one service). Against a
running container, `docker exec ndif ndif <verb>`.

- **Without a bind mount for `/root/.cache/huggingface`** the image's own
  `VOLUME` gives you an anonymous one: weights survive a restart, not a
  `docker rm`.
- **`--shm-size 4g` is a floor.** Ray's plasma store lives in `/dev/shm`.
- **`--gpus all` hands over every GPU;** `--gpus '"device=0,1"'` restricts it.
- **The image runs untrusted code in-process even with auth on** — set
  `NDIF_MODEL_IMPORT_PATH` to the sandbox actor before anyone else submits
  (`docs/operating/production.md#the-sandboxed-actor-class`). The sandbox is
  process separation, not a hardened jail (same user, filesystem, network and
  GPUs); it keeps untrusted requests from reading or editing weights and from
  training.
- **Weights land under `<cache>/hub/models--<org>--<name>`** in the mounted
  cache.
- **A port already taken: remap only the host side**
  (`-p 127.0.0.1:<newport>:<port>`). Moving 9000 also needs
  `-e NDIF_OBJECT_STORE_PUBLIC_URL=http://localhost:<newport>`.
- **The dashboard is opt-in:** `-e NDIF_SERVICE="all dashboard"` plus
  `-p 127.0.0.1:8081:8081`.

## Route 2 — the compose dev stack

Needs `just` (`curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin`,
with `~/.local/bin` on `PATH`).

```bash
git clone https://github.com/ndif-team/ndif.git && cd ndif
just up
```

The first `just up` builds one image and runs it three times — as `api`, `ray`
and `dashboard` — selected per container by `NDIF_SERVICE`. Expect ten minutes
or more, ~8 GB of image and ~8 GB of build cache under Docker's data root.
`just --list` shows the recipes; each is a one-line
`docker compose -f docker/docker-compose.yml ...`.

| Recipe | Use it for |
|---|---|
| `just up [svc...]` | start, **and** apply an `environment:` edit (it recreates the container) |
| `just ta [svc...]` | after any `src/` change: down → build → up. Always takes the whole stack down first |
| `just restart [svc...]` | restart processes with their *existing* environment — applies neither code nor config |
| `just nnsight` | print which nnsight would be bind-mounted |

What a fresh install runs into:

- **Every port is published on `0.0.0.0`**, unauthenticated: the API, RustFS,
  the admin dashboard (dev mode, no login), the Ray dashboard, Grafana, and also
  Redis 6379, Postgres 5432, Loki, Influx, Prometheus and the Ray client port
  10001 — anyone who reaches Redis or 10001 runs code on the cluster. Edit each
  `ports:` entry in `docker/docker-compose.yml` to `"127.0.0.1:6379:6379"` form.
  A `docker-compose.override.yml` is not read: the justfile passes `-f`.
- **The project name is fixed at `dev`.** An existing `dev` stack on the host is
  taken over, and existing `dev-*` images are retagged by the next build.
- **The HF cache path is** `${HOME}/.cache/huggingface`, written as root;
  weights land under `<cache>/hub/models--<org>--<name>`, and a shared cache
  makes every model in it appear as COLD. To move it, edit the `ray` service's
  volume line in `docker/docker-compose.yml`, or symlink the host path.
- **A host port already taken:** remap only the host side of the `ports:`
  entry, `"127.0.0.1:<newport>:<port>"`. If 9000 is remapped, set
  `NDIF_OBJECT_STORE_PUBLIC_URL: http://localhost:<newport>` on **both** `api`
  and `ray` — each signs presigned URLs.
- **`api` waits on `postgres` and `influxdb` health checks** even though it runs
  without either. If `api` never starts and has no logs, check those two in
  `just ps`.
- **Only `dashboard_data` is a named volume.** Result blobs, metrics, logs and
  the Postgres database (with every API key) are lost on `just down`; weights
  survive in the host HF cache.

**A local nnsight** is mounted by `docker/docker-compose.nnsight.yml`. Set
`NNSIGHT_PATH=<checkout>/src/nnsight` — the package directory — and carry it on
every `just` call. Auto-detection without it runs `python -c "import nnsight"`
and needs an editable nnsight; a host with only `python3` always detects
nothing, silently. **`just nnsight` is the only reliable pre-check** —
`just --dry-run up` always shows the overlay file. A fresh checkout lacks the
compiled `_c` extension, so every remote `.save()` fails with
`AttributeError: 'Tensor' object has no attribute 'save'` until you build it:

```bash
docker run --rm --entrypoint python -v <checkout>:/src -w /src dev-ray setup.py build_ext --inplace
```

It runs as root, so the build artifacts it leaves in the checkout are
root-owned.

`nnsight.__version__` reports the image's version under the mount; check
`grep nnsight /proc/self/mountinfo` inside the container instead. A mounted line:

```text
1517 1490 259:2 /u/me/nnsight/src/nnsight /usr/local/lib/python3.12/site-packages/nnsight rw,relatime - ext4 /dev/nvme0n1p2 rw
```

The fourth column, the source, is relative to the root of its filesystem, so it
need not match the host path you exported.
`docs/developing/nnsight-integration.md#developing-against-a-local-nnsight`.

## Route 3 — from source with the `ndif` CLI

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126   # FIRST
pip install -r requirements.txt                                         # from a checkout: the image's exact pins
pip install ".[api,ray,metrics,postgres,dashboard]"
```

torch goes first: with no wheel present pip takes PyPI's default, the CUDA 13
build, and torch is deliberately not in `requirements.txt`. Without a checkout,
`pip install "ndif[api,ray]"` after torch. The extras gate `api`, `ray`,
`metrics`, `postgres` (API-key auth), `dashboard`, `dev`.

```bash
ndif doctor
ndif start                 # redis, rustfs, ray, api in dependency order, detached
ndif start dashboard       # opt-in; a bare `ndif start` never pulls it in
ndif info                  # tracked PIDs + reachability of each endpoint
ndif stop
```

State lives under `NDIF_HOME` (`~/.ndif`). `ndif stop` only knows about
processes it started.

- **`ndif doctor` requires `redis-server` and `rustfs` on `PATH`.**
  `redis-server`: `conda install --override-channels -c conda-forge redis-server`.
  `rustfs`: a release from https://github.com/rustfs/rustfs/releases (zip,
  `.deb` or `.rpm`), or lift it from the image the Dockerfile pins:

  ```bash
  cid=$(docker create rustfs/rustfs:1.0.1)
  docker cp "$cid:/usr/bin/rustfs" /usr/local/bin/rustfs && docker rm "$cid"
  ```

  With an existing S3-compatible store, point `NDIF_OBJECT_STORE_URL` at it and
  run `ndif start redis ray api`.
- **Keep `NDIF_RAY_TEMP_DIR` short.** Ray's unix sockets go under
  `<dir>/session_<ts>_<pid>/sockets/` and AF_UNIX paths cap at 107 bytes; a long
  dir makes `ray start` die with `validate_socket_filename failed` in the ray log.
- **A ✓ from `ndif start` means the process was alive two seconds later.** If
  `/connected` still says `reconnecting` after two minutes, check `ndif info`
  and `$NDIF_HOME/logs/ray.log`.
- **No sandbox by default.** Nothing sets the actor class, so untrusted code runs
  in-process until you set `NDIF_MODEL_IMPORT_PATH` (and even then the sandbox
  is process separation, not a hardened jail — it stops weight reads, edits and
  training, not access to the user, filesystem, network or GPUs).

## More than one GPU machine

A node joins an existing head with `NDIF_RAY_HEAD_ADDRESS=<head>:6385` and
`ndif start ray` (or `NDIF_SERVICE=ray` in the image). Ports, the compose-head
caveat, verification and drain: `docs/runbooks/add-a-gpu-node.md`.
