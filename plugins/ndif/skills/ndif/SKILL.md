---
name: ndif
description: "Run your own NDIF server, the backend behind nnsight's remote=True, on your own GPUs. Use for any question about self-hosting, operating or fixing an NDIF deployment, and when someone pastes an NDIF server log. Covers standing one up (docker run ndif/ndif, the compose stack, or pip install and `ndif start`), deploying, pinning, scaling and sizing models (ndif deploy/evict/status/queue, models.yaml, padding factor, HOT/WARM/COLD), turning on auth and the sandbox, reading the logs, and symptoms such as a container that won't start, no GPU, 503 'compute backend is reconnecting', requests stuck in QUEUED or DEPLOYING, CUDA out of memory on deploy, or a result that won't download. Also for changing the server's own code: the request lifecycle, model actors, tests and releases. Not for calling the public ndif.us service from client code, which is the nnsight skill's remote guide."
---

# NDIF

NDIF is the server behind nnsight's `remote=True`. It holds model weights on
GPUs, receives a serialized `with model.trace(...)` block from a client, runs it
against the real forward pass, and returns the saved values. This skill is for
running your own: standing a server up, running models on it, fixing it, and
changing its code. Calling the public **ndif.us** service from client code (API
keys, writing efficient remote traces) is the `nnsight` skill's remote guide.

Four services make a server. **api** takes requests over HTTP and holds the
*dispatcher*, the one process that talks to Ray and feeds each model's queue.
**ray** is a Ray head with the *controller* (which places models on GPUs) and
one *model actor* per loaded replica. **redis** carries the request queue and
status updates. An **object store** (RustFS, or S3) holds results too large to
send inline.

`ndif <command> --help` prints every flag, so this skill gives only what the
help does not. Paths in backticks that begin with `docs/` are pages of the ndif
repository, `https://github.com/ndif-team/ndif`.

## Two defaults are open

Every route below starts in this state. It is right for a server only you can
reach and wrong the moment a second person can reach port 8001.

- **No authentication, and every request is trusted.** With no
  `NDIF_POSTGRES_URL` on the API there are no API keys. A trusted request's
  block runs inside the model process, next to the weights, and may make the
  server load a checkpoint that runs its own repository code
  (`trust_remote_code`).
- **No sandbox unless a sandbox actor class is selected.** Turning auth on makes
  keys without the `trusted` tag untrusted, but an untrusted block runs in a
  separate runner process only when the model actor class is a sandbox one:
  `NDIF_MODEL_IMPORT_PATH=ndif.services.ray.sandbox.model.SandboxModelActor` on
  the `ray` service. Only the compose stack selects it. The published image and
  a from-source install run every block in-process, auth or not.

Publish ports on `127.0.0.1` until both are set. The procedure, and what the
sandbox does and does not protect, is
[production and auth](references/operate-production-and-auth.md).

## Stand a server up

| Route | Start it with | Take it when |
|---|---|---|
| Published image | `docker run --gpus all ndif/ndif` | You want a server, not a checkout. One container runs redis, the object store, ray and the api. |
| Compose stack | `just up` in a clone of the repository | You are changing NDIF, or want Postgres, the sandbox actor and the telemetry tier (Loki, InfluxDB, Prometheus, Grafana) beside it. |
| From source | `pip install "ndif[api,ray]"`, then `ndif start` | No Docker, or you want ordinary host processes. |

All three need an NVIDIA GPU and driver, and the first two the NVIDIA Container
Toolkit. None needs any configuration. The published image:

```bash
docker run -d --name ndif --gpus all --shm-size 4g \
    -p 127.0.0.1:8001:8001 -p 127.0.0.1:9000:9000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    ndif/ndif
```

- **8001** is the API, the only port a client posts to.
- **9000** is the object store, and the client fetches from it: a result over
  the inline cap (8 MiB compressed) and every non-blocking result come back as
  a presigned URL, signed for `NDIF_OBJECT_STORE_PUBLIC_URL` (default
  `http://localhost:9000`). Publish it on another host port, or serve clients on
  another machine, and that variable must name the address clients use:
  `-p 127.0.0.1:9100:9000 -e NDIF_OBJECT_STORE_PUBLIC_URL=http://localhost:9100`.
- The cache mount keeps downloaded weights when the container is removed. The
  container writes there as root.
- `--shm-size 4g` is a floor: Ray's object store lives in `/dev/shm`.
- `--gpus '"device=0,1"'` (that quoting) hands over only some cards.
  `-e HF_TOKEN` passes your token through for gated checkpoints.

```bash
curl localhost:8001/ping        # "pong": the web process is up, nothing more
curl localhost:8001/connected   # {"status":"connected"}: the dispatcher has reached Ray
docker exec ndif ndif doctor    # versions, binaries, the GPU as torch sees it, connectivity
```

Ray takes about half a minute to boot. Until then `/connected` and `/request`
answer 503 `compute backend is reconnecting` and the log prints
`Error connecting to Ray` every second; that is normal. Images are published
one per CUDA line: `docker run --rm ndif/ndif version` prints what a tag
carries, and `doctor` says whether its torch can use your driver
(`torch sees 1 GPU(s) (driver CUDA 12.5, torch CUDA 12.6)`).

Run the `ndif` commands below where they can reach the server:
`docker exec ndif ndif ...` for the image,
`docker compose -f docker/docker-compose.yml exec ray ndif ...` for compose,
plain `ndif ...` from source. The other two routes, a GPU box you share, a
second GPU machine: [selfhost](references/selfhost.md).

## Prove it works

On the client, with `pip install nnsight`:

```python
import nnsight
nnsight.CONFIG.API.HOST = "http://localhost:8001"

from nnsight.modeling.transformers import TransformersModel
model = TransformersModel("openai-community/gpt2", task="text-generation")

with model.trace("The Eiffel Tower is in the city of", remote=True):
    hidden = model.transformer.h[-1].output.save()

print(hidden.shape)   # torch.Size([1, 10, 768])
```

Nothing has to be deployed first. The first request for a checkpoint loads it,
and the client shows `RECEIVED`, `PROVISIONING`, `DEPLOYING` (weights
downloading and loading), `QUEUED`, `DISPATCHED`, `RUNNING`, `COMPLETED`; later
requests go straight to `QUEUED`. No API key is needed while auth is off.

A client on another machine needs two addresses: the API in `CONFIG.API.HOST`,
and the object store under exactly the host and port in
`NDIF_OBJECT_STORE_PUBLIC_URL`, because a presigned URL's signature covers
them. With the ports on loopback as above, tunnel both:
`ssh -fN -L 8001:localhost:8001 -L 9000:localhost:9000 user@server`.

## Run models

```bash
ndif status                                    # HOT / WARM / COLD models, and free GPU memory by NDIF's accounting
ndif deploy meta-llama/Llama-3.1-8B --pinned   # load now, and never evict it to make room
ndif evict meta-llama/Llama-3.1-8B             # unload every replica, pinned or not
ndif queue                                     # per model: replicas, queue depth, the request each replica is running
ndif kill <request_id>                         # cancel a queued or a running request
```

- **HOT** is on a GPU and serving. **WARM** is parked in host RAM with its actor
  alive, quick to bring back, and does not serve. **COLD** is a checkpoint found
  in the server's Hugging Face cache and nothing more, so a shared cache lists
  models nobody deployed.
- **Deploy is additive.** `ndif deploy gpt2` twice gives two replicas. Shrink
  with `ndif evict`, or reconcile to a file with `ndif deploy -f models.yaml --sync`.
- **The model key is the identity.** Deploy prints it:
  `nnsight.modeling.transformers.TransformersModel:{"repo_id": "openai-community/gpt2", "revision": null, "task": "text-generation"}`.
  The revision, the pipeline task and `trust_remote_code` are part of it, and the
  client sends the key it computed, so a model deployed with `--task`,
  `--revision` or `--trust-remote-code` is reached only by a client that loads
  with the same, and later `evict`, `scale` and `restart` need the same flags.
- **Pinned means the controller will not evict it.** To make room the
  controller evicts only replicas that are unpinned and at least an hour old;
  an incoming `--pinned` deploy may also evict younger unpinned ones. `ndif
  evict` ignores both rules.
- **Read the output.** A deploy that fails prints `✗ <model_key>: <reason>` and
  exits non-zero. `ndif evict` and `ndif restart` that match nothing print a
  `✗` line and exit 0.
- **The deployed set lives in memory.** Restarting the `ray` service forgets it;
  only the model keys in `NDIF_DEPLOYMENTS` are loaded, pinned, at boot.

`curl localhost:8001/status` returns the same JSON as `ndif status
--json-output`. Scaling, restarting, `models.yaml`, loading at boot, the
dashboard, quantized and tensor-parallel deployments:
[operate](references/operate.md).

## Will a model fit

NDIF keeps a ledger of reserved bytes per GPU and never measures the card. A
deploy is refused when the ledger says no, and runs out of memory when the
ledger said yes and the card disagreed.

```text
reserved = ceil(base * (1 + padding_factor) + padding_bias)
base     = parameter count * bytes per element (2 for bfloat16, the default dtype)
defaults : padding_factor 0.15, padding_bias 524288000 (500 MiB)
cards    = ceil(reserved / bytes per card), each charged reserved / cards
```

The count is the one the Hub publishes:
`curl -s https://huggingface.co/api/models/meta-llama/Llama-3.1-8B | jq .safetensors.total`
prints `8030261248`. So Llama-3.1-8B has a base of 16.06 GB and reserves
18,993,889,871 bytes (18.99 GB). A 70.6 B-parameter model reserves 162.8 GB:
on 48 GB cards, which report 50.9 GB to torch, that is `ceil(162.8 / 50.9)` =
4 cards at 40.7 GB each.

- `ndif status` prints the ledger (`GPU Memory: 46.7 / 47.4 GB free`, in GiB);
  `ndif status --verbose` has each card's `memory_bytes` and
  `available_memory_bytes`; after a deploy, `ndif status --json-output` has
  each replica's `size_bytes` and `gpus`.
- The ledger cannot see anything NDIF did not place. Compare `nvidia-smi`
  before trusting it on a shared card, and force the placement with
  `ndif deploy --gpus N`.
- The padding is the replica's whole budget for activations, and each replica
  is capped at its reservation. A block that allocates more fails with
  `CUDA out of memory ... 800.55 MiB allowed` on an almost empty card, and the
  replica keeps serving. Raise it per model with `--padding-factor` or
  `--padding-bias`.
- `CANT_ACCOMMODATE` means no node has room after every eviction the rules
  allow: evict something, deploy `--pinned`, or lower the reservation with
  `--dtype` or a measured `--size-bytes`.

Why a deploy was refused, what the ledger misses, and how to change the
defaults: [sizing and placement](references/operate-sizing-and-placement.md).

## When something is wrong

The logs are in two places, and the second is the one people miss.

| Route | API, dispatcher and queue (`ndif.api`, `ndif.queue.*`, `ndif.request`) | Controller and model actors (`ndif.controller`, `ndif.modeling`) |
|---|---|---|
| Image | `docker logs ndif` | `docker exec ndif bash -c "grep -h <text> /tmp/ray/session_latest/logs/worker-*.out"` |
| Compose | `just logs api` | the same files inside the `ray` container; `just logs ray` has only Ray's boot. Or Grafana and Loki, `{service="model"}` |
| Source | `ndif logs api` | `$NDIF_RAY_TEMP_DIR/session_latest/logs/worker-*.out` (default `/tmp/ray`) |

| What you see | Look here first |
|---|---|
| `docker run` exits with `could not select device driver "" with capabilities: [[gpu]]` | The NVIDIA Container Toolkit is missing or Docker was not restarted after installing it: `sudo nvidia-ctk runtime configure --runtime=docker`, restart Docker |
| 503 `compute backend is reconnecting`, or `/connected` not 200 two minutes after start | The dispatcher has no Ray connection. Read the ray log from its first error after `Starting Ray head node`; the api log repeats `Error connecting to Ray` |
| `cuda_memory_bytes: 0` in the ray log, `Nodes: 0` in `ndif status`, a deploy fails `No GPU nodes available.` | torch sees no GPU. `ndif doctor`: no `--gpus`, or an image whose CUDA line is newer than the driver |
| A request stays in `PROVISIONING` or `DEPLOYING` | Weights are downloading or loading and there is no timeout. `ndif status` shows `DEPLOYING`; the controller log shows a failed load |
| A request stays in `QUEUED` | `ndif queue`: what the replica is running and for how long. `ndif kill <id>` stops it |
| A request stays in `RECEIVED` and `ndif queue` prints `No response from the dispatcher` | The dispatcher process is gone though `/connected` still says connected. Restart the api |
| `Could not deploy this model. CANT_ACCOMMODATE...`, or `Error starting model.` | The ledger is full, or the load failed. The controller log has `Analyzing deployment ... with size <bytes>` and `failed during create` |
| `CUDA out of memory ... MiB allowed` inside a block | The replica's own cap, above. Not a full card |
| `COMPLETED`, then the client cannot download the result | `NDIF_OBJECT_STORE_PUBLIC_URL` is not an address the client reaches, or port 9000 is not published |
| `The model architecture on this server doesn't match...` or `Your request payload could not be read (...)` | Client and server package versions differ: `ndif env` against the client's `pip show nnsight transformers` |

## Guides

| To | Read |
|---|---|
| run the compose stack or a from-source install, share a GPU box, reach the server from another machine, or add a second GPU machine | [selfhost](references/selfhost.md) |
| change a setting: which variable, on which service, how it takes effect, which ports and HTTP routes exist, what survives a restart | [configuration](references/selfhost-configuration.md) |
| pin, scale, restart or evict a model, keep a `models.yaml`, load models at boot, kill a request, use the dashboard, serve a quantized or tensor-parallel model or a PEFT adapter | [operate](references/operate.md) |
| work out why a deploy was refused or ran out of memory, make room, or change the padding | [sizing and placement](references/operate-sizing-and-placement.md) |
| let other people use the server: API keys, tags, the sandbox, the dashboard login, what NDIF does not provide | [production and auth](references/operate-production-and-auth.md) |
| fix a server that will not start, hangs, runs out of memory or returns nothing | [troubleshoot](references/troubleshoot.md) |
| decode an error message a user or a log gave you | [error messages](references/troubleshoot-error-messages.md) |
| follow one request through the logs, query Loki, read the metrics and Grafana | [reading logs](references/troubleshoot-reading-logs.md) |
| change the server's code: where a request goes, where code belongs, a new model actor, the tests, a release | [develop](references/develop.md) |
