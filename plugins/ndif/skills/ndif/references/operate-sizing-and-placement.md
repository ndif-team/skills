# Sizing, placement and eviction

`SKILL.md` has the formula and how to read the ledger. This page is how the
estimate and the placement are made, what the ledger cannot see, how to read a
refusal, and how to change the defaults.

## The estimate

The controller sizes a model from its key alone, before anything loads, and
remembers the answer per key and dtype.

1. `base` is the parameter count the Hub publishes for the repository times
   the dtype's bytes per element. A repository that publishes no count is built
   on the meta device (structure, no weights) and counted.
2. `reserved = ceil(base + base * padding_factor + padding_bias)`.
3. `--size-bytes` replaces step 1 with a number you measured. Padding still
   applies on top, and the deploy then works with the Hub unreachable.

```bash
ndif status --json-output | jq -c '.deployments[] | select(.model_key) | {repo_id, replica_id, dtype, n_params, size_bytes, gpus, pinned}'
# {"repo_id":"openai-community/gpt2","replica_id":"05052","dtype":"bfloat16","n_params":137022720,"size_bytes":839440256,"gpus":1,"pinned":false}
```

137,022,720 parameters at 2 bytes is 274,045,440; times 1.15, plus 524,288,000,
is the 839,440,256 reserved.

The actor is held to its reservation twice: as accelerate's `max_memory` when
it loads across several cards, and as a per-process allocator cap of its share
on each card when it runs, which is the `N MiB allowed` in an out-of-memory
error. A sandbox runner is a separate process and outside that cap.

## Placement

- **Cards**: `ceil(reserved / bytes per card)`, or `--gpus N`. With tensor
  parallelism on, the count is rounded up to one the model shards into evenly.
- **Each card is charged its share**, `ceil(reserved / cards)`, so a model a
  little over one card takes two at about half each and the rest of both stays
  usable.
- **Bytes per card is the node's total GPU memory divided by its card count**,
  read once when the node first appears. A node with mixed card sizes is
  mis-accounted, every card being taken for the average, and a node whose
  hardware changes has to leave Ray and rejoin.
- **Across several cards without tensor parallelism**, whole layers are spread
  with accelerate's balanced device map.
- **Among nodes**, one that holds a WARM copy and has free room wins, then one
  with free room, then one where something must be evicted. Ties break at
  random. Biggest models are placed first, and the first refusal ends that
  model's remaining replicas.

## What the ledger cannot see

Each of these is real memory on the card that the controller never subtracted.

- **Anything NDIF did not place**: another job, or a model actor left behind
  by a controller that was restarted. This is the usual reason "the numbers
  said it fit".
- **The CUDA context**, a few hundred MiB per process per card. A WARM replica
  keeps its process and its context. The 500 MiB bias covers one.
- **A restarted controller.** The ledger is in memory, so a new controller
  starts with every card marked free.
- **User code.** Actors see every GPU on the node, so a block can allocate on a
  card its replica was not given.

## Reading a refusal

| Message | Means |
|---|---|
| `CANT_ACCOMMODATE: placed 0 of 1 new replicas before the cluster ran out of room.` | Nodes exist and none can fit the reservation after every eviction the rules allow. It carries no arithmetic: the controller log has `Analyzing deployment of <key> ... with size <bytes>` |
| `asked for N GPUs, but this model shards at most M ways...` | The placement asked for cannot be made. Choose a divisor of M, or `--max-tp 0` |
| `No GPU nodes available.` | The controller has no node that reports a GPU. Not a memory problem: `ndif doctor` |
| A Hub error (repository not found, gated repo, 401, no network) | Sizing failed before placement. Fix the name, set `HF_TOKEN` where the deploy runs, or pass `--size-bytes` |
| `initialization failed - ...` with `torch.OutOfMemoryError` | The load ran out of memory: the estimate was low, or the card held something the ledger cannot see. The controller logs `Deployment <key> failed during create`, removes the actor and returns its bytes. A client waiting on the model gets `Error starting model.` |
| `RuntimeError: '<weight>' is on 'cpu', expected one of CUDA devices [...]` at load | A multi-card load overflowed its budget onto the CPU and the actor refused to serve it. Read it as "the estimate was too low" |
| `CUDA out of memory ... N MiB allowed` inside a block | The replica's own cap. The weights fit and the activations did not; the replica keeps serving. Raise the padding, or save less |

The controller's log is in Ray's worker logs, not the container's:

```bash
docker exec ndif bash -c "grep -hE 'Analyzing deployment|cannot be deployed|Deploying .* on |Evicting|failed during' /tmp/ray/session_latest/logs/worker-*.out"
# ... [ndif.controller] => Analyzing deployment of <key> (replica 1/1) with size 839440256...
# ... [ndif.controller] => Deploying <key> (replica 1/1) on 172.17.0.2 because FREE. Requiring evictions: []
```

Compare that size with what each card has by the ledger and by the driver:

```bash
ndif status --verbose | jq '.cluster.nodes[] | {name, gpus: [.resources.gpu_details[] | {index, free_gb: (.available_memory_bytes/1e9)}]}'
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

## Making room

1. Extra replicas the autoscaler added: `ndif evict <checkpoint> --replica <id>`.
2. Unpinned models nobody is using. `ndif queue` lists models that have had
   traffic and what each replica is running.
3. Pinned models, which `ndif evict` removes like any other.

WARM replicas hold no GPU memory. The ledger is updated as soon as `evict`
returns, and the controller evicts by itself only under the rules in
[operate](operate.md#pinning-eviction-and-autoscaling).

## Right-sizing

| Situation | Change |
|---|---|
| Blocks on a big model run out of activation room | `--padding-factor`, which scales with the model |
| Small models fail on fixed overhead | `--padding-bias` |
| A model lands just over one card | Lower its padding so it fits one, or accept the second card |
| A quantized model, or any whose estimate is wrong | `--size-bytes`, measured |
| A shared card with less free than its total | `--gpus N`, to spread the reservation |
| Nothing fits | Another GPU node: [selfhost](selfhost.md#a-second-gpu-machine) |

With the telemetry tier running, measure before guessing: the `gpu_mem`
measurement records each request's `extra_bytes` above the resident weights.
Compare a high percentile of it with `padding_factor * base + padding_bias`.

The same fields work per model in `models.yaml` (`padding_factor`,
`padding_bias`, `size_bytes`, `gpus`, `max_tp`). `ndif export` does not carry
them, so add them back to an exported file.

## Changing a controller variable

The padding defaults, `NDIF_DEFAULT_DTYPE`,
`NDIF_MINIMUM_DEPLOYMENT_TIME_SECONDS`, `NDIF_MODEL_CACHE_PERCENTAGE` and the
actor classes are read once when the controller starts, so a new value needs a
new controller, and that is an outage: every model actor goes with it, and only
`NDIF_DEPLOYMENTS` and the dashboard's schedule come back by themselves.

| Route | Do |
|---|---|
| Compose | edit the `ray` service's `environment:`, then `just up ray`. Not `just restart ray` (old environment) and not `just ta ray` (whole stack down) |
| Image | `docker rm -f ndif`, then `docker run` again with the new `-e` |
| Source | change the environment, `ndif stop ray`, `ndif start ray` |

For the next twenty seconds or so requests fail with `Error submitting request
to model deployment.`, `Error starting model.` or a 503, until the API's
dispatcher has reconnected to the new Ray. Then check the controller took it:

```bash
ndif status --verbose | jq '{default_padding_factor, default_padding_bias, minimum_deployment_time_seconds, model_cache_percentage}'
# {"default_padding_factor": 0.3, "default_padding_bias": 524288000, "minimum_deployment_time_seconds": 3600, "model_cache_percentage": 0.7}
```

The `dtype` under `.cluster.evaluator` there is torch's process default, not
`NDIF_DEFAULT_DTYPE`; each replica's real dtype is in `ndif status
--json-output`.

`NDIF_MODEL_CACHE_PERCENTAGE` is the share of host RAM WARM models may hold. It
does nothing for a GPU shortage. Set too low, an evicted replica has nowhere to
park and the next request pays a load from disk.

`docs/runbooks/model-oom-on-deploy.md` and
`docs/concepts/deployments-and-eviction.md` go further.
