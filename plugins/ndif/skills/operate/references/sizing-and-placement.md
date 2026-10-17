# Sizing, placement, and eviction

The one sentence to hold onto: **NDIF's GPU accounting is a model, not a
measurement.** The controller reserves bytes in its own per-GPU ledger from an
estimate made before anything loads, and never reads the card's free memory. A
deploy is refused when the ledger says no; it OOMs when the ledger said yes and
the card disagreed. Source of record: `docs/runbooks/model-oom-on-deploy.md`,
`docs/concepts/deployments-and-eviction.md`,
`docs/operating/models-and-deployment.md`.

## How the estimate is made

`ModelEvaluator`, memoized per model key, asks nnsight to describe the
checkpoint without loading it:

1. For a Hugging Face repo, `base` is the Hub's **published parameter count** ×
   the dtype's bytes per element. Buffers are not in that count.
2. A repo that publishes no count is built on the **meta device** (config and
   module tree, no weights) and counted instead.
3. Pad:

```text
padded_size = ceil(base + base * padding_factor + padding_bias)
```

| Knob | Env var | Default |
|---|---|---|
| proportional headroom | `NDIF_DEFAULT_PADDING_FACTOR` | `0.15` |
| flat headroom | `NDIF_DEFAULT_PADDING_BIAS` | `524288000` (500 MiB) |
| dtype driving `element_size()` | `NDIF_DEFAULT_DTYPE` | `bfloat16` |

For a 7B model in bf16: base ~14 GB, padded ~14 × 1.15 + 0.5 ≈ 16.6 GB. That is
the number the ledger reserves, and that padding is the entire budget for
activations, KV cache, CUDA workspaces and the CUDA context.

The actor holds itself to its share twice: accelerate's `max_memory` at load, and
a per-process allocator cap at run time of `share / card_total` on each assigned
card — the replica's padded **share** of that card, not size × factor. That is
why a block can die with `CUDA out of memory ... N MiB allowed` on a nearly empty
card: it hit its own budget instead of trampling a co-tenant. A sandbox runner
process is outside this cap.

The memo recomputes when the dtype or `trust_remote_code` changes, since element
sizes differ and repo code can build a different architecture. The controller pins a concrete dtype onto every `DeploymentConfig`
before evaluating so the estimate and the load cannot diverge.

## How placement decides

`gpus_needed = ceil(size / per_gpu_memory)` (or `gpus` if you set it), rounded up
to a degree the model shards into evenly when it will be tensor-parallel. Each
card the replica lands on is charged `ceil(size / gpus_needed)` — its **share**.
So a model 1.01× a card takes two cards at about half each and the other half of
both stays usable by other models. It is a step, not a cliff. Without tensor
parallelism a multi-card replica spreads whole layers over its cards with
accelerate's `device_map="balanced"`.

Worked example, a 27B bf16 model on 80 GB cards that have only 56 GB actually
free: base = 27.4e9 × 2 B = 54.8 GB, padded = 54.8 × 1.15 + 0.5 = 63.5 GB. The
ledger sees 80 GB free per card and places it on one; the card cannot hold it.
`--gpus 2` charges 31.8 GB to each card. An 8B model (16.06 GB base → 18.99 GB
padded) fits one.

`per_gpu_memory` is `cuda_memory_bytes // total_gpus`, computed once when the
node first appears. **A node with mixed card sizes is mis-accounted** — every GPU
is assumed to be the average. Node capacity is also read only once: to change a
node's advertised resources, drain it, stop Ray, and rejoin.

Nodes score themselves, lower better: `CACHED_AND_FREE` (1 — holds a WARM copy
*and* has free room) < `FREE` (2) < `CACHED_AND_FULL` (3 — WARM, but something
must be evicted) < `FULL` (4) < `CANT_ACCOMMODATE` (5 — won't fit even after
every legal eviction). Ties across nodes break randomly; the first
`CANT_ACCOMMODATE` ends that model's replica loop.

## What the ledger cannot see

Every item here is real memory on the card that the controller never subtracted:

- **Anything NDIF did not place.** The ledger starts each GPU at full capacity
  and is only ever decremented by NDIF's own placements. A stray training job, or
  a model actor left behind by a killed controller, is invisible — the single
  most common reason "the numbers said it fit".
- **The CUDA context**, roughly 400 MiB per process per device. Three actors
  sharing a card means three contexts; the 500 MiB bias covers one.
- **Controller restarts.** The ledger is in-memory; a replaced controller rebuilds
  with every GPU marked fully free while the surviving detached actors still hold
  their weights.
- **Fragmentation.** Actors run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  to blunt it, not remove it.

Actors also run with `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`, so Ray does
not mask `CUDA_VISIBLE_DEVICES` — targeting is done entirely by `max_memory` and
the per-process caps, device indices in NDIF's logs are real node-global indices,
and nothing stops user code inside a block from touching a card the replica was
not assigned.

## HOT / WARM / COLD

| Level | Weights | Actor | Serves? | Costs |
|---|---|---|---|---|
| HOT | on the assigned GPUs | alive | yes | GPU memory from the ledger |
| WARM | in host RAM on the same node | alive | no — raises `CachedActorError` | the node's CPU cache budget, plus a process slot |
| COLD | in that node's HF cache only | none | no | disk |

HOT and WARM are real controller state; COLD is synthesized for reporting.
Transitions are actor methods, not restarts: `to_cache` cancels any in-flight
execution, moves the module to CPU, lifts the per-process GPU caps and empties
the CUDA cache; `from_cache` re-applies caps for the (possibly *different*) GPUs,
recomputes a balanced device map, strips accelerate's old hooks and re-dispatches.
The replica keeps its `replica_id` across a demotion, so the Ray actor name is
stable.

**`NDIF_MODEL_CACHE_PERCENTAGE` (0.9) scales the node's host RAM into the WARM
cache budget.** It is not a GPU knob. Its one indirect GPU effect: a tight budget
makes a HOT→WARM demotion fail into a delete, so the next request pays a full
disk load.

## The eviction rules

<!-- test: skip -->
```python
def evictable(self, deployment, pinned) -> bool:
    if deployment.pinned:
        return False
    if (not pinned
            and self.minimum_deployment_time_seconds is not None
            and time.time() - deployment.deployed < self.minimum_deployment_time_seconds):
        return False
    return True
```

- A **pinned** deployment is never evicted by the controller — not to make room,
  not by autoscaling. It says nothing about an explicit `ndif evict`.
- A deployment **younger than `NDIF_MINIMUM_DEPLOYMENT_TIME_SECONDS`** (3600) is
  protected, *unless* the incoming model is itself pinned. Pinning is the
  override.

Per GPU the controller sorts evictable occupants by allocated bytes ascending and
takes the smallest until enough is freed.

What pinning does **not** do: stop `ndif evict`; survive a controller restart
(state is in-memory — it is rebuilt from `NDIF_DEPLOYMENTS` and whatever the
dashboard's reconcile cron re-pushes); keep the actor alive if it dies; or
reserve capacity for future replicas.

## Reading the failure

| Message | Means |
|---|---|
| `CANT_ACCOMMODATE: placed N of M new replicas before the cluster ran out of room.` | Nodes exist; none can fit the padded size, even after every legal eviction. The message carries no arithmetic — get the sized bytes from the controller log below. Placement stops for that model at the first failure. |
| A node-stated reason, e.g. `asked for N GPUs, but this model shards at most M ways…` | The request cannot be placed as asked. Pick a `gpus` that divides M, or `--max-tp 0` |
| `No GPU nodes available.` | The controller has zero nodes reporting a `GPU` resource — a different failure entirely. |
| An evaluator traceback (gated repo, 401, unknown repo id, no network) | Sizing failed *before* placement. Not a memory problem. Set `HF_TOKEN` wherever the deploy runs |
| An HF "requires you to execute the code" / `trust_remote_code` error | The deployment is untrusted and the repo ships custom modelling code. Redeploy `--trusted` |
| `RuntimeError: '<weight>' is on 'cpu', expected one of CUDA devices [...]` | A multi-card load's budget was too small, so accelerate offloaded the overflow to CPU and the post-load check refused to serve a half-CPU model. Read it as "the estimate was too low". |
| `torch.cuda.OutOfMemoryError` at load | The estimate was wrong, or something the ledger cannot see holds the memory. The controller logs `Deployment <model_key> failed during create: <error>`, kills the actor and returns its bytes; users waiting on the model get `Error starting model.` |
| `CUDA out of memory ... N MiB allowed` **inside a block** | The per-process allocator cap. The weights fit; the activations did not. The replica stays up. Raise padding, or save less. |

The controller is a Ray actor, so its own output does not reach
`just logs ray` — it goes to Ray's log directory inside the container:

```bash
docker compose -f docker/docker-compose.yml exec ray \
  bash -c "grep -hE 'Analyzing deployment|cannot be deployed|Deploying .* on ' \
           /tmp/ray/session_latest/logs/worker-*.out"
```

That prints the padded byte count the controller computed. Compare it with what
the cluster has:

```bash
ndif status --verbose | jq '.cluster.nodes[] |
  {name, gpus: [.resources.gpu_details[] | {index, free_gb: (.available_memory_bytes/1073741824)}]}'
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

## Making room

In order of preference:

1. **Extra replicas of a hot model** — autoscaling adds up to 3 per model and
   never removes them. `ndif evict <checkpoint> --replica <id>`.
2. **Unpinned models nobody is using** — `ndif queue` shows which models have
   live Processors; anything HOT and absent from that list is idle.
3. **WARM replicas** — they hold no GPU memory but consume the CPU cache budget,
   and a demotion needs CPU headroom to succeed at all.
4. **Pinned models** — `ndif evict` removes them like any other.

The ledger releases memory as soon as `evict` returns; there is no sync tick to
wait for.

## Right-sizing

**Measure first.** The actor records each request's *extra* GPU footprint
(`peak - baseline` per device, on top of the resident weights) as the `gpu_mem`
Influx measurement, fields `baseline_bytes` / `peak_bytes` / `extra_bytes`. Chart
`extra_bytes` for the model, take a high percentile, and compare it against
`padding_factor × base + padding_bias`. If the p99 exceeds your padding, the
model is under-provisioned and will OOM under load, not at deploy.

| Situation | Lever |
|---|---|
| Big models starved of activation room | raise `NDIF_DEFAULT_PADDING_FACTOR` (scales with the model) |
| Small models OOMing on fixed overhead | raise `NDIF_DEFAULT_PADDING_BIAS` |
| A model sits just over a one-GPU boundary | lower padding so it fits one, or accept the second card |
| Quantized deploy under-estimated | give it a measured `size_bytes` — `nf4` on Llama-3.2-1B estimates 0.62 GB against 1.07 GB really allocated, which 15% padding does not absorb |
| Nothing fits at all | add capacity — `docs/runbooks/add-a-gpu-node.md` |

The `NDIF_DEFAULT_*` values and `NDIF_MINIMUM_DEPLOYMENT_TIME_SECONDS` are read
once when the controller imports its module, so changing one means a new
controller. On compose, set it in the `ray` service's `environment:` and
**recreate**:

```bash
just up ray
```

`just restart ray` keeps the old environment; `just ta ray` runs `just down` first
and tears the whole stack down. Recreating `ray` is an outage: the controller and
every model actor go, queued and in-flight requests are errored, and only
`NDIF_DEPLOYMENTS` models (and scheduled ones, on the next reconcile) come back
on their own. The first request afterwards errors `Error starting model.` — the
API still holds its old compute connection, and its log shows a
`ConnectionError` — and the next may 503 until the API reconnects. On a
from-source install, change the environment `ndif start` runs under, then
`ndif stop ray` and `ndif start ray`.

Verify the controller picked it up:

```bash
ndif status --verbose | jq '{default_padding_factor, default_padding_bias, minimum_deployment_time_seconds}'
```

The `dtype` under `.cluster.evaluator` is torch's process default, not
`NDIF_DEFAULT_DTYPE`; read each replica's real dtype from
`ndif status --json-output`.

A per-model `padding_factor` / `padding_bias` / `size_bytes` / `gpus` raises it
for just the models that need it — via `ndif deploy` flags or a `models.yaml`
entry (the dashboard's deploy form takes `padding_factor` only). `ndif export`
does not carry the sizing fields, so add them back to an exported file.
