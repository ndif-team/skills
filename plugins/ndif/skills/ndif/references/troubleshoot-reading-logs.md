# Reading the logs and the metrics

`SKILL.md` has where each route keeps its logs. The split to remember: the API
container's output has the API, dispatcher and queue; the controller and the
model actors write to Ray's worker logs,
`/tmp/ray/session_latest/logs/worker-*.out` where ray runs.

## Who logs what

| Logger | Process | Tells you |
|---|---|---|
| `ndif.api` | API workers | refusals (`api request rejected: 403 path=/request ...`) and unhandled 500s |
| `ndif.request` | wherever the request is | one record per status change, with `stage`, `prev_stage`, `prev_stage_ms` |
| `ndif.queue.dispatcher` | dispatcher | `Error connecting to Ray`, `Connected to Ray`, reconnects |
| `ndif.queue.processor` | dispatcher | `request enqueued` with `queue_size` and `replicas`, autoscaling, provisioning errors |
| `ndif.queue.replica` | dispatcher | evictions and re-queues, `request errored during execution` |
| `ndif.controller` | Ray head | `Analyzing deployment ... with size`, `Deploying ... on <node> because FREE`, `Evicting`, `failed during create` |
| `ndif.modeling` | model actor | `GPU 0: limited to N bytes`, `Loaded ... onto GPUs [0]`, `model execution timed out` / `cancelled` / `errored` |

A console record is one line with its fields appended as `key=value`:

```text
2026-11-17 01:48:39 INFO    [ndif.request] Added to Queue at position 1. model_key=... request_id=eb2280a6d7fe4dea82f4575889dcfcc6 api_key=local stage=queued prev_stage=received prev_stage_ms=1.76 (request.py:177)
```

The Ray dashboard on 8265 (Actors) shows each model actor with its node, pid
and log. Actors are named `<replica id>:ModelActor:<model key>` in the `NDIF`
namespace. NDIF does not use Ray Serve, so there is no `serve status`.

## One request, end to end

The client prints the request id in brackets on every status line; `ndif
queue` lists the ids that are queued or running. Then grep both places:

```bash
docker logs ndif 2>&1 | grep <request id>
docker exec ndif bash -c "grep -h <request id> /tmp/ray/session_latest/logs/worker-*.out"
```

`ndif.request` gives the timeline. A cold request, from the telemetry tier:

```text
service  stage         prev_stage    prev_stage_ms
api      received
api      provisioning  received      192.58
api      deploying     provisioning  1102.72
api      queued        deploying     9601.21
api      dispatched    queued        5.14
model    running       dispatched    16.47
model    completed     running       412.91
```

The last stage reached says who owns the failure:

| Last stage | Owner |
|---|---|
| none | Refused at the door: look for `api request rejected` under `ndif.api` |
| `received` | Never taken off the Redis list: the dispatcher |
| `provisioning`, `deploying` | Placing or loading the model failed: the controller log |
| `queued` | Waiting behind other work, or purged with its queue |
| `dispatched` | The call to the actor failed, or the replica was evicted under it |
| `running` | The block failed, timed out or was killed: `ndif.modeling` has the record and, for an error, the traceback |

What is left to look at afterwards depends on how the job was submitted:

| | Blocking (the default) | Non-blocking |
|---|---|---|
| Statuses reach the client | over Redis pub/sub to its websocket | by polling `GET /response/{id}` |
| Kept on the server | nothing | the latest status, as `responses/<id>.json` in the object store |
| The user's `print()` | streamed as `LOG` statuses | dropped |
| The result | `<id>.pt` only when it was too large to send inline | `<id>.pt`, only on success |

So for a blocking job the logs and metrics are the only record. Without auth
there is no `email` on any of them; per-user attribution is a reason to turn
auth on.

## Loki

With `NDIF_LOKI_URL` set (compose sets it), every record is also shipped as
JSON. The stream labels are few on purpose: `service` (`api`, `ray`, `model`),
`environment`, `logger`, `severity`, `model_key`. Everything else
(`request_id`, `replica_id`, `email`, `api_key`, `stage`, `exec_ms`, `host`,
`pid`) is in the JSON line and needs `| json`.

```logql
{service="model"} | json                                                    # every model actor
{environment=~".+", logger="ndif.request"} | json | request_id=`<id>`       # one request's timeline, across services
{logger="ndif.request"} | json | stage=`error`                              # every failed request
```

`service="model"` appears in no configuration file: the controller sets it on
each actor it creates. `{service="ray"}` has the controller and the Ray node
only. In Grafana (3000 on compose) `request_id` in a log line is a link to
every record for that request.

## Metrics

With `NDIF_INFLUX_URL` set, six InfluxDB measurements, all tagged `service`,
`environment` and `model_key`, the per-request ones also `api_key` and `email`,
with `request_id` as a field:

| Measurement | Records | Fields |
|---|---|---|
| `request_size` | the payload at the door | `payload_bytes`, `ip_address`, `user_agent` |
| `status_time` | time spent in each status, tagged by the status left | `duration_ms` |
| `execution_time` | time on the model actor, tagged `status` (`completed`, `error`, `timeout`, `cancelled`, `preempted`) | `deserialize_ms`, `exec_ms`, `upload_ms` |
| `gpu_mem` | GPU memory a request used above the resident weights, per `gpu_index` | `baseline_bytes`, `peak_bytes`, `extra_bytes` |
| `response_size` | the result blob | `response_bytes`, `compressed` |
| `model_load_time` | weights onto the GPU, tagged `load_type` (`initial`, `from_cache`) | `duration_ms`, `num_gpus` |

- A large `status_time` for `QUEUED` beside a successful run is a user who saw
  "hung", not "failed".
- A large `deserialize_ms` is a large serialized block; a large `upload_ms` a
  large result.
- `gpu_mem.extra_bytes` is what to compare with a model's padding.
- No metrics is not evidence of no requests: the writer is off without the URL
  and fails open with it.

The compose Grafana ships nine dashboards over these (overview, requests,
latency, errors, GPU and model load, sizes, users, Ray). Prometheus holds only
Ray's own metrics. `docs/operating/observability.md` has the panels and more
queries.
