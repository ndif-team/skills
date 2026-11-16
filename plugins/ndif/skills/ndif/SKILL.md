---
name: ndif
description: "Run your own NDIF server, the backend behind nnsight's remote=True, on your own GPUs. Use for any question about self-hosting, operating or fixing an NDIF deployment, and when someone pastes an NDIF server log. Covers standing one up (docker run ndif/ndif, the compose stack, or pip install and `ndif start`), deploying, pinning, scaling and sizing models (ndif deploy/evict/status/queue, models.yaml, padding factor, HOT/WARM/COLD), turning on auth and the sandbox, reading the logs, and symptoms such as a container that won't start, no GPU, 503 'compute backend is reconnecting', requests stuck in QUEUED or DEPLOYING, CUDA out of memory on deploy, or a result that won't download. Also for changing the server's own code: the request lifecycle, model actors, tests and releases. Not for calling the public ndif.us service from client code, which is the nnsight skill's remote guide."
---

# NDIF

NDIF is the server behind nnsight. It holds model weights on GPUs, receives a
serialized `with model.trace(...)` block from a client, runs it against the real
forward pass, and sends the saved values back. This skill is for running your
own: standing a server up, running models on it, fixing it, and changing its
code.

Calling the public **ndif.us** service from client code (API keys, model
availability, writing efficient remote traces) is the `nnsight` skill's remote
guide, not this skill.

## Guides

Read the guide that matches the task before acting. Each one is self-contained.

| Task | Guide | Depth |
|---|---|---|
| Stand a server up: the published `ndif/ndif` image, the compose dev stack, or a from-source `ndif start`; prerequisites, tags, ports, volumes, the first remote trace | [selfhost](references/selfhost.md) | [routes](references/selfhost-routes.md), [configuration](references/selfhost-configuration.md) |
| Run models on it: deploy, evict, pin, scale, `models.yaml`, sizing and padding, HOT/WARM/COLD, the dashboard, telemetry, auth | [operate](references/operate.md) | [CLI commands](references/operate-cli-commands.md), [sizing and placement](references/operate-sizing-and-placement.md), [production and auth](references/operate-production-and-auth.md) |
| Something is broken: it won't start, requests hang, a deploy runs out of memory, a result won't download, versions disagree | [troubleshoot](references/troubleshoot.md) | [error messages](references/troubleshoot-error-messages.md), [reading logs](references/troubleshoot-reading-logs.md) |
| Change the server's code: the request lifecycle, the process map, trusted and untrusted execution, the model-actor hooks, the test suite, releases | [develop](references/develop.md) | [request lifecycle](references/develop-request-lifecycle.md), [extending](references/develop-extending.md) |

Paths in backticks that begin with `docs/` are pages in the ndif repository,
`https://github.com/ndif-team/ndif`.

## Before anyone else can reach the server

Both defaults are open. With no `NDIF_POSTGRES_URL` there is no authentication
and every request is trusted, so a caller's code runs inside the model process,
next to the weights. Turning auth on does not sandbox anything by itself: only a
sandbox model-actor class runs an untrusted request in a separate process, and
only the compose stack selects one. Read
[Read this before you start](references/selfhost.md#read-this-before-you-start)
before a second person can reach port 8001.
