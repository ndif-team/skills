#!/usr/bin/env python3
"""Print the module paths you need to write nnsight code against a model.

Never guess a module path. Run this first:

    python inspect_model.py openai-community/gpt2
    python inspect_model.py meta-llama/Llama-3.1-8B --prompt "The capital of France is"
    python inspect_model.py google/gemma-3-27b-it --grep attn
    python inspect_model.py AntonV/mamba2-130m-hf --task text-generation
    python inspect_model.py nvidia/Hymba-1.5B-Instruct --trust-remote-code

By default the model is built on the `meta` device (no weights downloaded beyond
the config/index), so this is cheap even for very large checkpoints. `--prompt`
adds a `model.scan(...)` pass that reports the *shape and type* of each key
module's output — which is how you find out whether `.output` is a tensor or a
tuple without loading weights.

**The task decides the class, and the class decides the tree.** With no `--task`
nnsight asks the Hub, and a repo whose `pipeline_tag` is `image-text-to-text`
builds a `...ForConditionalGeneration` whose layers are at
`model.model.language_model.layers[i]`, where `--task text-generation` builds a
`...ForCausalLM` with `model.model.layers[i]`. The `task` and `architecture`
lines at the top say which one you got; check them before copying a path.
"""

from __future__ import annotations

import argparse
import sys


def _config(model, *names, default=None):
    # Multimodal configs (Gemma3, Llava, ...) nest the LM's dims one level down.
    configs = [model.config]
    for nested in ("text_config", "llm_config", "language_model_config"):
        sub = getattr(model.config, nested, None)
        if sub is not None:
            configs.append(sub)
    for config in configs:
        for name in names:
            value = getattr(config, name, None)
            if value is not None:
                return value
    return default


def direct_children(envoy):
    """(relative_name, envoy) for the envoy's immediate children, in tree order."""
    prefix = envoy.path + "."
    children = []
    for path, child in envoy.named_modules():
        if path.startswith(prefix) and "." not in path[len(prefix) :]:
            children.append((path[len(prefix) :], child))
    return children


def mixture_children(envoy):
    """Direct children of a mixture-of-experts sub-block, or [] if it is not one.

    An MoE block's `.output` is a plain tensor, so a probe that stops at the block's
    own children reports `mlp Tensor(...)` and never names the router or the expert
    module — which are the two things you actually intervene on.
    """
    children = direct_children(envoy)
    if any(name in ("experts", "router") for name, _ in children):
        return children
    return []


def find_mixture_block(layers, count: int):
    """(index, [(relative_name, envoy)]) for the first block holding experts.

    DeepSeek-V3, GLM-4 and Ernie make their first layers dense, so block 0 is not
    representative; the router only appears further up. Only the first few blocks
    are looked at — no checkpoint keeps more than a handful dense, and walking a
    126-layer tree to find nothing is not worth the wait.
    """
    for index in range(min(count, 8)):
        block = layers[index]
        found = []
        for name, child in direct_children(block):
            found.extend((f"{name}.{inner}", grandchild)
                         for inner, grandchild in mixture_children(child))
        # Gemma-4 hangs router/experts off the decoder layer itself.
        found.extend((name, child) for name, child in mixture_children(block))
        if found:
            return index, found
    return None


def find_layer_list(model):
    """The repeated-block ModuleList (transformer.h, model.layers, ...)."""
    best = None
    for path, envoy in model.named_modules():
        module = envoy._module
        if module.__class__.__name__ != "ModuleList":
            continue
        children = list(module.children())
        if len(children) < 2:
            continue
        if len({type(c).__name__ for c in children}) != 1:
            continue
        if best is None or len(children) > best[2]:
            best = (path, envoy, len(children))
    return best


def describe(value) -> str:
    if isinstance(value, tuple):
        return "tuple(" + ", ".join(describe(item) for item in value) + ")"
    if hasattr(value, "shape"):
        return f"Tensor{tuple(value.shape)}"
    return type(value).__name__


def probe(model, prompt: str, found, dispatch: bool):
    """Report each block child's output type in true execution order.

    Registration order (what `print(model)` shows) is NOT execution order — on
    Llama the layernorms are registered after `self_attn` but run before it. The
    forward-pass order below is the order nnsight requires you to access them in.
    """
    path, layers, count = found
    block = layers[0]
    seen: list[tuple[str, str]] = []
    handles = []

    watched = [(f"{path}[0].{name}", child) for name, child in direct_children(block)]
    mixture = find_mixture_block(layers, count)
    if mixture is not None:
        index, experts = mixture
        watched.extend((f"{path}[{index}].{name}", child) for name, child in experts)

    for label, child in watched:
        module = child._module

        def hook(_module, _args, output, label=label):
            seen.append((label, describe(output)))

        handles.append(module.register_forward_hook(hook))
    handles.append(
        block._module.register_forward_hook(
            lambda _m, _a, output: seen.append((f"{path}[0]", describe(output)))
        )
    )

    runner, label = (model.trace, "model.trace — real weights") if dispatch else (
        model.scan,
        "model.scan — no weights, no compute",
    )
    print(f"## block internals in forward-pass order ({label})")
    try:
        with runner(prompt):
            pass
    except Exception as exc:  # noqa: BLE001 — surfaced to the user, not swallowed
        kind = type(exc).__name__
        print(f"  probe failed: {kind}: {str(exc).splitlines()[0]}")
        if not dispatch:
            if kind == "GuardOnDataDependentSymNode":
                print("  this pipeline branches on tensor data, which fake tensors can't supply")
            else:
                # MoE checkpoints land here in float32: _grouped_mm's meta kernel
                # rejects a dtype its real kernel accepts. Not a control-flow problem.
                print("  an op in this forward has no fake/meta kernel for the dtype in use")
            print("  re-run with --dispatch to probe with real weights")
        print()
        return
    finally:
        for handle in handles:
            handle.remove()

    width = max((len(name) for name, _ in seen), default=0)
    for name, description in seen:
        print(f"  {name:<{width}}  {description}")
    print()
    print("  access these in the order shown — a later-then-earlier read is an OutOfOrderError")
    print("  a tuple output means you index it (out[0]); a Tensor output you use directly")


def summarize(repo_id: str, prompt: str | None, grep: str | None, depth: int, dispatch: bool,
              task: str | None = None, trust_remote_code: bool = False):
    from nnsight import TransformersModel

    kwargs = {}
    if task:
        kwargs["task"] = task
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    model = TransformersModel(repo_id, dispatch=dispatch, **kwargs)

    print(f"# {repo_id}")
    print(f"model_type   {getattr(model.config, 'model_type', '?')}")
    print(f"task         {getattr(model, 'task', '?')}"
          f"{'' if task else '   (inferred from the Hub — --task builds a different class)'}")
    print(f"architecture {type(model._module).__name__}")
    print(f"layers       {_config(model, 'num_hidden_layers', 'n_layer', default='?')}")
    print(f"hidden       {_config(model, 'hidden_size', 'n_embd', default='?')}")
    print(f"heads        {_config(model, 'num_attention_heads', 'n_head', default='?')}")
    kv = _config(model, "num_key_value_heads")
    if kv is not None:
        print(f"kv heads     {kv}   (grouped-query attention — heads share KV)")
    print()

    found = find_layer_list(model)
    if found is None:
        print("no repeated-block ModuleList found; falling back to the full tree")
    else:
        path, layers, count = found
        block = layers[0]
        print("## key paths")
        print(f"layers        {path}[i]          ({count} blocks)")
        children = direct_children(block)
        for name, _ in children:
            print(f"  block child {path}[i].{name}")
        mixture = find_mixture_block(layers, count)
        if mixture is not None:
            index, experts = mixture
            where = f"{path}[{index}]" if index else f"{path}[i]"
            for name, _ in experts:
                print(f"  mixture     {where}.{name}")
            if index:
                dense = "layer 0 is" if index == 1 else f"layers 0-{index - 1} are"
                print(f"  NOTE: {dense} dense; the router first appears at [{index}]")
        shadowed = [name for name, _ in children if name in ("output", "input", "inputs")]
        if shadowed:
            print(f"  NOTE: child module(s) named {shadowed} collide with nnsight's own")
            print(f"        properties, so they move: reach them as "
                  f"{', '.join('.E_' + name for name in shadowed)}")
        for role, candidates in (
            ("embeddings", ["transformer.wte", "model.embed_tokens", "gpt_neox.embed_in", "embed_tokens"]),
            ("final norm", ["transformer.ln_f", "model.norm", "gpt_neox.final_layer_norm", "norm"]),
            ("unembed", ["lm_head", "embed_out"]),
        ):
            for candidate in candidates:
                try:
                    model.get(candidate)
                except Exception:
                    continue
                print(f"{role:<13} model.{candidate}")
                break
        print()

    if grep:
        print(f"## paths matching {grep!r}")
        for path, _ in model.named_modules():
            if grep in path:
                print(f"  {path}")
        print()

    if prompt and found is not None:
        probe(model, prompt, found, dispatch)

    if depth:
        print("\n## tree")
        for path, _ in model.named_modules():
            if path.count(".") <= depth:
                print(f"  {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id")
    parser.add_argument("--prompt", help="scan this prompt and report output shapes/types")
    parser.add_argument("--grep", help="only show module paths containing this substring")
    parser.add_argument("--depth", type=int, default=0, help="print the module tree to this depth")
    parser.add_argument("--dispatch", action="store_true", help="load real weights (needed for some remote-code models)")
    parser.add_argument("--task", help="pipeline task; required for a repo with no pipeline_tag, and it "
                                       "decides which model class (and so which tree) gets built")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="allow the repo's own modeling code; without it a custom architecture "
                             "stops at an interactive prompt and dies with EOFError")
    args = parser.parse_args(argv)

    summarize(args.repo_id, args.prompt, args.grep, args.depth, args.dispatch,
              task=args.task, trust_remote_code=args.trust_remote_code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
