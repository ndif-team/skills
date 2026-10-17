#!/usr/bin/env python3
"""Print what nnterp makes of a checkpoint before you write a trace against it.

    python inspect_family.py openai-community/gpt2
    python inspect_family.py HuggingFaceTB/SmolLM2-135M-Instruct --layer 5
    python inspect_family.py Qwen/Qwen3.5-9B --eager
    python inspect_family.py meta-llama/Llama-3.1-70B --eager --layer 40
    python inspect_family.py llava-hf/llava-1.5-7b-hf --task image-text-to-text

The model is built WITHOUT dispatch: it sits on the `meta` device, no weights are
downloaded beyond the config, so this is cheap for a 70B checkpoint. Everything
printed is decided by the config and the family module, which is exactly what
`support()` reads.

It prints, in order:

- the family module the checkpoint resolved to, and its `RENAME` aliases
- the standard name -> native path table (`model.layers[i].self_attn` is
  `model.transformer.h.0.attn` on GPT-2, `model.model.layers.0.self_attn` on
  Llama), which blocks have `self_attn` vs `linear_attn`, and whether `mlp` exists
- the sizes off the root (`num_layers`, `hidden_size`, `num_heads`, `num_kv_heads`,
  `head_dim`, `qk_head_dim`, `vocab_size`, `intermediate_size`)
- `model.support()` (or `support(layer=N)` with `--layer`): `None` means available,
  anything else is the reason a read would raise `nnterp.Unavailable`
- with `--task image-text-to-text` on a vision-language wrapper, the tower: its
  standard name -> native path table (`vision`, `vision.layers`,
  `vision.patch_embed`, `vision.norm`, `projector`), its sizes, and
  `model.vision.support()`
- one block's repr, which lists every standard value with its description

`attn_implementation` is transformers' default unless you pass it, and the
default is `sdpa` on every family that supports it, so a plain build reports the
whole attention interior (queries, keys, values, scores, probabilities, head
outputs) as unavailable with the reason `... this model runs 'sdpa'; load with
attn_implementation='eager'`. That is the answer for a plain load. `--eager`
builds with `attn_implementation="eager"` and shows what a load that reads the
interior will report.

`--task` defaults to `text-generation`, the text-only load: on a vision-language
wrapper it has no processor, so the tower serves nothing. `--task
image-text-to-text` builds the wrapper with its processor (the processor's files
come from the Hub, like the config) and shows what an image-carrying load reports.
"""

from __future__ import annotations

import argparse
import sys

ROOT_NAMES = ("embed_tokens", "layers", "norm", "lm_head")
BLOCK_NAMES = ("self_attn", "linear_attn", "mlp", "input_layernorm", "post_attention_layernorm")
VISION_NAMES = ("vision", "vision.layers", "vision.patch_embed", "vision.norm", "projector")
VISION_SIZES = ("num_layers", "hidden_size", "num_heads", "head_dim", "intermediate_size", "patch_size", "image_size")
DEFAULT_TASK = "text-generation"


def native(model, path: str) -> str | None:
    """The native dotted path behind a standard name, or None when it does not resolve."""
    try:
        return model.get(path).path
    except Exception:  # noqa: BLE001 — a missing alias is the answer, not a failure
        return None


def print_kv(rows: list[tuple[str, str]]) -> None:
    width = max((len(k) for k, _ in rows), default=0)
    for key, value in rows:
        print(f"  {key:<{width}}  {value}")


def print_sizes(owner, names) -> None:
    rows = []
    for name in names:
        try:
            rows.append((name, str(getattr(owner, name))))
        except Exception as exc:  # noqa: BLE001 — report, do not stop
            rows.append((name, f"({type(exc).__name__}: {exc})"))
    print_kv(rows)


def print_support(support: dict, n_layers: int) -> None:
    if not support:
        print("  (empty)")
        return
    width = max(len(k) for k in support)
    for key, value in support.items():
        if value is None:
            print(f"  {key:<{width}}  None")
        elif isinstance(value, dict):
            reasons = sorted(set(value.values()))
            blocks = sorted(value)
            where = "every block" if len(blocks) == n_layers else f"blocks {blocks}"
            if len(reasons) == 1:
                print(f"  {key:<{width}}  {where}: {reasons[0]}")
            else:
                print(f"  {key:<{width}}  {where}:")
                for block in blocks:
                    print(f"  {'':<{width}}    {block}: {value[block]}")
        else:
            print(f"  {key:<{width}}  {value}")


def summarize_vision(model) -> None:
    """The tower of a vision-language wrapper: names, sizes, support()."""
    vision = getattr(model, "vision", None)
    if vision is None:
        print("## vision tower: none (no model.vision: a text-only checkpoint, or a tower nnterp does not name)")
        print()
        return
    print(f"## vision tower ({type(vision).__name__}): standard name -> native path")
    rows = [(f"model.{name}", native(model, name) or "(none)") for name in VISION_NAMES]
    for name in BLOCK_NAMES:
        path = native(model, f"vision.layers.0.{name}")
        if path is not None:
            rows.append((f"model.vision.layers[0].{name}", path))
    print_kv(rows)
    print()
    print("## vision sizes (the tower's; model.num_layers / hidden_size are the text model's)")
    print_sizes(vision, VISION_SIZES)
    print()
    tower_layers = vision.num_layers if native(model, "vision.layers") is not None else 0
    print("## model.vision.support()   None = available on every tower block; else {block: reason}")
    print_support(vision.support(), tower_layers)
    if not vision.support():
        print("  no image reaches this load: load with task='image-text-to-text'")
    print()


def summarize(repo_id: str, layer: int | None, eager: bool, trust_remote_code: bool, dtype: str | None,
              task: str = DEFAULT_TASK) -> int:
    import nnterp  # noqa: F401 — import nnterp before any transformers.models module
    from nnterp import StandardizedTransformer

    kwargs = {}
    if eager:
        kwargs["attn_implementation"] = "eager"
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if dtype:
        import torch

        kwargs["dtype"] = getattr(torch, dtype)
    if task != DEFAULT_TASK:
        kwargs["task"] = task

    model = StandardizedTransformer(repo_id, **kwargs)

    print(f"# {repo_id}")
    print(f"model_type           {model.config.model_type}")
    print(f"family               {model.family.__name__}")
    print(f"architecture         {type(model._module).__name__}")
    impl = getattr(model.config, "_attn_implementation", None)
    note = "" if eager else "   (transformers' default; pass --eager to see an eager load)"
    print(f"attn_implementation  {impl}{note}")
    print(f"dispatched           {model.dispatched}   (meta device, no weights)")
    if task != DEFAULT_TASK:
        print(f"task                 {task}")
    print()

    print("## RENAME (native -> standard alias)")
    print_kv([(k, str(v)) for k, v in model.family.RENAME.items()])
    print()

    print("## standard name -> native path")
    rows: list[tuple[str, str]] = []
    for name in ROOT_NAMES:
        rows.append((f"model.{name}", native(model, name) or "(missing)"))
    n_layers = model.num_layers
    index = 0 if layer is None else layer
    if not 0 <= index < n_layers:
        print(f"--layer {layer} is out of range for {n_layers} blocks", file=sys.stderr)
        return 2
    rows.append((f"model.layers[{index}]", native(model, f"layers.{index}") or "(missing)"))
    for name in BLOCK_NAMES:
        path = native(model, f"layers.{index}.{name}")
        if path is not None:
            rows.append((f"model.layers[{index}].{name}", path))
    print_kv(rows)
    missing = [name for name in ("self_attn", "linear_attn", "mlp") if native(model, f"layers.{index}.{name}") is None]
    if missing:
        print(f"  block {index} has no {', '.join(missing)}")
    print()

    # A hybrid has self_attn on some blocks and linear_attn on others; say which.
    softmax = [i for i, block in enumerate(model.layers) if getattr(block, "self_attn", None) is not None]
    linear = [i for i, block in enumerate(model.layers) if getattr(block, "linear_attn", None) is not None]
    no_mlp = [i for i, block in enumerate(model.layers) if getattr(block, "mlp", None) is None]
    if linear:
        kind = type(model.layers[linear[0]].linear_attn).__mro__[1].__name__   # LinearAttention, SelectiveScan, StateSpace
        print(f"## recurrent mixer ({kind}): which blocks have which mixer")
        print(f"  self_attn   on {softmax}")
        print(f"  linear_attn on {linear}")
        print("  decide these lists OUTSIDE the trace, as above; getattr inside a trace can trip a served value")
        print("  nnterp.route_kernels(model.family, 'torch') before the first trace for the kernel values and the per-token state")
        print()
    moe = [i for i, block in enumerate(model.layers) if isinstance(getattr(block, "mlp", None), nnterp.Moe)]
    if moe:
        print(f"## mixture of experts on blocks: {moe if len(moe) < n_layers else 'every block'}")
        print()
    if no_mlp:
        print(f"## blocks with no mlp module: {no_mlp if len(no_mlp) < n_layers else 'every block'}")
        if not linear:
            print("  (OPT and XGLM keep fc1/fc2 on the block; layers[i].fc2.output is what the block adds)")
        print()

    print("## sizes")
    print_sizes(model, ("num_layers", "hidden_size", "num_heads", "num_kv_heads", "head_dim", "qk_head_dim",
                        "vocab_size", "intermediate_size"))
    print(f"  returns_tuple        {type(model.layers[index]).returns_tuple}   "
          "(True: the block's raw .output is a tuple; layer_output is the tensor either way)")
    print()

    if task != DEFAULT_TASK:
        summarize_vision(model)

    if layer is None:
        print("## model.support()   None = available on every block; else {block: reason}")
        support = model.support()
    else:
        print(f"## model.support(layer={layer})   None = available; else the reason")
        support = model.support(layer=layer)
    print_support(support, n_layers)
    print()

    print(f"## repr(model.layers[{index}])   every standard value, with its description")
    print(repr(model.layers[index]))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id")
    parser.add_argument("--layer", type=int, help="report support(layer=N) and this block's paths and repr")
    parser.add_argument("--eager", action="store_true",
                        help='build with attn_implementation="eager", as a load that reads the attention interior must')
    parser.add_argument("--dtype", help="torch dtype name to build with (float32, bfloat16); rarely needed on meta")
    parser.add_argument("--trust-remote-code", action="store_true", help="allow the repo's own modeling code")
    parser.add_argument("--task", default=DEFAULT_TASK,
                        help='the load\'s task (default text-generation); "image-text-to-text" builds a vision-language '
                             "wrapper with its processor and prints its tower")
    args = parser.parse_args(argv)
    return summarize(args.repo_id, args.layer, args.eager, args.trust_remote_code, args.dtype, args.task)


if __name__ == "__main__":
    sys.exit(main())
