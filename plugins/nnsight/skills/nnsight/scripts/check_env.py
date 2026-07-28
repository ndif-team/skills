#!/usr/bin/env python3
"""Report the environment an nnsight experiment will run in.

    python check_env.py            # local versions + devices
    python check_env.py --remote   # also query NDIF: key, deployed models, env diff

Run this first when something behaves unexpectedly — a version mismatch between
your machine and NDIF, a missing API key, or a model that is COLD rather than
RUNNING explains a large share of "my remote trace hangs / errors".
"""

from __future__ import annotations

import argparse
import os
import platform
import sys


def local() -> None:
    print("## local")
    print(f"python        {platform.python_version()}  ({sys.executable})")

    try:
        import nnsight

        print(f"nnsight       {nnsight.__version__}")
    except ImportError:
        print("nnsight       NOT INSTALLED")
        return

    import torch

    print(f"torch         {torch.__version__}")
    try:
        import transformers

        print(f"transformers  {transformers.__version__}")
    except ImportError:
        print("transformers  not installed")

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            total = properties.total_memory / 1024**3
            free = torch.cuda.mem_get_info(index)[0] / 1024**3
            print(f"cuda:{index}        {properties.name}  {free:.1f} GiB free of {total:.1f} GiB")
    else:
        print("cuda          unavailable — models will run on CPU")

    from nnsight import CONFIG

    key = CONFIG.API.APIKEY
    print(f"NDIF host     {CONFIG.API.HOST}")
    print(
        "NDIF key      "
        + ("set" if key else "not set — required by api.ndif.us, not by local deployments")
    )
    for variable in ("NDIF_API_KEY", "NDIF_HOST", "HF_TOKEN", "NNSIGHT_DEBUG"):
        if variable in os.environ:
            print(f"env           {variable} is set (overrides the config file)")
    print(f"debug mode    {CONFIG.APP.DEBUG}")
    print()


def remote() -> None:
    import nnsight

    print("## NDIF")
    try:
        status = nnsight.status()
    except Exception as exc:  # noqa: BLE001 — the reason matters to the user
        print(f"  could not reach {nnsight.CONFIG.API.HOST}: {type(exc).__name__}: {exc}")
        return

    print(status)
    print()
    print("## environment diff (local vs NDIF)")
    try:
        print(nnsight.compare())
    except Exception as exc:  # noqa: BLE001
        print(f"  unavailable: {type(exc).__name__}: {exc}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--remote", action="store_true", help="also query NDIF status and env diff")
    args = parser.parse_args(argv)

    local()
    if args.remote:
        remote()
    return 0


if __name__ == "__main__":
    sys.exit(main())
