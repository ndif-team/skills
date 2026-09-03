"""Execute every runnable code block in every skill.

A markdown file is one test: its python blocks run in document order in a shared
namespace, so later blocks can build on earlier ones. See docblocks.py for the
directive syntax that controls skipping.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

import conftest
from docblocks import REPO_ROOT, Block, extract_blocks, markdown_files


def _files_with_code() -> list[Path]:
    return [path for path in markdown_files() if any(b.lang == "python" for b in extract_blocks(path))]


def _skip_reason(block: Block, run_slow: bool) -> str | None:
    if "remote" in block.flags and not conftest.ndif_host():
        return "needs a reachable NDIF_HOST (e.g. NDIF_HOST=http://localhost:8001)"
    if "gpu" in block.flags and not conftest.cuda_available():
        return "needs CUDA"
    if "slow" in block.flags and not run_slow:
        return "needs --run-slow"
    return None


def _record(block: Block, status: str, detail: str = "") -> None:
    conftest.RESULTS.append(
        {
            "file": str(block.path.relative_to(REPO_ROOT)),
            "line": block.line,
            "status": status,
            "flags": sorted(block.flags),
            "detail": detail,
        }
    )


def _check_expected_error(block: Block, error: BaseException | None) -> None:
    expected = block.expect_error
    if error is None:
        pytest.fail(f"{block.location}: expected {expected} but the block ran cleanly")
    names = {cls.__name__ for cls in type(error).__mro__}
    if expected not in names:
        raise AssertionError(
            f"{block.location}: expected {expected}, got {type(error).__name__}: {error}"
        ) from error


def _write_block(block: Block, tmp_dir: Path) -> Path:
    """nnsight reads the with-block's source off disk, so blocks must be real files."""
    script = tmp_dir / f"{block.path.stem}_L{block.line}.py"
    script.write_text(block.code)
    return script


@pytest.mark.docs
@pytest.mark.parametrize("path", _files_with_code(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_markdown_code_blocks(path: Path, request, tmp_path: Path):
    run_slow = request.config.getoption("--run-slow")
    namespace: dict = {}
    ran_any = False

    for block in extract_blocks(path):
        if block.lang != "python":
            continue

        if "skip" in block.flags:
            if "nocompile" not in block.flags:
                try:
                    compile(block.code, block.location, "exec")
                except SyntaxError as exc:
                    raise AssertionError(f"{block.location}: syntax error in skipped block: {exc}") from exc
                _record(block, "compiled")
            else:
                _record(block, "skipped", "nocompile")
            continue

        reason = _skip_reason(block, run_slow)
        if reason is not None:
            compile(block.code, block.location, "exec")
            _record(block, "skipped", reason)
            continue

        compile(block.code, block.location, "exec")  # syntax first, for a clean message
        script = _write_block(block, tmp_path)
        error: BaseException | None = None
        try:
            # run_path executes the file (so inspect.getsource works inside nnsight)
            # and returns its globals, which carry into the next block.
            namespace = {
                key: value
                for key, value in runpy.run_path(str(script), init_globals=namespace).items()
                if not (key.startswith("__") and key.endswith("__"))
            }
        except BaseException as exc:  # noqa: BLE001 - re-raised below unless expected
            error = exc

        if block.expect_error is not None:
            _check_expected_error(block, error)
            _record(block, "ran", f"raised {block.expect_error}")
        elif error is not None:
            raise AssertionError(f"{block.location} failed: {type(error).__name__}: {error}") from error
        else:
            _record(block, "ran")
        ran_any = True

    if not ran_any:
        pytest.skip("no executable blocks (all skipped or gated)")
