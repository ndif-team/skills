"""Execute an agent's answer and decide whether it passed.

Code and debug tasks run in a **subprocess**, for three reasons:

1. nnsight compiles a trace block by reading its source off disk, so agent code
   has to be a real file — an `exec` of a string raises `WithBlockNotFoundError`.
2. A wrong intervention can segfault (torch C++ errors inside an interleaving
   greenlet) or hang. A long grid run cannot afford to lose the whole process.
3. Each task gets a clean CUDA context and a clean nnsight trace cache.

The subprocess writes a sentinel line with its verdict; anything else it prints
is captured as diagnostics.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .registry import Task, TaskKind

SENTINEL = "__EVAL_VERDICT__"

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class TaskOutcome:
    task_id: str
    passed: bool
    reason: str = ""
    stdout: str = ""
    stderr: str = ""
    error_type: str = ""
    duration_seconds: float = 0.0
    extracted_code: str = ""
    chosen_index: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "reason": self.reason,
            "error_type": self.error_type,
            "duration_seconds": round(self.duration_seconds, 3),
            "chosen_index": self.chosen_index,
            "stdout_tail": self.stdout[-2000:],
            "stderr_tail": self.stderr[-2000:],
            "extracted_code": self.extracted_code,
        }


def extract_code(response: str) -> str:
    """Pull the python out of an agent response.

    Agents mostly comply with "one fenced block", but not always: some prepend
    prose, some emit several blocks. Concatenating every block is wrong when one
    is an example of what *not* to do, so prefer the longest block, which in
    practice is the answer.
    """
    blocks = CODE_BLOCK_RE.findall(response)
    if blocks:
        return max(blocks, key=len).strip()
    return response.strip()


MCQ_PATTERNS = [
    re.compile(r"\banswer\s*(?:is)?\s*[:\-]?\s*\(?([A-J])\)?", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-J])\)?\s*[\.\):]?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\bchoose\s+\(?([A-J])\)?", re.IGNORECASE),
    re.compile(r"\boption\s+\(?([A-J])\)?", re.IGNORECASE),
    re.compile(r"\*\*\(?([A-J])\)?\*\*"),
]


def parse_mcq_answer(response: str, n_choices: int) -> Optional[int]:
    """Return a 0-based choice index, or None if nothing parses."""
    text = response.strip()
    if not text:
        return None

    for pattern in MCQ_PATTERNS:
        match = pattern.search(text)
        if match:
            index = ord(match.group(1).upper()) - ord("A")
            if 0 <= index < n_choices:
                return index

    # A bare 1-based number ("2") or "Answer: 2".
    number = re.search(r"\banswer\s*(?:is)?\s*[:\-]?\s*([1-9])\b", text, re.IGNORECASE)
    if number is None and re.fullmatch(r"[1-9]", text):
        number = re.match(r"([1-9])", text)
    if number:
        index = int(number.group(1)) - 1
        if 0 <= index < n_choices:
            return index

    # Last resort: a lone letter anywhere in a short response.
    if len(text) <= 40:
        letters = re.findall(r"\b([A-J])\b", text)
        if letters:
            index = ord(letters[-1].upper()) - ord("A")
            if 0 <= index < n_choices:
                return index
    return None


# Placeholders are substituted with str.replace, not str.format — the footer is
# full of braces and f-strings, and double-escaping them is a bug factory.
HARNESS_FOOTER = '''

# --- eval harness ------------------------------------------------------
def __eval_report():
    import json as __json
    import sys as __sys

    __sys.path.insert(0, "__EVALS_ROOT__")
    from evalkit.registry import get_task, load_all

    load_all()
    __task = get_task("__TASK_ID__")
    __namespace = dict(globals())
    try:
        __passed = bool(__task.verify(__namespace))
        __reason = "" if __passed else "verify() returned False"
    except Exception as __exc:
        __passed = False
        __reason = "verify() raised " + type(__exc).__name__ + ": " + str(__exc)
    print("__SENTINEL__ " + __json.dumps({"passed": __passed, "reason": __reason}))


__eval_report()
'''


def _build_script(task: Task, agent_code: str, evals_root: Path) -> str:
    footer = (
        HARNESS_FOOTER.replace("__EVALS_ROOT__", str(evals_root))
        .replace("__TASK_ID__", task.id)
        .replace("__SENTINEL__", SENTINEL)
    )
    return f"import torch\n{task.setup_code}\n\n{agent_code}\n{footer}"


def run_code_task(
    task: Task,
    response: str,
    *,
    evals_root: Optional[Path] = None,
    python: Optional[str] = None,
    workdir: Optional[Path] = None,
) -> TaskOutcome:
    evals_root = evals_root or Path(__file__).resolve().parent.parent
    code = extract_code(response)
    outcome = TaskOutcome(task_id=task.id, passed=False, extracted_code=code)

    if not code:
        outcome.reason = "no code in response"
        outcome.error_type = "EmptyResponse"
        return outcome

    directory = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="nnsight-eval-"))
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / f"{task.id}.py"
    script.write_text(_build_script(task, code, evals_root))

    environment = dict(os.environ)
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    started = time.time()
    try:
        completed = subprocess.run(
            [python or sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=task.timeout_seconds,
            cwd=str(directory),
            env=environment,
        )
        outcome.stdout = completed.stdout
        outcome.stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        outcome.duration_seconds = time.time() - started
        outcome.reason = f"timed out after {task.timeout_seconds}s"
        outcome.error_type = "Timeout"
        outcome.stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        outcome.stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return outcome
    outcome.duration_seconds = time.time() - started

    for line in reversed(outcome.stdout.splitlines()):
        if line.startswith(SENTINEL):
            verdict = json.loads(line[len(SENTINEL) :])
            outcome.passed = bool(verdict["passed"])
            outcome.reason = verdict.get("reason", "")
            if not outcome.passed and not outcome.error_type:
                outcome.error_type = "VerifyFailed"
            return outcome

    # No verdict: the script died before reaching the footer.
    outcome.reason = "execution failed before verification"
    outcome.error_type = _classify_error(outcome.stderr)
    return outcome


TRACEBACK_RE = re.compile(r"^(\w+(?:Error|Exception|Warning))\b", re.MULTILINE)


def _classify_error(stderr: str) -> str:
    """Best-effort exception class from a traceback, for failure taxonomy."""
    if not stderr.strip():
        return "NoOutput"
    names = TRACEBACK_RE.findall(stderr)
    if names:
        return names[-1]
    for marker in ("Segmentation fault", "core dumped", "CUDA out of memory"):
        if marker.lower() in stderr.lower():
            return marker.split()[0]
    return "Unknown"


def run_mcq_task(task: Task, response: str) -> TaskOutcome:
    index = parse_mcq_answer(response, len(task.choices))
    outcome = TaskOutcome(task_id=task.id, passed=False, chosen_index=index)
    if index is None:
        outcome.reason = "could not parse a choice from the response"
        outcome.error_type = "UnparseableAnswer"
        return outcome
    outcome.passed = index == task.correct_index
    if not outcome.passed:
        chosen = chr(ord("A") + index)
        correct = chr(ord("A") + task.correct_index)
        outcome.reason = f"chose {chosen}, correct is {correct}"
        outcome.error_type = "WrongChoice"
    return outcome


def run_task(task: Task, response: str, **kwargs) -> TaskOutcome:
    if task.kind is TaskKind.MCQ:
        return run_mcq_task(task, response)
    return run_code_task(task, response, **kwargs)
