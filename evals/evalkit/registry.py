"""Task registry.

Three task kinds, all scored the same way at the top level (pass / fail):

- ``CODE``  — the agent writes nnsight code; we execute it and call ``verify``.
- ``DEBUG`` — the agent is given broken code plus the symptom, returns a fixed
  version; execution and verification are identical to ``CODE``.
- ``MCQ``   — the agent picks one of N choices; we compare to ``correct_index``.

Ported from nnsight's ``tests/agent-evals`` (dev branch) and extended with the
DEBUG kind and per-task topic tags that map onto skills in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class Difficulty(Enum):
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class TaskKind(Enum):
    CODE = "code"
    DEBUG = "debug"
    MCQ = "mcq"


@dataclass
class Task:
    id: str
    name: str
    difficulty: Difficulty
    kind: TaskKind = TaskKind.CODE

    # CODE / DEBUG
    prompt: str = ""
    setup_code: str = ""
    buggy_code: str = ""          # DEBUG only: what the agent is asked to fix
    symptom: str = ""             # DEBUG only: what the user reports seeing
    verify: Optional[Callable[[dict], bool]] = None
    expected_output_description: str = ""
    timeout_seconds: int = 180
    #: A hand-written correct answer. `audit.py` runs it to prove the task is
    #: solvable and the verifier accepts a good solution — no LLM calls needed.
    reference_solution: str = ""

    # MCQ
    question: str = ""
    choices: list[str] = field(default_factory=list)
    correct_index: int = -1
    explanation: str = ""

    # Which skill / doc area this exercises. Used for per-topic breakdowns.
    tags: list[str] = field(default_factory=list)

    def user_prompt(self) -> str:
        """The exact text handed to the agent."""
        if self.kind is TaskKind.MCQ:
            lines = [self.question, ""]
            for index, choice in enumerate(self.choices):
                lines.append(f"{chr(ord('A') + index)}. {choice}")
            lines.append("")
            lines.append("Answer with a single letter.")
            return "\n".join(lines)

        parts = [self.prompt.strip()]
        if self.kind is TaskKind.DEBUG:
            parts.append(
                "\nThe code below is broken. "
                f"Symptom:\n{self.symptom.strip()}\n\n"
                f"```python\n{self.buggy_code.strip()}\n```"
            )
        if self.setup_code.strip():
            parts.append(
                "\nThis setup has already run; do not repeat it:\n"
                f"```python\n{self.setup_code.strip()}\n```"
            )
        parts.append(
            "\nReply with a single ```python code block and nothing else. "
            "Do not reload the model."
        )
        return "\n".join(parts)

    def to_dict(self) -> dict:
        base = {
            "id": self.id,
            "name": self.name,
            "difficulty": self.difficulty.value,
            "kind": self.kind.value,
            "tags": self.tags,
        }
        if self.kind is TaskKind.MCQ:
            base.update(
                {
                    "question": self.question,
                    "choices": self.choices,
                    "correct_index": self.correct_index,
                    "explanation": self.explanation,
                }
            )
        else:
            base.update(
                {
                    "prompt": self.prompt,
                    "setup_code": self.setup_code,
                    "buggy_code": self.buggy_code,
                    "symptom": self.symptom,
                    "expected_output_description": self.expected_output_description,
                    "timeout_seconds": self.timeout_seconds,
                }
            )
        return base


TASKS: dict[str, Task] = {}


def register_task(task: Task) -> Task:
    if task.id in TASKS:
        raise ValueError(f"duplicate task id: {task.id}")
    TASKS[task.id] = task
    return task


def register_mcq(
    *,
    id: str,
    name: str,
    difficulty: Difficulty,
    question: str,
    choices: list[str],
    correct_index: int,
    explanation: str = "",
    tags: Optional[list[str]] = None,
) -> Task:
    return register_task(
        Task(
            id=id,
            name=name,
            difficulty=difficulty,
            kind=TaskKind.MCQ,
            question=question,
            choices=list(choices),
            correct_index=correct_index,
            explanation=explanation,
            tags=tags or [],
        )
    )


def register_debug(
    *,
    id: str,
    name: str,
    difficulty: Difficulty,
    symptom: str,
    buggy_code: str,
    setup_code: str,
    verify: Callable[[dict], bool],
    prompt: str = "Fix the bug in this nnsight code.",
    expected_output_description: str = "",
    tags: Optional[list[str]] = None,
    timeout_seconds: int = 180,
) -> Task:
    return register_task(
        Task(
            id=id,
            name=name,
            difficulty=difficulty,
            kind=TaskKind.DEBUG,
            prompt=prompt,
            symptom=symptom,
            buggy_code=buggy_code,
            setup_code=setup_code,
            verify=verify,
            expected_output_description=expected_output_description,
            tags=tags or [],
            timeout_seconds=timeout_seconds,
        )
    )


def get_task(task_id: str) -> Optional[Task]:
    return TASKS.get(task_id)


def all_tasks() -> list[Task]:
    return list(TASKS.values())


def select(
    *,
    ids: Optional[list[str]] = None,
    kinds: Optional[list[TaskKind]] = None,
    difficulties: Optional[list[Difficulty]] = None,
    tags: Optional[list[str]] = None,
) -> list[Task]:
    tasks = all_tasks()
    if ids:
        wanted = set(ids)
        tasks = [t for t in tasks if t.id in wanted]
    if kinds:
        tasks = [t for t in tasks if t.kind in kinds]
    if difficulties:
        tasks = [t for t in tasks if t.difficulty in difficulties]
    if tags:
        wanted = set(tags)
        tasks = [t for t in tasks if wanted & set(t.tags)]
    return sorted(tasks, key=lambda t: t.id)


def load_all() -> None:
    """Import every task module so registration happens."""
    from . import tasks  # noqa: F401


__all__ = [
    "Difficulty",
    "TaskKind",
    "Task",
    "TASKS",
    "register_task",
    "register_mcq",
    "register_debug",
    "get_task",
    "all_tasks",
    "select",
    "load_all",
]
