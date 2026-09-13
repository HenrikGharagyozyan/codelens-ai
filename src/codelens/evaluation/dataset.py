"""Benchmark questions: loading, validation and the frozen dev/test split.

The split is not a field anyone chooses. It is derived from a hash of the
question id, so a question cannot quietly move from `test` to `dev` after it
turned out to be hard. Tuning (weights, depths, prompt wording) may look at
`dev` only; `test` is reported, never optimised against.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = ("lookup", "call_chain")
SPLITS = ("dev", "test")

# One question in TEST_SHARE is held out.
TEST_SHARE = 3


class DatasetError(ValueError):
    """A benchmark file that would silently produce wrong numbers."""


@dataclass(frozen=True)
class BenchmarkRepo:
    """A third-party repository pinned to one exact commit."""

    name: str
    url: str
    ref: str  # the tag that was cloned
    commit: str  # the commit that tag must resolve to


@dataclass(frozen=True)
class Question:
    id: str
    repo: str
    query: str
    category: str
    expected_files: tuple[str, ...]
    # What a correct answer says, in a sentence or two. The LLM judge grades
    # against it; it is never shown to the model being evaluated.
    reference: str
    split: str


def split_for(question_id: str) -> str:
    """The frozen split of a question, derived from its id alone."""
    digest = hashlib.sha256(question_id.encode("utf-8")).digest()
    return "test" if digest[0] % TEST_SHARE == 0 else "dev"


def load_repos(path: Path) -> dict[str, BenchmarkRepo]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: BenchmarkRepo(name=name, **spec) for name, spec in data["repos"].items()}


def load_questions(path: Path) -> list[Question]:
    """Loads one benchmark file, refusing anything that would skew the results."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    repo = data["repo"]

    questions: list[Question] = []
    seen: set[str] = set()
    for raw in data["questions"]:
        question = Question(
            id=raw["id"],
            repo=repo,
            query=raw["query"],
            category=raw["category"],
            expected_files=tuple(raw["expected_files"]),
            reference=raw["reference"],
            split=raw["split"],
        )
        _validate(question, seen)
        seen.add(question.id)
        questions.append(question)

    return questions


def _validate(question: Question, seen: set[str]) -> None:
    where = f"{question.repo}/{question.id}"
    if question.id in seen:
        raise DatasetError(f"{where}: duplicate question id")
    if question.category not in CATEGORIES:
        raise DatasetError(f"{where}: unknown category {question.category!r}")
    if question.split not in SPLITS:
        raise DatasetError(f"{where}: unknown split {question.split!r}")
    if question.split != split_for(question.id):
        raise DatasetError(f"{where}: split {question.split!r} differs from the frozen {split_for(question.id)!r}")
    if not question.expected_files:
        raise DatasetError(f"{where}: no expected files")
    if not question.reference.strip():
        raise DatasetError(f"{where}: empty reference answer")


def missing_files(questions: list[Question], repo_root: Path) -> list[tuple[str, str]]:
    """(question id, path) for every expected file that is not in the checkout."""
    return [(q.id, path) for q in questions for path in q.expected_files if not (repo_root / path).is_file()]
