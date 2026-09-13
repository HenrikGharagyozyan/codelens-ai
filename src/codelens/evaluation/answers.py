"""Grading answers: citation correctness from the index, correctness from an LLM judge.

Two independent signals, on purpose:

- Citation correctness needs no model at all. Every `path:line` in a raw
  answer is checked against the index with the same `CitationVerifier` the CLI
  uses, *before* repair, so the number measures what the model got right on
  its own.
- Correctness is judged by an LLM against a short reference answer written with
  the question. The judge never sees which retrieval configuration produced
  the answer, so it cannot favour one.
"""

import hashlib
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from codelens.context.citations import CitationVerifier
from codelens.evaluation.dataset import Question

T = TypeVar("T")

VERDICT_SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

JUDGE_SYSTEM = """You grade answers to questions about a source code repository.

You are given the question, a short reference answer written by someone who
read the code, and a candidate answer. Grade the candidate against the
reference:

- "correct": it identifies the same place in the code (file, class or function)
  and describes the mechanism consistently with the reference. Extra correct
  detail is fine. Missing minor detail is fine.
- "partial": it points to the right file or area but names the wrong function,
  misses the main point, or mixes a correct statement with a wrong one.
- "incorrect": it points to the wrong place, contradicts the reference, or does
  not answer (for example, says the context is insufficient).

Judge substance, not style or length. Do not reward confident wording.
Reply with JSON only: {"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}"""


class LLM(Protocol):
    def complete(self, system: str, prompt: str) -> str: ...


class Answerer(Protocol):
    """What `codelens ask` uses: `GeminiClient.ask(context, question)`."""

    def ask(self, context_chunks: str, question: str) -> str: ...


class GeminiLLM:
    """A plain completion call on the Gemini client the CLI already configures."""

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        from codelens.llm.gemini import GeminiClient

        base = GeminiClient()
        self.client = base.client
        self.model = model or base.model_name
        self.temperature = temperature

    def complete(self, system: str, prompt: str) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(system_instruction=system, temperature=self.temperature)
        response = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        return response.text or ""


# "Please retry in 21.07s." in the message, or "'retryDelay': '21s'" in the details.
RETRY_AFTER_RE = re.compile(r"retry(?:Delay)?\W{0,4}(?:in\s+)?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


class QuotaExhausted(RuntimeError):
    """A daily quota ran out: retrying today is pointless, and the cache keeps what was done."""


def retry_after(exc: Exception) -> float | None:
    """The delay the API asked for in a rate-limit error, if it named one."""
    match = RETRY_AFTER_RE.search(str(exc))
    return float(match.group(1)) if match else None


def call_with_retries(fn: Callable[[], T], attempts: int = 6, base_delay: float = 2.0) -> T:
    """Retries transient API failures, waiting as long as a rate-limit error asks."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if "PerDay" in str(exc):
                raise QuotaExhausted("daily API quota exhausted; cached results are kept, rerun later") from exc
            if attempt == attempts - 1:
                raise
            requested = retry_after(exc)
            time.sleep(requested + 1 if requested is not None else base_delay * 2**attempt)
    raise AssertionError("unreachable")


class RateLimiter:
    """Spaces calls at least 60/rpm seconds apart, across all worker threads.

    Free API tiers count requests per minute per model; staying under the
    limit is cheaper than bouncing off it and backing off.
    """

    def __init__(self, rpm: float | None, clock: Callable[[], float] = time.monotonic, sleep=time.sleep):
        self.interval = 60.0 / rpm if rpm else 0.0
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            self._sleep(start - now)


class RateLimitedAnswerer:
    def __init__(self, answerer: "Answerer", limiter: RateLimiter):
        self._answerer = answerer
        self._limiter = limiter
        self.model_name = answerer.model_name  # type: ignore[attr-defined]

    def ask(self, context_chunks: str, question: str) -> str:
        self._limiter.wait()
        return self._answerer.ask(context_chunks, question)


class RateLimitedLLM:
    def __init__(self, llm: "LLM", limiter: RateLimiter):
        self._llm = llm
        self._limiter = limiter
        self.model = getattr(llm, "model", "unknown")

    def complete(self, system: str, prompt: str) -> str:
        self._limiter.wait()
        return self._llm.complete(system, prompt)


def cache_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


class JsonCache:
    """A small persistent cache, so re-running the benchmark does not re-bill the API."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, key: str) -> dict | None:
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=1, sort_keys=True), encoding="utf-8")
            tmp.replace(self.path)


# ------------------------------------------------------------------ citations


@dataclass(frozen=True)
class CitationScore:
    total: int  # `path:line` citations found in the answer
    valid: int  # pointing at a real symbol or inside a real chunk
    corrected: int  # the right file and symbol, but the wrong line
    invalid: int  # an unknown file, or a line with nothing there
    mentions_expected: bool  # the answer names at least one expected file

    @property
    def precision(self) -> float | None:
        return self.valid / self.total if self.total else None


def score_citations(verifier: CitationVerifier, answer: str, expected_files: tuple[str, ...]) -> CitationScore:
    checks = verifier.verify(answer)
    return CitationScore(
        total=len(checks),
        valid=sum(1 for c in checks if c.status == "ok"),
        corrected=sum(1 for c in checks if c.status == "corrected"),
        invalid=sum(1 for c in checks if c.status in ("unknown_file", "no_symbol")),
        # Plain mentions count too: the prompt allows citing a file without a
        # line when the line is not in the context.
        mentions_expected=any(path in answer for path in expected_files),
    )


# ---------------------------------------------------------------------- judge


@dataclass(frozen=True)
class Verdict:
    label: str
    reason: str

    @property
    def score(self) -> float:
        return VERDICT_SCORES[self.label]


def judge_prompt(question: Question, answer: str) -> str:
    return (
        f"QUESTION:\n{question.query}\n\n"
        f"REFERENCE ANSWER:\n{question.reference}\n\n"
        f"CANDIDATE ANSWER:\n{answer.strip() or '(empty)'}\n"
    )


def parse_verdict(text: str) -> Verdict:
    """Pulls the JSON verdict out of a reply, tolerating code fences and prose."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in judge reply: {text[:200]!r}")
    data = json.loads(match.group(0))
    label = str(data.get("verdict", "")).strip().lower()
    if label not in VERDICT_SCORES:
        raise ValueError(f"unknown verdict {label!r}")
    return Verdict(label=label, reason=str(data.get("reason", "")).strip())


class Judge:
    def __init__(self, llm: LLM, model_name: str, cache: JsonCache | None = None, attempts: int = 6):
        self.llm = llm
        self.model_name = model_name
        self.cache = cache
        self.attempts = attempts

    def grade(self, question: Question, answer: str) -> Verdict:
        prompt = judge_prompt(question, answer)
        key = cache_key("judge", self.model_name, JUDGE_SYSTEM, prompt)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return Verdict(hit["label"], hit["reason"])

        # A malformed reply is retried like a network error.
        verdict = call_with_retries(lambda: parse_verdict(self.llm.complete(JUDGE_SYSTEM, prompt)), self.attempts)
        if self.cache is not None:
            self.cache.put(key, {"label": verdict.label, "reason": verdict.reason})
        return verdict


def generate_answer(
    answerer: Answerer, model_name: str, question: Question, context: str, cache: JsonCache | None = None
) -> str:
    """The answer `codelens ask` would give for this context, cached by context."""
    key = cache_key("answer", model_name, question.query, context)
    if cache is not None and (hit := cache.get(key)) is not None:
        return hit["answer"]

    answer = call_with_retries(lambda: answerer.ask(context, question.query) or "")
    if cache is not None:
        cache.put(key, {"answer": answer})
    return answer
