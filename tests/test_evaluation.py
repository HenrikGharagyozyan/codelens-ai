"""Tests for the third-party benchmark tooling: datasets, metrics, citations, judge, report."""

import json
from pathlib import Path

import pytest

from codelens.context.citations import CitationVerifier
from codelens.evaluation import answers as answers_module
from codelens.evaluation.answers import (
    JsonCache,
    Judge,
    QuotaExhausted,
    RateLimiter,
    call_with_retries,
    generate_answer,
    judge_prompt,
    parse_verdict,
    retry_after,
    score_citations,
)
from codelens.evaluation.dataset import (
    CATEGORIES,
    DatasetError,
    Question,
    load_questions,
    load_repos,
    missing_files,
    split_for,
)
from codelens.evaluation.metrics import first_hit, ranked_files, score_retrieval
from codelens.evaluation.report import (
    README_END,
    README_START,
    answer_table,
    replace_between_markers,
    retrieval_table,
)

BENCHMARKS = Path(__file__).resolve().parent.parent / "scripts" / "benchmarks"
BENCH_CHECKOUTS = Path(__file__).resolve().parent.parent / ".bench"


def question(id="q-1", split=None, **overrides) -> dict:
    raw = {
        "id": id,
        "query": "Where is X?",
        "category": "lookup",
        "expected_files": ["pkg/x.py"],
        "reference": "In pkg/x.py.",
        "split": split or split_for(id),
    }
    raw.update(overrides)
    return raw


def write_benchmark(tmp_path, questions) -> Path:
    path = tmp_path / "bench.json"
    path.write_text(json.dumps({"repo": "demo", "questions": questions}), encoding="utf-8")
    return path


class TestSplit:
    def test_is_deterministic(self):
        assert [split_for(f"x-{i}") for i in range(50)] == [split_for(f"x-{i}") for i in range(50)]

    def test_holds_out_roughly_a_third(self):
        share = sum(split_for(f"x-{i}") == "test" for i in range(600)) / 600

        assert 0.28 < share < 0.39


class TestLoadQuestions:
    def test_loads_a_valid_file(self, tmp_path):
        loaded = load_questions(write_benchmark(tmp_path, [question()]))

        assert loaded == [
            Question("q-1", "demo", "Where is X?", "lookup", ("pkg/x.py",), "In pkg/x.py.", split_for("q-1"))
        ]

    @pytest.mark.parametrize(
        ("questions", "message"),
        [
            ([question(), question()], "duplicate"),
            ([question(category="trivia")], "unknown category"),
            ([question(expected_files=[])], "no expected files"),
            ([question(reference="  ")], "empty reference"),
        ],
    )
    def test_rejects_files_that_would_skew_results(self, tmp_path, questions, message):
        with pytest.raises(DatasetError, match=message):
            load_questions(write_benchmark(tmp_path, questions))

    def test_a_question_cannot_be_moved_to_another_split(self, tmp_path):
        moved = "dev" if split_for("q-1") == "test" else "test"

        with pytest.raises(DatasetError, match="frozen"):
            load_questions(write_benchmark(tmp_path, [question(split=moved)]))

    def test_missing_files_are_reported(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "x.py").write_text("")
        qs = load_questions(write_benchmark(tmp_path, [question(), question("q-2", expected_files=["pkg/gone.py"])]))

        assert missing_files(qs, tmp_path) == [("q-2", "pkg/gone.py")]


@pytest.fixture(scope="module")
def suites():
    repos = load_repos(BENCHMARKS / "repos.json")
    return repos, {name: load_questions(BENCHMARKS / f"{name}.json") for name in repos}


class TestShippedBenchmarks:
    """The benchmark files in the repository itself."""

    def test_every_repository_has_a_question_file(self, suites):
        repos, questions = suites

        assert set(questions) == set(repos) and len(repos) >= 3

    def test_there_are_enough_questions(self, suites):
        _, questions = suites

        assert 50 <= sum(len(qs) for qs in questions.values()) <= 100

    def test_both_splits_cover_both_categories(self, suites):
        _, questions = suites
        every = [q for qs in questions.values() for q in qs]

        for split in ("dev", "test"):
            assert {q.category for q in every if q.split == split} == set(CATEGORIES)

    def test_expected_files_exist_in_the_pinned_checkout(self, suites):
        repos, questions = suites
        checked = 0
        for name, repo in repos.items():
            root = BENCH_CHECKOUTS / name
            if not (root / ".git").exists():
                continue
            assert missing_files(questions[name], root) == []
            checked += 1
        if not checked:
            pytest.skip("no benchmark checkouts; run scripts/eval_benchmark.py once")


class TestRetrievalMetrics:
    def test_chunks_collapse_into_files(self):
        chunks = [{"metadata": {"file_path": p}} for p in ["a.py", "a.py", "b.py", None]]

        assert ranked_files(chunks) == ["a.py", "b.py"]

    def test_first_hit_takes_the_best_expected_file(self):
        assert first_hit(["a.py", "b.py", "c.py"], ("c.py", "b.py")) == 2
        assert first_hit(["a.py"], ("z.py",)) is None

    def test_scores(self):
        scores = score_retrieval([1, 3, None, 2])

        assert scores.recall == {1: 0.25, 3: 0.75, 5: 0.75}
        assert scores.mrr == pytest.approx((1 + 1 / 3 + 0 + 1 / 2) / 4)

    def test_no_ranks_score_zero(self):
        assert score_retrieval([]).mrr == 0.0


class TestCitationScore:
    def test_counts_valid_and_invalid_citations(self, populated_db):
        answer = "`connect` is defined at src/db.py:42; see also src/ghost.py:3."

        score = score_citations(CitationVerifier(populated_db), answer, ("src/db.py",))

        assert (score.total, score.valid, score.invalid, score.corrected) == (2, 1, 1, 0)
        assert score.precision == 0.5
        assert score.mentions_expected is True

    def test_a_wrong_line_for_a_known_symbol_is_counted_as_corrected(self, populated_db):
        score = score_citations(CitationVerifier(populated_db), "`connect` lives in src/db.py:137.", ("src/db.py",))

        assert (score.valid, score.corrected) == (0, 1)

    def test_an_answer_without_citations_has_no_precision(self, populated_db):
        score = score_citations(CitationVerifier(populated_db), "It is in the database module.", ("src/db.py",))

        assert score.precision is None
        assert score.mentions_expected is False


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system, prompt):
        self.prompts.append((system, prompt))
        return self.replies.pop(0)


class FakeAnswerer:
    def __init__(self):
        self.calls = 0

    def ask(self, context, question):
        self.calls += 1
        return f"answer to {question}"


def sample_question() -> Question:
    return Question("q-1", "demo", "Where is X?", "lookup", ("pkg/x.py",), "In pkg/x.py, function x.", "dev")


class TestJudge:
    @pytest.mark.parametrize(
        ("reply", "label"),
        [
            ('{"verdict": "correct", "reason": "same function"}', "correct"),
            ('```json\n{"verdict": "Partial", "reason": "right file"}\n```', "partial"),
            ('Here you go: {"verdict": "incorrect", "reason": "wrong file"}', "incorrect"),
        ],
    )
    def test_parses_verdicts(self, reply, label):
        assert parse_verdict(reply).label == label

    @pytest.mark.parametrize("reply", ["no json here", '{"verdict": "great"}'])
    def test_rejects_unusable_replies(self, reply):
        with pytest.raises(ValueError):
            parse_verdict(reply)

    def test_prompt_contains_reference_and_answer_but_not_the_config(self):
        prompt = judge_prompt(sample_question(), "It is in pkg/x.py.")

        assert "In pkg/x.py, function x." in prompt
        assert "It is in pkg/x.py." in prompt
        assert "graph" not in prompt.lower()

    def test_grades_and_scores(self, tmp_path):
        judge = Judge(FakeLLM(['{"verdict": "partial", "reason": "r"}']), "m", JsonCache(tmp_path / "c.json"))

        assert judge.grade(sample_question(), "answer").score == 0.5

    def test_a_malformed_reply_is_retried(self, monkeypatch):
        monkeypatch.setattr(answers_module.time, "sleep", lambda _: None)
        llm = FakeLLM(["oops", '{"verdict": "correct", "reason": "r"}'])

        assert Judge(llm, "m").grade(sample_question(), "answer").label == "correct"
        assert len(llm.prompts) == 2

    def test_cached_verdicts_are_not_requested_again(self, tmp_path):
        cache = JsonCache(tmp_path / "c.json")
        llm = FakeLLM(['{"verdict": "correct", "reason": "r"}'])
        Judge(llm, "m", cache).grade(sample_question(), "answer")

        again = Judge(FakeLLM([]), "m", JsonCache(tmp_path / "c.json")).grade(sample_question(), "answer")

        assert again.label == "correct"


class TestAnswersAndRetries:
    def test_answers_are_cached_per_context(self, tmp_path):
        cache = JsonCache(tmp_path / "c.json")
        answerer = FakeAnswerer()

        generate_answer(answerer, "m", sample_question(), "CTX", cache)
        generate_answer(answerer, "m", sample_question(), "CTX", cache)
        generate_answer(answerer, "m", sample_question(), "OTHER CTX", cache)

        assert answerer.calls == 2

    def test_transient_failures_are_retried(self, monkeypatch):
        monkeypatch.setattr(answers_module.time, "sleep", lambda _: None)
        attempts = iter([RuntimeError("503"), RuntimeError("503"), "ok"])

        def flaky():
            outcome = next(attempts)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        assert call_with_retries(flaky) == "ok"

    def test_persistent_failures_surface(self, monkeypatch):
        monkeypatch.setattr(answers_module.time, "sleep", lambda _: None)

        def broken():
            raise RuntimeError("down")

        with pytest.raises(RuntimeError):
            call_with_retries(broken, attempts=2)


class TestReport:
    RECORDS = [
        {
            "category": "lookup",
            "retrieval": {"hybrid": 1, "hybrid+graph": 1},
            "answers": {
                "hybrid": {
                    "score": 1.0,
                    "verdict": "correct",
                    "citations": {"total": 2, "valid": 2, "mentions_expected": True},
                }
            },
        },
        {
            "category": "call_chain",
            "retrieval": {"hybrid": None, "hybrid+graph": 2},
            "answers": {
                "hybrid": {
                    "score": 0.0,
                    "verdict": "incorrect",
                    "citations": {"total": 2, "valid": 1, "mentions_expected": False},
                }
            },
        },
    ]

    def test_retrieval_table_has_overall_and_per_category_mrr(self):
        table = retrieval_table(self.RECORDS, ["hybrid", "hybrid+graph"])

        assert "| hybrid | 0.500 | 50% | 50% | 50% | 0.000 | 1.000 |" in table
        assert "| hybrid+graph | 0.750 | 50% | 100% | 100% | 0.500 | 1.000 |" in table

    def test_answer_table(self):
        table = answer_table(self.RECORDS, ["hybrid", "hybrid+graph"])

        assert "| hybrid | 0.50 | 50% | 75% | 2.0 | 50% |" in table
        assert "hybrid+graph" not in table  # no graded answers for it

    def test_replaces_only_the_generated_block(self):
        text = f"intro\n{README_START}\nold\n{README_END}\noutro"

        assert replace_between_markers(text, "new") == f"intro\n{README_START}\nnew\n{README_END}\noutro"

    def test_missing_markers_are_an_error(self):
        with pytest.raises(ValueError):
            replace_between_markers("no markers", "new")


class TestRateLimits:
    RATE_LIMITED = (
        "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: generate_content_free_tier_requests. "
        "Please retry in 21.06933914s. 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', "
        "'retryDelay': '21s'"
    )

    @pytest.mark.parametrize(
        ("message", "delay"),
        [("Please retry in 21.069s.", 21.069), ("{'retryDelay': '7s'}", 7.0), ("503 Service Unavailable", None)],
    )
    def test_reads_the_delay_the_api_asked_for(self, message, delay):
        assert retry_after(RuntimeError(message)) == delay

    def test_a_rate_limit_waits_as_long_as_asked(self, monkeypatch):
        slept = []
        monkeypatch.setattr(answers_module.time, "sleep", slept.append)
        outcomes = iter([RuntimeError(self.RATE_LIMITED), "ok"])

        def call():
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        assert call_with_retries(call) == "ok"
        assert slept == [pytest.approx(22.06933914)]

    def test_a_daily_quota_stops_at_once(self, monkeypatch):
        monkeypatch.setattr(answers_module.time, "sleep", lambda _: pytest.fail("must not retry"))
        calls = []

        def call():
            calls.append(1)
            raise RuntimeError("429 quotaId: 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'")

        with pytest.raises(QuotaExhausted):
            call_with_retries(call)
        assert len(calls) == 1

    def test_the_limiter_spaces_calls_evenly(self):
        slept = []
        limiter = RateLimiter(rpm=5, clock=lambda: 100.0, sleep=slept.append)

        for _ in range(3):
            limiter.wait()

        assert slept == [12.0, 24.0]

    def test_no_limit_means_no_waiting(self):
        limiter = RateLimiter(rpm=None, sleep=lambda _: pytest.fail("must not sleep"))

        limiter.wait()
