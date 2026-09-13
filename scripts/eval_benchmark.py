"""Benchmark CodeLens on third-party repositories: retrieval metrics and graded answers.

    uv run python scripts/eval_benchmark.py                        # retrieval only
    uv run python scripts/eval_benchmark.py --answers              # + LLM answers, judge, citations
    uv run python scripts/eval_benchmark.py --answers --update-readme

The repositories and their pinned commits are in scripts/benchmarks/repos.json,
the questions in scripts/benchmarks/<repo>.json. Checkouts and indexes live in
.bench/ (gitignored); LLM answers and verdicts are cached in .bench/.cache/, so
a rerun only pays for what changed.

Every question has a frozen split derived from its id. Tune on `--split dev`;
`test` is the held-out set and is what the README reports.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from codelens.evaluation.answers import (  # noqa: E402
    GeminiLLM,
    JsonCache,
    Judge,
    QuotaExhausted,
    RateLimitedAnswerer,
    RateLimitedLLM,
    RateLimiter,
    generate_answer,
    score_citations,
)
from codelens.evaluation.dataset import load_questions, load_repos, missing_files  # noqa: E402
from codelens.evaluation.metrics import first_hit, ranked_files  # noqa: E402
from codelens.evaluation.report import (  # noqa: E402
    answer_summary,
    answer_table,
    replace_between_markers,
    retrieval_scores,
    retrieval_table,
)
from codelens.evaluation.workspace import Workspace  # noqa: E402

BENCH_DIR = ROOT / "scripts" / "benchmarks"
RESULTS_DIR = BENCH_DIR / "results"
WORKSPACE = ROOT / ".bench"
CACHE = WORKSPACE / ".cache" / "llm.json"
README = ROOT / "README.md"

RETRIEVAL_CONFIGS = ("vector", "keyword", "hybrid", "hybrid+graph")
ANSWER_CONFIGS = ("hybrid", "hybrid+graph")
DEPTH = 20  # chunks fetched per query before collapsing to files
CONTEXT_LIMIT = 5  # chunks passed to the model, as `codelens ask` does


def search(config, retriever, db, vector_store, query):
    if config == "vector":
        return vector_store.search(query, limit=DEPTH)
    if config == "keyword":
        return [{"metadata": {"file_path": row["file_path"]}} for row in db.search_chunks_keyword(query, limit=DEPTH)]
    if config == "hybrid":
        return retriever._hybrid_search(query, limit=DEPTH)
    return retriever._graph_search(query, limit=DEPTH)


def evaluate_repo(repo, questions, workspace, args, llm_parts) -> list[dict]:
    from codelens.context.retriever import ContextRetriever

    root = workspace.checkout(repo)
    missing = missing_files(questions, root)
    if missing:
        raise SystemExit(f"{repo.name}: expected files missing from the checkout: {missing}")

    print(f"[{repo.name}] indexing {root} (reused when unchanged)...", flush=True)
    db, vector_store = workspace.open_index(repo, rebuild=args.reindex)
    try:
        retriever = ContextRetriever(db, vector_store)
        records = []
        for q in questions:
            record = {
                "id": q.id,
                "repo": q.repo,
                "category": q.category,
                "split": q.split,
                "query": q.query,
                "retrieval": {
                    config: first_hit(
                        ranked_files(search(config, retriever, db, vector_store, q.query)), q.expected_files
                    )
                    for config in RETRIEVAL_CONFIGS
                },
            }
            records.append(record)

        if args.answers:
            # Answers cost API calls, retrieval does not: grade only the chosen split.
            chosen = [(q, r) for q, r in zip(questions, records) if args.answer_split in ("all", q.split)]
            if chosen:
                grade_answers([q for q, _ in chosen], [r for _, r in chosen], db, vector_store, llm_parts, args.workers)
        return records
    finally:
        db.close()


def grade_answers(questions, records, db, vector_store, llm_parts, workers) -> None:
    """Answer every question the way `codelens ask` would, then score the answers."""
    from codelens.context.citations import CitationVerifier
    from codelens.context.retriever import ContextRetriever

    answerer, judge, cache = llm_parts
    retrievers = {
        config: ContextRetriever(db, vector_store, graph_expansion=config == "hybrid+graph")
        for config in ANSWER_CONFIGS
    }

    # SQLite stays on this thread; only the API calls run in parallel.
    jobs = []
    for q, record in zip(questions, records):
        record["answers"] = {}
        for config, retriever in retrievers.items():
            jobs.append((record, q, config, retriever.build_context(q.query, limit=CONTEXT_LIMIT) or ""))

    def answer(job):
        _, q, _, context = job
        # With no context `codelens ask` does not call the model at all.
        return generate_answer(answerer, answerer.model_name, q, context, cache) if context else ""

    with ThreadPoolExecutor(workers) as pool:
        answers = list(pool.map(answer, jobs))

    verifier = CitationVerifier(db)
    for (record, q, config, _), text in zip(jobs, answers):
        record["answers"][config] = {
            "answer": text,
            "citations": asdict(score_citations(verifier, text, q.expected_files)),
        }

    with ThreadPoolExecutor(workers) as pool:
        verdicts = list(pool.map(lambda pair: judge.grade(pair[0][1], pair[1]), zip(jobs, answers)))

    for (record, _, config, _), verdict in zip(jobs, verdicts):
        record["answers"][config].update(verdict=verdict.label, score=verdict.score, reason=verdict.reason)


def render_report(records: list[dict], meta: dict) -> str:
    """The Markdown block that goes into the README and results/latest.md."""
    repos = ", ".join(f"{name} {ref}" for name, ref in meta["repos"].items())
    provenance = f"_Generated by `scripts/eval_benchmark.py` on {meta['date']}. Repositories: {repos}."
    if meta.get("answer_model"):
        provenance += f" Answer model `{meta['answer_model']}`, judge `{meta['judge_model']}`."
    lines = [provenance + "_", ""]

    for split, title in (("test", "Held-out test split"), ("dev", "Dev split")):
        subset = [r for r in records if r["split"] == split]
        if not subset:
            continue
        chains = sum(1 for r in subset if r["category"] == "call_chain")
        lines += [
            f"**{title}**: {len(subset)} questions ({len(subset) - chains} lookup, {chains} call-chain).",
            "",
            "Retrieval (does the right file reach the top of the ranking):",
            "",
            retrieval_table(subset, RETRIEVAL_CONFIGS),
            "",
        ]
        if any("answers" in r for r in subset):
            lines += ["Answers (LLM-judged correctness, citations checked against the index):", ""]
            lines += [answer_table(subset, ANSWER_CONFIGS), ""]

    lines += ["Per repository, all splits:", "", per_repo_table(records), ""]
    return "\n".join(lines).rstrip() + "\n"


def per_repo_table(records: list[dict]) -> str:
    from codelens.evaluation.report import markdown_table

    header = ["Repository", "Questions", "hybrid MRR", "hybrid+graph MRR"]
    has_answers = any("answers" in r for r in records)
    if has_answers:
        header += ["hybrid correctness", "hybrid+graph correctness"]

    rows = []
    for repo in sorted({r["repo"] for r in records}):
        subset = [r for r in records if r["repo"] == repo]
        row = [repo, str(len(subset))]
        row += [f"{retrieval_scores(subset, config).mrr:.3f}" for config in ANSWER_CONFIGS]
        if has_answers:
            row += [f"{answer_summary(subset, config).get('correctness', 0):.2f}" for config in ANSWER_CONFIGS]
        rows.append(row)
    return markdown_table(header, rows)


def print_misses(records: list[dict]) -> None:
    for record in records:
        rank = record["retrieval"]["hybrid+graph"]
        verdicts = {c: a.get("verdict") for c, a in record.get("answers", {}).items()}
        if rank is None or rank > 5 or "incorrect" in verdicts.values():
            print(f"  {record['id']:<9} {record['split']:<4} rank={rank} {verdicts} {record['query'][:70]}")


def main() -> int:
    repos = load_repos(BENCH_DIR / "repos.json")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repos", nargs="+", choices=sorted(repos), default=sorted(repos))
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    parser.add_argument("--answers", action="store_true", help="generate and grade answers (calls the LLM API)")
    parser.add_argument("--judge-model", help="model for the judge (default: the answer model)")
    parser.add_argument(
        "--answer-split",
        choices=("dev", "test", "all"),
        default="test",
        help="which questions get answers graded (default: the held-out test split)",
    )
    parser.add_argument("--limit", type=int, help="first N questions per repository, for a quick run")
    parser.add_argument("--workers", type=int, default=4, help="parallel API calls")
    parser.add_argument("--rpm", type=float, help="requests per minute for the answer model (free tier: 5)")
    parser.add_argument("--judge-rpm", type=float, help="requests per minute for the judge model")
    parser.add_argument("--reindex", action="store_true", help="rebuild the indexes (after changing the indexer)")
    parser.add_argument("--update-readme", action="store_true", help="write the tables into README.md")
    parser.add_argument("--verbose", action="store_true", help="list the questions that were missed")
    args = parser.parse_args()

    llm_parts = None
    meta = {"date": date.today().isoformat(), "repos": {name: repos[name].ref for name in args.repos}}
    if args.answers:
        from codelens.llm.gemini import GeminiClient

        # Each model has its own rate limit, so each gets its own limiter.
        answerer = RateLimitedAnswerer(GeminiClient(), RateLimiter(args.rpm))
        judge_llm = RateLimitedLLM(GeminiLLM(model=args.judge_model), RateLimiter(args.judge_rpm))
        cache = JsonCache(CACHE)
        llm_parts = (answerer, Judge(judge_llm, judge_llm.model, cache), cache)
        meta |= {"answer_model": answerer.model_name, "judge_model": judge_llm.model}

    workspace = Workspace(WORKSPACE)
    records: list[dict] = []
    for name in args.repos:
        questions = load_questions(BENCH_DIR / f"{name}.json")
        if args.split != "all":
            questions = [q for q in questions if q.split == args.split]
        if args.limit:
            questions = questions[: args.limit]
        try:
            records += evaluate_repo(repos[name], questions, workspace, args, llm_parts)
        except QuotaExhausted as exc:
            print(f"\nStopped: {exc}. Everything answered so far is cached in {CACHE}.")
            return 2

    report = render_report(records, meta)
    print()
    print(report)
    if args.verbose:
        print("Missed (hybrid+graph rank > 5, or an answer judged incorrect):")
        print_misses(records)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "latest.json").write_text(
        json.dumps({"meta": meta, "records": records}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (RESULTS_DIR / "latest.md").write_text(report, encoding="utf-8")

    if args.update_readme:
        README.write_text(replace_between_markers(README.read_text(encoding="utf-8"), report), encoding="utf-8")
        print("README.md updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
