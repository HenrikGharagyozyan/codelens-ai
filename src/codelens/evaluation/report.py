"""Aggregating per-question results into the tables that go into the README."""

from collections.abc import Iterable

from codelens.evaluation.metrics import K_VALUES, RetrievalScores, score_retrieval

README_START = "<!-- benchmark:start -->"
README_END = "<!-- benchmark:end -->"


def retrieval_scores(records: list[dict], config: str) -> RetrievalScores:
    return score_retrieval([r["retrieval"][config] for r in records])


def retrieval_table(records: list[dict], configs: Iterable[str]) -> str:
    """Rows: configs. Columns: MRR and Recall@K overall, then MRR per category."""
    categories = sorted({r["category"] for r in records})
    header = ["Config", "MRR"] + [f"R@{k}" for k in K_VALUES] + [f"{c} MRR" for c in categories]
    rows = []
    for config in configs:
        overall = retrieval_scores(records, config)
        row = [config, f"{overall.mrr:.3f}"] + [f"{overall.recall[k]:.0%}" for k in K_VALUES]
        for category in categories:
            subset = [r for r in records if r["category"] == category]
            row.append(f"{retrieval_scores(subset, config).mrr:.3f}")
        rows.append(row)
    return markdown_table(header, rows)


def answer_summary(records: list[dict], config: str) -> dict:
    graded = [r["answers"][config] for r in records if config in r.get("answers", {})]
    if not graded:
        return {}
    total = sum(a["citations"]["total"] for a in graded)
    valid = sum(a["citations"]["valid"] for a in graded)
    return {
        "count": len(graded),
        "correctness": sum(a["score"] for a in graded) / len(graded),
        "correct_share": sum(1 for a in graded if a["verdict"] == "correct") / len(graded),
        "citation_precision": valid / total if total else None,
        "citations_per_answer": total / len(graded),
        "mentions_expected": sum(1 for a in graded if a["citations"]["mentions_expected"]) / len(graded),
    }


def answer_table(records: list[dict], configs: Iterable[str]) -> str:
    header = ["Config", "Correctness", "Fully correct", "Citation precision", "Cites per answer", "Names right file"]
    rows = []
    for config in configs:
        s = answer_summary(records, config)
        if not s:
            continue
        precision = "n/a" if s["citation_precision"] is None else f"{s['citation_precision']:.0%}"
        rows.append(
            [
                config,
                f"{s['correctness']:.2f}",
                f"{s['correct_share']:.0%}",
                precision,
                f"{s['citations_per_answer']:.1f}",
                f"{s['mentions_expected']:.0%}",
            ]
        )
    return markdown_table(header, rows)


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def replace_between_markers(text: str, body: str, start: str = README_START, end: str = README_END) -> str:
    """Swaps the generated block between the markers, leaving the rest untouched."""
    if start not in text or end not in text:
        raise ValueError(f"markers {start!r} and {end!r} not found")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return f"{before}{start}\n{body.strip()}\n{end}{after}"
