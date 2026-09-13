"""Retrieval metrics: where the first expected file lands in the ranking."""

from dataclasses import dataclass

K_VALUES = (1, 3, 5)


def ranked_files(chunks: list[dict]) -> list[str]:
    """Collapses a chunk ranking into a file ranking.

    Retrieval is chunk-level but the ground truth is file-level, so three chunks
    from the same file count as one result, not three.
    """
    files: list[str] = []
    for chunk in chunks:
        path = chunk["metadata"].get("file_path")
        if path and path not in files:
            files.append(path)
    return files


def first_hit(files: list[str], expected: tuple[str, ...] | list[str]) -> int | None:
    """1-based rank of the first expected file, or None if none was retrieved."""
    ranks = [files.index(path) + 1 for path in expected if path in files]
    return min(ranks) if ranks else None


@dataclass(frozen=True)
class RetrievalScores:
    count: int
    recall: dict[int, float]  # Recall@K for each K in K_VALUES
    mrr: float


def score_retrieval(ranks: list[int | None]) -> RetrievalScores:
    count = len(ranks)
    if count == 0:
        return RetrievalScores(0, {k: 0.0 for k in K_VALUES}, 0.0)
    recall = {k: sum(1 for r in ranks if r is not None and r <= k) / count for k in K_VALUES}
    mrr = sum(1.0 / r for r in ranks if r) / count
    return RetrievalScores(count, recall, mrr)
