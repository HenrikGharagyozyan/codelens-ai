"""Retrieval evaluation: how often does the right file reach the top K?

Run against an up-to-date index:

    uv run codelens index .
    uv run python scripts/eval_recall.py

The point is not the absolute number but the comparison: vector-only against
keyword-only against hybrid. If hybrid does not beat both, the fusion is not
earning its complexity.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from codelens.context.retriever import ContextRetriever  # noqa: E402
from codelens.indexer.vector_store import VectorStore  # noqa: E402
from codelens.repository.db import DatabaseManager  # noqa: E402

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
K_VALUES = (1, 3, 5)


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["queries"]


def ranked_files(chunks: list[dict]) -> list[str]:
    """Collapses a chunk ranking into a file ranking.

    Retrieval is chunk-level but the ground truth is file-level, so three chunks
    from the same file must count as one result, not three. Without this a file
    that dominates the top of the list makes Recall@3 look identical to
    Recall@1 for reasons that have nothing to do with relevance.
    """
    files: list[str] = []
    for chunk in chunks:
        path = chunk["metadata"].get("file_path")
        if path and path not in files:
            files.append(path)
    return files


class Retrievers:
    """The three configurations under comparison."""

    def __init__(self, db: DatabaseManager, vector_store: VectorStore):
        self.db = db
        self.vector_store = vector_store
        self.retriever = ContextRetriever(db, vector_store)

    def vector(self, query: str, limit: int) -> list[dict]:
        return self.vector_store.search(query, limit=limit)

    def keyword(self, query: str, limit: int) -> list[dict]:
        rows = self.db.search_chunks_keyword(query, limit=limit)
        return [
            {"metadata": {"file_path": row["file_path"], "symbol_name": row["symbol_name"]}}
            for row in rows
        ]

    def hybrid(self, query: str, limit: int) -> list[dict]:
        return self.retriever._hybrid_search(query, limit=limit)


def evaluate(search_fn, dataset: list[dict], depth: int) -> dict:
    """Runs one configuration over the dataset and returns its metrics."""
    hits = {k: 0 for k in K_VALUES}
    reciprocal_ranks = []
    per_query = []

    for item in dataset:
        expected = item["expected_file"]
        files = ranked_files(search_fn(item["query"], depth))

        rank = files.index(expected) + 1 if expected in files else None
        for k in K_VALUES:
            if rank is not None and rank <= k:
                hits[k] += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        per_query.append(
            {"query": item["query"], "expected": expected, "rank": rank, "files": files}
        )

    total = len(dataset)
    return {
        "recall": {k: hits[k] / total for k in K_VALUES},
        "mrr": sum(reciprocal_ranks) / total,
        "per_query": per_query,
    }


def print_per_query(result: dict) -> None:
    for row in result["per_query"]:
        rank = row["rank"]
        mark = f"rank {rank}" if rank else "MISS"
        status = "OK  " if rank and rank <= 3 else "    "
        print(f"  {status}{mark:<8} {row['query'][:58]}")
        if rank is None:
            print(f"          expected {row['expected']}")
            print(f"          got      {row['files'][:4]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depth",
        type=int,
        default=20,
        help="chunks fetched per query before collapsing to files (default: 20)",
    )
    parser.add_argument("--verbose", action="store_true", help="show every query's rank")
    args = parser.parse_args()

    dataset = load_dataset()
    db = DatabaseManager()

    if db.get_symbol_count() == 0:
        print("The index is empty. Run 'uv run codelens index .' first.")
        return 1

    try:
        retrievers = Retrievers(db, VectorStore())
        configurations = {
            "vector": retrievers.vector,
            "keyword": retrievers.keyword,
            "hybrid": retrievers.hybrid,
        }

        print(f"Evaluating {len(dataset)} queries at depth {args.depth}\n")

        results = {}
        for name, search_fn in configurations.items():
            results[name] = evaluate(search_fn, dataset, args.depth)
            if args.verbose:
                print(f"--- {name} ---")
                print_per_query(results[name])
                print()

        header = f"{'config':<10}" + "".join(f"{f'R@{k}':>9}" for k in K_VALUES) + f"{'MRR':>9}"
        print(header)
        print("-" * len(header))
        for name, result in results.items():
            row = f"{name:<10}"
            row += "".join(f"{result['recall'][k]:>8.0%} " for k in K_VALUES)
            row += f"{result['mrr']:>8.3f} "
            print(row)

        best = max(results, key=lambda name: results[name]["mrr"])
        print(f"\nBest by MRR: {best}")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
