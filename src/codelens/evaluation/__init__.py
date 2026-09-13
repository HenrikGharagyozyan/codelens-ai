"""Evaluation on third-party repositories: retrieval metrics and graded answers.

The self-evaluation in `scripts/eval_recall.py` asks questions about CodeLens
itself, which is exactly the repository whose heuristics were tuned on it.
This package measures the same thing on code CodeLens has never seen, and goes
one step further: it grades the final answers, not only the retrieval.
"""
