"""Project-wide defaults.

Both stores live in the working directory and are rebuilt by `codelens index`,
so they are build artifacts rather than configuration — but keeping the names in
one place stops them drifting apart between modules.
"""

DB_PATH = ".codelens.db"
VECTOR_DB_PATH = ".codelens_vector"

# Let call-graph neighbours of the best hits into the search ranking.
# scripts/eval_recall.py compares both settings; flip this if it stops winning.
GRAPH_EXPANSION = True
