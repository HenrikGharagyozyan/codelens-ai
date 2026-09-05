"""Project-wide defaults.

Both stores live in the working directory and are rebuilt by `codelens index`,
so they are build artifacts rather than configuration — but keeping the names in
one place stops them drifting apart between modules.
"""

DB_PATH = ".codelens.db"
VECTOR_DB_PATH = ".codelens_vector"
