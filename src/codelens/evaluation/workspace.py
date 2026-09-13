"""Checking benchmark repositories out and indexing each one into its own store.

Every repository gets its own SQLite file and vector store under
`<root>/.index/<name>/`, away from the checkout (so the index never indexes
itself) and away from the developer's own `.codelens.db`.
"""

import subprocess
from pathlib import Path

from codelens.evaluation.dataset import BenchmarkRepo
from codelens.repository.schema import SCHEMA_VERSION


class CheckoutError(RuntimeError):
    """The checkout is not the commit the benchmark was written against."""


class Workspace:
    def __init__(self, root: Path):
        self.root = Path(root)

    def checkout_path(self, repo: BenchmarkRepo) -> Path:
        return self.root / repo.name

    def index_path(self, repo: BenchmarkRepo) -> Path:
        return self.root / ".index" / repo.name

    def checkout(self, repo: BenchmarkRepo) -> Path:
        """Clones the pinned tag once, then insists it is still the pinned commit."""
        path = self.checkout_path(repo)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--quiet", "--depth", "1", "--branch", repo.ref, repo.url, str(path)],
                check=True,
            )

        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if head != repo.commit:
            # A moved tag would change the answers under the questions.
            raise CheckoutError(f"{repo.name}: expected commit {repo.commit}, found {head}")
        return path

    def stamp(self, repo: BenchmarkRepo) -> str:
        return f"{repo.commit} schema={SCHEMA_VERSION}"

    def open_index(self, repo: BenchmarkRepo, rebuild: bool = False):
        """Returns (DatabaseManager, VectorStore) for the repository, indexing it when needed.

        The index is reused while the commit and schema version match. Changes
        to the chunker or the resolver do not bump either, so pass
        `rebuild=True` after changing them.
        """
        # Imported here: chromadb and the embedding model are heavy, and the
        # dataset tooling should not pay for them.
        from codelens.indexer.runner import CodebaseIndexer
        from codelens.indexer.vector_store import VectorStore
        from codelens.repository.db import DatabaseManager

        store = self.index_path(repo)
        store.mkdir(parents=True, exist_ok=True)
        stamp_file = store / "stamp.txt"

        db = DatabaseManager(store / "codelens.db")
        vector_store = VectorStore(store / "vector")

        current = stamp_file.read_text(encoding="utf-8").strip() if stamp_file.exists() else None
        if rebuild or current != self.stamp(repo) or db.get_symbol_count() == 0:
            CodebaseIndexer(str(self.checkout_path(repo)), db=db, vector_store=vector_store).run()
            stamp_file.write_text(self.stamp(repo), encoding="utf-8")

        return db, vector_store
