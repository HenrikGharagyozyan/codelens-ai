"""Shared fixtures and test doubles for the CodeLens suite.

Nothing here touches the network, the user's real `.codelens.db`, or the
ChromaDB embedding model: every heavy dependency is replaced by an in-memory
double so the suite stays fast and deterministic in CI.
"""

import pytest

from codelens.repository.db import DatabaseManager


@pytest.fixture
def db(tmp_path):
    """A DatabaseManager backed by a throwaway SQLite file."""
    manager = DatabaseManager(tmp_path / "test.db")
    yield manager
    manager.close()


@pytest.fixture
def populated_db(db):
    """A database holding a small but realistic index.

    Layout:
        src/app.py   Service (class, l.5) -> Service.run (method, l.10)
        src/db.py    connect (function, l.42)
        Service.run calls connect and print.
    """
    db.insert_file("src/app.py", "py", 100, 20)
    db.insert_file("src/db.py", "py", 80, 15)

    db.insert_symbol("src/app.py::Service", "Service", "class", "src/app.py", 5)
    db.insert_symbol("src/app.py::Service.run", "run", "method", "src/app.py", 10)
    db.insert_symbol("src/db.py::connect", "connect", "function", "src/db.py", 42)

    db.insert_call("src/app.py::Service.run", "connect", 12)
    db.insert_call("src/app.py::Service.run", "print", 13)

    db.conn.execute(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "src/app.py::Service:5",
            "src/app.py",
            "Service",
            "class",
            5,
            18,
            "class Service:\n    pass",
        ),
    )
    db.conn.commit()
    return db


class FakeCollection:
    """Stand-in for a chromadb collection that records what it was given."""

    def __init__(self, name="code_chunks", metadata=None):
        self.name = name
        self.metadata = metadata
        self.upserts: list[dict] = []
        self.queries: list[tuple[list[str], int]] = []
        self.query_result: dict | None = None

    def upsert(self, ids, documents, metadatas):
        self.upserts.append({"ids": ids, "documents": documents, "metadatas": metadatas})

    def query(self, query_texts, n_results):
        self.queries.append((query_texts, n_results))
        if self.query_result is not None:
            return self.query_result
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


class FakeChromaClient:
    """Stand-in for chromadb.PersistentClient with no on-disk state."""

    def __init__(self, path=None):
        self.path = path
        self.collections: dict[str, FakeCollection] = {}
        self.deleted: list[str] = []

    def get_or_create_collection(self, name, metadata=None):
        if name not in self.collections:
            self.collections[name] = FakeCollection(name, metadata)
        return self.collections[name]

    def delete_collection(self, name):
        self.deleted.append(name)
        if name not in self.collections:
            raise ValueError(f"Collection {name} does not exist.")
        del self.collections[name]


class FakeVectorStore:
    """Minimal VectorStore replacement driven by a canned result list."""

    def __init__(self, results=None):
        self.results = results or []
        self.added: list = []
        self.cleared = False
        self.queries: list[tuple[str, int]] = []

    def search(self, query, limit=5):
        self.queries.append((query, limit))
        return self.results[:limit]

    def add_chunks(self, chunks):
        self.added.extend(chunks)

    def clear(self):
        self.cleared = True


@pytest.fixture
def fake_chroma_client(monkeypatch):
    """Replaces chromadb.PersistentClient so no embedding model is loaded."""
    from codelens.indexer import vector_store as vector_store_module

    created: list[FakeChromaClient] = []

    def _factory(path):
        client = FakeChromaClient(path=path)
        created.append(client)
        return client

    monkeypatch.setattr(vector_store_module.chromadb, "PersistentClient", _factory)
    return created
