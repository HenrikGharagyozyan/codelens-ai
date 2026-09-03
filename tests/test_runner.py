"""End-to-end tests for CodebaseIndexer over a real (tiny) repository on disk.

Only the vector store is faked; the scanner, parser, chunker and SQLite layer
all run for real, so this is the test that catches wiring mistakes between them.
"""

import pytest

from codelens.indexer.runner import CodebaseIndexer
from tests.conftest import FakeVectorStore

APP_PY = '''"""Application entry point."""

from db import connect


class Base:
    pass


class Service(Base):
    """Does the work."""

    def run(self):
        return connect()


def main():
    return Service().run()
'''

DB_PY = """def connect():
    return "connection"
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A two-file Python project, with the CWD moved into it.

    CodebaseIndexer builds a DatabaseManager with the default path, so the CWD
    decides where `.codelens.db` lands. Moving it keeps the developer's real
    index untouched.
    """
    (tmp_path / "app.py").write_text(APP_PY, encoding="utf-8")
    (tmp_path / "db.py").write_text(DB_PY, encoding="utf-8")
    (tmp_path / "notes.md").write_text("# Notes\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def fake_store():
    """Returns a FakeVectorStore so no embedding model is loaded during indexing."""
    return FakeVectorStore()


@pytest.fixture
def indexed(project, fake_store):
    """Runs a full index and yields (indexer, result)."""
    indexer = CodebaseIndexer(str(project), vector_store=fake_store)
    result = indexer.run()
    yield indexer, result
    indexer.db.close()


class TestRunSummary:
    def test_reports_the_file_and_symbol_counts_and_db_path(self, indexed, project):
        _, (files_count, symbols_count, db_path) = indexed

        assert files_count == 3
        # Base, Service, Service.run, main, connect
        assert symbols_count == 5
        assert db_path == (project / ".codelens.db").resolve()

    def test_creates_the_database_file(self, indexed, project):
        assert (project / ".codelens.db").exists()


class TestFileIndexing:
    def test_records_every_scanned_file_including_non_python(self, indexed):
        indexer, _ = indexed
        rows = indexer.db.conn.execute("SELECT path, language FROM files").fetchall()

        assert {row["path"] for row in rows} == {"app.py", "db.py", "notes.md"}

    def test_only_python_files_produce_symbols(self, indexed):
        indexer, _ = indexed
        rows = indexer.db.conn.execute("SELECT DISTINCT file_path FROM symbols").fetchall()

        assert {row["file_path"] for row in rows} == {"app.py", "db.py"}


class TestSymbolIndexing:
    def test_classes_methods_and_functions_are_typed_correctly(self, indexed):
        indexer, _ = indexed
        rows = indexer.db.conn.execute("SELECT name, type FROM symbols").fetchall()
        types = {row["name"]: row["type"] for row in rows}

        assert types == {
            "Base": "class",
            "Service": "class",
            "run": "method",
            "main": "function",
            "connect": "function",
        }

    def test_symbol_ids_are_namespaced_by_file_and_class(self, indexed):
        indexer, _ = indexed
        ids = {row["id"] for row in indexer.db.conn.execute("SELECT id FROM symbols")}

        assert "app.py::Service" in ids
        assert "app.py::Service.run" in ids
        assert "db.py::connect" in ids

    def test_symbol_paths_are_relative_to_the_repository_root(self, indexed):
        indexer, _ = indexed
        rows = indexer.db.conn.execute("SELECT file_path FROM symbols").fetchall()

        assert all(not row["file_path"].startswith("/") for row in rows)

    def test_line_numbers_point_at_the_definitions(self, indexed):
        indexer, _ = indexed
        row = indexer.db.get_symbol_at("db.py", 1)

        assert row["name"] == "connect"


class TestRelationships:
    def test_call_edges_record_the_real_call_line(self, indexed):
        indexer, _ = indexed
        rows = indexer.db.get_outgoing_calls("app.py::Service.run")

        assert [(row["callee_name"], row["line_number"]) for row in rows] == [("connect", 14)]

    def test_incoming_calls_resolve_across_files(self, indexed):
        indexer, _ = indexed
        rows = indexer.db.get_incoming_calls("connect")

        assert [row["caller_name"] for row in rows] == ["run"]

    def test_imports_are_recorded_against_their_file(self, indexed):
        indexer, _ = indexed
        rows = indexer.db.conn.execute("SELECT * FROM imports").fetchall()

        assert [(row["file_path"], row["module"], row["name"]) for row in rows] == [
            ("app.py", "db", "connect")
        ]

    def test_inheritance_edges_are_recorded(self, indexed):
        """Regression: the runner used to call a method the database never had."""
        indexer, _ = indexed
        rows = indexer.db.conn.execute("SELECT class_id, base_name FROM inherits").fetchall()

        edges = [(row["class_id"], row["base_name"]) for row in rows]

        assert edges == [("app.py::Service", "Base")]


class TestChunkingAndVectors:
    def test_chunks_are_persisted_for_every_symbol(self, indexed):
        indexer, _ = indexed
        rows = indexer.db.conn.execute("SELECT symbol_name FROM chunks").fetchall()

        assert {row["symbol_name"] for row in rows} == {"Base", "Service", "run", "main", "connect"}

    def test_the_same_chunks_are_sent_to_the_vector_store(self, indexed, fake_store):
        indexer, _ = indexed
        db_ids = {row["chunk_id"] for row in indexer.db.conn.execute("SELECT chunk_id FROM chunks")}

        assert {c.chunk_id for c in fake_store.added} == db_ids

    def test_chunk_line_ranges_match_the_source(self, indexed):
        indexer, _ = indexed
        row = indexer.db.conn.execute(
            "SELECT start_line, end_line FROM chunks WHERE symbol_name = 'connect'"
        ).fetchone()

        assert (row["start_line"], row["end_line"]) == (1, 2)


class TestReindexing:
    def test_clears_sqlite_and_the_vector_store_before_indexing(self, project, fake_store):
        indexer = CodebaseIndexer(str(project), vector_store=fake_store)
        indexer.db.insert_file("stale.py", "py", 1, 1)
        indexer.db.insert_symbol("stale::x", "x", "function", "stale.py", 1)

        indexer.run()

        rows = indexer.db.conn.execute("SELECT path FROM files").fetchall()
        assert "stale.py" not in {row["path"] for row in rows}
        assert fake_store.cleared is True
        indexer.db.close()

    def test_running_twice_does_not_duplicate_rows(self, project, fake_store):
        indexer = CodebaseIndexer(str(project), vector_store=fake_store)
        first_files, first_symbols, _ = indexer.run()
        second_files, second_symbols, _ = indexer.run()

        assert (first_files, first_symbols) == (second_files, second_symbols)
        calls = indexer.db.get_outgoing_calls("app.py::Service.run")
        assert len(calls) == 1
        indexer.db.close()


class TestDegenerateRepositories:
    def test_an_empty_repository_indexes_to_zero(self, tmp_path, monkeypatch, fake_store):
        monkeypatch.chdir(tmp_path)
        indexer = CodebaseIndexer(str(tmp_path), vector_store=fake_store)

        files_count, symbols_count, _ = indexer.run()

        # `.codelens.db` itself is not decodable as UTF-8, so the scanner skips it.
        assert (files_count, symbols_count) == (0, 0)
        indexer.db.close()

    def test_a_file_with_a_syntax_error_does_not_stop_the_run(self, project, fake_store):
        (project / "broken.py").write_text("def oops(:\n", encoding="utf-8")
        indexer = CodebaseIndexer(str(project), vector_store=fake_store)

        files_count, symbols_count, _ = indexer.run()

        assert files_count == 4  # 3 sources + broken.py
        assert symbols_count == 5
        indexer.db.close()
