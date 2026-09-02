"""Tests for DatabaseManager: schema, inserts, queries and index resets."""

import sqlite3
from dataclasses import dataclass

import pytest

from codelens.repository.db import DatabaseManager

EXPECTED_TABLES = {
    "files",
    "symbols",
    "calls",
    "chunks",
    "chat_sessions",
    "chat_messages",
    "imports",
    "inherits",
}


@dataclass
class ChunkStub:
    """Duck-typed stand-in for indexer.chunker.Chunk."""

    chunk_id: str
    file_path: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    content: str


def table_names(db: DatabaseManager) -> set[str]:
    rows = db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {row["name"] for row in rows}


class TestSchema:
    def test_creates_every_table(self, db):
        assert EXPECTED_TABLES <= table_names(db)

    def test_creates_the_lookup_indexes(self, db):
        rows = db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'").fetchall()
        names = {row["name"] for row in rows}

        assert {
            "idx_calls_caller",
            "idx_calls_callee",
            "idx_symbols_name",
            "idx_chunks_symbol",
        } <= names

    def test_creates_the_database_file_on_disk(self, tmp_path):
        path = tmp_path / "nested.db"
        manager = DatabaseManager(path)

        assert path.exists()
        manager.close()

    def test_rows_are_accessible_by_column_name(self, db):
        db.insert_file("a.py", "py", 1, 1)
        row = db.conn.execute("SELECT * FROM files").fetchone()

        assert row["path"] == "a.py"

    def test_reopening_an_existing_database_preserves_data(self, tmp_path):
        path = tmp_path / "reopen.db"
        first = DatabaseManager(path)
        first.insert_file("a.py", "py", 1, 1)
        first.close()

        second = DatabaseManager(path)
        assert second.conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"] == 1
        second.close()

    def test_close_releases_the_connection(self, tmp_path):
        manager = DatabaseManager(tmp_path / "closed.db")
        manager.close()

        with pytest.raises(sqlite3.ProgrammingError):
            manager.conn.execute("SELECT 1")


class TestInserts:
    def test_insert_file_stores_all_columns(self, db):
        db.insert_file("src/app.py", "py", 1024, 42)
        row = db.conn.execute("SELECT * FROM files").fetchone()

        assert (row["path"], row["language"], row["size"], row["lines"]) == (
            "src/app.py",
            "py",
            1024,
            42,
        )

    def test_insert_file_replaces_an_existing_path(self, db):
        db.insert_file("src/app.py", "py", 1, 1)
        db.insert_file("src/app.py", "py", 2, 2)

        rows = db.conn.execute("SELECT * FROM files").fetchall()
        assert len(rows) == 1
        assert rows[0]["lines"] == 2

    def test_insert_symbol_replaces_an_existing_id(self, db):
        db.insert_symbol("id", "run", "function", "a.py", 1)
        db.insert_symbol("id", "run", "method", "a.py", 7)

        rows = db.conn.execute("SELECT * FROM symbols").fetchall()
        assert len(rows) == 1
        assert rows[0]["type"] == "method"
        assert rows[0]["line_number"] == 7

    def test_insert_call_appends_rather_than_replacing(self, db):
        db.insert_call("caller", "callee", 3)
        db.insert_call("caller", "callee", 9)

        rows = db.conn.execute("SELECT line_number FROM calls ORDER BY line_number").fetchall()
        assert [row["line_number"] for row in rows] == [3, 9]

    def test_insert_import_keeps_null_module_and_alias(self, db):
        db.insert_import("a.py", None, "os", None)
        row = db.conn.execute("SELECT * FROM imports").fetchone()

        assert row["module"] is None
        assert row["name"] == "os"
        assert row["alias"] is None

    def test_insert_import_stores_from_imports_with_aliases(self, db):
        db.insert_import("a.py", "typing", "Optional", "Opt")
        row = db.conn.execute("SELECT * FROM imports").fetchone()

        assert (row["module"], row["name"], row["alias"]) == ("typing", "Optional", "Opt")

    def test_insert_inherit_records_a_base_class(self, db):
        db.insert_inherit("a.py::View", "BaseView")
        row = db.conn.execute("SELECT * FROM inherits").fetchone()

        assert (row["class_id"], row["base_name"]) == ("a.py::View", "BaseView")

    def test_insert_inherit_supports_multiple_bases(self, db):
        db.insert_inherit("a.py::View", "Mixin")
        db.insert_inherit("a.py::View", "BaseView")

        rows = db.conn.execute("SELECT base_name FROM inherits ORDER BY base_name").fetchall()
        assert [row["base_name"] for row in rows] == ["BaseView", "Mixin"]


class TestSymbolSearch:
    def test_search_symbols_matches_a_substring(self, populated_db):
        results = populated_db.search_symbols("erv")

        assert [row["name"] for row in results] == ["Service"]

    def test_search_symbols_returns_nothing_for_an_unknown_name(self, populated_db):
        assert populated_db.search_symbols("nope") == []

    def test_search_symbols_caps_the_result_count(self, db):
        for i in range(30):
            db.insert_symbol(f"id{i}", f"handler_{i}", "function", "a.py", i + 1)

        assert len(db.search_symbols("handler")) == 15

    def test_get_symbol_at_returns_the_definition_on_that_line(self, populated_db):
        row = populated_db.get_symbol_at("src/db.py", 42)

        assert row["name"] == "connect"

    def test_get_symbol_at_returns_none_off_a_definition(self, populated_db):
        assert populated_db.get_symbol_at("src/db.py", 43) is None

    def test_get_symbols_in_file_is_ordered_by_line(self, populated_db):
        rows = populated_db.get_symbols_in_file("src/app.py")

        assert [row["name"] for row in rows] == ["Service", "run"]

    def test_get_symbols_in_file_is_scoped_to_that_file(self, populated_db):
        rows = populated_db.get_symbols_in_file("src/db.py")

        assert [row["name"] for row in rows] == ["connect"]

    def test_get_symbol_locations_maps_names_to_places(self, populated_db):
        locations = populated_db.get_symbol_locations(["connect", "Service"])

        assert locations["connect"] == [("src/db.py", 42)]
        assert locations["Service"] == [("src/app.py", 5)]

    def test_get_symbol_locations_omits_unknown_names(self, populated_db):
        assert populated_db.get_symbol_locations(["enumerate"]) == {}

    def test_get_symbol_locations_short_circuits_on_an_empty_list(self, populated_db):
        assert populated_db.get_symbol_locations([]) == {}

    def test_get_symbol_locations_lists_every_definition_of_a_name(self, db):
        db.insert_symbol("a.py::run", "run", "function", "a.py", 1)
        db.insert_symbol("b.py::run", "run", "function", "b.py", 9)

        assert sorted(db.get_symbol_locations(["run"])["run"]) == [("a.py", 1), ("b.py", 9)]


class TestCallGraph:
    def test_get_outgoing_calls_lists_callees_with_lines(self, populated_db):
        rows = populated_db.get_outgoing_calls("src/app.py::Service.run")

        assert sorted((row["callee_name"], row["line_number"]) for row in rows) == [
            ("connect", 12),
            ("print", 13),
        ]

    def test_get_outgoing_calls_is_empty_for_a_leaf_symbol(self, populated_db):
        assert populated_db.get_outgoing_calls("src/db.py::connect") == []

    def test_get_incoming_calls_resolves_the_caller_symbol(self, populated_db):
        rows = populated_db.get_incoming_calls("connect")

        assert len(rows) == 1
        assert rows[0]["caller_name"] == "run"
        assert rows[0]["file_path"] == "src/app.py"
        assert rows[0]["line_number"] == 12

    def test_get_incoming_calls_ignores_calls_from_unknown_symbols(self, db):
        db.insert_call("ghost::caller", "target", 1)

        assert db.get_incoming_calls("target") == []


class TestChunks:
    def test_save_chunks_persists_every_field(self, db):
        db.save_chunks([ChunkStub("a.py::run:1", "a.py", "run", "function", 1, 4, "def run(): ...")])
        row = db.conn.execute("SELECT * FROM chunks").fetchone()

        assert row["chunk_id"] == "a.py::run:1"
        assert row["symbol_name"] == "run"
        assert row["symbol_type"] == "function"
        assert (row["start_line"], row["end_line"]) == (1, 4)
        assert row["content"] == "def run(): ..."

    def test_save_chunks_clears_stale_chunks_first(self, db):
        db.save_chunks([ChunkStub("old", "a.py", "old", "function", 1, 2, "old")])
        db.save_chunks([ChunkStub("new", "a.py", "new", "function", 1, 2, "new")])

        rows = db.conn.execute("SELECT chunk_id FROM chunks").fetchall()
        assert [row["chunk_id"] for row in rows] == ["new"]

    def test_save_chunks_with_an_empty_list_wipes_the_table(self, db):
        db.save_chunks([ChunkStub("old", "a.py", "old", "function", 1, 2, "old")])
        db.save_chunks([])

        assert db.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"] == 0

    def test_keyword_search_matches_the_code_content(self, populated_db):
        assert len(populated_db.search_chunks_keyword("class Service")) == 1

    def test_keyword_search_matches_the_symbol_name(self, populated_db):
        assert len(populated_db.search_chunks_keyword("Service")) == 1

    def test_keyword_search_matches_the_file_path(self, populated_db):
        assert len(populated_db.search_chunks_keyword("src/app.py")) == 1

    def test_keyword_search_returns_nothing_on_a_miss(self, populated_db):
        assert populated_db.search_chunks_keyword("kubernetes") == []

    def test_keyword_search_honours_the_limit(self, db):
        db.save_chunks(
            [ChunkStub(f"c{i}", "a.py", f"handler_{i}", "function", i, i + 1, "body") for i in range(10)]
        )

        assert len(db.search_chunks_keyword("handler", limit=3)) == 3


class TestChatHistory:
    def test_messages_come_back_in_chronological_order(self, db):
        db.create_chat_session("s1", title="First chat")
        db.add_chat_message("s1", "user", "hello")
        db.add_chat_message("s1", "model", "hi there")

        history = db.get_chat_history("s1")

        assert [(row["role"], row["content"]) for row in history] == [
            ("user", "hello"),
            ("model", "hi there"),
        ]

    def test_history_is_scoped_to_one_session(self, db):
        db.create_chat_session("s1")
        db.create_chat_session("s2")
        db.add_chat_message("s1", "user", "only mine")

        assert db.get_chat_history("s2") == []

    def test_creating_a_session_stores_the_title(self, db):
        db.create_chat_session("s1", title="Explain the retriever")

        assert db.get_recent_sessions()[0]["title"] == "Explain the retriever"

    def test_a_session_gets_a_default_title(self, db):
        db.create_chat_session("s1")

        assert db.get_recent_sessions()[0]["title"] == "New Chat Session"

    def test_recent_sessions_honour_the_limit(self, db):
        for i in range(8):
            db.create_chat_session(f"s{i}")

        assert len(db.get_recent_sessions(limit=3)) == 3

    def test_recent_sessions_is_empty_for_a_fresh_database(self, db):
        assert db.get_recent_sessions() == []


class TestClearAllIndexedData:
    def test_wipes_every_index_table(self, populated_db):
        populated_db.insert_import("src/app.py", "typing", "List", None)
        populated_db.insert_inherit("src/app.py::Service", "Base")

        populated_db.clear_all_indexed_data()

        for table in ("files", "symbols", "calls", "chunks", "imports", "inherits"):
            count = populated_db.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            assert count == 0, f"{table} was not cleared"

    def test_preserves_chat_history(self, populated_db):
        populated_db.create_chat_session("s1")
        populated_db.add_chat_message("s1", "user", "keep me")

        populated_db.clear_all_indexed_data()

        assert len(populated_db.get_chat_history("s1")) == 1
        assert len(populated_db.get_recent_sessions()) == 1

    def test_is_safe_to_run_on_an_empty_database(self, db):
        db.clear_all_indexed_data()

        assert db.conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"] == 0
