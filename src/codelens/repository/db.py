import sqlite3
from pathlib import Path

from codelens.config import DB_PATH
from codelens.repository.chat import ChatRepository
from codelens.repository.fts import build_match_query
from codelens.repository.schema import DROP_INDEX_DDL, SCHEMA_DDL, SCHEMA_VERSION


class DatabaseManager:
    """The code index: files, symbols, calls, imports, inheritance and chunks.

        Chat history lives in the same file but is reached through `.chat`, since it
        is the one thing here that re-indexing must not touch.
    """

    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = Path(db_path)
        # Connect to the database file (if it doesn't exist, it will be created automatically)
        self.conn = sqlite3.connect(self.db_path)
        # This setting allows accessing columns by name: row['name']
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self.chat = ChatRepository(self.conn)

    def _create_tables(self):
        """Creates the schema, rebuilding the index tables after a version bump."""
        current = self.conn.execute("PRAGMA user_version").fetchone()[0]

        # The DDL is all `IF NOT EXISTS`, so an existing table would keep its old
        # definition forever. On a version change we drop the index tables and
        # let them be recreated; chat history is not among them and survives.
        if current and current != SCHEMA_VERSION:
            with self.conn:
                self.conn.executescript(DROP_INDEX_DDL)

        # The with block automatically commits the transaction if there are no errors
        with self.conn:
            self.conn.executescript(SCHEMA_DDL)
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def insert_file(self, path: str, language: str, size: int, lines: int):
        with self.conn:
            # INSERT OR REPLACE updates the record if it already exists
            self.conn.execute(
                "INSERT OR REPLACE INTO files (path, language, size, lines) VALUES (?, ?, ?, ?)",
                (path, language, size, lines),
            )

    def insert_symbol(
        self, symbol_id: str, name: str, sym_type: str, file_path: str, line_number: int
    ):
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO symbols (id, name, type, file_path, line_number) "
                "VALUES (?, ?, ?, ?, ?)",
                (symbol_id, name, sym_type, file_path, line_number),
            )

    def insert_call(self, caller_id: str, callee_name: str, line_number: int):
        with self.conn:
            self.conn.execute(
                "INSERT INTO calls (caller_id, callee_name, line_number) VALUES (?, ?, ?)",
                (caller_id, callee_name, line_number),
            )

    def insert_import(self, file_path: str, module: str | None, name: str, alias: str | None):
        with self.conn:
            self.conn.execute(
                "INSERT INTO imports (file_path, module, name, alias) VALUES (?, ?, ?, ?)",
                (file_path, module, name, alias),
            )

    def insert_inherit(self, class_id: str, base_name: str):
        with self.conn:
            self.conn.execute(
                "INSERT INTO inherits (class_id, base_name) VALUES (?, ?)", (class_id, base_name)
            )

    # New batch insert methods for performance when indexing many symbols/files at once
    def insert_files_batch(self, rows: list[tuple[str, str, int, int]]):
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO files (path, language, size, lines) VALUES (?, ?, ?, ?)", rows
            )

    def insert_symbols_batch(self, rows: list[tuple[str, str, str, str, int]]):
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO symbols (id, name, type, file_path, line_number) VALUES (?, ?, ?, ?, ?)", rows
            )

    def insert_calls_batch(self, rows: list[tuple[str, str, int]]):
        with self.conn:
            self.conn.executemany(
                "INSERT INTO calls (caller_id, callee_name, line_number) VALUES (?, ?, ?)", rows
            )

    def insert_imports_batch(self, rows: list[tuple[str, str | None, str, str | None]]):
        with self.conn:
            self.conn.executemany(
                "INSERT INTO imports (file_path, module, name, alias) VALUES (?, ?, ?, ?)", rows
            )

    def insert_inherits_batch(self, rows: list[tuple[str, str]]):
        with self.conn:
            self.conn.executemany(
                "INSERT INTO inherits (class_id, base_name) VALUES (?, ?)", rows
            )

    def search_symbols(self, query: str) -> list[sqlite3.Row]:
        """Searches for symbols by partial name match."""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT * FROM symbols WHERE name LIKE ? LIMIT 15",
                (f"%{query}%",),
            )
            return cursor.fetchall()

    def search_chunks_keyword(self, query: str, limit: int = 10) -> list[sqlite3.Row]:
        """Lexical search using SQLite FTS5 (BM25) for true relevance ranking."""
        # Quoting the whole query would make this an exact-phrase search, which
        # matches nothing for a natural-language question. Split it into terms so
        # BM25 can rank by term overlap and rarity.
        match_query = build_match_query(query)
        if match_query is None:
            return []

        with self.conn:
            cursor = self.conn.execute(
                """
                SELECT chunks.* 
                FROM chunks 
                JOIN chunks_fts ON chunks.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ? 
                ORDER BY bm25(chunks_fts) 
                LIMIT ?
                """,
                (match_query, limit),
            )
            return cursor.fetchall()

    def get_symbol_locations(self, names: list[str]) -> dict[str, list[tuple[str, int]]]:
        """
        Resolves symbol names to their real (file_path, line_number) locations.
        """
        if not names:
            return {}

        placeholders = ",".join("?" * len(names))
        with self.conn:
            cursor = self.conn.execute(
                f"SELECT name, file_path, line_number FROM symbols WHERE name IN ({placeholders})",
                names,
            )
            locations: dict[str, list[tuple[str, int]]] = {}
            for row in cursor:
                locations.setdefault(row["name"], []).append((row["file_path"], row["line_number"]))
            return locations

    def get_symbol_at(self, file_path: str, line_number: int) -> sqlite3.Row | None:
        """Returns the symbol defined exactly at file_path:line_number, if any."""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT * FROM symbols WHERE file_path = ? AND line_number = ?",
                (file_path, line_number),
            )
            return cursor.fetchone()

    def get_symbols_in_file(self, file_path: str) -> list[sqlite3.Row]:
        """Returns every symbol defined in a file, ordered by line number."""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT * FROM symbols WHERE file_path = ? ORDER BY line_number", (file_path,)
            )
            return cursor.fetchall()

    def get_outgoing_calls(self, symbol_id: str) -> list[sqlite3.Row]:
        """Returns a list of all functions called by the specified symbol."""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT callee_name, line_number FROM calls WHERE caller_id = ?", (symbol_id,)
            )
            return cursor.fetchall()

    def get_incoming_calls(self, callee_name: str) -> list[sqlite3.Row]:
        """Returns a list of functions/symbols that call the specified callee_name."""
        with self.conn:
            cursor = self.conn.execute(
                """
                SELECT s.name AS caller_name, s.file_path, c.line_number 
                FROM calls c
                JOIN symbols s ON c.caller_id = s.id
                WHERE c.callee_name = ?
            """,
                (callee_name,),
            )
            return cursor.fetchall()

    def get_symbol_count(self) -> int:
        """Returns the total number of unique symbols in the database."""
        with self.conn:
            cursor = self.conn.execute("SELECT COUNT(*) FROM symbols")
            return cursor.fetchone()[0]

    def save_chunks(self, chunks: list) -> None:
        """Saves semantic code chunks to the database."""
        with self.conn:
            # Clearing chunks will automatically fire the AFTER DELETE trigger for FTS
            self.conn.execute("DELETE FROM chunks")
            self.conn.executemany(
                """
                INSERT INTO chunks
                    (chunk_id, file_path, symbol_name, symbol_type, start_line, end_line, content)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    (
                        c.chunk_id,
                        c.file_path,
                        c.symbol_name,
                        c.symbol_type,
                        c.start_line,
                        c.end_line,
                        c.content,
                    )
                    for c in chunks
                ],
            )

    def clear_all_indexed_data(self):
        """Fully clears the old index data before a new scan."""
        with self.conn:
            self.conn.execute("DELETE FROM calls")
            self.conn.execute("DELETE FROM symbols")
            self.conn.execute("DELETE FROM chunks")  # Triggers sync with chunks_fts automatically
            self.conn.execute("DELETE FROM files")
            self.conn.execute("DELETE FROM imports")
            self.conn.execute("DELETE FROM inherits")

    def close(self):
        self.conn.close()
