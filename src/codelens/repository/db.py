import json
import sqlite3
from pathlib import Path

from codelens.config import DB_PATH
from codelens.repository.chat import ChatRepository
from codelens.repository.fts import build_match_query
from codelens.repository.schema import DROP_INDEX_DDL, SCHEMA_DDL, SCHEMA_VERSION

# Calls that stayed unresolved for want of information, as opposed to calls we
# know leave the repository (builtins, third-party code). Only these can be
# offered as name-matched guesses.
UNRESOLVED_CALL = "(c.callee_id IS NULL AND (c.resolution IS NULL OR c.resolution = 'unresolved'))"


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

    # ------------------------------------------------------------------ writes

    def insert_file(self, path: str, language: str, size: int, lines: int):
        with self.conn:
            # INSERT OR REPLACE updates the record if it already exists
            self.conn.execute(
                "INSERT OR REPLACE INTO files (path, language, size, lines) VALUES (?, ?, ?, ?)",
                (path, language, size, lines),
            )

    def insert_symbol(
        self,
        symbol_id: str,
        name: str,
        sym_type: str,
        file_path: str,
        line_number: int,
        *,
        qualname: str | None = None,
        end_line: int | None = None,
        signature: str | None = None,
        decorators: list[str] | None = None,
        parent_id: str | None = None,
    ):
        row = (
            symbol_id,
            name,
            qualname or name,
            sym_type,
            file_path,
            line_number,
            end_line,
            signature,
            json.dumps(decorators) if decorators else None,
            parent_id,
        )
        self.insert_symbols_batch([row])

    def insert_call(
        self,
        caller_id: str,
        callee_name: str,
        line_number: int,
        receiver: str | None = None,
        callee_id: str | None = None,
        resolution: str | None = None,
    ):
        self.insert_calls_batch([(caller_id, callee_name, line_number, receiver, callee_id, resolution)])

    def insert_import(self, file_path: str, module: str | None, name: str, alias: str | None, level: int = 0):
        self.insert_imports_batch([(file_path, module, name, alias, level)])

    def insert_inherit(self, class_id: str, base_name: str, base_id: str | None = None):
        self.insert_inherits_batch([(class_id, base_name, base_id)])

    # Batch inserts: one transaction per table instead of one per row.
    def insert_files_batch(self, rows: list[tuple[str, str, int, int]]):
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO files (path, language, size, lines) VALUES (?, ?, ?, ?)", rows
            )

    def insert_symbols_batch(self, rows: list[tuple]):
        """Rows: (id, name, qualname, type, file_path, line_number, end_line, signature, decorators, parent_id)."""
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO symbols "
                "(id, name, qualname, type, file_path, line_number, end_line, signature, decorators, parent_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def insert_calls_batch(self, rows: list[tuple]):
        """Rows: (caller_id, callee_name, line_number, receiver, callee_id, resolution)."""
        with self.conn:
            self.conn.executemany(
                "INSERT INTO calls (caller_id, callee_name, line_number, receiver, callee_id, resolution) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    def insert_imports_batch(self, rows: list[tuple]):
        """Rows: (file_path, module, name, alias, level)."""
        with self.conn:
            self.conn.executemany(
                "INSERT INTO imports (file_path, module, name, alias, level) VALUES (?, ?, ?, ?, ?)", rows
            )

    def insert_inherits_batch(self, rows: list[tuple]):
        """Rows: (class_id, base_name, base_id)."""
        with self.conn:
            self.conn.executemany("INSERT INTO inherits (class_id, base_name, base_id) VALUES (?, ?, ?)", rows)

    # ----------------------------------------------------------------- symbols

    def search_symbols(self, query: str) -> list[sqlite3.Row]:
        """Searches for symbols by partial name match."""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT * FROM symbols WHERE name LIKE ? LIMIT 15",
                (f"%{query}%",),
            )
            return cursor.fetchall()

    def get_symbol(self, symbol_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM symbols WHERE id = ?", (symbol_id,)).fetchone()

    def find_symbols(self, query: str, limit: int = 15) -> list[sqlite3.Row]:
        """Finds symbols for a user-typed reference, best match tier first.

        Tries, in order: the full id (`src/app.py::Service.run`), the qualified
        name (`Service.run`), the bare name (`run`), then a substring of the
        qualified name. The first tier with any hit wins, so `run` never loses
        to `runner` just because both contain the letters.
        """
        tiers = (
            ("id = ?", query),
            ("qualname = ?", query),
            ("name = ?", query),
            ("qualname LIKE ?", f"%{query}%"),
        )
        for condition, value in tiers:
            rows = self.conn.execute(
                f"SELECT * FROM symbols WHERE {condition} "
                # Prefer definitions over variables, and real code over tests.
                "ORDER BY type = 'variable', file_path LIKE '%test%', file_path, line_number LIMIT ?",
                (value, limit),
            ).fetchall()
            if rows:
                return rows
        return []

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
            cursor = self.conn.execute("SELECT * FROM symbols WHERE file_path = ? ORDER BY line_number", (file_path,))
            return cursor.fetchall()

    def get_symbol_count(self) -> int:
        """Returns the total number of unique symbols in the database."""
        with self.conn:
            cursor = self.conn.execute("SELECT COUNT(*) FROM symbols")
            return cursor.fetchone()[0]

    # -------------------------------------------------------------- call graph

    def get_outgoing_calls(self, symbol_id: str) -> list[sqlite3.Row]:
        """Returns every call made by the specified symbol, resolved or not."""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT callee_name, line_number, receiver, callee_id, resolution FROM calls WHERE caller_id = ?",
                (symbol_id,),
            )
            return cursor.fetchall()

    def get_incoming_calls(self, callee_name: str) -> list[sqlite3.Row]:
        """Returns the symbols containing a call to `callee_name`, matched by name only.

        Kept for symbols that have no id to resolve against; prefer
        `get_callers`, which follows resolved edges.
        """
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

    def get_callers(self, symbol_id: str) -> list[sqlite3.Row]:
        """Resolved call edges pointing at `symbol_id`, with the calling symbol."""
        return self.conn.execute(
            """
            SELECT c.caller_id, c.line_number, c.resolution,
                   s.name AS caller_name, s.qualname AS caller_qualname, s.type AS caller_type,
                   s.file_path AS caller_file, s.line_number AS caller_line
            FROM calls c
            JOIN symbols s ON s.id = c.caller_id
            WHERE c.callee_id = ?
            ORDER BY s.file_path, c.line_number
            """,
            (symbol_id,),
        ).fetchall()

    def get_callees(self, symbol_id: str) -> list[sqlite3.Row]:
        """Every call made by `symbol_id`, joined with its target when it was resolved."""
        return self.conn.execute(
            """
            SELECT c.callee_name, c.line_number, c.receiver, c.callee_id, c.resolution,
                   s.qualname AS callee_qualname, s.type AS callee_type,
                   s.file_path AS callee_file, s.line_number AS callee_line
            FROM calls c
            LEFT JOIN symbols s ON s.id = c.callee_id
            WHERE c.caller_id = ?
            ORDER BY c.line_number
            """,
            (symbol_id,),
        ).fetchall()

    def get_name_matched_callers(self, callee_name: str) -> list[sqlite3.Row]:
        """Unresolved calls that merely share the name `callee_name`.

        These are guesses, not edges: the resolver could not tell what the call
        targets. Callers must label them as such.
        """
        return self.conn.execute(
            f"""
            SELECT c.caller_id, c.line_number, c.resolution,
                   s.name AS caller_name, s.qualname AS caller_qualname, s.type AS caller_type,
                   s.file_path AS caller_file, s.line_number AS caller_line
            FROM calls c
            JOIN symbols s ON s.id = c.caller_id
            WHERE c.callee_name = ? AND {UNRESOLVED_CALL}
            ORDER BY s.file_path, c.line_number
            """,
            (callee_name,),
        ).fetchall()

    def get_call_resolution_counts(self) -> dict[str, int]:
        """How many call sites were resolved, and how, keyed by `calls.resolution`."""
        rows = self.conn.execute(
            "SELECT COALESCE(resolution, 'unresolved') AS how, COUNT(*) AS n FROM calls GROUP BY how"
        ).fetchall()
        return {row["how"]: row["n"] for row in rows}

    # ------------------------------------------------------------------ chunks

    def get_chunk_at(self, file_path: str, start_line: int) -> sqlite3.Row | None:
        """The symbol chunk starting at file_path:start_line (module summaries excluded)."""
        return self.conn.execute(
            "SELECT * FROM chunks WHERE file_path = ? AND start_line = ? AND symbol_type != 'module' LIMIT 1",
            (file_path, start_line),
        ).fetchone()

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
