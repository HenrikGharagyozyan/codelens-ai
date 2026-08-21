import sqlite3
from pathlib import Path


class DatabaseManager:
    def __init__(self, db_path: str | Path = ".codelens.db"):
        self.db_path = Path(db_path)
        # Connect to the database file (if it doesn't exist, it will be created automatically)
        self.conn = sqlite3.connect(self.db_path)
        # This setting allows accessing columns by name: row['name']
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Creates tables if they do not yet exist."""
        # The with block automatically commits the transaction if there are no errors
        with self.conn:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    language TEXT NOT NULL,
                    size INTEGER,
                    lines INTEGER
                );

                CREATE TABLE IF NOT EXISTS symbols (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,  -- 'class' or 'function'
                    file_path TEXT NOT NULL,
                    line_number INTEGER,
                    FOREIGN KEY (file_path) REFERENCES files(path)
                );

                CREATE TABLE IF NOT EXISTS calls (
                    caller_id TEXT,
                    callee_name TEXT,
                    line_number INTEGER,
                    FOREIGN KEY (caller_id) REFERENCES symbols(id)
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_path TEXT,
                    symbol_name TEXT,
                    symbol_type TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    content TEXT
                );
            """)

    def insert_file(self, path: str, language: str, size: int, lines: int):
        with self.conn:
            # INSERT OR REPLACE updates the record if it already exists
            self.conn.execute(
                "INSERT OR REPLACE INTO files (path, language, size, lines) VALUES (?, ?, ?, ?)",
                (path, language, size, lines)
            )

    def insert_symbol(self, symbol_id: str, name: str, sym_type: str, file_path: str, line_number: int):
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO symbols (id, name, type, file_path, line_number) VALUES (?, ?, ?, ?, ?)",
                (symbol_id, name, sym_type, file_path, line_number)
            )

    def insert_call(self, caller_id: str, callee_name: str, line_number: int):
        with self.conn:
            self.conn.execute(
                "INSERT INTO calls (caller_id, callee_name, line_number) VALUES (?, ?, ?)",
                (caller_id, callee_name, line_number)
            )

    def search_symbols(self, query: str) -> list[sqlite3.Row]:
        """Searches for symbols by partial name match."""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT * FROM symbols WHERE name LIKE ? LIMIT 15",
                (f"%{query}%",)  # % means any text before and after the query
            )
            return cursor.fetchall()

    def get_outgoing_calls(self, symbol_id: str) -> list[sqlite3.Row]:
        """Returns a list of all functions called by the specified symbol."""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT callee_name, line_number FROM calls WHERE caller_id = ?",
                (symbol_id,)
            )
            return cursor.fetchall()

    def save_chunks(self, chunks: list) -> None:
        """Saves semantic code chunks to the database."""
        with self.conn:
            self.conn.execute("DELETE FROM chunks")  # Очищаем старые чанки при переиндексации
            self.conn.executemany("""
                INSERT INTO chunks (chunk_id, file_path, symbol_name, symbol_type, start_line, end_line, content)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                (c.chunk_id, c.file_path, c.symbol_name, c.symbol_type, c.start_line, c.end_line, c.content)
                for c in chunks
            ])
    
    def close(self):
        self.conn.close()