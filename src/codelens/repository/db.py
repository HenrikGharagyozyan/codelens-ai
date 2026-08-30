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


                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT NOT NULL, -- 'user' or 'model'
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
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

    def search_chunks_keyword(self, query: str, limit: int = 10) -> list[sqlite3.Row]:
        """Lexical search for chunks by keyword match in code content or symbol name."""
        with self.conn:
            # We search for the query in file paths, symbol names, or the raw code content
            cursor = self.conn.execute(
                """SELECT * FROM chunks 
                   WHERE content LIKE ? 
                      OR symbol_name LIKE ? 
                      OR file_path LIKE ? 
                   LIMIT ?""",
                (f"%{query}%", f"%{query}%", f"%{query}%", limit)
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
        
    def get_incoming_calls(self, callee_name: str) -> list[sqlite3.Row]:
        """Returns a list of functions/symbols that call the specified callee_name."""
        with self.conn:
            cursor = self.conn.execute("""
                SELECT s.name AS caller_name, s.file_path, c.line_number 
                FROM calls c
                JOIN symbols s ON c.caller_id = s.id
                WHERE c.callee_name = ?
            """, (callee_name,))
            return cursor.fetchall()

    def save_chunks(self, chunks: list) -> None:
        """Saves semantic code chunks to the database."""
        with self.conn:
            self.conn.execute("DELETE FROM chunks")  # Clear stale chunks during reindexing
            self.conn.executemany("""
                INSERT INTO chunks (chunk_id, file_path, symbol_name, symbol_type, start_line, end_line, content)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                (c.chunk_id, c.file_path, c.symbol_name, c.symbol_type, c.start_line, c.end_line, c.content)
                for c in chunks
            ])

    def create_chat_session(self, session_id: str, title: str = "New Chat Session"):
        with self.conn:
            self.conn.execute("INSERT INTO chat_sessions (id, title) VALUES (?, ?)", (session_id, title))

    def add_chat_message(self, session_id: str, role: str, content: str):
        with self.conn:
            self.conn.execute(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content)
            )

    def get_chat_history(self, session_id: str) -> list[sqlite3.Row]:
        """Returns the chat history for a specific session in chronological order."""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,)
            )
            return cursor.fetchall()
            
    def get_recent_sessions(self, limit: int = 5) -> list[sqlite3.Row]:
        """Returns a list of recent chat sessions."""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT id, title, created_at FROM chat_sessions ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            return cursor.fetchall()
    
    def close(self):
        self.conn.close()