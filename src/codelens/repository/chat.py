"""Persistence for chat sessions.

Conversation history is deliberately kept apart from the code index: it is the
only data in the database that survives `codelens index`, and it has no
relationship to symbols, calls or chunks.
"""

import sqlite3


class ChatRepository:
    """Chat sessions and messages, stored in the same SQLite file as the index."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def create_session(self, session_id: str, title: str = "New Chat Session"):
        with self.conn:
            self.conn.execute(
                "INSERT INTO chat_sessions (id, title) VALUES (?, ?)", (session_id, title)
            )

    def add_message(self, session_id: str, role: str, content: str):
        with self.conn:
            self.conn.execute(
                "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )

    def get_history(self, session_id: str) -> list[sqlite3.Row]:
        """Returns the chat history for a specific session in chronological order."""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT role, content FROM chat_messages "
                "WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            )
            return cursor.fetchall()

    def get_recent_sessions(self, limit: int = 5) -> list[sqlite3.Row]:
        """Returns a list of recent chat sessions, newest first."""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT id, title, created_at FROM chat_sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return cursor.fetchall()
