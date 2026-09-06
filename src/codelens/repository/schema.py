"""Database schema definition."""

# Bump this whenever SCHEMA_DDL changes in a way that existing databases cannot
# simply grow into. Every statement below is `IF NOT EXISTS`, so an already
# created table is never altered -- without a version check, a schema change
# would silently apply only to brand-new databases (this is exactly how a
# tokenizer change to chunks_fts went unnoticed once).
SCHEMA_VERSION = 2

# Index tables, dropped and rebuilt on a version change. The chat tables are
# deliberately absent: conversations must survive both re-indexing and upgrades.
INDEX_OBJECTS = (
    ("trigger", "chunks_ai"),
    ("trigger", "chunks_ad"),
    ("trigger", "chunks_au"),
    ("table", "chunks_fts"),
    ("table", "chunks"),
    ("table", "inherits"),
    ("table", "imports"),
    ("table", "calls"),
    ("table", "symbols"),
    ("table", "files"),
)

DROP_INDEX_DDL = "\n".join(f"DROP {kind.upper()} IF EXISTS {name};" for kind, name in INDEX_OBJECTS)

SCHEMA_DDL = """
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

CREATE TABLE IF NOT EXISTS imports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT,
    module TEXT,
    name TEXT,
    alias TEXT,
    FOREIGN KEY (file_path) REFERENCES files(path)
);

CREATE TABLE IF NOT EXISTS inherits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id TEXT,
    base_name TEXT,
    FOREIGN KEY (class_id) REFERENCES symbols(id)
);

CREATE INDEX IF NOT EXISTS idx_calls_caller ON calls(caller_id);
CREATE INDEX IF NOT EXISTS idx_calls_callee ON calls(callee_name);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol_name);

-- FTS5 Virtual Table for true hybrid search (BM25)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    symbol_name,
    file_path,
    content='chunks',
    content_rowid='rowid',
    -- Porter stemming so a question's wording does not have to match the code's:
    -- "paths"/"path", "configured"/"configures" and "parsing"/"parse" all unify.
    tokenize='porter unicode61'
);

-- Triggers to keep FTS index synced with the chunks table automatically
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, symbol_name, file_path)
    VALUES (new.rowid, new.content, new.symbol_name, new.file_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, symbol_name, file_path)
    VALUES ('delete', old.rowid, old.content, old.symbol_name, old.file_path);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, symbol_name, file_path)
    VALUES ('delete', old.rowid, old.content, old.symbol_name, old.file_path);
    INSERT INTO chunks_fts(rowid, content, symbol_name, file_path)
    VALUES (new.rowid, new.content, new.symbol_name, new.file_path);
END;
"""
