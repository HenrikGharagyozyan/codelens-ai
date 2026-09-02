# CodeLens AI

[![CI](https://github.com/HenrikGharagyozyan/codelens-ai/actions/workflows/ci.yaml/badge.svg)](https://github.com/HenrikGharagyozyan/codelens-ai/actions/workflows/ci.yaml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Ask your codebase questions and get answers with citations you can trust.**

CodeLens is a developer tool, not another RAG chatbot. It parses your repository
into an AST symbol graph, indexes it with hybrid search, and gives the LLM real
line numbers — so every `file.py:42` in an answer points at code that actually
exists.

```
$ codelens ask "How does hybrid search work?"

Hybrid search is implemented by _hybrid_search in
src/codelens/context/retriever.py:23. It fuses semantic vector search and
lexical keyword search using Reciprocal Rank Fusion...

Citations: 5/5 verified against the index.
```

---

## Why not just embed everything?

Plain vector RAG over code has three failure modes CodeLens is built to avoid.

**1. Fixed-size chunks cut code in half.** A 500-token window splits a class
between two chunks and neither one is understandable. CodeLens chunks on AST
boundaries — a function, a method, a class skeleton — so every chunk is a
complete unit of meaning.

**2. Semantic search alone can't find `HttpClient`.** Exact identifiers are a
lexical problem, not a semantic one. CodeLens runs both searches and merges them
with Reciprocal Rank Fusion, so `HttpClient` and *"where do we open outbound
connections?"* both land on the right code.

**3. Retrieved code has no context around it.** Finding `Database.connect()`
doesn't tell you who calls it. CodeLens stores a call graph in SQLite and
attaches callers and callees — with their own verified locations — to every
retrieved chunk.

And the part most tools get wrong: **line numbers**. If the model only sees
`(File: db.py)`, it will invent a number when asked to cite one. CodeLens
prefixes every context line with its real line number, resolves every mentioned
symbol to a verified location, and re-checks the finished answer against the
index — correcting wrong numbers and stripping ones it cannot confirm.

---

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/HenrikGharagyozyan/codelens-ai.git
cd codelens-ai
uv sync
```

`ask` and `chat` need a Gemini API key — they call `gemini-3.6-flash`. Create a
`.env` in the project root:

```bash
GEMINI_API_KEY=your_key_here
```

`.env` is already in `.gitignore`. The other commands (`index`, `search`, `graph`,
`inspect`) work without a key.

---

## Quick start

```bash
# 1. Index a repository (a few seconds for a small project)
uv run codelens index .

# 2. Ask a question
uv run codelens ask "Where is the database connection created?"

# 3. Or start a conversation
uv run codelens chat
```

---

## Commands

| Command | What it does |
|---|---|
| `codelens index [path]` | Scan, parse, chunk, and embed a repository (default: `.`) |
| `codelens ask "<question>"` | One-shot question, answered with verified citations |
| `codelens chat` | Interactive session; the model searches the codebase itself |
| `codelens search <name>` | Exact symbol lookup by name |
| `codelens search-semantic "<query>"` | Vector search by meaning |
| `codelens graph <symbol>` | Show what a symbol calls |
| `codelens inspect <file.py>` | Dump the AST symbols of one file |
| `codelens inspect-chunks [--limit N]` | Inspect the semantic chunks that were indexed |
| `codelens init` | Placeholder — prints a confirmation, no side effects yet |

Every command takes `--help`.

### `search-semantic`

Finds code by meaning, not by name:

```
$ codelens search-semantic "how are code chunks embedded"

Top semantic matches for: 'how are code chunks embedded'

1. save_chunks in src/codelens/repository/db.py (Score: 0.5552)

   def save_chunks(self, chunks: list) -> None:
       """Saves semantic code chunks to the database."""
   ...
```

### `graph`

Walks the call graph built from the AST:

```
$ codelens graph build_context

Dependency Graph for: build_context
(src/codelens/context/retriever.py::ContextRetriever.build_context)

This symbol calls:
  ├── _hybrid_search()
  ├── get_symbols_in_file()
  ├── get_incoming_calls()
```

### `chat`

A conversation with tool access — the model decides when to search:

```
You: Where is authentication handled?
CodeLens: 🔍 searching codebase for 'authentication'...
          Authentication runs in src/auth/middleware.py:31 ...

You: Who calls it?
CodeLens: AuthMiddleware.handle() at src/auth/middleware.py:58 ...
```

Sessions are stored in SQLite, so you can resume an earlier conversation on
startup. History survives re-indexing.

---

## How it works

```
Repository
    │
    ▼
RepositoryScanner ──── walks the tree, honours .gitignore
    │
    ▼
PythonAstVisitor ───── classes, methods, functions, base classes, call sites
    │
    ├──────────────────────────────┐
    ▼                              ▼
SemanticChunker              DatabaseManager
  AST-boundary chunks          symbols · calls · chunks (SQLite)
    │                              │
    ▼                              │
VectorStore                        │
  ChromaDB embeddings              │
    │                              │
    └──────────┬───────────────────┘
               ▼
        ContextRetriever
          vector search + keyword search  ──► RRF fusion
          + call-graph expansion
          + real line numbers
               │
               ▼
          GeminiClient
               │
               ▼
        CitationVerifier ──── checks every file:line against the index
               │
               ▼
        Answer + verified citations
```

### Retrieval in three stages

**Hybrid search.** Vector search (ChromaDB, cosine) and keyword search (SQLite)
each return their top 10. Both rankings are fused with Reciprocal Rank Fusion:

```
score(chunk) = Σ  1 / (k + rank + 1)        k = 60
```

RRF needs no score normalisation between the two systems — only their rank
order — which is what makes it robust when one retriever returns distances and
the other returns nothing comparable.

**Graph expansion.** Each surviving chunk is enriched from SQLite with what
calls it and what it calls. Every neighbour is resolved to a real
`file_path:line_number`; symbols outside the repository are labelled
`(external, no location)` so the model does not invent a location for `len()`.

**Line-accurate context.** Code reaches the model already numbered:

```
23 | def _hybrid_search(self, query: str, limit = 4) -> list[dict]:
24 |     """
```

The model reads line numbers instead of estimating them. Chunks that contain
nested definitions also carry an explicit map of where each one starts.

### Citation verification

Even with all of that, a model can still emit a number nobody gave it.
`CitationVerifier` parses every `path:line` out of the finished answer and
checks it against the index:

- correct → left alone
- wrong line for a known symbol → **corrected** to the real one
- unverifiable → the line number is **stripped**, leaving just the file

```
Citations: 14/15 verified against the index.
  fixed: _get_exact_symbol_id -> src/codelens/context/retriever.py:10
```

---

## Storage

| Where | What |
|---|---|
| `.codelens.db` | SQLite: files, symbols, calls, chunks, chat sessions |
| `.codelens_vector/` | ChromaDB: chunk embeddings |

Both are rebuilt from scratch on every `codelens index`, so the vector store and
the symbol tables never drift apart. Chat history is preserved across re-indexing.

Both paths are covered by `.gitignore` — the index is a build artifact, not source.

---

## Limitations

Worth knowing before you point this at a large repository.

- **Python only.** Other languages are scanned for file metadata but produce no
  symbols, no chunks, and no graph edges. Tree-sitter support is on the roadmap.
- **Every index is a full rebuild.** There is no incremental mode yet, so
  re-indexing a monorepo costs the same as indexing it the first time.
- **Call edges are matched by name.** A call to `connect()` links to every symbol
  named `connect` in the index; the graph does not resolve which one is meant.
- **The lexical half of hybrid search is SQL `LIKE`,** not BM25 — it finds
  identifiers reliably but ranks them crudely.
- **Gemini only.** `ask` and `chat` are wired directly to the Google GenAI SDK;
  there is no provider abstraction yet.

---

## Development

```bash
uv sync

uv run pytest          # 260 tests
uv run ruff check .    # lint
uv run ruff format .   # format
```

Lint and tests both run in CI on every push and pull request.

| Test module | Covers |
|---|---|
| `test_db.py` | Schema, inserts, symbol/chunk queries, call graph, chat history, index reset |
| `test_citations.py` | Citation regex, verification, line correction, answer repair |
| `test_cli.py` | Every command via Typer's `CliRunner`, plus lazy `AppContext` wiring |
| `test_retriever.py` | RRF ranking, line numbering, prompt-context assembly |
| `test_parser.py` | AST extraction, plus syntax errors, binary and missing files |
| `test_gemini.py` | Client config, prompt building, tool sessions, stream parsing |
| `test_chunker.py` | Chunk boundaries, docstring handling, path normalisation |
| `test_runner.py` | A full index over a real repository on disk, end to end |
| `test_scanner.py` | `.gitignore` rules, binary skipping, file metadata |
| `test_vector_store.py` | Metadata written to ChromaDB and results parsed back |

The suite makes no network calls and loads no embedding model: ChromaDB and the
Gemini SDK are replaced by in-memory doubles, `load_dotenv` is neutralised so a
real `.env` cannot leak into a test, and everything that writes runs under a
temporary directory — your own `.codelens.db` is never touched.

---

## Project status

Working today:

- Repository scanning with `.gitignore` support
- Python AST parsing — classes, methods, functions, base classes, call sites
- Symbol table and call graph in SQLite
- AST-boundary semantic chunking
- Vector search (ChromaDB) and keyword search (SQLite)
- Hybrid retrieval with Reciprocal Rank Fusion
- Call-graph context expansion
- `import` extraction and inheritance edges, stored in SQLite
- LLM answering with verified citations
- Interactive chat with persistent sessions and tool use

On the roadmap:

- Using the `imports` and `inherits` tables in retrieval — they are indexed today,
  but the retriever does not read them yet
- FTS5/BM25 in place of `LIKE` for the lexical half of hybrid search
- Multi-hop call chains in graph retrieval
- Git integration — `git log`/`blame` aware answers
- Incremental indexing driven by file hashes
- Retrieval evaluation: Recall@K across vector / hybrid / hybrid+graph
- Multi-language parsing via Tree-sitter
- MCP server, so other AI clients can query the index

---

## License

MIT — see [LICENSE](LICENSE).
