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
| `codelens graph <symbol> [--depth N] [--direction callers\|callees\|both]` | Tree of who calls a symbol and what it calls |
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

Walks the resolved call graph, in either direction, to a chosen depth. The
symbol can be a name (`run`), a qualified name (`Service.run`) or a full id.

```
$ codelens graph ask --direction callees

Call graph for: ask (src/codelens/cli/commands_search.py::ask)
function at src/codelens/cli/commands_search.py:58

Calls
├── ContextRetriever.build_context()  src/codelens/context/retriever.py:336
│   ├── ContextRetriever.search()  src/codelens/context/retriever.py:152
│   └── ContextRetriever._render_chunk()  src/codelens/context/retriever.py:184
├── GeminiClient.ask()  src/codelens/llm/gemini.py:46
├── CitationVerifier.repair()  src/codelens/context/citations.py:114
│   └── CitationVerifier.verify()  src/codelens/context/citations.py:89
├── _report_citations()  src/codelens/cli/commands_search.py:93
└── builtin / external: status(), print(), Markdown()
```

```
$ codelens graph VectorStore.add_chunks --direction callers --depth 3

Called by
└── CodebaseIndexer._build_and_store_chunks  src/codelens/indexer/runner.py:184
    └── CodebaseIndexer.run  src/codelens/indexer/runner.py:41
        └── index  src/codelens/cli/commands_index.py:19
```

Calls the resolver could not follow are shown separately and marked, never
merged into the tree: `? caller ... name match only`.

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
PythonAstVisitor ───── classes, methods, functions, call sites and their receivers
    │
    ▼
SymbolTable ────────── resolves every call site to the symbol it actually calls
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

**Graph expansion.** The best hits pull their call-graph neighbours (up to two
hops, at most six per hit) into the ranking, scored as a decaying fraction of
the hit that led to them. A question like *"which command ends up writing
embeddings?"* matches the code that writes embeddings; expansion is how the
command a few hops up gets into the results.

Each retrieved chunk then carries its neighbourhood: resolved callers and
callees with real `file_path:line_number` locations, and multi-hop chains such
as `main -> Service.run -> connect`. Symbols outside the repository are
labelled `(external, no location)`, and callers that merely share a name are
labelled `[name match only]`, so the model can tell a fact from a guess.

**Line-accurate context.** Code reaches the model already numbered:

```
23 | def _hybrid_search(self, query: str, limit = 4) -> list[dict]:
24 |     """
```

The model reads line numbers instead of estimating them. Chunks that contain
nested definitions also carry an explicit map of where each one starts.

### Call resolution

A call to `connect()` could mean any of several `connect`s. The resolver follows
Python's own lookup rules, as far as they can be followed without running the
code:

| Call | Resolved through | Label |
|---|---|---|
| `helper()` | a definition in the same file | `direct` |
| `connect()` | an import, including relative imports and re-exports | `import` |
| `self.save()`, `cls()` | the enclosing class, then its bases | `self` |
| `super().save()` | the bases only | `super` |
| `db.connect()` | an imported module | `module` |
| `self.db.query()` | the receiver's type: annotations, `x = Foo()`, `self.x = ...`, property and return types | `typed` |
| `len(x)`, `names.append(...)` | builtins and builtin containers | `builtin` |
| `os.path.join(...)` | a module outside the repository | `external` |

Anything else stays `unresolved` instead of being guessed: a wrong edge is
worse than a missing one, because the model treats the graph as fact.
`codelens index` reports the split, e.g.
`Call sites: 1845 (34% resolved to a symbol, 41% builtin or external, 25% unresolved)`.
Most of the unresolved remainder is test code calling methods on untyped
fixtures; in `src/` it is about 7%.

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
- **Call resolution is static.** Calls through untyped parameters, dynamic
  dispatch (`getattr`, callbacks, dependency injection) and objects of unknown
  type stay unresolved. They are listed as `[name match only]` guesses, never
  as edges.
- **Gemini only.** `ask` and `chat` are wired directly to the Google GenAI SDK;
  there is no provider abstraction yet.

---

## Development

```bash
uv sync

uv run pytest          # full suite, no network
uv run ruff check .    # lint
uv run ruff format .   # format
```

Lint and tests both run in CI on every push and pull request.

| Test module | Covers |
|---|---|
| `test_db.py` | Schema, inserts, symbol/chunk queries, call graph, chat history, index reset |
| `test_resolver.py` | Every call-resolution rule, and every case where the resolver must not guess |
| `test_call_graph.py` | Callers, callees, multi-hop chains, depth limits, budgets, recursion |
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

## Evaluation

`scripts/eval_recall.py` measures whether the right file reaches the top of
the ranking, over 25 questions about this repository. *Lookup* questions
describe the code that answers them; *call-chain* questions describe a callee
and expect its caller, or the reverse. Every call-chain answer was checked
against the source with `grep`, not against the resolver's own output.

| Config | Lookup MRR | Call-chain MRR | Call-chain R@5 | All MRR | All R@5 |
|---|---|---|---|---|---|
| vector | 0.661 | 0.172 | 40% | 0.465 | 64% |
| keyword (FTS5/BM25) | 0.529 | 0.342 | 60% | 0.454 | 68% |
| hybrid (RRF) | 0.833 | 0.233 | 60% | 0.593 | 76% |
| **hybrid + graph** | **0.844** | **0.433** | **80%** | **0.680** | **88%** |

Graph expansion roughly doubles MRR on call-chain questions without costing
anything on lookups, which is why it is on by default (`GRAPH_EXPANSION` in
`config.py`). The dataset is small and drawn from one repository, so treat the
numbers as a regression check, not a benchmark.

---

## Project status

Working today:

- Repository scanning with `.gitignore` support
- Python AST parsing — classes, methods, functions, base classes, call sites
- Symbol table with qualified names, signatures, decorators and parents
- Call graph resolved to symbol ids, with multi-hop chains and budgets
- AST-boundary semantic chunking
- Vector search (ChromaDB) and keyword search (SQLite)
- Hybrid retrieval: FTS5/BM25 and embeddings, fused with Reciprocal Rank Fusion
- Graph expansion of the ranking and of the prompt context
- Retrieval evaluation across vector / keyword / hybrid / hybrid+graph
- `import` extraction and inheritance edges, stored in SQLite
- LLM answering with verified citations
- Interactive chat with persistent sessions and tool use

On the roadmap:

- Git integration — `git log`/`blame` aware answers
- Incremental indexing driven by file hashes
- A larger evaluation set across several open-source repositories
- Multi-language parsing via Tree-sitter
- MCP server, so other AI clients can query the index

---

## License

MIT — see [LICENSE](LICENSE).
