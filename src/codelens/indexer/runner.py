import json
from collections import Counter
from dataclasses import dataclass, field

from rich.progress import track

from codelens.console import console
from codelens.graph.resolver import SymbolTable
from codelens.indexer.chunker import SemanticChunker
from codelens.indexer.vector_store import VectorStore
from codelens.parser.models import Function, Symbol
from codelens.parser.python_parser import ParsedFile, parse_file
from codelens.repository.db import DatabaseManager
from codelens.repository.scanner import RepositoryScanner


@dataclass
class IndexRows:
    """Every row one indexing run writes, collected so each table is written in one batch."""

    files: list[tuple] = field(default_factory=list)
    symbols: list[tuple] = field(default_factory=list)
    calls: list[tuple] = field(default_factory=list)
    imports: list[tuple] = field(default_factory=list)
    inherits: list[tuple] = field(default_factory=list)


class CodebaseIndexer:
    def __init__(
        self,
        path: str = ".",
        db: DatabaseManager | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.path = path
        self.db = db if db is not None else DatabaseManager()
        self.vector_store = vector_store if vector_store is not None else VectorStore()
        # How the call sites of the last run were resolved, keyed by resolution label.
        self.call_stats: Counter[str] = Counter()

    def run(self):
        # Clear both SQLite tables and ChromaDB vector store
        self.db.clear_all_indexed_data()
        self.vector_store.clear()
        self.call_stats = Counter()

        scanner = RepositoryScanner(self.path)
        repo = scanner.scan()

        rows = IndexRows()
        parsed_files: dict[str, ParsedFile] = {}

        for f in track(repo.files, description="Indexing files..."):
            rows.files.append((str(f.path), f.language, f.size, f.lines))

            if f.language == "py":
                rel_path = str(f.path)
                # The parser records the repository-relative path directly, so nothing
                # downstream has to rewrite `file_path` afterwards.
                parsed_files[rel_path] = parse_file(repo.root / f.path, record_as=rel_path)

        # Calls can only be resolved once every file has been parsed: `connect()`
        # in app.py may be defined in db.py, which the loop reaches later.
        self._assign_ids(parsed_files)
        table = SymbolTable(parsed_files)
        for rel_path, parsed in parsed_files.items():
            self._collect_rows(rel_path, parsed, table, rows)

        with console.status("[bold blue]Writing to database...", spinner="dots"):
            self.db.insert_files_batch(rows.files)
            self.db.insert_imports_batch(rows.imports)
            self.db.insert_symbols_batch(rows.symbols)
            self.db.insert_inherits_batch(rows.inherits)
            self.db.insert_calls_batch(rows.calls)

        all_symbols = [sym for parsed in parsed_files.values() for sym in self._chunkable(parsed)]
        self._build_and_store_chunks(all_symbols)

        symbols_count = self.db.get_symbol_count()

        return len(repo.files), symbols_count, self.db.db_path.absolute()

    def _get_unique_id(self, base_id: str, line_number: int, seen_ids: set) -> str:
        """Ensures symbol IDs are unique, appending line number on collisions (e.g., @property)."""
        if base_id not in seen_ids:
            seen_ids.add(base_id)
            return base_id

        alt_id = f"{base_id}::{line_number}"
        if alt_id in seen_ids:
            # Fallback for edge cases where even the line number is identical
            counter = 1
            while f"{alt_id}_{counter}" in seen_ids:
                counter += 1
            alt_id = f"{alt_id}_{counter}"

        seen_ids.add(alt_id)
        return alt_id

    def _assign_ids(self, parsed_files: dict[str, ParsedFile]) -> None:
        """Gives every symbol its id (`path::Qual.name`) and every method its parent."""
        seen_ids: set[str] = set()

        for rel_path, parsed in parsed_files.items():
            class_ids: dict[str, str] = {}

            for cls in parsed.classes:
                cls.symbol_id = self._get_unique_id(f"{rel_path}::{cls.qualname}", cls.line_number, seen_ids)
                # A nested class belongs to the class it is written in.
                parent_qualname = cls.qualname.rpartition(".")[0]
                cls.parent_id = class_ids.get(parent_qualname) if parent_qualname else None
                class_ids.setdefault(cls.qualname, cls.symbol_id)

                for method in cls.methods:
                    method.symbol_id = self._get_unique_id(
                        f"{rel_path}::{method.qualname}", method.line_number, seen_ids
                    )
                    method.parent_id = cls.symbol_id

            for sym in [*parsed.functions, *parsed.variables]:
                sym.symbol_id = self._get_unique_id(f"{rel_path}::{sym.qualname}", sym.line_number, seen_ids)

    def _collect_rows(self, rel_path: str, parsed: ParsedFile, table: SymbolTable, rows: IndexRows) -> None:
        for imp in parsed.imports:
            rows.imports.append((rel_path, imp.module, imp.name, imp.alias, imp.level))

        for cls in parsed.classes:
            rows.symbols.append(self._symbol_row(cls, "class"))

            assert cls.symbol_id is not None
            for base, base_id in zip(cls.bases, table.base_ids(cls.symbol_id)):
                rows.inherits.append((cls.symbol_id, base, base_id))

            for method in cls.methods:
                rows.symbols.append(self._symbol_row(method, "method"))
                self._collect_calls(method, table, rows)

        for func in parsed.functions:
            rows.symbols.append(self._symbol_row(func, "function"))
            self._collect_calls(func, table, rows)

        for var in parsed.variables:
            rows.symbols.append(self._symbol_row(var, "variable"))

    def _collect_calls(self, func: Function, table: SymbolTable, rows: IndexRows) -> None:
        assert func.symbol_id is not None
        for call in func.calls:
            callee_id, resolution = table.resolve_call(func.symbol_id, call)
            self.call_stats[resolution] += 1
            rows.calls.append((func.symbol_id, call.name, call.line, call.receiver, callee_id, resolution))

    @staticmethod
    def _symbol_row(sym: Symbol, sym_type: str) -> tuple:
        return (
            sym.symbol_id,
            sym.name,
            sym.qualname,
            sym_type,
            sym.file_path,
            sym.line_number,
            sym.end_line_number,
            sym.signature,
            json.dumps(sym.decorators) if sym.decorators else None,
            sym.parent_id,
        )

    @staticmethod
    def _chunkable(parsed: ParsedFile) -> list[Symbol]:
        symbols: list[Symbol] = []

        # The module summary is chunked but not stored as a symbol: it is a
        # retrieval aid, not something the call graph should ever point at.
        if parsed.module is not None:
            symbols.append(parsed.module)

        for cls in parsed.classes:
            symbols.append(cls)
            symbols.extend(cls.methods)

        symbols.extend(parsed.functions)
        symbols.extend(parsed.variables)
        return symbols

    def _build_and_store_chunks(self, symbols: list):
        with console.status("[bold green]Chunking codebase...", spinner="dots"):
            chunker = SemanticChunker(self.path)
            chunks = chunker.create_chunks(symbols)
            self.db.save_chunks(chunks)
            self.vector_store.add_chunks(chunks)
