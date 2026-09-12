from rich.progress import track

from codelens.console import console
from codelens.indexer.chunker import SemanticChunker
from codelens.indexer.vector_store import VectorStore
from codelens.parser.python_parser import parse_file
from codelens.repository.db import DatabaseManager
from codelens.repository.scanner import RepositoryScanner


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

    def run(self):
        # Clear both SQLite tables and ChromaDB vector store
        self.db.clear_all_indexed_data()
        self.vector_store.clear()

        scanner = RepositoryScanner(self.path)
        repo = scanner.scan()

        all_symbols = []

        # Batch collections
        file_rows = []
        symbol_rows = []
        call_rows = []
        import_rows = []
        inherit_rows = []
        seen_ids = set()

        for f in track(repo.files, description="Indexing files..."):
            self.db.insert_file(str(f.path), f.language, f.size, f.lines)

            if f.language == "py":
                file_symbols = self._index_file(
                    f, repo.root, symbol_rows, call_rows, import_rows, inherit_rows, seen_ids
                )
                all_symbols.extend(file_symbols)

        # Execute batch inserts in a single transaction-like burst
        with console.status("[bold blue]Writing to database...", spinner="dots"):
            self.db.insert_files_batch(file_rows)
            self.db.insert_imports_batch(import_rows)
            self.db.insert_symbols_batch(symbol_rows)
            self.db.insert_inherits_batch(inherit_rows)
            self.db.insert_calls_batch(call_rows)

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

    def _index_file(self, f, root, symbol_rows, call_rows, import_rows, inherit_rows, seen_ids) -> list:
        rel_path = str(f.path)

        # The parser records the repository-relative path directly, so nothing
        # downstream has to rewrite `file_path` afterwards.
        parsed = parse_file(root / f.path, record_as=rel_path)

        for imp in parsed.imports:
            import_rows.append((rel_path, imp.module, imp.name, imp.alias))

        file_symbols = []

        # The module summary is chunked but not stored as a symbol: it is a
        # retrieval aid, not something the call graph should ever point at.
        if parsed.module is not None:
            file_symbols.append(parsed.module)

        for cls in parsed.classes:
            self._persist_class(cls, rel_path, symbol_rows, call_rows, inherit_rows, seen_ids)
            file_symbols.append(cls)
            file_symbols.extend(cls.methods)

        for func in parsed.functions:
            self._persist_function(func, rel_path, symbol_rows, call_rows, seen_ids)
            file_symbols.append(func)

        return file_symbols

    def _persist_class(self, cls, rel_path: str, symbol_rows, call_rows, inherit_rows, seen_ids):
        base_id = f"{rel_path}::{cls.name}"
        sym_id = self._get_unique_id(base_id, cls.line_number, seen_ids)

        symbol_rows.append((sym_id, cls.name, "class", rel_path, cls.line_number))

        for base in cls.bases:
            inherit_rows.append((sym_id, base))

        for method in cls.methods:
            meth_base_id = f"{rel_path}::{cls.name}.{method.name}"
            meth_id = self._get_unique_id(meth_base_id, method.line_number, seen_ids)
            
            symbol_rows.append((meth_id, method.name, "method", rel_path, method.line_number))

            for call_name, call_line in method.calls:
                call_rows.append((meth_id, call_name, call_line))

    def _persist_function(self, func, rel_path: str, symbol_rows, call_rows, seen_ids):
        base_id = f"{rel_path}::{func.name}"
        sym_id = self._get_unique_id(base_id, func.line_number, seen_ids)

        symbol_rows.append((sym_id, func.name, "function", rel_path, func.line_number))

        for call_name, call_line in func.calls:
            call_rows.append((sym_id, call_name, call_line))

    def _build_and_store_chunks(self, symbols: list):
        with console.status("[bold green]Chunking codebase...", spinner="dots"):
            chunker = SemanticChunker(self.path)
            chunks = chunker.create_chunks(symbols)
            self.db.save_chunks(chunks)
            self.vector_store.add_chunks(chunks)
