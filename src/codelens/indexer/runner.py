from rich.console import Console
from rich.progress import track

from codelens.indexer.chunker import SemanticChunker
from codelens.indexer.vector_store import VectorStore
from codelens.parser.python_parser import parse_python_file
from codelens.repository.db import DatabaseManager
from codelens.repository.scanner import RepositoryScanner

console = Console()


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

        symbols_count = 0
        all_symbols = []

        for f in track(repo.files, description="Indexing files..."):
            self.db.insert_file(str(f.path), f.language, f.size, f.lines)

            if f.language == "py":
                file_symbols = self._index_file(f, repo.root)
                all_symbols.extend(file_symbols)

        self._build_and_store_chunks(all_symbols)

        symbols_count = self.db.get_symbol_count()

        return len(repo.files), symbols_count, self.db.db_path.absolute()

    def _index_file(self, f, root) -> list:
        file_path = root / f.path
        rel_path = str(f.path)
        classes, functions, imports = parse_python_file(file_path)

        for imp in imports:
            self.db.insert_import(rel_path, imp.module, imp.name, imp.alias)

        file_symbols = []

        # Force relative paths for all symbols and persist them
        for cls in classes:
            cls.file_path = rel_path
            for method in cls.methods:
                method.file_path = rel_path

            self._persist_class(cls, rel_path)
            file_symbols.append(cls)
            file_symbols.extend(cls.methods)

        for func in functions:
            func.file_path = rel_path
            self._persist_function(func, rel_path)
            file_symbols.append(func)

        return file_symbols

    def _persist_class(self, cls, rel_path: str):
        sym_id = f"{rel_path}::{cls.name}"
        self.db.insert_symbol(sym_id, cls.name, "class", rel_path, cls.line_number)

        for base in cls.bases:
            self.db.insert_inherit(sym_id, base)

        for method in cls.methods:
            meth_id = f"{rel_path}::{cls.name}.{method.name}"
            self.db.insert_symbol(meth_id, method.name, "method", rel_path, method.line_number)

            for call_name, call_line in method.calls:
                self.db.insert_call(meth_id, call_name, call_line)

    def _persist_function(self, func, rel_path: str):
        sym_id = f"{rel_path}::{func.name}"
        self.db.insert_symbol(sym_id, func.name, "function", rel_path, func.line_number)

        for call_name, call_line in func.calls:
            self.db.insert_call(sym_id, call_name, call_line)

    def _build_and_store_chunks(self, symbols: list):
        with console.status("[bold green]Chunking codebase...", spinner="dots"):
            chunker = SemanticChunker(self.path)
            chunks = chunker.create_chunks(symbols)
            self.db.save_chunks(chunks)
            self.vector_store.add_chunks(chunks)
