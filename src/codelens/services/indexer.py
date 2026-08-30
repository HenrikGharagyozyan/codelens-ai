import shutil
from pathlib import Path
from rich.console import Console
from rich.progress import track

from codelens.repository.scanner import RepositoryScanner
from codelens.parser.python_parser import parse_python_file
from codelens.repository.db import DatabaseManager
from codelens.indexer.chunker import SemanticChunker
from codelens.indexer.vector_store import VectorStore

console = Console()


class CodebaseIndexer:
    def __init__(self, path: str = "."):
        self.path = path

        # Forcibly remove old databases before reindexing
        db_path = Path(".codelens.db")
        chroma_path = Path(".codelens_vector")

        if db_path.exists():
            db_path.unlink()
        if chroma_path.exists() and chroma_path.is_dir():
            shutil.rmtree(chroma_path)

        self.db = DatabaseManager()

    def run(self):
        scanner = RepositoryScanner(self.path)
        repo = scanner.scan()

        symbols_count = 0
        all_symbols = []

        for f in track(repo.files, description="Indexing files..."):
            self.db.insert_file(str(f.path), f.language, f.size, f.lines)

            if f.language == "py":
                full_path = repo.root / f.path
                classes, functions = parse_python_file(full_path)

                # Force relative paths for all symbols
                for cls in classes:
                    cls.file_path = str(f.path)
                    for method in cls.methods:
                        method.file_path = str(f.path)
                for func in functions:
                    func.file_path = str(f.path)

                all_symbols.extend(classes)
                for cls in classes:
                    all_symbols.extend(cls.methods)
                all_symbols.extend(functions)

                for cls in classes:
                    sym_id = f"{f.path}::{cls.name}"
                    self.db.insert_symbol(sym_id, cls.name, "class", str(f.path), cls.line_number)
                    symbols_count += 1

                    for method in cls.methods:
                        meth_id = f"{f.path}::{cls.name}.{method.name}"
                        self.db.insert_symbol(meth_id, method.name, "method", str(f.path), method.line_number)
                        symbols_count += 1

                        for call_name in method.calls:
                            self.db.insert_call(meth_id, call_name, method.line_number)

                for func in functions:
                    sym_id = f"{f.path}::{func.name}"
                    self.db.insert_symbol(sym_id, func.name, "function", str(f.path), func.line_number)
                    symbols_count += 1

                    for call_name in func.calls:
                        self.db.insert_call(sym_id, call_name, func.line_number)

        with console.status("[bold green]Chunking codebase...", spinner="dots"):
            chunker = SemanticChunker(self.path)
            chunks = chunker.create_chunks(all_symbols)
            self.db.save_chunks(chunks)

            vector_store = VectorStore()
            vector_store.add_chunks(chunks)

        return len(repo.files), symbols_count, self.db.db_path.absolute()