from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from codelens.parser.models import Class, Function, Symbol


@dataclass
class Chunk:
    """Represents a meaningful piece of code for vectorization."""

    chunk_id: str  # Example: "src/server.py::HTTPServer.handle"
    file_path: str
    symbol_name: str
    start_line: int
    end_line: int
    content: str  # The original source code
    symbol_type: str  # "function" or "class"


class SemanticChunker:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    def create_chunks(self, symbols: list[Symbol]) -> list[Chunk]:
        """Converts abstract symbols into actual pieces of code."""
        chunks = []

        for file_path, file_symbols in self._group_by_file(symbols).items():
            full_path, rel_path = self._resolve_paths(file_path)
            if not full_path.exists():
                continue

            # Read the entire file into memory (code files are small)
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            chunks.extend(self._build_chunk(sym, lines, rel_path) for sym in file_symbols)

        return chunks

    @staticmethod
    def _group_by_file(symbols: list[Symbol]) -> dict[str, list[Symbol]]:
        """Groups symbols by file so each file is read only once."""
        grouped: dict[str, list[Symbol]] = defaultdict(list)
        for sym in symbols:
            grouped[sym.file_path].append(sym)
        return grouped

    def _resolve_paths(self, file_path: str) -> tuple[Path, str]:
        """Returns (absolute path for reading, relative path for the database).

        Symbols may carry either kind of path, but the database must only ever
        see paths relative to the repository root.
        """
        path = Path(file_path)
        if not path.is_absolute():
            return self.root_dir / path, str(path)

        try:
            return path, str(path.relative_to(self.root_dir))
        except ValueError:
            # Outside the indexed root: keep the name so the chunk is still usable.
            return path, path.name

    def _build_chunk(self, sym: Symbol, lines: list[str], rel_path: str) -> Chunk:
        """Slices one symbol out of its file."""
        # Indexing in the Python AST starts at 1, while arrays start at 0
        start_idx = sym.line_number - 1

        if isinstance(sym, Class):
            content, end_idx = self._render_class(sym, lines, start_idx)
            sym_type = "class"
        else:
            content, end_idx = self._render_body(sym, lines, start_idx)
            sym_type = "function" if isinstance(sym, Function) else "symbol"

        return Chunk(
            chunk_id=f"{sym.file_path}::{sym.name}:{sym.line_number}",
            file_path=rel_path,
            symbol_name=sym.name,
            start_line=sym.line_number,
            end_line=end_idx,
            content=content,
            symbol_type=sym_type,
        )

    def _render_class(self, sym: Class, lines: list[str], start_idx: int) -> tuple[str, int]:
        """Renders a class as a skeleton: its header, docstring and method names.

        The bodies of the methods are indexed as chunks of their own, so a class
        chunk stops right before the first method declaration.
        """
        if sym.methods:
            end_idx = min(m.line_number for m in sym.methods) - 1  # Exclusive
        else:
            end_idx = sym.end_line_number or len(lines)

        header_code = "".join(lines[start_idx:end_idx]).strip()

        # Ensure docstring is explicitly added if it somehow got lost
        doc_str = ""
        if sym.docstring and sym.docstring not in header_code:
            doc_str = f'\n    """{sym.docstring}"""'

        method_names = [m.name for m in sym.methods]
        methods_str = f"\n# Methods: {', '.join(method_names)}" if method_names else ""

        return f"{header_code}{doc_str}{methods_str}".strip(), end_idx

    @staticmethod
    def _render_body(sym: Symbol, lines: list[str], start_idx: int) -> tuple[str, int]:
        """Renders a function or a bare symbol as its verbatim source lines."""
        end_idx = sym.end_line_number or len(lines)
        return "".join(lines[start_idx:end_idx]).strip(), end_idx
