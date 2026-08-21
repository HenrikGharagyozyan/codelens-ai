from dataclasses import dataclass
from pathlib import Path
from codelens.parser.models import Symbol, Function, Class


@dataclass
class Chunk:
    """Represents a meaningful piece of code for vectorization."""
    chunk_id: str          # Example: "src/server.py::HTTPServer.handle"
    file_path: str
    symbol_name: str
    start_line: int
    end_line: int
    content: str           # The original source code
    symbol_type: str       # "function" or "class"


class SemanticChunker:
    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)

    def create_chunks(self, symbols: list[Symbol]) -> list[Chunk]:
        """Converts abstract symbols into actual pieces of code."""
        chunks = []
        
        # Group symbols by file to avoid reading the same file 100 times
        symbols_by_file: dict[str, list[Symbol]] = {}
        for sym in symbols:
            if sym.file_path not in symbols_by_file:
                symbols_by_file[sym.file_path] = []
            symbols_by_file[sym.file_path].append(sym)
            
        for file_path, file_symbols in symbols_by_file.items():
            full_path = self.root_dir / file_path
            if not full_path.exists():
                continue
                
            # Read the entire file into memory (code files are small)
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for sym in file_symbols:
                # Indexing in the Python AST starts at 1, while arrays start at 0
                start_idx = sym.line_number - 1
                end_idx = sym.end_line_number if sym.end_line_number else len(lines)
                
                # Extract the exact piece of code
                code_content = "".join(lines[start_idx:end_idx])
                
                # Determine the type for metadata
                if isinstance(sym, Class):
                    sym_type = "class"
                elif isinstance(sym, Function):
                    sym_type = "function"
                else:
                    sym_type = "symbol"
                    
                chunk = Chunk(
                    chunk_id=f"{sym.file_path}::{sym.name}",
                    file_path=sym.file_path,
                    symbol_name=sym.name,
                    start_line=sym.line_number,
                    end_line=end_idx,
                    content=code_content.strip(),
                    symbol_type=sym_type
                )
                chunks.append(chunk)
                
        return chunks