import typer
from rich.console import Console
from pathlib import Path
from rich.progress import track

from codelens.repository.scanner import RepositoryScanner
from codelens.parser.python_parser import parse_python_file
from codelens.repository.db import DatabaseManager


# Create the main CLI application
app = typer.Typer(help="CodeLens AI - Codebase Analysis & Graph Tool")
console = Console()


@app.command()
def init():
    """Initialize CodeLens in the current directory."""
    console.print("[green]CodeLens initialized successfully![/green]")


@app.command()
def index(path: str = typer.Argument(".", help="Path to the repository to index")):
    """Parse and index the repository."""
    console.print(f"[blue]Indexing repository at:[/blue] {path}")

    scanner = RepositoryScanner(path)
    repo = scanner.scan()
    db = DatabaseManager()

    symbols_count = 0

    # track() automaticly draw progress bar in terminal
    for f in track(repo.files, description="Indexing files..."):
        # Save file metadata
        db.insert_file(str(f.path), f.language, f.size, f.lines)

        # If the file is Python - build AST and extract symbols
        if f.language == "py":
            full_path = repo.root / f.path
            classes, functions = parse_python_file(full_path)

            # Save classes and their methods
            for cls in classes:
                sym_id = f"{f.path}::{cls.name}"
                db.insert_symbol(sym_id, cls.name, "class", str(f.path), cls.line_number)
                symbols_count += 1

                for method in cls.methods:
                    meth_id = f"{f.path}::{cls.name}.{method.name}"
                    db.insert_symbol(meth_id, method.name, "method", str(f.path), method.line_number)
                    symbols_count += 1

            # Save global functions
            for func in functions:
                sym_id = f"{f.path}::{func.name}"
                db.insert_symbol(sym_id, func.name, "function", str(f.path), func.line_number)
                symbols_count += 1

    console.print("\n[bold green]Index complete![/bold green]")
    console.print(f"Files indexed: {len(repo.files)}")
    console.print(f"Symbols extracted: {symbols_count}")
    console.print(f"Database saved to: {db.db_path.absolute()}")
            

@app.command()
def inspect(file: str = typer.Argument(..., help="Path to a Python file to inspect")):
    """Parse a single Python file and show its AST symbols."""
    path = Path(file)
    
    if not path.exists() or path.suffix != '.py':
        console.print("[red]Error: Please provide a valid Python file.[/red]")
        raise typer.Exit(1)
        
    classes, functions = parse_python_file(path)
    
    console.print(f"[bold green]Parsed symbols in:[/bold green] {file}\n")
    
    # Print clases and methods
    for cls in classes:
        base_str = f"({', '.join(cls.bases)})" if cls.bases else ""
        console.print(f"[blue]Class:[/blue] {cls.name}{base_str} (line {cls.line_number})")
        for method in cls.methods:
            console.print(f"  ├── [cyan]Method:[/cyan] {method.name}({', '.join(method.args)})")
            
    # Print global functions
    if functions:
        console.print("\n[magenta]Global Functions:[/magenta]")
    for func in functions:
        console.print(f"  ├── {func.name}({', '.join(func.args)}) (line {func.line_number})")


@app.command()
def ask(question: str):
    """Ask the LLM a question about the indexed codebase."""
    console.print(f"[yellow]Asking LLM:[/yellow] {question}")


@app.command()
def graph(symbol: str):
    """Show the dependency graph for a specific symbol."""
    console.print(f"[magenta]Building graph for:[/magenta] {symbol}")



if __name__ == "__main__":
    app()