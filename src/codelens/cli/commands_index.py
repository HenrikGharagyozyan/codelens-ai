from pathlib import Path

import typer
from rich.console import Console

from codelens.indexer.runner import CodebaseIndexer
from codelens.parser.python_parser import parse_python_file

app = typer.Typer(help="Indexing and inspection commands")
console = Console()


@app.command()
def init():
    """Initialize CodeLens in the current directory."""
    console.print("[green]CodeLens initialized successfully![/green]")


@app.command()
def index(path: str = typer.Argument(".", help="Path to the repository to index")):
    """Parse and index the repository."""
    console.print(f"[blue]Indexing repository at:[/blue] {path}")

    indexer = CodebaseIndexer(path)
    files_count, symbols_count, db_path = indexer.run()

    console.print("\n[bold green]Index complete![/bold green]")
    console.print(f"Files indexed: {files_count}")
    console.print(f"Symbols extracted: {symbols_count}")
    console.print(f"Database saved to: {db_path}")


@app.command()
def inspect(file: str = typer.Argument(..., help="Path to a Python file to inspect")):
    """Parse a single Python file and show its AST symbols."""
    path = Path(file)

    if not path.exists() or path.suffix != ".py":
        console.print("[red]Error: Please provide a valid Python file.[/red]")
        raise typer.Exit(1)

    classes, functions, _ = parse_python_file(path)

    console.print(f"[bold green]Parsed symbols in:[/bold green] {file}\n")

    # Print classes and methods
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
