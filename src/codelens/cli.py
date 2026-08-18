import typer
from rich.console import Console
from pathlib import Path

from codelens.repository.scanner import RepositoryScanner
from codelens.parser.python_parser import parse_python_file


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

    console.print(f"[green]Successfully scanned:[/green] {repo.root}")
    console.print(f"[green]Found text files:[/green] {len(repo.files)}")

    # Print the first 3 files as a sanity check
    if repo.files:
        console.print("\n[yellow]Sample files found:[/yellow]")
        for f in repo.files[:3]:
            console.print(f"  - {f.path} ({f.lines} lines, {f.size} bytes)")


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