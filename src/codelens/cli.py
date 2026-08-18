import typer
from rich.console import Console

from codelens.repository.scanner import RepositoryScanner


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
def ask(question: str):
    """Ask the LLM a question about the indexed codebase."""
    console.print(f"[yellow]Asking LLM:[/yellow] {question}")

@app.command()
def graph(symbol: str):
    """Show the dependency graph for a specific symbol."""
    console.print(f"[magenta]Building graph for:[/magenta] {symbol}")

if __name__ == "__main__":
    app()