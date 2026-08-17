import typer
from rich.console import Console

# Создаем главное приложение CLI
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
    # Здесь позже мы вызовем наш Scanner и Parser

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