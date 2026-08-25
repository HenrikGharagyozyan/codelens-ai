import typer
from rich.console import Console
from codelens.repository.db import DatabaseManager

app = typer.Typer(help="Debugging and graph visualization commands")
console = Console()


@app.command()
def graph(symbol: str = typer.Argument(..., help="Symbol name to build graph for")):
    """Show the dependency graph for a specific symbol."""
    db = DatabaseManager()

    # Find the symbol itself
    results = db.search_symbols(symbol)
    if not results:
        console.print(f"[red]Symbol '{symbol}' not found in index.[/red]")
        return

    # Find an exact name match
    target = None
    for row in results:
        if row['name'] == symbol:
            target = row
            break

    # If there is no exact match, use the first result
    if not target:
        target = results[0]
        
    sym_id = target['id']

    console.print(f"[bold magenta]Dependency Graph for:[/bold magenta] {target['name']} ({sym_id})\n")

    # Get all calls made by this function
    calls = db.get_outgoing_calls(sym_id)

    if not calls:
        console.print("[dim]This symbol doesn't call any other known functions.[/dim]")
        return

    console.print("This symbol calls:")
    # Use set to remove duplicates (if a function is called multiple times)
    unique_calls = set(row['callee_name'] for row in calls)
    for call in unique_calls:
        console.print(f"  ├── [cyan]{call}()[/cyan]")


@app.command(name="inspect-chunks")
def inspect_chunks(limit: int = 3):
    """View extracted semantic chunks from the database."""
    db = DatabaseManager()
    chunks = db.conn.execute("SELECT chunk_id, start_line, end_line, content FROM chunks LIMIT ?", (limit,)).fetchall()
    
    if not chunks:
        console.print("[red]No chunks found. Run 'uv run codelens index .' first.[/red]")
        return
        
    for row in chunks:
        console.print(f"[bold cyan]Chunk:[/bold cyan] {row['chunk_id']} (Lines: {row['start_line']}-{row['end_line']})")
        console.print(f"```python\n{row['content']}\n```\n")
        console.print("-" * 50)