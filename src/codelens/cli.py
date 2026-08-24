import typer
from rich.console import Console
from rich.progress import track
from rich.markdown import Markdown
from pathlib import Path

from codelens.repository.scanner import RepositoryScanner
from codelens.parser.python_parser import parse_python_file
from codelens.repository.db import DatabaseManager
from codelens.llm.gemini import GeminiClient
from codelens.indexer.chunker import SemanticChunker
from codelens.indexer.vector_store import VectorStore


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
    all_symbols = []  # Collect all symbols for chunker

    # track() automaticly draw progress bar in terminal
    for f in track(repo.files, description="Indexing files..."):
        # Save file metadata
        db.insert_file(str(f.path), f.language, f.size, f.lines)

        # If the file is Python - build AST and extract symbols
        if f.language == "py":
            full_path = repo.root / f.path
            classes, functions = parse_python_file(full_path)

            all_symbols.extend(classes)
            for cls in classes:
                all_symbols.extend(cls.methods)
            all_symbols.extend(functions)

            # Save classes and their methods
            for cls in classes:
                sym_id = f"{f.path}::{cls.name}"
                db.insert_symbol(sym_id, cls.name, "class", str(f.path), cls.line_number)
                symbols_count += 1

                for method in cls.methods:
                    meth_id = f"{f.path}::{cls.name}.{method.name}"
                    db.insert_symbol(meth_id, method.name, "method", str(f.path), method.line_number)
                    symbols_count += 1

                    # Save calls inside the method
                    for call_name in method.calls:
                        db.insert_call(meth_id, call_name, method.line_number)

            # Save global functions
            for func in functions:
                sym_id = f"{f.path}::{func.name}"
                db.insert_symbol(sym_id, func.name, "function", str(f.path), func.line_number)
                symbols_count += 1

                # Save calls inside the global function
                for call_name in func.calls:
                    db.insert_call(sym_id, call_name, func.line_number)

    with console.status("[bold green]Chunking codebase...", spinner="dots"):
        chunker = SemanticChunker(path)
        chunks = chunker.create_chunks(all_symbols)
        db.save_chunks(chunks)

        vector_store = VectorStore()
        vector_store.add_chunks(chunks)

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
def search(query: str = typer.Argument(..., help="Symbol name to search for")):
    """Search for a symbol in the indexed database."""
    db = DatabaseManager()
    results = db.search_symbols(query)
    
    if not results:
        console.print(f"[yellow]No symbols found matching:[/yellow] '{query}'")
        return
        
    console.print(f"[bold green]Found {len(results)} symbols matching:[/bold green] '{query}'\n")
    
    for row in results:
        # Since we configured sqlite3.Row, we can access columns by names
        sym_type = row['type'].upper()
        color = "blue" if sym_type == "CLASS" else "magenta"
        
        console.print(f"[{color}]{sym_type}[/{color}] {row['name']}")
        console.print(f"  └── [dim]{row['id']} (line {row['line_number']})[/dim]")


@app.command()
def ask(question: str = typer.Argument(..., help="Ask a question about the codebase")):
    """Ask the LLM a question about the indexed codebase using RAG."""
    vector_store = VectorStore()

    with console.status("[bold cyan]Searching codebase context...", spinner="dots"):
        # Retrieve the 5 code chunks most similar in meaning
        results = vector_store.search(question, limit=5)

    if not results:
        console.print("[yellow]No relevant context found in the codebase.[/yellow]")
        return

    with console.status("[bold magenta]Analyzing code with Gemini...", spinner="dots"):
        try:
            client = GeminiClient()
            # Pass the list of vector results directly to the client
            answer = client.ask(results, question)
        except Exception as e:
            console.print(f"[bold red]Error communicating with LLM:[/bold red] {e}")
            return

    console.print(f"\n[bold green]Question:[/bold green] {question}\n")
    console.print(Markdown(answer))


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


@app.command(name="search-semantic")
def search_semantic(query: str = typer.Argument(..., help="Query to search by meaning")):
    """Search the codebase by semantic meaning using Embeddings."""
    vector_store = VectorStore()
    
    with console.status("[bold cyan]Searching vector database...", spinner="dots"):
        results = vector_store.search(query, limit=3)
        
    if not results:
        console.print("[yellow]No relevant code found.[/yellow]")
        return
        
    console.print(f"[bold green]Top semantic matches for:[/bold green] '{query}'\n")
    
    for idx, res in enumerate(results, 1):
        meta = res['metadata']
        console.print(f"{idx}. [bold cyan]{meta['symbol_name']}[/bold cyan] in [dim]{meta['file_path']}[/dim] (Score: {res['distance']:.4f})")
        # Print the first 3 lines of code for a preview
        preview_lines = res['document'].split('\n')[:3]
        preview = '\n'.join(preview_lines) + ('\n...' if len(preview_lines) >= 3 else '')
        console.print(f"```python\n{preview}\n```\n")


if __name__ == "__main__":
    app()