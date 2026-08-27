import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from codelens.repository.db import DatabaseManager
from codelens.llm.gemini import GeminiClient
from codelens.indexer.vector_store import VectorStore
from codelens.context.retriever import ContextRetriever


app = typer.Typer(help="Search and LLM query commands")
console = Console()


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


@app.command()
def ask(question: str = typer.Argument(..., help="Ask a question about the codebase")):
    """Ask the LLM a question about the indexed codebase using RAG with Call Graph."""
    db = DatabaseManager()
    vector_store = VectorStore()
    retriever = ContextRetriever(db, vector_store)

    with console.status("[bold cyan]Searching codebase and building graph context...", spinner="dots"):
        # The retriever now searches vectors and loads relationships from SQLite itself
        context = retriever.build_context(question, limit=5)

    if not context:
        console.print("[yellow]No relevant context found in the codebase.[/yellow]")
        return

    with console.status("[bold magenta]Analyzing code with Gemini...", spinner="dots"):
        try:
            client = GeminiClient()
            # Pass the assembled text context with the call graph
            answer = client.ask(context, question)
        except Exception as e:
            console.print(f"[bold red]Error communicating with LLM:[/bold red] {e}")
            return

    console.print(f"\n[bold green]Question:[/bold green] {question}\n")
    console.print(Markdown(answer))


@app.command()
def chat():
    """Start an interactive chat session about the codebase with history."""
    db = DatabaseManager()
    vector_store = VectorStore()
    retriever = ContextRetriever(db, vector_store)

    # Define the tool function that Gemini will call
    def search_codebase(query: str) -> str:
        """
        Searches the repository using vector embeddings and AST graph relationships.
        Use this tool whenever you need to look up code implementation, functions, or architecture.
        """
        # Show a status in the console when the model decides to search
        with console.status(f"[bold cyan]🔍 Gemini requested codebase search for: '{query}'...", spinner="dots"):
            context = retriever.build_context(query, limit=4)
            return context if context else "No relevant code found."
    
    try:
        client = GeminiClient()
        chat_session = client.start_chat_with_tools(search_codebase)
    except Exception as e:
        console.print(f"[bold red]Error initializing Gemini client:[/bold red] {e}")
        return

    console.print("[bold green]🤖 Welcome to CodeLens Interactive Chat![/bold green]")
    console.print("[dim]Ask questions about the codebase. Type 'exit' or 'quit' to end the session.\n[/dim]")

    while True:
        try:
            question = Prompt.ask("[bold blue]You[/bold blue]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not question.strip():
            continue

        if question.strip().lower() in ["exit", "quit"]:
            console.print("[dim]Goodbye![/dim]")
            break

        console.print("\n[bold green]CodeLens:[/bold green]\n")

        try:
            # Send the message directly. If the model decides to call search_codebase,
            # the SDK automatically runs the function, sends the result to the model, and returns the final response.
            for chunk in client.send_chat_message_stream(chat_session, question):
                print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            console.print(f"\n[bold red]Error communicating with LLM:[/bold red] {e}")
            continue