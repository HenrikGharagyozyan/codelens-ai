import uuid
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt, IntPrompt

from codelens.cli.context import AppContext


app = typer.Typer(help="Search and LLM query commands")
console = Console()


@app.command()
def search(ctx: typer.Context, query: str = typer.Argument(..., help="Symbol name to search for")):
    """Search for a symbol in the indexed database."""
    app_ctx: AppContext = ctx.obj
    results = app_ctx.db.search_symbols(query)
    
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
def search_semantic(ctx: typer.Context, query: str = typer.Argument(..., help="Query to search by meaning")):
    """Search the codebase by semantic meaning using Embeddings."""
    app_ctx = ctx.obj
    
    with console.status("[bold cyan]Searching vector database...", spinner="dots"):
        results = app_ctx.vector_store.search(query, limit=3)
        
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
def ask(ctx: typer.Context, question: str = typer.Argument(..., help="Ask a question about the codebase")):
    """Ask the LLM a question about the indexed codebase using RAG with Call Graph."""
    app_ctx: AppContext = ctx.obj

    with console.status("[bold cyan]Searching codebase and building graph context...", spinner="dots"):
        # The retriever now searches vectors and loads relationships from SQLite itself
        context = app_ctx.retriever.build_context(question, limit=5)

    if not context:
        console.print("[yellow]No relevant context found in the codebase.[/yellow]")
        return

    with console.status("[bold magenta]Analyzing code with Gemini...", spinner="dots"):
        try:
            # Pass the assembled text context with the call graph
            answer = app_ctx.gemini.ask(context, question)
        except Exception as e:
            console.print(f"[bold red]Error communicating with LLM:[/bold red] {e}")
            return

    console.print(f"\n[bold green]Question:[/bold green] {question}\n")
    console.print(Markdown(answer))


@app.command()
def chat(ctx: typer.Context):
    """Start an interactive chat session about the codebase with history."""
    app_ctx: AppContext = ctx.obj

    # Define the tool function that Gemini will call
    def search_codebase(query: str) -> str:
        """
        Searches the repository using vector embeddings and AST graph relationships.
        Use this tool whenever you need to look up code implementation, functions, or architecture.
        """
        # Show a status in the console when the model decides to search
        with console.status(f"[bold cyan]🔍 Gemini requested codebase search for: '{query}'...", spinner="dots"):
            context = app_ctx.retriever.build_context(query, limit=4)
            return context if context else "No relevant code found."

    # History logic
    recent_sessions = app_ctx.db.get_recent_sessions(limit=5)
    session_id = None
    history_dicts = None
    session_created = False

    if recent_sessions:
        console.print("\n[bold cyan]Recent chat sessions:[/bold cyan]")
        for idx, sess in enumerate(recent_sessions, 1):
            console.print(f"  {idx}. [bold]{sess['title']}[/bold] [dim]({sess['created_at']})[/dim]")
        console.print(f"  0. [bold green]Start a NEW session[/bold green]")

    valid_choices = [str(i) for i in range(len(recent_sessions) + 1)]
    choice = IntPrompt.ask("\nSelect a session to continue (or 0 for new)", choices=valid_choices, default=0)

    if choice > 0:
        selected = recent_sessions[choice - 1]
        session_id = selected['id']
        session_created = True
        
        # Extract chat history from the database
        history_rows = app_ctx.db.get_chat_history(session_id)
        history_dicts = [{'role': row['role'], 'content': row['content']} for row in history_rows]
        console.print(f"\n[dim]Continuing session: {selected['title']}[/dim]\n")

    if not session_id:
        session_id = str(uuid.uuid4())
        session_created = False
        console.print("\n[dim]Starting a new chat session...[/dim]\n")

    
    try:
        chat_session = app_ctx.gemini.start_chat_with_tools(search_codebase, history_dicts=history_dicts)
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

        # Create a session record in the database when the first message is sent
        if not session_created:
            title = question[:40] + ("..." if len(question) > 40 else "")
            app_ctx.db.create_chat_session(session_id, title=title)
            session_created = True

        console.print("\n[bold green]CodeLens:[/bold green]\n")

        try:
            # Save the user's question
            app_ctx.db.add_chat_message(session_id, "user", question)

            # Collect and print the stream
            full_response = ""
            for chunk in app_ctx.gemini.send_chat_message_stream(chat_session, question):
                print(chunk, end="", flush=True)
                full_response += chunk
            print("\n")
            
            # Save the model's response
            app_ctx.db.add_chat_message(session_id, "model", full_response)

        except Exception as e:
            console.print(f"\n[bold red]Error communicating with LLM:[/bold red] {e}")
            continue