import typer
from rich.markdown import Markdown

from codelens.cli.context import AppContext
from codelens.console import console

app = typer.Typer(help="Search and LLM query commands")


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
        sym_type = row["type"].upper()
        color = "blue" if sym_type == "CLASS" else "magenta"

        console.print(f"[{color}]{sym_type}[/{color}] {row['name']}")
        console.print(f"  └── [dim]{row['id']} (line {row['line_number']})[/dim]")


@app.command(name="search-semantic")
def search_semantic(
    ctx: typer.Context, query: str = typer.Argument(..., help="Query to search by meaning")
):
    """Search the codebase by semantic meaning using Embeddings."""
    app_ctx = ctx.obj

    with console.status("[bold cyan]Searching vector database...", spinner="dots"):
        results = app_ctx.vector_store.search(query, limit=3)

    if not results:
        console.print("[yellow]No relevant code found.[/yellow]")
        return

    console.print(f"[bold green]Top semantic matches for:[/bold green] '{query}'\n")

    for idx, res in enumerate(results, 1):
        meta = res["metadata"]
        console.print(
            f"{idx}. [bold cyan]{meta['symbol_name']}[/bold cyan] "
            f"in [dim]{meta['file_path']}[/dim] (Score: {res['distance']:.4f})"
        )
        # Print the first 3 lines of code for a preview
        preview_lines = res["document"].split("\n")[:3]
        preview = "\n".join(preview_lines) + ("\n..." if len(preview_lines) >= 3 else "")
        console.print(f"```python\n{preview}\n```\n")


@app.command()
def ask(
    ctx: typer.Context,
    question: str = typer.Argument(..., help="Ask a question about the codebase"),
):
    """Ask the LLM a question about the indexed codebase using RAG with Call Graph."""
    app_ctx: AppContext = ctx.obj

    with console.status(
        "[bold cyan]Searching codebase and building graph context...", spinner="dots"
    ):
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

    if not answer:
        console.print("[yellow]The model returned an empty response.[/yellow]")
        return

    # Last line of defence: rewrite any citation that disagrees with the index.
    answer, checks = app_ctx.verifier.repair(answer)

    console.print(f"\n[bold green]Question:[/bold green] {question}\n")
    console.print(Markdown(answer))
    _report_citations(checks)


def _report_citations(checks) -> None:
    """Prints a short audit of the citations found in an answer."""
    if not checks:
        return

    corrected = [c for c in checks if c.status == "corrected"]
    dropped = [c for c in checks if c.status in ("no_symbol", "unknown_file")]
    ok_count = sum(1 for c in checks if c.is_valid)

    console.print(f"\n[dim]Citations: {ok_count}/{len(checks)} verified against the index.[/dim]")
    for c in corrected:
        console.print(f"[yellow]  fixed:[/yellow] {c.symbol} -> {c.path}:{c.corrected_line}")
    for c in dropped:
        console.print(f"[red]  unverifiable line dropped:[/red] {c.path}:{c.line}")
