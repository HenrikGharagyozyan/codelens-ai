import uuid
from typing import Callable

import typer
from rich.prompt import IntPrompt, Prompt

from codelens.cli.commands_search import _report_citations
from codelens.cli.context import AppContext
from codelens.console import console

app = typer.Typer(help="Interactive chat commands")


def _make_search_tool(app_ctx: AppContext) -> Callable:
    def search_codebase(query: str) -> str:
        """
        Searches the repository using vector embeddings and AST graph relationships.
        Use this tool whenever you need to look up code implementation, functions, or architecture.
        """
        with console.status(
            f"[bold cyan]🔍 Gemini requested codebase search for: '{query}'...", spinner="dots"
        ):
            context = app_ctx.retriever.build_context(query, limit=4)
            return context if context else "No relevant code found."

    return search_codebase


def _select_session(app_ctx: AppContext) -> tuple[str, list[dict] | None, bool]:
    recent_sessions = app_ctx.db.chat.get_recent_sessions(limit=5)

    if recent_sessions:
        console.print("\n[bold cyan]Recent chat sessions:[/bold cyan]")
        for idx, sess in enumerate(recent_sessions, 1):
            console.print(
                f"  {idx}. [bold]{sess['title']}[/bold] [dim]({sess['created_at']})[/dim]"
            )
        console.print("  0. [bold green]Start a NEW session[/bold green]")

    valid_choices = [str(i) for i in range(len(recent_sessions) + 1)]
    choice = IntPrompt.ask(
        "\nSelect a session to continue (or 0 for new)", choices=valid_choices, default=0
    )

    if choice > 0:
        selected = recent_sessions[choice - 1]
        session_id = selected["id"]
        history_rows = app_ctx.db.chat.get_history(session_id)
        history_dicts = [{"role": row["role"], "content": row["content"]} for row in history_rows]

        console.print(f"\n[dim]Continuing session: {selected['title']}[/dim]\n")
        return session_id, history_dicts, True

    console.print("\n[dim]Starting a new chat session...[/dim]\n")
    return str(uuid.uuid4()), None, False


def _chat_turn(app_ctx: AppContext, chat_session, session_id: str, question: str):
    console.print("\n[bold green]CodeLens:[/bold green]\n")
    try:
        app_ctx.db.chat.add_message(session_id, "user", question)

        full_response = ""
        for chunk in app_ctx.gemini.send_chat_message_stream(chat_session, question):
            print(chunk, end="", flush=True)
            full_response += chunk
        print("\n")

        repaired, checks = app_ctx.verifier.repair(full_response)
        _report_citations(checks)

        app_ctx.db.chat.add_message(session_id, "model", repaired)

    except Exception as e:
        console.print(f"\n[bold red]Error communicating with LLM:[/bold red] {e}")


@app.command()
def chat(ctx: typer.Context):
    """Start an interactive chat session about the codebase with history."""
    app_ctx: AppContext = ctx.obj

    session_id, history_dicts, session_created = _select_session(app_ctx)
    search_tool = _make_search_tool(app_ctx)

    try:
        chat_session = app_ctx.gemini.start_chat_with_tools(
            search_tool, history_dicts=history_dicts
        )
    except Exception as e:
        console.print(f"[bold red]Error initializing Gemini client:[/bold red] {e}")
        return

    console.print("[bold green]🤖 Welcome to CodeLens Interactive Chat![/bold green]")
    console.print(
        "[dim]Ask questions about the codebase. Type 'exit' or 'quit' to end the session.\n[/dim]"
    )

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
            app_ctx.db.chat.create_session(session_id, title=title)
            session_created = True

        _chat_turn(app_ctx, chat_session, session_id, question)
