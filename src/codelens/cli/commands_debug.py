from enum import Enum

import typer
from rich.markup import escape
from rich.tree import Tree

from codelens.cli.context import AppContext
from codelens.console import console
from codelens.graph.call_graph import CallGraph, GraphNode
from codelens.graph.resolver import BUILTIN, EXTERNAL

app = typer.Typer(help="Debugging and graph visualization commands")

MAX_CHILDREN = 15  # per tree node, so a hub function does not flood the terminal
MAX_NAME_MATCHES = 5


class Direction(str, Enum):
    callers = "callers"
    callees = "callees"
    both = "both"


@app.command()
def graph(
    ctx: typer.Context,
    symbol: str = typer.Argument(..., help="Symbol name, qualified name (Class.method) or full id"),
    depth: int = typer.Option(2, "--depth", "-d", min=1, max=6, help="How many hops to follow"),
    direction: Direction = typer.Option(Direction.both, "--direction", help="Which edges to follow"),
):
    """Show who calls a symbol and what it calls, as a tree."""
    app_ctx: AppContext = ctx.obj

    matches = app_ctx.db.find_symbols(symbol)
    if not matches:
        console.print(f"[red]Symbol '{escape(symbol)}' not found in index.[/red]")
        return

    node = GraphNode.from_row(matches[0])
    call_graph = CallGraph(app_ctx.db)

    console.print(f"[bold magenta]Call graph for:[/bold magenta] {escape(node.qualname)} ({escape(node.symbol_id)})")
    console.print(f"[dim]{node.type} at {node.location}[/dim]")
    if node.signature:
        console.print(f"[cyan]{escape(node.signature)}[/cyan]")
    if len(matches) > 1:
        others = ", ".join(row["id"] for row in matches[1:4])
        console.print(
            f"[yellow]{len(matches) - 1} other match(es): {escape(others)}. Pass the full id to pick one.[/yellow]"
        )
    console.print()

    if direction in (Direction.callers, Direction.both):
        console.print(_callers_tree(app_ctx, call_graph, node, depth))
    if direction in (Direction.callees, Direction.both):
        console.print(_callees_tree(app_ctx, call_graph, node, depth))


def _callers_tree(app_ctx: AppContext, call_graph: CallGraph, node: GraphNode, depth: int) -> Tree:
    tree = Tree("[bold]Called by[/bold]")
    _add_callers(tree, call_graph, node.symbol_id, depth, frozenset({node.symbol_id}))

    listed = {edge.caller.symbol_id for edge in call_graph.callers(node.symbol_id)}
    guesses = [row for row in app_ctx.db.get_name_matched_callers(node.name) if row["caller_id"] not in listed]
    for row in guesses[:MAX_NAME_MATCHES]:
        caller = row["caller_qualname"] or row["caller_name"]
        tree.add(
            f"[yellow]? {escape(caller)}[/yellow]  "
            f"[dim]{row['caller_file']}:{row['caller_line']}, call at line {row['line_number']}, "
            f"name match only[/dim]"
        )

    if not tree.children:
        tree.add("[dim]No known callers.[/dim]")
    return tree


def _add_callers(branch: Tree, call_graph: CallGraph, symbol_id: str, depth: int, on_path: frozenset[str]) -> None:
    edges = call_graph.callers(symbol_id)
    for edge in edges[:MAX_CHILDREN]:
        caller = edge.caller
        label = f"{escape(caller.qualname)}  [dim]{caller.location}, calls at line {edge.line}[/dim]"
        if caller.symbol_id in on_path:
            branch.add(f"{label} [dim](recursive)[/dim]")
            continue
        child = branch.add(label)
        if depth > 1:
            _add_callers(child, call_graph, caller.symbol_id, depth - 1, on_path | {caller.symbol_id})
    if len(edges) > MAX_CHILDREN:
        branch.add(f"[dim]... {len(edges) - MAX_CHILDREN} more[/dim]")


def _callees_tree(app_ctx: AppContext, call_graph: CallGraph, node: GraphNode, depth: int) -> Tree:
    tree = Tree("[bold]Calls[/bold]")
    rows = app_ctx.db.get_callees(node.symbol_id)
    if not rows:
        tree.add("[dim]This symbol doesn't call any other known functions.[/dim]")
        return tree

    resolved: dict[str, object] = {}
    unresolved: list[str] = []
    outside: list[str] = []

    for row in rows:
        name = row["callee_name"]
        if row["callee_id"]:
            if row["callee_id"] not in resolved:
                resolved[row["callee_id"]] = row
        elif row["resolution"] in (BUILTIN, EXTERNAL):
            if f"{name}()" not in outside:
                outside.append(f"{name}()")
        elif name not in unresolved:
            unresolved.append(name)

    for callee_id, row in list(resolved.items())[:MAX_CHILDREN]:
        label = f"{escape(row['callee_qualname'])}()  [dim]{row['callee_file']}:{row['callee_line']}[/dim]"  # type: ignore[index]
        child = tree.add(label)
        if depth > 1:
            _add_callees(child, call_graph, callee_id, depth - 1, frozenset({node.symbol_id, callee_id}))

    for name in unresolved[:MAX_CHILDREN]:
        tree.add(f"[yellow]{escape(name)}()[/yellow]  [dim]unresolved: target unknown statically[/dim]")
    if outside:
        tree.add(f"[dim]builtin / external: {escape(', '.join(outside))}[/dim]")
    return tree


def _add_callees(branch: Tree, call_graph: CallGraph, symbol_id: str, depth: int, on_path: frozenset[str]) -> None:
    edges = call_graph.callees(symbol_id)
    for edge in edges[:MAX_CHILDREN]:
        callee = edge.callee
        label = f"{escape(callee.qualname)}()  [dim]{callee.location}[/dim]"
        if callee.symbol_id in on_path:
            branch.add(f"{label} [dim](recursive)[/dim]")
            continue
        child = branch.add(label)
        if depth > 1:
            _add_callees(child, call_graph, callee.symbol_id, depth - 1, on_path | {callee.symbol_id})
    if len(edges) > MAX_CHILDREN:
        branch.add(f"[dim]... {len(edges) - MAX_CHILDREN} more[/dim]")


@app.command(name="inspect-chunks")
def inspect_chunks(ctx: typer.Context, limit: int = 3):
    """View extracted semantic chunks from the database."""
    app_ctx: AppContext = ctx.obj
    chunks = app_ctx.db.conn.execute(
        "SELECT chunk_id, start_line, end_line, content FROM chunks LIMIT ?", (limit,)
    ).fetchall()

    if not chunks:
        console.print("[red]No chunks found. Run 'uv run codelens index .' first.[/red]")
        return

    for row in chunks:
        console.print(f"[bold cyan]Chunk:[/bold cyan] {row['chunk_id']} (Lines: {row['start_line']}-{row['end_line']})")
        console.print(f"```python\n{row['content']}\n```\n")
        console.print("-" * 50)
