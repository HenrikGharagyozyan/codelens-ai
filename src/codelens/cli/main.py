import typer

from codelens.cli import commands_debug, commands_index, commands_query
from codelens.cli.context import AppContext

app = typer.Typer(help="CodeLens AI - Codebase Analysis & Graph Tool", no_args_is_help=True)


@app.callback()
def main(ctx: typer.Context):
    """Main entry point and context initialization."""
    # Create the context once. Heavy dependencies load only when needed.
    ctx.obj = AppContext()


# Gather all commands into a single application without prefixes
app.add_typer(commands_index.app)
app.add_typer(commands_query.app)
app.add_typer(commands_debug.app)

if __name__ == "__main__":
    app()
