import typer
from codelens.cli import commands_index, commands_query, commands_debug

app = typer.Typer(
    help="CodeLens AI - Codebase Analysis & Graph Tool",
    no_args_is_help=True
)

# Собираем все команды в единое приложение без префиксов
app.add_typer(commands_index.app)
app.add_typer(commands_query.app)
app.add_typer(commands_debug.app)

if __name__ == "__main__":
    app()