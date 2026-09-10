from dataclasses import dataclass, field


@dataclass
class Import:
    file_path: str
    module: str | None
    name: str
    alias: str | None


@dataclass
class Symbol:
    name: str
    file_path: str
    line_number: int
    end_line_number: int | None = None  # For chunking
    docstring: str | None = None  # Saved documentation for LLM context

    @property
    def symbol_id(self) -> str:
        """Generates a composite ID to prevent collisions (e.g., @property getters/setters)."""
        return f"{self.file_path}::{self.name}::{self.line_number}"


@dataclass
class Function(Symbol):
    args: list[str] = field(default_factory=list)
    calls: list[tuple[str, int]] = field(default_factory=list)
    is_async: bool = False


@dataclass
class Module(Symbol):
    """A file as a whole: its docstring and what it defines at the top level.

    Indexed as its own chunk so a question about a file's *purpose* ("where are
    the storage paths configured?") can match the module summary, rather than
    having to match one of the constants inside it.
    """

    top_level_names: list[str] = field(default_factory=list)


@dataclass
class Class(Symbol):
    methods: list[Function] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)  # base clases for inheritance
