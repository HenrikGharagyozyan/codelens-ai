from dataclasses import dataclass, field
from typing import NamedTuple

# Receiver recorded for a call made on an expression the resolver cannot follow,
# such as `items[0].save()` or `"".join(parts)`.
UNKNOWN_RECEIVER = "<expr>"


class CallSite(NamedTuple):
    """One call expression inside a function body.

    `receiver` is what the call was made on, as a dotted path in which `()`
    marks the result of a call:

        foo()                 receiver None
        self.save()           receiver "self"
        self.db.query()       receiver "self.db"
        Service().run()       receiver "Service()"
        super().__init__()    receiver "super()"
        items[0].save()       receiver "<expr>"

    Keeping the receiver is what lets the graph resolver tell `self.db.query()`
    apart from every other method called `query` in the repository.
    """

    name: str
    line: int
    receiver: str | None = None


@dataclass
class Import:
    file_path: str
    module: str | None
    name: str
    alias: str | None
    # 0 for an absolute import; 1 for `from . import x`, 2 for `from .. import x`.
    level: int = 0


@dataclass
class Symbol:
    name: str
    file_path: str
    line_number: int
    end_line_number: int | None = None  # For chunking
    docstring: str | None = None  # Saved documentation for LLM context
    # Dotted path inside the file: "Service", "Service.run", "Outer.Inner.method".
    qualname: str = ""
    signature: str | None = None  # "def run(self, retries: int = 3) -> bool"
    decorators: list[str] = field(default_factory=list)  # ["property", "app.command()"]
    # Assigned by the indexer once every symbol in the repository is known.
    symbol_id: str | None = None
    parent_id: str | None = None

    def __post_init__(self):
        if not self.qualname:
            self.qualname = self.name


@dataclass
class Function(Symbol):
    args: list[str] = field(default_factory=list)
    calls: list[CallSite] = field(default_factory=list)
    is_async: bool = False
    returns: str | None = None  # The return annotation as written, if any
    # What the resolver knows about local names: parameter annotations and
    # simple assignments such as `indexer = CodebaseIndexer(path)`. A name that
    # is local but of unknown type maps to None, so it can never be mistaken
    # for a global of the same name.
    local_types: dict[str, str | None] = field(default_factory=dict)
    # The type of the returned value: the annotation when there is one,
    # otherwise inferred from the first `return` statement.
    return_type: str | None = None


@dataclass
class Variable(Symbol):
    """A module-level name bound by an assignment, such as `DB_PATH = ...`."""

    # What the value is, when it can be told statically: `console = Console()`
    # gives "Console()", which lets `console.print()` be resolved elsewhere.
    value_type: str | None = None


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
    bases: list[str] = field(default_factory=list)  # Short base class names, for display
    # The base expressions as dotted paths, aligned with `bases`; None where a
    # base is not a plain name chain (e.g. `class A(make_base())`).
    base_refs: list[str | None] = field(default_factory=list)
    # Types of instance attributes: class-level annotations (`root: Path`) and
    # assignments in methods (`self.db = DatabaseManager()`).
    attr_types: dict[str, str] = field(default_factory=dict)
