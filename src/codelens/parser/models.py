from dataclasses import dataclass, field


@dataclass
class Symbol:
    name: str
    file_path: str
    line_number: int
    end_line_number: int | None = None  # For chunking

@dataclass
class Function(Symbol):
    args: list[str] = field(default_factory=list)
    calls: list[tuple[str, int]] = field(default_factory=list)
    is_async: bool = False


@dataclass
class Class(Symbol):
    methods: list[Function] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)  # base clases for inheritance
