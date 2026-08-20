from dataclasses import dataclass
from pathlib import Path


@dataclass
class File:
    """Represents a single file in the repository."""
    path: Path          # Path relative to the repository root
    language: str       # Extension (for example, 'py', 'cpp')
    size: int           # Size in bytes
    lines: int          # Number of lines


@dataclass
class Repository:
    """Represents the whole indexed repository."""
    root: Path          # Absolute path to the project root
    files: list[File]   # List of all discovered files
