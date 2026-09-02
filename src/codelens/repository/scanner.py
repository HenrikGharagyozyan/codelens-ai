from pathlib import Path

import pathspec

from .models import File, Repository


class RepositoryScanner:
    def __init__(self, root_path: str | Path):
        # Turn the string into a Path object and get the absolute path (resolve)
        self.root = Path(root_path).resolve()
        self.ignore_spec = self._load_gitignore()

    def _load_gitignore(self) -> pathspec.PathSpec:
        """Loads ignore rules from .gitignore plus the default exclusions."""
        lines = [".git/", "__pycache__/", ".venv/", "node_modules/", "build/", "dist/"]

        gitignore_path = self.root / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, "r", encoding="utf-8") as f:
                lines.extend(f.readlines())

        # Build an object that can match paths against the gitignore rules
        return pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, lines)

    def scan(self) -> Repository:
        """Scans the directory and returns a Repository object."""
        files = []

        # rglob("*") recursively finds all files and directories
        for file_path in self.root.rglob("*"):
            if not file_path.is_file():
                continue

            # Get the path relative to the project root for the gitignore check
            rel_path = file_path.relative_to(self.root)

            # as_posix() replaces \ with / (important for pathspec on Windows)
            if self.ignore_spec.match_file(rel_path.as_posix()):
                continue

            try:
                # Try to count the lines.
                # If the file is binary, a UnicodeDecodeError is raised
                with open(file_path, "r", encoding="utf-8") as f:
                    lines_count = sum(1 for _ in f)

                # Get the file extension without the dot (for example, 'py')
                ext = file_path.suffix.lstrip(".") or "unknown"
                stat = file_path.stat()

                files.append(
                    File(path=rel_path, language=ext, size=stat.st_size, lines=lines_count)
                )
            except UnicodeDecodeError:
                # Skip binary files
                pass

        return Repository(root=self.root, files=files)
