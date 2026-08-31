"""Post-generation verification of the file:line citations an LLM produces.

The retriever hands the model verified line numbers, but nothing stops a model
from emitting one anyway. This module is the last line of defence: it checks
every citation in an answer against the index and rewrites the ones that are
demonstrably wrong.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from codelens.repository.db import DatabaseManager

# Matches `src/pkg/mod.py:42`, optionally followed by `-58`.
# The path must contain a separator or a known code extension so we don't match
# things like `ratio:60` in prose.
CITATION_RE = re.compile(
    r"(?P<path>(?:[\w.\-]+/)*[\w.\-]+\.(?:py|pyi|js|ts|tsx|go|rs|c|h|cpp|hpp|java))"
    r":(?P<line>\d+)"
    r"(?:-(?P<end>\d+))?"
)


@dataclass
class CitationCheck:
    """One citation found in an answer and what the index says about it."""
    path: str
    line: int
    status: str           # "ok" | "corrected" | "unknown_file" | "no_symbol"
    corrected_line: int | None = None
    symbol: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.status == "ok"


class CitationVerifier:
    """Verifies (and optionally repairs) file:line citations against the index."""

    def __init__(self, db: DatabaseManager, root: Path | str = "."):
        self.db = db
        self.root = Path(root)

    def _known_files(self) -> set[str]:
        rows = self.db.conn.execute("SELECT path FROM files").fetchall()
        return {row['path'] for row in rows}

    def check(self, path: str, line: int) -> CitationCheck:
        """Checks a single citation, correcting the line when we can."""
        if path not in self._known_files():
            return CitationCheck(path, line, "unknown_file")

        # Exact hit: a symbol really is defined at that line.
        symbol = self.db.get_symbol_at(path, line)
        if symbol is not None:
            return CitationCheck(path, line, "ok", symbol=symbol['name'])

        # The line may legitimately point inside a symbol's body rather than at
        # its definition. Accept it if it falls within a known chunk's range.
        inside = self.db.conn.execute(
            "SELECT symbol_name FROM chunks WHERE file_path = ? AND start_line <= ? AND end_line >= ? LIMIT 1",
            (path, line, line)
        ).fetchone()
        if inside is not None:
            return CitationCheck(path, line, "ok", symbol=inside['symbol_name'])

        return CitationCheck(path, line, "no_symbol")

    def check_named(self, symbol_name: str, path: str, line: int) -> CitationCheck:
        """
        Checks a citation we can tie to a symbol name, and corrects the line
        number to where that symbol is actually defined.
        """
        locations = self.db.get_symbol_locations([symbol_name]).get(symbol_name, [])
        for loc_path, loc_line in locations:
            if loc_path == path and loc_line == line:
                return CitationCheck(path, line, "ok", symbol=symbol_name)

        # Same file, wrong line -> we know the right answer, so fix it.
        for loc_path, loc_line in locations:
            if loc_path == path:
                return CitationCheck(path, line, "corrected", corrected_line=loc_line, symbol=symbol_name)

        return self.check(path, line)

    def verify(self, answer: str) -> list[CitationCheck]:
        """Returns a check for every citation found in the answer text."""
        checks = []
        for match in CITATION_RE.finditer(answer):
            path = match.group("path")
            line = int(match.group("line"))

            # If the model named the symbol near the citation, use the stronger
            # name-aware check, which can repair the line instead of only
            # flagging it.
            window = answer[max(0, match.start() - 200):match.start()]
            names = re.findall(r"`([A-Za-z_][\w.]*)`", window)
            resolved = None
            for name in reversed(names):
                candidate = name.split(".")[-1]
                if self.db.get_symbol_locations([candidate]).get(candidate):
                    resolved = candidate
                    break

            if resolved:
                checks.append(self.check_named(resolved, path, line))
            else:
                checks.append(self.check(path, line))
        return checks

    def repair(self, answer: str) -> tuple[str, list[CitationCheck]]:
        """
        Rewrites the answer so every citation matches the index.

        Corrected citations are replaced with the true line; citations we cannot
        resolve are stripped down to the file path so the user is never shown a
        number that does not exist.
        """
        checks = self.verify(answer)
        if not checks:
            return answer, checks

        result = []
        cursor = 0
        for match, check in zip(CITATION_RE.finditer(answer), checks):
            result.append(answer[cursor:match.start()])

            if check.status == "corrected" and check.corrected_line is not None:
                result.append(f"{check.path}:{check.corrected_line}")
            elif check.status in ("no_symbol", "unknown_file"):
                result.append(check.path)
            else:
                result.append(match.group(0))

            cursor = match.end()

        result.append(answer[cursor:])
        return "".join(result), checks
