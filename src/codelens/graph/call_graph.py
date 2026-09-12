"""Walking the resolved call graph: neighbours, multi-hop chains and budgets.

Only resolved edges are followed. An unresolved call has no target to walk to,
and following name matches would compound one guess into a chain of them.

Every walk is bounded twice: by depth (how many hops) and by budget (how many
paths or nodes). The graph of a real repository fans out fast; without a
budget, three hops from a utility function reach half the codebase and the
result is noise, not context.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from codelens.repository.db import DatabaseManager

TEST_MARKERS = ("test_", "/tests/")


def is_test_path(path: str | None) -> bool:
    """The same heuristic hybrid search uses to push test code down the ranking."""
    lowered = (path or "").lower()
    return any(marker in lowered for marker in TEST_MARKERS)


@dataclass(frozen=True)
class GraphNode:
    symbol_id: str
    name: str
    qualname: str
    type: str
    file_path: str
    line_number: int
    signature: str | None = None

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.line_number}"

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "GraphNode":
        return cls(
            symbol_id=row["id"],
            name=row["name"],
            qualname=row["qualname"] or row["name"],
            type=row["type"],
            file_path=row["file_path"],
            line_number=row["line_number"],
            signature=row["signature"],
        )


@dataclass(frozen=True)
class CallEdge:
    caller: GraphNode
    callee: GraphNode
    line: int  # where the call happens, in the caller's file
    resolution: str | None


def chain_nodes(chain: list[CallEdge]) -> list[GraphNode]:
    """The symbols along a chain of edges, from the first caller to the last callee."""
    if not chain:
        return []
    return [chain[0].caller] + [edge.callee for edge in chain]


class CallGraph:
    """A read-only view of the resolved call graph stored in SQLite."""

    def __init__(self, db: DatabaseManager):
        self.db = db
        self._nodes: dict[str, GraphNode | None] = {}

    def node(self, symbol_id: str) -> GraphNode | None:
        if symbol_id not in self._nodes:
            row = self.db.get_symbol(symbol_id)
            self._nodes[symbol_id] = GraphNode.from_row(row) if row else None
        return self._nodes[symbol_id]

    def callers(self, symbol_id: str) -> list[CallEdge]:
        """One edge per distinct caller, real code before tests."""
        target = self.node(symbol_id)
        if target is None:
            return []

        edges: dict[str, CallEdge] = {}
        for row in self.db.get_callers(symbol_id):
            caller = self.node(row["caller_id"])
            if caller is not None and caller.symbol_id not in edges:
                edges[caller.symbol_id] = CallEdge(caller, target, row["line_number"], row["resolution"])

        return sorted(edges.values(), key=lambda e: (is_test_path(e.caller.file_path), e.caller.file_path, e.line))

    def callees(self, symbol_id: str) -> list[CallEdge]:
        """One edge per distinct resolved callee, in the order the calls appear."""
        source = self.node(symbol_id)
        if source is None:
            return []

        edges: dict[str, CallEdge] = {}
        for row in self.db.get_callees(symbol_id):
            callee_id = row["callee_id"]
            if callee_id is None or callee_id in edges:
                continue
            callee = self.node(callee_id)
            if callee is not None:
                edges[callee_id] = CallEdge(source, callee, row["line_number"], row["resolution"])

        return list(edges.values())

    def caller_chains(
        self, symbol_id: str, depth: int = 3, max_paths: int = 4, fanout: int = 3
    ) -> list[list[CallEdge]]:
        """Paths that end at `symbol_id`, each ordered from the outermost caller inwards.

        `main -> Application.start -> Database.connect` for `Database.connect`.
        """
        paths = self._walk(symbol_id, self.callers, lambda edge: edge.caller.symbol_id, depth, max_paths, fanout)
        return [list(reversed(path)) for path in paths]

    def callee_chains(
        self, symbol_id: str, depth: int = 3, max_paths: int = 4, fanout: int = 3
    ) -> list[list[CallEdge]]:
        """Paths that start at `symbol_id` and follow what it calls."""
        return self._walk(symbol_id, self.callees, lambda edge: edge.callee.symbol_id, depth, max_paths, fanout)

    def neighbourhood(
        self, symbol_id: str, depth: int = 2, max_nodes: int = 8, fanout: int = 4
    ) -> list[tuple[GraphNode, int]]:
        """Symbols within `depth` hops in either direction, nearest first, as (node, hops)."""
        seen = {symbol_id}
        found: list[tuple[GraphNode, int]] = []
        frontier = [symbol_id]

        for hop in range(1, depth + 1):
            next_frontier = []
            for current in frontier:
                edges = self.callers(current)[:fanout] + self.callees(current)[:fanout]
                for edge in edges:
                    other = edge.caller if edge.callee.symbol_id == current else edge.callee
                    if other.symbol_id in seen:
                        continue
                    seen.add(other.symbol_id)
                    found.append((other, hop))
                    next_frontier.append(other.symbol_id)
                    if len(found) >= max_nodes:
                        return found
            frontier = next_frontier

        return found

    @staticmethod
    def _walk(
        start: str,
        step: Callable[[str], list[CallEdge]],
        far_end: Callable[[CallEdge], str],
        depth: int,
        max_paths: int,
        fanout: int,
    ) -> list[list[CallEdge]]:
        """Depth-first enumeration of maximal paths, cycle-free and within budget."""
        paths: list[list[CallEdge]] = []

        def visit(current: str, path: list[CallEdge], on_path: frozenset[str]) -> None:
            if len(paths) >= max_paths:
                return
            edges = [] if len(path) >= depth else [e for e in step(current) if far_end(e) not in on_path][:fanout]
            if not edges:
                if path:
                    paths.append(path)
                return
            for edge in edges:
                visit(far_end(edge), path + [edge], on_path | {far_end(edge)})

        visit(start, [], frozenset({start}))
        return paths
