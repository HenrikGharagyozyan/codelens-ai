from codelens.graph.call_graph import CallEdge, CallGraph, chain_nodes, is_test_path
from codelens.graph.resolver import BUILTIN, EXTERNAL
from codelens.indexer.vector_store import VectorStore
from codelens.llm.prompts import CONTEXT_PREAMBLE, NAME_MATCH_LABEL
from codelens.repository.db import DatabaseManager

# Hybrid search
CANDIDATES = 10  # fetched from each retriever before fusion
RRF_K = 60  # standard Reciprocal Rank Fusion constant
TEST_PENALTY = 0.5  # test code ranks lower unless the query asks about tests

# Graph expansion during search: the best hits pull their call-graph
# neighbours into the ranking, at a score that decays with every hop.
GRAPH_SEEDS = 5
GRAPH_DEPTH = 2
GRAPH_DECAY = 0.5
GRAPH_MAX_NEIGHBOURS = 6

# Call chains rendered into the prompt context, per retrieved chunk.
CHAIN_DEPTH = 3
CHAIN_BUDGET = 3
MAX_LISTED = 15  # direct callers or callees listed per chunk
MAX_NAME_MATCHES = 3  # unresolved same-name callers, offered as guesses


class ContextRetriever:
    def __init__(self, db: DatabaseManager, vector_store: VectorStore, graph_expansion: bool = False):
        self.db = db
        self.vector_store = vector_store
        self.graph = CallGraph(db)
        # Whether `search` lets call-graph neighbours into the ranking. Chains
        # are rendered into the context either way.
        self.graph_expansion = graph_expansion

    def _get_exact_symbol_id(self, symbol_name: str, file_path: str, start_line: int, end_line: int):
        """Helper method for finding the exact symbol ID using physical file bounds."""
        if not symbol_name or not file_path or start_line is None or end_line is None:
            return None

        with self.db.conn:
            cursor = self.db.conn.execute(
                """
                SELECT id FROM symbols
                WHERE name = ? AND file_path = ?
                AND line_number >= ? AND line_number <= ?
                """,
                (symbol_name, file_path, start_line, end_line),
            )
            row = cursor.fetchone()
            return row["id"] if row else None

    def _symbol_id_for(self, meta: dict) -> str | None:
        """The symbol a chunk was cut from; module summaries have none."""
        if meta.get("symbol_type") == "module":
            return None
        return self._get_exact_symbol_id(
            meta.get("symbol_name"), meta.get("file_path"), meta.get("start_line"), meta.get("end_line")
        )

    @staticmethod
    def _chunk_from_row(row) -> dict:
        """Shapes a `chunks` row like a vector-store hit."""
        return {
            "chunk_id": row["chunk_id"],
            "document": row["content"],
            "metadata": {
                "chunk_id": row["chunk_id"],
                "symbol_name": row["symbol_name"],
                "symbol_type": row["symbol_type"],
                "file_path": row["file_path"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
            },
        }

    # ------------------------------------------------------------------ search

    def _fused_scores(self, query: str) -> tuple[dict[str, float], dict[str, dict]]:
        """Reciprocal Rank Fusion of vector and keyword search, with tests pushed down."""
        vector_results = self.vector_store.search(query, limit=CANDIDATES)
        keyword_results = self.db.search_chunks_keyword(query, limit=CANDIDATES)

        scores: dict[str, float] = {}
        chunks: dict[str, dict] = {}

        for rank, res in enumerate(vector_results):
            chunk_id = res["id"]
            chunks[chunk_id] = {"chunk_id": chunk_id, "document": res["document"], "metadata": res["metadata"]}
            # Formula: 1 / (k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        for rank, row in enumerate(keyword_results):
            chunk_id = row["chunk_id"]
            chunks.setdefault(chunk_id, self._chunk_from_row(row))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        if "test" not in query.lower():
            for chunk_id in scores:
                if is_test_path(chunks[chunk_id]["metadata"].get("file_path")):
                    scores[chunk_id] *= TEST_PENALTY

        return scores, chunks

    def _hybrid_search(self, query: str, limit=4) -> list[dict]:
        """
        Combines Semantic Vector Search (Chroma) and Lexical Keyword Search (SQLite)
        using the Reciprocal Rank Fusion (RRF) algorithm.
        """
        scores, chunks = self._fused_scores(query)
        ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        return [chunks[cid] for cid in ranked[:limit]]

    def _graph_search(self, query: str, limit=4, depth: int = GRAPH_DEPTH, decay: float = GRAPH_DECAY) -> list[dict]:
        """Hybrid search whose best hits also pull their call-graph neighbours in.

        A question like "which command ends up writing embeddings?" matches the
        code that writes embeddings, not the command. The command is a caller a
        few hops up; expansion lets it into the ranking, scored as a fraction of
        the hit that led to it.
        """
        scores, chunks = self._fused_scores(query)
        asking_for_tests = "test" in query.lower()
        seeds = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:GRAPH_SEEDS]

        # Bonuses are computed from the pre-expansion scores, so the result does
        # not depend on the order in which seeds are expanded.
        bonus: dict[str, float] = {}
        for seed in seeds:
            symbol_id = self._symbol_id_for(chunks[seed]["metadata"])
            if symbol_id is None:
                continue

            for node, hops in self.graph.neighbourhood(symbol_id, depth=depth, max_nodes=GRAPH_MAX_NEIGHBOURS):
                row = self.db.get_chunk_at(node.file_path, node.line_number)
                if row is None:
                    continue
                neighbour = self._chunk_from_row(row)
                chunk_id = neighbour["chunk_id"]
                chunks.setdefault(chunk_id, neighbour)

                gain = scores[seed] * decay**hops
                if not asking_for_tests and is_test_path(node.file_path):
                    gain *= TEST_PENALTY
                bonus[chunk_id] = bonus.get(chunk_id, 0.0) + gain

        for chunk_id, gain in bonus.items():
            scores[chunk_id] = scores.get(chunk_id, 0.0) + gain

        ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        return [chunks[cid] for cid in ranked[:limit]]

    def search(self, query: str, limit: int = 4) -> list[dict]:
        if self.graph_expansion:
            return self._graph_search(query, limit=limit)
        return self._hybrid_search(query, limit=limit)

    # --------------------------------------------------------------- rendering

    def _number_lines(self, code: str, start_line: int) -> str:
        """Prefixes every source line with its real line number in the file."""
        lines = code.split("\n")
        width = len(str(start_line + len(lines) - 1))
        return "\n".join(f"{start_line + offset:>{width}} | {line}" for offset, line in enumerate(lines))

    def _format_related(self, label: str, names: list[str]) -> str:
        """Renders call-graph neighbours with their verified file:line locations."""
        locations = self.db.get_symbol_locations(names)

        rendered = []
        for name in names:
            places = locations.get(name)
            if places:
                # A name can be defined in several files; list every location so
                # the model picks one that exists instead of inventing one.
                cites = ", ".join(f"{path}:{line}" for path, line in sorted(places))
                rendered.append(f"`{name}` ({cites})")
            else:
                # Not indexed (stdlib, third-party). Say so explicitly, otherwise
                # the model will happily make a location up.
                rendered.append(f"`{name}` (external, no location)")

        return f"{label} {'; '.join(rendered)}"

    def _render_chunk(self, res: dict, idx: int) -> str:
        """Formats a single retrieved chunk and its call graph into Markdown."""
        meta = res["metadata"]
        symbol_name = meta.get("symbol_name")
        symbol_type = meta.get("symbol_type")
        file_path = meta.get("file_path")
        start_line = meta.get("start_line")
        end_line = meta.get("end_line")

        block = list(self._render_code(res["document"], idx, meta))

        # Check by symbol_type instead of the fragile "global" string
        if symbol_name and symbol_type != "module":
            block.extend(self._render_call_graph(symbol_name, file_path, start_line, end_line))

        return "\n".join(block)

    def _render_code(self, doc: str, idx: int, meta: dict) -> tuple[str, str]:
        """Returns the (header, fenced code) pair for one chunk."""
        symbol_name = meta.get("symbol_name")
        file_path = meta.get("file_path")
        start_line = meta.get("start_line")
        end_line = meta.get("end_line")

        if start_line is None or start_line < 0:
            # Without a trustworthy anchor we must not print numbered lines.
            header = f"### Chunk {idx}: {symbol_name} (File: {file_path}, line numbers unavailable)"
            return header, f"```py\n{doc}\n```"

        header = (
            f"### Chunk {idx}: {symbol_name} "
            f"(File: {file_path}, lines {start_line}-{end_line}) "
            f"-> cite as {file_path}:{start_line}"
        )
        return header, f"```py\n{self._number_lines(doc, start_line)}\n```"

    def _render_call_graph(
        self, symbol_name: str, file_path: str | None, start_line: int | None, end_line: int | None
    ) -> list[str]:
        """Renders the verified neighbourhood of a symbol: nested defs, callers, callees, chains."""
        sections = []

        nested = self._render_nested_definitions(file_path, start_line, end_line)
        if nested:
            sections.append(nested)

        sym_id = self._get_exact_symbol_id(symbol_name, file_path, start_line, end_line)

        callers = self._render_callers(symbol_name, sym_id)
        if callers:
            sections.append(callers)

        if sym_id:
            callees = self._render_callees(symbol_name, sym_id)
            if callees:
                sections.append(callees)
            sections.extend(self._render_chains(symbol_name, sym_id))

        return sections

    def _render_callers(self, symbol_name: str, sym_id: str | None) -> str | None:
        label = f"**What calls `{symbol_name}`:**"

        if sym_id is None:
            # No symbol to follow edges from: fall back to matching the name.
            incoming = self.db.get_incoming_calls(symbol_name)
            if not incoming:
                return None
            callers = sorted(set(row["caller_name"] for row in incoming))
            return self._format_related(label, callers)

        resolved = self.graph.callers(sym_id)
        entries = [f"`{edge.caller.qualname}` ({edge.caller.location})" for edge in resolved]
        listed = {edge.caller.symbol_id for edge in resolved}

        guesses = 0
        for row in self.db.get_name_matched_callers(symbol_name):
            if row["caller_id"] in listed or guesses >= MAX_NAME_MATCHES:
                continue
            listed.add(row["caller_id"])
            guesses += 1
            caller = row["caller_qualname"] or row["caller_name"]
            entries.append(f"`{caller}` ({row['caller_file']}:{row['caller_line']}) {NAME_MATCH_LABEL}")

        if not entries:
            return None
        return f"{label} {'; '.join(entries[:MAX_LISTED])}"

    def _render_callees(self, symbol_name: str, sym_id: str) -> str | None:
        resolved, guessed, external = [], [], []
        seen: set[str] = set()

        for row in self.db.get_callees(sym_id):
            name = row["callee_name"]
            key = row["callee_id"] or name
            if key in seen:
                continue
            seen.add(key)

            if row["callee_id"]:
                resolved.append(f"`{row['callee_qualname']}` ({row['callee_file']}:{row['callee_line']})")
            elif row["resolution"] in (BUILTIN, EXTERNAL):
                external.append(f"`{name}` (external, no location)")
            else:
                places = self.db.get_symbol_locations([name]).get(name)
                if places:
                    cites = ", ".join(f"{path}:{line}" for path, line in sorted(places))
                    guessed.append(f"`{name}` ({cites}) {NAME_MATCH_LABEL}")
                else:
                    external.append(f"`{name}` (external, no location)")

        entries = (resolved + guessed + external)[:MAX_LISTED]
        if not entries:
            return None
        return f"**What `{symbol_name}` calls:** {'; '.join(entries)}"

    def _render_chains(self, symbol_name: str, sym_id: str) -> list[str]:
        """Multi-hop chains through resolved calls; single hops are already listed above."""
        sections = []

        upward = [c for c in self.graph.caller_chains(sym_id, depth=CHAIN_DEPTH, max_paths=CHAIN_BUDGET) if len(c) > 1]
        if upward:
            lines = "\n".join(f"- {self._format_chain(chain)}" for chain in upward)
            sections.append(f"**Call chains leading to `{symbol_name}` (verified):**\n{lines}")

        downward = [
            c for c in self.graph.callee_chains(sym_id, depth=CHAIN_DEPTH, max_paths=CHAIN_BUDGET) if len(c) > 1
        ]
        if downward:
            lines = "\n".join(f"- {self._format_chain(chain)}" for chain in downward)
            sections.append(f"**Call chains starting at `{symbol_name}` (verified):**\n{lines}")

        return sections

    @staticmethod
    def _format_chain(chain: list[CallEdge]) -> str:
        return " -> ".join(f"`{node.qualname}` ({node.location})" for node in chain_nodes(chain))

    def _render_nested_definitions(
        self, file_path: str | None, start_line: int | None, end_line: int | None
    ) -> str | None:
        """Lists definitions living inside this chunk, with their exact lines."""
        if not file_path or start_line is None or end_line is None:
            return None

        nested = [row for row in self.db.get_symbols_in_file(file_path) if start_line < row["line_number"] <= end_line]
        if not nested:
            return None

        listing = "; ".join(f"`{row['name']}` -> {file_path}:{row['line_number']}" for row in nested)
        return f"**Definitions inside this chunk:** {listing}"

    def build_context(self, query: str, limit: int = 4) -> str | None:
        """Build enriched context (code + call graph) for the LLM."""

        results = self.search(query, limit=limit)

        if not results:
            return None

        blocks = [self._render_chunk(res, idx) for idx, res in enumerate(results, 1)]
        final_context = "\n\n---\n\n".join(blocks)

        return f"{CONTEXT_PREAMBLE}{final_context}"
