from codelens.indexer.vector_store import VectorStore
from codelens.repository.db import DatabaseManager
from codelens.llm.prompts import CONTEXT_PREAMBLE


class ContextRetriever:
    def __init__(self, db: DatabaseManager, vector_store: VectorStore):
        self.db = db
        self.vector_store = vector_store

    def _get_exact_symbol_id(self, symbol_name: str, file_path: str):
        """Helper method for finding the exact symbol ID."""
        if not symbol_name or not file_path:
            return None

        with self.db.conn:
            cursor = self.db.conn.execute(
                "SELECT id FROM symbols WHERE name = ? AND file_path = ?", (symbol_name, file_path)
            )
            row = cursor.fetchone()
            return row["id"] if row else None

    def _hybrid_search(self, query: str, limit=4) -> list[dict]:
        """
        Combines Semantic Vector Search (Chroma) and Lexical Keyword Search (SQLite)
        using the Reciprocal Rank Fusion (RRF) algorithm.
        """
        # Fetch top candidates from both sources (ask for more than limit to rank them)
        vector_results = self.vector_store.search(query, limit=10)
        keyword_results = self.db.search_chunks_keyword(query, limit=10)

        rrf_k = 60  # Standard constant for RRF
        scores = {}
        chunks_data = {}

        # Score Vector Results
        for rank, res in enumerate(vector_results):
            chunk_id = res["id"]
            chunks_data[chunk_id] = {
                "chunk_id": chunk_id,
                "document": res["document"],
                "metadata": res["metadata"],
            }
            # Formula: 1 / (k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Score Keyword Results
        for rank, row in enumerate(keyword_results):
            chunk_id = row["chunk_id"]
            # If the keyword chunk isn't in vector results, build its dictionary structure
            if chunk_id not in chunks_data:
                chunks_data[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document": row["content"],
                    "metadata": {
                        "chunk_id": chunk_id,
                        "symbol_name": row["symbol_name"],
                        "symbol_type": row["symbol_type"],
                        "file_path": row["file_path"],
                        "start_line": row["start_line"],
                        "end_line": row["end_line"],
                    },
                }
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort everything by final RRF score (highest to lowest)
        ranked_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

        # Return only the best `limit` chunks
        return [chunks_data[cid] for cid in ranked_chunk_ids[:limit]]

    def _number_lines(self, code: str, start_line: int) -> str:
        """Prefixes every source line with its real line number in the file."""
        lines = code.split("\n")
        width = len(str(start_line + len(lines) - 1))
        return "\n".join(
            f"{start_line + offset:>{width}} | {line}" for offset, line in enumerate(lines)
        )

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
        doc = res["document"]
        meta = res["metadata"]

        symbol_name = meta.get("symbol_name")
        file_path = meta.get("file_path")
        start_line = meta.get("start_line")
        end_line = meta.get("end_line")

        if start_line is None or start_line < 0:
            body = f"```py\n{doc}\n```"
            header = f"### Chunk {idx}: {symbol_name} (File: {file_path}, line numbers unavailable)"
        else:
            body = f"```py\n{self._number_lines(doc, start_line)}\n```"
            header = (
                f"### Chunk {idx}: {symbol_name} "
                f"(File: {file_path}, lines {start_line}-{end_line}) "
                f"-> cite as {file_path}:{start_line}"
            )

        block = [header, body]

        if symbol_name and symbol_name != "global":
            if file_path and start_line is not None and end_line is not None:
                nested = [
                    row
                    for row in self.db.get_symbols_in_file(file_path)
                    if start_line < row["line_number"] <= end_line
                ]
                if nested:
                    listing = "; ".join(
                        f"`{row['name']}` -> {file_path}:{row['line_number']}" for row in nested
                    )
                    block.append(f"**Definitions inside this chunk:** {listing}")

            incoming = self.db.get_incoming_calls(symbol_name)
            if incoming:
                callers = sorted(set(row["caller_name"] for row in incoming))
                block.append(self._format_related(f"**What calls `{symbol_name}`:**", callers))

            sym_id = self._get_exact_symbol_id(symbol_name, file_path)
            if sym_id:
                outgoing = self.db.get_outgoing_calls(sym_id)
                if outgoing:
                    callees = sorted(set(row["callee_name"] for row in outgoing))
                    block.append(self._format_related(f"**What `{symbol_name}` calls:**", callees))

        return "\n".join(block)
    

    def build_context(self, query: str, limit: int = 4) -> str | None:
        """Build enriched context (code + call graph) for the LLM."""

        results = self._hybrid_search(query, limit=limit)

        if not results:
            return None

        blocks = [self._render_chunk(res, idx) for idx, res in enumerate(results, 1)]
        final_context = "\n\n---\n\n".join(blocks)

        return f"{CONTEXT_PREAMBLE}{final_context}"
