from codelens.repository.db import DatabaseManager
from codelens.indexer.vector_store import VectorStore


class ContextRetriever:
    def __init__(self, db: DatabaseManager, vector_store: VectorStore):
        self.db = db
        self.vector_store = vector_store

    def _get_exact_symbol_id(self, symbol_name: str, file_path: str):
        """Helper method for finding the exact symbol ID."""
        if not symbol_name:
            return None
            
        symbols = self.db.search_symbols(symbol_name)
        for sym in symbols:
            # Find an exact match by name and file
            if sym['name'] == symbol_name and sym['file_path'] == file_path:
                return sym['id']
        return None

    def _hybrid_search(self, query: str, limit = 4) -> list[dict]:
        """
        Combines Semantic Vector Search (Chroma) and Lexical Keyword Search (SQLite)
        using the Reciprocal Rank Fusion (RRF) algorithm.
        """
        # Fetch top candidates from both sources (ask for more than limit to rank them)
        vector_results = self.vector_store.search(query, limit=10)
        keyword_results = self.db.search_chunks_keyword(query, limit=10)

        rrf_k = 60 # Standard constant for RRF
        scores = {}
        chunks_data = {}

        # Score Vector Results
        for rank, res in enumerate(vector_results):
            chunk_id = res['metadata']['chunk_id']
            chunks_data[chunk_id] = {
                'chunk_id': chunk_id,
                'document': res['document'],
                'metadata': res['metadata']
            }
            # Formula: 1 / (k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Score Keyword Results
        for rank, row in enumerate(keyword_results):
            chunk_id = row['chunk_id']
            # If the keyword chunk isn't in vector results, build its dictionary structure
            if chunk_id not in chunks_data:
                chunks_data[chunk_id] = {
                    'chunk_id': chunk_id,
                    'document': row['content'],
                    'metadata': {
                        'chunk_id': chunk_id,
                        'symbol_name': row['symbol_name'],
                        'symbol_type': row['symbol_type'],
                        'file_path': row['file_path'],
                        'start_line': row['start_line'],
                        'end_line': row['end_line']
                    }
                }
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort everything by final RRF score (highest to lowest)
        ranked_chunk_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
        
        # Return only the best `limit` chunks
        return [chunks_data[cid] for cid in ranked_chunk_ids[:limit]]



    def build_context(self, query: str, limit: int = 4) -> str | None:
        """Build enriched context (code + call graph) for the LLM."""
        
        # Find relevant code chunks by meaning (vector search)
        vector_results = self.vector_store.search(query, limit=limit)
        
        if not vector_results:
            return None

        context_blocks = []
        
        # Enrich each chunk with its call graph from SQLite
        for idx, res in enumerate(vector_results, 1):
            doc = res['document']
            meta = res['metadata']
            
            symbol_name = meta.get('symbol_name')
            file_path = meta.get('file_path')
            
            block = [
                f"### Chunk {idx}: {symbol_name} (File: {file_path})",
                f"```py\n{doc}\n```"
            ]

            # If this is a specific symbol (not the file's global scope)
            if symbol_name and symbol_name != "global":
                # What calls this function?
                incoming = self.db.get_incoming_calls(symbol_name)
                if incoming:
                    callers = sorted(list(set([row['caller_name'] for row in incoming])))
                    block.append(f"**What calls `{symbol_name}`:** {', '.join(callers)}")
                
                # What does this function call?
                sym_id = self._get_exact_symbol_id(symbol_name, file_path)
                if sym_id:
                    outgoing = self.db.get_outgoing_calls(sym_id)
                    if outgoing:
                        callees = sorted(list(set([row['callee_name'] for row in outgoing])))
                        block.append(f"**What `{symbol_name}` calls:** {', '.join(callees)}")
            
            context_blocks.append("\n".join(block))

        # Combine everything into a single text block for Gemini
        final_context = "\n\n---\n\n".join(context_blocks)
        
        return f"THE FOLLOWING CONTEXT IS PROVIDED FROM THE PROJECT KNOWLEDGE BASE (WITH CODE AND CALL GRAPH):\n\n{final_context}"
    