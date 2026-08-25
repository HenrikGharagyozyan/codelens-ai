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