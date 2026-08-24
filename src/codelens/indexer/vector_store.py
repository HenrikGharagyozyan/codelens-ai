import chromadb
from pathlib import Path
from codelens.indexer.chunker import Chunk


class VectorStore:
    def __init__(self, db_path: str | Path = ".codelens_vector"):
        # Create a local database next to our SQLite database
        self.client = chromadb.PersistentClient(path=str(db_path))
        # HNSW with cosine similarity is the standard for text search
        self.collection = self.client.get_or_create_collection(
            name="code_chunks",
            metadata={"hnsw:space": "cosine"} 
        )

    def add_chunks(self, chunks: list[Chunk]):
        """Converts chunks into vectors and stores them in ChromaDB."""
        if not chunks:
            return

        # ChromaDB accepts data as lists
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "file_path": c.file_path, 
                "symbol_name": c.symbol_name, 
                "symbol_type": c.symbol_type
            } 
            for c in chunks
        ]
        
        # upsert will update data if we reindex the project
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Finds the most similar code chunks by the meaning of the query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )
        
        # ChromaDB returns a complex structure; simplify it slightly for the CLI
        parsed_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                parsed_results.append({
                    "id": results['ids'][0][i],
                    "document": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] # How close in meaning (closer to 0 is better)
                })
        return parsed_results