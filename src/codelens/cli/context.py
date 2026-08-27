from dataclasses import dataclass

@dataclass
class AppContext:
    _db = None
    _vector_store = None
    _retriever = None
    _gemini_client = None

    @property
    def db(self):
        if self._db is None:
            # Импортируем только при обращении, чтобы ускорить старт CLI
            from codelens.repository.db import DatabaseManager
            self._db = DatabaseManager()
        return self._db

    @property
    def vector_store(self):
        if self._vector_store is None:
            from codelens.indexer.vector_store import VectorStore
            self._vector_store = VectorStore()
        return self._vector_store

    @property
    def retriever(self):
        if self._retriever is None:
            from codelens.context.retriever import ContextRetriever
            self._retriever = ContextRetriever(self.db, self.vector_store)
        return self._retriever

    @property
    def gemini(self):
        if self._gemini_client is None:
            from codelens.llm.gemini import GeminiClient
            self._gemini_client = GeminiClient()
        return self._gemini_client